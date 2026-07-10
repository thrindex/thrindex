//! Ratification measurement binary for ADR-0010 Part II.
//!
//! Runs when the ≥100-sample frozen SHD test set is available and produces the
//! distribution data needed to set final `CONFORMANCE_ENVELOPE_V0` thresholds.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p conformance --bin ratify_envelope -- \
//!     --artifact /path/to/model.thx \
//!     --data-dir  conformance/fixtures/shd_100
//! ```
//!
//! `--data-dir` must point to a directory containing `frozen/sample_NNN.json` files
//! (≥100 of them). Run `scripts/freeze_shd_fixtures.py` first if the directory does
//! not exist — that is ADR-0010 Part II Step 1.
//!
//! # What this measures
//!
//! For each frozen sample:
//! 1. Load the artifact into a `ResolvedModel` (f32, ADR-0007).
//! 2. Apply **per-channel symmetric int8** quantization to every Dense layer's weight
//!    matrix. Each output-neuron's row gets its own scale
//!    `s_n = max_abs(row_n) / 127`. Rounding: **round-half-to-even** (banker's rounding,
//!    consistent with `thrindex-numerics`). This is the realistic hardware noise model
//!    per ADR-0010 Part II self-critique #2.
//! 3. Run both the f32 model and the quantized model through `thrindex_sim::sim::run`.
//!    These are two genuinely different executions — f32 weights vs int8 dequantized
//!    weights. Any ~0 error here is a real result, not a compare-to-self artefact.
//! 4. Compute `per_neuron_rate_errors(float_raster, int8_raster)` for all output neurons.
//!
//! # Output
//!
//! - Fixture-set fingerprint (CRC32 of all fixture files, hash-pinned for traceability).
//! - Full per-neuron rate error distribution across ALL (neuron × sample) pairs.
//! - Per-sample mean/max distribution.
//! - Per-tensor int8 comparison (gap vs per-channel).
//! - Prediction-agreement breakdown.
//! - Recommended final T_mean, T_max, P_min — paste into ADR-0010 Part II Amendment.
//!
//! # Ratification gate
//!
//! After running, paste the full output into the ADR-0010 Part II Amendment section.
//! Submit for founder approval. No backend may be Certified until the Amendment is merged.
use std::io::Write as _;
use std::path::PathBuf;

use conformance::metric::{
    hamming_fraction, max_rate_error, mean_rate_error, per_neuron_rate_errors, prediction,
};
use thrindex_sim::{
    SimConfig, SimOutput,
    model::{DenseLayer, ResolvedLayer, ResolvedModel, load_from_str},
    sim::run as sim_run,
};

// ─── CLI ─────────────────────────────────────────────────────────────────────

fn print_usage(bin: &str) {
    eprintln!("Usage: {bin} --artifact <model.thx> --data-dir <fixture_dir>");
    eprintln!();
    eprintln!("  --artifact    Path to a trained model.thx (SHD architecture).");
    eprintln!("  --data-dir    Directory whose frozen/ subdir contains sample_NNN.json files.");
    eprintln!("  --n-samples   Number of samples to use (default: 100, must be ≤ available).");
    eprintln!();
    eprintln!("ADR-0010 Part II ratification measurement. Output contains the fixture set");
    eprintln!("fingerprint and recommended thresholds for CONFORMANCE_ENVELOPE_V0.");
}

struct Args {
    artifact: PathBuf,
    data_dir: PathBuf,
    n_samples: usize,
}

fn parse_args() -> Option<Args> {
    let raw: Vec<String> = std::env::args().collect();
    let mut artifact: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut n_samples = 100usize;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--artifact" if i + 1 < raw.len() => { artifact = Some(PathBuf::from(&raw[i + 1])); i += 2; }
            "--data-dir" if i + 1 < raw.len() => { data_dir = Some(PathBuf::from(&raw[i + 1])); i += 2; }
            "--n-samples" if i + 1 < raw.len() => { n_samples = raw[i + 1].parse().ok()?; i += 2; }
            _ => i += 1,
        }
    }
    Some(Args { artifact: artifact?, data_dir: data_dir?, n_samples })
}

// ─── Rounding ─────────────────────────────────────────────────────────────────

/// Round-half-to-even (banker's rounding), consistent with `thrindex-numerics`.
///
/// `f32::round()` rounds half-away-from-zero; this function ties to the nearest even
/// integer instead. The difference matters only on exact 0.5 boundaries, but at
/// quantization boundaries (e.g. when a weight falls exactly halfway between two int8
/// levels) it produces systematically less biased rounding error.
fn round_half_to_even(x: f32) -> f32 {
    let floor = x.floor();
    let frac = x - floor;
    if (frac - 0.5).abs() < 1e-6 {
        // Tie: round to even.
        let floor_i = floor as i64;
        if floor_i % 2 == 0 { floor } else { floor + 1.0 }
    } else {
        x.round()
    }
}

// ─── Quantization ─────────────────────────────────────────────────────────────

/// Per-channel symmetric int8: each output neuron (row) gets its own scale.
///
/// `scale_n = max_abs(row_n) / 127`
/// `q_n[k]  = clamp(round_half_to_even(w[n,k] / scale_n), -127, 127)`
/// `dq_n[k] = q_n[k] * scale_n`
///
/// This is the realistic hardware noise model per ADR-0010 Part II.
/// Per-channel is 4–8× more accurate than per-tensor for large matrices with
/// weight-magnitude variation across output neurons.
fn quantize_per_channel_int8(weights: &[f32], out_features: usize) -> Vec<f32> {
    debug_assert_eq!(weights.len() % out_features, 0, "weights not divisible by out_features");
    let in_features = weights.len() / out_features;
    let mut out = weights.to_vec();
    for out_n in 0..out_features {
        let start = out_n * in_features;
        let row = &weights[start..start + in_features];
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs < 1e-12 { 1.0_f32 } else { max_abs / 127.0 };
        for (k, &w) in row.iter().enumerate() {
            let q = round_half_to_even(w / scale).clamp(-127.0, 127.0) as i8;
            out[start + k] = q as f32 * scale;
        }
    }
    out
}

/// Per-tensor symmetric int8: single scale over the entire matrix.
///
/// Included for comparison with per-channel to quantify the gap.
/// Per-tensor is what naive quantization does; it overstates the error budget
/// because outlier rows dominate the single global scale.
fn quantize_per_tensor_int8(weights: &[f32]) -> Vec<f32> {
    let max_abs = weights.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs < 1e-12 { 1.0_f32 } else { max_abs / 127.0 };
    weights
        .iter()
        .map(|&w| {
            let q = round_half_to_even(w / scale).clamp(-127.0, 127.0) as i8;
            q as f32 * scale
        })
        .collect()
}

/// Apply per-channel int8 to all Dense layers in a resolved model.
fn quantize_model_per_channel(model: &ResolvedModel) -> ResolvedModel {
    let layers = model
        .layers
        .iter()
        .map(|layer| match layer {
            ResolvedLayer::Dense(d) => {
                let q_weights = quantize_per_channel_int8(&d.weights, d.out_features);
                // Biases are left in f32 — most hardware backends do not quantize biases.
                let q_bias = d.bias.clone();
                ResolvedLayer::Dense(DenseLayer {
                    in_features: d.in_features,
                    out_features: d.out_features,
                    weights: q_weights,
                    bias: q_bias,
                })
            }
            other => other.clone(),
        })
        .collect();
    ResolvedModel { layers, target: model.target.clone() }
}

/// Apply per-tensor int8 to all Dense layers in a resolved model.
fn quantize_model_per_tensor(model: &ResolvedModel) -> ResolvedModel {
    let layers = model
        .layers
        .iter()
        .map(|layer| match layer {
            ResolvedLayer::Dense(d) => {
                let q_weights = quantize_per_tensor_int8(&d.weights);
                ResolvedLayer::Dense(DenseLayer {
                    in_features: d.in_features,
                    out_features: d.out_features,
                    weights: q_weights,
                    bias: d.bias.clone(),
                })
            }
            other => other.clone(),
        })
        .collect();
    ResolvedModel { layers, target: model.target.clone() }
}

// ─── Frozen fixture loading ───────────────────────────────────────────────────

struct Fixture {
    /// File content, used for hashing.
    raw: String,
    spikes: Vec<Vec<f32>>,
    #[allow(dead_code)]
    label: usize,
}

fn load_fixtures(data_dir: &PathBuf, n_samples: usize) -> Result<Vec<Fixture>, String> {
    let frozen_dir = data_dir.join("frozen");
    if !frozen_dir.exists() {
        return Err(format!(
            "Frozen fixture directory {:?} not found.\n\
             Run `uv run python scripts/freeze_shd_fixtures.py \
             --data-dir /tmp/shd --out-dir {:?}` first (ADR-0010 Part II Step 1).",
            frozen_dir, data_dir
        ));
    }
    let mut fixtures = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let path = frozen_dir.join(format!("sample_{i:03}.json"));
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        let label = v["label"].as_u64().ok_or("missing label")? as usize;

        // Auto-detect format: sparse_events_v1 (new) or dense "spikes" (legacy).
        let spikes: Vec<Vec<f32>> = if v["events"].is_array() {
            // Sparse events format (sparse_events_v1): [[t, u], ...]
            let t_size = v["T"].as_u64().unwrap_or(100) as usize;
            let n_in = v["N_in"].as_u64().unwrap_or(700) as usize;
            let mut dense = vec![vec![0.0f32; n_in]; t_size];
            if let Some(events) = v["events"].as_array() {
                for event in events {
                    if let Some(pair) = event.as_array()
                        && pair.len() == 2
                    {
                        let t = pair[0].as_u64().unwrap_or(0) as usize;
                        let u = pair[1].as_u64().unwrap_or(0) as usize;
                        if t < t_size && u < n_in {
                            dense[t][u] = 1.0;
                        }
                    }
                }
            }
            dense
        } else if let Some(spikes_arr) = v["spikes"].as_array() {
            // Dense format (legacy, backward-compatible).
            spikes_arr
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect()
                })
                .collect()
        } else {
            return Err(format!(
                "{}: expected either 'events' (sparse_events_v1) or 'spikes' (dense) field",
                path.display()
            ));
        };

        fixtures.push(Fixture { raw, spikes, label });
    }
    Ok(fixtures)
}

/// Compute CRC32 fingerprint for the fixture set.
///
/// Algorithm: CRC32 of each file's bytes, concatenated as 8-char lowercase hex,
/// then CRC32 of the full concatenation. Stable: depends only on file content,
/// not on filesystem order (files are loaded in sorted index order).
///
/// This fingerprint must appear in the ADR-0010 Part II Amendment so that a
/// ratified threshold is traceable to the exact frozen set that produced it.
fn fixture_fingerprint(fixtures: &[Fixture]) -> String {
    let mut combined = String::new();
    for f in fixtures {
        let crc = crc32fast::hash(f.raw.as_bytes());
        combined.push_str(&format!("{crc:08x}"));
    }
    let set_crc = crc32fast::hash(combined.as_bytes());
    format!("crc32={set_crc:08x} ({} files)", fixtures.len())
}

// ─── Distribution helpers ─────────────────────────────────────────────────────

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_dist(sorted: &[f64], label: &str) {
    println!("  {label}");
    println!("    p50: {:.4e}   p90: {:.4e}   p95: {:.4e}   p99: {:.4e}   max: {:.4e}",
        percentile(sorted, 50.0),
        percentile(sorted, 90.0),
        percentile(sorted, 95.0),
        percentile(sorted, 99.0),
        sorted.last().cloned().unwrap_or(0.0),
    );
}

fn run_sim(model: &ResolvedModel, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let config = SimConfig { threads: 1 };
    let output: SimOutput = sim_run(model, &[input.to_vec()], &config)
        .expect("sim::run failed");
    output.spikes.into_iter().next().expect("batch size 1")
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── round_half_to_even ──────────────────────────────────────────────────

    #[test]
    fn round_half_to_even_rounds_down_on_even_floor() {
        // 2.5 → floor=2 (even) → 2.0
        let r = round_half_to_even(2.5_f32);
        assert_eq!(r, 2.0, "2.5 must round to 2 (even)");
    }

    #[test]
    fn round_half_to_even_rounds_up_on_odd_floor() {
        // 3.5 → floor=3 (odd) → 4.0
        let r = round_half_to_even(3.5_f32);
        assert_eq!(r, 4.0, "3.5 must round to 4 (even)");
    }

    #[test]
    fn round_half_to_even_negative_tie() {
        // -2.5 → floor=-3 (odd) → -2.0
        let r = round_half_to_even(-2.5_f32);
        assert_eq!(r, -2.0, "-2.5 must round to -2 (even)");
    }

    #[test]
    fn round_half_to_even_non_tie_behaves_normally() {
        assert_eq!(round_half_to_even(2.3_f32), 2.0);
        assert_eq!(round_half_to_even(2.7_f32), 3.0);
        assert_eq!(round_half_to_even(-1.3_f32), -1.0);
    }

    // ── quantize_per_channel_int8 ───────────────────────────────────────────

    #[test]
    fn per_channel_uses_per_row_scale() {
        // Row 0: max_abs = 1.0 → scale = 1/127 → all elements quantized with row-0 scale.
        // Row 1: max_abs = 0.5 → scale = 0.5/127 → all elements quantized with row-1 scale.
        // The two rows must get different scales.
        let weights = vec![
            1.0_f32, 0.5,  // row 0: max_abs=1.0
            0.5_f32, 0.25, // row 1: max_abs=0.5
        ];
        let q = quantize_per_channel_int8(&weights, 2); // out_features=2
        // Row 0: scale_0 = 1.0/127. 1.0 → round(127) = 127 → dq = 127/127 = 1.0.
        //        0.5 → round(63.5) = 64 (even) → dq = 64/127 ≈ 0.50394.
        // Row 1: scale_1 = 0.5/127. 0.5 → round(127) = 127 → dq = 127*(0.5/127) = 0.5.
        //        0.25 → round(63.5) = 64 (even) → dq = 64*(0.5/127) ≈ 0.25197.
        let expected_row0_w0 = 127.0_f32 / 127.0; // = 1.0
        let expected_row1_w0 = 127.0_f32 * (0.5 / 127.0); // = 0.5
        assert!((q[0] - expected_row0_w0).abs() < 1e-5, "row0 max weight should dequantize near 1.0");
        assert!((q[2] - expected_row1_w0).abs() < 1e-5, "row1 max weight should dequantize near 0.5");
        // Crucially: the two rows are NOT the same (per-channel, not per-tensor).
        assert!((q[1] - q[3]).abs() > 1e-6, "row0 and row1 non-max weights should differ (different scales)");
    }

    #[test]
    fn per_channel_identity_on_max_weight() {
        // The maximum weight in each row must dequantize to exactly the max (127 × scale = max_abs).
        let weights = vec![0.8_f32, 0.3, -0.8, 0.1_f32, 0.7, -0.1];
        // out_features = 2, in_features = 3
        let q = quantize_per_channel_int8(&weights, 2);
        // Row 0: max_abs = 0.8, element 0 = 0.8 (max), must dequantize to 0.8.
        assert!((q[0] - 0.8_f32).abs() < 1e-5, "max element should round-trip to original: {}", q[0]);
        // Row 1: max_abs = 0.7, element 4 = 0.7 (max), must dequantize to 0.7.
        assert!((q[4] - 0.7_f32).abs() < 1e-5, "max element should round-trip to original: {}", q[4]);
    }

    #[test]
    fn per_channel_clamps_to_127() {
        // A single-element row: the max is the only element. Should round-trip exactly.
        let w = vec![0.5_f32];
        let q = quantize_per_channel_int8(&w, 1);
        // scale = 0.5/127, w/scale = 127.0, round = 127.
        assert!((q[0] - 0.5_f32).abs() < 1e-5);
    }

    #[test]
    fn per_channel_is_strictly_per_row_not_global() {
        // If this were per-tensor, row 1's 0.001 element would lose essentially all precision.
        // Per-channel: row 1's scale = 0.001/127, so 0.001 maps exactly to 127 → recovers 0.001.
        let weights = vec![
            100.0_f32, 50.0, // row 0: large
            0.001_f32, 0.0,  // row 1: tiny — per-tensor would crush this to 0
        ];
        let q = quantize_per_channel_int8(&weights, 2);
        // Row 1 max = 0.001; scale = 0.001/127 ≈ 7.87e-6.
        // 0.001 / scale = 127.0 → rounds to 127 → dequantize = 0.001.
        assert!((q[2] - 0.001_f32).abs() < 1e-4, "row1 max should be preserved, got {}", q[2]);

        // Per-tensor would have scale = 100.0/127 ≈ 0.787; 0.001/0.787 → rounds to 0.
        // Prove per-tensor would fail:
        let q_tensor = quantize_per_tensor_int8(&weights);
        assert_eq!(q_tensor[2], 0.0, "per-tensor crushes 0.001 to 0 when large row exists");
    }

    // ── fixture_fingerprint ─────────────────────────────────────────────────

    #[test]
    fn fingerprint_is_deterministic() {
        let f1 = Fixture { raw: "abc".to_string(), spikes: vec![], label: 0 };
        let f2 = Fixture { raw: "def".to_string(), spikes: vec![], label: 1 };
        let fp1 = fixture_fingerprint(&[f1, f2]);
        let f3 = Fixture { raw: "abc".to_string(), spikes: vec![], label: 0 };
        let f4 = Fixture { raw: "def".to_string(), spikes: vec![], label: 1 };
        let fp2 = fixture_fingerprint(&[f3, f4]);
        assert_eq!(fp1, fp2, "fingerprint must be deterministic");
    }

    #[test]
    fn fingerprint_changes_on_content_change() {
        let f1 = Fixture { raw: "abc".to_string(), spikes: vec![], label: 0 };
        let fp1 = fixture_fingerprint(&[f1]);
        let f2 = Fixture { raw: "abd".to_string(), spikes: vec![], label: 0 };
        let fp2 = fixture_fingerprint(&[f2]);
        assert_ne!(fp1, fp2, "fingerprint must change when file content changes");
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = match parse_args() {
        Some(a) => a,
        None => {
            print_usage(&std::env::args().next().unwrap_or_default());
            std::process::exit(1);
        }
    };

    // ── Load and parse artifact ────────────────────────────────────────────────
    let artifact_json = match std::fs::read_to_string(&args.artifact) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("E0201: cannot read artifact {}: {e}", args.artifact.display());
            std::process::exit(1);
        }
    };
    let float_model = match load_from_str(&artifact_json) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("E0201: artifact parse failed: {e}");
            std::process::exit(1);
        }
    };

    // ── Build quantized models ─────────────────────────────────────────────────
    let int8_ch_model = quantize_model_per_channel(&float_model);
    let int8_tensor_model = quantize_model_per_tensor(&float_model);

    // ── Load frozen fixtures ───────────────────────────────────────────────────
    let fixtures = match load_fixtures(&args.data_dir, args.n_samples) {
        Ok(f) => f,
        Err(e) => { eprintln!("Cannot load fixtures: {e}"); std::process::exit(1); }
    };

    let fingerprint = fixture_fingerprint(&fixtures);

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" ADR-0010 Part II — Ratification Measurement");
    println!(" artifact:     {}", args.artifact.display());
    println!(" fixture set:  {fingerprint}");
    println!(" n_samples:    {}", fixtures.len());
    println!(" quantization: per-channel symmetric int8 (round-half-to-even)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ── Run float vs per-channel int8 ─────────────────────────────────────────
    // Separate storage for per-sample aggregates and the full per-neuron flat list.
    let n = fixtures.len();
    let mut per_sample_mean_ch: Vec<f64> = Vec::with_capacity(n);
    let mut per_sample_max_ch: Vec<f64> = Vec::with_capacity(n);
    let mut per_sample_mean_tensor: Vec<f64> = Vec::with_capacity(n);
    let mut per_sample_max_tensor: Vec<f64> = Vec::with_capacity(n);
    let mut all_neuron_errors_ch: Vec<f64> = Vec::new();   // all (neuron, sample) pairs
    let mut all_neuron_errors_tensor: Vec<f64> = Vec::new();
    let mut hammings: Vec<f64> = Vec::with_capacity(n);
    let mut pred_agrees = 0usize;
    let mut boundary_agrees = 0usize;
    let mut boundary_total = 0usize;

    print!("Running samples");
    std::io::stdout().flush().ok();

    for (i, fixture) in fixtures.iter().enumerate() {
        // Three independent runs: float, per-channel int8, per-tensor int8.
        let ref_raster = run_sim(&float_model, &fixture.spikes);
        let ch_raster = run_sim(&int8_ch_model, &fixture.spikes);
        let tensor_raster = run_sim(&int8_tensor_model, &fixture.spikes);

        // Per-channel errors.
        let errors_ch = per_neuron_rate_errors(&ref_raster, &ch_raster);
        all_neuron_errors_ch.extend_from_slice(&errors_ch);
        per_sample_mean_ch.push(mean_rate_error(&errors_ch));
        per_sample_max_ch.push(max_rate_error(&errors_ch));

        // Per-tensor errors (for gap comparison).
        let errors_tensor = per_neuron_rate_errors(&ref_raster, &tensor_raster);
        all_neuron_errors_tensor.extend_from_slice(&errors_tensor);
        per_sample_mean_tensor.push(mean_rate_error(&errors_tensor));
        per_sample_max_tensor.push(max_rate_error(&errors_tensor));

        // Hamming (informational, float vs per-channel only).
        hammings.push(hamming_fraction(&ref_raster, &ch_raster));

        // Prediction agreement.
        let ref_pred = prediction(&ref_raster);
        let ch_pred = prediction(&ch_raster);
        if ref_pred == ch_pred { pred_agrees += 1; }

        // Boundary: samples where argmax margin in reference is small.
        if !ref_raster.is_empty() {
            let n_neurons = ref_raster[0].len();
            if n_neurons > 1 {
                let mut counts: Vec<f64> = (0..n_neurons)
                    .map(|k| ref_raster.iter().map(|f| if f[k] > 0.5 { 1.0 } else { 0.0 }).sum())
                    .collect();
                counts.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let margin = (counts[0] - counts[1]).abs();
                if margin < 3.0 {  // fewer than 3 spike difference at top-2
                    boundary_total += 1;
                    if ref_pred == ch_pred { boundary_agrees += 1; }
                }
            }
        }

        if (i + 1) % 10 == 0 { print!(" {}", i + 1); std::io::stdout().flush().ok(); }
    }
    println!(" done.");

    // ── Sort all distributions ─────────────────────────────────────────────────
    all_neuron_errors_ch.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_neuron_errors_tensor.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per_sample_mean_ch.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per_sample_max_ch.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per_sample_mean_tensor.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per_sample_max_tensor.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_h = hammings.clone();
    sorted_h.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_neurons_per_sample = all_neuron_errors_ch.len() / n;
    println!();
    println!("─── FLOAT vs PER-CHANNEL INT8 ──────────────────────────────────────");
    println!("  {} samples × {} output neurons = {} data points",
        n, n_neurons_per_sample, all_neuron_errors_ch.len());
    println!();
    println!("  Full per-neuron error distribution (ALL neurons × ALL samples):");
    print_dist(&all_neuron_errors_ch, "per-neuron rate error");
    println!();
    println!("  Per-sample mean_rate_error (one number per sample):");
    print_dist(&per_sample_mean_ch, "per-sample mean");
    println!();
    println!("  Per-sample max_rate_error (worst neuron per sample):");
    print_dist(&per_sample_max_ch, "per-sample max");

    println!();
    println!("─── FLOAT vs PER-TENSOR INT8 (gap comparison) ──────────────────────");
    println!("  Full per-neuron error distribution:");
    print_dist(&all_neuron_errors_tensor, "per-neuron rate error");
    println!();
    println!("  Per-sample mean_rate_error:");
    print_dist(&per_sample_mean_tensor, "per-sample mean");
    println!();
    println!("  Per-sample max_rate_error:");
    print_dist(&per_sample_max_tensor, "per-sample max");

    // Per-channel vs per-tensor gap.
    let ch_p99 = percentile(&all_neuron_errors_ch, 99.0);
    let tensor_p99 = percentile(&all_neuron_errors_tensor, 99.0);
    let gap_ratio = if ch_p99 > 0.0 { tensor_p99 / ch_p99 } else { f64::INFINITY };
    println!();
    println!("  Gap: per-tensor p99 / per-channel p99 = {tensor_p99:.4e} / {ch_p99:.4e} = {gap_ratio:.1}×");
    println!("  (ADR-0010 Part II self-critique: expected 4–8× gap; gaps outside this band");
    println!("  indicate unusual weight distribution or architecture — investigate before ratifying)");

    println!();
    println!("─── Hamming fraction (informational, not pass/fail) ─────────────────");
    println!("  mean: {:.4e}   max: {:.4e}",
        hammings.iter().sum::<f64>() / n as f64,
        sorted_h.last().cloned().unwrap_or(0.0));

    println!();
    println!("─── Prediction agreement (float vs per-channel int8) ─────────────────");
    let pred_frac = pred_agrees as f64 / n as f64;
    println!("  Overall: {:.2}%  ({pred_agrees}/{n} samples)", pred_frac * 100.0);
    if boundary_total > 0 {
        let bound_frac = boundary_agrees as f64 / boundary_total as f64;
        println!("  Near-argmax boundary cases: {:.2}%  ({boundary_agrees}/{boundary_total})",
            bound_frac * 100.0);
        println!("  NOTE: boundary agreement ({:.2}%) is the conservative gate anchor.", bound_frac * 100.0);
    } else {
        println!("  No near-argmax samples detected — all argmax margins ≥ 3 spikes.");
        println!("  P_min derivation: use overall agreement ({:.4}) × 0.95 = {:.4}.",
            pred_frac, pred_frac * 0.95);
    }

    // ── Recommended thresholds ────────────────────────────────────────────────
    let obs_max_mean = per_sample_mean_ch.last().cloned().unwrap_or(0.0);
    let obs_max_max = per_sample_max_ch.last().cloned().unwrap_or(0.0);
    let rec_t_mean = obs_max_mean * 2.5;
    let rec_t_max = obs_max_max * 2.0;
    let rec_pred = if boundary_total > 0 {
        let bound_frac = boundary_agrees as f64 / boundary_total as f64;
        // Conservative: floor at 90% of observed boundary agreement.
        (bound_frac * 0.90).min(0.99)
    } else {
        (pred_frac * 0.95).min(0.99)
    };

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" RECOMMENDED FINAL THRESHOLDS");
    println!(" Paste into ADR-0010 Part II Amendment. Founder approval required.");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Fixture set fingerprint: {fingerprint}");
    println!("  Quantization model: per-channel symmetric int8, round-half-to-even.");
    println!();
    println!("  T_mean_threshold = {rec_t_mean:.4e}");
    println!("    (derivation: max observed per-sample mean_rate_error = {obs_max_mean:.4e}, × 2.5)");
    println!();
    println!("  T_max_threshold  = {rec_t_max:.4e}");
    println!("    (derivation: max observed per-sample max_rate_error  = {obs_max_max:.4e}, × 2.0)");
    println!();
    println!("  pred_agreement   = {rec_pred:.4}");
    if boundary_total > 0 {
        let bound_frac = boundary_agrees as f64 / boundary_total as f64;
        println!("    (derivation: boundary-case agreement = {bound_frac:.4}, × 0.90)");
    } else {
        println!("    (derivation: overall agreement = {pred_frac:.4}, × 0.95 — no boundary cases)");
    }
    println!();
    println!("  Confirm int4 per-channel FAILS both T_mean and T_max.");
    println!("  If int4 also passes, the thresholds are too loose — tighten before ratifying.");
    println!();

    // ── Sanity vs DRAFT thresholds ────────────────────────────────────────────
    let draft = &conformance::CONFORMANCE_ENVELOPE_V0_DRAFT;
    println!("─── Comparison with CONFORMANCE_ENVELOPE_v0_DRAFT ──────────────────");
    let status_mean = if obs_max_mean <= draft.t_mean_threshold { "OK — within draft" } else { "EXCEEDS draft anchor" };
    let status_max = if obs_max_max <= draft.t_max_threshold { "OK — within draft" } else { "EXCEEDS draft anchor" };
    let status_pred = if pred_frac >= draft.pred_agreement_min { "OK — within draft" } else { "BELOW draft anchor" };
    println!("  measured max_mean  {obs_max_mean:.4e}  vs  DRAFT T_mean {:.4e}  → {status_mean}",
        draft.t_mean_threshold);
    println!("  measured max_max   {obs_max_max:.4e}  vs  DRAFT T_max  {:.4e}  → {status_max}",
        draft.t_max_threshold);
    println!("  measured pred_frac {pred_frac:.4}         vs  DRAFT P_min {:.4}         → {status_pred}",
        draft.pred_agreement_min);
    println!();
    println!("NOTE: This output does not constitute certification. ADR-0010 Part II");
    println!("must be amended by the founder before any backend can claim THRINDEX Certified.");
}
