//! Ratification measurement binary for ADR-0010 Part II.
//!
//! Run this once the frozen ≥100-sample SHD fixture set exists in `--data-dir`.
//! The output contains the int8 vs int4 gap analysis needed to set final
//! `CONFORMANCE_ENVELOPE_V0` thresholds.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p conformance --bin ratify_envelope -- \
//!     --artifact templates/keyword-spotting/model.thx \
//!     --data-dir conformance/fixtures/shd_ratify_v1
//! ```
//!
//! When `--n-samples` is omitted, the binary auto-discovers all `sample_NNN.json`
//! files in `<data-dir>/frozen/` and uses all of them.  This ensures the fixture-set
//! fingerprint always matches the committed manifest.
//!
//! # What this measures
//!
//! For each frozen sample, three independent simulations are run:
//! 1. **Reference** (`float f32`, no quantization) → raster R.
//! 2. **Int8-per-channel** (should PASS): scale per output neuron = `max_abs / 127`.
//! 3. **Int4-per-channel** (should FAIL): scale per output neuron = `max_abs / 7`.
//! 4. **Int8-per-tensor** (informational): single global scale = `max_abs / 127`.
//!
//! Rounding throughout: **round-half-to-even** (banker's rounding, consistent with
//! `thrindex-numerics` and IEEE 754).
//!
//! # Threshold derivation principle
//!
//! The `CONFORMANCE_ENVELOPE_V0` thresholds are derived from the **gap** between
//! int8 (should-pass) and int4 (should-fail) aggregate metrics:
//!
//!   `T = int8_agg + (int4_agg - int8_agg) × 0.40`
//!
//! This places the threshold 40% of the way into the gap from the int8 side,
//! giving int8 60% of the gap as headroom and int4 60% of the gap as failure margin.
//!
//! The exact harness aggregates are used (`agg_mean = mean of per-sample
//! mean_rate_errors`; `agg_max = max of per-sample max_rate_errors`) so the
//! recommended thresholds are directly comparable to what `run_conformance` computes.
//!
//! # Ratification gate
//!
//! Paste the full output into ADR-0010 Part II Amendment. Submit for founder approval.
//! No backend may be Certified until the Amendment is merged.
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
    eprintln!("Usage: {bin} --artifact <model.thx> --data-dir <fixture_dir> [--n-samples N]");
    eprintln!();
    eprintln!("  --artifact    Path to the trained .thx artifact.");
    eprintln!("  --data-dir    Directory whose frozen/ subdir has sample_NNN.json files.");
    eprintln!("  --n-samples   Samples to load (default: 0 = ALL available).");
    eprintln!();
    eprintln!("ADR-0010 Part II ratification. Produces int8 vs int4 gap analysis.");
}

struct Args {
    artifact: PathBuf,
    data_dir: PathBuf,
    /// 0 = auto-discover all fixtures.
    n_samples: usize,
}

fn parse_args() -> Option<Args> {
    let raw: Vec<String> = std::env::args().collect();
    let mut artifact: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut n_samples = 0usize; // 0 = auto-discover
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
/// `f32::round()` rounds half-away-from-zero. This function ties to the nearest even
/// integer — unbiased on exact 0.5 boundaries, matching IEEE 754 default rounding mode.
fn round_half_to_even(x: f32) -> f32 {
    let floor = x.floor();
    let frac = x - floor;
    if (frac - 0.5).abs() < 1e-6 {
        let floor_i = floor as i64;
        if floor_i % 2 == 0 { floor } else { floor + 1.0 }
    } else {
        x.round()
    }
}

// ─── Quantization ─────────────────────────────────────────────────────────────

/// Per-channel symmetric int8 (should-PASS reference).
///
/// `scale_n = max_abs(row_n) / 127`, round-half-to-even, clamp [-127, 127].
fn quantize_per_channel_int8(weights: &[f32], out_features: usize) -> Vec<f32> {
    debug_assert_eq!(weights.len() % out_features, 0);
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

/// Per-channel symmetric int4 (should-FAIL reference).
///
/// `scale_n = max_abs(row_n) / 7`, round-half-to-even, clamp [-7, 7].
/// 4-bit signed symmetric range: [-7, 7] (excludes -8 for sign symmetry).
/// Produces 18× more quantization steps than int4 naive (2^4/2 = 7 vs 2^8/2 = 127
/// levels above zero), meaning roughly sqrt(127/7) ≈ 4.3× more error per weight.
fn quantize_per_channel_int4(weights: &[f32], out_features: usize) -> Vec<f32> {
    debug_assert_eq!(weights.len() % out_features, 0);
    let in_features = weights.len() / out_features;
    let mut out = weights.to_vec();
    for out_n in 0..out_features {
        let start = out_n * in_features;
        let row = &weights[start..start + in_features];
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs < 1e-12 { 1.0_f32 } else { max_abs / 7.0 };
        for (k, &w) in row.iter().enumerate() {
            let q = round_half_to_even(w / scale).clamp(-7.0, 7.0) as i8;
            out[start + k] = q as f32 * scale;
        }
    }
    out
}

/// Per-tensor symmetric int8 (informational — for gap comparison).
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

fn apply_quantization<F>(model: &ResolvedModel, quantize_weights: F) -> ResolvedModel
where
    F: Fn(&[f32], usize) -> Vec<f32>,
{
    let layers = model
        .layers
        .iter()
        .map(|layer| match layer {
            ResolvedLayer::Dense(d) => {
                ResolvedLayer::Dense(DenseLayer {
                    in_features: d.in_features,
                    out_features: d.out_features,
                    weights: quantize_weights(&d.weights, d.out_features),
                    // Biases are left in f32 — most hardware backends do not quantize biases.
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
    raw: String,
    spikes: Vec<Vec<f32>>,
    #[allow(dead_code)]
    label: usize,
}

/// Count how many `sample_NNN.json` files are available in `frozen_dir`.
fn count_available_fixtures(frozen_dir: &std::path::Path) -> usize {
    let mut count = 0;
    loop {
        if !frozen_dir.join(format!("sample_{count:03}.json")).exists() {
            break;
        }
        count += 1;
    }
    count
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

    let available = count_available_fixtures(&frozen_dir);
    if available == 0 {
        return Err(format!("No sample_NNN.json files found in {:?}.", frozen_dir));
    }

    let to_load = if n_samples == 0 { available } else { n_samples.min(available) };

    let mut fixtures = Vec::with_capacity(to_load);
    for i in 0..to_load {
        let path = frozen_dir.join(format!("sample_{i:03}.json"));
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        let label = v["label"].as_u64().ok_or("missing label")? as usize;

        // Auto-detect format: sparse_events_v1 (new) or dense "spikes" (legacy).
        let spikes: Vec<Vec<f32>> = if v["events"].is_array() {
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

/// CRC32 fixture-set fingerprint — matches `ratify_envelope::fixture_fingerprint()`.
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
    println!("    {label}");
    println!("      p50: {:.4e}   p90: {:.4e}   p95: {:.4e}   p99: {:.4e}   max: {:.4e}",
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

// ─── Per-condition metrics ────────────────────────────────────────────────────

struct ConditionMetrics {
    /// Sorted all-neuron × all-samples rate errors.
    all_neuron_errors: Vec<f64>,
    /// Sorted per-sample mean_rate_error.
    per_sample_mean: Vec<f64>,
    /// Sorted per-sample max_rate_error.
    per_sample_max: Vec<f64>,
    /// Harness agg_mean: mean of per_sample_mean (raw, unsorted).
    agg_mean: f64,
    /// Harness agg_max: max of per_sample_max.
    agg_max: f64,
    /// Overall prediction agreement.
    pred_agree: f64,
    /// Boundary-case prediction agreement.
    boundary_agree_frac: Option<f64>,
    /// Hamming (informational).
    mean_hamming: f64,
    max_hamming: f64,
}

fn collect_condition(
    ref_rasters: &[Vec<Vec<f32>>],
    test_rasters: &[Vec<Vec<f32>>],
) -> ConditionMetrics {
    let n = ref_rasters.len();
    let mut all_neuron_errors: Vec<f64> = Vec::new();
    let mut per_sample_mean_raw: Vec<f64> = Vec::with_capacity(n);
    let mut per_sample_max_raw: Vec<f64> = Vec::with_capacity(n);
    let mut hammings: Vec<f64> = Vec::with_capacity(n);
    let mut pred_agrees = 0usize;
    let mut boundary_agrees = 0usize;
    let mut boundary_total = 0usize;

    for (ref_r, test_r) in ref_rasters.iter().zip(test_rasters.iter()) {
        let errors = per_neuron_rate_errors(ref_r, test_r);
        all_neuron_errors.extend_from_slice(&errors);
        per_sample_mean_raw.push(mean_rate_error(&errors));
        per_sample_max_raw.push(max_rate_error(&errors));
        hammings.push(hamming_fraction(ref_r, test_r));

        if prediction(ref_r) == prediction(test_r) { pred_agrees += 1; }

        if !ref_r.is_empty() {
            let n_neurons = ref_r[0].len();
            if n_neurons > 1 {
                let mut counts: Vec<f64> = (0..n_neurons)
                    .map(|k| ref_r.iter().map(|f| if f[k] > 0.5 { 1.0 } else { 0.0 }).sum())
                    .collect();
                counts.sort_by(|a, b| b.partial_cmp(a).unwrap());
                if (counts[0] - counts[1]).abs() < 3.0 {
                    boundary_total += 1;
                    if prediction(ref_r) == prediction(test_r) { boundary_agrees += 1; }
                }
            }
        }
    }

    // Harness-exact aggregates.
    let agg_mean = per_sample_mean_raw.iter().sum::<f64>() / n as f64;
    let agg_max = per_sample_max_raw.iter().cloned().fold(0.0_f64, f64::max);
    let pred_agree = pred_agrees as f64 / n as f64;
    let boundary_agree_frac = if boundary_total > 0 {
        Some(boundary_agrees as f64 / boundary_total as f64)
    } else {
        None
    };
    let mean_hamming = hammings.iter().sum::<f64>() / n as f64;
    let max_hamming = hammings.iter().cloned().fold(0.0_f64, f64::max);

    all_neuron_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per_sample_mean_raw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per_sample_max_raw.sort_by(|a, b| a.partial_cmp(b).unwrap());

    ConditionMetrics {
        all_neuron_errors,
        per_sample_mean: per_sample_mean_raw,
        per_sample_max: per_sample_max_raw,
        agg_mean,
        agg_max,
        pred_agree,
        boundary_agree_frac,
        mean_hamming,
        max_hamming,
    }
}

fn print_condition(label: &str, m: &ConditionMetrics, n_neurons: usize, n_samples: usize) {
    println!("─── {} ───────────────────────────────────────────────", label);
    println!("  {} samples × {} output neurons = {} data points",
        n_samples, n_neurons, m.all_neuron_errors.len());
    println!("  Harness aggregates (exact values used by run_conformance):");
    println!("    agg_mean (mean of per-sample means):  {:.4e}", m.agg_mean);
    println!("    agg_max  (max  of per-sample maxes):  {:.4e}", m.agg_max);
    println!("    pred_agree:                            {:.4}  ({:.1}%)",
        m.pred_agree, m.pred_agree * 100.0);
    if let Some(ba) = m.boundary_agree_frac {
        println!("    boundary-case pred_agree:              {:.4}  ({:.1}%)", ba, ba * 100.0);
    } else {
        println!("    boundary-case pred_agree:              n/a (no near-tie samples)");
    }
    println!("  Full distributions:");
    print_dist(&m.all_neuron_errors, "per-neuron error (ALL neurons × ALL samples)");
    print_dist(&m.per_sample_mean, "per-sample mean_rate_error");
    print_dist(&m.per_sample_max, "per-sample max_rate_error (worst neuron)");
    println!("  Hamming (informational): mean={:.4e}  max={:.4e}", m.mean_hamming, m.max_hamming);
    println!();
}

// ─── Gap-based threshold derivation ──────────────────────────────────────────

struct GapResult {
    threshold: f64,
    #[allow(dead_code)]
    /// How far into the gap the threshold sits (0.0 = at int8, 1.0 = at int4).
    gap_fraction: f64,
    gap_ratio: f64,
    /// Headroom above int8 as a fraction of int8.
    int8_headroom_pct: f64,
    /// Margin below int4 as a fraction of int4.
    int4_margin_pct: f64,
    int8_passes: bool,
    int4_fails: bool,
    /// True if the gap is too narrow for a clean separation.
    gap_too_narrow: bool,
}

fn derive_threshold(int8_val: f64, int4_val: f64, gap_frac: f64) -> GapResult {
    let gap_ratio = if int8_val > 0.0 { int4_val / int8_val } else { f64::INFINITY };
    let gap_too_narrow = gap_ratio < 1.2;
    let threshold = if gap_too_narrow || int4_val <= int8_val {
        int8_val * 1.5  // fallback: 50% headroom above int8 only
    } else {
        int8_val + (int4_val - int8_val) * gap_frac
    };
    let int8_passes = int8_val <= threshold;
    let int4_fails = int4_val > threshold;
    let gap_fraction = if (int4_val - int8_val).abs() > 1e-15 {
        (threshold - int8_val) / (int4_val - int8_val)
    } else { 0.0 };
    let int8_headroom_pct = if int8_val > 0.0 { (threshold / int8_val - 1.0) * 100.0 } else { 100.0 };
    let int4_margin_pct = if int4_val > 0.0 { (1.0 - threshold / int4_val) * 100.0 } else { 100.0 };
    GapResult { threshold, gap_fraction, gap_ratio, int8_headroom_pct, int4_margin_pct, int8_passes, int4_fails, gap_too_narrow }
}

fn check(passes: bool) -> &'static str { if passes { "✓" } else { "✗ PROBLEM" } }

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── round_half_to_even ──────────────────────────────────────────────────

    #[test]
    fn round_half_to_even_rounds_down_on_even_floor() {
        assert_eq!(round_half_to_even(2.5_f32), 2.0, "2.5 must round to 2 (even)");
    }

    #[test]
    fn round_half_to_even_rounds_up_on_odd_floor() {
        assert_eq!(round_half_to_even(3.5_f32), 4.0, "3.5 must round to 4 (even)");
    }

    #[test]
    fn round_half_to_even_negative_tie() {
        assert_eq!(round_half_to_even(-2.5_f32), -2.0, "-2.5 must round to -2 (even)");
    }

    #[test]
    fn round_half_to_even_non_tie_behaves_normally() {
        assert_eq!(round_half_to_even(2.3_f32), 2.0);
        assert_eq!(round_half_to_even(2.7_f32), 3.0);
    }

    // ── quantize_per_channel_int8 ───────────────────────────────────────────

    #[test]
    fn per_channel_int8_uses_per_row_scale() {
        let weights = vec![1.0_f32, 0.5, 0.5_f32, 0.25];
        let q = quantize_per_channel_int8(&weights, 2);
        let expected_row0_w0 = 127.0_f32 / 127.0;
        let expected_row1_w0 = 127.0_f32 * (0.5 / 127.0);
        assert!((q[0] - expected_row0_w0).abs() < 1e-5);
        assert!((q[2] - expected_row1_w0).abs() < 1e-5);
        assert!((q[1] - q[3]).abs() > 1e-6, "rows must use different scales");
    }

    #[test]
    fn per_channel_int8_identity_on_max_weight() {
        let weights = vec![0.8_f32, 0.3, -0.8, 0.1_f32, 0.7, -0.1];
        let q = quantize_per_channel_int8(&weights, 2);
        assert!((q[0] - 0.8_f32).abs() < 1e-5);
        assert!((q[4] - 0.7_f32).abs() < 1e-5);
    }

    #[test]
    fn per_channel_int8_preserves_tiny_row_per_channel() {
        let weights = vec![100.0_f32, 50.0, 0.001_f32, 0.0];
        let q8 = quantize_per_channel_int8(&weights, 2);
        let q8_tensor = quantize_per_tensor_int8(&weights);
        assert!((q8[2] - 0.001_f32).abs() < 1e-4, "per-channel preserves tiny row: {}", q8[2]);
        assert_eq!(q8_tensor[2], 0.0, "per-tensor crushes tiny row to 0");
    }

    // ── quantize_per_channel_int4 ───────────────────────────────────────────

    #[test]
    fn per_channel_int4_uses_scale_7() {
        // Row 0: max=0.7, scale=0.7/7=0.1. 0.7/0.1=7.0 → 7 → dq=0.7.
        let weights = vec![0.7_f32, 0.0, 0.35_f32, 0.0];
        let q = quantize_per_channel_int4(&weights, 2);
        assert!((q[0] - 0.7_f32).abs() < 1e-5, "max weight should round-trip: {}", q[0]);
    }

    #[test]
    fn per_channel_int4_clamps_to_pm7() {
        let weights = vec![1.0_f32, -1.0];
        let q = quantize_per_channel_int4(&weights, 1);
        // scale = 1.0/7. 1.0/scale = 7.0 → clamp → 7 → dq = 1.0.
        assert!((q[0] - 1.0_f32).abs() < 1e-5);
        assert!((q[1] + 1.0_f32).abs() < 1e-5);
    }

    #[test]
    fn per_channel_int4_has_more_error_than_int8() {
        // int4 has 127/7 ≈ 18× fewer quantization steps → more rounding error.
        let weights: Vec<f32> = (0..128).map(|i| (i as f32) / 127.0).collect();
        let q8 = quantize_per_channel_int8(&weights, 1);
        let q4 = quantize_per_channel_int4(&weights, 1);
        let err8: f64 = weights.iter().zip(q8.iter()).map(|(w, q)| (w - q).abs() as f64).sum();
        let err4: f64 = weights.iter().zip(q4.iter()).map(|(w, q)| (w - q).abs() as f64).sum();
        assert!(err4 > err8 * 2.0, "int4 must have >2× more error than int8: err4={err4:.4e} err8={err8:.4e}");
    }

    #[test]
    fn int4_max_representable_is_7() {
        let weights = vec![0.5_f32];
        let q4 = quantize_per_channel_int4(&weights, 1);
        // scale = 0.5/7, 0.5/scale = 7.0 → clamp 7 → dq = 7 × 0.5/7 = 0.5.
        assert!((q4[0] - 0.5_f32).abs() < 1e-5);
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
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn fingerprint_changes_on_content_change() {
        let f1 = Fixture { raw: "abc".to_string(), spikes: vec![], label: 0 };
        let fp1 = fixture_fingerprint(&[f1]);
        let f2 = Fixture { raw: "abd".to_string(), spikes: vec![], label: 0 };
        let fp2 = fixture_fingerprint(&[f2]);
        assert_ne!(fp1, fp2);
    }

    // ── derive_threshold ────────────────────────────────────────────────────

    #[test]
    fn threshold_sits_in_gap() {
        let r = derive_threshold(0.005, 0.020, 0.40);
        assert!(r.int8_passes, "int8 must pass");
        assert!(r.int4_fails, "int4 must fail");
        assert!(!r.gap_too_narrow);
        // threshold should be 0.005 + 0.40*(0.020-0.005) = 0.005+0.006 = 0.011
        assert!((r.threshold - 0.011).abs() < 1e-10);
    }

    #[test]
    fn threshold_flags_narrow_gap() {
        let r = derive_threshold(0.010, 0.011, 0.40);
        assert!(r.gap_too_narrow, "gap ratio 1.1× should be flagged");
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

    // ── Load artifact ──────────────────────────────────────────────────────────
    let artifact_json = match std::fs::read_to_string(&args.artifact) {
        Ok(s) => s,
        Err(e) => { eprintln!("E0201: cannot read artifact {}: {e}", args.artifact.display()); std::process::exit(1); }
    };
    let float_model = match load_from_str(&artifact_json) {
        Ok(m) => m,
        Err(e) => { eprintln!("E0201: artifact parse failed: {e}"); std::process::exit(1); }
    };

    // ── Build quantized models ─────────────────────────────────────────────────
    // apply_quantization passes (weights, out_features) per Dense layer — matches both functions.
    let int8_ch_model = apply_quantization(&float_model, quantize_per_channel_int8);
    let int4_ch_model = apply_quantization(&float_model, quantize_per_channel_int4);
    let int8_ts_model = apply_quantization(&float_model, |w, _| quantize_per_tensor_int8(w));

    // ── Load frozen fixtures ───────────────────────────────────────────────────
    let fixtures = match load_fixtures(&args.data_dir, args.n_samples) {
        Ok(f) => f,
        Err(e) => { eprintln!("Cannot load fixtures: {e}"); std::process::exit(1); }
    };

    let fingerprint = fixture_fingerprint(&fixtures);
    let n = fixtures.len();

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" ADR-0010 Part II — Ratification Measurement");
    println!(" artifact:     {}", args.artifact.display());
    println!(" fixture set:  {fingerprint}");
    if args.n_samples == 0 || args.n_samples >= n {
        println!("               (ALL committed fixtures in frozen/)");
    } else {
        println!("               (WARNING: only {n} of {} available — run without --n-samples to use all)",
            count_available_fixtures(&args.data_dir.join("frozen")));
    }
    println!(" n_samples:    {n}");
    println!(" comparison:   int8-per-channel (should PASS) vs int4-per-channel (should FAIL)");
    println!(" rounding:     round-half-to-even throughout");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ── Run all four conditions ────────────────────────────────────────────────
    print!("Running samples");
    std::io::stdout().flush().ok();

    let mut ref_rasters:    Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
    let mut int8_ch_rasters: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
    let mut int4_ch_rasters: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
    let mut int8_ts_rasters: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);

    for (i, fixture) in fixtures.iter().enumerate() {
        ref_rasters.push(run_sim(&float_model, &fixture.spikes));
        int8_ch_rasters.push(run_sim(&int8_ch_model, &fixture.spikes));
        int4_ch_rasters.push(run_sim(&int4_ch_model, &fixture.spikes));
        int8_ts_rasters.push(run_sim(&int8_ts_model, &fixture.spikes));
        if (i + 1) % 10 == 0 { print!(" {}", i + 1); std::io::stdout().flush().ok(); }
    }
    println!(" done.");
    println!();

    // ── Collect metrics ────────────────────────────────────────────────────────
    let m_int8 = collect_condition(&ref_rasters, &int8_ch_rasters);
    let m_int4 = collect_condition(&ref_rasters, &int4_ch_rasters);
    let m_ts   = collect_condition(&ref_rasters, &int8_ts_rasters);

    let n_neurons = if !ref_rasters.is_empty() && !ref_rasters[0].is_empty() {
        ref_rasters[0][0].len()
    } else { 0 };

    // ── Print distributions ────────────────────────────────────────────────────
    print_condition("FLOAT vs INT8-PER-CHANNEL  [should PASS]", &m_int8, n_neurons, n);
    print_condition("FLOAT vs INT4-PER-CHANNEL  [should FAIL]", &m_int4, n_neurons, n);

    println!("─── FLOAT vs INT8-PER-TENSOR  [informational] ──────────────────────");
    println!("  Harness aggregates: agg_mean={:.4e}  agg_max={:.4e}  pred_agree={:.4}",
        m_ts.agg_mean, m_ts.agg_max, m_ts.pred_agree);
    print_dist(&m_ts.all_neuron_errors, "per-neuron error");
    print_dist(&m_ts.per_sample_mean, "per-sample mean");
    print_dist(&m_ts.per_sample_max, "per-sample max");
    let ts_ch_p99_ratio = if percentile(&m_int8.all_neuron_errors, 99.0) > 0.0 {
        percentile(&m_ts.all_neuron_errors, 99.0) / percentile(&m_int8.all_neuron_errors, 99.0)
    } else { f64::INFINITY };
    println!("  Gap int8-tensor p99 / int8-channel p99 = {:.4e} / {:.4e} = {:.1}×",
        percentile(&m_ts.all_neuron_errors, 99.0),
        percentile(&m_int8.all_neuron_errors, 99.0),
        ts_ch_p99_ratio);
    println!();

    // ── Gap analysis ───────────────────────────────────────────────────────────
    // Thresholds are placed 40% into the gap from the int8 side.
    const GAP_FRAC: f64 = 0.40;

    let t_mean_gap = derive_threshold(m_int8.agg_mean, m_int4.agg_mean, GAP_FRAC);
    let t_max_gap  = derive_threshold(m_int8.agg_max,  m_int4.agg_max,  GAP_FRAC);

    // For P_min, we use boundary-case agreement as the conservative anchor (lower bound).
    let int8_pred_anchor = m_int8.boundary_agree_frac.unwrap_or(m_int8.pred_agree);
    let int4_pred_anchor = m_int4.boundary_agree_frac.unwrap_or(m_int4.pred_agree);
    // P_min must be ≤ int8 and > int4 (both overall and boundary).
    // The anchor for int8 is the lower of overall/boundary; for int4 the higher.
    let int8_pred_lower = m_int8.pred_agree.min(int8_pred_anchor);
    let int4_pred_upper = m_int4.pred_agree.max(int4_pred_anchor);
    // Place P_min 40% into the gap (higher P_min → harder to pass).
    let p_min = if int8_pred_lower > int4_pred_upper {
        int4_pred_upper + (int8_pred_lower - int4_pred_upper) * (1.0 - GAP_FRAC)
    } else {
        // No separation — use int8_lower with small margin
        (int8_pred_lower * 0.95).min(0.99)
    };
    let p_min_int8_passes = int8_pred_lower >= p_min;
    let p_min_int4_fails  = int4_pred_upper <  p_min;

    println!("─── GAP ANALYSIS ────────────────────────────────────────────────────");
    println!("  Thresholds placed {:.0}% into the gap from the int8 side.", GAP_FRAC * 100.0);
    println!("  (int8 gets {:.0}% of gap as headroom; int4 gets {:.0}% as failure margin)",
        (1.0 - GAP_FRAC) * 100.0, GAP_FRAC * 100.0);
    println!();

    println!("  T_MEAN  (harness uses: mean of per-sample mean_rate_errors)");
    println!("    int8 agg_mean = {:.4e}   (must be ≤ T_mean for int8 to PASS)", m_int8.agg_mean);
    println!("    int4 agg_mean = {:.4e}   (must be >  T_mean for int4 to FAIL)", m_int4.agg_mean);
    println!("    Ratio int4/int8 = {:.2}×", t_mean_gap.gap_ratio);
    if t_mean_gap.gap_too_narrow {
        println!("    ⚠ GAP TOO NARROW — int4 and int8 are too close; cannot draw a clean line.");
        println!("      Falling back to int8_agg_mean × 1.5 = {:.4e}.", t_mean_gap.threshold);
    } else {
        println!("    Gap candidate: {:.4e} + {:.4e} × {:.2} = {:.4e}",
            m_int8.agg_mean, m_int4.agg_mean - m_int8.agg_mean, GAP_FRAC, t_mean_gap.threshold);
        println!("    int8 headroom:  +{:.1}%  (threshold / int8_agg_mean - 1)", t_mean_gap.int8_headroom_pct);
        println!("    int4 margin:    -{:.1}%  (1 - threshold / int4_agg_mean)", t_mean_gap.int4_margin_pct);
    }
    println!("    {} int8 PASSES  {} int4 FAILS",
        check(t_mean_gap.int8_passes), check(t_mean_gap.int4_fails));
    println!();

    println!("  T_MAX   (harness uses: max of per-sample max_rate_errors across all samples)");
    println!("    int8 agg_max  = {:.4e}   (must be ≤ T_max for int8 to PASS)", m_int8.agg_max);
    println!("    int4 agg_max  = {:.4e}   (must be >  T_max for int4 to FAIL)", m_int4.agg_max);
    println!("    Ratio int4/int8 = {:.2}×", t_max_gap.gap_ratio);
    if t_max_gap.gap_too_narrow {
        println!("    ⚠ GAP TOO NARROW for T_max. int8 barely passes; int4 barely fails.");
        println!("      Candidate: {:.4e} (int8_agg_max × 1.5 fallback).", t_max_gap.threshold);
    } else {
        println!("    Gap candidate: {:.4e} + {:.4e} × {:.2} = {:.4e}",
            m_int8.agg_max, m_int4.agg_max - m_int8.agg_max, GAP_FRAC, t_max_gap.threshold);
        println!("    int8 headroom:  +{:.1}%", t_max_gap.int8_headroom_pct);
        println!("    int4 margin:    -{:.1}%", t_max_gap.int4_margin_pct);
    }
    println!("    {} int8 PASSES  {} int4 FAILS",
        check(t_max_gap.int8_passes), check(t_max_gap.int4_fails));
    println!();

    println!("  P_MIN   (harness uses: prediction_agreement fraction)");
    println!("    int8 overall={:.4}  boundary={:.4}  anchor={:.4}",
        m_int8.pred_agree, m_int8.boundary_agree_frac.unwrap_or(f64::NAN), int8_pred_lower);
    println!("    int4 overall={:.4}  boundary={:.4}  anchor={:.4}",
        m_int4.pred_agree, m_int4.boundary_agree_frac.unwrap_or(f64::NAN), int4_pred_upper);
    println!("    Candidate P_min = {:.4}", p_min);
    println!("    {} int8 PASSES  {} int4 FAILS",
        check(p_min_int8_passes), check(p_min_int4_fails));
    println!();

    // ── Recommended thresholds ────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" RECOMMENDED FINAL THRESHOLDS (gap-derived, not multiplier)");
    println!(" Paste into ADR-0010 Part II Amendment. Founder approval required.");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Fixture set fingerprint: {fingerprint}");
    println!("  Ratified against: per-channel int8 (should-pass), per-channel int4 (should-fail).");
    println!("  Quantization: symmetric, round-half-to-even, per-output-channel scale.");
    println!("  Scope: one reference model (SHD 64.66%, Dense 700→512→20). v0 envelope.");
    println!();
    println!("  T_mean_threshold = {:.4e}", t_mean_gap.threshold);
    println!("    {} int8 agg_mean = {:.4e}  ≤  {:.4e} → PASSES",
        check(t_mean_gap.int8_passes), m_int8.agg_mean, t_mean_gap.threshold);
    println!("    {} int4 agg_mean = {:.4e}  >  {:.4e} → FAILS",
        check(t_mean_gap.int4_fails), m_int4.agg_mean, t_mean_gap.threshold);
    println!();
    println!("  T_max_threshold  = {:.4e}", t_max_gap.threshold);
    println!("    {} int8 agg_max  = {:.4e}  ≤  {:.4e} → PASSES",
        check(t_max_gap.int8_passes), m_int8.agg_max, t_max_gap.threshold);
    println!("    {} int4 agg_max  = {:.4e}  >  {:.4e} → FAILS",
        check(t_max_gap.int4_fails), m_int4.agg_max, t_max_gap.threshold);
    println!();
    println!("  pred_agreement_min = {:.4}", p_min);
    println!("    {} int8 pred_agree = {:.4}  ≥  {:.4} → PASSES",
        check(p_min_int8_passes), int8_pred_lower, p_min);
    println!("    {} int4 pred_agree = {:.4}  <  {:.4} → FAILS",
        check(p_min_int4_fails), int4_pred_upper, p_min);
    println!();

    let overall_ok = t_mean_gap.int8_passes && t_mean_gap.int4_fails
        && t_max_gap.int8_passes && t_max_gap.int4_fails
        && p_min_int8_passes && p_min_int4_fails;
    if overall_ok {
        println!("  ✓ All six conditions satisfied — clean int8/int4 separation.");
        println!("    The above thresholds are suitable for ADR-0010 Part II Amendment.");
    } else {
        println!("  ✗ PROBLEM: not all conditions satisfied — see gap analysis above.");
        println!("    Do NOT ratify until each condition shows ✓.");
    }
    println!();

    // ── Comparison with DRAFT ─────────────────────────────────────────────────
    let draft = &conformance::CONFORMANCE_ENVELOPE_V0_DRAFT;
    println!("─── Comparison with CONFORMANCE_ENVELOPE_v0_DRAFT ──────────────────");
    println!("                     INT8 measured   DRAFT   Status");
    let s_mean = if m_int8.agg_mean <= draft.t_mean_threshold { "OK" } else { "EXCEEDS" };
    let s_max  = if m_int8.agg_max  <= draft.t_max_threshold  { "OK" } else { "EXCEEDS" };
    let s_pred = if m_int8.pred_agree >= draft.pred_agreement_min { "OK" } else { "BELOW" };
    println!("  agg_mean:          {:.4e}       {:.4e}   {}",
        m_int8.agg_mean, draft.t_mean_threshold, s_mean);
    println!("  agg_max:           {:.4e}       {:.4e}   {}",
        m_int8.agg_max, draft.t_max_threshold, s_max);
    println!("  pred_agree:        {:.4}         {:.4}   {}",
        m_int8.pred_agree, draft.pred_agreement_min, s_pred);
    println!();
    println!("NOTE: This output does not constitute certification.");
    println!("ADR-0010 Part II must be amended by the founder before any backend");
    println!("can claim THRINDEX Certified. DRAFT envelope remains in effect.");
}
