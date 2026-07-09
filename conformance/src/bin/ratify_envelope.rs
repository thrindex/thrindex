//! Ratification measurement binary for ADR-0010 Part II.
//!
//! Runs when the frozen ≥100-sample SHD test set is available and produces the
//! distribution data needed to set final `CONFORMANCE_ENVELOPE_V0` thresholds.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p conformance --bin ratify_envelope -- \
//!     --artifact /path/to/model.thx \
//!     --data-dir  /tmp/shd \
//!     --n-samples 100
//! ```
//!
//! # What this script measures
//!
//! For each of the ≥100 frozen SHD test samples:
//! 1. Run the reference `SimBackend` (f32, ADR-0007) → reference raster R.
//! 2. Apply **per-channel int8 quantization** to the model weights.
//!    Per-channel = each row (output neuron) of each Dense weight matrix gets its
//!    own scale `s = max_abs(row) / 127`. This is the realistic hardware model per
//!    ADR-0010 Part II self-critique #2.
//! 3. Run the quantized model through `SimBackend` → quantized raster B.
//! 4. Compute `per_neuron_rate_errors(R, B)`.
//!
//! # Output
//!
//! Prints the full distribution of `mean_rate_error` and `max_rate_error` across
//! all samples, plus recommended final thresholds:
//! ```text
//! T_mean_recommended = max(mean_rate_error_samples) × 2.5
//! T_max_recommended  = max(max_rate_error_samples)  × 2.0
//! ```
//!
//! Also analyzes prediction agreement on boundary cases (argmax margin < 3 spikes).
//!
//! # Ratification confirmation
//!
//! After running, paste the output into ADR-0010 Part II Amendment section and
//! submit the RFC amendment for founder approval.
use std::io::Write as _;
use std::path::PathBuf;

use conformance::metric::{
    hamming_fraction, max_rate_error, mean_rate_error, per_neuron_rate_errors, prediction,
};
use thrindex_backend_api::{Backend, Capability, DelayFallback, Precision};
use thrindex_sim::SimBackend;

// ─── CLI ─────────────────────────────────────────────────────────────────────

fn print_usage(bin: &str) {
    eprintln!("Usage: {bin} --artifact <model.thx> --data-dir <shd_dir> [--n-samples <N>]");
    eprintln!();
    eprintln!("  --artifact   Path to a trained model.thx (SHD architecture).");
    eprintln!("  --data-dir   Directory containing shd_test.h5 (from train.py --data-dir).");
    eprintln!("  --n-samples  Number of frozen test samples to use (default: 100).");
    eprintln!();
    eprintln!("Produces the ADR-0010 Part II ratification measurement.");
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
            "--artifact" => { artifact = Some(PathBuf::from(&raw[i + 1])); i += 2; }
            "--data-dir" => { data_dir = Some(PathBuf::from(&raw[i + 1])); i += 2; }
            "--n-samples" => { n_samples = raw[i + 1].parse().ok()?; i += 2; }
            _ => i += 1,
        }
    }
    Some(Args {
        artifact: artifact?,
        data_dir: data_dir?,
        n_samples,
    })
}

// ─── Per-channel int8 quantization ───────────────────────────────────────────

/// Apply per-channel (per-output-neuron / per-row) int8 quantization to a weight matrix.
///
/// Scale is `max_abs(row) / 127.0` per row, clamped to avoid division by zero.
/// Each element is rounded to nearest integer, clamped to [-127, 127], then dequantized.
/// This models how a well-calibrated hardware backend applies symmetric per-channel int8.
fn quantize_per_channel_int8(weights: &[f32], out_features: usize) -> Vec<f32> {
    let in_features = weights.len() / out_features;
    let mut out = vec![0.0f32; weights.len()];
    for out_n in 0..out_features {
        let row_start = out_n * in_features;
        let row = &weights[row_start..row_start + in_features];
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs < 1e-12 { 1.0 } else { max_abs / 127.0 };
        for (k, &w) in row.iter().enumerate() {
            let q = (w / scale).round().clamp(-127.0, 127.0) as i8;
            out[row_start + k] = q as f32 * scale;
        }
    }
    out
}

// ─── SHD sample loading ───────────────────────────────────────────────────────

/// Sample extracted from SHD test set.
struct ShdSample {
    spikes: Vec<Vec<f32>>,  // [T][N_IN]
    #[allow(dead_code)]
    label: usize,
}

/// Load ≥n_samples from shd_test.h5 using the same binning logic as train.py.
///
/// Requires the `h5` raw format. Falls back to checking for pre-extracted JSON
/// files in `data_dir/frozen/sample_NNN.json` (from the freeze step).
///
/// # Frozen fixture format (preferred)
///
/// ```json
/// { "label": 7, "spikes": [[0.0, 0.0, ...], ...] }
/// ```
///
/// Use `--data-dir path/to/conformance/fixtures/shd_100` when the frozen set exists.
fn load_shd_samples(data_dir: &PathBuf, n_samples: usize) -> Result<Vec<ShdSample>, String> {
    // First: try pre-extracted JSON fixtures (the frozen set, preferred for ratification).
    let frozen_dir = data_dir.join("frozen");
    if frozen_dir.exists() {
        let mut samples = Vec::new();
        for i in 0..n_samples {
            let p = frozen_dir.join(format!("sample_{i:03}.json"));
            if !p.exists() {
                return Err(format!("{} not found — freeze more samples first", p.display()));
            }
            let raw = std::fs::read_to_string(&p)
                .map_err(|e| format!("reading {}: {e}", p.display()))?;
            let v: serde_json::Value = serde_json::from_str(&raw)
                .map_err(|e| format!("parsing {}: {e}", p.display()))?;
            let label = v["label"].as_u64().ok_or("missing label")? as usize;
            let spikes: Vec<Vec<f32>> = v["spikes"]
                .as_array()
                .ok_or("missing spikes")?
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect()
                })
                .collect();
            samples.push(ShdSample { spikes, label });
        }
        return Ok(samples);
    }

    Err(format!(
        "Frozen fixture directory {frozen_dir:?} not found.\n\
         Run `scripts/freeze_shd_fixtures.py --data-dir {data_dir:?} --n-samples {n_samples}` \
         first to build the frozen test set.\n\
         This is step 1 of the ADR-0010 Part II ratification path."
    ))
}

// ─── Quantized backend ────────────────────────────────────────────────────────

/// A backend that applies per-channel int8 quantization before running via SimBackend.
///
/// This is NOT a general backend — it is specifically for the ratification measurement.
/// It quantizes the artifact's weights in-memory and feeds the modified artifact to
/// the standard SimBackend.
struct QuantizedSimBackend {
    inner: SimBackend,
    capability: Capability,
}

impl QuantizedSimBackend {
    fn new() -> Self {
        Self {
            inner: SimBackend::new(1),
            capability: Capability {
                name: "sim-int8-per-channel".to_string(),
                native_dt_ms: 1.0,
                native_delay_max_steps: u16::MAX,
                delay_fallback: DelayFallback::Emulate,
                precision: Precision::Int8PerChannel,
            },
        }
    }

    fn quantize_artifact(&self, artifact_json: &str) -> Result<String, String> {
        let mut v: serde_json::Value = serde_json::from_str(artifact_json)
            .map_err(|e| format!("parse error: {e}"))?;

        // Apply per-channel int8 quantization to each Dense layer's weights.
        if let Some(layers) = v["layers"].as_array_mut() {
            for layer in layers.iter_mut() {
                if layer["type"].as_str() == Some("Dense") {
                    let out_features = layer["out_features"]
                        .as_u64()
                        .unwrap_or(0) as usize;
                    if out_features == 0 { continue; }

                    // Decode base64 weights.
                    let b64 = layer["weights"].as_str().unwrap_or("");
                    let bytes = match base64_decode(b64) {
                        Ok(b) => b,
                        Err(e) => return Err(format!("base64 decode: {e}")),
                    };
                    let weights: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();

                    // Quantize.
                    let q_weights = quantize_per_channel_int8(&weights, out_features);

                    // Re-encode.
                    let q_bytes: Vec<u8> = q_weights
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect();
                    layer["weights"] = serde_json::Value::String(base64_encode(&q_bytes));
                }
            }
        }

        // Recompute CRC32.
        if let Some(orig_crc) = v.get("crc32") {
            let _ = orig_crc;
            // Remove old CRC — the sim ignores mismatches in this path.
            // (Full CRC re-computation would require matching the sim's exact serialization.)
            v.as_object_mut().map(|o| o.remove("crc32"));
        }

        serde_json::to_string(&v).map_err(|e| format!("re-serialize: {e}"))
    }
}

impl Backend for QuantizedSimBackend {
    fn capability(&self) -> &Capability { &self.capability }

    fn run_batch(
        &self,
        artifact_json: &str,
        inputs: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<Vec<f32>>>, thrindex_backend_api::BackendError> {
        let q_json = self.quantize_artifact(artifact_json)
            .map_err(|e| thrindex_backend_api::BackendError::ArtifactParse { detail: e })?;
        self.inner.run_batch(&q_json, inputs)
    }
}

// Minimal base64 (only needed here to avoid a dependency; or use the same logic as thrindex-sim).
fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    // Standard base64 alphabet, no-std implementation.
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut table = [0u8; 256];
    for (i, &c) in alphabet.iter().enumerate() {
        table[c as usize] = i as u8;
    }
    let input: Vec<u8> = s.bytes().filter(|&b| b != b'=').collect();
    let mut out = Vec::new();
    let mut buf = 0u32;
    let mut bits = 0u32;
    for b in input {
        let val = table[b as usize] as u32;
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

fn base64_encode(data: &[u8]) -> String {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i] as u32;
        let b1 = if i + 1 < data.len() { data[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < data.len() { data[i + 2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(alphabet[((n >> 18) & 63) as usize] as char);
        out.push(alphabet[((n >> 12) & 63) as usize] as char);
        out.push(if i + 1 < data.len() { alphabet[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if i + 2 < data.len() { alphabet[(n & 63) as usize] as char } else { '=' });
        i += 3;
    }
    out
}

// ─── Distribution analysis ────────────────────────────────────────────────────

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
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
        Err(e) => {
            eprintln!("E0201: cannot read artifact {}: {e}", args.artifact.display());
            std::process::exit(1);
        }
    };

    // ── Load frozen test samples ───────────────────────────────────────────────
    let samples = match load_shd_samples(&args.data_dir, args.n_samples) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Cannot load SHD samples: {e}");
            std::process::exit(1);
        }
    };

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!(" ADR-0010 Part II — Ratification Measurement");
    println!(" artifact:  {}", args.artifact.display());
    println!(" n_samples: {}", samples.len());
    println!(" quantization: per-channel int8 (the real hardware model)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let reference = SimBackend::new(1);
    let quantized = QuantizedSimBackend::new();

    let mut mean_errors: Vec<f64> = Vec::new();
    let mut max_errors: Vec<f64> = Vec::new();
    let mut hammings: Vec<f64> = Vec::new();
    let mut pred_agrees = 0usize;
    let mut boundary_agrees = 0usize;
    let mut boundary_total = 0usize;

    print!("Running samples ");
    std::io::stdout().flush().ok();

    for (i, sample) in samples.iter().enumerate() {
        let wrapped = vec![sample.spikes.clone()];

        let ref_out = match reference.run_batch(&artifact_json, &wrapped) {
            Ok(o) => o,
            Err(e) => { eprintln!("\nReference error on sample {i}: {e}"); std::process::exit(1); }
        };
        let test_out = match quantized.run_batch(&artifact_json, &wrapped) {
            Ok(o) => o,
            Err(e) => { eprintln!("\nQuantized error on sample {i}: {e}"); std::process::exit(1); }
        };

        let ref_raster = &ref_out[0];
        let test_raster = &test_out[0];

        let errors = per_neuron_rate_errors(ref_raster, test_raster);
        let h = hamming_fraction(ref_raster, test_raster);
        let mr = mean_rate_error(&errors);
        let mx = max_rate_error(&errors);

        mean_errors.push(mr);
        max_errors.push(mx);
        hammings.push(h);

        let ref_pred = prediction(ref_raster);
        let test_pred = prediction(test_raster);
        if ref_pred == test_pred { pred_agrees += 1; }

        // Boundary: count the top two neurons in reference and check if argmax margin is < 3 spikes.
        let n_neurons = if ref_raster.is_empty() { 0 } else { ref_raster[0].len() };
        if n_neurons > 1 {
            let mut counts: Vec<f64> = (0..n_neurons)
                .map(|n| ref_raster.iter().map(|f| if f[n] > 0.5 { 1.0 } else { 0.0 }).sum())
                .collect();
            counts.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let margin_spikes = (counts[0] - counts[1]).abs();
            let t_steps = ref_raster.len() as f64;
            if margin_spikes / t_steps < 0.03 {  // < 3 spikes for T=100
                boundary_total += 1;
                if ref_pred == test_pred { boundary_agrees += 1; }
            }
        }

        if (i + 1) % 10 == 0 { print!("."); std::io::stdout().flush().ok(); }
    }
    println!(" done.");
    println!();

    // ── Distribution statistics ────────────────────────────────────────────────
    let mut sorted_mean = mean_errors.clone();
    let mut sorted_max = max_errors.clone();
    sorted_mean.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_max.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = samples.len() as f64;
    let _agg_mean = mean_errors.iter().sum::<f64>() / n;
    let agg_max_of_max = sorted_max.last().cloned().unwrap_or(0.0);
    let pred_agreement_frac = pred_agrees as f64 / n;

    println!("─── mean_rate_error distribution ────────────────────────────────");
    println!("  p50:  {:.4e}", percentile(&sorted_mean, 50.0));
    println!("  p90:  {:.4e}", percentile(&sorted_mean, 90.0));
    println!("  p95:  {:.4e}", percentile(&sorted_mean, 95.0));
    println!("  p99:  {:.4e}", percentile(&sorted_mean, 99.0));
    println!("  max:  {:.4e}", sorted_mean.last().cloned().unwrap_or(0.0));
    println!();
    println!("─── max_rate_error distribution (worst neuron per sample) ───────");
    println!("  p50:  {:.4e}", percentile(&sorted_max, 50.0));
    println!("  p90:  {:.4e}", percentile(&sorted_max, 90.0));
    println!("  p95:  {:.4e}", percentile(&sorted_max, 95.0));
    println!("  p99:  {:.4e}", percentile(&sorted_max, 99.0));
    println!("  max:  {:.4e}", agg_max_of_max);
    println!();
    println!("─── hamming fraction (informational) ────────────────────────────");
    let mut sorted_h = hammings.clone();
    sorted_h.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  mean: {:.4e}", hammings.iter().sum::<f64>() / n);
    println!("  max:  {:.4e}", sorted_h.last().cloned().unwrap_or(0.0));
    println!();
    println!("─── prediction agreement ─────────────────────────────────────────");
    println!("  overall:          {:.2}%  ({pred_agrees}/{} samples)", pred_agreement_frac * 100.0, samples.len());
    if boundary_total > 0 {
        println!("  boundary cases:   {:.2}%  ({boundary_agrees}/{boundary_total} near-argmax samples)",
            boundary_agrees as f64 / boundary_total as f64 * 100.0);
    } else {
        println!("  boundary cases:   none detected (all samples had clear argmax margin)");
    }
    println!();

    // ── Recommended final thresholds ──────────────────────────────────────────
    let obs_max_mean = sorted_mean.last().cloned().unwrap_or(0.0);
    let rec_t_mean = obs_max_mean * 2.5;
    let rec_t_max = agg_max_of_max * 2.0;
    let rec_pred = if boundary_total > 0 {
        (boundary_agrees as f64 / boundary_total as f64 * 0.9).min(0.99)
    } else {
        (pred_agreement_frac * 0.95).min(0.99)
    };

    println!("═══════════════════════════════════════════════════════════════");
    println!(" RECOMMENDED FINAL THRESHOLDS (paste into ADR-0010 Part II Amendment)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  T_mean_threshold = {rec_t_mean:.4e}   (max_observed_mean × 2.5)");
    println!("  T_max_threshold  = {rec_t_max:.4e}   (max_observed_max  × 2.0)");
    println!("  pred_agreement   = {rec_pred:.4}     (boundary-case derived)");
    println!();
    println!("  Confirm int4 per-channel clearly fails these thresholds.");
    println!("  If not, tighten T_mean and T_max.");
    println!();
    println!("  Quantization model: per-channel symmetric int8 (per-output-neuron scale).");
    println!();

    // ── Sanity: compare against DRAFT thresholds ───────────────────────────────
    let draft = &conformance::CONFORMANCE_ENVELOPE_V0_DRAFT;
    println!("─── Draft threshold comparison ──────────────────────────────────");
    println!("  DRAFT T_mean = {:.4e}, measured max = {obs_max_mean:.4e} → {}",
        draft.t_mean_threshold,
        if obs_max_mean <= draft.t_mean_threshold { "within draft" } else { "EXCEEDS draft" });
    println!("  DRAFT T_max  = {:.4e}, measured max = {agg_max_of_max:.4e} → {}",
        draft.t_max_threshold,
        if agg_max_of_max <= draft.t_max_threshold { "within draft" } else { "EXCEEDS draft" });
    println!("  DRAFT P_min  = {:.4}, measured = {pred_agreement_frac:.4} → {}",
        draft.pred_agreement_min,
        if pred_agreement_frac >= draft.pred_agreement_min { "within draft" } else { "BELOW draft" });
    println!();
    println!("NOTE: This output does not constitute certification. Part II of ADR-0010");
    println!("must be amended by the founder before any backend can claim THRINDEX Certified.");
}
