// crates/thrindex-backends/akida/runtime/src/main.rs
//
// akida-runtime — Tier-2 device-side inference binary.
// Built only when `--features hardware` is active.
//
// CLI
// ───
//   akida-runtime <fbz_path> [--pcie-addr ADDR]
//
//   fbz_path:   path to the .fbz produced by akida_compile.py
//   --pcie-addr ADDR: PCIe device address (default "0001:01:00.0")
//
// stdin  (JSON, newline-terminated or until EOF)
// ──────
//   {"batch": [[[f32,...]], ...]}
//   Shape: [N_samples][T=1][n_features]
//   T must be exactly 1; AKD1500 is stateless single-frame (E0407 if T≠1).
//
// stdout on success
// ─────────────────
//   {"outputs": [[f32,...], ...]}
//   Shape: [N_samples][out_features]
//   exit code 0
//
// stdout on error
// ───────────────
//   {"error": "E04XX: <human-readable message>"}
//   exit code 1
//
// Input quantization (f32 → u8)
// ─────────────────────────────
//   4-bit feature encoding into a u8 container:
//     step 1:  v4 = clamp(round(f32 * 15.0), 0, 15) as u8   [0..15 range]
//     step 2:  u8 = v4 * 17                                   [maps 15→255]
//   This matches InputData(input_bits=4) in akida_compile.py.

use std::process;

// ── Stub entry point (hardware feature on but THRINDEX_AKIDA_ENGINE_PATH not set) ──

#[cfg(not(akida_engine_available))]
fn main() {
    eprintln!(
        "akida-runtime: built without Engine Library \
         (THRINDEX_AKIDA_ENGINE_PATH was not set at compile time). \
         Rebuild with THRINDEX_AKIDA_ENGINE_PATH pointing to the Engine Library source."
    );
    process::exit(1);
}

// ── Full implementation (Engine Library present at build time) ─────────────────

#[cfg(akida_engine_available)]
use std::io::{self, Read};

#[cfg(akida_engine_available)]
use serde::{Deserialize, Serialize};

#[cfg(akida_engine_available)]
use thrindex_backend_akida::ffi;

// ── JSON protocol types ───────────────────────────────────────────────────────

#[cfg(akida_engine_available)]
#[derive(Deserialize)]
struct BatchInput {
    /// [N_samples][timesteps][features]
    batch: Vec<Vec<Vec<f32>>>,
}

#[cfg(akida_engine_available)]
#[derive(Serialize)]
struct BatchOutput {
    /// [N_samples][out_features]
    outputs: Vec<Vec<f32>>,
}

#[cfg(akida_engine_available)]
#[derive(Serialize)]
struct ErrorOutput {
    error: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[cfg(akida_engine_available)]
fn main() {
    let result = run_from_args();
    match result {
        Ok(output) => {
            let json = serde_json::to_string(&output).expect("serialize output");
            println!("{json}");
            process::exit(0);
        }
        Err(msg) => {
            let json = serde_json::to_string(&ErrorOutput { error: msg }).expect("serialize error");
            println!("{json}");
            process::exit(1);
        }
    }
}

#[cfg(akida_engine_available)]
fn run_from_args() -> Result<BatchOutput, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut fbz_path: Option<String> = None;
    let mut pcie_addr = String::from("0001:01:00.0");

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--pcie-addr" if i + 1 < args.len() => {
                pcie_addr = args[i + 1].clone();
                i += 2;
            }
            flag if flag.starts_with("--") => {
                return Err(format!("E0204: unknown flag {flag}"));
            }
            _ => {
                fbz_path = Some(args[i].clone());
                i += 1;
            }
        }
    }

    let fbz_path = fbz_path
        .ok_or_else(|| "E0204: usage: akida-runtime <fbz_path> [--pcie-addr ADDR]".to_owned())?;

    let mut stdin_buf = String::new();
    io::stdin()
        .read_to_string(&mut stdin_buf)
        .map_err(|e| format!("E0204: cannot read stdin: {e}"))?;

    run(&fbz_path, &pcie_addr, &stdin_buf)
}

// ── Core inference logic ──────────────────────────────────────────────────────

#[cfg(akida_engine_available)]
fn run(fbz_path: &str, pcie_addr: &str, stdin_json: &str) -> Result<BatchOutput, String> {
    let input: BatchInput =
        serde_json::from_str(stdin_json).map_err(|e| format!("E0204: invalid stdin JSON: {e}"))?;

    for (idx, sample) in input.batch.iter().enumerate() {
        if sample.len() != 1 {
            return Err(format!(
                "E0407 [temporal-input-not-supported]: \
                 sample[{idx}] has T={t} timesteps; \
                 AKD1500 is a stateless single-frame backend (T must equal 1); \
                 split the spike train into individual frames before calling run_batch.",
                t = sample.len()
            ));
        }
    }

    if input.batch.is_empty() {
        return Ok(BatchOutput { outputs: vec![] });
    }

    let n_samples = input.batch.len();
    let n_features = input.batch[0][0].len();

    let mut input_u8: Vec<u8> = Vec::with_capacity(n_samples * n_features);
    for sample in &input.batch {
        for &v in &sample[0] {
            input_u8.push(quantize_f32_to_u8(v));
        }
    }

    let fbz = std::fs::read(fbz_path)
        .map_err(|e| format!("E0204: cannot read .fbz at {fbz_path}: {e}"))?;
    if fbz.is_empty() {
        return Err(format!("E0204: .fbz at {fbz_path} is empty"));
    }

    let mut device = ffi::create_pcie_device(pcie_addr)
        .map_err(|e| format!("E0204: device init failed for PCIe addr {pcie_addr}: {e}"))?;

    ffi::device_program(device.pin_mut(), &fbz)
        .map_err(|e| format!("E0204: device program failed: {e}"))?;

    let output_floats = ffi::akida_run_batch(
        device.pin_mut(),
        &input_u8,
        n_samples as u32,
        n_features as u32,
    )
    .map_err(|e| format!("E0204: inference failed: {e}"))?;

    if output_floats.is_empty() {
        return Err("E0204: akida_run_batch returned empty output".to_owned());
    }
    let out_features = output_floats.len() / n_samples;
    if out_features == 0 {
        return Err(format!(
            "E0204: output length {} is not divisible by n_samples {n_samples}",
            output_floats.len()
        ));
    }

    let outputs: Vec<Vec<f32>> = output_floats
        .chunks(out_features)
        .map(|c: &[f32]| c.to_vec())
        .collect();

    Ok(BatchOutput { outputs })
}

// ── Quantization helper ───────────────────────────────────────────────────────

/// f32 (range [0, 1]) → u8 (4-bit value × 17)
///
/// InputData(input_bits=4) represents features as 4-bit unsigned integers (0..15).
/// The Engine Library expects them as u8 values where 4-bit n maps to u8 n×17:
///   0 → 0,  1 → 17,  7 → 119,  8 → 136,  15 → 255
/// This is the standard AKD1500 encoding for 4-bit inputs.
#[cfg(any(akida_engine_available, test))]
#[inline]
fn quantize_f32_to_u8(v: f32) -> u8 {
    let v4 = (v * 15.0).round().clamp(0.0, 15.0) as u8;
    v4.saturating_mul(17)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_boundaries() {
        assert_eq!(quantize_f32_to_u8(0.0), 0);
        assert_eq!(quantize_f32_to_u8(1.0), 255);
        assert_eq!(quantize_f32_to_u8(0.5), 136);
        assert_eq!(quantize_f32_to_u8(-1.0), 0);
        assert_eq!(quantize_f32_to_u8(2.0), 255);
    }

    #[cfg(akida_engine_available)]
    #[test]
    fn t_not_1_rejected() {
        let json = r#"{"batch": [[[0.1, 0.2], [0.3, 0.4]]]}"#; // T=2
        let result = run("dummy.fbz", "0001:01:00.0", json);
        let err = result.unwrap_err();
        assert!(err.starts_with("E0407"), "expected E0407, got: {err}");
    }

    #[cfg(akida_engine_available)]
    #[test]
    fn empty_batch_returns_empty() {
        let json = r#"{"batch": []}"#;
        let output = run("dummy.fbz", "0001:01:00.0", json).unwrap();
        assert!(output.outputs.is_empty());
    }
}
