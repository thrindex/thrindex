//! BrainChip AKD1500 backend implementation of [`thrindex_backend_api::Backend`].
//!
//! ## Responsibilities
//!
//! - Declare AKD1500 capability via `AkidaBackend::capability()`.
//! - Run pre-flight validation of `.thx` artifacts (E0401–E0404).
//! - Guard against temporal inputs with T > 1 (E0407).
//! - Without `hardware` feature: return `Err(E0204)` stub after pre-flight.
//! - With `hardware` feature: spawn the `akida-runtime` binary and parse its
//!   JSON response.
//!
//! ## Subprocess protocol (hardware feature)
//!
//! `run_batch` finds `akida-runtime` via:
//!   1. `THRINDEX_AKIDA_RUNTIME` environment variable
//!   2. Sibling of the current executable
//!   3. `"akida-runtime"` on `$PATH`
//!
//! PCIe address comes from `THRINDEX_AKIDA_DEVICE` (default `"0001:01:00.0"`).
use std::path::{Path, PathBuf};

use thrindex_backend_api::{Backend, BackendError, Capability};

use crate::capability::akd1500_capability;
use crate::error::AkidaError;
use crate::validate::validate_artifact;

/// AKD1500 backend handle.
///
/// Constructed with a path to the compiled `.fbz` program produced by Tier-1
/// (`python/akida_compile.py`).
pub struct AkidaBackend {
    fbz_path: PathBuf,
    capability: Capability,
}

impl AkidaBackend {
    /// Construct a backend handle pointing at a compiled `.fbz` program.
    ///
    /// Does not open or validate the file at construction time; the first
    /// `run_batch` call will do that via the Engine Library subprocess.
    pub fn new(fbz_path: &Path) -> Self {
        AkidaBackend {
            fbz_path: fbz_path.to_owned(),
            capability: akd1500_capability(),
        }
    }
}

impl Backend for AkidaBackend {
    fn capability(&self) -> &Capability {
        &self.capability
    }

    /// Run inference on a batch of inputs.
    ///
    /// # Pre-flight checks (always run, regardless of feature flags)
    ///
    /// 1. Validates the `.thx` artifact for AKD1500 compatibility (E0401–E0404).
    /// 2. Rejects any input with `T > 1` timesteps (E0407).
    ///
    /// # Hardware path (`hardware` feature)
    ///
    /// Spawns `akida-runtime <fbz_path> --pcie-addr <ADDR>` with the
    /// serialised batch JSON on stdin, parses the JSON response from stdout,
    /// and wraps each output row as `[[f32; out_features]]` (T=1).
    ///
    /// # Stub path (no `hardware` feature)
    ///
    /// Returns `Err(E0204)` after pre-flight checks.
    fn run_batch(
        &self,
        artifact_json: &str,
        inputs: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<Vec<f32>>>, BackendError> {
        // Step 1: pre-flight artifact validation (E0401–E0404).
        validate_artifact(artifact_json)?;

        // Step 2: E0407 — reject T > 1 temporal sequences.
        if let Some(sample) = inputs.first()
            && sample.len() != 1
        {
            let batch = inputs.len();
            let timesteps = sample.len();
            let features = sample.first().map(Vec::len).unwrap_or(0);
            return Err(AkidaError::TemporalInputNotSupported {
                batch,
                timesteps,
                features,
            }
            .into());
        }

        // Step 3: delegate to hardware or stub.
        self.run_batch_after_validation(inputs)
    }
}

// ── Hardware path ─────────────────────────────────────────────────────────────

#[cfg(feature = "hardware")]
impl AkidaBackend {
    fn run_batch_after_validation(
        &self,
        inputs: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<Vec<f32>>>, BackendError> {
        self.run_batch_hardware(inputs)
    }

    fn run_batch_hardware(
        &self,
        inputs: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<Vec<f32>>>, BackendError> {
        use std::io::Write as _;
        use std::process::{Command, Stdio};

        let binary = find_akida_runtime();
        let pcie_addr =
            std::env::var("THRINDEX_AKIDA_DEVICE").unwrap_or_else(|_| "0001:01:00.0".to_owned());

        // Encode inputs as {"batch": [[[f32,...]], ...]} — shape [N][T=1][features]
        let json_input = serde_json::json!({ "batch": inputs });
        let json_bytes = serde_json::to_vec(&json_input).map_err(|e| BackendError::Execution {
            detail: format!("E0204: failed to serialise batch input: {e}"),
        })?;

        let mut child = Command::new(&binary)
            .arg(&self.fbz_path)
            .arg("--pcie-addr")
            .arg(&pcie_addr)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| BackendError::Execution {
                detail: format!(
                    "E0204: failed to spawn akida-runtime at {binary}: {e}\n\
                     Hint: set THRINDEX_AKIDA_RUNTIME to the binary path.",
                    binary = binary.display()
                ),
            })?;

        // Write JSON batch to stdin; ignore broken-pipe (child may exit early)
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(&json_bytes).ok();
        }

        let output = child
            .wait_with_output()
            .map_err(|e| BackendError::Execution {
                detail: format!("E0204: akida-runtime wait failed: {e}"),
            })?;

        // On non-zero exit, show the first 512 bytes of the stdout JSON error
        if !output.status.success() {
            let out = String::from_utf8_lossy(&output.stdout);
            return Err(BackendError::Execution {
                detail: format!(
                    "E0204: akida-runtime exited {status}: {out}",
                    status = output.status,
                    out = &out[..out.len().min(512)]
                ),
            });
        }

        let result: serde_json::Value =
            serde_json::from_slice(&output.stdout).map_err(|e| BackendError::Execution {
                detail: format!("E0204: invalid JSON from akida-runtime: {e}"),
            })?;

        // Propagate structured errors returned by the binary
        if let Some(err_val) = result.get("error") {
            return Err(BackendError::Execution {
                detail: err_val
                    .as_str()
                    .unwrap_or("unknown error from akida-runtime")
                    .to_owned(),
            });
        }

        let outputs_arr = result["outputs"]
            .as_array()
            .ok_or_else(|| BackendError::Execution {
                detail: "E0204: 'outputs' missing or not an array in akida-runtime response"
                    .to_owned(),
            })?;

        // Shape: [N][out_features] → [N][T=1][out_features]
        let shaped: Vec<Vec<Vec<f32>>> = outputs_arr
            .iter()
            .map(|sample_val| {
                let flat: Vec<f32> = sample_val
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                vec![flat]
            })
            .collect();

        Ok(shaped)
    }
}

#[cfg(feature = "hardware")]
fn find_akida_runtime() -> PathBuf {
    // 1. Explicit override
    if let Ok(path) = std::env::var("THRINDEX_AKIDA_RUNTIME") {
        return PathBuf::from(path);
    }
    // 2. Sibling of current executable (works in release builds + cargo test)
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("akida-runtime");
        if sibling.exists() {
            return sibling;
        }
    }
    // 3. Rely on $PATH
    PathBuf::from("akida-runtime")
}

// ── Stub path (no hardware feature) ──────────────────────────────────────────

#[cfg(not(feature = "hardware"))]
impl AkidaBackend {
    fn run_batch_after_validation(
        &self,
        _inputs: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<Vec<f32>>>, BackendError> {
        Err(BackendError::Execution {
            detail: format!(
                "E0204: AkidaBackend hardware inference is not compiled in \
                 (fbz_path={path}). \
                 Build with --features hardware and set THRINDEX_AKIDA_ENGINE_PATH \
                 to enable Engine Library FFI (RFC-004 / ADR-0011 items 8–9).",
                path = self.fbz_path.display()
            ),
        })
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> AkidaBackend {
        AkidaBackend::new(Path::new("/dev/null"))
    }

    const VALID_ARTIFACT: &str = r#"{
        "format_version": "m2-draft",
        "target": "akida-akd1500",
        "model": {"layers": []},
        "metadata": {}
    }"#;

    const LIF_ARTIFACT: &str = r#"{
        "format_version": "m2-draft",
        "target": "akida-akd1500",
        "model": {
            "layers": [
                {"type": "lif", "threshold": 1.0, "alpha": 0.9, "reset": "subtract"}
            ]
        },
        "metadata": {}
    }"#;

    #[test]
    fn capability_name() {
        assert_eq!(backend().capability().name, "akida-akd1500");
    }

    #[test]
    fn e0407_temporal_input_rejected() {
        let b = backend();
        // 1 sample × 5 timesteps × 4 features — T > 1 must fire E0407.
        let inputs = vec![vec![vec![0.0f32; 4]; 5]];
        let err = b.run_batch(VALID_ARTIFACT, &inputs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("E0407"), "expected E0407 in:\n{msg}");
        assert!(msg.contains("T=5"), "expected timestep count in:\n{msg}");
    }

    #[test]
    fn e0401_lif_rejected_before_hardware() {
        let b = backend();
        let inputs = vec![vec![vec![0.0f32; 4]; 1]];
        let err = b.run_batch(LIF_ARTIFACT, &inputs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("E0401"), "expected E0401 in:\n{msg}");
    }

    // This test only makes sense without hardware — with hardware, run_batch
    // would attempt to spawn the subprocess instead of returning E0204.
    #[cfg(not(feature = "hardware"))]
    #[test]
    fn valid_artifact_t1_passes_preflight_fails_hardware_stub() {
        let b = backend();
        let inputs = vec![vec![vec![0.0f32; 4]; 1]];
        let err = b.run_batch(VALID_ARTIFACT, &inputs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("E0204"), "expected E0204 stub in:\n{msg}");
        assert!(!msg.contains("E0401"), "should not contain E0401:\n{msg}");
        assert!(!msg.contains("E0407"), "should not contain E0407:\n{msg}");
    }
}
