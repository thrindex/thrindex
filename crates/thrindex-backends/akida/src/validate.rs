//! Pre-flight validation of `.thx` JSON artifacts for AKD1500 compatibility.
//!
//! Emits [`BackendError`] for the first unsupported construct found.
//! E0401/E0402/E0403/E0404 are detected here before any Python or C++ code runs.
//! E0407 (T > 1 temporal input) is detected in [`crate::backend`] at `run_batch` call time.
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use thrindex_backend_api::BackendError;

use crate::capability::AKD1500_TARGET_NAME;
use crate::error::AkidaError;

/// Validate a raw `.thx` JSON artifact string for AKD1500 compatibility.
///
/// Returns `Ok(())` if the artifact contains only AKD1500-compatible constructs:
/// - `target == "akida-akd1500"` (E0404 if not)
/// - No `Lif` layers (E0401 if present)
/// - No synaptic delays (E0402 if present)
/// - All weight tensors finite (E0403 if NaN/inf found)
///
/// Fails on the **first** violation; call once before invoking any Python or C++ code.
pub fn validate_artifact(artifact_json: &str) -> Result<(), BackendError> {
    let root: serde_json::Value =
        serde_json::from_str(artifact_json).map_err(|e| BackendError::ArtifactParse {
            detail: format!("akida: JSON parse failed: {e}"),
        })?;

    // E0404 — wrong compile target.
    let target = root
        .get("target")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if target != AKD1500_TARGET_NAME {
        return Err(AkidaError::WrongTarget {
            actual_target: target.to_owned(),
        }
        .into());
    }

    // Walk layer list.
    let empty = vec![];
    let layers = root
        .get("model")
        .and_then(|m| m.get("layers"))
        .and_then(serde_json::Value::as_array)
        .unwrap_or(&empty);

    for (index, layer) in layers.iter().enumerate() {
        let layer_type = layer
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");

        match layer_type {
            // E0401 — LIF layer: no membrane potential, no alpha, no reset on AKD1500.
            "lif" => {
                let threshold = layer
                    .get("threshold")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let alpha = layer
                    .get("alpha")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let reset = layer
                    .get("reset")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
                    .to_owned();
                return Err(AkidaError::LifNotSupported {
                    index,
                    threshold,
                    alpha,
                    reset,
                }
                .into());
            }

            // Dense / Conv2d: check for delays (E0402) and non-finite weights (E0403).
            "dense" | "conv2d" => {
                // E0402 — any delay encoding rejects the model.
                let has_delays =
                    layer.get("delays_b64").is_some() || layer.get("delays_sparse").is_some();
                if has_delays {
                    let encoding = layer
                        .get("delays_encoding")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("present")
                        .to_owned();
                    return Err(AkidaError::DelaysNotSupported { index, encoding }.into());
                }

                // E0403 — non-finite weights cannot be quantized.
                if let Some(w_b64) = layer.get("weights_b64").and_then(serde_json::Value::as_str) {
                    let count = count_nonfinite_f32(w_b64);
                    if count > 0 {
                        return Err(AkidaError::NonFiniteWeights { index, count }.into());
                    }
                }
            }

            // Unknown layer types pass here; the Python compile step will reject them.
            _ => {}
        }
    }

    Ok(())
}

/// Decode a base64 LE-f32 weight blob and count NaN/inf values.
/// Returns 0 on decode error (malformed base64 is an artifact issue, not a weight issue).
fn count_nonfinite_f32(b64: &str) -> usize {
    let Ok(bytes) = BASE64.decode(b64) else {
        return 0;
    };
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .filter(|f| !f.is_finite())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lif_artifact() -> &'static str {
        r#"{
            "format_version": "m2-draft",
            "target": "akida-akd1500",
            "model": {
                "layers": [
                    {"type": "lif", "threshold": 1.0, "alpha": 0.9, "reset": "subtract"}
                ]
            },
            "metadata": {}
        }"#
    }

    fn dense_with_delays_artifact() -> &'static str {
        r#"{
            "format_version": "m2-draft",
            "target": "akida-akd1500",
            "model": {
                "layers": [
                    {
                        "type": "dense",
                        "in_features": 4, "out_features": 2,
                        "weights_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                        "delays_b64": "AAAA",
                        "delays_encoding": "dense"
                    }
                ]
            },
            "metadata": {}
        }"#
    }

    fn wrong_target_artifact() -> &'static str {
        r#"{
            "format_version": "m2-draft",
            "target": "sim",
            "model": {"layers": []},
            "metadata": {}
        }"#
    }

    fn valid_dense_only_artifact() -> &'static str {
        // weights_b64: 8 LE f32 zeros (4 bytes × 8 = 32 bytes → base64)
        r#"{
            "format_version": "m2-draft",
            "target": "akida-akd1500",
            "model": {
                "layers": [
                    {
                        "type": "dense",
                        "in_features": 2, "out_features": 4,
                        "weights_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
                    }
                ]
            },
            "metadata": {}
        }"#
    }

    #[test]
    fn lif_rejection() {
        let err = validate_artifact(lif_artifact()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("E0401"), "expected E0401 in:\n{msg}");
        assert!(msg.contains("lif"), "expected 'lif' mention in:\n{msg}");
        assert!(msg.contains("threshold=1"), "expected threshold in:\n{msg}");
    }

    #[test]
    fn delay_rejection() {
        let err = validate_artifact(dense_with_delays_artifact()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("E0402"), "expected E0402 in:\n{msg}");
        assert!(msg.contains("dense"), "expected encoding in:\n{msg}");
    }

    #[test]
    fn wrong_target() {
        let err = validate_artifact(wrong_target_artifact()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("E0404"), "expected E0404 in:\n{msg}");
        assert!(msg.contains("sim"), "expected actual target in:\n{msg}");
    }

    #[test]
    fn valid_dense_only() {
        validate_artifact(valid_dense_only_artifact())
            .expect("Dense-only artifact with correct target should pass validation");
    }
}
