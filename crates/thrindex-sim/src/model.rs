//! `.thx` artifact model definition and deserialisation (ADR-0006).
//!
//! The format is JSON with base64-encoded little-endian f32 weight arrays.
//! `format_version` is checked first; unknown version → `E0002`.
//! Derived constants (`alpha`, `alpha_syn`) are **read from the artifact** —
//! they are resolved at compile time and the simulator never calls `exp` (ADR-0007 §4).

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::Deserialize;

use crate::error::SimError;

/// The supported format version string (ADR-0006).
pub const SUPPORTED_FORMAT_VERSION: &str = "m2-draft";

/// The minimum thrindex version required to read this format.
pub const MIN_THRINDEX_VERSION: &str = "0.2.0";

// ── Top-level artifact ────────────────────────────────────────────────────────

/// Deserialised `.thx` artifact (ADR-0006).
#[derive(Debug, Deserialize)]
pub struct ThxArtifact {
    pub format_version: String,
    pub thrindex_version: String,
    pub target: String,
    pub model: ThxModel,
    pub metadata: ThxMetadata,
}

#[derive(Debug, Deserialize)]
pub struct ThxModel {
    /// Layers stored as raw JSON values to avoid the `serde_json`
    /// `arbitrary_precision` + internally-tagged enum incompatibility.
    ///
    /// When the `core` binary links both `thrindex-artifact` (which requests
    /// `serde_json/arbitrary_precision`) and `thrindex-sim`, Cargo feature-unifies
    /// `serde_json` so that `arbitrary_precision` is active for the whole binary.
    /// With that feature enabled, `#[serde(tag = "type")]` internally-tagged
    /// deserialization routes numeric fields through serde's `Content` buffer
    /// instead of through `serde_json`'s own number path — producing
    /// `{"$serde_json::private::Number": "…"}` objects where `f32` scalars are
    /// expected, causing `E0008`.
    ///
    /// Storing layers as `Value` (as `thrindex-artifact`'s `WireModel` already
    /// does) and dispatching manually on the `"type"` string field avoids the
    /// code path entirely.
    pub layers: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ThxMetadata {
    pub compiled_at: String,
    /// The exact UTF-8 bytes that were hashed for the CRC32.
    /// Stored in the artifact so that the Rust loader never needs to
    /// re-serialise JSON (which would risk key-order drift vs. Python).
    pub model_canonical: String,
    pub crc32: String,
}

// ── Raw layer representations (as stored in JSON) ────────────────────────────
//
// These structs are intentionally NOT combined into a `#[serde(tag = "type")]`
// enum — see the `ThxModel::layers` comment above for the reason.

#[derive(Debug, Deserialize)]
pub(crate) struct DenseRaw {
    pub in_features: usize,
    pub out_features: usize,
    pub weights_b64: String,
    pub bias_b64: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LifRaw {
    pub threshold: f32,
    /// `exp(-dt/tau_mem)` — resolved at compile time (ADR-0007, correction 4).
    pub alpha: f32,
    /// `exp(-dt/tau_syn)` — `None` when synaptic dynamics are disabled.
    pub alpha_syn: Option<f32>,
    pub reset: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Conv2dRaw {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub weights_b64: String,
    pub bias_b64: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FlattenRaw {
    pub start_dim: i32,
    pub end_dim: i32,
}

// ── Resolved layer representations (weights decoded, ready for sim) ───────────

/// A layer with weights fully decoded from base64 into `Vec<f32>`.
#[derive(Debug, Clone)]
pub enum ResolvedLayer {
    Dense(DenseLayer),
    Lif(LifLayer),
    Conv2d(Conv2dLayer),
    Flatten(FlattenLayer),
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub in_features: usize,
    pub out_features: usize,
    /// Row-major `[out_features × in_features]`.
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct LifLayer {
    pub threshold: f32,
    /// Leak coefficient — read directly from artifact; never recomputed.
    pub alpha: f32,
    /// Optional synaptic leak coefficient.
    pub alpha_syn: Option<f32>,
    pub reset: ResetMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResetMode {
    Subtract,
    Zero,
}

#[derive(Debug, Clone)]
pub struct Conv2dLayer {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    /// `[out_channels, in_channels, kernel_h, kernel_w]` row-major.
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

/// Flatten layer: reshape spatial output to a 1-D feature vector.
///
/// The simulator currently treats this as a no-op because simulation operates on
/// 1-D feature vectors throughout.  Full spatial shape propagation is deferred (M3).
#[derive(Debug, Clone)]
pub struct FlattenLayer {
    pub start_dim: i32,
    pub end_dim: i32,
}

/// A fully-resolved, simulation-ready model.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub layers: Vec<ResolvedLayer>,
    pub target: String,
}

// ── Decoding helpers ──────────────────────────────────────────────────────────

/// Decode a base64 little-endian f32 array.
fn decode_weights(
    b64: &str,
    expected: usize,
    layer_idx: usize,
    field: &str,
) -> Result<Vec<f32>, SimError> {
    let bytes = BASE64
        .decode(b64)
        .map_err(|e| SimError::Base64DecodeError {
            layer_idx,
            field: field.to_string(),
            detail: e.to_string(),
        })?;

    if bytes.len() != expected * 4 {
        return Err(SimError::WeightShapeMismatch {
            layer_idx,
            declared: expected,
            decoded: bytes.len(),
            got_f32: bytes.len() / 4,
        });
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunks_exact guarantees 4 bytes")))
        .collect())
}

fn decode_weights_opt(
    b64: Option<&String>,
    expected: usize,
    layer_idx: usize,
    field: &str,
) -> Result<Option<Vec<f32>>, SimError> {
    match b64 {
        Some(s) => Ok(Some(decode_weights(s, expected, layer_idx, field)?)),
        None => Ok(None),
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and resolve a `.thx` artifact from disk.
///
/// Checks `format_version` before any further parsing (ADR-0006).
/// Verifies CRC32 integrity of the `model` JSON block.
///
/// # Errors
///
/// Returns [`SimError`] for any of: `E0001` (not found), `E0002` (wrong version),
/// `E0003` (unknown layer type), `E0004` (dimension mismatch), `E0005` (invalid LIF
/// params), `E0006/E0007` (weight decode errors), `E0008` (JSON parse), `E0009` (CRC32).
pub fn load(path: &str) -> Result<ResolvedModel, SimError> {
    let content = std::fs::read_to_string(path).map_err(|_| SimError::ArtifactNotFound {
        path: path.to_string(),
    })?;
    load_from_str(&content)
}

/// Parse and resolve a `.thx` artifact from a JSON string in memory.
///
/// Identical to [`load`] except it accepts a pre-read string rather than a file path.
/// Used for backward-compatibility tests (frozen M2 fixture strings) and by
/// `thrindex-compiler`'s integration tests.
///
/// # Errors
///
/// Returns [`SimError`] for `E0002` (wrong version), `E0003`–`E0009` (see [`load`]).
/// Does **not** produce `E0001` (`ArtifactNotFound`) — there is no path.
pub fn load_from_str(json: &str) -> Result<ResolvedModel, SimError> {
    // Parse the raw JSON.
    let artifact: ThxArtifact =
        serde_json::from_str(json).map_err(|e| SimError::JsonParseError {
            detail: e.to_string(),
        })?;

    // Version gate — first check (ADR-0006).
    if artifact.format_version != SUPPORTED_FORMAT_VERSION {
        return Err(SimError::UnsupportedFormatVersion {
            version: artifact.format_version,
            supported: SUPPORTED_FORMAT_VERSION.to_string(),
            min_thrindex: MIN_THRINDEX_VERSION.to_string(),
        });
    }

    // CRC32 integrity check.
    verify_crc32(&artifact)?;

    // Resolve all layers.
    resolve_model(&artifact)
}

/// Verify that `metadata.model_canonical` hashes to `metadata.crc32`.
///
/// `model_canonical` is the exact UTF-8 string Python hashed — stored in the artifact
/// to avoid JSON re-serialisation key-order drift between Python and Rust.
fn verify_crc32(artifact: &ThxArtifact) -> Result<(), SimError> {
    let computed = crc32fast::hash(artifact.metadata.model_canonical.as_bytes());
    let computed_hex = format!("{computed:08x}");
    if computed_hex != artifact.metadata.crc32.to_lowercase() {
        return Err(SimError::IntegrityCheckFailed);
    }
    Ok(())
}

fn resolve_model(artifact: &ThxArtifact) -> Result<ResolvedModel, SimError> {
    let mut layers = Vec::with_capacity(artifact.model.layers.len());
    let mut prev_out: Option<usize> = None;

    for (idx, raw) in artifact.model.layers.iter().enumerate() {
        let resolved = resolve_layer(raw, idx)?;

        // Dimension continuity check (skip LIF — it does not change width).
        let (in_dim, out_dim) = layer_dims(&resolved);
        if let (Some(prev), Some(expected_in)) = (prev_out, in_dim)
            && prev != expected_in
        {
            return Err(SimError::DimensionMismatch {
                layer_idx: idx,
                prev_idx: idx - 1,
                expected: expected_in,
                got: prev,
            });
        }
        if let Some(o) = out_dim {
            prev_out = Some(o);
        }

        layers.push(resolved);
    }

    Ok(ResolvedModel {
        layers,
        target: artifact.target.clone(),
    })
}

/// Dispatch a raw JSON layer value to the appropriate `ResolvedLayer`.
///
/// Dispatches on the `"type"` string field and deserialises the value directly
/// into the concrete typed struct (e.g. `DenseRaw`).  This avoids going through
/// `#[serde(tag = "type")]` internally-tagged enum deserialization, which is
/// incompatible with `serde_json`'s `arbitrary_precision` feature (activated by
/// Cargo feature unification when `thrindex-artifact` is in the same binary).
fn resolve_layer(v: &serde_json::Value, idx: usize) -> Result<ResolvedLayer, SimError> {
    let layer_type = v["type"].as_str().ok_or_else(|| SimError::JsonParseError {
        detail: format!("layer[{idx}] missing \"type\" field"),
    })?;

    match layer_type {
        "dense" => {
            let d: DenseRaw =
                serde_json::from_value(v.clone()).map_err(|e| SimError::JsonParseError {
                    detail: format!("layer[{idx}] dense parse error: {e}"),
                })?;
            let weights = decode_weights(
                &d.weights_b64,
                d.out_features * d.in_features,
                idx,
                "weights",
            )?;
            let bias = decode_weights_opt(d.bias_b64.as_ref(), d.out_features, idx, "bias")?;
            Ok(ResolvedLayer::Dense(DenseLayer {
                in_features: d.in_features,
                out_features: d.out_features,
                weights,
                bias,
            }))
        }
        "lif" => {
            let l: LifRaw =
                serde_json::from_value(v.clone()).map_err(|e| SimError::JsonParseError {
                    detail: format!("layer[{idx}] lif parse error: {e}"),
                })?;
            let reset = match l.reset.as_str() {
                "subtract" => ResetMode::Subtract,
                "zero" => ResetMode::Zero,
                other => {
                    return Err(SimError::InvalidLifParam {
                        layer_idx: idx,
                        message: format!("unknown reset mode \"{other}\""),
                        reason: "only \"subtract\" and \"zero\" are supported".into(),
                        fix:
                            "recompile from a model that uses reset=\"subtract\" or reset=\"zero\""
                                .into(),
                    });
                }
            };
            // Guard: alpha must be in (0, 1] to be a valid leak coefficient.
            if !(l.alpha > 0.0 && l.alpha <= 1.0) {
                return Err(SimError::InvalidLifParam {
                    layer_idx: idx,
                    message: format!("alpha={} is outside (0, 1]", l.alpha),
                    reason: "alpha = exp(-dt/tau_mem); tau_mem must be > dt (1 ms)".into(),
                    fix: "recompile from a model with tau_mem > 1.0 ms".into(),
                });
            }
            Ok(ResolvedLayer::Lif(LifLayer {
                threshold: l.threshold,
                alpha: l.alpha,
                alpha_syn: l.alpha_syn,
                reset,
            }))
        }
        "conv2d" => {
            let c: Conv2dRaw =
                serde_json::from_value(v.clone()).map_err(|e| SimError::JsonParseError {
                    detail: format!("layer[{idx}] conv2d parse error: {e}"),
                })?;
            let n_weights = c.out_channels * c.in_channels * c.kernel_h * c.kernel_w;
            let weights = decode_weights(&c.weights_b64, n_weights, idx, "weights")?;
            let bias = decode_weights_opt(c.bias_b64.as_ref(), c.out_channels, idx, "bias")?;
            Ok(ResolvedLayer::Conv2d(Conv2dLayer {
                in_channels: c.in_channels,
                out_channels: c.out_channels,
                kernel_h: c.kernel_h,
                kernel_w: c.kernel_w,
                stride: c.stride,
                padding: c.padding,
                weights,
                bias,
            }))
        }
        "flatten" => {
            let f: FlattenRaw =
                serde_json::from_value(v.clone()).map_err(|e| SimError::JsonParseError {
                    detail: format!("layer[{idx}] flatten parse error: {e}"),
                })?;
            Ok(ResolvedLayer::Flatten(FlattenLayer {
                start_dim: f.start_dim,
                end_dim: f.end_dim,
            }))
        }
        other => Err(SimError::JsonParseError {
            detail: format!(
                "layer[{idx}] unknown type \"{other}\"; expected \"dense\", \"lif\", \"conv2d\", or \"flatten\""
            ),
        }),
    }
}

/// Returns `(input_features, output_features)` for dimension-continuity checking.
/// LIF layers return `(None, None)` — they neither change nor consume the feature count.
/// Flatten and Conv2d return `(None, None)` — spatial shape tracking is deferred (M3).
fn layer_dims(layer: &ResolvedLayer) -> (Option<usize>, Option<usize>) {
    match layer {
        ResolvedLayer::Dense(d) => (Some(d.in_features), Some(d.out_features)),
        ResolvedLayer::Conv2d(c) => (Some(c.in_channels), Some(c.out_channels)),
        ResolvedLayer::Lif(_) | ResolvedLayer::Flatten(_) => (None, None),
    }
}
