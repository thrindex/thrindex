//! Typed layer representations for programmatic access.
//!
//! These types are used for **reading** layer data only — they are never used
//! to re-serialise a `.thx` artifact.  That role belongs to the raw
//! `Vec<serde_json::Value>` in `WireModel` (see `wire.rs`).
//!
//! ## Float types
//!
//! All float fields are `f64`.  The compiler computes them as `f32` but stores
//! them in JSON via `serde_json::json!({})` which promotes `f32 → f64` through
//! the `Into<Value>` path.  Deserialising to `f64` therefore gives the exact
//! value the compiler wrote.
//!
//! ## Deserialisation
//!
//! Call [`parse_layer`] to convert a raw `serde_json::Value` to a `Layer`.

use serde::Deserialize;

use crate::error::ArtifactError;

/// A single typed layer.
#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Lif(LifLayer),
    Conv2d(Conv2dLayer),
    Flatten(FlattenLayer),
}

/// Fully-connected layer.
#[derive(Debug, Clone, Deserialize)]
pub struct DenseLayer {
    pub in_features: usize,
    pub out_features: usize,
    /// Base64 of little-endian f32 weights, row-major `[out_features × in_features]`.
    pub weights_b64: String,
    /// Base64 of little-endian f32 biases `[out_features]`, or `None`.
    pub bias_b64: Option<String>,
    /// Present only for layers with explicit axonal delays (ADR-0009).
    pub delays_b64: Option<String>,
    /// `"dense"` or `"sparse"` — present iff `delays_b64` is present.
    pub delays_encoding: Option<String>,
}

/// Leaky Integrate-and-Fire neuron layer.
///
/// ## Why f64, not f32
///
/// `alpha = exp(-dt/tau_mem)` is computed in f64 and truncated to `f32` before
/// being passed to `serde_json::json!({})`.  However `json!{}` converts
/// `f32 → f64` via `Into<Value>`, so the serialised value is the f64
/// representation of that bit-pattern.  Using `f64` here gives the exact value.
#[derive(Debug, Clone, Deserialize)]
pub struct LifLayer {
    pub threshold: f64,
    pub alpha: f64,
    /// `None` when synaptic dynamics are disabled.
    pub alpha_syn: Option<f64>,
    /// `"subtract"` or `"zero"`.
    pub reset: String,
}

/// 2D convolutional layer.
#[derive(Debug, Clone, Deserialize)]
pub struct Conv2dLayer {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    /// Base64 of little-endian f32 weights `[out_channels, in_channels, kH, kW]`.
    pub weights_b64: String,
    /// Base64 of little-endian f32 biases `[out_channels]`, or `None`.
    pub bias_b64: Option<String>,
}

/// Spatial-to-feature flatten (no learned parameters).
#[derive(Debug, Clone, Deserialize)]
pub struct FlattenLayer {
    /// First dimension to flatten (torch convention; default 1).
    pub start_dim: i32,
    /// Last dimension to flatten inclusive (torch convention; default -1).
    pub end_dim: i32,
}

// ── Internal wire helper (only for deserialisation) ───────────────────────────

/// Internal discriminated union used to route deserialization.
///
/// Identical to `Layer` but with the `#[serde(tag)]` attribute so that
/// serde_json can dispatch on the `"type"` field in the raw Value.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum LayerWire {
    Dense(DenseLayer),
    Lif(LifLayer),
    Conv2d(Conv2dLayer),
    Flatten(FlattenLayer),
}

/// Convert a raw JSON `Value` (as stored in `WireModel.layers`) to a typed [`Layer`].
///
/// # Errors
///
/// Returns [`ArtifactError::InvalidContent`] if the value is not a valid layer object.
pub fn parse_layer(v: &serde_json::Value) -> Result<Layer, ArtifactError> {
    let wire: LayerWire =
        serde_json::from_value(v.clone()).map_err(|e| ArtifactError::InvalidContent {
            detail: format!("layer parse error: {e}"),
        })?;
    Ok(match wire {
        LayerWire::Dense(d) => Layer::Dense(d),
        LayerWire::Lif(l) => Layer::Lif(l),
        LayerWire::Conv2d(c) => Layer::Conv2d(c),
        LayerWire::Flatten(f) => Layer::Flatten(f),
    })
}
