//! Graph IR type definitions (ADR-0008, ADR-0009).
//!
//! The two-level parameter split is enforced **by type**:
//! - [`GraphLif`] holds `tau_mem` / `tau_syn` (continuous, ms) — no `alpha`.
//! - [`GraphDense`] holds optional [`Delays`] (integer step counts relative to
//!   `GraphModel::dt_ms`) — no target-specific resolved `delay_steps`.
//! - [`GraphConv2d`] has no `delays` field; Conv2d-delay support is deferred.

use serde::{Deserialize, Serialize};

// ── Top-level model ──────────────────────────────────────────────────────────

/// The pre-lowering Graph IR representation of an SNN model.
///
/// Stores only **continuous** parameters plus the canonical timestep `dt_ms`.
/// Resolved constants (`alpha`, `delay_steps`) do not appear here — they are
/// derived by `thrindex-compiler::lower` for a specific target's native `dt`.
///
/// # Example
///
/// ```json
/// {
///   "dt_ms": 1.0,
///   "layers": [
///     { "type": "dense", "in_features": 4, "out_features": 4,
///       "weights_b64": "...", "bias_b64": null, "delays": null },
///     { "type": "lif", "tau_mem": 10.0, "tau_syn": null,
///       "threshold": 1.0, "reset": "subtract" }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphModel {
    /// Canonical authored timestep in milliseconds (ADR-0008).
    ///
    /// Lowering resolves `alpha = exp(-dt_eff / tau_mem)` at the target's native `dt`,
    /// which may differ from this value (retiming advisory emitted when it does).
    pub dt_ms: f64,

    /// Ordered layers. The sequence defines the forward-pass graph.
    pub layers: Vec<GraphLayer>,
}

// ── Layer enum ───────────────────────────────────────────────────────────────

/// A single layer in the Graph IR.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GraphLayer {
    /// Fully-connected (Dense) layer, optionally with per-connection delays.
    Dense(GraphDense),

    /// Leaky Integrate-and-Fire neuron layer (continuous parameters — no `alpha`).
    Lif(GraphLif),

    /// 2-D convolutional layer.
    ///
    /// Conv2d-delay support is **deferred** (ADR-0009 v1 scope).
    /// Reopening trigger: a design partner ships a Conv2d-delay model where the
    /// absence of delay support is a proven constraint.
    Conv2d(GraphConv2d),
}

// ── Layer types ───────────────────────────────────────────────────────────────

/// Dense (fully-connected) layer in the Graph IR.
///
/// Weight encoding: base64 little-endian `f32`, row-major `[out_features × in_features]`.
/// Delays are per-connection and optional (absent ⇒ all delays are 0; no bytes in the
/// target artifact). See [`Delays`] for encoding details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDense {
    /// Number of input features.
    pub in_features: usize,

    /// Number of output features.
    pub out_features: usize,

    /// Weights as base64 little-endian `f32` (row-major `out × in`).
    pub weights_b64: String,

    /// Optional bias as base64 little-endian `f32` (length `out_features`).
    pub bias_b64: Option<String>,

    /// Per-connection delays (ADR-0009).
    ///
    /// `None` ⇒ no delays on this layer; no `delays_b64` field in the target artifact.
    /// This is the common case (M2 models, any zero-delay model) and costs zero bytes.
    pub delays: Option<Delays>,
}

/// LIF neuron layer in the Graph IR.
///
/// # Two-level rule (ADR-0008, binding)
///
/// This struct has **no `alpha` field** by design. It is structurally impossible to
/// store a pre-resolved `alpha` in the Graph IR. The continuous `tau_mem` is the
/// source of truth; `alpha = exp(-dt_eff / tau_mem)` is computed by the lower pass
/// at the target's effective `dt`.
///
/// # Reset mode
///
/// `reset` is deserialized as a `String` and validated to `ResetMode` by the
/// validate pass in `thrindex-compiler` (which produces E0107 when invalid).
/// Parsing it eagerly here would prevent §30-format errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphLif {
    /// Membrane time constant in milliseconds (continuous). Must be `> dt_ms`.
    ///
    /// `alpha` is **not** stored here — it is resolved at lower time.
    pub tau_mem: f64,

    /// Synaptic time constant in milliseconds (continuous). `None` = single-state LIF.
    ///
    /// `alpha_syn` is **not** stored here — it is resolved at lower time.
    pub tau_syn: Option<f64>,

    /// Firing threshold (dimensionless).
    pub threshold: f32,

    /// Reset mode string — validated to [`ResetMode`] by the validate pass (E0107).
    pub reset: String,
}

/// Conv2d layer in the Graph IR.
///
/// Weight encoding: base64 little-endian `f32`, `[out, in, kH, kW]` row-major.
///
/// **No `delays` field** — Conv2d-delay support is deferred (ADR-0009 v1 scope).
/// The absence is intentional and makes the state unrepresentable, not just unimplemented.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConv2d {
    /// Input channels.
    pub in_channels: usize,

    /// Output channels.
    pub out_channels: usize,

    /// Kernel height.
    pub kernel_h: usize,

    /// Kernel width.
    pub kernel_w: usize,

    /// Stride `[height, width]`.
    pub stride: [usize; 2],

    /// Padding `[height, width]`.
    pub padding: [usize; 2],

    /// Weights as base64 little-endian `f32`, `[out, in, kH, kW]`.
    pub weights_b64: String,

    /// Optional bias as base64 little-endian `f32` (length `out_channels`).
    pub bias_b64: Option<String>,
}

// ── Delays ────────────────────────────────────────────────────────────────────

/// Per-connection delay encoding for a Dense layer (ADR-0009).
///
/// Units: integer multiples of `GraphModel::dt_ms` (ADR-0008).
///
/// Two encodings are supported; the compiler picks the smaller one:
/// - **Dense**: one `u16` per connection (`in_features × out_features` values),
///   stored as base64 little-endian `u16`.
/// - **Sparse**: `(connection_index: u32, delay_steps: u16)` pairs, sorted by
///   index, stored as base64 little-endian `(u32, u16)` pairs.
///
/// ## Size honesty (ADR-0009 v1 scope)
///
/// Dense per-connection delays cost `sizeof(u16) × in_features × out_features` bytes
/// per layer. For a 700→512 layer this is ≈700 KB — roughly doubling that layer's
/// payload. Sparse encoding does **not** help when every synapse has a delay (the
/// SOTA case). This cost is accepted in v1; delay *compression* is deferred to a
/// future RFC (trigger: a design partner ships dense-delay models where artifact size
/// is a proven, measured constraint).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "encoding", rename_all = "snake_case")]
pub enum Delays {
    /// One `u16` per connection, base64 LE.
    Dense {
        /// Base64 little-endian `u16` array, length = `in_features × out_features`.
        delays_b64: String,
    },
    /// Sorted `(u32 connection_index, u16 delay_steps)` pairs, base64 LE.
    Sparse {
        /// Base64 little-endian `(u32, u16)` pairs.
        delays_b64: String,
    },
}

// ── Reset mode ────────────────────────────────────────────────────────────────

/// Membrane reset mode after a spike.
///
/// Parsed from [`GraphLif::reset`] by the validate pass (E0107 on failure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResetMode {
    /// Membrane potential decremented by threshold: `mem -= threshold`.
    Subtract,
    /// Membrane potential clamped to zero: `mem = 0`.
    Zero,
}

impl TryFrom<&str> for ResetMode {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "subtract" => Ok(ResetMode::Subtract),
            "zero" => Ok(ResetMode::Zero),
            other => Err(format!(
                "unknown reset mode {other:?}; expected \"subtract\" or \"zero\""
            )),
        }
    }
}
