//! Capability descriptor — what a backend declares it can do (§16, ADR-0008, ADR-0009).
//!
//! ## Why this exists
//!
//! Core crates (`thrindex-ir`, `thrindex-compiler`, `thrindex-sim`) must never
//! hardcode hardware-specific constants such as native timestep, delay limits, or
//! weight precision. The [`Capability`] struct is how each backend declares its own
//! constraints. The compiler reads it to decide:
//!   - whether to retime (ADR-0008 `dt` negotiation),
//!   - whether to lower delays natively or emulate (ADR-0009 policy P3),
//!   - which quantization path to take (future RFC-004).
//!
//! UNVERIFIED hardware figures (e.g. Loihi 2's 62/63 step cap, SpiNNaker's 16-slot ring)
//! must be declared in each backend's `Capability::default()`, not here. No number in
//! this file should ever be chip-specific.
use serde::{Deserialize, Serialize};

/// What a backend can do. Declared by each backend; read by the compiler and harness.
///
/// All timing constants are owned here — never in core crates (ADR-0008 §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    /// Short identifier, e.g. `"sim"`, `"loihi2"`, `"spinnaker"`.
    pub name: String,

    /// Native algorithmic timestep in milliseconds (ADR-0008).
    /// The compiler retimes if this differs from the artifact's authored `dt_ms`.
    pub native_dt_ms: f64,

    /// Maximum synaptic delay in timesteps this backend can route natively (ADR-0009).
    /// `0` means no native delay support — delays must be emulated or rejected.
    pub native_delay_max_steps: u16,

    /// What to do when a delay exceeds `native_delay_max_steps` (ADR-0009 P3).
    pub delay_fallback: DelayFallback,

    /// Weight precision the backend executes at.
    pub precision: Precision,
}

/// Compiler policy when a delay exceeds the backend's declared cap (ADR-0009).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DelayFallback {
    /// Emulate the over-cap delay using a spike-history ring buffer. The compile
    /// report prints the emulation cost (memory and ops per step).
    Emulate,
    /// Reject the artifact with `E0103` (delay exceeds target maximum).
    Reject,
}

/// Weight precision the backend executes at.
///
/// Does not prescribe the *format* (that is RFC-004 / ADR future). States what
/// the backend does, so the conformance harness can set expectations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Precision {
    /// IEEE 754 float32 — the reference simulator's precision (ADR-0007).
    Float32,
    /// Per-tensor symmetric int8.
    Int8PerTensor,
    /// Per-channel (per-row) symmetric int8. The realistic hardware model for
    /// well-calibrated backends; preferred for conformance measurements.
    Int8PerChannel,
    /// Per-tensor symmetric int4.
    Int4PerTensor,
    /// Vendor-specific; description is informational.
    Custom(String),
}
