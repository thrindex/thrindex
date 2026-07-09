//! The `Backend` trait — the plugin contract every execution target implements.
use crate::{BackendError, Capability};

/// An execution backend: takes a `.thx` artifact and a batch of input spike trains,
/// returns output spike rasters.
///
/// ## Contract
///
/// - **Inputs:** `inputs[b][t][n]` — batch `b`, timestep `t`, input neuron `n`. Values
///   are `{0.0, 1.0}` (binary spike); backends must not assume other values.
/// - **Outputs:** `outputs[b][t][n]` — same batch dimension, same timestep count `T`,
///   output neuron count from the model's final layer.
/// - **Determinism:** same artifact + same inputs → same outputs for all valid calls.
///   The conformance harness will call `run_batch` multiple times and assert identical
///   results (self-determinism per ADR-0007).
/// - **Thread safety:** `Backend` is `Send + Sync`. The harness may call backends from
///   multiple threads concurrently for different samples.
///
/// ## Conformance
///
/// The `conformance/` harness compares this backend's output against the reference
/// `SimBackend` (from `thrindex-sim`) using the `per_neuron_rate_error` metric
/// (ADR-0010 Part I). Pass/fail uses `CONFORMANCE_ENVELOPE_v0_DRAFT` until the
/// ratification measurement produces `CONFORMANCE_ENVELOPE_v0`.
pub trait Backend: Send + Sync {
    /// Declare what this backend can do. Called once before any run.
    fn capability(&self) -> &Capability;

    /// Run a batch of samples through the model.
    ///
    /// `artifact_json` is the raw `.thx` JSON string (ADR-0006).
    /// `inputs` has shape `[batch, timesteps, in_features]`.
    ///
    /// Returns `[batch, timesteps, out_features]` or a [`BackendError`].
    fn run_batch(
        &self,
        artifact_json: &str,
        inputs: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<Vec<f32>>>, BackendError>;
}
