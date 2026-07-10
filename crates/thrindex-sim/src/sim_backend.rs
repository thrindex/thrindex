//! Reference `SimBackend` — the software simulator implementing the [`Backend`] trait.
//!
//! This is the reference implementation that every hardware backend must match within
//! the `CONFORMANCE_ENVELOPE_vN` tolerances (ADR-0010). It executes in f32 (ADR-0007)
//! and is self-deterministic (same inputs → byte-identical outputs, any thread count).
//!
//! ## Usage in conformance
//!
//! The conformance harness runs both the backend-under-test and `SimBackend` on the
//! same frozen test set, then computes `per_neuron_rate_error` between their rasters.
use thrindex_backend_api::{Backend, BackendError, Capability, DelayFallback, Precision};

use crate::{SimConfig, model, sim};

/// The reference software simulator, exposed as a [`Backend`].
///
/// Capability declares:
/// - `native_dt_ms = 1.0` (ADR-0005 canonical step)
/// - `native_delay_max_steps = u16::MAX` (emulates any delay via ring buffer, ADR-0009)
/// - `delay_fallback = Emulate`
/// - `precision = Float32` (ADR-0007)
pub struct SimBackend {
    capability: Capability,
    /// Thread count for batch parallelism. `0` = Rayon default.
    threads: usize,
}

impl SimBackend {
    /// Create with explicit thread count. Use `0` for Rayon default.
    #[must_use]
    pub fn new(threads: usize) -> Self {
        Self {
            capability: Capability {
                name: "sim".to_string(),
                native_dt_ms: 1.0,
                native_delay_max_steps: u16::MAX,
                delay_fallback: DelayFallback::Emulate,
                precision: Precision::Float32,
            },
            threads,
        }
    }
}

impl Default for SimBackend {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Backend for SimBackend {
    fn capability(&self) -> &Capability {
        &self.capability
    }

    fn run_batch(
        &self,
        artifact_json: &str,
        inputs: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<Vec<f32>>>, BackendError> {
        let resolved =
            model::load_from_str(artifact_json).map_err(|e| BackendError::ArtifactParse {
                detail: e.to_string(),
            })?;

        let config = SimConfig {
            threads: self.threads,
        };
        let output = sim::run(&resolved, inputs, &config).map_err(|e| BackendError::Execution {
            detail: e.to_string(),
        })?;

        Ok(output.spikes)
    }
}
