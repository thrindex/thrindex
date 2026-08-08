//! Errors emitted by the AKD1500 backend (E0401–E0407).
//!
//! All errors follow the four-part format required by §30:
//! error code, Observed (interpolated values), Why, and What to do.
use thiserror::Error;
use thrindex_backend_api::BackendError;

/// AKD1500-specific validation errors.
///
/// Converts to [`BackendError::Execution`] via [`From`] so callers
/// holding a `BackendError` see the full four-part message.
#[derive(Debug, Error)]
pub enum AkidaError {
    /// E0401 — LIF layer present; AKD1500 implements bounded ReLU, not LIF.
    #[error(
        "E0401: layer[{index}] type=\"lif\" cannot be mapped to akida-akd1500.\n\
         Observed: .thx layer {index} has type \"lif\" with threshold={threshold}, \
         alpha={alpha}, reset=\"{reset}\".\n\
         Why: AKD1500 implements Akida 1.0. Akida 1.0's activation is bounded ReLU applied \
         per-inference-call to the integer dot product. It has no membrane potential, no \
         exponential leak (alpha term), and no spike-triggered reset. The leaky \
         integrate-and-fire dynamics encoded in this layer do not exist in the hardware.\n\
         What to do: (a) Use the 'sim' backend for SNN simulation. \
         (b) If you want AKD1500 inference, re-design the model without LIF layers \
         using the CNN workflow described in the BrainChip documentation. \
         There is no lossless conversion from LIF to bounded-ReLU; they are different \
         computational models.\n\
         Docs: https://docs.thrindex.com/errors/E0401"
    )]
    LifNotSupported {
        index: usize,
        threshold: f64,
        alpha: f64,
        reset: String,
    },

    /// E0402 — Synaptic delays present; AKD1500 (Akida 1.0) has no TNP.
    #[error(
        "E0402: layer[{index}] has synaptic delays but akida-akd1500 declares \
         native_delay_max_steps=0.\n\
         Observed: .thx layer {index} has \"{encoding}\" delay encoding; \
         delay_fallback=Reject per capability descriptor.\n\
         Why: AKD1500 is an Akida 1.0 device. Temporal Neural Processors (TNP) that \
         support buffered or recurrent delay processing are an Akida 2.0 feature not \
         present on AKD1500. There is no mechanism to emulate per-synapse delays in \
         a single-frame feedforward model.\n\
         What to do: Retrain without delays, or target a backend that supports them. \
         The 'sim' backend supports delays up to u16::MAX steps \
         (native_delay_max_steps=65535, delay_fallback=Emulate).\n\
         Docs: https://docs.thrindex.com/errors/E0402"
    )]
    DelaysNotSupported { index: usize, encoding: String },

    /// E0403 — Non-finite (NaN / Inf) values in weights; cannot quantize.
    #[error(
        "E0403: layer[{index}] weights_b64 contains non-finite values.\n\
         Observed: {count} NaN or Inf values found in decoded weight tensor.\n\
         Why: AKD1500 requires integer-quantizable weights. Quantization of non-finite \
         floats is undefined and would produce silent garbage output.\n\
         What to do: Check the training pipeline for numerical instability. \
         Re-export the model with finite weights.\n\
         Docs: https://docs.thrindex.com/errors/E0403"
    )]
    NonFiniteWeights { index: usize, count: usize },

    /// E0404 — Artifact compiled for a different target backend.
    #[error(
        "E0404: .thx artifact was compiled for target=\"{actual_target}\", \
         not \"akida-akd1500\".\n\
         Observed: metadata.target = \"{actual_target}\".\n\
         Why: A .thx artifact carries the target it was compiled for. Running it on a \
         different backend without recompilation may produce incorrect results because \
         resolved constants (alpha, dt_ms) are target-specific.\n\
         What to do: Recompile with --target akida-akd1500:\n\
         \x20\x20  thrindex build --target akida-akd1500 model.py\n\
         Docs: https://docs.thrindex.com/errors/E0404"
    )]
    WrongTarget { actual_target: String },

    /// E0407 — Temporal input (T > 1) passed to a single-frame stateless backend.
    ///
    /// This is the most dangerous failure mode: without this guard the backend
    /// would produce T independent feedforward responses silently, which look
    /// like valid output but have no relationship to T-step SNN temporal dynamics.
    #[error(
        "E0407: run_batch received inputs with T={timesteps} timesteps per sample; \
         akida-akd1500 requires T=1.\n\
         Observed: inputs shape [batch={batch}, T={timesteps}, features={features}]; T > 1.\n\
         Why: AKD1500 processes one spatial frame per model.forward() call. It holds no \
         temporal state between calls: there is no membrane potential that accumulates \
         across timesteps, no recurrent architecture, and no frame-to-frame buffering. \
         Iterating T frames independently through the same stateless model produces T \
         independent feedforward responses. This is NOT equivalent to T timesteps of SNN \
         temporal dynamics: any model whose behaviour depends on cross-timestep membrane \
         accumulation will produce silently incorrect output rather than a recognisable \
         error. E0407 is raised explicitly to prevent that silent failure.\n\
         What to do: Pass exactly one timestep frame per sample: inputs shape \
         [batch, 1, features]. AKD1500 performs spatial (not temporal) inference; \
         the 'timesteps' dimension must always be 1. For temporal SNN inference, \
         use the 'sim' backend.\n\
         Docs: https://docs.thrindex.com/errors/E0407"
    )]
    TemporalInputNotSupported {
        batch: usize,
        timesteps: usize,
        features: usize,
    },
}

impl From<AkidaError> for BackendError {
    fn from(e: AkidaError) -> Self {
        BackendError::Execution {
            detail: e.to_string(),
        }
    }
}
