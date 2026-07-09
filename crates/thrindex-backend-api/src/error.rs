//! Errors returned by backend execution.
use thiserror::Error;

/// Error produced by a [`super::Backend`] run.
#[derive(Debug, Error)]
pub enum BackendError {
    /// The artifact JSON could not be parsed or resolved.
    #[error("E0201: backend could not parse artifact\nWhy: {detail}\nFix: verify the artifact was produced by `thrindex compile` and is not corrupted\nDocs: https://docs.thrindex.com/errors/E0201")]
    ArtifactParse { detail: String },

    /// Input shape does not match the model's declared first-layer dimensions.
    #[error("E0202: input shape mismatch — expected {expected} features, got {got}\nWhy: the input tensor was prepared for a different model\nFix: re-encode inputs using `thrindex.encoders` with the model's declared input size\nDocs: https://docs.thrindex.com/errors/E0202")]
    InputShapeMismatch { expected: usize, got: usize },

    /// Backend produced an output raster with a different shape than the reference.
    #[error("E0203: backend output shape mismatch — expected [{t}, {n}], got [{got_t}, {got_n}]\nWhy: the backend returned fewer/more timesteps or neurons than the reference simulation\nFix: verify the backend correctly unrolls T timesteps and outputs all N_out neurons\nDocs: https://docs.thrindex.com/errors/E0203")]
    OutputShapeMismatch {
        t: usize,
        n: usize,
        got_t: usize,
        got_n: usize,
    },

    /// Backend-specific execution error.
    #[error("E0204: backend execution failed\nWhy: {detail}\nFix: check backend driver and hardware connection\nDocs: https://docs.thrindex.com/errors/E0204")]
    Execution { detail: String },
}
