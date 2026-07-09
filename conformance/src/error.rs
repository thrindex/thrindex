//! Errors produced by the conformance harness.
use thiserror::Error;

/// Error from the conformance harness (E02xx range, ADR-0010 / M4).
#[derive(Debug, Error)]
pub enum ConformanceError {
    /// Test set is smaller than `min_test_samples` in the envelope.
    ///
    /// E0205: enforces the `min_test_samples = 100` requirement. A harness run
    /// with fewer samples produces statistically invalid results and is refused.
    #[error("E0205: test set too small — {got} samples provided, minimum is {required}\nWhy: fewer samples than the envelope's `min_test_samples` cannot statistically characterise the metric distribution\nFix: provide ≥{required} samples; see ADR-0010 Part II for the ratification measurement specification\nDocs: https://docs.thrindex.com/errors/E0205")]
    TestSetTooSmall { got: usize, required: usize },

    /// Backend produced output with a different shape than the reference.
    #[error("E0206: backend output shape mismatch on sample {sample_idx} — reference [{ref_t}×{ref_n}], backend [{got_t}×{got_n}]\nWhy: the backend returned a different number of timesteps or output neurons\nFix: verify the backend runs exactly T={ref_t} timesteps and produces N_out={ref_n} neurons per the artifact\nDocs: https://docs.thrindex.com/errors/E0206")]
    OutputShapeMismatch {
        sample_idx: usize,
        ref_t: usize,
        ref_n: usize,
        got_t: usize,
        got_n: usize,
    },

    /// The backend reported an execution error for one or more samples.
    #[error("E0207: backend execution error on sample {sample_idx}: {detail}\nWhy: the backend-under-test returned an error during the conformance run\nFix: check the backend driver and the artifact compatibility\nDocs: https://docs.thrindex.com/errors/E0207")]
    BackendExecution { sample_idx: usize, detail: String },

    /// The reference sim reported an error — this is a harness bug.
    #[error("E0208: reference simulator error on sample {sample_idx}: {detail}\nWhy: the reference SimBackend returned an error — this indicates a harness or artifact bug, not a backend issue\nFix: verify the artifact is valid; file a bug if the artifact was produced by `thrindex compile`\nDocs: https://docs.thrindex.com/errors/E0208")]
    ReferenceError { sample_idx: usize, detail: String },
}
