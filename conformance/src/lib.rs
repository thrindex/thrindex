//! THRINDEX Certified conformance harness.
//!
//! ## Structure (ADR-0010)
//!
//! This crate implements two separable things, matching the ADR-0010 two-part structure:
//!
//! **Part I (decided):** the metric and pass-rule structure.
//! - [`metric`] — `per_neuron_rate_error`, `mean_rate_error`, `max_rate_error`,
//!   `prediction_agreement`, `hamming_fraction` (informational).
//! - [`harness`] — runs reference + backend, computes metrics, produces a report.
//! - [`report`] — [`ConformanceReport`]: structured result with `passed()` + `render()`.
//!
//! **Part II (provisional):** [`CONFORMANCE_ENVELOPE_V0_DRAFT`].
//! - **Not certification-valid.** No backend may be certified or rejected on these numbers.
//! - Becomes `CONFORMANCE_ENVELOPE_V0` only after the ratification measurement
//!   (per-channel int8, ≥100-sample frozen SHD set) — see ADR-0010 Part II.
//! - All code paths that use this constant must propagate the `DRAFT` status to
//!   the conformance report header. The harness enforces this.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use conformance::{CONFORMANCE_ENVELOPE_V0_DRAFT, harness, report};
//! use thrindex_sim::SimBackend;
//!
//! // Reference and backend-under-test are both Backend implementors.
//! let reference = SimBackend::default();
//! let backend_under_test = SimBackend::new(1); // replace with real backend
//!
//! // Load artifact and frozen test samples (≥100 for real certification).
//! let artifact_json = std::fs::read_to_string("model.thx").unwrap();
//! let inputs: Vec<Vec<Vec<f32>>> = vec![]; // load from fixtures
//!
//! let report = harness::run_conformance(
//!     &backend_under_test,
//!     &reference,
//!     &artifact_json,
//!     &inputs,
//!     &CONFORMANCE_ENVELOPE_V0_DRAFT,
//! ).unwrap();
//!
//! println!("{}", report.render());
//! println!("Passed: {}", report.passed(&CONFORMANCE_ENVELOPE_V0_DRAFT));
//! ```
pub mod error;
pub mod harness;
pub mod metric;
pub mod report;

pub use report::ConformanceReport;
use report::EnvelopeStatus;

/// The conformance envelope used during M4 development.
///
/// # ⚠ DRAFT — NOT CERTIFICATION-VALID ⚠
///
/// These constants are provisional engineering anchors derived from 3 SHD test samples
/// and per-tensor int8 quantization. No backend may be certified or rejected on these
/// values. The envelope becomes `CONFORMANCE_ENVELOPE_V0` only after the ratification
/// measurement described in ADR-0010 Part II.
///
/// The `status` field is [`EnvelopeStatus::Draft`]. The conformance harness asserts
/// this and includes `"DRAFT — not certification-valid"` in every report header when
/// this envelope is used.
pub const CONFORMANCE_ENVELOPE_V0_DRAFT: ConformanceEnvelope = ConformanceEnvelope {
    version: "v0_draft",
    status: EnvelopeStatus::Draft,
    t_mean_threshold: 0.02,
    t_max_threshold: 0.10,
    pred_agreement_min: 0.99,
    min_test_samples: 100,
};

/// A versioned conformance envelope (ADR-0010 Part I §9 versioning rules).
///
/// The numeric thresholds are declared here and nowhere else. Changing them
/// requires bumping `version` and a snapshot-test update (enforced in CI).
#[derive(Debug, Clone, PartialEq)]
pub struct ConformanceEnvelope {
    /// Semantic version string, e.g. `"v0_draft"`, `"v0"`, `"v1"`.
    pub version: &'static str,
    /// Whether this envelope is final or provisional.
    pub status: EnvelopeStatus,
    /// Mean per-neuron rate error threshold (ADR-0010, scaling rule applies for T < 50).
    pub t_mean_threshold: f64,
    /// Max per-neuron rate error threshold (worst-case output neuron).
    pub t_max_threshold: f64,
    /// Minimum prediction-agreement fraction over the test set.
    pub pred_agreement_min: f64,
    /// Minimum frozen test set size. Runs with fewer samples are rejected.
    pub min_test_samples: usize,
}
