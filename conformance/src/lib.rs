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
//! **Part II (ratified 2026-07-10):** [`CONFORMANCE_ENVELOPE_V0`].
//! - **Final law.** Backends may be certified against this envelope.
//! - Ratified on 120 frozen SHD test samples (crc32=e2ebd845) against the SHD
//!   keyword-spotting model (Dense 700→512→20, 64.66% accuracy).
//! - Per-channel int8 PASSES; per-channel int4 FAILS on all three metrics (ratio 9–16×).
//! - See ADR-0010 Part II Amendment for full evidence.
//!
//! The superseded DRAFT constant [`CONFORMANCE_ENVELOPE_V0_DRAFT`] is retained for
//! historical reference and backwards compatibility in the harness.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use conformance::{CONFORMANCE_ENVELOPE_V0, harness, report};
//! use thrindex_sim::SimBackend;
//!
//! // Reference and backend-under-test are both Backend implementors.
//! let reference = SimBackend::default();
//! let backend_under_test = SimBackend::new(1); // replace with real backend
//!
//! // Load artifact and frozen test samples (≥100 for certification).
//! let artifact_json = std::fs::read_to_string("model.thx").unwrap();
//! let inputs: Vec<Vec<Vec<f32>>> = vec![]; // load from fixtures
//!
//! let report = harness::run_conformance(
//!     &backend_under_test,
//!     &reference,
//!     &artifact_json,
//!     &inputs,
//!     &CONFORMANCE_ENVELOPE_V0,
//! ).unwrap();
//!
//! println!("{}", report.render());
//! println!("Certified: {}", report.passed(&CONFORMANCE_ENVELOPE_V0));
//! ```
pub mod error;
pub mod harness;
pub mod metric;
pub mod report;

pub use report::ConformanceReport;
use report::EnvelopeStatus;

/// THRINDEX conformance envelope v0 — **final law, ratified 2026-07-10**.
///
/// Backends certified against this envelope carry the badge `[THRINDEX Certified v0]`.
///
/// # Ratification evidence (crc32=e2ebd845, 120 frozen SHD samples)
///
/// Quantization model: per-channel symmetric int8, round-half-to-even.
///
/// ```text
/// int8-per-channel  agg_mean=5.28e-3  agg_max=0.100  pred=0.983  → PASSES
/// int4-per-channel  agg_mean=8.38e-2  agg_max=0.920  pred=0.617  → FAILS
/// Separation: T_mean 15.9×  T_max 9.2×  pred_agree gap 0.37
/// ```
///
/// See ADR-0010 Part II Amendment for full derivation, scope, and reopening triggers.
///
/// # Versioning (ADR-0010 Part I §9)
///
/// This constant is snapshot-tested. Any change fails CI and requires a corresponding
/// RFC amendment and founder approval. Tightening any threshold is a breaking change;
/// all v0-certified backends must be explicitly re-certified under the new version.
pub const CONFORMANCE_ENVELOPE_V0: ConformanceEnvelope = ConformanceEnvelope {
    version: "v0",
    status: EnvelopeStatus::Final,
    t_mean_threshold: 0.020,
    t_max_threshold: 0.130,
    pred_agreement_min: 0.900,
    min_test_samples: 100,
};

/// THRINDEX conformance envelope v0 DRAFT — **superseded by [`CONFORMANCE_ENVELOPE_V0`]**.
///
/// Retained for historical reference and backward compatibility in the harness
/// (`--envelope v0_draft`). Do not use for new certification runs.
///
/// # ⚠ SUPERSEDED — use [`CONFORMANCE_ENVELOPE_V0`] for all new work ⚠
///
/// These constants were provisional engineering anchors. The ratified final values
/// are in [`CONFORMANCE_ENVELOPE_V0`]. The key differences:
/// - `T_max`: DRAFT 0.10 (zero margin over int8 observed) → V0 **0.130** (+30% headroom)
/// - `pred_agreement_min`: DRAFT 0.99 (unreachable; int8 observed 0.983) → V0 **0.900**
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
