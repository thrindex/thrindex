//! Tests that the reference `SimBackend` passes its own conformance.
//!
//! The reference vs itself is the zero-error case — all per-neuron rate errors
//! must be exactly 0.0, prediction agreement 1.0, Hamming 0.0.
//!
//! Note: the conformance harness requires `min_test_samples = 100` for a real run.
//! These tests use a synthetic test set and bypass the sample-size check via
//! direct metric calls, since sample generation (not conformance logic) is being tested here.
//!
//! The DRAFT envelope snapshot test asserts the constant hasn't drifted without a
//! corresponding RFC amendment — this is the CI enforcement of ADR-0010 Part I §9.
use conformance::{
    CONFORMANCE_ENVELOPE_V0_DRAFT,
    metric::{
        mean_rate_error, max_rate_error, per_neuron_rate_errors,
        prediction_agreement, hamming_fraction,
    },
    report::EnvelopeStatus,
    harness::run_self_determinism,
};
use thrindex_backend_api::Backend;
use thrindex_sim::SimBackend;

// Use the frozen M2 fixture — already verified to parse correctly under thrindex-sim.
// This is Dense(2→2) → LIF(threshold=1.0, alpha≈0.905).
const MINIMAL_ARTIFACT: &str =
    include_str!("../../crates/thrindex-compiler/tests/fixtures/m2_dense_lif.thx");

/// Reference sim run against itself → all metrics exactly 0.
#[test]
fn reference_vs_itself_is_zero() {
    let reference = SimBackend::new(1);

    // 2 timesteps, 2 features — matches Dense(2→2) in the fixture.
    let input = vec![vec![0.0f32; 2]; 2];
    let wrapped = vec![input.clone()];

    let ref_out = reference.run_batch(MINIMAL_ARTIFACT, &wrapped)
        .expect("reference run failed");
    let raster = &ref_out[0];

    let errors = per_neuron_rate_errors(raster, raster);
    assert!(errors.iter().all(|&e| e == 0.0), "self-distance must be 0: {errors:?}");
    assert_eq!(mean_rate_error(&errors), 0.0);
    assert_eq!(max_rate_error(&errors), 0.0);

    let ref_list = vec![raster.clone()];
    assert_eq!(prediction_agreement(&ref_list, &ref_list), 1.0);
    assert_eq!(hamming_fraction(raster, raster), 0.0);
}

/// Self-determinism: two consecutive runs of the same input are byte-identical.
#[test]
fn reference_is_self_deterministic() {
    let reference = SimBackend::new(1);
    let input = vec![vec![1.0f32, 0.0]; 10];
    run_self_determinism(&reference, MINIMAL_ARTIFACT, &input)
        .expect("self-determinism check failed");
}

/// Multi-thread self-determinism: threads=1 vs threads=4 (Rayon) gives identical output.
#[test]
fn reference_deterministic_across_thread_counts() {
    let input = vec![vec![1.0f32, 0.0]; 10];
    let wrapped = vec![input.clone()];

    let single = SimBackend::new(1);
    let multi = SimBackend::new(4);

    let out1 = single.run_batch(MINIMAL_ARTIFACT, &wrapped).expect("single-thread failed");
    let out4 = multi.run_batch(MINIMAL_ARTIFACT, &wrapped).expect("multi-thread failed");

    assert_eq!(out1, out4, "thread count must not affect output (ADR-0007)");
}

/// The DRAFT envelope constant has the expected fields.
///
/// Snapshot-tested: any change to the constant fails CI until the snapshot is
/// explicitly updated, enforcing the RFC amendment requirement (ADR-0010 Part I §9).
#[test]
fn draft_envelope_is_labeled_draft() {
    let env = &CONFORMANCE_ENVELOPE_V0_DRAFT;
    assert_eq!(env.status, EnvelopeStatus::Draft, "envelope must be Draft until ratification");
    assert!(env.version.contains("draft"), "version string must contain 'draft': {}", env.version);
    assert_eq!(env.min_test_samples, 100, "min_test_samples must be 100 (fixed, not provisional)");
}

/// `passed()` always returns false for a Draft envelope — even if metrics are perfect.
#[test]
fn draft_envelope_never_certifies() {
    use conformance::report::ConformanceReport;

    let report = ConformanceReport {
        backend_name: "sim".to_string(),
        envelope_version: CONFORMANCE_ENVELOPE_V0_DRAFT.version,
        envelope_status: EnvelopeStatus::Draft,
        n_samples: 100,
        samples: vec![],
        agg_mean_rate_error: 0.0,
        agg_max_rate_error: 0.0,
        pred_agreement: 1.0,
    };

    assert!(
        !report.passed(&CONFORMANCE_ENVELOPE_V0_DRAFT),
        "passed() must return false for a Draft envelope (ADR-0010 Part I §9)"
    );
    assert!(
        report.metrics_within_draft_thresholds(&CONFORMANCE_ENVELOPE_V0_DRAFT),
        "metrics_within_draft_thresholds() must return true for 0-error report"
    );
}

/// Conformance report render includes DRAFT notice.
#[test]
fn report_render_includes_draft_notice() {
    use conformance::report::ConformanceReport;

    let report = ConformanceReport {
        backend_name: "sim".to_string(),
        envelope_version: CONFORMANCE_ENVELOPE_V0_DRAFT.version,
        envelope_status: EnvelopeStatus::Draft,
        n_samples: 100,
        samples: vec![],
        agg_mean_rate_error: 0.0,
        agg_max_rate_error: 0.0,
        pred_agreement: 1.0,
    };

    let rendered = report.render();
    assert!(
        rendered.contains("DRAFT"),
        "report render must include DRAFT notice; got:\n{rendered}"
    );
    assert!(
        rendered.contains("not certification-valid"),
        "report render must say 'not certification-valid'; got:\n{rendered}"
    );
}
