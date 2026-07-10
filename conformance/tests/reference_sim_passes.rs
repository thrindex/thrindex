//! Tests that the reference `SimBackend` passes its own conformance.
//!
//! The reference vs itself is the zero-error case — all per-neuron rate errors
//! must be exactly 0.0, prediction agreement 1.0, Hamming 0.0.
//!
//! The V0 envelope snapshot test (`draft_envelope_is_labeled_draft`,
//! `v0_envelope_constants`) asserts that the constants haven't drifted without
//! a corresponding RFC amendment — this is the CI enforcement of ADR-0010 Part I §9.
use conformance::{
    CONFORMANCE_ENVELOPE_V0, CONFORMANCE_ENVELOPE_V0_DRAFT,
    harness::{run_conformance, run_self_determinism},
    metric::{
        hamming_fraction, max_rate_error, mean_rate_error, per_neuron_rate_errors,
        prediction_agreement,
    },
    report::EnvelopeStatus,
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

    let ref_out = reference
        .run_batch(MINIMAL_ARTIFACT, &wrapped)
        .expect("reference run failed");
    let raster = &ref_out[0];

    let errors = per_neuron_rate_errors(raster, raster);
    assert!(
        errors.iter().all(|&e| e == 0.0),
        "self-distance must be 0: {errors:?}"
    );
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

    let out1 = single
        .run_batch(MINIMAL_ARTIFACT, &wrapped)
        .expect("single-thread failed");
    let out4 = multi
        .run_batch(MINIMAL_ARTIFACT, &wrapped)
        .expect("multi-thread failed");

    assert_eq!(out1, out4, "thread count must not affect output (ADR-0007)");
}

/// The DRAFT envelope constant has the expected fields.
///
/// Snapshot-tested: any change to the constant fails CI until the snapshot is
/// explicitly updated, enforcing the RFC amendment requirement (ADR-0010 Part I §9).
#[test]
fn draft_envelope_is_labeled_draft() {
    let env = &CONFORMANCE_ENVELOPE_V0_DRAFT;
    assert_eq!(
        env.status,
        EnvelopeStatus::Draft,
        "DRAFT envelope must have Draft status"
    );
    assert!(
        env.version.contains("draft"),
        "DRAFT version string must contain 'draft': {}",
        env.version
    );
    assert_eq!(env.min_test_samples, 100, "min_test_samples must be 100");
}

/// V0 envelope constants — snapshot guard for ADR-0010 Part I §9.
///
/// Any change to these values fails CI and requires a founder-approved RFC amendment.
#[test]
fn v0_envelope_constants() {
    let env = &CONFORMANCE_ENVELOPE_V0;
    assert_eq!(env.status, EnvelopeStatus::Final, "V0 must be Final");
    assert_eq!(env.version, "v0");
    assert_eq!(
        env.t_mean_threshold, 0.020,
        "T_mean must be 0.020 (ADR-0010 Part II Amendment)"
    );
    assert_eq!(
        env.t_max_threshold, 0.130,
        "T_max must be 0.130 (ADR-0010 Part II Amendment)"
    );
    assert_eq!(
        env.pred_agreement_min, 0.900,
        "P_min must be 0.900 (ADR-0010 Part II Amendment)"
    );
    assert_eq!(env.min_test_samples, 100);
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
        envelope_t_mean: CONFORMANCE_ENVELOPE_V0_DRAFT.t_mean_threshold,
        envelope_t_max: CONFORMANCE_ENVELOPE_V0_DRAFT.t_max_threshold,
        envelope_p_min: CONFORMANCE_ENVELOPE_V0_DRAFT.pred_agreement_min,
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

/// The float reference sim passes its own V0 envelope.
///
/// Reference vs itself produces zero rate error and 1.0 prediction agreement on
/// every sample — it must always clear the V0 thresholds decisively. This is the
/// THRINDEX Certified badge test for the reference implementation.
#[test]
fn v0_envelope_certifies_reference_sim() {
    // Use 100 identical minimal inputs to satisfy min_test_samples=100.
    // Reference vs itself gives zero error regardless of content.
    let inputs: Vec<Vec<Vec<f32>>> = (0..100).map(|_| vec![vec![0.0f32; 2]; 2]).collect();

    let reference = SimBackend::new(1);
    let backend = SimBackend::new(1);

    let report = run_conformance(
        &backend,
        &reference,
        MINIMAL_ARTIFACT,
        &inputs,
        &CONFORMANCE_ENVELOPE_V0,
    )
    .expect("conformance run must succeed");

    assert_eq!(
        report.agg_mean_rate_error, 0.0,
        "float ref vs itself: zero mean error"
    );
    assert_eq!(
        report.agg_max_rate_error, 0.0,
        "float ref vs itself: zero max error"
    );
    assert_eq!(
        report.pred_agreement, 1.0,
        "float ref vs itself: perfect prediction agreement"
    );

    assert!(
        report.passed(&CONFORMANCE_ENVELOPE_V0),
        "reference SimBackend must be THRINDEX Certified [v0] — \
         0 error passes all V0 thresholds decisively"
    );

    let rendered = report.render();
    assert!(
        rendered.contains("PASS — THRINDEX Certified [v0]"),
        "report render must show certification badge; got:\n{rendered}"
    );
}

/// Conformance report render includes DRAFT notice for Draft envelopes.
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
        envelope_t_mean: CONFORMANCE_ENVELOPE_V0_DRAFT.t_mean_threshold,
        envelope_t_max: CONFORMANCE_ENVELOPE_V0_DRAFT.t_max_threshold,
        envelope_p_min: CONFORMANCE_ENVELOPE_V0_DRAFT.pred_agreement_min,
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
