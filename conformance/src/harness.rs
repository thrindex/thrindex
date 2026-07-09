//! Conformance harness: runs reference + backend, computes metrics, produces a report.
//!
//! ## Protocol
//!
//! 1. Assert test set size ≥ `envelope.min_test_samples` (E0205 if not).
//! 2. For each sample:
//!    a. Run the reference [`Backend`] (expected: `SimBackend`).
//!    b. Run the backend-under-test.
//!    c. Assert output shapes match (E0206 if not).
//!    d. Compute `per_neuron_rate_errors`, `hamming_fraction`, FSL error.
//!    e. Record prediction agreement.
//! 3. Aggregate: mean of `mean_rate_error`, max of `max_rate_error`, fraction of agrees.
//! 4. Return [`ConformanceReport`].
//!
//! ## Self-determinism check
//!
//! Call [`run_self_determinism`] to assert that the backend-under-test returns
//! byte-identical output on two consecutive runs of the same inputs. This is
//! required by ADR-0007 for any backend that claims deterministic execution.
use thrindex_backend_api::Backend;

use crate::{
    ConformanceEnvelope,
    error::ConformanceError,
    metric::{
        hamming_fraction, mean_first_spike_latency_error, mean_rate_error, max_rate_error,
        per_neuron_rate_errors, prediction, prediction_agreement,
    },
    report::{ConformanceReport, SampleMetrics},
};

/// Run the full conformance protocol and return a [`crate::ConformanceReport`].
///
/// `inputs` is `[n_samples][timesteps][features]` — the frozen test set.
/// `envelope` is the envelope to evaluate against (use [`crate::CONFORMANCE_ENVELOPE_V0_DRAFT`]
/// during M4 development).
///
/// The report header will state DRAFT if the envelope is provisional.
///
/// # Errors
///
/// - [`ConformanceError::TestSetTooSmall`] if `inputs.len() < envelope.min_test_samples`.
/// - [`ConformanceError::ReferenceError`] if the reference sim fails on any sample.
/// - [`ConformanceError::BackendExecution`] if the backend-under-test fails on any sample.
/// - [`ConformanceError::OutputShapeMismatch`] if backend output shape != reference.
pub fn run_conformance(
    backend: &dyn Backend,
    reference: &dyn Backend,
    artifact_json: &str,
    inputs: &[Vec<Vec<f32>>],
    envelope: &ConformanceEnvelope,
) -> Result<ConformanceReport, ConformanceError> {
    let n_samples = inputs.len();
    if n_samples < envelope.min_test_samples {
        return Err(ConformanceError::TestSetTooSmall {
            got: n_samples,
            required: envelope.min_test_samples,
        });
    }

    let mut sample_metrics = Vec::with_capacity(n_samples);
    let mut all_ref_rasters: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_samples);
    let mut all_test_rasters: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_samples);

    for (idx, input) in inputs.iter().enumerate() {
        let wrapped = vec![input.clone()];

        let ref_out = reference
            .run_batch(artifact_json, &wrapped)
            .map_err(|e| ConformanceError::ReferenceError {
                sample_idx: idx,
                detail: e.to_string(),
            })?;

        let test_out = backend
            .run_batch(artifact_json, &wrapped)
            .map_err(|e| ConformanceError::BackendExecution {
                sample_idx: idx,
                detail: e.to_string(),
            })?;

        let ref_raster = &ref_out[0];
        let test_raster = &test_out[0];

        let ref_t = ref_raster.len();
        let ref_n = if ref_t > 0 { ref_raster[0].len() } else { 0 };
        let got_t = test_raster.len();
        let got_n = if got_t > 0 { test_raster[0].len() } else { 0 };

        if ref_t != got_t || ref_n != got_n {
            return Err(ConformanceError::OutputShapeMismatch {
                sample_idx: idx,
                ref_t,
                ref_n,
                got_t,
                got_n,
            });
        }

        let errors = per_neuron_rate_errors(ref_raster, test_raster);
        let hamming = hamming_fraction(ref_raster, test_raster);
        let fsl_error = mean_first_spike_latency_error(ref_raster, test_raster);

        sample_metrics.push(SampleMetrics {
            hamming,
            mean_rate_error: mean_rate_error(&errors),
            max_rate_error: max_rate_error(&errors),
            mean_fsl_error: fsl_error,
            prediction_agrees: prediction(ref_raster) == prediction(test_raster),
        });

        all_ref_rasters.push(ref_raster.clone());
        all_test_rasters.push(test_raster.clone());
    }

    let agg_mean = sample_metrics
        .iter()
        .map(|m| m.mean_rate_error)
        .sum::<f64>()
        / n_samples as f64;
    let agg_max = sample_metrics
        .iter()
        .map(|m| m.max_rate_error)
        .fold(0.0_f64, f64::max);
    let pred_agree = prediction_agreement(&all_ref_rasters, &all_test_rasters);

    Ok(ConformanceReport {
        backend_name: backend.capability().name.clone(),
        envelope_version: envelope.version,
        envelope_status: envelope.status,
        n_samples,
        samples: sample_metrics,
        agg_mean_rate_error: agg_mean,
        agg_max_rate_error: agg_max,
        pred_agreement: pred_agree,
    })
}

/// Assert that the backend returns identical output on two consecutive runs of the same
/// input (self-determinism check, ADR-0007).
///
/// # Errors
///
/// - [`ConformanceError::BackendExecution`] if either run fails.
pub fn run_self_determinism(
    backend: &dyn Backend,
    artifact_json: &str,
    sample: &[Vec<f32>],
) -> Result<(), ConformanceError> {
    let wrapped = vec![sample.to_vec()];
    let run1 = backend
        .run_batch(artifact_json, &wrapped)
        .map_err(|e| ConformanceError::BackendExecution { sample_idx: 0, detail: e.to_string() })?;
    let run2 = backend
        .run_batch(artifact_json, &wrapped)
        .map_err(|e| ConformanceError::BackendExecution { sample_idx: 0, detail: e.to_string() })?;

    let r1 = &run1[0];
    let r2 = &run2[0];
    assert_eq!(
        r1.len(),
        r2.len(),
        "self-determinism: run1 and run2 have different timestep counts"
    );
    for (t, (f1, f2)) in r1.iter().zip(r2.iter()).enumerate() {
        for (n, (&v1, &v2)) in f1.iter().zip(f2.iter()).enumerate() {
            assert_eq!(
                v1, v2,
                "self-determinism failure at t={t}, n={n}: run1={v1}, run2={v2}"
            );
        }
    }
    Ok(())
}
