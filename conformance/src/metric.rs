//! Conformance metric functions (ADR-0010 Part I).
//!
//! All metrics operate on **discrete step-indexed binary rasters** — `[timesteps][neurons]`
//! where each cell is `{0.0, 1.0}` — per the ADR-0008 representation.
//!
//! ## Primary metric: per-neuron spike-count rate error
//!
//! For each output neuron `n` over `T` timesteps:
//! ```text
//! count[n]      = Σ_{t} 1[spike(t, n)]
//! rate_error[n] = |count_R[n] - count_B[n]| / T
//! ```
//!
//! Pass condition (ADR-0010):
//! ```text
//! mean_n(rate_error[n]) ≤ T_mean   AND   max_n(rate_error[n]) ≤ T_max
//! ```
//!
//! ## Scaling rule for short trials (T < 50)
//!
//! For T < 50, the effective thresholds are:
//! ```text
//! effective T_mean = max(T_mean_threshold, 1.0 / T)   // at most 1 spike difference
//! effective T_max  = max(T_max_threshold,  2.0 / T)   // at most 2 spike differences
//! ```
//!
//! ## Silent and rare-firing neurons
//!
//! - Silent in both (count = 0): `rate_error[n] = 0.0` — never causes a failure.
//! - Fires once (count = 1), missed by backend: `rate_error[n] = 1/T` — naturally bounded.
//! - No minimum floor is needed; the absolute normalization handles it.

/// Compute per-neuron spike-count rate error between two rasters.
///
/// `ref_raster` and `test_raster` have shape `[timesteps][neurons]`.
/// Returns a `Vec` of length `n_neurons` with `rate_error[n] ∈ [0, 1]`.
///
/// # Panics
///
/// Panics if the rasters have different shapes.
#[must_use]
pub fn per_neuron_rate_errors(ref_raster: &[Vec<f32>], test_raster: &[Vec<f32>]) -> Vec<f64> {
    assert_eq!(
        ref_raster.len(),
        test_raster.len(),
        "rasters must have the same number of timesteps"
    );
    if ref_raster.is_empty() {
        return vec![];
    }
    let t = ref_raster.len() as f64;
    let n_neurons = ref_raster[0].len();
    assert_eq!(
        n_neurons,
        test_raster[0].len(),
        "rasters must have the same number of neurons"
    );

    (0..n_neurons)
        .map(|n| {
            let count_r: f64 = ref_raster
                .iter()
                .map(|frame| if frame[n] > 0.5 { 1.0 } else { 0.0 })
                .sum();
            let count_b: f64 = test_raster
                .iter()
                .map(|frame| if frame[n] > 0.5 { 1.0 } else { 0.0 })
                .sum();
            (count_r - count_b).abs() / t
        })
        .collect()
}

/// Mean rate error over all neurons (the primary aggregate metric).
#[must_use]
pub fn mean_rate_error(errors: &[f64]) -> f64 {
    if errors.is_empty() {
        return 0.0;
    }
    errors.iter().sum::<f64>() / errors.len() as f64
}

/// Max rate error over all neurons (worst-case neuron).
#[must_use]
pub fn max_rate_error(errors: &[f64]) -> f64 {
    errors.iter().cloned().fold(0.0_f64, f64::max)
}

/// Effective T_mean threshold, applying the short-trial scaling rule (ADR-0010 Part I §4).
#[must_use]
pub fn effective_t_mean(t_mean_threshold: f64, t_steps: usize) -> f64 {
    if t_steps < 50 {
        f64::max(t_mean_threshold, 1.0 / t_steps as f64)
    } else {
        t_mean_threshold
    }
}

/// Effective T_max threshold, applying the short-trial scaling rule (ADR-0010 Part I §4).
#[must_use]
pub fn effective_t_max(t_max_threshold: f64, t_steps: usize) -> f64 {
    if t_steps < 50 {
        f64::max(t_max_threshold, 2.0 / t_steps as f64)
    } else {
        t_max_threshold
    }
}

/// Prediction: `argmax_n(count[n])` over the raster.
///
/// Returns the neuron index with the highest spike count. Ties broken by lowest index.
#[must_use]
pub fn prediction(raster: &[Vec<f32>]) -> usize {
    if raster.is_empty() || raster[0].is_empty() {
        return 0;
    }
    let n_neurons = raster[0].len();
    let mut best_n = 0;
    let mut best_count: f64 = -1.0;
    for n in 0..n_neurons {
        let count: f64 = raster
            .iter()
            .map(|f| if f[n] > 0.5 { 1.0 } else { 0.0 })
            .sum();
        if count > best_count {
            best_count = count;
            best_n = n;
        }
    }
    best_n
}

/// Prediction agreement: fraction of samples where `argmax` of reference == `argmax` of backend.
///
/// `ref_rasters` and `test_rasters` are both `[batch][timesteps][neurons]`.
/// Returns a value in [0, 1].
///
/// # Panics
///
/// Panics if batch sizes differ.
#[must_use]
pub fn prediction_agreement(ref_rasters: &[Vec<Vec<f32>>], test_rasters: &[Vec<Vec<f32>>]) -> f64 {
    assert_eq!(
        ref_rasters.len(),
        test_rasters.len(),
        "batch sizes must match"
    );
    if ref_rasters.is_empty() {
        return 1.0;
    }
    let agree = ref_rasters
        .iter()
        .zip(test_rasters.iter())
        .filter(|(r, t)| prediction(r) == prediction(t))
        .count();
    agree as f64 / ref_rasters.len() as f64
}

/// Hamming fraction: proportion of `(t, n)` cells where spike values differ.
///
/// **Informational metric only** — not a pass/fail criterion (ADR-0010 Part I §6).
/// Logged in the conformance report for diagnostic comparison with rate-error metrics.
#[must_use]
pub fn hamming_fraction(ref_raster: &[Vec<f32>], test_raster: &[Vec<f32>]) -> f64 {
    let mut total = 0u64;
    let mut diff = 0u64;
    for (fr, ft) in ref_raster.iter().zip(test_raster.iter()) {
        for (&vr, &vt) in fr.iter().zip(ft.iter()) {
            total += 1;
            if (vr > 0.5) != (vt > 0.5) {
                diff += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        diff as f64 / total as f64
    }
}

/// Mean first-spike latency error (steps) over neurons that fire in **both** rasters.
///
/// **Informational metric only** — not a pass/fail criterion (ADR-0010 Part I §6).
/// Returns 0.0 if no neuron fires in both rasters.
#[must_use]
pub fn mean_first_spike_latency_error(ref_raster: &[Vec<f32>], test_raster: &[Vec<f32>]) -> f64 {
    if ref_raster.is_empty() || ref_raster[0].is_empty() {
        return 0.0;
    }
    let n_neurons = ref_raster[0].len();
    let mut total = 0.0f64;
    let mut count = 0u64;
    for n in 0..n_neurons {
        let first_r = ref_raster.iter().position(|f| f[n] > 0.5);
        let first_t = test_raster.iter().position(|f| f[n] > 0.5);
        if let (Some(fr), Some(ft)) = (first_r, first_t) {
            total += (fr as i64 - ft as i64).unsigned_abs() as f64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster(spikes: &[(usize, usize)], t: usize, n: usize) -> Vec<Vec<f32>> {
        let mut r = vec![vec![0.0f32; n]; t];
        for &(step, neuron) in spikes {
            r[step][neuron] = 1.0;
        }
        r
    }

    #[test]
    fn self_distance_is_zero() {
        let r = raster(&[(0, 0), (2, 1), (5, 0)], 10, 4);
        let errors = per_neuron_rate_errors(&r, &r);
        assert!(
            errors.iter().all(|&e| e == 0.0),
            "self-distance must be exactly 0"
        );
        assert_eq!(mean_rate_error(&errors), 0.0);
        assert_eq!(max_rate_error(&errors), 0.0);
    }

    #[test]
    fn silent_neuron_contributes_zero() {
        // Neuron 3 never fires in either raster.
        let ref_r = raster(&[(0, 0), (1, 1)], 10, 4);
        let test_r = raster(&[(0, 0), (1, 1)], 10, 4); // identical
        let errors = per_neuron_rate_errors(&ref_r, &test_r);
        assert_eq!(errors[3], 0.0, "silent neuron contributes exactly 0");
    }

    #[test]
    fn one_spike_miss_equals_one_over_t() {
        // Neuron 0 fires at t=0 in reference but never in backend (T=100).
        let ref_r = raster(&[(0, 0)], 100, 2);
        let test_r = raster(&[], 100, 2);
        let errors = per_neuron_rate_errors(&ref_r, &test_r);
        let expected = 1.0 / 100.0;
        assert!(
            (errors[0] - expected).abs() < 1e-12,
            "1-spike miss → rate_error = 1/T = {expected}; got {}",
            errors[0]
        );
        assert_eq!(errors[1], 0.0);
    }

    #[test]
    fn prediction_agreement_identical_rasters() {
        let r = vec![raster(&[(0, 2), (1, 2), (2, 2)], 10, 5)];
        assert_eq!(prediction_agreement(&r, &r), 1.0);
    }

    #[test]
    fn prediction_agreement_different_argmax() {
        // Reference: neuron 2 has highest count. Backend: neuron 4 has highest count.
        let ref_r = vec![raster(&[(0, 2), (1, 2), (2, 2)], 10, 5)];
        let test_r = vec![raster(&[(0, 4), (1, 4), (2, 4)], 10, 5)];
        assert_eq!(prediction_agreement(&ref_r, &test_r), 0.0);
    }

    #[test]
    fn short_trial_scaling_t10() {
        // For T=10, effective T_mean = max(0.02, 1/10) = 0.10.
        let eff = effective_t_mean(0.02, 10);
        assert!(
            (eff - 0.10).abs() < 1e-12,
            "expected 0.10 for T=10, got {eff}"
        );
    }

    #[test]
    fn short_trial_scaling_t100() {
        // For T=100, effective T_mean = 0.02 (no adjustment).
        let eff = effective_t_mean(0.02, 100);
        assert!(
            (eff - 0.02).abs() < 1e-12,
            "expected 0.02 for T=100, got {eff}"
        );
    }

    #[test]
    fn hamming_is_informational_not_pass_fail() {
        // Verify hamming computes correctly; used as log only.
        let ref_r = raster(&[(0, 0), (1, 1)], 5, 2);
        let test_r = raster(&[(0, 0)], 5, 2); // (1,1) missing
        let h = hamming_fraction(&ref_r, &test_r);
        // 1 cell differs out of 10 total.
        assert!((h - 0.1).abs() < 1e-12, "expected 0.1, got {h}");
    }
}
