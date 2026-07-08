//! LIF neuron single-timestep forward pass.
//!
//! Implements the canonical update order from ADR-0005, matching the Python SDK:
//!
//! ```text
//! (i)   mem_t   = alpha · mem_prev + x_t       (leak + integrate; no syn dynamics)
//!       syn_t   = alpha_syn · syn_prev + x_t   )
//!       mem_t   = alpha · mem_prev + syn_t     ) (with synaptic current)
//! (ii)  spk_t   = (mem_t >= threshold) as f32  (fire — hard threshold, no surrogate)
//! (iii) mem_out = mem_t − spk_t · threshold    (subtract reset)
//!       mem_out = mem_t · (1 − spk_t)          (zero reset)
//! ```
//!
//! `alpha` is **read from the artifact** (ADR-0007, correction 4); this function
//! never calls `exp`.  The simulator is deterministic by construction: no RNG, no
//! global state, no clock reads.

use crate::model::{LifLayer, ResetMode};

/// Per-neuron recurrent state for one LIF layer.
#[derive(Debug, Clone)]
pub struct LifState {
    /// Membrane potential, one value per neuron.
    pub mem: Vec<f32>,
    /// Synaptic current, one value per neuron; `None` when `tau_syn` is disabled.
    pub syn: Option<Vec<f32>>,
}

impl LifState {
    /// Initialise all-zero state for `n` neurons, with or without synaptic current.
    #[must_use]
    pub fn zeros(n: usize, has_syn: bool) -> Self {
        Self {
            mem: vec![0.0f32; n],
            syn: if has_syn { Some(vec![0.0f32; n]) } else { None },
        }
    }
}

/// Execute one timestep of the LIF layer.
///
/// Updates `state` in-place and returns the spike vector for this timestep.
/// The spike vector has the same length as `x` (and `state.mem`).
///
/// # Panics
///
/// Panics in debug builds if `x.len() != state.mem.len()`.
#[inline]
pub fn step(x: &[f32], state: &mut LifState, layer: &LifLayer) -> Vec<f32> {
    debug_assert_eq!(x.len(), state.mem.len());

    let n = x.len();
    let mut spikes = Vec::with_capacity(n);

    match state.syn.as_mut() {
        None => {
            // No synaptic dynamics: direct integration.
            for (xi, mi) in x.iter().zip(state.mem.iter_mut()) {
                // (i) Leak + integrate.
                let mem_t = layer.alpha.mul_add(*mi, *xi);
                // (ii) Fire.
                let spk = if mem_t >= layer.threshold {
                    1.0_f32
                } else {
                    0.0_f32
                };
                // (iii) Reset.
                *mi = match layer.reset {
                    ResetMode::Subtract => mem_t - spk * layer.threshold,
                    ResetMode::Zero => mem_t * (1.0 - spk),
                };
                spikes.push(spk);
            }
        }
        Some(syn) => {
            let alpha_syn = layer.alpha_syn.unwrap_or(1.0);
            for ((xi, mi), si) in x.iter().zip(state.mem.iter_mut()).zip(syn.iter_mut()) {
                // (i) Synaptic current, then membrane integration.
                let syn_t = alpha_syn.mul_add(*si, *xi);
                let mem_t = layer.alpha.mul_add(*mi, syn_t);
                // (ii) Fire.
                let spk = if mem_t >= layer.threshold {
                    1.0_f32
                } else {
                    0.0_f32
                };
                // (iii) Reset.
                *mi = match layer.reset {
                    ResetMode::Subtract => mem_t - spk * layer.threshold,
                    ResetMode::Zero => mem_t * (1.0 - spk),
                };
                *si = syn_t;
                spikes.push(spk);
            }
        }
    }

    spikes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ResetMode;

    fn make_lif(threshold: f32, alpha: f32, reset: ResetMode) -> LifLayer {
        LifLayer {
            threshold,
            alpha,
            alpha_syn: None,
            reset,
        }
    }

    /// Golden: single neuron, no input, threshold 1.0, alpha 0.9.
    /// mem stays at 0.0, no spikes.
    #[test]
    fn golden_no_input_no_spike() {
        let layer = make_lif(1.0, 0.9, ResetMode::Subtract);
        let mut state = LifState::zeros(1, false);
        let spk = step(&[0.0], &mut state, &layer);
        assert_eq!(spk, &[0.0]);
        assert!((state.mem[0] - 0.0).abs() < 1e-7);
    }

    /// Golden: single neuron receives input 1.0 for two timesteps.
    ///
    /// - t=0: `mem_t` = 0.9·0 + 1.0 = 1.0  → spk=1, `mem_out` = 1.0 - 1.0 = 0.0 (subtract)
    /// - t=1: `mem_t` = 0.9·0 + 1.0 = 1.0  → spk=1, `mem_out` = 0.0
    #[test]
    fn golden_spike_at_threshold_subtract() {
        let layer = make_lif(1.0, 0.9, ResetMode::Subtract);
        let mut state = LifState::zeros(1, false);

        let spk0 = step(&[1.0], &mut state, &layer);
        assert_eq!(spk0, &[1.0]);
        assert!((state.mem[0] - 0.0).abs() < 1e-7);

        let spk1 = step(&[1.0], &mut state, &layer);
        assert_eq!(spk1, &[1.0]);
        assert!((state.mem[0] - 0.0).abs() < 1e-7);
    }

    /// Golden: zero-reset mode.
    ///
    /// - t=0: `mem_t` = 0.9·0 + 1.0 = 1.0  → spk=1, `mem_out` = 1.0·(1-1) = 0.0
    #[test]
    fn golden_spike_zero_reset() {
        let layer = make_lif(1.0, 0.9, ResetMode::Zero);
        let mut state = LifState::zeros(1, false);
        let spk = step(&[1.0], &mut state, &layer);
        assert_eq!(spk, &[1.0]);
        assert!((state.mem[0] - 0.0).abs() < 1e-7);
    }

    /// Golden: sub-threshold accumulation.
    ///
    /// Input 0.5 per timestep, threshold 1.0, alpha ≈ 0.9048374.
    ///
    /// - t=0: `mem_t` ≈ 0.5          → no spike
    /// - t=1: `mem_t` ≈ 0.9524       → no spike
    /// - t=2: `mem_t` ≈ 1.36         → spike, `mem_out` ≈ 0.36
    #[test]
    fn golden_subthreshold_accumulation() {
        // alpha = exp(-1/10)
        #[allow(clippy::cast_possible_truncation)]
        let alpha: f32 = (-0.1_f64).exp() as f32;
        let layer = make_lif(1.0, alpha, ResetMode::Subtract);
        let mut state = LifState::zeros(1, false);

        let spk0 = step(&[0.5], &mut state, &layer);
        assert_eq!(spk0, &[0.0]);
        let mem_after_t0 = state.mem[0];
        assert!((mem_after_t0 - 0.5_f32).abs() < 1e-6);

        let spk1 = step(&[0.5], &mut state, &layer);
        assert_eq!(spk1, &[0.0]);
        let expected_t1 = alpha.mul_add(mem_after_t0, 0.5);
        assert!((state.mem[0] - expected_t1).abs() < 1e-6);

        let spk2 = step(&[0.5], &mut state, &layer);
        assert_eq!(spk2, &[1.0], "neuron should have fired by t=2");
    }

    /// Python↔Rust equivalence anchor: the exact alpha used by the Python SDK
    /// for `tau_mem=10` is `exp(-1/10)`.  Verify our step produces the same value.
    #[test]
    fn python_alpha_consistency() {
        // Python: alpha = math.exp(-LIF.DT / tau_mem) = math.exp(-1.0/10.0)
        let alpha_python: f64 = (-1.0_f64 / 10.0_f64).exp();
        #[allow(clippy::cast_possible_truncation)]
        let alpha_rust = alpha_python as f32;
        let layer = make_lif(1.0, alpha_rust, ResetMode::Subtract);
        let mut state = LifState::zeros(1, false);
        let spk = step(&[0.5], &mut state, &layer);
        assert_eq!(spk, &[0.0]);
        assert!((state.mem[0] - alpha_rust.mul_add(0.0, 0.5)).abs() < 1e-7);
    }
}
