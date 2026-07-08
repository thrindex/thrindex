//! Spike raster representation and utilities.
//!
//! A `SpikeRaster` holds the output spike trains from `sim::run` for a single sample,
//! along with pre-computed statistics used in the §29 transcript renderer.

/// Spike raster for one sample.
#[derive(Debug, Clone)]
pub struct SpikeRaster {
    /// Number of timesteps.
    pub timesteps: usize,
    /// Number of input features (used for transcript header).
    pub in_features: usize,
    /// Output spike train: `[timesteps, out_neurons]`.
    pub frames: Vec<Vec<f32>>,
}

impl SpikeRaster {
    /// Build a raster from the raw `SimOutput` frames for one sample.
    #[must_use]
    pub fn from_frames(frames: Vec<Vec<f32>>, in_features: usize) -> Self {
        let timesteps = frames.len();
        Self {
            timesteps,
            in_features,
            frames,
        }
    }

    /// Mean firing rate across all output neurons and all timesteps.
    #[must_use]
    pub fn output_spike_rate(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let n_neurons = self.frames[0].len();
        if n_neurons == 0 {
            return 0.0;
        }
        let total: f64 = self
            .frames
            .iter()
            .flat_map(|f| f.iter())
            .map(|&v| f64::from(v))
            .sum();
        // Precision loss is acceptable for an estimated rate.
        #[allow(clippy::cast_precision_loss)]
        let denom = self.timesteps as f64 * n_neurons as f64;
        total / denom
    }

    /// Return `(predicted_class, rate_score)` from the output layer's mean firing rates.
    ///
    /// `rate_score` is the mean rate of the winning class neuron.
    /// Returns `None` if there are no output neurons.
    ///
    /// # Panics
    ///
    /// Panics if the output frames contain `NaN` values (which the simulator never produces).
    #[must_use]
    pub fn top_prediction(&self) -> Option<(usize, f32)> {
        if self.frames.is_empty() {
            return None;
        }
        let n = self.frames[0].len();
        if n == 0 {
            return None;
        }
        let mut sums = vec![0.0_f32; n];
        for frame in &self.frames {
            for (i, &v) in frame.iter().enumerate() {
                sums[i] += v;
            }
        }
        // Precision loss is acceptable for an estimated rate.
        #[allow(clippy::cast_precision_loss)]
        let t = self.timesteps as f32;
        let rates: Vec<f32> = sums.iter().map(|s| s / t).collect();
        let (best_class, &best_rate) = rates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("no NaN in rates"))
            .expect("n > 0");
        Some((best_class, best_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spike_rate_all_ones() {
        let frames = vec![vec![1.0f32; 4]; 10];
        let raster = SpikeRaster::from_frames(frames, 8);
        assert!((raster.output_spike_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn spike_rate_all_zeros() {
        let frames = vec![vec![0.0f32; 4]; 10];
        let raster = SpikeRaster::from_frames(frames, 8);
        assert!((raster.output_spike_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn top_prediction_argmax() {
        // Neuron 2 fires every timestep, others fire never.
        let mut frames = vec![vec![0.0f32; 4]; 5];
        for f in &mut frames {
            f[2] = 1.0;
        }
        let raster = SpikeRaster::from_frames(frames, 8);
        let (cls, rate) = raster.top_prediction().unwrap();
        assert_eq!(cls, 2);
        assert!((rate - 1.0).abs() < 1e-6);
    }
}
