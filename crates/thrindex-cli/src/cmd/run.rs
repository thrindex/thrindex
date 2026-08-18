//! `thrindex run <model.thx>` — load artifact, run simulation, print §29 transcript.

use clap::Args;
use thrindex_sim::{
    SimConfig, model,
    raster::SpikeRaster,
    sim::{self, render_transcript},
};

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Path to the `.thx` artifact.
    pub artifact: String,

    /// Encoder seed (integer).  Passed through for transcript provenance only —
    /// the simulator itself contains no RNG (correction 2 / ADR-0007).
    #[arg(long, default_value = "0")]
    pub seed: u64,

    /// Number of threads.  0 = Rayon default (hardware concurrency).
    #[arg(long, default_value = "0")]
    pub threads: usize,
}

/// Execute `thrindex run`.  Returns the transcript string (for snapshot testing).
pub fn run(args: &RunArgs) -> Result<String, thrindex_sim::SimError> {
    let resolved = model::load(&args.artifact)?;

    // For the demo / golden-path invocation, generate a synthetic input spike train
    // whose dimensions match the model's first layer.  Real use: callers provide
    // pre-encoded spike trains via the Python SDK.
    let in_features = first_in_features(&resolved);
    let t_steps = 100_usize;
    let input = synthetic_input(in_features, t_steps, args.seed);

    let config = SimConfig {
        threads: args.threads,
    };
    let output = sim::run(&resolved, &[input], &config)?;

    let raster = SpikeRaster::from_frames(output.spikes[0].clone(), in_features);

    let transcript = render_transcript(
        &resolved,
        &output,
        &raster,
        Some(args.seed),
        env!("CARGO_PKG_VERSION"),
    );

    Ok(transcript)
}

/// Return the number of input features expected by the first layer.
fn first_in_features(model: &thrindex_sim::model::ResolvedModel) -> usize {
    use thrindex_sim::model::ResolvedLayer;
    model
        .layers
        .iter()
        .find_map(|l| match l {
            ResolvedLayer::Dense(d) => Some(d.in_features),
            ResolvedLayer::Conv2d(c) => Some(c.in_channels),
            ResolvedLayer::Lif(_) | ResolvedLayer::Flatten(_) => None,
        })
        .unwrap_or(1)
}

/// Generate a deterministic synthetic spike train from a seed.
///
/// Uses a simple linear-congruential generator — this is NOT the canonical Xoshiro256**
/// encoder.  It exists only to make `thrindex run` self-contained for the demo/golden
/// path without requiring an external spike file.  Real workloads supply pre-encoded
/// spike trains through the Python SDK.
fn synthetic_input(n_features: usize, t_steps: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed.wrapping_add(6_364_136_223_846_793_005);
    let mut frames = Vec::with_capacity(t_steps);
    for _ in 0..t_steps {
        let frame: Vec<f32> = (0..n_features)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Spike with ~20% probability.
                if (state >> 33).is_multiple_of(5) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        frames.push(frame);
    }
    frames
}
