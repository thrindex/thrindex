//! Temporal unrolling engine.
//!
//! Execution model (corrections 1a, 8):
//! - Parallelism is **across samples only** (Rayon).  Within one sample, the timestep
//!   loop is strictly sequential.  This guarantees:
//!   `threads=1` and `threads=N` on the same input → **byte-identical** spike rasters.
//! - The simulator contains **zero RNG** (correction 2).  All input spike trains are
//!   provided by the caller; the encoder stage (which holds the PRNG) runs before this.
//! - `alpha` and `alpha_syn` are read from the artifact — never recomputed (correction
//!   4 / ADR-0007).

use rayon::prelude::*;

use crate::error::SimError;
use crate::lif::{LifState, step as lif_step};
use crate::model::{DenseLayer, ResolvedLayer, ResolvedModel};
use crate::raster::SpikeRaster;
use crate::spatial::{self, Activation};

/// Configuration for a simulation run.
#[derive(Debug, Clone, Default)]
pub struct SimConfig {
    /// Thread count.  `1` = single-threaded.  `0` = Rayon default (hardware concurrency).
    pub threads: usize,
}

/// Output of a completed simulation run.
#[derive(Debug, Clone)]
pub struct SimOutput {
    /// Spike rasters: `[batch, timesteps, neurons_out]`.
    pub spikes: Vec<Vec<Vec<f32>>>,
    /// Aggregated statistics.
    pub stats: SimStats,
}

/// Per-run statistics.
#[derive(Debug, Clone, Default)]
pub struct SimStats {
    /// Total number of spikes fired across all samples, timesteps, and neurons.
    pub total_spikes: u64,
    /// Total synaptic operations (`pre_spikes` × `post_neurons`, summed over layers).
    pub synaptic_ops: u64,
    /// Wall-clock time of the simulation in seconds.
    pub wall_secs: f64,
}

/// Internal alias to keep complex-type clippy lint quiet.
type SampleResult = (Vec<Vec<f32>>, SimStats);

/// Run the simulator on a batch of input spike trains.
///
/// `inputs` has shape `[batch, timesteps, in_features]`.
/// Each sample is processed independently, then results are collected in order.
///
/// Thread count is set via `config.threads`:
/// - `0` → Rayon default pool (hardware concurrency).
/// - `1` → single-threaded; sequential for a single sample.
/// - `N` → Rayon pool capped at N threads.
///
/// **Self-determinism guarantee**: same `inputs`, same `model`, any `threads` value →
/// byte-identical `SimOutput` (ADR-0007, correction 1a, correction 8).
///
/// # Errors
///
/// Returns [`SimError`] if:
/// - Input dimensions do not match the model's first layer (`E0010`).
pub fn run(
    model: &ResolvedModel,
    inputs: &[Vec<Vec<f32>>],
    config: &SimConfig,
) -> Result<SimOutput, SimError> {
    if inputs.is_empty() {
        return Ok(SimOutput {
            spikes: vec![],
            stats: SimStats::default(),
        });
    }

    // Validate input dimensions against the first layer.
    let first_in = first_input_dim(model);
    if let Some(expected) = first_in {
        for sample in inputs {
            for frame in sample {
                if frame.len() != expected {
                    return Err(SimError::InputDimensionMismatch {
                        expected,
                        got: frame.len(),
                    });
                }
            }
        }
    }

    let start = std::time::Instant::now();

    // Build a Rayon pool sized to the requested thread count.
    let pool = build_pool(config.threads);

    // Process each sample in parallel (across samples; sequential within each sample).
    let results: Vec<Result<SampleResult, SimError>> = pool.install(|| {
        inputs
            .par_iter()
            .map(|sample| run_one_sample(model, sample))
            .collect()
    });

    // Collect results, propagating the first error.
    let mut all_spikes = Vec::with_capacity(inputs.len());
    let mut agg = SimStats::default();
    for res in results {
        let (spikes, sample_stats) = res?;
        agg.total_spikes += sample_stats.total_spikes;
        agg.synaptic_ops += sample_stats.synaptic_ops;
        all_spikes.push(spikes);
    }
    agg.wall_secs = start.elapsed().as_secs_f64();

    Ok(SimOutput {
        spikes: all_spikes,
        stats: agg,
    })
}

// ── Per-sample execution ──────────────────────────────────────────────────────

/// Run one input sample through the full model, one timestep at a time.
fn run_one_sample(model: &ResolvedModel, input: &[Vec<f32>]) -> Result<SampleResult, SimError> {
    let t_steps = input.len();
    if t_steps == 0 {
        return Ok((vec![], SimStats::default()));
    }

    // Allocate LIF states lazily (first timestep determines neuron count).
    let mut lif_states: Vec<Option<LifState>> = vec![None; model.layers.len()];
    let mut delay_lines: Vec<Option<DelayLine>> = vec![None; model.layers.len()];
    let mut output_frames: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
    let mut sample_stats = SimStats::default();

    for frame in input {
        let mut act = Activation::flat(frame.clone());

        for (i, layer) in model.layers.iter().enumerate() {
            match layer {
                ResolvedLayer::Dense(d) => {
                    let features = act.as_features(i)?.to_vec();
                    let (out, ops) = if let Some(delays) = &d.delays {
                        let line = delay_lines[i].get_or_insert_with(|| {
                            DelayLine::new(delays.iter().copied().max().unwrap_or(0))
                        });
                        dense_delayed(&features, d, delays, line)
                    } else {
                        dense_forward(&features, d)
                    };
                    sample_stats.synaptic_ops += ops;
                    act = Activation::flat(out);
                }
                ResolvedLayer::Conv2d(conv) => {
                    let (out, ops) = spatial::conv2d(&act, conv, i)?;
                    sample_stats.synaptic_ops += ops;
                    act = out;
                }
                ResolvedLayer::Flatten(flat) => {
                    act = spatial::flatten(&act, flat, i)?;
                }
                ResolvedLayer::Lif(lif) => {
                    if lif_states[i].is_none() {
                        lif_states[i] = Some(LifState::zeros(act.values.len(), lif.alpha_syn.is_some()));
                    }
                    let lif_state = lif_states[i].as_mut().expect("just initialised");
                    let spikes = lif_step(&act.values, lif_state, lif);
                    let spk_count = spikes.iter().filter(|&&v| v > 0.5).count() as u64;
                    sample_stats.total_spikes += spk_count;
                    act.values = spikes;
                }
            }
        }

        // Spatial output is already stored channel-height-width, width fastest,
        // which is the vector a later dense layer or the returned raster reads.
        output_frames.push(act.values);
    }

    Ok((output_frames, sample_stats))
}

// ── Layer kernels ─────────────────────────────────────────────────────────────

/// Dense (fully-connected) forward pass.
///
/// Returns `(output, synaptic_ops_count)`.
/// ops = number of multiply-accumulate operations = `spikes_in` × `out_features`.
fn dense_forward(x: &[f32], layer: &DenseLayer) -> (Vec<f32>, u64) {
    let m = layer.out_features;
    let n = layer.in_features;
    let mut out = match &layer.bias {
        Some(b) => b.clone(),
        None => vec![0.0f32; m],
    };

    // Count pre-synaptic spikes for energy model.
    let spike_count = x.iter().filter(|&&v| v > 0.5).count() as u64;
    let ops = spike_count * m as u64;

    for (i, row) in out.iter_mut().enumerate() {
        let w_row = &layer.weights[i * n..(i + 1) * n];
        let dot: f32 = w_row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
        *row += dot;
    }

    (out, ops)
}

/// Inputs seen by one dense layer, so a connection can read `d` steps back.
#[derive(Clone)]
struct DelayLine {
    cap: usize,
    slots: Vec<Vec<f32>>,
    next: usize,
    /// Timesteps stored so far, including the one just written.
    seen: u64,
}

impl DelayLine {
    fn new(max_delay: u16) -> Self {
        let cap = usize::from(max_delay) + 1;
        Self {
            cap,
            slots: Vec::with_capacity(cap),
            next: 0,
            seen: 0,
        }
    }

    fn read(&self, delay: u16, feature: usize) -> f32 {
        let delay = u64::from(delay);
        if delay >= self.seen {
            return 0.0;
        }
        let slot = (self.seen - 1 - delay) % self.cap as u64;
        self.slots[usize::try_from(slot).unwrap_or(0)][feature]
    }
}

/// Dense forward where connection `(out, inp)` reads the input from `delays` steps ago.
///
/// Delay 0 is the current frame. A delay longer than the steps seen so far reads 0.
fn dense_delayed(
    x: &[f32],
    layer: &DenseLayer,
    delays: &[u16],
    line: &mut DelayLine,
) -> (Vec<f32>, u64) {
    if line.slots.len() < line.cap {
        line.slots.push(x.to_vec());
    } else {
        line.slots[line.next] = x.to_vec();
    }
    line.seen += 1;
    let m = layer.out_features;
    let n = layer.in_features;
    let mut out = match &layer.bias {
        Some(b) => b.clone(),
        None => vec![0.0_f32; m],
    };
    let mut ops = 0_u64;
    for i in 0..m {
        let mut acc = out[i];
        let w_row = &layer.weights[i * n..(i + 1) * n];
        let d_row = &delays[i * n..(i + 1) * n];
        for (j, (&weight, &delay)) in w_row.iter().zip(d_row.iter()).enumerate() {
            let sample = line.read(delay, j);
            acc += weight * sample;
            if sample > 0.5 {
                ops += 1;
            }
        }
        out[i] = acc;
    }
    line.next = (line.next + 1) % line.cap;
    (out, ops)
}

/// Build a Rayon thread pool for the requested concurrency.
fn build_pool(threads: usize) -> rayon::ThreadPool {
    let n = if threads == 0 {
        rayon::current_num_threads()
    } else {
        threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool creation should not fail")
}

/// Return the expected number of input features for the first non-LIF layer.
fn first_input_dim(model: &ResolvedModel) -> Option<usize> {
    for layer in &model.layers {
        match layer {
            ResolvedLayer::Dense(d) => return Some(d.in_features),
            ResolvedLayer::Conv2d(c) => {
                return match (c.input_height, c.input_width) {
                    (Some(h), Some(w)) => Some(c.in_channels * h * w),
                    _ => None,
                };
            }
            ResolvedLayer::Lif(_) | ResolvedLayer::Flatten(_) => {}
        }
    }
    None
}

// ── Render the §29 transcript ─────────────────────────────────────────────────

/// Render a §29-format transcript string from a simulation result.
///
/// This function lives in the library so the Rust CLI and the Python CLI wrapper
/// both call the same rendering code — one implementation, two callers.
#[must_use]
pub fn render_transcript(
    model: &ResolvedModel,
    output: &SimOutput,
    raster: &SpikeRaster,
    seed: Option<u64>,
    version: &str,
) -> String {
    use std::fmt::Write as _;

    let mut s = String::new();
    let border = "═".repeat(55);
    let sep = "─".repeat(55);

    let seed_str = seed.map_or("(none)".to_string(), |v| v.to_string());
    writeln!(s, "{border}").ok();
    writeln!(
        s,
        " thrindex {version}  |  target: sim  |  seed: {seed_str}"
    )
    .ok();
    writeln!(s, "{border}").ok();
    writeln!(s, " model:  {}", model_summary(model)).ok();
    writeln!(
        s,
        " input:  {} timesteps × {} features",
        raster.timesteps, raster.in_features
    )
    .ok();
    writeln!(s, "{sep}").ok();

    // Prediction from mean firing rate.
    if let Some((pred_class, rate_score)) = raster.top_prediction() {
        writeln!(
            s,
            " prediction:  class {pred_class}  —  rate score {rate_score:.3}"
        )
        .ok();
    }
    writeln!(s, "{sep}").ok();

    // Spike stats.
    let out_rate = raster.output_spike_rate();
    writeln!(s, " output spike rate:   {:.1}%", out_rate * 100.0).ok();

    // Energy estimate (correction 6 format).
    let coefficient_pj: f64 = 0.5;
    // Cast is intentional: u64 to f64 may lose precision for very large counts,
    // but this is an estimate — precision loss is acceptable.
    #[allow(clippy::cast_precision_loss)]
    let energy_nj = output.stats.synaptic_ops as f64 * coefficient_pj * 1e-3;
    writeln!(s, " synaptic ops:        {}", output.stats.synaptic_ops).ok();
    writeln!(
        s,
        " modeled energy:      {energy_nj:.2} nJ  (coefficient: {coefficient_pj} pJ/syn-op — see docs/energy.md)",
    )
    .ok();
    writeln!(s, " sim wall time:       {:.3}s", output.stats.wall_secs).ok();
    writeln!(s, "{border}").ok();

    s
}

/// One-line architecture summary.
fn model_summary(model: &ResolvedModel) -> String {
    model
        .layers
        .iter()
        .map(|l| match l {
            ResolvedLayer::Dense(d) => format!("Dense({}→{})", d.in_features, d.out_features),
            ResolvedLayer::Lif(lif) => {
                // tau_mem = -1.0 / ln(alpha)  (inverse of ADR-0005 formula)
                let tau = -1.0_f64 / f64::from(lif.alpha).ln();
                format!("LIF(τ={tau:.0}ms)")
            }
            ResolvedLayer::Conv2d(c) => {
                format!(
                    "Conv2d({}→{},{}x{})",
                    c.in_channels, c.out_channels, c.kernel_h, c.kernel_w
                )
            }
            ResolvedLayer::Flatten(_) => "Flatten".to_string(),
        })
        .collect::<Vec<_>>()
        .join(" → ")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn tiny_dense_lif_model() -> ResolvedModel {
        use crate::model::{DenseLayer, LifLayer, ResetMode};
        ResolvedModel {
            target: "sim".into(),
            layers: vec![
                ResolvedLayer::Dense(DenseLayer {
                    in_features: 4,
                    out_features: 4,
                    weights: vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                        1.0,
                    ],
                    bias: None,
                    delays: None,
                }),
                ResolvedLayer::Lif(LifLayer {
                    threshold: 1.0,
                    alpha: 0.9_f32,
                    alpha_syn: None,
                    reset: ResetMode::Subtract,
                }),
            ],
        }
    }

    #[test]
    fn dense_delay_of_one_shifts_the_input() {
        use crate::model::{DenseLayer, ResolvedLayer};
        let model = ResolvedModel {
            target: "sim".into(),
            layers: vec![ResolvedLayer::Dense(DenseLayer {
                in_features: 1,
                out_features: 1,
                weights: vec![1.0],
                bias: None,
                delays: Some(vec![1]),
            })],
        };
        let sample = vec![vec![1.0], vec![0.0], vec![0.0]];
        let out = run(&model, &[sample], &SimConfig { threads: 1 }).unwrap();
        assert_eq!(out.spikes[0], vec![vec![0.0], vec![1.0], vec![0.0]]);
    }

    fn spike_input(t: usize, n: usize, value: f32) -> Vec<Vec<f32>> {
        vec![vec![value; n]; t]
    }

    // ── Self-determinism: threads=1 vs threads=4 (correction 1a, correction 8) ──

    #[test]
    fn self_determinism_threads_1_vs_4() {
        let model = tiny_dense_lif_model();
        let input = spike_input(20, 4, 0.5);

        let out1 = run(
            &model,
            std::slice::from_ref(&input),
            &SimConfig { threads: 1 },
        )
        .unwrap();
        let out4 = run(
            &model,
            std::slice::from_ref(&input),
            &SimConfig { threads: 4 },
        )
        .unwrap();

        assert_eq!(
            out1.spikes, out4.spikes,
            "threads=1 and threads=4 must produce identical spikes"
        );
    }

    #[test]
    fn self_determinism_repeated_run() {
        let model = tiny_dense_lif_model();
        let input = spike_input(20, 4, 0.8);

        let out_a = run(
            &model,
            std::slice::from_ref(&input),
            &SimConfig { threads: 1 },
        )
        .unwrap();
        let out_b = run(
            &model,
            std::slice::from_ref(&input),
            &SimConfig { threads: 1 },
        )
        .unwrap();

        assert_eq!(
            out_a.spikes, out_b.spikes,
            "repeated runs must produce identical spikes"
        );
    }

    // ── Basic correctness ──────────────────────────────────────────────────────

    #[test]
    fn identity_weights_spikes_pass_through() {
        let model = tiny_dense_lif_model();
        let input = spike_input(5, 4, 1.0);
        let out = run(
            &model,
            std::slice::from_ref(&input),
            &SimConfig { threads: 1 },
        )
        .unwrap();
        for frame in &out.spikes[0] {
            assert_eq!(frame, &[1.0, 1.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn zero_input_no_spikes() {
        let model = tiny_dense_lif_model();
        let input = spike_input(10, 4, 0.0);
        let out = run(
            &model,
            std::slice::from_ref(&input),
            &SimConfig { threads: 1 },
        )
        .unwrap();
        for frame in &out.spikes[0] {
            assert!(frame.iter().all(|&v| v < 0.5));
        }
    }

    #[test]
    fn batch_results_ordered() {
        let model = tiny_dense_lif_model();
        let inputs = vec![spike_input(5, 4, 1.0), spike_input(5, 4, 0.0)];
        let out = run(&model, &inputs, &SimConfig { threads: 2 }).unwrap();
        assert!(out.spikes[0][0].iter().all(|&v| v > 0.5));
        assert!(out.spikes[1][0].iter().all(|&v| v < 0.5));
    }

    #[test]
    fn synaptic_ops_counted() {
        let model = tiny_dense_lif_model();
        let input = spike_input(10, 4, 1.0);
        let out = run(
            &model,
            std::slice::from_ref(&input),
            &SimConfig { threads: 1 },
        )
        .unwrap();
        assert_eq!(out.stats.synaptic_ops, 160);
    }

    // ── §29 transcript snapshot ────────────────────────────────────────────────
    //
    // `render_transcript` is the SINGLE source of the §29 format.  Both the Rust
    // CLI binary and the Python `thrindex._cli` entry-point return/print the string
    // it produces — neither adds or removes content.  Pinning the output here
    // ensures neither frontend can drift without this test failing.

    #[test]
    fn transcript_format_snapshot() {
        let model = tiny_dense_lif_model();
        // 5 timesteps of full activation so the transcript has non-trivial stats.
        let input = spike_input(5, 4, 1.0);
        let config = SimConfig { threads: 1 };
        let output = run(&model, std::slice::from_ref(&input), &config).unwrap();
        let raster = crate::raster::SpikeRaster::from_frames(output.spikes[0].clone(), 4);
        let transcript = render_transcript(
            &model,
            &output,
            &raster,
            Some(0),
            // Pin to "test" so the snapshot is not tied to a specific semver string.
            "test",
        );
        // Redact the wall-time line — it is a real timer and varies between runs.
        let redacted: String = transcript
            .lines()
            .map(|l| {
                if l.trim_start().starts_with("sim wall time:") {
                    " sim wall time:       [time]"
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        insta::assert_snapshot!(redacted);
    }
}
