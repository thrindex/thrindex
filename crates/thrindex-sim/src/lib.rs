//! `thrindex-sim` — behavioral LIF simulator.
//!
//! Executes spiking neural network models compiled to the `.thx` artifact format
//! (ADR-0006) with f32 precision (ADR-0007).
//!
//! **Self-determinism guarantee**: same `.thx` + same pre-generated input +
//! any thread count 1..N → **byte-identical** spike raster, across all runs and
//! all CI platforms.
//!
//! The simulator contains **zero RNG**.  All input spike trains are provided by
//! the caller; the encoder stage (which holds the PRNG) runs before this function.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use thrindex_sim::{model, sim, raster, SimConfig};
//!
//! let resolved = model::load("model.thx")?;
//! let input: Vec<Vec<Vec<f32>>> = vec![/* [batch, T, features] */];
//! let output = sim::run(&resolved, &input, &SimConfig::default())?;
//! # Ok::<(), thrindex_sim::SimError>(())
//! ```

#![deny(unsafe_code)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod lif;
pub mod model;
pub mod raster;
pub mod sim;

pub use error::SimError;
pub use sim::{SimConfig, SimOutput, SimStats, render_transcript};
