//! Backend plugin contract for THRINDEX (§16, ADR-0010).
//!
//! ## Architecture
//!
//! This crate is **L1** in the layer law (ARCHITECTURE.md):
//! - Depends on: nothing beyond `std` and `serde`.
//! - Depended on by: `thrindex-sim` (L2, implements [`Backend`] as the reference),
//!   `conformance/` (the harness), and future hardware backend crates under
//!   `crates/thrindex-backends/<target>/`.
//!
//! ## What lives here
//!
//! - [`Capability`]: what a backend declares it can do. The compiler reads this to
//!   decide whether to lower natively, retime (ADR-0008), emulate delays (ADR-0009),
//!   or reject. Core crates never hardcode hardware constants; they read the descriptor.
//! - [`Backend`]: the trait every backend implements. The conformance harness calls it.
//! - [`BackendError`]: typed error returned by a backend run.
//!
//! ## Input / output contract
//!
//! Inputs and outputs are **untyped rasters**: `Vec<Vec<Vec<f32>>>` with shape
//! `[batch, timesteps, features]`. This avoids `thrindex-backend-api` depending on
//! `thrindex-sim`'s typed model structs, keeping the trait at L1. Backends receive the
//! `.thx` artifact JSON and parse it themselves (or delegate to `thrindex-sim`).
pub mod backend;
pub mod capability;
pub mod error;

pub use backend::Backend;
pub use capability::{Capability, DelayFallback, Precision};
pub use error::BackendError;
