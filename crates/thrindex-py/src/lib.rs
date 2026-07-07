//! `thrindex._core` — the compiled Rust extension module.
//!
//! This crate is intentionally thin: **zero SNN logic lives here**.  All
//! computation belongs in the core Rust crates (`thrindex-numerics`, future
//! `thrindex-sim`, etc.) that are consumed by this bridge via re-exports.
//!
//! In M1 the only surface exposed to Python is the Rust-side package version,
//! which lets the Python package version be derived from the single source of
//! truth in `Cargo.toml`.

#![deny(unsafe_code)]
#![warn(clippy::pedantic)]

use pyo3::prelude::*;

/// The `thrindex._core` Python extension module.
///
/// Exposed attributes:
/// - `__version__`: the crate version string, identical to `pyproject.toml`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
