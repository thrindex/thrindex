//! `thrindex-compiler` — capture → validate → lower pipeline.
//!
//! Takes a Graph IR JSON string produced by the Python SDK and emits a sealed
//! `.thx` artifact JSON string for a named target (currently only `"sim"`).
//!
//! ## Compiler architecture
//!
//! ```text
//! Graph IR JSON (from Python SDK)
//!         │
//!         ▼
//!    [capture]  — JSON → GraphModel (E0108 on parse failure)
//!         │
//!         ▼
//!    [validate] — semantic checks (E0105, E0107, E0101, E0106, E0109)
//!         │
//!         ▼
//!    [lower]    — resolve alpha, encode delays → .thx JSON
//!                 (E0101/E0102 at retiming time; advisory when dt differs)
//! ```
//!
//! ## Design authority
//!
//! - ADR-0008 (Graph IR time semantics / two-level parameter rule / retiming advisory)
//! - ADR-0009 (Synaptic delays: first-class, per-connection, capability-negotiated)

pub mod capture;
pub mod error;
pub mod lower;
pub mod validate;

pub use error::CompileError;
pub use lower::{CompileReport, SimCapability};

/// Run the full compilation pipeline: Graph IR JSON → `.thx` artifact JSON.
///
/// Returns [`CompileReport`] containing the artifact JSON and an optional retiming
/// advisory.  The advisory **MUST** be surfaced to the user (Python SDK prints it
/// to `stderr`; CLI prints it as a `WARN` line).
///
/// # Errors
///
/// Returns the first [`CompileError`] encountered; errors are in §30 four-part
/// format and are snapshot-tested in `crates/thrindex-compiler/tests/`.
pub fn compile(ir_json: &str, target: &str) -> Result<CompileReport, CompileError> {
    let model = capture::capture(ir_json)?;
    validate::validate(&model)?;
    lower::lower(&model, target)
}
