//! Capture pass: Graph IR JSON → [`GraphModel`].
//!
//! This pass is intentionally thin: its only job is serde deserialization.
//! Semantic validation (dimension checks, `tau_mem > dt`, reset-mode parsing)
//! lives in the validate pass to ensure every structural error is surfaced in
//! the §30 four-part format rather than as a raw serde message.

use thrindex_ir::GraphModel;

use crate::error::CompileError;

/// Deserialize a Graph IR JSON string into a [`GraphModel`].
///
/// Produces `E0108` if the JSON is malformed or missing required fields.
///
/// # Errors
///
/// Returns [`CompileError::IrJsonParseError`] for any serde deserialization failure.
pub fn capture(ir_json: &str) -> Result<GraphModel, CompileError> {
    serde_json::from_str(ir_json).map_err(|e| CompileError::IrJsonParseError {
        detail: e.to_string(),
    })
}
