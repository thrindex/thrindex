//! `E####` error codes for `thrindex-sim`.
//!
//! Codes are **stable** — never reused, never renumbered.
//! Every variant carries the §30 four-part body in its `Display` implementation:
//! What happened / Why / How to fix / Docs link.

use thiserror::Error;

/// All simulation errors.  Each variant maps 1:1 to a stable `E####` code.
#[derive(Debug, Error)]
pub enum SimError {
    /// E0001 — `.thx` file could not be read from disk.
    #[error(
        "E0001: artifact file not found: {path}\n\
         Why: the path does not exist or is not readable.\n\
         Fix: check the path with `ls {path}` and verify file permissions.\n\
         Docs: https://docs.thrindex.com/errors/E0001"
    )]
    ArtifactNotFound { path: String },

    /// E0002 — unknown `.thx` format version.
    #[error(
        "E0002: unsupported artifact format version: \"{version}\"\n\
         Why: this build of thrindex-sim understands only \"{supported}\".\n\
         Fix: rebuild the artifact with `thx.compile()` from thrindex >= {min_thrindex}.\n\
         Docs: https://docs.thrindex.com/errors/E0002"
    )]
    UnsupportedFormatVersion {
        version: String,
        supported: String,
        min_thrindex: String,
    },

    /// E0003 — unknown layer type in the model definition.
    #[error(
        "E0003: unknown layer type: \"{layer_type}\"\n\
         Why: the artifact was compiled with a newer thrindex that supports layer types \
         this simulator does not know.\n\
         Fix: upgrade thrindex-sim, or recompile the model with a compatible version.\n\
         Docs: https://docs.thrindex.com/errors/E0003"
    )]
    UnknownLayerType { layer_type: String },

    /// E0004 — dimension mismatch between consecutive layers.
    #[error(
        "E0004: layer dimension mismatch at layer {layer_idx}\n\
         Why: layer {layer_idx} expects {expected} input features but receives {got} \
         from layer {prev_idx}.\n\
         Fix: recompile the model — the artifact may be corrupt.\n\
         Docs: https://docs.thrindex.com/errors/E0004"
    )]
    DimensionMismatch {
        layer_idx: usize,
        prev_idx: usize,
        expected: usize,
        got: usize,
    },

    /// E0005 — LIF parameter out of valid range.
    #[error(
        "E0005: invalid LIF parameter in layer {layer_idx}: {message}\n\
         Why: {reason}\n\
         Fix: {fix}\n\
         Docs: https://docs.thrindex.com/errors/E0005"
    )]
    InvalidLifParam {
        layer_idx: usize,
        message: String,
        reason: String,
        fix: String,
    },

    /// E0006 — base64 weight decoding failed.
    #[error(
        "E0006: base64 decode error in layer {layer_idx} ({field}): {detail}\n\
         Why: the artifact's weight data is corrupt or was truncated.\n\
         Fix: recompile the model from scratch with `thx.compile()`.\n\
         Docs: https://docs.thrindex.com/errors/E0006"
    )]
    Base64DecodeError {
        layer_idx: usize,
        field: String,
        detail: String,
    },

    /// E0007 — weight array length does not match declared dimensions.
    #[error(
        "E0007: weight shape mismatch in layer {layer_idx}: declared {declared} f32 values \
         but decoded {decoded} bytes ({got_f32} f32 values).\n\
         Why: the artifact's weight data is corrupt.\n\
         Fix: recompile the model from scratch.\n\
         Docs: https://docs.thrindex.com/errors/E0007"
    )]
    WeightShapeMismatch {
        layer_idx: usize,
        declared: usize,
        decoded: usize,
        got_f32: usize,
    },

    /// E0008 — JSON deserialization error.
    #[error(
        "E0008: artifact JSON parse error: {detail}\n\
         Why: the file is not valid thrindex artifact JSON.\n\
         Fix: verify the file was produced by `thx.compile()` and was not corrupted in transfer.\n\
         Docs: https://docs.thrindex.com/errors/E0008"
    )]
    JsonParseError { detail: String },

    /// E0009 — CRC32 integrity check failed.
    #[error(
        "E0009: artifact integrity check failed (CRC32 mismatch)\n\
         Why: the artifact file was corrupted after it was compiled.\n\
         Fix: recompile or re-download the artifact.\n\
         Docs: https://docs.thrindex.com/errors/E0009"
    )]
    IntegrityCheckFailed,

    /// E0010 — input spike train dimensions do not match the model's input layer.
    #[error(
        "E0010: input dimension mismatch: model expects {expected} input neurons, \
         got {got}.\n\
         Why: the input spike train does not match the model's first layer.\n\
         Fix: verify your encoder output shape matches the model architecture.\n\
         Docs: https://docs.thrindex.com/errors/E0010"
    )]
    InputDimensionMismatch { expected: usize, got: usize },
}
