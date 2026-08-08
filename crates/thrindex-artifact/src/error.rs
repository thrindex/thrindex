use thiserror::Error;

/// All errors that can occur when parsing or validating a `.thx` artifact.
#[derive(Debug, Error)]
pub enum ArtifactError {
    /// The file is not valid JSON or the top-level structure is wrong.
    #[error("JSON parse error: {detail}")]
    JsonParse { detail: String },

    /// The `format_version` string is not in `SUPPORTED_VERSIONS`.
    ///
    /// The caller should surface the version to the user — an artifact from a
    /// newer compiler may require an updated version of this crate.
    #[error("unsupported format version {version:?}; this crate supports: {supported:?}")]
    UnsupportedFormatVersion {
        version: String,
        supported: &'static str,
    },

    /// The CRC32 stored in `metadata.crc32` does not match the re-computed
    /// hash of `metadata.model_canonical`.  The artifact is corrupt.
    #[error("CRC32 integrity check failed: artifact claims {expected}, computed {computed}")]
    IntegrityCheckFailed { expected: String, computed: String },

    /// A structural constraint was violated (e.g. empty layer list, unknown
    /// reset mode string).
    #[error("invalid artifact content: {detail}")]
    InvalidContent { detail: String },
}
