//! # thrindex-artifact
//!
//! Parser and validator for `.thx` deployment artifacts.
//!
//! ## Current format: `m2-draft`
//!
//! A single UTF-8 JSON file produced by `serde_json::to_string_pretty`.
//! Weights/biases are standard-base64 of little-endian IEEE-754 f32 arrays.
//! Integrity is a CRC32 over a sorted-key canonical JSON representation of
//! the model block.  Signing is deferred to M5 (`m3-signed` format).
//!
//! ## Format gaps (proposed `m2-platform` bump)
//!
//! The following fields are **missing** from `m2-draft` for Platform use.
//! They are derived by this crate from layer shapes as a stopgap; the
//! compiler must be updated to emit them as `m2-platform`:
//!
//! | Gap | Platform need | WS-1 workaround |
//! |---|---|---|
//! | `input_shape` | Run verification inference | Derived from first layer `in_features` |
//! | `output_shape` | Declare expected output | Derived from last Dense `out_features` |
//! | `resource_report.weight_count` | Synapse budget check | Derived from weight tensor shapes |
//! | `resource_report.lif_neuron_count` | Neuron budget check | Derived from preceding Dense `out_features` |
//! | `capability_hash` | Verify correct chip descriptor used | Not available; check deferred |
//! | `source_hash` | Full provenance chain | Not available; `thrindex_version` only |
//! | Signing fields | Authenticity + non-repudiation | Not available; deferred to M5 |
//!
//! ## Usage
//!
//! ```rust,no_run
//! use thrindex_artifact::parse_bytes;
//!
//! let bytes = std::fs::read("model.thx").unwrap();
//! let artifact = parse_bytes(&bytes).unwrap();
//!
//! println!("target: {}", artifact.target());
//! println!("content_hash: {}", artifact.content_hash());
//!
//! let rs = artifact.resource_summary();
//! println!("{} layers, {} total weights", rs.layer_count, rs.total_weight_count);
//! ```

mod artifact;
mod error;
mod integrity;
mod layers;
mod resource;
mod wire;

pub use artifact::{Artifact, SUPPORTED_VERSIONS};
pub use error::ArtifactError;
pub use layers::{Conv2dLayer, DenseLayer, Layer, LifLayer};
pub use resource::ResourceSummary;

/// Parse a `.thx` artifact from a UTF-8 string.
///
/// Validates format version, then verifies the CRC32 integrity check before
/// returning the artifact.
///
/// # Errors
///
/// - [`ArtifactError::JsonParse`] — not valid JSON or wrong top-level shape
/// - [`ArtifactError::UnsupportedFormatVersion`] — `format_version` not in [`SUPPORTED_VERSIONS`]
/// - [`ArtifactError::IntegrityCheckFailed`] — CRC32 mismatch
pub fn parse(content: &str) -> Result<Artifact, ArtifactError> {
    parse_bytes(content.as_bytes())
}

/// Parse a `.thx` artifact from raw bytes.
///
/// Identical to [`parse`] but accepts `&[u8]`; the SHA-256 content hash is
/// computed over these bytes before UTF-8 decoding.
///
/// # Errors
///
/// Same as [`parse`], plus [`ArtifactError::JsonParse`] if the bytes are
/// not valid UTF-8.
pub fn parse_bytes(bytes: &[u8]) -> Result<Artifact, ArtifactError> {
    let content_hash = integrity::sha256_hex(bytes);

    let content = std::str::from_utf8(bytes).map_err(|e| ArtifactError::JsonParse {
        detail: format!("file is not valid UTF-8: {e}"),
    })?;

    let wire: wire::WireArtifact =
        serde_json::from_str(content).map_err(|e| ArtifactError::JsonParse {
            detail: e.to_string(),
        })?;

    if !SUPPORTED_VERSIONS.contains(&wire.format_version.as_str()) {
        return Err(ArtifactError::UnsupportedFormatVersion {
            version: wire.format_version,
            supported: "m2-draft",
        });
    }

    let artifact = Artifact::from_wire(wire, content_hash);

    integrity::verify_crc32(
        artifact.raw_metadata().model_canonical.as_str(),
        artifact.raw_metadata().crc32.as_str(),
    )?;

    Ok(artifact)
}
