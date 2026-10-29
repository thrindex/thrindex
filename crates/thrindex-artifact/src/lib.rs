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
//! ## What a new compile records
//!
//! `format_version` stays `m2-draft`. New files add two optional fields.
//! Files that omit them still parse. The simulator ignores both.
//!
//! | Field | Where | Meaning |
//! |---|---|---|
//! | `model.resource_report` | inside the CRC'd model block | Geometry the compiler computed from the layers: shapes, weight count, LIF count. On parse it must match a fresh derivation from `layers`. A report that disagrees with the layers is rejected. |
//! | `metadata.capability_hash` | metadata | SHA-256 of the **sim** lower descriptor (`dt`, delay cap, float32). This is not a chip id. `thx.compile` only lowers for `sim`. |
//!
//! `source_hash` is intentionally absent. `thx.compile` receives an in-memory
//! model, not a source file. Hashing a file we do not have would be a false
//! provenance claim. `thrindex_version` identifies the toolchain.
//!
//! Signing fields are still absent. The Console signs the uploaded bytes.
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
pub use integrity::{expected_sim_capability_hash, hash_lower_capability};
pub use layers::{Conv2dLayer, DenseLayer, FlattenLayer, Layer, LifLayer};
pub use resource::{ResourceSummary, summarize_layers};

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

    integrity::verify_crc32(&wire.metadata.model_canonical, &wire.metadata.crc32)?;

    if let Some(report) = wire.model.resource_report.as_ref() {
        resource::check_report(report, &wire.model.layers)?;
    }
    integrity::check_capability_hash(&wire.target, wire.metadata.capability_hash.as_deref())?;

    Ok(Artifact::from_wire(wire, content_hash))
}
