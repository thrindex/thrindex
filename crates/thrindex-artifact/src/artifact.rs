//! The public [`Artifact`] type — parsed, validated, and enriched.

use crate::{
    error::ArtifactError,
    integrity,
    layers::{self, Layer},
    resource::{self, ResourceSummary},
    wire::{WireArtifact, WireMetadata},
};

/// Format version strings this crate can parse.
pub const SUPPORTED_VERSIONS: &[&str] = &["m2-draft"];

/// A parsed and integrity-verified `.thx` artifact.
///
/// Construct via [`crate::parse`] or [`crate::parse_bytes`].
///
/// ## What is NOT available from `m2-draft`
///
/// The following fields are required by the Platform but absent from the
/// current format.  They are proposed as additions in `m2-platform`
/// (see module documentation in `lib.rs`):
///
/// - `input_shape` / `output_shape` — derived by this crate from layer shapes
/// - `resource_report` — derived (imprecise; missing chip scheduling data)
/// - `capability_hash` — which chip capability descriptor was used; not stored
/// - `source_hash` — Python source identity; not stored
/// - signing fields — deferred to M5 / `m3-signed` format
#[derive(Debug, Clone)]
pub struct Artifact {
    wire: WireArtifact,
    /// SHA-256 hex of the raw UTF-8 bytes passed to the parser.
    content_hash: String,
}

impl Artifact {
    // ── Constructors (crate-private) ─────────────────────────────────────────

    pub(crate) fn from_wire(wire: WireArtifact, content_hash: String) -> Self {
        Self { wire, content_hash }
    }

    // ── Format identity ───────────────────────────────────────────────────────

    /// The `format_version` string from the artifact, e.g. `"m2-draft"`.
    pub fn format_version(&self) -> &str {
        &self.wire.format_version
    }

    /// The `thrindex_version` string recorded at compile time.
    pub fn thrindex_version(&self) -> &str {
        &self.wire.thrindex_version
    }

    /// The compilation target, e.g. `"sim"`.
    pub fn target(&self) -> &str {
        &self.wire.target
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// ISO-8601 compile timestamp.
    pub fn compiled_at(&self) -> &str {
        &self.wire.metadata.compiled_at
    }

    /// Timestep in milliseconds, if recorded by the compiler (`dt_ms` field).
    ///
    /// Present in artifacts produced by the M3 compiler and all
    /// keyword-spotting / event-camera templates.  Absent from the older
    /// `m2_dense_lif.thx` fixture.
    pub fn dt_ms(&self) -> Option<f64> {
        self.wire.metadata.dt_ms
    }

    /// SHA-256 hex (lowercase, 64 chars) of the raw artifact bytes.
    ///
    /// This is the content fingerprint the Platform stores in `artifacts.content_hash`.
    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }

    // ── Layer access ──────────────────────────────────────────────────────────

    /// Number of layers (cheap — reads from the raw `Vec<Value>` length).
    pub fn layer_count(&self) -> usize {
        self.wire.model.layers.len()
    }

    /// Parse all layers into typed structs.
    ///
    /// Parsed on demand (not cached) — call once and store the result if you
    /// need it repeatedly.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError::InvalidContent`] if any layer has an unknown
    /// `"type"` or a malformed structure.
    pub fn layers(&self) -> Result<Vec<Layer>, ArtifactError> {
        self.wire
            .model
            .layers
            .iter()
            .enumerate()
            .map(|(i, v)| {
                layers::parse_layer(v).map_err(|e| ArtifactError::InvalidContent {
                    detail: format!("layer {i}: {e}"),
                })
            })
            .collect()
    }

    // ── Integrity ─────────────────────────────────────────────────────────────

    /// Verify the CRC32 stored in `metadata.crc32` against `metadata.model_canonical`.
    ///
    /// This is the same check performed by `thrindex-sim` when loading a model.
    /// [`crate::parse`] and [`crate::parse_bytes`] call this automatically.
    pub fn verify_crc32(&self) -> Result<(), ArtifactError> {
        integrity::verify_crc32(
            &self.wire.metadata.model_canonical,
            &self.wire.metadata.crc32,
        )
    }

    // ── Re-serialisation ──────────────────────────────────────────────────────

    /// Re-serialise to pretty-printed JSON.
    ///
    /// For artifacts produced by the current compiler, the output is
    /// **byte-identical** to the original file.  See `wire.rs` for the
    /// design choices that make this possible.
    ///
    /// # Panics
    ///
    /// Only if serde_json fails to serialise a well-formed struct, which
    /// should never happen for a type constructed from valid JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.wire).expect("WireArtifact is always serialisable")
    }

    // ── Platform-derived data ─────────────────────────────────────────────────

    /// Derive resource figures from the layer geometry.
    ///
    /// These are **estimates** for WS-1 verification.  The `m2-platform`
    /// format bump will add a compiler-declared `resource_report` that
    /// supersedes this.
    pub fn resource_summary(&self) -> ResourceSummary {
        resource::derive(&self.wire.model.layers)
    }

    // ── Crate-internal ────────────────────────────────────────────────────────

    pub(crate) fn raw_metadata(&self) -> &WireMetadata {
        &self.wire.metadata
    }
}
