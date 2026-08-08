//! Wire format types — exactly match the on-disk JSON layout of a `.thx` file.
//!
//! ## Byte-identical round-trip contract
//!
//! These types are designed so that `serde_json::to_string_pretty(artifact)`
//! produces **byte-identical output** to the original file.  Two properties
//! make this work:
//!
//! ### 1. Layer storage as raw `Value`
//!
//! The compiler builds layer objects with `serde_json::json!({...})`.  Without
//! the `preserve_order` feature, `serde_json::Map` is a `BTreeMap`, so all
//! layer keys are sorted **alphabetically** (e.g. `bias_b64` before `type`).
//!
//! Storing layers as `Vec<serde_json::Value>` keeps the original key order
//! intact on both parse and re-serialise.  Using a typed enum with
//! `#[serde(tag = "type")]` would emit keys in struct-declaration order
//! (`type` first) — **not** matching the file.
//!
//! ### 2. Exact float strings via `arbitrary_precision`
//!
//! With `serde_json = { features = ["arbitrary_precision"] }`, every JSON
//! number is stored as its original decimal string internally.  Re-serialising
//! emits the identical string, even for fixture files written with older
//! serde_json versions that used non-ryu float formatting.
//!
//! ## Option serialisation rules (struct-level fields only)
//!
//! | Field | When `None` | Rule |
//! |---|---|---|
//! | `dt_ms` | → absent | `skip_serializing_if` (absent in pre-M3 artifacts) |
//!
//! Layer-level option handling is managed per-Value in the raw JSON.

use serde::{Deserialize, Serialize};

// ── Top-level artifact ────────────────────────────────────────────────────────

/// Top-level `.thx` artifact (ADR-0006, `m2-draft` format).
///
/// Field declaration order matches `ThxArtifact` in `lower.rs`, which is
/// what `#[derive(Serialize)]` uses — declaration order, not alphabetical.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireArtifact {
    pub format_version: String,
    pub thrindex_version: String,
    pub target: String,
    pub model: WireModel,
    pub metadata: WireMetadata,
}

/// The `model` block.
///
/// Layers are stored as raw `serde_json::Value` objects so that the original
/// key order (BTreeMap alphabetical, from `json!{}` without `preserve_order`)
/// is preserved on re-serialisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireModel {
    pub layers: Vec<serde_json::Value>,
}

/// The `metadata` block.
///
/// Field declaration order matches `ThxMetadata` in `lower.rs`.
/// `dt_ms` is absent from pre-M3 artifacts (`m2_dense_lif.thx` fixture).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMetadata {
    pub compiled_at: String,
    /// Compact, sorted-key JSON of the `model` block; the CRC32 input.
    pub model_canonical: String,
    pub crc32: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dt_ms: Option<f64>,
}
