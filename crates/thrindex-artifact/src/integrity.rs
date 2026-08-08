//! CRC32 integrity verification and SHA-256 content hashing.
//!
//! The CRC32 algorithm mirrors `thrindex-sim`'s `verify_crc32()` exactly:
//! hash the UTF-8 bytes of the stored `metadata.model_canonical` string and
//! compare the 8-hex-digit result against `metadata.crc32`.
//!
//! The SHA-256 content hash is a Platform addition: it is computed over the
//! raw input bytes (the `.thx` file as read from disk) and stored on the
//! `Artifact` type so the Platform can record a stable content fingerprint in
//! the `artifacts` table without re-serialising.

use sha2::{Digest, Sha256};

use crate::error::ArtifactError;

/// Verify the stored `crc32` against `model_canonical`.
///
/// Matches the algorithm in `thrindex-sim::model::verify_crc32`.
pub(crate) fn verify_crc32(model_canonical: &str, expected_hex: &str) -> Result<(), ArtifactError> {
    let computed = crc32fast::hash(model_canonical.as_bytes());
    let computed_hex = format!("{computed:08x}");
    if computed_hex != expected_hex.to_lowercase() {
        return Err(ArtifactError::IntegrityCheckFailed {
            expected: expected_hex.to_string(),
            computed: computed_hex,
        });
    }
    Ok(())
}

/// Compute the SHA-256 of `bytes` and return a lowercase 64-char hex string.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
