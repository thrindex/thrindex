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

/// A stored capability hash must be the sim descriptor hash.
///
/// Absent means an older file: accepted. Present on any target other than
/// `sim`, or present and different from [`expected_sim_capability_hash`],
/// is rejected. The compiler only lowers `sim`.
pub(crate) fn check_capability_hash(
    target: &str,
    stored: Option<&str>,
) -> Result<(), ArtifactError> {
    let Some(stored) = stored else {
        return Ok(());
    };
    if target != "sim" {
        return Err(ArtifactError::InvalidContent {
            detail: format!(
                "capability_hash is the sim lower descriptor, but this file says target {target:?}"
            ),
        });
    }
    let expected = expected_sim_capability_hash();
    if stored != expected {
        return Err(ArtifactError::InvalidContent {
            detail: format!(
                "capability_hash does not match the sim lower descriptor (file has {stored}, sim descriptor is {expected})"
            ),
        });
    }
    Ok(())
}

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

/// SHA-256 of the lower-pass descriptor that produced a `.thx`.
///
/// The preimage is compact JSON with alphabetically sorted keys. `native_dt_ms`
/// uses one fractional digit when it is a whole number, so `1.0` cannot become
/// `1` under a different printer. This hash identifies the **sim** descriptor.
/// It is not a chip id.
///
/// Returns `None` when `native_dt_ms` is not finite.
pub fn hash_lower_capability(
    target: &str,
    native_dt_ms: f64,
    native_delay_max_steps: u16,
    delay_fallback: &str,
    precision: &str,
) -> Option<String> {
    let dt = format_dt_ms(native_dt_ms)?;
    let preimage = format!(
        "{{\"delay_fallback\":{fallback},\"native_delay_max_steps\":{steps},\"native_dt_ms\":{dt},\"precision\":{precision},\"target\":{target}}}",
        fallback = quote_json(delay_fallback),
        steps = native_delay_max_steps,
        precision = quote_json(precision),
        target = quote_json(target),
    );
    Some(sha256_hex(preimage.as_bytes()))
}

/// Hash of the descriptor `thrindex-compiler` uses for `target = "sim"`.
///
/// `native_dt_ms` is `1.0`, the delay cap is `u16::MAX`, fallback is emulate,
/// precision is float32 (ADR-0007). Parse compares a stored hash to this.
pub fn expected_sim_capability_hash() -> String {
    hash_lower_capability("sim", 1.0, u16::MAX, "emulate", "float32").unwrap_or_else(
        || sha256_hex(br#"{"delay_fallback":"emulate","native_delay_max_steps":65535,"native_dt_ms":1.0,"precision":"float32","target":"sim"}"#),
    )
}

fn format_dt_ms(dt: f64) -> Option<String> {
    if !dt.is_finite() {
        return None;
    }
    if dt.fract() == 0.0 && dt.abs() < 1_000_000.0 {
        return Some(format!("{dt:.1}"));
    }
    serde_json::to_string(&dt).ok()
}

fn quote_json(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", u32::from(c)));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_capability_preimage_is_stable() {
        let hash = expected_sim_capability_hash();
        let again = hash_lower_capability("sim", 1.0, u16::MAX, "emulate", "float32").unwrap();
        assert_eq!(hash, again);
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn non_finite_dt_is_not_hashed() {
        assert!(hash_lower_capability("sim", f64::NAN, 1, "emulate", "float32").is_none());
    }
}
