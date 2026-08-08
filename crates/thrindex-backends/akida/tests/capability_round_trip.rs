//! Integration test: AKD1500 capability descriptor field values + serde round-trip.
//!
//! Proves the public API: akd1500_capability() returns the correct constant values,
//! and the Capability struct round-trips cleanly through serde_json.
use thrindex_backend_akida::{AKD1500_TARGET_NAME, AKD1500_WEIGHT_MAX, akd1500_capability};
use thrindex_backend_api::{DelayFallback, Precision};

/// Assert every field of the AKD1500 capability descriptor.
///
/// These values are pinned by ADR-0011 §3. Any change requires an ADR amendment.
#[test]
fn capability_field_values() {
    let cap = akd1500_capability();

    assert_eq!(
        cap.name, "akida-akd1500",
        "name must be the canonical target string"
    );

    assert_eq!(
        cap.native_dt_ms, 1.0,
        "native_dt_ms must be 1.0 (notional — LIF rejection fires before dt negotiation)"
    );

    assert_eq!(
        cap.native_delay_max_steps, 0,
        "native_delay_max_steps must be 0 — no TNP on Akida 1.0"
    );

    assert_eq!(
        cap.delay_fallback,
        DelayFallback::Reject,
        "delay_fallback must be Reject — no emulation path without TNP"
    );

    assert!(
        matches!(cap.precision, Precision::Int4PerTensor),
        "precision must be Int4PerTensor — 4-bit weights/activations (confirmed spike test)"
    );
}

/// Assert the public constants that downstream code is expected to use.
#[test]
fn public_constants() {
    assert_eq!(AKD1500_TARGET_NAME, "akida-akd1500");
    assert_eq!(
        AKD1500_WEIGHT_MAX, 7,
        "4-bit signed range [-7, 7]; scale = max|W| / 7"
    );
}

/// Capability round-trip: serialize to JSON and deserialize back — all fields survive intact.
///
/// This guards against accidentally making Capability non-serializable and ensures
/// the capability descriptor can be stored/transmitted by the compiler pipeline.
#[test]
fn capability_round_trip() {
    let original = akd1500_capability();
    let json = serde_json::to_string(&original).expect("Capability must be serializable");

    let roundtripped: thrindex_backend_api::Capability =
        serde_json::from_str(&json).expect("Capability must be deserializable");

    assert_eq!(roundtripped.name, original.name);
    assert_eq!(roundtripped.native_dt_ms, original.native_dt_ms);
    assert_eq!(
        roundtripped.native_delay_max_steps,
        original.native_delay_max_steps
    );
    assert_eq!(roundtripped.delay_fallback, original.delay_fallback);

    // Precision derives PartialEq is not guaranteed but its Debug does — check via JSON key.
    assert!(
        json.contains("\"Int4PerTensor\""),
        "serialized precision must be 'Int4PerTensor'; got: {json}"
    );
}
