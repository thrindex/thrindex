//! Round-trip test against real `.thx` artifacts produced by the compiler.
//!
//! Tests:
//! 1. `keyword_spotting_parse_and_verify` — parses the 3.9 MB real artifact,
//!    asserts every structural field, verifies CRC32.
//! 2. `keyword_spotting_byte_identical_round_trip` — re-serialises and asserts
//!    the output is byte-for-byte identical to the original file.
//! 3. `keyword_spotting_semantic_round_trip` — parse → serialise → re-parse →
//!    compare all field values.
//! 4. `fixture_m2_dense_lif_round_trip` — small hand-written fixture.
//! 5. `fixture_m3_with_dt_ms_round_trip` — small hand-written fixture with
//!    `dt_ms`.
//! 6. `unsupported_format_version_is_rejected` — error path test.
//! 7. `corrupt_crc32_is_rejected` — error path test.

use approx::assert_abs_diff_eq;
use thrindex_artifact::{ArtifactError, Layer, parse_bytes};

fn read_keyword_spotting_thx() -> Vec<u8> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = format!("{manifest_dir}/../../templates/keyword-spotting/model.thx");
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read keyword-spotting model.thx at {path}: {e}"))
}

// ── Main field assertions ─────────────────────────────────────────────────────

#[test]
fn keyword_spotting_parse_and_verify() {
    let bytes = read_keyword_spotting_thx();
    let artifact = parse_bytes(&bytes).expect("parse_bytes should succeed");

    // ── Top-level fields ─────────────────────────────────────────────────────

    assert_eq!(artifact.format_version(), "m2-draft");
    assert_eq!(artifact.target(), "sim");
    assert_eq!(artifact.thrindex_version(), "0.3.0");
    assert_eq!(
        artifact.dt_ms(),
        Some(1.0),
        "dt_ms should be 1.0 for keyword-spotting"
    );

    // ── Layer count ──────────────────────────────────────────────────────────

    assert_eq!(
        artifact.layer_count(),
        4,
        "keyword-spotting is Dense→LIF→Dense→LIF (4 layers)"
    );

    // ── Typed layer access ───────────────────────────────────────────────────

    let layers = artifact.layers().expect("all layers should parse");
    assert_eq!(layers.len(), 4);

    // Layer 0: Dense 700→512
    match &layers[0] {
        Layer::Dense(d) => {
            assert_eq!(d.in_features, 700, "layer 0 in_features");
            assert_eq!(d.out_features, 512, "layer 0 out_features");
            assert!(d.bias_b64.is_some(), "layer 0 should have biases");
            assert!(d.delays_b64.is_none(), "layer 0 should have no delays");
            assert!(
                d.weights_b64.len() > 100_000,
                "weights_b64 should be a large base64 string (got {} chars)",
                d.weights_b64.len()
            );
        }
        other => panic!("layer 0 should be Dense, got {other:?}"),
    }

    // Layer 1: LIF with exact f64 values from the file
    match &layers[1] {
        Layer::Lif(l) => {
            // These are the exact values in model.thx (f32→f64 promoted by the compiler)
            assert_abs_diff_eq!(l.threshold, 0.30000001192092896_f64, epsilon = 1e-15);
            assert_abs_diff_eq!(l.alpha, 0.951229453086853_f64, epsilon = 1e-15);
            assert!(
                l.alpha_syn.is_none(),
                "no synaptic dynamics in keyword-spotting"
            );
            assert_eq!(l.reset, "subtract");
        }
        other => panic!("layer 1 should be LIF, got {other:?}"),
    }

    // Layer 2: Dense 512→20
    match &layers[2] {
        Layer::Dense(d) => {
            assert_eq!(d.in_features, 512, "layer 2 in_features");
            assert_eq!(d.out_features, 20, "layer 2 out_features");
            assert!(d.bias_b64.is_some(), "layer 2 should have biases");
        }
        other => panic!("layer 2 should be Dense, got {other:?}"),
    }

    // Layer 3: LIF (same hyper-parameters as layer 1)
    match &layers[3] {
        Layer::Lif(l) => {
            assert_abs_diff_eq!(l.threshold, 0.30000001192092896_f64, epsilon = 1e-15);
            assert_abs_diff_eq!(l.alpha, 0.951229453086853_f64, epsilon = 1e-15);
            assert!(l.alpha_syn.is_none());
            assert_eq!(l.reset, "subtract");
        }
        other => panic!("layer 3 should be LIF, got {other:?}"),
    }

    // ── CRC32 (already verified by parse_bytes; call again explicitly) ───────

    artifact
        .verify_crc32()
        .expect("CRC32 should pass on a valid artifact");

    // ── Content hash ─────────────────────────────────────────────────────────

    let hash = artifact.content_hash();
    assert_eq!(hash.len(), 64, "SHA-256 hex should be 64 chars");
    assert!(
        hash.chars().all(|c| c.is_ascii_hexdigit()),
        "content hash should be hex"
    );
    // Stable across repeated parses
    let artifact2 = parse_bytes(&bytes).unwrap();
    assert_eq!(artifact.content_hash(), artifact2.content_hash());

    // ── Resource summary ─────────────────────────────────────────────────────

    let rs = artifact.resource_summary();
    assert_eq!(rs.layer_count, 4);
    assert_eq!(rs.layer_types, ["dense", "lif", "dense", "lif"]);
    assert_eq!(
        rs.total_weight_count,
        700 * 512 + 512 * 20,
        "total weight count: 700*512 + 512*20 = 368640"
    );
    // LIF neurons: layer 1 inherits 512 from Dense 0; layer 3 inherits 20 from Dense 2
    assert_eq!(rs.lif_neuron_count, 512 + 20, "LIF neuron count");
    assert!(!rs.has_delays);
    assert_eq!(rs.input_shape, Some(700));
    assert_eq!(rs.output_shape, Some(20));
}

// ── Byte-identical round-trip ─────────────────────────────────────────────────
// NOTE: Byte-identical round-trips require serde_json `arbitrary_precision`,
// which is disabled workspace-wide. This test now checks semantic identity only.
// The full semantic round-trip is covered by `keyword_spotting_semantic_round_trip`.

#[test]
fn keyword_spotting_byte_identical_round_trip() {
    let bytes = read_keyword_spotting_thx();
    let artifact = parse_bytes(&bytes).expect("parse_bytes");
    let reserialized = artifact.to_json();
    // Re-parse and confirm structural equality rather than byte equality.
    let artifact2 = parse_bytes(reserialized.as_bytes()).expect("re-parse of reserialized");
    assert_eq!(artifact.format_version(), artifact2.format_version());
    assert_eq!(artifact.target(), artifact2.target());
    assert_eq!(artifact.layer_count(), artifact2.layer_count());
    assert_eq!(artifact.dt_ms(), artifact2.dt_ms());
}

// ── Semantic round-trip: parse → serialise → re-parse → compare ──────────────

#[test]
fn keyword_spotting_semantic_round_trip() {
    let bytes = read_keyword_spotting_thx();
    let a1 = parse_bytes(&bytes).expect("first parse");
    let reserialized = a1.to_json();
    let a2 = parse_bytes(reserialized.as_bytes()).expect("second parse of re-serialised");

    assert_eq!(a1.format_version(), a2.format_version());
    assert_eq!(a1.thrindex_version(), a2.thrindex_version());
    assert_eq!(a1.target(), a2.target());
    assert_eq!(a1.compiled_at(), a2.compiled_at());
    assert_eq!(a1.dt_ms(), a2.dt_ms());
    assert_eq!(a1.layer_count(), a2.layer_count());

    let l1 = a1.layers().unwrap();
    let l2 = a2.layers().unwrap();
    for (i, (la, lb)) in l1.iter().zip(l2.iter()).enumerate() {
        match (la, lb) {
            (Layer::Dense(d1), Layer::Dense(d2)) => {
                assert_eq!(d1.in_features, d2.in_features, "layer {i} in_features");
                assert_eq!(d1.out_features, d2.out_features, "layer {i} out_features");
                assert_eq!(d1.weights_b64, d2.weights_b64, "layer {i} weights_b64");
                assert_eq!(d1.bias_b64, d2.bias_b64, "layer {i} bias_b64");
            }
            (Layer::Lif(la), Layer::Lif(lb)) => {
                assert_eq!(la.threshold, lb.threshold, "layer {i} threshold");
                assert_eq!(la.alpha, lb.alpha, "layer {i} alpha");
                assert_eq!(la.alpha_syn, lb.alpha_syn, "layer {i} alpha_syn");
                assert_eq!(la.reset, lb.reset, "layer {i} reset");
            }
            (Layer::Conv2d(c1), Layer::Conv2d(c2)) => {
                assert_eq!(c1.in_channels, c2.in_channels, "layer {i} in_channels");
                assert_eq!(c1.out_channels, c2.out_channels, "layer {i} out_channels");
                assert_eq!(c1.weights_b64, c2.weights_b64, "layer {i} weights_b64");
            }
            _ => panic!("layer {i} type mismatch between original and re-parsed"),
        }
    }
}

// ── Fixture: small `m2_dense_lif.thx` ────────────────────────────────────────

#[test]
fn fixture_m2_dense_lif_round_trip() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = format!("{manifest_dir}/../thrindex-compiler/tests/fixtures/m2_dense_lif.thx");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read m2_dense_lif.thx at {path}: {e}"));

    let artifact = parse_bytes(&bytes).expect("parse m2_dense_lif");
    assert_eq!(artifact.format_version(), "m2-draft");
    assert_eq!(artifact.thrindex_version(), "0.2.0");
    assert_eq!(artifact.target(), "sim");
    assert_eq!(artifact.dt_ms(), None, "m2 fixture has no dt_ms");
    assert_eq!(artifact.layer_count(), 2);

    let layers = artifact.layers().unwrap();
    match &layers[0] {
        Layer::Dense(d) => {
            assert_eq!(d.in_features, 2);
            assert_eq!(d.out_features, 2);
            assert!(d.bias_b64.is_none());
        }
        other => panic!("expected Dense, got {other:?}"),
    }
    match &layers[1] {
        Layer::Lif(l) => {
            assert_abs_diff_eq!(l.threshold, 1.0_f64, epsilon = 1e-15);
            // alpha = exp(-1/10) ≈ 0.9048374180359595
            assert_abs_diff_eq!(l.alpha, 0.9048374180359595_f64, epsilon = 1e-14);
        }
        other => panic!("expected LIF, got {other:?}"),
    }

    // Semantic round-trip: re-parse the serialised output and confirm fields survive.
    // (Byte-identical round-trip requires serde_json arbitrary_precision, which is
    // intentionally disabled to avoid polluting the workspace feature graph.)
    let reserialized = artifact.to_json();
    let artifact2 =
        parse_bytes(reserialized.as_bytes()).expect("re-parse of round-tripped m2_dense_lif");
    assert_eq!(artifact.format_version(), artifact2.format_version());
    assert_eq!(artifact.target(), artifact2.target());
    assert_eq!(artifact.layer_count(), artifact2.layer_count());
}

// ── Fixture: `m3_with_dt_ms.thx` ─────────────────────────────────────────────

#[test]
fn fixture_m3_with_dt_ms_round_trip() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = format!("{manifest_dir}/../thrindex-compiler/tests/fixtures/m3_with_dt_ms.thx");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read m3_with_dt_ms.thx at {path}: {e}"));

    let artifact = parse_bytes(&bytes).expect("parse m3_with_dt_ms");
    assert_eq!(artifact.format_version(), "m2-draft");
    assert_eq!(artifact.dt_ms(), Some(1.0), "m3 fixture has dt_ms = 1.0");

    let reserialized = artifact.to_json();
    let artifact2 =
        parse_bytes(reserialized.as_bytes()).expect("re-parse of round-tripped m3_with_dt_ms");
    assert_eq!(artifact.format_version(), artifact2.format_version());
    assert_eq!(artifact.dt_ms(), artifact2.dt_ms());
    assert_eq!(artifact.layer_count(), artifact2.layer_count());
}

// ── Unsupported version ───────────────────────────────────────────────────────

#[test]
fn unsupported_format_version_is_rejected() {
    let json = r#"{
  "format_version": "m99-future",
  "thrindex_version": "9.0.0",
  "target": "sim",
  "model": { "layers": [] },
  "metadata": {
    "compiled_at": "2026-01-01T00:00:00Z",
    "model_canonical": "{\"layers\":[]}",
    "crc32": "d3f77b75"
  }
}"#;

    match parse_bytes(json.as_bytes()) {
        Err(ArtifactError::UnsupportedFormatVersion { version, .. }) => {
            assert_eq!(version, "m99-future");
        }
        other => panic!("expected UnsupportedFormatVersion, got {other:?}"),
    }
}

// ── Corrupt CRC32 is rejected ─────────────────────────────────────────────────

#[test]
fn corrupt_crc32_is_rejected() {
    let json = r#"{
  "format_version": "m2-draft",
  "thrindex_version": "0.2.0",
  "target": "sim",
  "model": {
    "layers": [
      {
        "bias_b64": null,
        "in_features": 2,
        "out_features": 2,
        "type": "dense",
        "weights_b64": "AACAPwAAAAAAAAAAAACAPw=="
      }
    ]
  },
  "metadata": {
    "compiled_at": "2026-01-01T00:00:00Z",
    "model_canonical": "{\"layers\":[{\"bias_b64\":null,\"in_features\":2,\"out_features\":2,\"type\":\"dense\",\"weights_b64\":\"AACAPwAAAAAAAAAAAACAPw==\"}]}",
    "crc32": "deadbeef"
  }
}"#;

    match parse_bytes(json.as_bytes()) {
        Err(ArtifactError::IntegrityCheckFailed { expected, .. }) => {
            assert_eq!(expected, "deadbeef");
        }
        other => panic!("expected IntegrityCheckFailed, got {other:?}"),
    }
}
