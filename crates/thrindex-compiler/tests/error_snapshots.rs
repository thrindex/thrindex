//! Snapshot tests for §30-format compiler error messages (E0101–E0109).
//!
//! Run `cargo insta review` after any intentional message change.

use insta::assert_snapshot;
use thrindex_compiler::compile;

// ── E0108 — JSON parse failure ────────────────────────────────────────────────

#[test]
fn e0108_bad_json() {
    let err = compile("not json", "sim").unwrap_err();
    assert_snapshot!("e0108_bad_json", err.to_string());
}

#[test]
fn e0108_missing_dt_ms() {
    let err = compile(r#"{"layers":[]}"#, "sim").unwrap_err();
    assert_snapshot!("e0108_missing_dt_ms", err.to_string());
}

// ── E0105 — empty model ───────────────────────────────────────────────────────

#[test]
fn e0105_empty_model() {
    let err = compile(r#"{"dt_ms":1.0,"layers":[]}"#, "sim").unwrap_err();
    assert_snapshot!("e0105_empty_model", err.to_string());
}

// ── E0107 — invalid reset mode (validate pass, NOT serde) ────────────────────

#[test]
fn e0107_invalid_reset_mode() {
    // "reset" is deserialized as a String — serde accepts it — then validate rejects it.
    let ir = r#"{"dt_ms":1.0,"layers":[{"type":"lif","tau_mem":10.0,"tau_syn":null,"threshold":1.0,"reset":"hard"}]}"#;
    let err = compile(ir, "sim").unwrap_err();
    assert_snapshot!("e0107_invalid_reset_mode", err.to_string());
}

// ── E0101 — tau_mem too small ─────────────────────────────────────────────────

#[test]
fn e0101_tau_mem_too_small() {
    // tau_mem = 0.5 ms ≤ dt_ms = 1.0 ms — rejected in validate pass.
    let ir = r#"{"dt_ms":1.0,"layers":[{"type":"lif","tau_mem":0.5,"tau_syn":null,"threshold":1.0,"reset":"subtract"}]}"#;
    let err = compile(ir, "sim").unwrap_err();
    assert_snapshot!("e0101_tau_mem_too_small", err.to_string());
}

// ── E0106 — dimension mismatch ────────────────────────────────────────────────

#[test]
fn e0106_dimension_mismatch() {
    // Dense(4→8) then Dense(4→4) — 4 ≠ 8.
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    let zeros_4x8 = B64.encode(vec![0u8; 4 * 8 * 4]); // f32 weights 8 out × 4 in
    let zeros_4x4 = B64.encode(vec![0u8; 4 * 4 * 4]); // f32 weights 4 out × 4 in
    let ir = format!(
        r#"{{
          "dt_ms": 1.0,
          "layers": [
            {{"type":"dense","in_features":4,"out_features":8,"weights_b64":"{zeros_4x8}","bias_b64":null,"delays":null}},
            {{"type":"dense","in_features":4,"out_features":4,"weights_b64":"{zeros_4x4}","bias_b64":null,"delays":null}}
          ]
        }}"#
    );
    let err = compile(&ir, "sim").unwrap_err();
    assert_snapshot!("e0106_dimension_mismatch", err.to_string());
}

// ── Unknown target ─────────────────────────────────────────────────────────────

#[test]
fn unknown_target() {
    let ir = r#"{"dt_ms":1.0,"layers":[{"type":"lif","tau_mem":10.0,"tau_syn":null,"threshold":1.0,"reset":"subtract"}]}"#;
    let err = compile(ir, "other").unwrap_err();
    assert_snapshot!("unknown_target", err.to_string());
}

// ── Successful round-trip (smoke test) ───────────────────────────────────────

#[test]
fn compile_dense_lif_succeeds() {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    let weights = B64.encode(vec![0u8; 2 * 2 * 4]); // 2-out × 2-in f32 zeros
    let ir = format!(
        r#"{{
          "dt_ms": 1.0,
          "layers": [
            {{"type":"dense","in_features":2,"out_features":2,"weights_b64":"{weights}","bias_b64":null,"delays":null}},
            {{"type":"lif","tau_mem":10.0,"tau_syn":null,"threshold":1.0,"reset":"subtract"}}
          ]
        }}"#
    );
    let report = compile(&ir, "sim").expect("compile should succeed");
    assert!(report.advisory.is_none(), "no retiming advisory expected");

    // Verify the artifact has the expected fields.
    let artifact: serde_json::Value = serde_json::from_str(&report.thx_json).unwrap();
    assert_eq!(artifact["format_version"], "m2-draft");
    assert_eq!(artifact["target"], "sim");
    assert_eq!(artifact["metadata"]["dt_ms"], 1.0);
    assert_eq!(artifact["model"]["resource_report"]["input_shape"], 2);
    assert_eq!(artifact["model"]["resource_report"]["output_shape"], 2);
    assert_eq!(
        artifact["model"]["resource_report"]["total_weight_count"],
        4
    );
    assert_eq!(artifact["model"]["resource_report"]["lif_neuron_count"], 2);
    assert!(
        artifact["metadata"]["capability_hash"]
            .as_str()
            .unwrap()
            .len()
            == 64
    );

    // alpha = exp(-1.0 / 10.0) ≈ 0.9048374.
    let alpha = artifact["model"]["layers"][1]["alpha"].as_f64().unwrap();
    assert!(
        (alpha - (-0.1f64).exp()).abs() < 1e-6,
        "alpha should be exp(-dt/tau_mem)"
    );
}

#[test]
fn compile_dense_only_succeeds() {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    // Dense(4→2): row-major [out × in] = [[1,0,0,0],[0,1,0,0]] so all-ones input
    // produces a non-zero activation (needed for Console liveness without LIF).
    let mut raw = Vec::with_capacity(8 * 4);
    for v in [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let weights = B64.encode(&raw);
    let ir = format!(
        r#"{{
          "dt_ms": 1.0,
          "layers": [
            {{"type":"dense","in_features":4,"out_features":2,"weights_b64":"{weights}","bias_b64":null,"delays":null}}
          ]
        }}"#
    );
    let report = compile(&ir, "sim").expect("dense-only compile should succeed");
    assert!(report.advisory.is_none());
    let artifact: serde_json::Value = serde_json::from_str(&report.thx_json).unwrap();
    assert_eq!(artifact["format_version"], "m2-draft");
    assert_eq!(artifact["target"], "sim");
    assert_eq!(artifact["model"]["resource_report"]["input_shape"], 4);
    assert_eq!(artifact["model"]["resource_report"]["output_shape"], 2);
    assert_eq!(
        artifact["model"]["resource_report"]["total_weight_count"],
        8
    );
    assert_eq!(artifact["model"]["resource_report"]["lif_neuron_count"], 0);
    let layers = artifact["model"]["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0]["type"], "dense");
    assert!(
        layers.iter().all(|l| l["type"] != "lif"),
        "dense-only artifact must not contain LIF"
    );
    if let Ok(path) = std::env::var("WRITE_DENSE_FEEDFORWARD_THX") {
        std::fs::write(&path, &report.thx_json).expect("write dense-feedforward fixture");
    }
}
