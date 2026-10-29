//! A new `.thx` states its geometry and the sim descriptor it was lowered with.
//!
//! Older files omit both fields and still load. A stored report that disagrees
//! with the layers is rejected, as is a capability hash that is not the sim one.

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use thrindex_artifact::{expected_sim_capability_hash, parse};
use thrindex_compiler::compile;

fn dense_lif_ir() -> String {
    let weights = B64.encode(vec![0u8; 2 * 2 * 4]);
    format!(
        r#"{{
          "dt_ms": 1.0,
          "layers": [
            {{"type":"dense","in_features":2,"out_features":2,"weights_b64":"{weights}","bias_b64":null,"delays":null}},
            {{"type":"lif","tau_mem":10.0,"tau_syn":null,"threshold":1.0,"reset":"subtract"}}
          ]
        }}"#
    )
}

#[test]
fn compile_round_trips_through_the_parser() {
    let report = compile(&dense_lif_ir(), "sim").expect("compile");
    let artifact = parse(&report.thx_json).expect("parse");
    assert_eq!(artifact.to_json(), report.thx_json);
    assert_eq!(
        artifact.capability_hash(),
        Some(expected_sim_capability_hash().as_str())
    );
    assert!(artifact.has_resource_report());
    let summary = artifact.resource_summary();
    assert_eq!(summary.input_shape, Some(2));
    assert_eq!(summary.output_shape, Some(2));
    assert_eq!(summary.lif_neuron_count, 2);
    assert_eq!(summary.total_weight_count, 4);

    let value: serde_json::Value = serde_json::from_str(&report.thx_json).unwrap();
    let keys: Vec<&str> = value["model"]["layers"][0]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys.first().copied(),
        Some("bias_b64"),
        "layer keys stay alphabetical, got {keys:?}"
    );
}

#[test]
fn tampered_capability_hash_is_rejected() {
    let report = compile(&dense_lif_ir(), "sim").expect("compile");
    let mut value: serde_json::Value = serde_json::from_str(&report.thx_json).unwrap();
    value["metadata"]["capability_hash"] =
        serde_json::json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    let tampered = serde_json::to_string_pretty(&value).unwrap();
    let err = parse(&tampered).expect_err("bad capability hash");
    let msg = err.to_string();
    assert!(
        msg.contains("capability_hash"),
        "expected the hash mismatch to be named, got {msg}"
    );
}

#[test]
fn tampered_resource_report_is_rejected() {
    let report = compile(&dense_lif_ir(), "sim").expect("compile");
    let mut value: serde_json::Value = serde_json::from_str(&report.thx_json).unwrap();
    value["model"]["resource_report"]["input_shape"] = serde_json::json!(999);
    let tampered = serde_json::to_string_pretty(&value).unwrap();
    let err = parse(&tampered).expect_err("bad resource report");
    let msg = err.to_string();
    assert!(
        msg.contains("input_shape") && msg.contains("999"),
        "expected the disagreed field, got {msg}"
    );
}
