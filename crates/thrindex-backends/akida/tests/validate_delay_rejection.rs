//! Integration test: E0402 — Dense layer with delays rejected.
use thrindex_backend_akida::validate_artifact;

/// E0402 golden test: .thx with Dense + delays_b64 returns BackendError containing E0402.
#[test]
fn validate_delay_rejection() {
    const DELAY_ARTIFACT: &str = r#"{
        "format_version": "m2-draft",
        "target": "akida-akd1500",
        "model": {
            "layers": [
                {
                    "type": "dense",
                    "in_features": 4,
                    "out_features": 2,
                    "weights_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                    "delays_b64": "AAAA",
                    "delays_encoding": "dense"
                }
            ]
        },
        "metadata": {}
    }"#;

    let err = validate_artifact(DELAY_ARTIFACT)
        .expect_err("artifact with delays must be rejected by akida-akd1500 backend");
    let msg = err.to_string();

    assert!(
        msg.contains("E0402"),
        "BackendError must contain E0402; got:\n{msg}"
    );
    assert!(
        msg.contains("native_delay_max_steps=0"),
        "error must cite the zero delay cap; got:\n{msg}"
    );
    assert!(
        msg.contains("TNP"),
        "error must explain TNP is not present on Akida 1.0; got:\n{msg}"
    );
    assert!(
        msg.contains("dense"),
        "error must state the delay encoding type; got:\n{msg}"
    );
}
