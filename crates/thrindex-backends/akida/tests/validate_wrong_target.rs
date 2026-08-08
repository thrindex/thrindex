//! Integration test: E0404 — artifact compiled for a different target is rejected.
use thrindex_backend_akida::validate_artifact;

/// E0404 golden test: .thx with target="sim" returns BackendError containing E0404.
#[test]
fn validate_wrong_target() {
    const SIM_ARTIFACT: &str = r#"{
        "format_version": "m2-draft",
        "target": "sim",
        "model": {
            "layers": []
        },
        "metadata": {}
    }"#;

    let err = validate_artifact(SIM_ARTIFACT)
        .expect_err("sim-targeted artifact must be rejected by akida-akd1500 backend");
    let msg = err.to_string();

    assert!(
        msg.contains("E0404"),
        "BackendError must contain E0404; got:\n{msg}"
    );
    assert!(
        msg.contains("\"sim\""),
        "error must cite the actual target name; got:\n{msg}"
    );
    assert!(
        msg.contains("akida-akd1500"),
        "error must name the expected target; got:\n{msg}"
    );
    assert!(
        msg.contains("Recompile"),
        "error must suggest recompilation; got:\n{msg}"
    );
}

/// Empty-string target is rejected as E0404 (target="" ≠ "akida-akd1500").
#[test]
fn missing_target_field_rejected() {
    const NO_TARGET_ARTIFACT: &str = r#"{
        "format_version": "m2-draft",
        "model": {"layers": []},
        "metadata": {}
    }"#;

    let err = validate_artifact(NO_TARGET_ARTIFACT)
        .expect_err("artifact with no target field must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("E0404"),
        "missing target must trigger E0404; got:\n{msg}"
    );
}
