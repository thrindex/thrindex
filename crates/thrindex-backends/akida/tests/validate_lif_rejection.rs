//! Integration test: E0401 — LIF layer rejected, error message snapshot.
//!
//! Proves the public API path: validate_artifact(lif_json) → BackendError containing E0401.
//! Snapshot-tests the AkidaError::LifNotSupported message so any change to the
//! four-part error format fails CI.
use thrindex_backend_akida::{AkidaError, validate_artifact};

/// E0401 golden test: .thx with a single Lif layer returns BackendError containing E0401.
///
/// Snapshot is taken on the AkidaError text (the `detail` field inside BackendError::Execution)
/// rather than the BackendError wrapper, so changes to E0401 wording fail CI independently.
#[test]
fn validate_lif_rejection() {
    const LIF_ARTIFACT: &str = r#"{
        "format_version": "m2-draft",
        "target": "akida-akd1500",
        "model": {
            "layers": [
                {
                    "type": "lif",
                    "threshold": 1.0,
                    "alpha": 0.9,
                    "reset": "subtract"
                }
            ]
        },
        "metadata": {}
    }"#;

    let err = validate_artifact(LIF_ARTIFACT)
        .expect_err("LIF artifact must be rejected by akida-akd1500 backend");
    let msg = err.to_string();

    // The BackendError wrapper is E0204; the Akida-specific code E0401 lives in `detail`.
    assert!(
        msg.contains("E0401"),
        "BackendError must contain E0401 in detail; got:\n{msg}"
    );
    assert!(
        msg.contains("lif"),
        "error must mention the rejected layer type; got:\n{msg}"
    );
    assert!(
        msg.contains("bounded ReLU"),
        "error must explain the neuron model incompatibility; got:\n{msg}"
    );
    assert!(
        msg.contains("What to do:"),
        "error must include remediation advice; got:\n{msg}"
    );

    // Snapshot the AkidaError text directly to pin the four-part format.
    let e401 = AkidaError::LifNotSupported {
        index: 0,
        threshold: 1.0,
        alpha: 0.9,
        reset: "subtract".to_owned(),
    };
    let e401_msg = e401.to_string();
    insta::assert_snapshot!(
        &e401_msg,
        @r###"
E0401: layer[0] type="lif" cannot be mapped to akida-akd1500.
Observed: .thx layer 0 has type "lif" with threshold=1, alpha=0.9, reset="subtract".
Why: AKD1500 implements Akida 1.0. Akida 1.0's activation is bounded ReLU applied per-inference-call to the integer dot product. It has no membrane potential, no exponential leak (alpha term), and no spike-triggered reset. The leaky integrate-and-fire dynamics encoded in this layer do not exist in the hardware.
What to do: (a) Use the 'sim' backend for SNN simulation. (b) If you want AKD1500 inference, re-design the model without LIF layers using the CNN workflow described in the BrainChip documentation. There is no lossless conversion from LIF to bounded-ReLU; they are different computational models.
Docs: https://docs.thrindex.com/errors/E0401
"###
    );
}
