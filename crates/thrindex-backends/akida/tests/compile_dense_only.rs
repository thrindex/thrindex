//! Integration test: Dense-only .thx → .fbz via akida_compile.py.
//!
//! Gated by environment variable THRINDEX_AKIDA_COMPILE_TEST=1.
//! Requires the `akida` Python package installed (Python 3.10–3.12, Linux).
//!
//! To run:
//!   THRINDEX_AKIDA_COMPILE_TEST=1 cargo test -p thrindex-backend-akida compile_dense_only
//!
//! This test is skipped (returns immediately with a note to stderr) when the env var
//! is not set. It does NOT use #[ignore] so it shows as "ok" in the normal test run
//! and does not pollute the ignore count.
use std::path::PathBuf;
use std::process::Command;

const DENSE_ONLY_ARTIFACT: &str = r#"{
    "format_version": "m2-draft",
    "target": "akida-akd1500",
    "model": {
        "layers": [
            {
                "type": "dense",
                "in_features": 4,
                "out_features": 2,
                "weights_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
            }
        ]
    },
    "metadata": {}
}"#;

/// Smoke test: Dense(4→2) .thx artifact compiles to non-empty .fbz bytes via Tier-1 Python tool.
///
/// Gated by THRINDEX_AKIDA_COMPILE_TEST=1. Skipped silently when gate is not set.
///
/// When this test passes:
/// - akida_compile.py correctly reads a .thx, calls InputData + FullyConnected.set_variable(),
///   maps to AKD1500, and returns a valid .fbz byte payload.
/// - The confirmed spike-test path (set_variable with scale=max|W|/7) is verified end-to-end.
#[test]
fn compile_dense_only_smoke() {
    if std::env::var("THRINDEX_AKIDA_COMPILE_TEST").as_deref() != Ok("1") {
        eprintln!(
            "SKIP compile_dense_only_smoke: \
             set THRINDEX_AKIDA_COMPILE_TEST=1 (requires akida Python package, Linux)"
        );
        return;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let compile_script = manifest_dir.join("python").join("akida_compile.py");

    assert!(
        compile_script.exists(),
        "akida_compile.py not found at {}: check crate layout",
        compile_script.display()
    );

    let tmpdir = std::env::temp_dir();
    let artifact_path = tmpdir.join("thrindex_akida_smoke.thx");
    let fbz_path = tmpdir.join("thrindex_akida_smoke.fbz");

    std::fs::write(&artifact_path, DENSE_ONLY_ARTIFACT).expect("failed to write test artifact");

    let output = Command::new("python3")
        .arg(&compile_script)
        .arg(&artifact_path)
        .arg(&fbz_path)
        .output()
        .expect("failed to spawn akida_compile.py — is python3 on PATH?");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "akida_compile.py exited with status {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );

    let fbz = std::fs::read(&fbz_path).expect("failed to read .fbz output file");
    assert!(
        !fbz.is_empty(),
        "akida_compile.py produced empty .fbz — expected non-empty hardware program bytes"
    );

    eprintln!("OK: .fbz produced, {} bytes", fbz.len());
    eprintln!("compile output:\n{stderr}");
}
