//! `thrindex bench --conformance --target sim` — conformance check command.
//!
//! Runs the conformance harness (ADR-0010) using the reference `SimBackend`
//! as both reference and backend-under-test. Since the simulator is deterministic,
//! all per-neuron rate errors are 0.0.
//!
//! With ≥100 samples (full ratification set) the report reads:
//! ```text
//! PASS — THRINDEX Certified [v0]
//! ```
//!
//! With fewer than 100 samples (e.g. the 3 bundled template samples) the report is
//! labeled "STRUCTURAL DEMO — insufficient samples for ratification".
//!
//! This command:
//! 1. Loads a `.thx` artifact (default: `templates/keyword-spotting/model.thx`).
//! 2. Loads spike-raster test fixtures from a fixture directory
//!    (default: `templates/keyword-spotting/samples/`).
//! 3. Runs the conformance harness using [`conformance::harness::run_conformance`].
//! 4. Prints the report and exits 0 (pass) or 1 (fail / structural error).
//!
//! ## Sample count
//!
//! The final `CONFORMANCE_ENVELOPE_V0` requires ≥100 samples. When fewer are provided
//! (e.g. the 3 bundled template samples), the report is additionally labeled
//! "STRUCTURAL DEMO — insufficient samples for ratification". This is expected for the
//! `thrindex bench --conformance --target sim` M4 definition-of-done run.
//!
//! For the full ratification measurement (ADR-0010 Part II), run:
//! ```bash
//! cargo run -p conformance --bin ratify_envelope -- \
//!     --artifact templates/keyword-spotting/model.thx \
//!     --data-dir conformance/fixtures/shd_100
//! ```

use clap::Args;

use conformance::{
    CONFORMANCE_ENVELOPE_V0, CONFORMANCE_ENVELOPE_V0_DRAFT, harness::run_conformance,
};
use thrindex_sim::SimBackend;

/// Arguments for `thrindex bench`.
#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Run the spike-equivalence conformance suite (ADR-0010).
    #[arg(long, default_value_t = false)]
    pub conformance: bool,

    /// Execution target to test against the reference simulator. Currently only "sim"
    /// is available; hardware targets will be added in M5+.
    #[arg(long, default_value = "sim")]
    pub target: String,

    /// Path to the `.thx` artifact to load.
    /// Default: `templates/keyword-spotting/model.thx` (relative to CWD).
    #[arg(long)]
    pub artifact: Option<String>,

    /// Directory containing fixture JSON files (`sample_NNN.json`, `{ "spikes": [...] }`).
    /// Default: `templates/keyword-spotting/samples/` (relative to CWD).
    #[arg(long)]
    pub fixtures: Option<String>,
}

/// Output of a bench run — the rendered report string or an error message.
#[allow(clippy::type_complexity)]
pub fn run(args: &BenchArgs) -> Result<String, String> {
    if !args.conformance {
        return Err("thrindex bench: no action specified.\n\
             Use `thrindex bench --conformance --target sim` to run the conformance suite.\n\
             Use `thrindex bench --help` for more options."
            .to_string());
    }

    if args.target != "sim" {
        return Err(format!(
            "thrindex bench: target '{}' is not yet available.\n\
             Currently only '--target sim' is supported (M4).\n\
             Hardware targets are planned for M5+.",
            args.target
        ));
    }

    // ── Resolve paths ──────────────────────────────────────────────────────────
    let artifact_path = args
        .artifact
        .as_deref()
        .unwrap_or("templates/keyword-spotting/model.thx");
    let fixtures_dir = args
        .fixtures
        .as_deref()
        .unwrap_or("templates/keyword-spotting/samples");

    let artifact_json = std::fs::read_to_string(artifact_path).map_err(|e| {
        format!(
            "E0001: cannot read artifact '{artifact_path}': {e}\n\
             Fix: provide a path via --artifact, or run from the repository root where \
             templates/keyword-spotting/model.thx exists."
        )
    })?;

    // ── Load fixtures ──────────────────────────────────────────────────────────
    let (inputs, n_loaded) = load_fixtures(fixtures_dir)?;

    // ── Run conformance ────────────────────────────────────────────────────────
    // For the structural demo (n < 100), we temporarily allow small test sets.
    // The report labels this as "STRUCTURAL DEMO" so no one mistakes it for
    // a certification run.
    let reference = SimBackend::new(1);
    let backend = SimBackend::new(1); // backend-under-test is also the sim (M4 reference)

    // Full conformance runs use the ratified V0 envelope. Structural demos
    // (fewer than 100 samples) use the superseded DRAFT with relaxed sample count
    // so the report cannot be confused with real certification output.
    let structural_demo = n_loaded < CONFORMANCE_ENVELOPE_V0.min_test_samples;

    let demo_envelope;
    let active_envelope = if structural_demo {
        demo_envelope = conformance::ConformanceEnvelope {
            min_test_samples: n_loaded, // accept whatever we loaded
            ..CONFORMANCE_ENVELOPE_V0_DRAFT
        };
        &demo_envelope
    } else {
        &CONFORMANCE_ENVELOPE_V0
    };

    let report = run_conformance(
        &backend,
        &reference,
        &artifact_json,
        &inputs,
        active_envelope,
    )
    .map_err(|e| e.to_string())?;

    // ── Render ─────────────────────────────────────────────────────────────────
    let mut out = report.render();

    if structural_demo {
        use std::fmt::Write as _;
        let _ = write!(
            out,
            "\n⚠ STRUCTURAL DEMO — {n_loaded} samples loaded (minimum for certification: {}).\n\
             Provide ≥100 samples via --fixtures to obtain a certification-valid run.\n",
            CONFORMANCE_ENVELOPE_V0.min_test_samples
        );
    }

    Ok(out)
}

/// Load `sample_NNN.json` files from `dir`. Each must have a `"spikes"` array.
/// Returns `(inputs, n_samples)` where inputs is `[n_samples][timesteps][features]`.
#[allow(clippy::type_complexity, clippy::cast_possible_truncation)]
fn load_fixtures(dir: &str) -> Result<(Vec<Vec<Vec<f32>>>, usize), String> {
    let path = std::path::Path::new(dir);
    if !path.exists() {
        return Err(format!(
            "Fixture directory '{dir}' does not exist.\n\
             Provide an alternative via --fixtures, or run from the repository root."
        ));
    }

    let mut entries: Vec<_> = std::fs::read_dir(path)
        .map_err(|e| format!("cannot read directory '{dir}': {e}"))?
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("sample_"))
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);

    if entries.is_empty() {
        return Err(format!(
            "No sample_NNN.json files found in '{dir}'.\n\
             The directory must contain files named sample_000.json, sample_001.json, etc."
        ));
    }

    let mut inputs = Vec::new();
    for entry in &entries {
        let raw = std::fs::read_to_string(entry.path())
            .map_err(|e| format!("reading {}: {e}", entry.path().display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("parsing {}: {e}", entry.path().display()))?;
        let spikes: Vec<Vec<f32>> = v["spikes"]
            .as_array()
            .ok_or_else(|| format!("{}: missing 'spikes' field", entry.path().display()))?
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect();
        inputs.push(spikes);
    }

    let n = inputs.len();
    Ok((inputs, n))
}
