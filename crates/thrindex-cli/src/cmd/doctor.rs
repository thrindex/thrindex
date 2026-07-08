//! `thrindex doctor` — check the user's environment.
//!
//! Checks the user's runtime world, not a developer's build environment.
//! Correction 7: no Rust toolchain check — prebuilt-wheel users never have `rustc`.
//!
//! Checks:
//!   - Artifact readability (if `--check <file>` provided) → E0001/E0002/E0008/E0009
//!   - thrindex-sim version (self-check)

use clap::Args;

#[derive(Args, Debug)]
pub struct DoctorArgs {
    /// Optional path to a `.thx` artifact to test readability.
    #[arg(long)]
    pub check: Option<String>,

    /// Show verbose detail for each check.
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Debug)]
struct Check {
    label: String,
    status: CheckStatus,
    detail: Option<String>,
}

#[derive(Debug, PartialEq)]
enum CheckStatus {
    Ok,
    Advisory,
    Fail,
}

impl CheckStatus {
    fn icon(&self) -> &'static str {
        match self {
            Self::Ok => "OK ",
            Self::Advisory => "?  ",
            Self::Fail => "ERR",
        }
    }
}

/// Execute `thrindex doctor`.  Returns the rendered report string.
pub fn run(args: &DoctorArgs) -> String {
    let mut checks: Vec<Check> = Vec::new();

    // ── Check 1: thrindex-sim self-check ────────────────────────────────────
    checks.push(Check {
        label: format!("thrindex-sim {}", env!("CARGO_PKG_VERSION")),
        status: CheckStatus::Ok,
        detail: Some("core simulator library loaded successfully".into()),
    });

    // ── Check 2: artifact readability ────────────────────────────────────────
    if let Some(path) = &args.check {
        let result = thrindex_sim::model::load(path);
        match result {
            Ok(m) => checks.push(Check {
                label: format!("artifact: {path}"),
                status: CheckStatus::Ok,
                detail: Some(format!("{} layers — format OK", m.layers.len())),
            }),
            Err(e) => checks.push(Check {
                label: format!("artifact: {path}"),
                status: CheckStatus::Fail,
                detail: Some(e.to_string()),
            }),
        }
    }

    // ── Render ───────────────────────────────────────────────────────────────
    render_report(&checks, args.verbose)
}

fn render_report(checks: &[Check], verbose: bool) -> String {
    use std::fmt::Write as _;

    let border = "═".repeat(55);
    let mut s = String::new();
    writeln!(s, "{border}").ok();
    writeln!(s, " thrindex doctor v{}", env!("CARGO_PKG_VERSION")).ok();
    writeln!(s, "{border}").ok();

    let advisory_count = checks
        .iter()
        .filter(|c| c.status == CheckStatus::Advisory)
        .count();
    let fail_count = checks
        .iter()
        .filter(|c| c.status == CheckStatus::Fail)
        .count();

    for check in checks {
        let icon = check.status.icon();
        writeln!(s, " [{icon}]  {}", check.label).ok();
        if (verbose || check.status != CheckStatus::Ok)
            && let Some(detail) = &check.detail
        {
            for line in detail.lines() {
                writeln!(s, "        {line}").ok();
            }
        }
    }

    writeln!(s, "{border}").ok();
    if fail_count > 0 {
        writeln!(
            s,
            " {fail_count} failure(s). Run `thrindex doctor --verbose` for details."
        )
        .ok();
    } else if advisory_count > 0 {
        writeln!(
            s,
            " {advisory_count} advisory. Run `thrindex doctor --verbose` for details."
        )
        .ok();
    } else {
        writeln!(s, " All checks passed.").ok();
    }
    writeln!(s, "{border}").ok();

    s
}
