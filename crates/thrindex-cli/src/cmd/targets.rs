//! `thrindex targets` — list available simulation targets.
//!
//! For M2 there is exactly one target: `sim` (behavioral float32 simulator).
//! Hardware targets are added in M4 (post RFC-004).

use clap::Args;

#[derive(Args, Debug)]
pub struct TargetsArgs {}

/// Execute `thrindex targets`.  Returns the rendered table string.
pub fn run(_args: &TargetsArgs) -> String {
    use std::fmt::Write as _;

    let border = "═".repeat(55);
    let sep = "─".repeat(55);
    let mut s = String::new();

    writeln!(s, "{border}").ok();
    writeln!(
        s,
        " thrindex {}  —  available targets",
        env!("CARGO_PKG_VERSION")
    )
    .ok();
    writeln!(s, "{border}").ok();
    writeln!(s, " sim      Behavioral simulator (this build)").ok();
    writeln!(s, "          Precision: float32  |  ADR-0007").ok();
    writeln!(s, "          Deterministic, CPU-parallel (Rayon)").ok();
    writeln!(s, "{sep}").ok();
    writeln!(
        s,
        " Fixed-point targets: pending RFC-004 (target calibration)"
    )
    .ok();
    writeln!(s, "{border}").ok();

    s
}
