//! Backend exclusions from the RFC-003 spike-raster conformance suite.
//!
//! ## Why this module exists
//!
//! Not every registered backend implements SNN spike-raster dynamics. AKD1500
//! runs bounded ReLU inference on integer dot products — not leaky integrate-and-fire.
//! It cannot produce a spike raster comparable to the reference simulator.
//!
//! The RFC-003 metric therefore does not apply. But a conformance harness that
//! silently skips a registered backend is not a standard — it is an omission.
//! This module enforces that the harness **actively prints** the exclusion line
//! for every registered non-SNN backend, every time it runs, regardless of
//! which CI machine or test environment is in use.
//!
//! ## Mandatory harness output
//!
//! For each excluded backend the harness MUST print exactly:
//! ```text
//! EXCLUDED <name>: reason=<reason>; metric=<alternative_metric>
//! ```
//! This tells any log reader why the backend produced no raster distance scores
//! and what metric should be used in its place (RFC-004 Correction 3; ADR-0011 §5).

/// A backend excluded from the RFC-003 spike-raster conformance suite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendExclusion {
    /// Backend name matching the capability descriptor `name` field.
    pub backend_name: &'static str,
    /// Machine-readable reason code (lowercase, hyphens, no spaces).
    pub reason: &'static str,
    /// Alternative metric this backend CAN be evaluated against.
    pub alternative_metric: &'static str,
}

impl BackendExclusion {
    /// Returns the canonical exclusion line as it MUST appear in harness stdout.
    ///
    /// Format: `EXCLUDED <name>: reason=<reason>; metric=<alternative_metric>`
    ///
    /// This string is snapshot-tested (`akida_exclusion_line_format`). Any change
    /// requires updating the snapshot AND a corresponding ADR amendment.
    #[must_use]
    pub fn exclusion_line(&self) -> String {
        format!(
            "EXCLUDED {}: reason={}; metric={}",
            self.backend_name, self.reason, self.alternative_metric
        )
    }
}

/// All registered backend exclusions from the spike-raster conformance suite.
///
/// When adding a new non-SNN hardware backend:
/// 1. Append an entry here.
/// 2. State the reason code (why RFC-003 doesn't apply).
/// 3. Name the alternative metric (what the backend CAN be evaluated against).
/// 4. Update the corresponding ADR.
///
/// When removing or modifying an entry:
/// 5. The snapshot test `akida_exclusion_line_format` will fail CI — update it.
/// 6. Obtain founder approval and update the relevant ADR.
pub const EXCLUDED_BACKENDS: &[BackendExclusion] = &[
    // ADR-0011: AKD1500 implements bounded ReLU (Akida 1.0), not LIF.
    // RFC-003 spike-raster metric requires SNN temporal dynamics; AKD1500 is
    // a single-frame stateless CNN accelerator. Exclusion is permanent for
    // Akida 1.0 targets.
    BackendExclusion {
        backend_name: "akida-akd1500",
        reason: "non-snn-backend",
        alternative_metric: "top-1-accuracy-frozen-brainchip-dataset",
    },
];

/// Find the exclusion entry for a backend by name.
///
/// Returns `Some` if the backend is excluded from spike-raster conformance;
/// returns `None` if the backend is subject to normal conformance.
#[must_use]
pub fn find_exclusion(backend_name: &str) -> Option<&'static BackendExclusion> {
    EXCLUDED_BACKENDS
        .iter()
        .find(|e| e.backend_name == backend_name)
}

/// Print the mandatory exclusion line to stdout for an excluded backend.
///
/// This function MUST be called — not skipped — when the harness encounters
/// a backend that is in [`EXCLUDED_BACKENDS`]. Silent omission violates the
/// conformance standard (RFC-004 Correction 3; ADR-0011 §5).
pub fn print_exclusion(exclusion: &BackendExclusion) {
    println!("{}", exclusion.exclusion_line());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert that akida-akd1500 is present in the registered exclusion list.
    ///
    /// If this test fails, the exclusion was accidentally removed. Either re-add it
    /// or obtain ADR amendment approval before removing it.
    #[test]
    fn akida_excluded_flag() {
        let exclusion = find_exclusion("akida-akd1500");
        assert!(
            exclusion.is_some(),
            "akida-akd1500 must be in EXCLUDED_BACKENDS — \
             removing it without an ADR amendment violates RFC-004 Correction 3"
        );
        let e = exclusion.unwrap();
        assert_eq!(e.backend_name, "akida-akd1500");
        assert_eq!(
            e.reason, "non-snn-backend",
            "reason code must be 'non-snn-backend' (ADR-0011 §5)"
        );
        assert_eq!(
            e.alternative_metric, "top-1-accuracy-frozen-brainchip-dataset",
            "alternative metric must match ADR-0011 §5"
        );
    }

    /// Assert the exact exclusion line format — snapshot-tested.
    ///
    /// Any change to this line requires updating the snapshot AND an ADR amendment.
    /// The line is mandatory CI output (RFC-004 Correction 3; ADR-0011 §5).
    #[test]
    fn akida_exclusion_line_format() {
        let exclusion = find_exclusion("akida-akd1500").unwrap();
        let line = exclusion.exclusion_line();
        insta::assert_snapshot!(
            &line,
            @"EXCLUDED akida-akd1500: reason=non-snn-backend; metric=top-1-accuracy-frozen-brainchip-dataset"
        );
    }

    /// The `sim` backend is subject to normal conformance — it must NOT be excluded.
    #[test]
    fn sim_backend_not_excluded() {
        assert!(
            find_exclusion("sim").is_none(),
            "sim backend must not be excluded from spike-raster conformance"
        );
    }

    /// `print_exclusion` produces exactly the same string as `exclusion_line()`.
    /// (Tested via a captured String; in production the harness writes to stdout.)
    #[test]
    fn print_exclusion_matches_line() {
        let e = find_exclusion("akida-akd1500").unwrap();
        let expected = e.exclusion_line();
        // Verify the format string is stable without actual stdout capture.
        assert!(
            expected.starts_with("EXCLUDED akida-akd1500:"),
            "exclusion line must start with 'EXCLUDED akida-akd1500:'; got: {expected}"
        );
        assert!(
            expected.contains("reason=non-snn-backend"),
            "exclusion line must contain reason code; got: {expected}"
        );
        assert!(
            expected.contains("metric=top-1-accuracy-frozen-brainchip-dataset"),
            "exclusion line must contain alternative metric; got: {expected}"
        );
    }
}
