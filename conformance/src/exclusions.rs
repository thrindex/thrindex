//! Backend exclusions from the spike-raster conformance suite.
//!
//! The SDK ships one backend, the reference simulator, and it is not excluded.
//! The list stays so a future non-spiking backend would have to be named here
//! instead of disappearing from the report.
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
    /// Format is stable: `EXCLUDED <name>: reason=<reason>; metric=<metric>`.
    #[must_use]
    pub fn exclusion_line(&self) -> String {
        format!(
            "EXCLUDED {}: reason={}; metric={}",
            self.backend_name, self.reason, self.alternative_metric
        )
    }
}

/// Backends left out of the spike-raster suite. Empty: the simulator is in the suite.
pub const EXCLUDED_BACKENDS: &[BackendExclusion] = &[];

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
/// Called when the harness meets a backend listed in [`EXCLUDED_BACKENDS`].
pub fn print_exclusion(exclusion: &BackendExclusion) {
    println!("{}", exclusion.exclusion_line());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_backends_are_excluded() {
        assert!(EXCLUDED_BACKENDS.is_empty());
        assert!(find_exclusion("sim").is_none());
    }

    #[test]
    fn exclusion_line_names_the_backend() {
        let line = BackendExclusion {
            backend_name: "example",
            reason: "unspecified",
            alternative_metric: "none",
        }
        .exclusion_line();
        assert_eq!(line, "EXCLUDED example: reason=unspecified; metric=none");
    }
}
