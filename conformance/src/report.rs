//! Structured conformance report (ADR-0010 Part I).
use crate::ConformanceEnvelope;

/// Whether the envelope is final law or provisional draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeStatus {
    /// Certification-valid. Backends may be certified against this envelope.
    Final,
    /// Provisional — not certification-valid.
    ///
    /// No backend may be certified or rejected on a Draft envelope.
    /// Every report header must say "DRAFT — not certification-valid".
    Draft,
}

/// Per-sample metrics collected by the conformance harness.
#[derive(Debug, Clone)]
pub struct SampleMetrics {
    /// Hamming fraction (informational only).
    pub hamming: f64,
    /// Mean per-neuron rate error (primary metric).
    pub mean_rate_error: f64,
    /// Max per-neuron rate error (worst-case neuron).
    pub max_rate_error: f64,
    /// Mean first-spike latency error in timesteps (informational only).
    pub mean_fsl_error: f64,
    /// Whether reference and backend agree on the argmax prediction.
    pub prediction_agrees: bool,
}

/// Full conformance run result for a backend against the reference simulator.
#[derive(Debug, Clone)]
pub struct ConformanceReport {
    /// Backend name from capability descriptor.
    pub backend_name: String,
    /// Envelope version string (e.g. `"v0_draft"`, `"v0"`).
    pub envelope_version: &'static str,
    /// Whether the envelope is draft or final.
    pub envelope_status: EnvelopeStatus,
    /// Number of test samples evaluated.
    pub n_samples: usize,
    /// Per-sample breakdown.
    pub samples: Vec<SampleMetrics>,
    /// Aggregate mean of `mean_rate_error` across all samples.
    pub agg_mean_rate_error: f64,
    /// Aggregate max of `max_rate_error` across all samples (worst single neuron).
    pub agg_max_rate_error: f64,
    /// Prediction agreement fraction across all samples.
    pub pred_agreement: f64,
    /// T_mean threshold from the envelope used for this run (stored for self-contained rendering).
    pub envelope_t_mean: f64,
    /// T_max threshold from the envelope used for this run.
    pub envelope_t_max: f64,
    /// P_min threshold from the envelope used for this run.
    pub envelope_p_min: f64,
}

impl ConformanceReport {
    /// Returns `true` if this report passes the given envelope's thresholds.
    ///
    /// Always returns `false` for [`EnvelopeStatus::Draft`] envelopes — a draft
    /// envelope cannot certify a backend.
    #[must_use]
    pub fn passed(&self, envelope: &ConformanceEnvelope) -> bool {
        if self.envelope_status == EnvelopeStatus::Draft {
            return false;
        }
        self.agg_mean_rate_error <= envelope.t_mean_threshold
            && self.agg_max_rate_error <= envelope.t_max_threshold
            && self.pred_agreement >= envelope.pred_agreement_min
    }

    /// Returns `true` if the numeric thresholds pass (ignoring draft status).
    ///
    /// Used during M4 development to check whether metrics are in range without
    /// claiming certification. Reports this as "WOULD PASS (envelope is DRAFT)".
    #[must_use]
    pub fn metrics_within_draft_thresholds(&self, envelope: &ConformanceEnvelope) -> bool {
        self.agg_mean_rate_error <= envelope.t_mean_threshold
            && self.agg_max_rate_error <= envelope.t_max_threshold
            && self.pred_agreement >= envelope.pred_agreement_min
    }

    /// Render a human-readable conformance report.
    ///
    /// The header always states whether the envelope is DRAFT or FINAL.
    /// A DRAFT envelope report header explicitly says "not certification-valid".
    #[must_use]
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let border = "═".repeat(60);
        let sep = "─".repeat(60);

        let draft_notice = match self.envelope_status {
            EnvelopeStatus::Draft => "  ⚠ DRAFT — not certification-valid ⚠",
            EnvelopeStatus::Final => "",
        };

        writeln!(s, "{border}").ok();
        writeln!(s, " THRINDEX Conformance Report  [envelope: {}]{draft_notice}", self.envelope_version).ok();
        writeln!(s, " Backend: {}", self.backend_name).ok();
        writeln!(s, "{border}").ok();
        writeln!(s, " Samples evaluated:      {}", self.n_samples).ok();
        writeln!(s, " Agg mean rate error:    {:.4e}", self.agg_mean_rate_error).ok();
        writeln!(s, " Agg max rate error:     {:.4e}", self.agg_max_rate_error).ok();
        writeln!(s, " Prediction agreement:   {:.2}%", self.pred_agreement * 100.0).ok();
        writeln!(s, "{sep}").ok();

        // Use stored thresholds — the report is self-contained and does not need the
        // envelope to be passed in at render time. This avoids referencing global constants
        // and correctly handles both v0_draft and v0 (and future v1, v2, ...) reports.
        let metrics_pass = self.agg_mean_rate_error <= self.envelope_t_mean
            && self.agg_max_rate_error <= self.envelope_t_max
            && self.pred_agreement >= self.envelope_p_min;

        match self.envelope_status {
            EnvelopeStatus::Draft => {
                if metrics_pass {
                    writeln!(s, " WOULD PASS (draft thresholds) — not certification-valid").ok();
                } else {
                    writeln!(s, " WOULD FAIL (draft thresholds) — not certification-valid").ok();
                }
            }
            EnvelopeStatus::Final => {
                if metrics_pass {
                    writeln!(s, " PASS — THRINDEX Certified [{}]", self.envelope_version).ok();
                } else {
                    writeln!(s, " FAIL — does not meet conformance thresholds").ok();
                }
            }
        }

        writeln!(s, "{sep}").ok();
        writeln!(s, " Per-sample detail (hamming and FSL are informational):").ok();
        for (i, m) in self.samples.iter().enumerate() {
            writeln!(
                s,
                "  sample_{i:03}: mean_rate_err={:.3e}  max_rate_err={:.3e}  \
                 hamming={:.3e}  fsl_err={:.2}  pred={}",
                m.mean_rate_error,
                m.max_rate_error,
                m.hamming,
                m.mean_fsl_error,
                if m.prediction_agrees { "✓" } else { "✗" }
            )
            .ok();
        }
        writeln!(s, "{border}").ok();
        s
    }
}
