//! `E####` error codes for `thrindex-compiler`.
//!
//! Codes E0101–E0109 are the M3 compiler range (E0100–E0199, per `docs/errors/README.md`).
//! Each variant maps 1:1 to a stable `E####` code.
//! Every display string is the §30 four-part contract (What / Why / Fix / Docs),
//! and is snapshot-tested.

use thiserror::Error;

/// All compiler errors.  Each variant maps 1:1 to a stable `E####` code.
#[derive(Debug, Error)]
pub enum CompileError {
    // ── Retiming errors (shared with ADR-0008) ─────────────────────────────
    /// E0101 — membrane time constant is too small for the effective `dt`.
    ///
    /// Fires during the **validate** pass (when `tau_mem ≤ authored dt_ms`) and
    /// during the **lower** pass (when `tau_mem ≤ target native dt` after retiming).
    #[error(
        "E0101: LIF tau_mem too small for effective dt in layer {layer_idx}\n\
         Why: tau_mem={tau_mem:.4} ms is ≤ dt={effective_dt:.4} ms; the membrane \
         would lose all history within one step (ADR-0005 guard, ADR-0008).\n\
         Fix: set tau_mem > {effective_dt:.4} ms when constructing the LIF layer.\n\
         Docs: https://docs.thrindex.com/errors/E0101"
    )]
    TauMemTooSmall {
        layer_idx: usize,
        tau_mem: f64,
        effective_dt: f64,
    },

    /// E0102 — delay does not divide evenly into the target step ratio.
    ///
    /// Only fires during retiming (when target `dt ≠ authored `dt`). Delays are
    /// integer multiples of `dt`; re-scaling by `dt_a / dt_t` must yield an integer.
    #[error(
        "E0102: synaptic delay cannot be retimed in layer {layer_idx} connection {conn_idx}\n\
         Why: delay_steps={delay_steps} × (authored_dt={authored_dt:.4} ms / \
         target_dt={target_dt:.4} ms) = {ratio:.6} is not an integer; \
         sub-step delay precision is not available.\n\
         Fix: use a delay that is an integer multiple of target_dt / authored_dt, \
         or author the model at target_dt directly.\n\
         Docs: https://docs.thrindex.com/errors/E0102"
    )]
    RetimingDelayNotInteger {
        layer_idx: usize,
        conn_idx: u32,
        delay_steps: u16,
        authored_dt: f64,
        target_dt: f64,
        ratio: f64,
    },

    /// E0103 — delay exceeds the target's declared maximum.
    #[error(
        "E0103: synaptic delay exceeds target maximum in layer {layer_idx} connection {conn_idx}\n\
         Why: delay_steps={delay_steps} > native_delay_max_steps={max_steps} \
         for this target, and delay_fallback is \"reject\".\n\
         Fix: reduce the delay to ≤ {max_steps} steps, or use a target that \
         supports longer delays or emulation.\n\
         Docs: https://docs.thrindex.com/errors/E0103"
    )]
    DelayExceedsTargetMax {
        layer_idx: usize,
        conn_idx: u32,
        delay_steps: u16,
        max_steps: u16,
    },

    /// E0104 — target has no native delay support and emulation is disabled.
    #[error(
        "E0104: target has no native delay support and emulation is disabled in layer {layer_idx}\n\
         Why: this target's capability descriptor declares \
         native_delay_max_steps=0 and delay_fallback=\"reject\".\n\
         Fix: use the \"sim\" target (which emulates delays via ring buffers), \
         or remove delays from the model.\n\
         Docs: https://docs.thrindex.com/errors/E0104"
    )]
    NoNativeDelaySupport { layer_idx: usize },

    // ── Validate pass errors ───────────────────────────────────────────────
    /// E0105 — model contains no layers.
    #[error(
        "E0105: model has no layers\n\
         Why: an empty Sequential cannot be compiled or simulated.\n\
         Fix: add at least one layer to the model before calling thx.compile().\n\
         Docs: https://docs.thrindex.com/errors/E0105"
    )]
    EmptyModel,

    /// E0106 — consecutive layers have incompatible dimensions.
    #[error(
        "E0106: layer dimension mismatch at layer {layer_idx}\n\
         Why: layer {layer_idx} declares {expected} input features but layer \
         {prev_idx} produces {got} output features.\n\
         Fix: recheck the architecture — input/output sizes must chain correctly.\n\
         Docs: https://docs.thrindex.com/errors/E0106"
    )]
    DimensionMismatch {
        layer_idx: usize,
        prev_idx: usize,
        expected: usize,
        got: usize,
    },

    /// E0107 — LIF reset mode string is not recognized.
    ///
    /// Fires in the **validate** pass (not capture), because `GraphLif::reset` is
    /// deserialized as a plain `String` — serde cannot produce a §30-format error.
    /// The validate pass converts it via `ResetMode::try_from` and surfaces this code.
    #[error(
        "E0107: invalid LIF reset mode {mode:?} in layer {layer_idx}\n\
         Why: reset must be \"subtract\" or \"zero\" (Playbook §28 canonical vocab).\n\
         Fix: change reset to \"subtract\" or \"zero\" when constructing the LIF layer.\n\
         Docs: https://docs.thrindex.com/errors/E0107"
    )]
    InvalidResetMode { layer_idx: usize, mode: String },

    /// E0108 — JSON deserialization of the Graph IR failed.
    ///
    /// This is the capture-pass equivalent of E0008 for the Graph IR JSON format.
    #[error(
        "E0108: Graph IR JSON parse error: {detail}\n\
         Why: the input is not valid Graph IR JSON (produced by thx.compile's \
         Python extraction pass).\n\
         Fix: verify the Graph IR was produced by a current version of the \
         thrindex Python SDK.\n\
         Docs: https://docs.thrindex.com/errors/E0108"
    )]
    IrJsonParseError { detail: String },

    /// E0109 — delay array length does not match the declared Dense layer dimensions.
    #[error(
        "E0109: delay array length mismatch in layer {layer_idx}\n\
         Why: a dense delay array must have exactly in_features × out_features = \
         {expected} entries, but it has {got} entries.\n\
         Fix: recompile the model — the Graph IR delay data may be corrupt.\n\
         Docs: https://docs.thrindex.com/errors/E0109"
    )]
    DelayLengthMismatch {
        layer_idx: usize,
        expected: usize,
        got: usize,
    },
}
