//! Validate pass: semantic checks on a [`GraphModel`] before lowering.
//!
//! Produces the first encountered error and stops. Future work: collect all errors
//! and return a `Vec<CompileError>` for multi-error reporting.
//!
//! Checks performed, in order:
//! 1. E0105 — model is non-empty.
//! 2. E0107 — every `GraphLif::reset` string is a recognized [`ResetMode`].
//! 3. E0101 — every `GraphLif::tau_mem > model.dt_ms` (authored-dt guard).
//! 4. E0106 — consecutive connection-bearing and LIF layers have matching dimensions.
//! 5. E0109 — delay array lengths match `in_features × out_features` for Dense layers.

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};

use thrindex_ir::{Delays, GraphLayer, GraphModel, ResetMode};

use crate::error::CompileError;

/// Validate a [`GraphModel`] for semantic correctness.
///
/// # Errors
///
/// Returns the first semantic error encountered.
pub fn validate(model: &GraphModel) -> Result<(), CompileError> {
    // E0105 — non-empty model.
    if model.layers.is_empty() {
        return Err(CompileError::EmptyModel);
    }

    // Track the output width of the previous connection-bearing layer (Dense/Conv2d)
    // so we can verify continuity with the next Dense's `in_features`.
    let mut prev_out: Option<(usize, usize)> = None; // (layer_idx, out_features)

    for (idx, layer) in model.layers.iter().enumerate() {
        match layer {
            GraphLayer::Lif(lif) => {
                // E0107 — valid reset mode string.
                ResetMode::try_from(lif.reset.as_str()).map_err(|_| {
                    CompileError::InvalidResetMode {
                        layer_idx: idx,
                        mode: lif.reset.clone(),
                    }
                })?;

                // E0101 — tau_mem must exceed the authored dt_ms.
                if lif.tau_mem <= model.dt_ms {
                    return Err(CompileError::TauMemTooSmall {
                        layer_idx: idx,
                        tau_mem: lif.tau_mem,
                        effective_dt: model.dt_ms,
                    });
                }

                // LIF does not produce a new "width" — it preserves it.
            }

            GraphLayer::Dense(dense) => {
                // E0106 — check dimension continuity.
                if let Some((prev_idx, prev_width)) = prev_out
                    && dense.in_features != prev_width
                {
                    return Err(CompileError::DimensionMismatch {
                        layer_idx: idx,
                        prev_idx,
                        expected: dense.in_features,
                        got: prev_width,
                    });
                }

                // E0109 — validate delay array length for Dense layers.
                if let Some(delays) = &dense.delays {
                    let expected = dense.in_features * dense.out_features;
                    let bytes = decode_delay_bytes(delays)
                        .map_err(|e| CompileError::IrJsonParseError { detail: e })?;
                    let entry_count = match delays {
                        Delays::Dense { .. } => bytes.len() / 2, // u16 = 2 bytes each
                        Delays::Sparse { .. } => bytes.len() / 6, // (u32, u16) = 6 bytes
                    };
                    // Dense encoding must be exactly `expected` entries.
                    // Sparse encoding is validated by the lower pass (connection indices).
                    if matches!(delays, Delays::Dense { .. }) && entry_count != expected {
                        return Err(CompileError::DelayLengthMismatch {
                            layer_idx: idx,
                            expected,
                            got: entry_count,
                        });
                    }
                }

                prev_out = Some((idx, dense.out_features));
            }

            GraphLayer::Conv2d(conv) => {
                prev_out = Some((idx, conv.out_channels));
            }
        }
    }

    Ok(())
}

/// Decode the raw bytes from a [`Delays`] variant (shared Dense/Sparse path).
fn decode_delay_bytes(delays: &Delays) -> Result<Vec<u8>, String> {
    let b64 = match delays {
        Delays::Dense { delays_b64 } | Delays::Sparse { delays_b64 } => delays_b64,
    };
    B64.decode(b64)
        .map_err(|e| format!("delay base64 decode failed: {e}"))
}
