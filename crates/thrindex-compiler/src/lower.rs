//! Lower pass: [`GraphModel`] → sealed `.thx` artifact JSON.
//!
//! Implements:
//! - Two-level parameter resolution: `alpha = exp(-dt_eff / tau_mem)` (ADR-0008).
//! - Retiming advisory when target `dt ≠ authored `dt` (ADR-0008 §2).
//! - Delay encoding: dense/sparse passthrough; E0102 on non-integer retimed delays (ADR-0009).
//! - CRC32 artifact integrity (ADR-0006).
//! - Zero-delay = zero bytes: absent `delays` field in output (ADR-0009 §4).

use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use crc32fast::Hasher as Crc32Hasher;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use thrindex_ir::{Delays, GraphLayer, GraphModel, ResetMode};

use crate::error::CompileError;

// ── Target capability descriptor ──────────────────────────────────────────────

/// Backend capability descriptor (§16 capability descriptor, ADR-0009).
///
/// No core crate hardcodes chip-specific constants — backends declare their true
/// `native_delay_max_steps` here.  The `"sim"` target never rejects delays.
pub struct SimCapability {
    /// Native step size in milliseconds.
    pub native_dt_ms: f64,
    /// Maximum supported delay steps (0 = no native delay support).
    pub native_delay_max_steps: u16,
    /// Policy when delay exceeds cap or cap is 0.
    pub delay_fallback: DelayFallback,
}

/// Delay fallback policy when native support is insufficient.
pub enum DelayFallback {
    /// Emulate via ring-buffer (incurs memory/cycle cost, but works).
    Emulate,
    /// Reject with `E0103`/`E0104`.
    Reject,
}

fn capability_for(target: &str) -> Result<SimCapability, CompileError> {
    match target {
        "sim" => Ok(SimCapability {
            native_dt_ms: 1.0,
            native_delay_max_steps: u16::MAX,
            delay_fallback: DelayFallback::Emulate,
        }),
        other => Err(CompileError::IrJsonParseError {
            detail: format!("unknown target {other:?}; supported: \"sim\""),
        }),
    }
}

// ── Compile report ────────────────────────────────────────────────────────────

/// Output of a successful compilation.
#[derive(Debug)]
pub struct CompileReport {
    /// The complete `.thx` artifact as a JSON string.
    pub thx_json: String,
    /// Retiming advisory emitted when target `dt ≠ authored `dt`, or `None`.
    ///
    /// Callers **MUST** surface this to the user and **MUST NOT** assume models
    /// are equivalent under a changed `dt` without re-evaluation.
    pub advisory: Option<String>,
}

// ── .thx artifact types (target-side) ────────────────────────────────────────

/// Top-level `.thx` artifact (ADR-0006, `m2-draft` format).
#[derive(Serialize, Deserialize)]
struct ThxArtifact {
    format_version: String,
    thrindex_version: String,
    target: String,
    model: ThxModel,
    metadata: ThxMetadata,
}

/// The model block inside a `.thx` artifact.
#[derive(Serialize, Deserialize)]
struct ThxModel {
    layers: Vec<Value>,
}

/// Metadata block inside a `.thx` artifact.
#[derive(Serialize, Deserialize)]
struct ThxMetadata {
    compiled_at: String,
    /// The model block re-serialized with sorted keys (used for CRC32).
    model_canonical: String,
    crc32: String,
    /// Authored `dt_ms` from the Graph IR (ADR-0008 extension to ADR-0006 metadata).
    dt_ms: f64,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Lower a validated [`GraphModel`] to a sealed `.thx` artifact for `target`.
///
/// # Errors
///
/// Returns [`CompileError`] for unsupported targets or retiming failures (E0101/E0102).
///
/// # Advisory
///
/// If `authored_dt ≠ target native_dt`, a retiming advisory is included in
/// [`CompileReport::advisory`].  The model **MUST** be re-evaluated at the target
/// `dt`; the advisory text states this explicitly.
pub fn lower(model: &GraphModel, target: &str) -> Result<CompileReport, CompileError> {
    let cap = capability_for(target)?;

    let authored_dt = model.dt_ms;
    let target_dt = cap.native_dt_ms;
    let retiming = (target_dt - authored_dt).abs() > 1e-12;

    let advisory = if retiming {
        Some(format!(
            "RETIMING ADVISORY: authored dt={authored_dt:.4} ms → target dt={target_dt:.4} ms. \
             alpha and delay_steps have been re-derived at the target dt. \
             This changes the training regime; the model SHOULD be re-evaluated \
             at target dt={target_dt:.4} ms and MUST NOT be assumed equivalent to the \
             trained checkpoint. Any accuracy figure is an estimate only."
        ))
    } else {
        None
    };

    let mut thx_layers: Vec<Value> = Vec::with_capacity(model.layers.len());

    for (idx, layer) in model.layers.iter().enumerate() {
        let v = match layer {
            GraphLayer::Dense(dense) => {
                lower_dense(dense, idx, authored_dt, target_dt, &cap, &mut thx_layers)?;
                continue;
            }
            GraphLayer::Lif(lif) => lower_lif(lif, idx, target_dt)?,
            GraphLayer::Conv2d(conv) => lower_conv2d(conv),
        };
        thx_layers.push(v);
    }

    let thx_model = ThxModel { layers: thx_layers };

    // CRC32 is computed on the canonically serialized model block (sorted keys).
    let model_canonical = canonical_json(&serde_json::to_value(&thx_model).map_err(|e| {
        CompileError::IrJsonParseError {
            detail: e.to_string(),
        }
    })?);

    let mut hasher = Crc32Hasher::new();
    hasher.update(model_canonical.as_bytes());
    let crc32 = hasher.finalize();

    let compiled_at = now_iso8601();

    let artifact = ThxArtifact {
        format_version: "m2-draft".to_string(),
        thrindex_version: env!("CARGO_PKG_VERSION").to_string(),
        target: target.to_string(),
        model: thx_model,
        metadata: ThxMetadata {
            compiled_at,
            model_canonical,
            crc32: format!("{crc32:08x}"),
            dt_ms: authored_dt,
        },
    };

    let thx_json =
        serde_json::to_string_pretty(&artifact).map_err(|e| CompileError::IrJsonParseError {
            detail: e.to_string(),
        })?;

    Ok(CompileReport { thx_json, advisory })
}

// ── Layer lowering ────────────────────────────────────────────────────────────

fn lower_dense(
    dense: &thrindex_ir::GraphDense,
    layer_idx: usize,
    authored_dt: f64,
    target_dt: f64,
    cap: &SimCapability,
    out: &mut Vec<Value>,
) -> Result<(), CompileError> {
    let mut obj = serde_json::json!({
        "type": "dense",
        "in_features": dense.in_features,
        "out_features": dense.out_features,
        "weights_b64": dense.weights_b64,
        "bias_b64": dense.bias_b64,
    });

    // Delays: only emitted if present (zero-delay = zero bytes, ADR-0009 §4).
    if let Some(delays) = &dense.delays {
        let (delays_b64, encoding) = lower_delays(
            delays,
            layer_idx,
            authored_dt,
            target_dt,
            dense.in_features,
            dense.out_features,
            cap,
        )?;
        obj["delays_b64"] = Value::String(delays_b64);
        obj["delays_encoding"] = Value::String(encoding);
    }

    out.push(obj);
    Ok(())
}

fn lower_lif(
    lif: &thrindex_ir::GraphLif,
    layer_idx: usize,
    target_dt: f64,
) -> Result<Value, CompileError> {
    // E0101 at lower time (target_dt guard).
    if lif.tau_mem <= target_dt {
        return Err(CompileError::TauMemTooSmall {
            layer_idx,
            tau_mem: lif.tau_mem,
            effective_dt: target_dt,
        });
    }

    // Two-level resolution (ADR-0008): compute alpha from continuous tau_mem.
    let alpha = (-target_dt / lif.tau_mem).exp() as f32;
    let alpha_syn = lif.tau_syn.map(|ts| (-target_dt / ts).exp() as f32);

    // Reset mode was validated; parse safely.
    let reset_str = match ResetMode::try_from(lif.reset.as_str()) {
        Ok(ResetMode::Subtract) => "subtract",
        Ok(ResetMode::Zero) => "zero",
        // Unreachable after validate pass — but produce a safe fallback.
        Err(_) => "subtract",
    };

    Ok(serde_json::json!({
        "type": "lif",
        "threshold": lif.threshold,
        "alpha": alpha,
        "alpha_syn": alpha_syn,
        "reset": reset_str,
    }))
}

fn lower_conv2d(conv: &thrindex_ir::GraphConv2d) -> Value {
    serde_json::json!({
        "type": "conv2d",
        "in_channels": conv.in_channels,
        "out_channels": conv.out_channels,
        "kernel_h": conv.kernel_h,
        "kernel_w": conv.kernel_w,
        "stride": conv.stride,
        "padding": conv.padding,
        "weights_b64": conv.weights_b64,
        "bias_b64": conv.bias_b64,
    })
}

// ── Delay lowering ────────────────────────────────────────────────────────────

fn lower_delays(
    delays: &Delays,
    layer_idx: usize,
    authored_dt: f64,
    target_dt: f64,
    in_features: usize,
    out_features: usize,
    cap: &SimCapability,
) -> Result<(String, String), CompileError> {
    let retiming = (target_dt - authored_dt).abs() > 1e-12;

    match delays {
        Delays::Dense { delays_b64 } => {
            if !retiming && cap.native_delay_max_steps == u16::MAX {
                // Fast path: no retiming, no cap — pass bytes through as-is.
                return Ok((delays_b64.clone(), "dense".to_string()));
            }
            let raw = decode_b64(delays_b64, layer_idx)?;
            let steps_in = parse_u16_le(&raw, layer_idx)?;

            let mut steps_out = Vec::with_capacity(steps_in.len());
            for (i, &s) in steps_in.iter().enumerate() {
                let s_out = retime_step(s, authored_dt, target_dt, layer_idx, i as u32)?;
                check_cap(s_out, layer_idx, i as u32, cap, in_features * out_features)?;
                steps_out.push(s_out);
            }

            Ok((encode_u16_le(&steps_out), "dense".to_string()))
        }

        Delays::Sparse { delays_b64 } => {
            if !retiming && cap.native_delay_max_steps == u16::MAX {
                return Ok((delays_b64.clone(), "sparse".to_string()));
            }
            let raw = decode_b64(delays_b64, layer_idx)?;
            let pairs = parse_sparse_pairs(&raw, layer_idx)?;

            let mut out_pairs: Vec<(u32, u16)> = Vec::with_capacity(pairs.len());
            for (idx_conn, steps) in pairs {
                let s_out = retime_step(steps, authored_dt, target_dt, layer_idx, idx_conn)?;
                check_cap(s_out, layer_idx, idx_conn, cap, in_features * out_features)?;
                out_pairs.push((idx_conn, s_out));
            }

            Ok((encode_sparse_pairs(&out_pairs), "sparse".to_string()))
        }
    }
}

/// Retime a single delay step from `authored_dt` to `target_dt`.
///
/// Produces E0102 if the scaled value is not an integer.
fn retime_step(
    steps: u16,
    authored_dt: f64,
    target_dt: f64,
    layer_idx: usize,
    conn_idx: u32,
) -> Result<u16, CompileError> {
    if (target_dt - authored_dt).abs() <= 1e-12 {
        return Ok(steps);
    }
    let ratio = f64::from(steps) * authored_dt / target_dt;
    let rounded = ratio.round();
    if (ratio - rounded).abs() > 1e-9 {
        return Err(CompileError::RetimingDelayNotInteger {
            layer_idx,
            conn_idx,
            delay_steps: steps,
            authored_dt,
            target_dt,
            ratio,
        });
    }
    Ok(rounded as u16)
}

/// Check a delay step against the target's declared maximum.
fn check_cap(
    steps: u16,
    layer_idx: usize,
    conn_idx: u32,
    cap: &SimCapability,
    _total_connections: usize,
) -> Result<(), CompileError> {
    if cap.native_delay_max_steps == 0 {
        match cap.delay_fallback {
            DelayFallback::Reject => {
                return Err(CompileError::NoNativeDelaySupport { layer_idx });
            }
            DelayFallback::Emulate => {}
        }
    } else if steps > cap.native_delay_max_steps {
        match cap.delay_fallback {
            DelayFallback::Reject => {
                return Err(CompileError::DelayExceedsTargetMax {
                    layer_idx,
                    conn_idx,
                    delay_steps: steps,
                    max_steps: cap.native_delay_max_steps,
                });
            }
            DelayFallback::Emulate => {}
        }
    }
    Ok(())
}

// ── Byte codec helpers ────────────────────────────────────────────────────────

fn decode_b64(s: &str, layer_idx: usize) -> Result<Vec<u8>, CompileError> {
    B64.decode(s).map_err(|e| CompileError::IrJsonParseError {
        detail: format!("layer {layer_idx}: base64 decode failed: {e}"),
    })
}

fn parse_u16_le(bytes: &[u8], layer_idx: usize) -> Result<Vec<u16>, CompileError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(CompileError::IrJsonParseError {
            detail: format!("layer {layer_idx}: delay byte count not divisible by 2"),
        });
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn parse_sparse_pairs(bytes: &[u8], layer_idx: usize) -> Result<Vec<(u32, u16)>, CompileError> {
    if !bytes.len().is_multiple_of(6) {
        return Err(CompileError::IrJsonParseError {
            detail: format!("layer {layer_idx}: sparse delay byte count not divisible by 6"),
        });
    }
    Ok(bytes
        .chunks_exact(6)
        .map(|c| {
            let idx = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let steps = u16::from_le_bytes([c[4], c[5]]);
            (idx, steps)
        })
        .collect())
}

fn encode_u16_le(steps: &[u16]) -> String {
    let bytes: Vec<u8> = steps.iter().flat_map(|s| s.to_le_bytes()).collect();
    B64.encode(&bytes)
}

fn encode_sparse_pairs(pairs: &[(u32, u16)]) -> String {
    let bytes: Vec<u8> = pairs
        .iter()
        .flat_map(|(idx, steps)| idx.to_le_bytes().into_iter().chain(steps.to_le_bytes()))
        .collect();
    B64.encode(&bytes)
}

// ── Canonical JSON (sorted keys) ──────────────────────────────────────────────

/// Re-serialize a JSON value with lexicographically sorted object keys.
///
/// Used to produce the canonical model representation for CRC32.
fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let sorted: BTreeMap<&str, &Value> = map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            let inner: Vec<String> = sorted
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap(),
                        canonical_json(v)
                    )
                })
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        Value::Array(arr) => {
            let inner: Vec<String> = arr.iter().map(canonical_json).collect();
            format!("[{}]", inner.join(","))
        }
        _ => serde_json::to_string(v).unwrap_or_default(),
    }
}

// ── Timestamp ─────────────────────────────────────────────────────────────────

/// Return the current UTC time as ISO-8601, or a fixed string if unavailable.
fn now_iso8601() -> String {
    // ENGINEERING.md: determinism principle — callers can override in tests via the
    // `CompileReport` rather than via global state.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Minimal ISO-8601 UTC without external deps (chrono, time — §12 stack).
    let s = secs % 60;
    let m_tot = secs / 60;
    let m = m_tot % 60;
    let h_tot = m_tot / 60;
    let h = h_tot % 24;
    let days = h_tot / 24;

    // Gregorian day decomposition (1970-01-01 epoch).
    let (year, month, day) = days_to_ymd(days);

    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Convert days since 1970-01-01 to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970u64;
    loop {
        let leap = is_leap(year);
        let yd = if leap { 366 } else { 365 };
        if days < yd {
            break;
        }
        days -= yd;
        year += 1;
    }
    let leap = is_leap(year);
    let months = [
        31u64,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u64;
    for m_days in months {
        if days < m_days {
            break;
        }
        days -= m_days;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}
