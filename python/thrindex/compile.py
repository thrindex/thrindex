"""Model compiler: serialize a :class:`thrindex.snn.Sequential` to a ``.thx`` artifact.

## Architecture (M3 — ADR-0008, ADR-0009)

The compile path is now split into two passes:

1. **Python extraction pass** (this module): walks the PyTorch model and emits a
   *Graph IR JSON* string containing **continuous parameters** — ``tau_mem``, ``tau_syn``,
   ``threshold``, ``reset``.  ``alpha`` is **never computed in Python** (two-level rule,
   ADR-0008).

2. **Rust compilation pass** (``thrindex._core.compile_to_thx``): runs capture →
   validate → lower, resolving ``alpha = exp(-dt/tau_mem)`` at the target's effective
   ``dt``, encoding delays, computing CRC32, and emitting the sealed ``.thx`` JSON.

## Why the split?

Resolving ``alpha`` in Python and resolving it in Rust for the same ``tau_mem``/``dt``
pair may differ at the last ULP due to Python's ``math.exp`` vs Rust's ``libm::exp``.
Centralising resolution in Rust ensures a single, audited code path and that every
artifact's ``alpha`` matches what the simulator would compute (ADR-0007 §4).
"""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from thrindex.snn.conv2d import Conv2d as ThxConv2d
from thrindex.snn.dense import Dense as ThxDense
from thrindex.snn.lif import LIF
from thrindex.snn.sequential import Sequential

if TYPE_CHECKING:
    pass

__all__ = ["compile_model"]


def compile_model(
    model: nn.Module,
    path: str | Path,
    target: str = "sim",
    version: str = "0.3.0",
) -> None:
    """Compile *model* to a ``.thx`` artifact at *path*.

    Parameters
    ----------
    model:
        A :class:`thrindex.snn.Sequential` — the only supported container for M3.
    path:
        Output path.  The ``.thx`` extension is conventional but not enforced.
    target:
        Target identifier.  ``"sim"`` is the only valid value for M3.
    version:
        thrindex SDK version.  Used for informational metadata only; the Rust
        compiler writes its own ``CARGO_PKG_VERSION`` into the artifact.
    """
    if not isinstance(model, Sequential):
        raise TypeError(
            f"compile_model expects a thrindex.snn.Sequential, got {type(model).__name__}"
        )

    ir_json = _build_ir_json(model)
    thx_json, advisory = _compile_via_rust(ir_json, target)

    if advisory is not None:
        print(f"THRINDEX WARNING: {advisory}", file=sys.stderr)

    Path(path).write_text(thx_json, encoding="utf-8")


# ── Graph IR extraction (Python pass) ─────────────────────────────────────────


def _build_ir_json(model: Sequential, dt_ms: float = 1.0) -> str:
    """Walk *model* and produce a Graph IR JSON string.

    The Graph IR carries **continuous** parameters only (ADR-0008 two-level rule):
    - LIF: ``tau_mem``, ``tau_syn``, ``threshold``, ``reset`` — **no ``alpha``**.
    - Dense: weights, optional bias, optional delays (not yet exposed in M3 SDK).
    - Conv2d: weights, channels, kernel shape, stride, padding — **no delays**
      (Conv2d-delay support deferred, ADR-0009 v1 scope).
    """
    layers: list[dict[str, Any]] = [_serialise_layer_ir(layer) for layer in model.layers]
    ir: dict[str, Any] = {"dt_ms": dt_ms, "layers": layers}
    return json.dumps(ir)


def _serialise_layer_ir(layer: nn.Module) -> dict[str, Any]:
    if isinstance(layer, ThxDense):
        return _serialise_dense_ir(layer)
    if isinstance(layer, ThxConv2d):
        return _serialise_conv2d_ir(layer)
    if isinstance(layer, LIF):
        return _serialise_lif_ir(layer)
    raise TypeError(
        f"unsupported layer type for compile_model: {type(layer).__name__}. "
        "Only Dense, Conv2d, and LIF are supported."
    )


def _serialise_dense_ir(layer: ThxDense) -> dict[str, Any]:
    w = layer.linear.weight.detach().cpu()
    b: Tensor | None = layer.linear.bias  # type: ignore[assignment]
    return {
        "type": "dense",
        "in_features": w.shape[1],
        "out_features": w.shape[0],
        "weights_b64": _to_b64(w),
        "bias_b64": _to_b64(b.detach().cpu()) if b is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
        "delays": None,
    }


def _serialise_lif_ir(layer: LIF) -> dict[str, Any]:
    # ADR-0008 two-level rule: emit tau_mem (continuous), NOT alpha (resolved).
    # alpha = exp(-dt / tau_mem) is computed by the Rust lower pass.
    return {
        "type": "lif",
        "tau_mem": float(layer.tau_mem),
        "tau_syn": float(layer.tau_syn) if layer.tau_syn is not None else None,
        "threshold": float(layer.threshold),
        "reset": layer.reset,
    }


def _serialise_conv2d_ir(layer: ThxConv2d) -> dict[str, Any]:
    conv = layer.conv
    w = conv.weight.detach().cpu()
    b: Tensor | None = conv.bias  # type: ignore[assignment]
    out_c, in_c, kh, kw = w.shape
    # Conv2d carries no delays (ADR-0009 v1 scope — deferred).
    return {
        "type": "conv2d",
        "in_channels": in_c,
        "out_channels": out_c,
        "kernel_h": kh,
        "kernel_w": kw,
        "stride": list(conv.stride),
        "padding": list(conv.padding),
        "weights_b64": _to_b64(w),
        "bias_b64": _to_b64(b.detach().cpu()) if b is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
    }


# ── Rust compilation pass ─────────────────────────────────────────────────────


def _compile_via_rust(ir_json: str, target: str) -> tuple[str, str | None]:
    """Call ``thrindex._core.compile_to_thx`` to run the Rust compilation pipeline.

    Returns ``(thx_json, advisory_or_none)``.

    Raises ``ValueError`` (with §30-format E#### message) on any compile error.
    Raises ``ImportError`` when the native extension is not installed (development mode).
    """
    try:
        from thrindex import _core  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "thrindex._core is not installed.  "
            "Build it with `maturin develop` or `pip install thrindex`."
        ) from exc

    thx_json, advisory = _core.compile_to_thx(ir_json, target)
    return thx_json, advisory


# ── Encoding helpers ───────────────────────────────────────────────────────────


def _to_b64(t: Tensor) -> str:
    """Encode a tensor as base64 little-endian f32 bytes (ADR-0006).

    Uses ``struct.pack`` (stdlib) to avoid a numpy dependency — CI installs the
    CPU-only torch wheel which ships without numpy.
    """
    flat: list[float] = t.to(dtype=torch.float32).contiguous().reshape(-1).tolist()
    raw = struct.pack(f"<{len(flat)}f", *flat)
    return base64.b64encode(raw).decode("ascii")
