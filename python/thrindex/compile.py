"""Model compiler: serialize a :class:`thrindex.snn.Sequential` to a ``.thx`` artifact.

The compiler resolves all derived constants at compile time (correction 4 / ADR-0007):
- ``alpha = exp(-dt / tau_mem)``
- ``alpha_syn = exp(-dt / tau_syn)``

Weights are stored as base64-encoded little-endian f32 arrays (ADR-0006).
The simulator reads ``alpha`` directly and never calls ``exp``.
"""

from __future__ import annotations

import base64
import json
import math
from datetime import UTC, datetime
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

# The format version string written into every artifact (ADR-0006).
_FORMAT_VERSION = "m2-draft"


def compile_model(
    model: nn.Module,
    path: str | Path,
    target: str = "sim",
    version: str = "0.2.0",
) -> None:
    """Compile *model* to a ``.thx`` artifact at *path*.

    Parameters
    ----------
    model:
        A :class:`thrindex.snn.Sequential` — the only supported container for M2.
    path:
        Output path.  The ``.thx`` extension is conventional but not enforced.
    target:
        Target identifier.  ``"sim"`` is the only valid value for M2.
    version:
        thrindex SDK version written into artifact metadata.
    """
    if not isinstance(model, Sequential):
        raise TypeError(
            f"compile_model expects a thrindex.snn.Sequential, got {type(model).__name__}"
        )

    layers_json: list[dict[str, Any]] = []
    for layer in model.layers:
        layers_json.append(_serialise_layer(layer))

    model_block = {"layers": layers_json}

    # Canonical model JSON (sorted keys, compact) — stored in metadata so that
    # both Python and Rust hash the SAME bytes without serde re-serialisation drift.
    model_canonical = json.dumps(model_block, sort_keys=True, separators=(",", ":"))
    crc = _crc32_hex(model_canonical.encode("utf-8"))

    artifact: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "thrindex_version": version,
        "target": target,
        "model": model_block,
        "metadata": {
            "compiled_at": datetime.now(UTC).isoformat(),
            # model_canonical is the exact bytes that were hashed.
            # The Rust loader reads this field and recomputes crc32 — no serde drift.
            "model_canonical": model_canonical,
            "crc32": crc,
        },
    }

    Path(path).write_text(json.dumps(artifact, indent=2), encoding="utf-8")


# ── Layer serialisers ──────────────────────────────────────────────────────────


def _serialise_layer(layer: nn.Module) -> dict[str, Any]:
    if isinstance(layer, ThxDense):
        return _serialise_dense(layer)
    if isinstance(layer, ThxConv2d):
        return _serialise_conv2d(layer)
    if isinstance(layer, LIF):
        return _serialise_lif(layer)
    raise TypeError(
        f"unsupported layer type for compile_model: {type(layer).__name__}. "
        "Only Dense, Conv2d, and LIF are supported in M2."
    )


def _serialise_dense(layer: ThxDense) -> dict[str, Any]:
    w = layer.linear.weight.detach().cpu()
    # nn.Linear.bias is typed as Parameter in PyTorch stubs but is None when
    # bias=False.  Explicit annotation avoids a spurious pyright comparison warning.
    b: Tensor | None = layer.linear.bias  # type: ignore[assignment]
    return {
        "type": "dense",
        "in_features": w.shape[1],
        "out_features": w.shape[0],
        "weights_b64": _to_b64(w),
        "bias_b64": _to_b64(b.detach().cpu()) if b is not None else None,
    }


def _serialise_lif(layer: LIF) -> dict[str, Any]:
    # Resolve derived constants at compile time — the simulator NEVER calls exp.
    # ADR-0005: alpha = exp(-dt / tau_mem), dt = 1.0 ms canonical.
    alpha: float = math.exp(-LIF.DT / layer.tau_mem)
    alpha_syn: float | None = (
        math.exp(-LIF.DT / layer.tau_syn) if layer.tau_syn is not None else None
    )
    return {
        "type": "lif",
        "threshold": float(layer.threshold),
        "alpha": alpha,
        "alpha_syn": alpha_syn,
        "reset": layer.reset,
    }


def _serialise_conv2d(layer: ThxConv2d) -> dict[str, Any]:
    conv = layer.conv
    w = conv.weight.detach().cpu()
    b: Tensor | None = conv.bias  # type: ignore[assignment]
    out_c, in_c, kh, kw = w.shape
    return {
        "type": "conv2d",
        "in_channels": in_c,
        "out_channels": out_c,
        "kernel_h": kh,
        "kernel_w": kw,
        "stride": list(conv.stride),
        "padding": list(conv.padding),
        "weights_b64": _to_b64(w),
        "bias_b64": _to_b64(b.detach().cpu()) if b is not None else None,
    }


# ── Encoding helpers ───────────────────────────────────────────────────────────


def _to_b64(t: Tensor) -> str:
    """Encode a tensor as base64 little-endian f32 bytes (ADR-0006)."""
    raw = t.to(dtype=torch.float32).contiguous().numpy().astype("<f4").tobytes()
    return base64.b64encode(raw).decode("ascii")


def _crc32_hex(data: bytes) -> str:
    """Return a CRC32 of *data* as an 8-character lowercase hex string.

    Uses ``zlib.crc32`` (stdlib), which returns the same value as the
    ``crc32fast`` Rust crate for the same input bytes.
    """
    import zlib

    value = zlib.crc32(data) & 0xFFFF_FFFF
    return f"{value:08x}"
