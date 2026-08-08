#!/usr/bin/env python3
"""
crates/thrindex-backends/akida/python/akida_compile.py

Tier-1 compiler: converts a .thx JSON artifact into a BrainChip AKD1500
hardware program (.fbz byte payload).

Usage:
    python akida_compile.py <artifact.thx> <output.fbz>

Requirements:
    pip install akida numpy
    Python 3.10-3.12 (akida 2.x package constraint), aarch64 or x86_64 Linux

Confirmed working API (spike test, aarch64, Python 3.11.9, akida 2.19.2):
    from akida import Model, AKD1500
    from akida.layers.input_data import InputData
    from akida.layers.fully_connected import FullyConnected
    InputData(input_shape=(1,1,N), input_bits=4)
    FullyConnected(units=M, weights_bits=4, act_bits=4)
    layer.set_variable("weights", W_int8)   # shape (1,1,in,out), values in [-7,7]
    model.map(AKD1500(), hw_only=True)
    model.to_buffer()                        # returns bytes → write to .fbz

Quantization (4-bit signed, per-tensor):
    scale = max(|W_f32|) / 7   (not /127 — AKD1500 constraint confirmed on Pi)
    W_int8 = clip(round(W_f32 / scale * 7), -7, 7).astype(int8)
    quantization_error_per_weight ≤ scale/2

See ADR-0011 / RFC-004 for full design rationale, rejection policy, and
conformance exclusion details.
"""
from __future__ import annotations

import json
import sys
import base64
import struct
from pathlib import Path

import numpy as np

try:
    from akida import Model, AKD1500
    from akida.layers.input_data import InputData
    from akida.layers.fully_connected import FullyConnected
except ImportError as e:
    print(f"ERROR: akida package not available: {e}", file=sys.stderr)
    print("Install: pip install akida  (Python 3.10-3.12, Linux only)", file=sys.stderr)
    sys.exit(1)


# ── Constants (ADR-0011) ──────────────────────────────────────────────────────

AKD1500_TARGET = "akida-akd1500"
# 4-bit signed symmetric range [-7, 7]; int8 container.
WEIGHT_MAX_INT4 = 7


# ── Weight quantization ───────────────────────────────────────────────────────

def quantize_f32_to_int4(w_f32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize f32 weight matrix to 4-bit signed values in an int8 container.

    4-bit signed range: [-7, 7] (symmetric, avoids -8 to keep scale uniform).
    Quantization scale: scale = max(|W|) / 7
    Quantization error per weight: ≤ scale / 2  (~7× coarser than int8).

    Returns:
        (W_int8, scale): int8 array with values in [-7, 7], and the scale factor.
    """
    max_abs = float(np.max(np.abs(w_f32)))
    if max_abs == 0.0:
        return np.zeros_like(w_f32, dtype=np.int8), 1.0
    scale = max_abs / WEIGHT_MAX_INT4
    w_scaled = np.round(w_f32 / scale * WEIGHT_MAX_INT4)
    w_clipped = np.clip(w_scaled, -WEIGHT_MAX_INT4, WEIGHT_MAX_INT4)
    return w_clipped.astype(np.int8), scale


# ── .thx parsing ─────────────────────────────────────────────────────────────

def decode_f32_b64(b64_str: str) -> np.ndarray:
    """Decode a base64 LE-f32 weight blob to a numpy float32 array."""
    raw = base64.b64decode(b64_str)
    n = len(raw) // 4
    return np.array(struct.unpack_from(f"<{n}f", raw), dtype=np.float32)


def compile_artifact(artifact_json: str) -> bytes:
    """Convert a .thx JSON artifact string to an AKD1500 .fbz byte payload.

    Raises:
        ValueError: if the artifact is not compatible with AKD1500.
        RuntimeError: if model.map() or model.to_buffer() fails.
    """
    artifact = json.loads(artifact_json)

    # Target check (E0404).
    target = artifact.get("target", "")
    if target != AKD1500_TARGET:
        raise ValueError(
            f"E0404: artifact compiled for target='{target}', not '{AKD1500_TARGET}'. "
            f"Recompile with --target {AKD1500_TARGET}."
        )

    layers_raw = artifact.get("model", {}).get("layers", [])

    # Pre-flight: reject LIF (E0401) and delays (E0402) immediately.
    for i, layer in enumerate(layers_raw):
        ltype = layer.get("type", "")
        if ltype == "lif":
            raise ValueError(
                f"E0401: layer[{i}] type='lif' cannot be compiled for AKD1500. "
                f"AKD1500 implements bounded ReLU, not LIF. "
                f"Use the 'sim' backend for SNN simulation."
            )
        if layer.get("delays_b64") or layer.get("delays_sparse"):
            enc = layer.get("delays_encoding", "present")
            raise ValueError(
                f"E0402: layer[{i}] has '{enc}' delays; AKD1500 (Akida 1.0) has no TNP. "
                f"Retrain without delays or use a backend that supports them."
            )

    # Build Akida model.
    model = Model()

    # InputData layer: declare spatial shape and 4-bit input encoding.
    # AKD1500 constraint: FullyConnected only accepts 4-bit inputs (input_bits=4).
    first_dense = next((l for l in layers_raw if l.get("type") == "dense"), None)
    if first_dense is None:
        raise ValueError(
            "E0405: no Dense layers in artifact. AKD1500 requires at least one Dense layer."
        )
    in_features = int(first_dense["in_features"])
    model.add(InputData(input_shape=(1, 1, in_features), input_bits=4))

    # Build each Dense layer.
    for i, layer in enumerate(layers_raw):
        ltype = layer.get("type", "")
        if ltype != "dense":
            raise ValueError(
                f"E0406: layer[{i}] type='{ltype}' is not supported on AKD1500. "
                f"Only 'dense' layers are supported in the current implementation. "
                f"Conv2d support is planned for a future item in the build sequence."
            )

        out_features = int(layer["out_features"])
        cur_in = int(layer["in_features"])

        # Quantize f32 weights → int8 container for 4-bit values.
        w_f32 = decode_f32_b64(layer["weights_b64"]).reshape(out_features, cur_in)
        w_int8, scale = quantize_f32_to_int4(w_f32)

        print(
            f"  layer[{i}] dense: in={cur_in} out={out_features} "
            f"scale={scale:.6f} max_abs_w={float(np.max(np.abs(w_f32))):.6f}",
            file=sys.stderr,
        )

        akida_layer = FullyConnected(
            units=out_features,
            name=f"dense_{i}",
            weights_bits=4,
            act_bits=4,
        )
        # Reshape to Akida expected shape: (1, 1, in_features, out_features).
        model.add(akida_layer)
        akida_layer.set_variable("weights", w_int8.T.reshape(1, 1, cur_in, out_features))

    # Map to virtual AKD1500 and serialise.
    try:
        model.map(AKD1500(), hw_only=True)
    except Exception as e:
        raise RuntimeError(
            f"model.map() failed: {type(e).__name__}: {e}. "
            f"Check that layer sizes are within AKD1500 NP capacity."
        ) from e

    try:
        fbz_bytes = model.to_buffer()
    except Exception as e:
        raise RuntimeError(f"model.to_buffer() failed: {type(e).__name__}: {e}") from e

    return fbz_bytes


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: akida_compile.py <artifact.thx> <output.fbz>", file=sys.stderr)
        sys.exit(2)

    artifact_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        artifact_json = artifact_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"ERROR: cannot read artifact: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Compiling {artifact_path} → {output_path} ...", file=sys.stderr)
    try:
        fbz = compile_artifact(artifact_json)
    except (ValueError, RuntimeError) as e:
        print(f"COMPILE ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    output_path.write_bytes(fbz)
    print(f"OK: wrote {len(fbz)} bytes to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
