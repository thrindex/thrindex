"""Generate synthetic N-MNIST-shaped spike-train samples for the event-camera demo.

Run ONCE from the repo root to regenerate:
    python templates/event-camera/_generate_samples.py

This is a developer/maintainer script, not user-facing.
It writes three .json spike-train files to templates/event-camera/samples/
and a placeholder model.thx with random (seeded) weights.

N-MNIST format recap
---------------------
- 34 × 34 pixels, 2 polarities (ON / OFF)
- Each spatial frame is flattened to a 2312-element spike vector
- T = 20 timesteps (each 2 ms → covers the 40 ms N-MNIST recording window)

The weights are Xavier-uniform initialised (seeded) — NOT a trained model.
Run train.py to obtain benchmark-quality weights and a real model.thx.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

H, W, C = 34, 34, 2       # N-MNIST sensor dimensions
N_IN = C * H * W           # 2312
T = 20                     # time bins (2 ms each)
HIDDEN1 = 1024
HIDDEN2 = 256
N_CLASSES = 10
TAU_MEM = 20.0

SAMPLES_DIR = Path(__file__).parent / "samples"

# ── Helpers ────────────────────────────────────────────────────────────────────


def _to_b64(tensor: torch.Tensor) -> str:
    raw = tensor.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    return base64.b64encode(raw).decode("ascii")


def _pack_f32(values: list[float]) -> str:
    raw = struct.pack(f"<{len(values)}f", *values)
    return base64.b64encode(raw).decode("ascii")


# ── Generate 3 synthetic spike trains ─────────────────────────────────────────

# Synthetic N-MNIST-like event statistics:
#   - ON events (polarity 0): concentrated in the first half of the recording,
#     spatial density follows a Gaussian blob (simulates a saccade over a digit edge).
#   - OFF events (polarity 1): concentrated in the second half, sparser.
#   - Overall sparsity: ~1–3% per frame per polarity, matching real N-MNIST statistics.

for sample_idx in range(3):
    g = torch.Generator()
    g.manual_seed(2000 + sample_idx)

    # Simulate a digit-like activation blob: Gaussian weight map over the 34×34 grid
    cx = 10 + (sample_idx * 7) % 20     # blob centre x (varies per sample)
    cy = 10 + (sample_idx * 5) % 20     # blob centre y
    xs = torch.arange(W, dtype=torch.float32).unsqueeze(0)   # [1, W]
    ys = torch.arange(H, dtype=torch.float32).unsqueeze(1)   # [H, 1]
    spatial_weight = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 8.0 ** 2))
    # [H, W], values in (0, 1]

    frames = torch.zeros(T, C, H, W)
    for t in range(T):
        # ON polarity: active early in the saccade
        on_rate = spatial_weight * 0.15 * max(0.0, 1.0 - t / (T * 0.6))
        # OFF polarity: trails behind
        off_rate = spatial_weight * 0.08 * max(0.0, t / T - 0.3)
        frames[t, 0] = torch.bernoulli(on_rate.clamp(0, 1), generator=g)
        frames[t, 1] = torch.bernoulli(off_rate.clamp(0, 1), generator=g)

    # Flatten spatial: [T, C*H*W]
    spikes_flat = frames.reshape(T, N_IN)

    label = sample_idx  # digits 0, 1, 2

    sample = {
        "label": label,
        "label_name": str(label),
        "timesteps": T,
        "n_features": N_IN,
        "source": "synthetic — N-MNIST-shaped (2 pol × 34×34, T=20). "
                  "Run train.py for real N-MNIST benchmark.",
        "spikes": spikes_flat.tolist(),
    }
    out_path = SAMPLES_DIR / f"sample_{sample_idx:03d}.json"
    out_path.write_text(json.dumps(sample, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out_path}  (sparsity: {spikes_flat.mean():.3f})")

# ── Generate placeholder model.thx (random weights) ──────────────────────────

torch.manual_seed(0)
w1 = torch.nn.init.xavier_uniform_(torch.empty(HIDDEN1, N_IN))
b1 = torch.zeros(HIDDEN1)
w2 = torch.nn.init.xavier_uniform_(torch.empty(HIDDEN2, HIDDEN1))
b2 = torch.zeros(HIDDEN2)
w3 = torch.nn.init.xavier_uniform_(torch.empty(N_CLASSES, HIDDEN2))
b3 = torch.zeros(N_CLASSES)

alpha = math.exp(-1.0 / TAU_MEM)

layers: list[dict] = [
    {
        "type": "dense",
        "in_features": N_IN,
        "out_features": HIDDEN1,
        "weights_b64": _to_b64(w1),
        "bias_b64": _to_b64(b1),
    },
    {"type": "lif", "threshold": 0.5, "alpha": alpha, "alpha_syn": None, "reset": "subtract"},
    {
        "type": "dense",
        "in_features": HIDDEN1,
        "out_features": HIDDEN2,
        "weights_b64": _to_b64(w2),
        "bias_b64": _to_b64(b2),
    },
    {"type": "lif", "threshold": 0.5, "alpha": alpha, "alpha_syn": None, "reset": "subtract"},
    {
        "type": "dense",
        "in_features": HIDDEN2,
        "out_features": N_CLASSES,
        "weights_b64": _to_b64(w3),
        "bias_b64": _to_b64(b3),
    },
    {"type": "lif", "threshold": 0.5, "alpha": alpha, "alpha_syn": None, "reset": "subtract"},
]

import zlib
from datetime import datetime, timezone

model_block: dict = {"layers": layers}
canonical = json.dumps(model_block, sort_keys=True, separators=(",", ":"))
crc = f"{zlib.crc32(canonical.encode()) & 0xFFFF_FFFF:08x}"

artifact = {
    "format_version": "m2-draft",
    "thrindex_version": "0.3.0",
    "target": "sim",
    "model": model_block,
    "metadata": {
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "model_canonical": canonical,
        "crc32": crc,
        "note": "PLACEHOLDER — random weights. Run train.py for benchmark weights.",
    },
}

thx_path = Path(__file__).parent / "model.thx"
thx_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
print(f"Wrote {thx_path}  ({thx_path.stat().st_size / 1024:.0f} KB)")
print("Done. Run train.py to replace model.thx with trained weights (≥ 95.0%).")
