"""Generate synthetic sensor-data samples for the anomaly-detection demo.

Run ONCE from the repo root to regenerate:
    python templates/anomaly-detection/_generate_samples.py

This is a developer/maintainer script, not user-facing.
It writes three .json files to templates/anomaly-detection/samples/:
  - sample_000.json  — normal operating point
  - sample_001.json  — Fault A (bearing wear: channels 0–31 elevated)
  - sample_002.json  — Fault B (overheating: channels 48–79 elevated)

And a placeholder model.thx with random (seeded) weights.
Run train.py to obtain benchmark-quality weights and a real model.thx.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

N_IN = 128
HIDDEN1 = 256
HIDDEN2 = 64
N_CLASSES = 2
T = 100
TAU_MEM = 10.0

SAMPLES_DIR = Path(__file__).parent / "samples"

# ── Helpers ────────────────────────────────────────────────────────────────────


def _to_b64(tensor: torch.Tensor) -> str:
    raw = tensor.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    return base64.b64encode(raw).decode("ascii")


def _tanh_normalise(x: torch.Tensor) -> torch.Tensor:
    """Squash to [0, 1] using tanh — same normalisation as train.py."""
    return (x.tanh() + 1.0) / 2.0


# ── Generate 3 samples ────────────────────────────────────────────────────────

SAMPLE_SPECS = [
    {
        "label": 0,
        "label_name": "normal",
        "description": "Normal operating point — all channels centred near zero.",
        "fn": lambda g: torch.randn(N_IN, generator=g) * 0.8,
    },
    {
        "label": 1,
        "label_name": "anomaly — Fault A (bearing wear)",
        "description": "Channels 0–31 elevated by +3σ; simulates early bearing wear. "
                       "After tanh normalisation: rate ≈ 0.995 vs 0.50 for normal.",
        "fn": lambda g: (
            lambda d: (d.__setitem__(slice(0, 32), d[:32] + 3.0), d)[1]
        )(torch.randn(N_IN, generator=g) * 0.8),
    },
    {
        "label": 1,
        "label_name": "anomaly — Fault B (overheating)",
        "description": "Channels 48–79 elevated by +5σ; simulates thermal runaway. "
                       "After tanh normalisation: rate ≈ 0.9999 vs 0.50 for normal.",
        "fn": lambda g: (
            lambda d: (d.__setitem__(slice(48, 80), d[48:80] + 5.0), d)[1]
        )(torch.randn(N_IN, generator=g) * 0.8),
    },
]

for i, spec in enumerate(SAMPLE_SPECS):
    g = torch.Generator()
    g.manual_seed(3000 + i)
    raw: torch.Tensor = spec["fn"](g)  # type: ignore[operator]
    normed = _tanh_normalise(raw)      # [N_IN], values in [0, 1]

    # Rate-encode to spike train: [T, N_IN]
    enc_g = torch.Generator()
    enc_g.manual_seed(3100 + i)
    spikes = torch.bernoulli(
        normed.unsqueeze(0).expand(T, -1), generator=enc_g
    )

    sample = {
        "label": spec["label"],
        "label_name": spec["label_name"],
        "description": spec["description"],
        "timesteps": T,
        "n_features": N_IN,
        "source": "synthetic — run train.py for benchmark weights.",
        "raw_sensor": normed.tolist(),    # [N_IN] — normalised readings
        "spikes": spikes.tolist(),         # [T, N_IN] — rate-encoded
    }
    out_path = SAMPLES_DIR / f"sample_{i:03d}.json"
    out_path.write_text(json.dumps(sample, separators=(",", ":")), encoding="utf-8")
    sparsity = spikes.mean().item()
    print(f"Wrote {out_path}  label={spec['label']} ({spec['label_name']})  sparsity={sparsity:.3f}")

# ── Placeholder model.thx ─────────────────────────────────────────────────────

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
    {"type": "lif", "threshold": 1.0, "alpha": alpha, "alpha_syn": None, "reset": "subtract"},
    {
        "type": "dense",
        "in_features": HIDDEN1,
        "out_features": HIDDEN2,
        "weights_b64": _to_b64(w2),
        "bias_b64": _to_b64(b2),
    },
    {"type": "lif", "threshold": 1.0, "alpha": alpha, "alpha_syn": None, "reset": "subtract"},
    {
        "type": "dense",
        "in_features": HIDDEN2,
        "out_features": N_CLASSES,
        "weights_b64": _to_b64(w3),
        "bias_b64": _to_b64(b3),
    },
    {"type": "lif", "threshold": 1.0, "alpha": alpha, "alpha_syn": None, "reset": "subtract"},
]

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
        "note": "PLACEHOLDER — random weights. Run train.py for benchmark weights (AUROC ≥ 0.90).",
    },
}

thx_path = Path(__file__).parent / "model.thx"
thx_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
print(f"Wrote {thx_path}  ({thx_path.stat().st_size / 1024:.0f} KB)")
print("Done. Run train.py to replace model.thx with trained weights.")
