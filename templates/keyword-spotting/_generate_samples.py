"""Generate synthetic SHD-format spike train samples for the keyword-spotting demo.

Run ONCE from the repo root to regenerate:
    python templates/keyword-spotting/_generate_samples.py

This is a developer/maintainer script, not user-facing.
It writes three .json spike-train files to templates/keyword-spotting/samples/
and a placeholder model.thx with random (seeded) weights.

Dataset: Spiking Heidelberg Digits (SHD) format
- 700 cochlea channels (input neurons)
- 100 timesteps (1ms each, ADR-0005 canonical)
- 20 classes (digits 0-9, English + German)

The weights are Xavier-uniform initialised (seeded) — this is NOT a trained model.
Run templates/keyword-spotting/train.py to get real trained weights.
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


def _to_b64(tensor: torch.Tensor) -> str:
    raw = tensor.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    return base64.b64encode(raw).decode("ascii")


def _crc32_hex(data: bytes) -> str:
    value = zlib.crc32(data) & 0xFFFF_FFFF
    return f"{value:08x}"


# ── Generate 3 synthetic spike trains ─────────────────────────────────────────

N_IN = 700
T = 100
N_CLASSES = 20

samples_dir = Path(__file__).parent / "samples"

for sample_idx in range(3):
    g = torch.Generator()
    g.manual_seed(1000 + sample_idx)
    # SHD-style: temporal structure — higher spike density at start, then fading.
    t_indices = torch.arange(T, dtype=torch.float32)
    rate_envelope = torch.exp(-t_indices / 30.0) * 0.4 + 0.05  # ~5–45%
    spikes = torch.zeros(T, N_IN)
    for t in range(T):
        spikes[t] = torch.bernoulli(torch.full((N_IN,), rate_envelope[t].item()), generator=g)

    label = sample_idx % N_CLASSES  # synthetic label

    sample_data = {
        "label": label,
        "timesteps": T,
        "n_features": N_IN,
        # Store as list-of-lists (JSON); real pipeline uses binary .pt files.
        "spikes": spikes.tolist(),
    }
    path = samples_dir / f"sample_{sample_idx:03d}.json"
    path.write_text(json.dumps(sample_data, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {path}")

# ── Generate placeholder model.thx (random weights) ──────────────────────────

HIDDEN = 512

torch.manual_seed(0)
# Xavier uniform init — same scale as training would use.
w1 = torch.nn.init.xavier_uniform_(torch.empty(HIDDEN, N_IN))
b1 = torch.zeros(HIDDEN)
w2 = torch.nn.init.xavier_uniform_(torch.empty(N_CLASSES, HIDDEN))
b2 = torch.zeros(N_CLASSES)

TAU_MEM = 20.0
alpha = math.exp(-1.0 / TAU_MEM)

layers: list[dict] = [
    {
        "type": "dense",
        "in_features": N_IN,
        "out_features": HIDDEN,
        "weights_b64": _to_b64(w1),
        "bias_b64": _to_b64(b1),
    },
    {
        "type": "lif",
        "threshold": 1.0,
        "alpha": alpha,
        "alpha_syn": None,
        "reset": "subtract",
    },
    {
        "type": "dense",
        "in_features": HIDDEN,
        "out_features": N_CLASSES,
        "weights_b64": _to_b64(w2),
        "bias_b64": _to_b64(b2),
    },
    {
        "type": "lif",
        "threshold": 1.0,
        "alpha": alpha,
        "alpha_syn": None,
        "reset": "subtract",
    },
]

model_block: dict = {"layers": layers}
model_canonical = json.dumps(model_block, sort_keys=True, separators=(",", ":"))
crc = _crc32_hex(model_canonical.encode("utf-8"))

artifact = {
    "format_version": "m2-draft",
    "thrindex_version": "0.2.0",
    "target": "sim",
    "model": model_block,
    "metadata": {
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "model_canonical": model_canonical,
        "crc32": crc,
        "note": "PLACEHOLDER — random weights. Run train.py for benchmark weights.",
    },
}

thx_path = Path(__file__).parent / "model.thx"
thx_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
print(f"Wrote {thx_path}  ({thx_path.stat().st_size / 1024:.0f} KB)")
print("Done. Run train.py to replace model.thx with trained weights.")
