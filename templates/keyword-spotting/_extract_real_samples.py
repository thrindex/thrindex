"""Extract 3 real SHD test samples to replace the synthetic placeholders.

Run once from the repo root after the training run:
    uv run python templates/keyword-spotting/_extract_real_samples.py

Requires /tmp/shd/shd_test.h5 (produced by train.py --data-dir /tmp/shd).

Dataset: Spiking Heidelberg Digits (SHD)
  Cramer et al. 2020, IEEE TNNLS, doi:10.1109/TNNLS.2020.3044364
  License: CC BY 4.0  https://zenkelab.org/resources/spiking-heidelberg-datasets/

Samples selected:
  sample_000: first test sample from class 0  (digit "zero", English)
  sample_001: first test sample from class 7  (digit "seven", English)
  sample_002: first test sample from class 14 (digit "four",  German)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    raise SystemExit("h5py required: uv add h5py")

N_IN = 700
T = 100
MAX_TIME = 1.4  # seconds — SHD recording length
DT = MAX_TIME / T

HDF5_PATH = Path("/tmp/shd/shd_test.h5")
SAMPLES_DIR = Path(__file__).parent / "samples"
TARGET_LABELS = [0, 7, 14]

if not HDF5_PATH.exists():
    raise SystemExit(f"{HDF5_PATH} not found — run train.py --data-dir /tmp/shd first.")

with h5py.File(HDF5_PATH, "r") as f:
    spike_times = f["spikes"]["times"][:]
    spike_units = f["spikes"]["units"][:]
    labels = f["labels"][:]

for out_idx, target in enumerate(TARGET_LABELS):
    candidates = np.where(labels == target)[0]
    if len(candidates) == 0:
        raise RuntimeError(f"No test sample found for label {target}")
    src_idx = int(candidates[0])

    spikes: list[list[float]] = [[0.0] * N_IN for _ in range(T)]
    t_idx = (spike_times[src_idx] / DT).astype("int64").clip(0, T - 1)
    u_idx = spike_units[src_idx].astype("int64").clip(0, N_IN - 1)
    for t, u in zip(t_idx, u_idx):
        spikes[t][u] = 1.0

    sample = {
        "label": int(labels[src_idx]),
        "timesteps": T,
        "n_features": N_IN,
        "spikes": spikes,
        "source": (
            "SHD test set — Cramer et al. 2020 (doi:10.1109/TNNLS.2020.3044364), "
            "CC BY 4.0, https://zenkelab.org/resources/spiking-heidelberg-datasets/"
        ),
    }

    path = SAMPLES_DIR / f"sample_{out_idx:03d}.json"
    path.write_text(json.dumps(sample, separators=(",", ":")), encoding="utf-8")
    n_spikes = sum(sum(row) for row in spikes)
    print(f"Wrote {path}  label={labels[src_idx]}  spikes={int(n_spikes)}")

print("Done. Commit templates/keyword-spotting/samples/ to replace synthetic placeholders.")
