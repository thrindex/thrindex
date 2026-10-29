"""Spiking Heidelberg Digits.

The binning matches the keyword-spotting template: each recording is ~1.4 s,
spread across ``timesteps`` equal bins, into 700 cochlear channels. Train and
test are the files published by the Zenke lab. This function does not reshuffle
them.
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import torch
from torch import Tensor

from thrindex.datasets.types import SpikeDataset

__all__ = ["bin_shd_units", "load_shd"]

_URLS = {
    "train": "https://zenkelab.org/datasets/shd_train.h5.gz",
    "test": "https://zenkelab.org/datasets/shd_test.h5.gz",
}
_N_UNITS = 700
_MAX_TIME_S = 1.4


def bin_shd_units(
    times: Tensor,
    units: Tensor,
    *,
    timesteps: int,
    n_units: int = _N_UNITS,
    max_time_s: float = _MAX_TIME_S,
) -> Tensor:
    """Place one recording's spike times into ``timesteps`` bins.

    ``times`` is in seconds. ``units`` is the cochlear channel. A spike past
    the last bin is clipped into that bin. The result is ``[timesteps, n_units]``
    with values in ``{0, 1}``.
    """
    if timesteps < 1:
        raise ValueError(f"timesteps={timesteps} must be at least 1")
    frame = torch.zeros(timesteps, n_units)
    if times.numel() == 0:
        return frame
    dt = max_time_s / timesteps
    index = (times / dt).to(dtype=torch.long).clamp(0, timesteps - 1)
    channel = units.to(dtype=torch.long).clamp(0, n_units - 1)
    frame[index, channel] = 1.0
    return frame


def load_shd(
    root: str | Path,
    split: str = "train",
    *,
    timesteps: int = 100,
    max_time_s: float = _MAX_TIME_S,
    n_units: int = _N_UNITS,
) -> SpikeDataset:
    """Load one official SHD split.

    Parameters
    ----------
    root:
        Directory for the ``.h5`` files. Missing files are downloaded.
    split:
        ``"train"`` or ``"test"``.
    timesteps:
        Number of bins across ``max_time_s``. The keyword-spotting template
        uses 100.
    """
    if split not in _URLS:
        raise ValueError(f"split={split!r} is not an SHD split; use 'train' or 'test'")
    try:
        import h5py  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read SHD. Install it with: pip install 'thrindex[datasets]'"
        ) from exc

    directory = Path(root)
    path = _ensure_h5(directory, split)
    with h5py.File(path, "r") as handle:
        all_times = handle["spikes"]["times"][:]
        all_units = handle["spikes"]["units"][:]
        labels = torch.tensor(handle["labels"][:], dtype=torch.long)

    frames = torch.zeros(len(labels), timesteps, n_units)
    for i in range(len(labels)):
        frames[i] = bin_shd_units(
            torch.as_tensor(all_times[i], dtype=torch.float32),
            torch.as_tensor(all_units[i], dtype=torch.long),
            timesteps=timesteps,
            n_units=n_units,
            max_time_s=max_time_s,
        )
    return SpikeDataset(
        spikes=frames,
        labels=labels,
        split=split,
        timesteps=timesteps,
        source="shd",
    )


def _ensure_h5(directory: Path, split: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shd_{split}.h5"
    if path.exists():
        return path
    gz_path = directory / f"shd_{split}.h5.gz"
    urllib.request.urlretrieve(_URLS[split], gz_path)  # noqa: S310 — the URL is a fixed table
    with gzip.open(gz_path, "rb") as gz_file:
        path.write_bytes(gz_file.read())
    gz_path.unlink()
    return path
