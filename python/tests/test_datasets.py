"""Binning that does not download a dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from thrindex.datasets import bin_shd_units, load_shd
from thrindex.datasets.events import load_cifar10_dvs, tonic_frame_to_torch


def test_shd_bin_places_a_spike_in_the_right_channel() -> None:
    # 1.4 s across 10 bins → bin width 0.14 s. t=0.14 is the start of bin 1.
    frame = bin_shd_units(
        torch.tensor([0.14]),
        torch.tensor([3]),
        timesteps=10,
        n_units=8,
        max_time_s=1.4,
    )
    assert frame.shape == (10, 8)
    assert frame[1, 3].item() == 1.0
    assert frame.sum().item() == 1.0


def test_shd_bin_clips_a_spike_past_the_window() -> None:
    frame = bin_shd_units(
        torch.tensor([5.0]),
        torch.tensor([0]),
        timesteps=4,
        n_units=2,
    )
    assert frame[3, 0].item() == 1.0


def test_load_shd_reads_the_hdf5_layout(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    import numpy as np  # pyright: ignore[reportMissingImports]

    path = tmp_path / "shd_train.h5"
    times = [np.array([0.0, 0.14], dtype=np.float32), np.array([], dtype=np.float32)]
    units = [np.array([3, 3], dtype=np.int64), np.array([], dtype=np.int64)]
    with h5py.File(path, "w") as handle:
        time_type = h5py.vlen_dtype(np.float32)
        unit_type = h5py.vlen_dtype(np.int64)
        handle.create_dataset("spikes/times", data=times, dtype=time_type)
        handle.create_dataset("spikes/units", data=units, dtype=unit_type)
        handle.create_dataset("labels", data=np.array([4, 1], dtype=np.int64))

    loaded = load_shd(tmp_path, "train", timesteps=10, n_units=8, max_time_s=1.4)
    assert loaded.split == "train"
    assert loaded.labels.tolist() == [4, 1]
    assert loaded.spikes.shape == (2, 10, 8)
    assert loaded.spikes[0, 0, 3].item() == 1.0
    assert loaded.spikes[0, 1, 3].item() == 1.0
    assert loaded.spikes[1].sum().item() == 0.0


def test_tonic_frame_swaps_width_and_height() -> None:
    # ToFrame layout is (T, C, W, H). A 2-wide, 3-tall image.
    frame = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    out = tonic_frame_to_torch(frame)
    assert out.shape == (1, 1, 3, 2)
    assert out[0, 0, 1, 0].item() == frame[0, 0, 0, 1].item()


def test_cifar10_dvs_refuses_a_fake_split() -> None:
    with pytest.raises(ValueError, match="all"):
        load_cifar10_dvs("/tmp/does-not-download", split="train")
