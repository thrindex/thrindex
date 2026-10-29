"""Event-camera datasets via tonic.

N-MNIST and DVS Gesture have a published train/test cut. CIFAR10-DVS does not:
tonic ships the whole set, and :func:`load_cifar10_dvs` only accepts
``split="all"``. Frames are ``[N, T, C, H, W]`` so a convolution can read them.
A dense model flattens the last three axes.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from thrindex.datasets.types import SpikeDataset

__all__ = ["load_cifar10_dvs", "load_dvs_gesture", "load_nmnist"]


def load_nmnist(
    root: str | Path,
    split: str = "train",
    *,
    timesteps: int = 20,
) -> SpikeDataset:
    """N-MNIST, Orchard et al. 2015. Sensor ``(34, 34, 2)``."""
    return _load_tonic("NMNIST", root, split, timesteps=timesteps, source="nmnist")


def load_dvs_gesture(
    root: str | Path,
    split: str = "train",
    *,
    timesteps: int = 20,
) -> SpikeDataset:
    """DVS Gesture, Amir et al. 2017. Sensor ``(128, 128, 2)``."""
    return _load_tonic("DVSGesture", root, split, timesteps=timesteps, source="dvs_gesture")


def load_cifar10_dvs(
    root: str | Path,
    split: str = "all",
    *,
    timesteps: int = 20,
) -> SpikeDataset:
    """CIFAR10-DVS, Li et al. 2017. Sensor ``(128, 128, 2)``.

    There is no official train/test split in the tonic distribution.
    ``split`` must be ``"all"``.
    """
    if split != "all":
        raise ValueError(
            f"split={split!r} is not a CIFAR10-DVS split. "
            "The published files have no train/test cut; use split='all' and "
            "hold out rows yourself."
        )
    return _load_tonic("CIFAR10DVS", root, split, timesteps=timesteps, source="cifar10_dvs")


def tonic_frame_to_torch(sample: Tensor) -> Tensor:
    """Turn a tonic ``ToFrame`` array into ``[T, C, H, W]``.

    ``ToFrame`` documents its layout as ``(time, polarity, x, y)``.
    ``sensor_size`` is ``(x, y, p)``. ``snn.Conv2d`` reads ``(batch, channel, height, width)``,
    so x and y are swapped here. Values above 1 are clamped: several events in
    one bin are one spike, which is what the event-camera template trained with.
    """
    if sample.ndim != 4:
        raise RuntimeError(
            f"tonic frame has shape {tuple(sample.shape)}; expected [T, C, W, H]"
        )
    return sample.clamp(0.0, 1.0).permute(0, 1, 3, 2).contiguous()


def _load_tonic(
    class_name: str,
    root: str | Path,
    split: str,
    *,
    timesteps: int,
    source: str,
) -> SpikeDataset:
    if timesteps < 1:
        raise ValueError(f"timesteps={timesteps} must be at least 1")
    try:
        import tonic  # pyright: ignore[reportMissingImports]
        import tonic.transforms as transforms  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "tonic is required for event-camera datasets. "
            "Install it with: pip install 'thrindex[datasets]'"
        ) from exc

    dataset_cls = getattr(tonic.datasets, class_name)
    sensor = tuple(dataset_cls.sensor_size)
    frame = transforms.ToFrame(sensor_size=sensor, n_time_bins=timesteps)
    kwargs: dict[str, Any] = {"save_to": str(root), "transform": frame}
    parameters = inspect.signature(dataset_cls.__init__).parameters
    if "train" in parameters:
        if split not in ("train", "test"):
            raise ValueError(f"split={split!r} must be 'train' or 'test' for {class_name}")
        kwargs["train"] = split == "train"
    dataset = dataset_cls(**kwargs)

    frames: list[Tensor] = []
    labels: list[int] = []
    for index in range(len(dataset)):
        sample, label = dataset[index]
        tensor = tonic_frame_to_torch(torch.as_tensor(sample, dtype=torch.float32))
        frames.append(tensor)
        labels.append(int(label))
    if not frames:
        spikes = torch.zeros(0, timesteps, sensor[2], sensor[1], sensor[0])
    else:
        spikes = torch.stack(frames, dim=0)
    return SpikeDataset(
        spikes=spikes,
        labels=torch.tensor(labels, dtype=torch.long),
        split=split,
        timesteps=timesteps,
        source=source,
    )
