"""Return type shared by the dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

__all__ = ["SpikeDataset"]


@dataclass(frozen=True)
class SpikeDataset:
    """One split of a spike dataset.

    ``spikes`` is ``[N, T, ...]`` with time as the second axis, which is the
    layout the Heidelberg files and the tonic frame transform produce. Training
    code permutes that to ``[T, N, ...]`` before :class:`thrindex.snn.Sequential`.

    ``split`` is the published split name. CIFAR10-DVS has no official
    train/test cut, so its split is ``"all"``.
    """

    spikes: Tensor
    labels: Tensor
    split: str
    timesteps: int
    source: str
