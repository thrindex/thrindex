"""Spike datasets.

Loaders return tensors. They do not train. SHD is read from the Heidelberg
``.h5`` files. The event-camera sets are read through ``tonic``, which is an
optional install::

    pip install "thrindex[datasets]"

The base ``pip install thrindex`` stays limited to PyTorch.
"""

from thrindex.datasets.events import load_cifar10_dvs, load_dvs_gesture, load_nmnist
from thrindex.datasets.shd import bin_shd_units, load_shd
from thrindex.datasets.types import SpikeDataset

__all__ = [
    "SpikeDataset",
    "bin_shd_units",
    "load_cifar10_dvs",
    "load_dvs_gesture",
    "load_nmnist",
    "load_shd",
]
