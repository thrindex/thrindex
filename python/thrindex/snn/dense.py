"""Dense (fully-connected) SNN layer.

A thin wrapper around :class:`torch.nn.Linear` that fits into a
:class:`~thrindex.snn.sequential.Sequential` temporal unrolling loop.
No internal recurrent state — the layer is called once per timestep with
the current spike (or current) input.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["Dense"]


class Dense(nn.Module):
    """Fully-connected layer compatible with SNN ``Sequential`` unrolling.

    Wraps :class:`~torch.nn.Linear` with no additional behaviour.  In a
    ``Sequential(Dense(...), LIF(...))`` chain the dense layer converts the
    incoming spike vector (or any continuous input) into a weighted current
    that the following ``LIF`` integrates into its membrane potential.

    Parameters
    ----------
    in_features:
        Size of each input sample.
    out_features:
        Size of each output sample.
    bias:
        If ``True`` (default), adds a learnable bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear transformation to the input tensor."""
        return self.linear(x)
