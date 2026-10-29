"""Dense (fully-connected) SNN layer.

A thin wrapper around :class:`torch.nn.Linear` that fits into a
:class:`~thrindex.snn.sequential.Sequential` temporal unrolling loop.
The layer itself holds no recurrent buffer. A non-zero ``delay`` asks
``Sequential`` to feed it the input from that many timesteps ago, the same
shift the simulator applies after compile.
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
    delay:
        Synaptic delay in timesteps, applied to every connection. ``0`` stores
        no delay bytes in the artifact. The value must fit in a ``u16``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        delay: int = 0,
    ) -> None:
        super().__init__()
        if delay < 0 or delay > 65535:
            raise ValueError(
                f"delay={delay} must be an integer timestep in 0..=65535. "
                "The artifact stores each connection delay as a u16."
            )
        self.delay = delay
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear transformation to the input tensor."""
        return self.linear(x)
