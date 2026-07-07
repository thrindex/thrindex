"""SNN-compatible 2-D convolutional layer.

Wraps :class:`torch.nn.Conv2d` for use inside a
:class:`~thrindex.snn.sequential.Sequential` temporal unrolling loop.
Operates on a single spatial frame ``[batch, C, H, W]`` per timestep; the
``LIF`` layer that follows applies element-wise spiking across the feature map.

Note: when chaining ``Conv2d → LIF → Dense``, a spatial-to-feature flattening
step is needed between the ``LIF`` output and ``Dense`` input (e.g., add a
:class:`torch.nn.Flatten` layer in the ``Sequential``).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["Conv2d"]


class Conv2d(nn.Module):
    """2-D convolutional layer compatible with SNN ``Sequential`` unrolling.

    Parameters
    ----------
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias:
        Forwarded verbatim to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution to a single spatial frame ``[batch, C, H, W]``."""
        return self.conv(x)
