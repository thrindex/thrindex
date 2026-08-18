"""``thrindex.snn.Flatten`` — spatial-to-feature flatten layer.

Subclasses :class:`torch.nn.Flatten` so all PyTorch behaviour (forward pass,
state dict, parameter iteration) is inherited without reimplementation.

Compile support: :func:`thrindex.compile.compile_model` recognises any
:class:`torch.nn.Flatten` instance (including this subclass) and serialises it
as a ``flatten`` IR node.  The Rust compiler propagates it into the ``.thx``
artifact.  The simulator treats it as a no-op until spatial shape tracking is
implemented.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["Flatten"]


class Flatten(nn.Flatten):
    """SNN-compatible flatten layer.

    Identical to :class:`torch.nn.Flatten` in every respect.  Provided so that
    users can write::

        import thrindex.snn as snn

        model = snn.Sequential(
            snn.Conv2d(1, 16, 3),
            snn.LIF(tau_mem=10.0),
            snn.Flatten(),                  # flattens [batch, 16, H, W] → [batch, 16*H*W]
            snn.Dense(16 * H * W, 10),
            snn.LIF(tau_mem=10.0),
        )

    without importing from ``torch.nn`` directly.

    Parameters
    ----------
    start_dim:
        First dimension to flatten (default 1 = first non-batch dim).
    end_dim:
        Last dimension to flatten inclusive (default -1 = last dim).
    """
