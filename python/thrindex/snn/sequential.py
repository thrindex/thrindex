"""Temporal unrolling container for SNN layers.

``Sequential`` wraps a list of layers and unrolls them across the time dimension
of the input.  It owns and threads the recurrent ``LIFState`` objects so that
neither the caller nor the ``LIF`` modules need to manage state explicitly.

State is initialized to zeros lazily on the first timestep of each forward call
(shape inference is automatic) and reset to zeros at the start of every new
``forward`` call.  See ``adr/0004-lif-state-ownership.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from thrindex.snn.lif import LIF, LIFState

__all__ = ["Sequential"]


class Sequential(nn.Module):
    """Temporally-unrolled SNN container.

    Accepts any sequence of :class:`~torch.nn.Module` layers and unrolls them
    across the time axis ``T`` of the input tensor.  ``LIF`` layers are
    identified at runtime; their state is initialized to zeros and threaded
    across timesteps.  All other layers are called without state.

    Parameters
    ----------
    *layers:
        Modules to apply in order at each timestep.  ``LIF`` layers are handled
        specially (state threading); everything else is called with ``layer(x)``.

    Input shape:
        ``[T, batch, *]`` — time-first convention.

    Output shape:
        ``[T, batch, *]`` — the spike output of the last layer at each timestep.

    Example::

        import thrindex.snn as snn

        model = snn.Sequential(
            snn.Dense(784, 1000),
            snn.LIF(threshold=1.0, tau_mem=5.0),
            snn.Dense(1000, 10),
            snn.LIF(threshold=1.0, tau_mem=5.0),
        )
        spikes = model(encoded_input)  # [T, batch, 10]
    """

    def __init__(self, *layers: nn.Module) -> None:
        super().__init__()
        # Register as a ModuleList so parameters are discovered by optimizers.
        self._layers = nn.ModuleList(layers)

    # Expose layers as a property so tests and repr can iterate them.
    @property
    def layers(self) -> Sequence[nn.Module]:
        return list(self._layers)

    def forward(self, x: Tensor) -> Tensor:
        """Unroll all layers across the ``T`` time steps of ``x``.

        Parameters
        ----------
        x:
            Input tensor, shape ``[T, batch, *]``.

        Returns
        -------
        Tensor
            Output of the last layer at each timestep, shape ``[T, batch, *]``.
        """
        T = x.shape[0]

        # LIF state indexed by layer position.  Initialized lazily on t=0.
        lif_states: dict[int, LIFState] = {}

        spikes_out: list[Tensor] = []
        for t in range(T):
            h: Tensor = x[t]
            for i, layer in enumerate(self._layers):
                if isinstance(layer, LIF):
                    if i not in lif_states:
                        # Lazy init: zero membrane with the same shape as the
                        # current feature tensor (inferred from the layer before it).
                        lif_states[i] = LIFState(
                            mem=torch.zeros_like(h),
                            syn=(
                                torch.zeros_like(h)
                                if layer.tau_syn is not None
                                else None
                            ),
                        )
                    spk, lif_states[i] = layer(h, lif_states[i])
                    h = spk
                else:
                    h = layer(h)  # type: ignore[assignment]
            spikes_out.append(h)

        return torch.stack(spikes_out, dim=0)

    def extra_repr(self) -> str:
        lines = [f"  ({i}): {layer}" for i, layer in enumerate(self._layers)]
        return "\n" + "\n".join(lines) + "\n"
