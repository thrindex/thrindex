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
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from thrindex.snn.dense import Dense
from thrindex.snn.lif import LIF, LIFState

__all__ = ["RecordedForward", "Sequential"]


@dataclass
class RecordedForward:
    """Output of :meth:`Sequential.forward_recorded`.

    ``output`` matches :meth:`Sequential.forward`. ``layer_spikes`` maps a layer
    index to ``[T, batch, ...]``. ``membrane`` does the same for each recorded
    LIF, and the value is the potential before reset.
    """

    output: Tensor
    layer_spikes: dict[int, Tensor]
    membrane: dict[int, Tensor]


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
        return self._unroll(x, record=False).output

    def forward_recorded(
        self,
        x: Tensor,
        *,
        layers: Sequence[int] | None = None,
    ) -> RecordedForward:
        """Same unroll as :meth:`forward`, plus the tensors a probe needs.

        ``layers`` selects which layer indices are kept. ``None`` keeps every
        layer. LIF entries in ``membrane`` are the potential compared with the
        threshold, before reset. ``output`` is identical to :meth:`forward`.
        """
        chosen = None if layers is None else frozenset(layers)
        return self._unroll(x, record=True, chosen=chosen)

    def _unroll(
        self,
        x: Tensor,
        *,
        record: bool,
        chosen: frozenset[int] | None = None,
    ) -> RecordedForward:
        time_steps = x.shape[0]
        lif_states: dict[int, LIFState] = {}
        # Past inputs to each delayed Dense, oldest first, not including this step.
        delay_hist: dict[int, list[Tensor]] = {}
        spikes_out: list[Tensor] = []
        kept_spikes: dict[int, list[Tensor]] = {}
        kept_mem: dict[int, list[Tensor]] = {}

        def _keep(index: int) -> bool:
            return record and (chosen is None or index in chosen)

        for _t in range(time_steps):
            h: Tensor = x[_t]
            for i, layer in enumerate(self._layers):
                mem_t: Tensor | None = None
                if isinstance(layer, LIF):
                    if i not in lif_states:
                        lif_states[i] = LIFState(
                            mem=torch.zeros_like(h),
                            syn=(
                                torch.zeros_like(h)
                                if layer.tau_syn is not None
                                else None
                            ),
                        )
                    spk, lif_states[i], mem_t = layer.step(h, lif_states[i])
                    h = spk
                elif isinstance(layer, Dense) and layer.delay > 0:
                    h = _delayed_dense(layer, h, delay_hist, i)
                else:
                    h = layer(h)  # type: ignore[assignment]
                if _keep(i):
                    kept_spikes.setdefault(i, []).append(h)
                    if mem_t is not None:
                        kept_mem.setdefault(i, []).append(mem_t)
            spikes_out.append(h)

        return RecordedForward(
            output=torch.stack(spikes_out, dim=0),
            layer_spikes={i: torch.stack(frames, dim=0) for i, frames in kept_spikes.items()},
            membrane={i: torch.stack(frames, dim=0) for i, frames in kept_mem.items()},
        )

    def extra_repr(self) -> str:
        lines = [f"  ({i}): {layer}" for i, layer in enumerate(self._layers)]
        return "\n" + "\n".join(lines) + "\n"


def _delayed_dense(
    layer: Dense,
    current: Tensor,
    history: dict[int, list[Tensor]],
    index: int,
) -> Tensor:
    """Apply ``layer`` to the input from ``layer.delay`` steps ago.

    The current input is stored after the read, so delay 1 uses the previous
    step and the first step sees zeros. The stored tensors stay in the
    autograd graph.
    """
    past = history.setdefault(index, [])
    if len(past) >= layer.delay:
        delayed = past[-layer.delay]
    else:
        delayed = torch.zeros_like(current)
    past.append(current)
    if len(past) > layer.delay:
        past.pop(0)
    return layer(delayed)
