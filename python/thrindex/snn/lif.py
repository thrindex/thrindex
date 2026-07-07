"""Leaky Integrate-and-Fire neuron module.

Design decisions are recorded in:
- ``adr/0004-lif-state-ownership.md``: why ``LIF`` is stateless and ``Sequential``
  owns the membrane tensors.
- ``adr/0005-lif-leak-discretization.md``: why alpha = exp(−dt / tau_mem) rather
  than the Euler approximation 1 − dt / tau_mem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
from torch import Tensor

from thrindex.snn.surrogate import FastSigmoid

__all__ = ["LIF", "LIFState"]


@dataclass
class LIFState:
    """Recurrent state of a single ``LIF`` layer.

    Attributes
    ----------
    mem:
        Membrane potential, shape ``[batch, *]``.  Initialized to zeros by
        ``Sequential`` at the start of each forward call.
    syn:
        Synaptic current (only used when ``tau_syn`` is set on the ``LIF``).
        ``None`` for single-state LIF neurons.
    """

    mem: Tensor
    syn: Tensor | None = None


class LIF(nn.Module):
    """Leaky Integrate-and-Fire neuron layer.

    Per-timestep update order (ADR-0005, verbatim):

    .. code-block::

        (i)  mem_t   = alpha * mem_prev + x_t          # leak + integrate
        (ii) spk_t   = H(mem_t − threshold)            # fire (Heaviside; surrogate in backward)
        (iii) mem_out = mem_t − spk_t * threshold      # reset (subtract mode)
              OR
              mem_out = mem_t * (1 − spk_t.detach())   # reset (zero mode)

    where ``alpha = exp(−dt / tau_mem)`` (exact exponential, ADR-0005) and
    ``dt = 1.0 ms`` (canonical timestep, ``LIF.DT``).

    The ``spk_t.detach()`` in zero-mode reset severs the gradient path through
    the reset gate.  Gradients should flow via the surrogate spike, not via the
    binary reset mask (which is not differentiable and would create a spurious
    second gradient path through ``mem_t``).

    Parameters
    ----------
    threshold:
        Firing threshold (dimensionless; typically 1.0 in normalized models).
    tau_mem:
        Membrane time constant in ms.  Must be strictly greater than ``dt = 1.0 ms``.
    tau_syn:
        Synaptic time constant in ms.  ``None`` = no synaptic filtering (single-state
        LIF).  A float value adds an exponentially decaying synaptic current as a
        second state variable (LIF-SRM / two-compartment model).
    reset:
        Reset mode after a spike.  ``"subtract"`` (membrane −= threshold) or
        ``"zero"`` (membrane → 0).  See ``adr/0004`` for the precise definitions.
    delay:
        Synaptic transmission delay in timesteps.  Only ``0`` is supported in M1;
        any other value raises ``NotImplementedError`` pointing at RFC-002.
    beta:
        Slope of the fast-sigmoid surrogate gradient.  Default 25 matches the
        snnTorch reference implementation for the benchmark in ``docs/validation.md``.
    """

    DT: Final[float] = 1.0  # canonical timestep in ms (ADR-0005); never a constructor param

    def __init__(
        self,
        threshold: float = 1.0,
        tau_mem: float = 10.0,
        tau_syn: float | None = None,
        reset: str = "subtract",
        delay: int = 0,
        beta: float = 25.0,
    ) -> None:
        super().__init__()
        if tau_mem <= self.DT:
            raise ValueError(
                f"tau_mem={tau_mem!r} ms must be strictly greater than dt={self.DT} ms. "
                "A time constant at or below the timestep produces a membrane that loses "
                "all history within a single step, which is almost certainly a parameter "
                "error.  See adr/0005-lif-leak-discretization.md."
            )
        if tau_syn is not None and tau_syn <= self.DT:
            raise ValueError(
                f"tau_syn={tau_syn!r} ms must be strictly greater than dt={self.DT} ms."
            )
        if reset not in ("subtract", "zero"):
            raise ValueError(
                f"reset must be 'subtract' or 'zero', got {reset!r}.  "
                "See Playbook §28 for canonical parameter vocabulary."
            )
        if delay != 0:
            raise NotImplementedError(
                f"delay={delay} is not yet implemented.  "
                "Synaptic delays are governed by RFC-002, which is open.  "
                "Only delay=0 is supported in M1."
            )

        self.threshold = threshold
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.reset = reset
        self.beta = beta

        # alpha = exp(−dt / tau_mem)  [ADR-0005: exact exponential]
        # Stored as plain floats; multiplied against tensors in forward.
        self.alpha: float = math.exp(-self.DT / tau_mem)
        self.alpha_syn: float | None = (
            math.exp(-self.DT / tau_syn) if tau_syn is not None else None
        )

    def forward(self, x: Tensor, state: LIFState) -> tuple[Tensor, LIFState]:
        """Single-timestep forward pass.

        Parameters
        ----------
        x:
            Input current, shape ``[batch, *]``.  For a ``Dense`` → ``LIF`` chain
            this is the weighted sum from the dense layer.
        state:
            Previous ``LIFState`` carrying ``mem`` (and optionally ``syn``).

        Returns
        -------
        spk:
            Spike tensor, values in ``{0.0, 1.0}``, same shape as ``x``.
        new_state:
            Updated ``LIFState`` after integration, firing, and reset.
        """
        mem_prev = state.mem

        if self.tau_syn is not None and self.alpha_syn is not None:
            # Two-state LIF-SRM: exponentially filtered synaptic current.
            syn_prev = state.syn if state.syn is not None else torch.zeros_like(x)
            syn_t = self.alpha_syn * syn_prev + x
            mem_t = self.alpha * mem_prev + syn_t
        else:
            # Single-state LIF: input delivered directly to membrane.
            syn_t = None
            mem_t = self.alpha * mem_prev + x

        # (ii) Fire: Heaviside forward, fast-sigmoid surrogate backward.
        # Function.apply is typed as (...) -> Unknown in torch stubs; cast explicitly.
        spk: Tensor = FastSigmoid.apply(mem_t - self.threshold, self.beta)  # type: ignore[assignment]

        # (iii) Reset.
        if self.reset == "subtract":
            mem_out: Tensor = mem_t - spk * self.threshold
        else:
            # zero mode: mem goes to 0 on spike.
            # spk.detach() severs the gradient through the reset gate — gradients
            # must flow via the surrogate spike path, not the reset mask.
            mem_out = mem_t * (1.0 - spk.detach())

        return spk, LIFState(mem=mem_out, syn=syn_t)

    def extra_repr(self) -> str:
        parts = [
            f"threshold={self.threshold}",
            f"tau_mem={self.tau_mem}",
            f"alpha={self.alpha:.6f}",
            f"reset={self.reset!r}",
        ]
        if self.tau_syn is not None:
            parts.append(f"tau_syn={self.tau_syn}")
        return ", ".join(parts)
