"""Spike encoders: convert continuous inputs to spike trains.

Three encoders are provided:

- :func:`rate` — Bernoulli-sampled spikes proportional to input value.
  **Requires an explicit** :class:`torch.Generator`; never touches global RNG state.
- :func:`latency` — deterministic temporal coding; higher inputs spike earlier.
- :func:`delta` — deterministic change coding; spikes on absolute change above a threshold.

Design constraint (Playbook §23, no hidden global state):
    ``rate`` requires an explicit ``generator`` argument.  Callers are responsible for
    constructing and seeding the generator:

    .. code-block:: python

        gen = torch.Generator()
        gen.manual_seed(42)
        spikes = thx.encoders.rate(x, T=25, generator=gen)
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["delta", "latency", "rate"]


def rate(x: Tensor, T: int, generator: torch.Generator) -> Tensor:
    """Rate (Bernoulli) encoder.

    Each element of ``x`` is treated as a spike probability and sampled
    independently at each of the ``T`` timesteps.

    Parameters
    ----------
    x:
        Input tensor with values in ``[0, 1]``, shape ``[batch, *]``.
        Values outside ``[0, 1]`` are clamped before sampling.
    T:
        Number of timesteps to generate.
    generator:
        Explicit :class:`torch.Generator` for reproducibility.  This function
        **never** modifies global torch RNG state.  Construct and seed the
        generator in the calling scope:
        ``gen = torch.Generator(); gen.manual_seed(seed)``.

    Returns
    -------
    Tensor
        Spike tensor with values in ``{0.0, 1.0}``, shape ``[T, *x.shape]``.
    """
    x_clamped = x.clamp(0.0, 1.0)
    # Expand along the time dimension: [T, batch, *]
    x_expanded = x_clamped.unsqueeze(0).expand(T, *x_clamped.shape)
    return torch.bernoulli(x_expanded, generator=generator)


def latency(x: Tensor, T: int) -> Tensor:
    """Latency (time-to-first-spike) encoder.

    Higher input values produce earlier spikes.  An input value of ``1.0``
    fires at ``t = 0``; a value of ``0.0`` never fires within ``T`` timesteps.
    The spike time is ``floor((1 − x) * T)``, clamped to ``[0, T)``.

    This encoder is **deterministic** — no RNG is used.

    Parameters
    ----------
    x:
        Input tensor with values in ``[0, 1]``, shape ``[batch, *]``.
    T:
        Number of timesteps.

    Returns
    -------
    Tensor
        Spike tensor with values in ``{0.0, 1.0}``, shape ``[T, *x.shape]``.
        Each spatial location fires at most once.
    """
    x_clamped = x.clamp(0.0, 1.0)
    # Spike time: 0 for x=1, T-1 for x just above 0, T (never) for x=0.
    spike_time = ((1.0 - x_clamped) * T).long().clamp(0, T)  # [batch, *]
    # Build output: t_index tensor [T, 1, ...] vs spike_time [1, batch, *]
    t_idx = torch.arange(T, device=x.device).view(
        (T,) + (1,) * x_clamped.dim()
    )  # [T, 1, ...]
    spikes = (t_idx == spike_time.unsqueeze(0)).float()  # [T, batch, *]
    return spikes


def delta(x: Tensor, T: int, threshold: float = 0.1) -> Tensor:
    """Delta (change) encoder.

    Emits a spike at timestep ``t`` wherever the absolute change in ``x``
    from ``t − 1`` to ``t`` exceeds ``threshold``.  At ``t = 0`` the change
    is defined as ``|x[0] − 0|``.

    This encoder is **deterministic** — no RNG is used.

    Parameters
    ----------
    x:
        Input tensor, shape ``[T, batch, *]``.  Each element ``x[t]``
        represents the signal at timestep ``t``.
    T:
        Expected number of timesteps; must equal ``x.shape[0]``.
    threshold:
        Minimum absolute change required to emit a spike.

    Returns
    -------
    Tensor
        Spike tensor with values in ``{0.0, 1.0}``, shape ``[T, batch, *]``.

    Raises
    ------
    ValueError
        If ``x.shape[0] != T``.
    """
    if x.shape[0] != T:
        raise ValueError(
            f"delta encoder: x.shape[0]={x.shape[0]} does not match T={T}."
        )
    # Prepend a zero frame to compute the difference at t=0.
    x_prev = torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)  # [T, batch, *]
    return ((x - x_prev).abs() > threshold).float()
