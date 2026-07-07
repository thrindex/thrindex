"""Surrogate gradient functions for spiking neuron training.

The forward pass of a spiking neuron is a Heaviside step function, which has
zero gradient almost everywhere and is undefined at the threshold.  Surrogate
gradient methods replace the backward pass with a smooth proxy that provides a
useful learning signal while leaving the forward (spike/no-spike) decision exact.

Reference: Neftci, Mostafa & Zenke (2019). "Surrogate Gradient Learning in
Spiking Neural Networks." IEEE Signal Processing Magazine, 36(6), 51–63.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    pass

__all__ = ["FastSigmoid"]


class FastSigmoid(torch.autograd.Function):
    """Heaviside spike with fast-sigmoid surrogate gradient.

    **Forward:** :math:`H(x) = \\mathbb{1}[x \\ge 0]`

    **Backward (surrogate):**

    .. math::

        \\frac{\\partial \\hat{H}}{\\partial x} =
            \\frac{\\beta}{2(1 + \\beta |x|)^2}

    where ``x = U − threshold`` (the shifted membrane potential) and ``β``
    is the slope parameter that controls how sharply the proxy peaks at zero.
    A larger ``β`` makes the surrogate a closer approximation to the true
    (zero-gradient) Heaviside, at the cost of vanishing gradients further
    from the threshold.

    Usage::

        spk = FastSigmoid.apply(mem - threshold, beta)
        loss = spk.sum()
        loss.backward()  # uses the surrogate, not the Heaviside gradient
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, beta: float) -> Tensor:  # noqa: ANN401
        """Heaviside forward; saves ``x`` and ``beta`` for the backward pass."""
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x >= 0.0).float()

    @staticmethod
    def backward(  # type: ignore[override]  # torch uses variadic *grad_outputs; our forward has 1 output
        ctx: Any,  # noqa: ANN401
        grad_output: Tensor,
    ) -> tuple[Tensor, None]:
        """Fast-sigmoid surrogate gradient: β / (2(1 + β|x|)²) · upstream."""
        (x,) = ctx.saved_tensors
        beta: float = ctx.beta
        surrogate = beta / (2.0 * (1.0 + beta * x.abs()) ** 2)
        return surrogate * grad_output, None
