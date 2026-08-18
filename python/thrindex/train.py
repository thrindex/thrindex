"""Training utilities for spiking neural networks.

Provides loss functions and helpers that plug into standard PyTorch training loops.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

__all__ = ["accuracy", "rate_loss"]


def rate_loss(spikes: Tensor, labels: Tensor) -> Tensor:
    """Cross-entropy loss on mean firing rates over time.

    The spike tensor is averaged over the time dimension to produce a per-sample
    firing rate for each output class.  Standard cross-entropy is then applied,
    treating the rates as unnormalized logits.

    This is the standard training objective for rate-coded SNN classifiers and
    matches the reference in:

        Eshraghian et al. (2023). "Training Spiking Neural Networks Using
        Lessons From Deep Learning." Proc. IEEE, 111(9), 1016–1054.
        DOI: 10.1109/JPROC.2023.3308088

    Parameters
    ----------
    spikes:
        Output spike tensor, shape ``[T, batch, n_classes]``.
    labels:
        Ground-truth class indices, shape ``[batch]``.

    Returns
    -------
    Tensor
        Scalar cross-entropy loss, differentiable with respect to ``spikes``.
    """
    # Mean firing rate over T timesteps: [batch, n_classes]
    mean_rates = spikes.mean(dim=0)
    return F.cross_entropy(mean_rates, labels)


def accuracy(spikes: Tensor, labels: Tensor) -> float:
    """Classification accuracy: fraction of correct predictions.

    The predicted class for each sample is the output neuron with the highest
    total spike count over ``T`` timesteps (rate decoding, argmax).

    Parameters
    ----------
    spikes:
        Output spike tensor, shape ``[T, batch, n_classes]``.
    labels:
        Ground-truth class indices, shape ``[batch]``.

    Returns
    -------
    float
        Accuracy in ``[0.0, 1.0]``.  Not differentiable — use :func:`rate_loss`
        for the training objective.

    Example
    -------
    ::

        spikes = model(encoded_input)                    # [T, batch, n_classes]
        loss = rate_loss(spikes, labels)
        acc = accuracy(spikes, labels)                   # e.g. 0.924
        print(f"loss={loss.item():.4f}  acc={acc:.3f}")
    """
    mean_rates = spikes.mean(dim=0)           # [batch, n_classes]
    predictions = mean_rates.argmax(dim=1)    # [batch]
    return (predictions == labels).float().mean().item()
