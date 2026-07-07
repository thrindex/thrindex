"""Training utilities for spiking neural networks.

Provides loss functions and helpers that plug into standard PyTorch training loops.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

__all__ = ["rate_loss"]


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
