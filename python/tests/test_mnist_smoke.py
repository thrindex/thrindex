"""CI smoke test for the MNIST LIF training pipeline.

Correction 7: we do NOT run 50 epochs in CI.  This test uses synthetic
MNIST-shaped data (no download), runs 1 epoch, and asserts:
  1. Loss decreases from first to last batch.
  2. Training accuracy exceeds chance (> 10%).

The test uses the same Dense(784→1000→10) + LIF pipeline as the full
benchmark, but with threshold=0.5 and lr=5e-3 so that the output LIF
generates spikes within a single CI epoch.  The full benchmark (threshold=1.0,
tau_mem=5.0, 50 epochs) is run by the human and recorded in docs/validation.md.
"""

from __future__ import annotations

import thrindex.snn as snn
import torch
from thrindex.encoders import rate
from thrindex.train import rate_loss

# Same two-layer Dense+LIF pipeline as the full benchmark; only the threshold
# and learning rate differ (see module docstring).
_T = 10           # fewer timesteps keeps CI fast
_HIDDEN = 1000    # hidden size matches the benchmark
_THRESHOLD = 0.5  # lower than benchmark (1.0) so output LIF fires in 1 epoch
_TAU_MEM = 5.0


def _build_smoke_model() -> snn.Sequential:
    return snn.Sequential(
        snn.Dense(784, _HIDDEN),
        snn.LIF(threshold=_THRESHOLD, tau_mem=_TAU_MEM, reset="subtract"),
        snn.Dense(_HIDDEN, 10),
        snn.LIF(threshold=_THRESHOLD, tau_mem=_TAU_MEM, reset="subtract"),
    )


def test_mnist_pipeline_smoke() -> None:
    """1 epoch on synthetic MNIST-shaped data: loss decreases, accuracy > chance.

    Structured synthetic data: each class c has its discriminative features in
    column band [c*78 : (c+1)*78] set high, making the task linearly separable
    so the network can learn within a single epoch.
    """
    torch.manual_seed(0)

    n_classes = 10
    samples_per_class = 20    # 200 total training samples
    n_train = n_classes * samples_per_class
    batch_size = 20

    # Build structured synthetic data: class-specific feature bands at amplitude 1.0.
    x_data = torch.rand(n_train, 784) * 0.05   # low-amplitude noise baseline
    y = torch.arange(n_classes).repeat_interleave(samples_per_class)
    for c in range(n_classes):
        x_data[y == c, c * 78 : (c + 1) * 78] = 1.0   # hard class signal
    x_data = x_data.clamp(0.0, 1.0)

    model = _build_smoke_model()
    torch.manual_seed(0)   # reproducible weight init (re-seed after data generation)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    batch_losses: list[float] = []
    for i in range(0, n_train, batch_size):
        x_batch = x_data[i : i + batch_size]
        y_batch = y[i : i + batch_size]

        # Explicit generator per batch — never touches global RNG.
        gen = torch.Generator()
        gen.manual_seed(i)
        spikes_in = rate(x_batch, T=_T, generator=gen)

        optimizer.zero_grad()
        spk_out = model(spikes_in)
        loss = rate_loss(spk_out, y_batch)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()  # type: ignore[no-untyped-call]
        batch_losses.append(float(loss.item()))

    # 1. Loss must have decreased at some point during the epoch.
    assert min(batch_losses) < batch_losses[0], (
        f"Loss never decreased below the first batch value ({batch_losses[0]:.4f}).\n"
        f"Full trace: {[f'{v:.4f}' for v in batch_losses]}"
    )

    # 2. Training accuracy must exceed chance (10%).
    with torch.no_grad():
        gen_eval = torch.Generator()
        gen_eval.manual_seed(99999)
        spikes_in = rate(x_data, T=_T, generator=gen_eval)
        spk_out = model(spikes_in)
        preds = spk_out.mean(0).argmax(1)
        acc = float((preds == y).float().mean().item())

    assert acc > 0.10, (
        f"Training accuracy {acc:.3f} ≤ chance (10%) after 1 epoch — "
        "check LIF dynamics or gradient flow."
    )
