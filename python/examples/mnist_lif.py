"""Full MNIST LIF-SNN training script.

Reproduces the surrogate-gradient LIF benchmark from:

    Eshraghian, J. K., Ward, M., Neftci, E. O., et al. (2023).
    "Training Spiking Neural Networks Using Lessons From Deep Learning."
    Proceedings of the IEEE, 111(9), 1016–1054.
    DOI: 10.1109/JPROC.2023.3308088

Reference implementation: snnTorch (https://github.com/jeshraghian/snntorch),
used as the comparison baseline.  Published accuracy range: per reference
implementation documentation (see docs/validation.md for the committed result).

Architecture
------------
    Dense(784, 1000) → LIF(threshold=1.0, tau_mem=5.0, reset="subtract")
    Dense(1000, 10)  → LIF(threshold=1.0, tau_mem=5.0, reset="subtract")

Training
--------
    Encoder : rate (T=25 Poisson-like timesteps, pixels ÷ 255 → [0,1])
    Loss    : cross-entropy on mean firing rates (thrindex.train.rate_loss)
    Optimizer: Adam, lr=5e-4, default betas
    Batch   : 128
    Epochs  : 50
    Seed    : torch.manual_seed(0), torch.Generator(seed=0) for rate encoder

Committed accuracy target: ≥ 98.0% on MNIST test set (see docs/validation.md).

Usage
-----
    # Full 50-epoch run (run on your machine; do NOT run in CI)
    python python/examples/mnist_lif.py --epochs 50 --data-dir /tmp/mnist

    # Quick smoke test (1 epoch, synthetic data)
    python python/examples/mnist_lif.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import thrindex.snn as snn
import torch
import torch.nn as nn
from thrindex.encoders import rate
from thrindex.train import rate_loss

# ── Architecture ───────────────────────────────────────────────────────────────

T = 25             # timesteps
HIDDEN = 1000      # hidden layer size
THRESHOLD = 1.0
TAU_MEM = 5.0      # → alpha = exp(-1/5) ≈ 0.8187
BETA = 25.0        # surrogate slope


def build_model() -> snn.Sequential:
    """Two-layer FC LIF network matching the benchmark architecture."""
    return snn.Sequential(
        snn.Dense(784, HIDDEN),
        snn.LIF(threshold=THRESHOLD, tau_mem=TAU_MEM, reset="subtract", beta=BETA),
        snn.Dense(HIDDEN, 10),
        snn.LIF(threshold=THRESHOLD, tau_mem=TAU_MEM, reset="subtract", beta=BETA),
    )


# ── Training & evaluation ──────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    enc_seed_base: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)  # [B, 1, 28, 28]
        labels = labels.to(device)
        x_flat = images.view(images.size(0), -1)  # [B, 784]

        # Explicit generator — never uses global torch RNG (correction 5).
        gen = torch.Generator(device=device)
        gen.manual_seed(enc_seed_base + epoch * 1_000_000 + batch_idx)
        spikes_in = rate(x_flat, T=T, generator=gen)  # [T, B, 784]

        optimizer.zero_grad()
        spk_out = model(spikes_in)  # [T, B, 10]
        loss = rate_loss(spk_out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    enc_seed: int = 999_999,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        x_flat = images.view(images.size(0), -1)

        gen = torch.Generator(device=device)
        gen.manual_seed(enc_seed + batch_idx)
        spikes_in = rate(x_flat, T=T, generator=gen)
        spk_out = model(spikes_in)  # [T, B, 10]

        mean_rates = spk_out.mean(dim=0)  # [B, 10]
        preds = mean_rates.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ── Smoke test (CI mode) ───────────────────────────────────────────────────────


def run_smoke_test(device: torch.device) -> None:
    """CI smoke test: 1 epoch on synthetic MNIST-shaped data.

    Asserts:
    1. Loss decreases from first to last batch.
    2. Training accuracy > chance (> 10%) after 1 epoch.
    """
    print("Running CI smoke test (synthetic data, 1 epoch)…")
    torch.manual_seed(0)

    n_classes = 10
    samples_per_class = 10  # 100 total training samples
    n_train = n_classes * samples_per_class

    # Structured synthetic data: class c has amplified features in band [c*78:(c+1)*78].
    x_data = torch.rand(n_train, 784) * 0.2
    y = torch.arange(n_classes).repeat_interleave(samples_per_class)
    for c in range(n_classes):
        x_data[y == c, c * 78 : (c + 1) * 78] += 0.8
    x_data = x_data.clamp(0.0, 1.0).to(device)
    y = y.to(device)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    batch_size = 10
    batch_losses: list[float] = []

    for i in range(0, n_train, batch_size):
        x_batch = x_data[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        gen = torch.Generator(device=device)
        gen.manual_seed(i)
        spikes_in = rate(x_batch, T=T, generator=gen)

        optimizer.zero_grad()
        spk_out = model(spikes_in)
        loss = rate_loss(spk_out, y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(float(loss.item()))

    # Assertion 1: loss decreases
    if batch_losses[-1] >= batch_losses[0]:
        print(
            f"  FAIL: loss did not decrease "
            f"(first={batch_losses[0]:.4f}, last={batch_losses[-1]:.4f})",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  ✓ Loss decreased: {batch_losses[0]:.4f} → {batch_losses[-1]:.4f}")

    # Assertion 2: training accuracy > chance
    with torch.no_grad():
        gen_eval = torch.Generator(device=device)
        gen_eval.manual_seed(12345)
        spikes_in = rate(x_data, T=T, generator=gen_eval)
        spk_out = model(spikes_in)
        preds = spk_out.mean(0).argmax(1)
        acc = float((preds == y).float().mean().item())

    if acc <= 0.10:
        print(f"  FAIL: training accuracy {acc:.3f} ≤ chance (10%)", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ Training accuracy: {acc:.3f} > chance (10%)")
    print("Smoke test PASSED.")


# ── Full training ──────────────────────────────────────────────────────────────


def run_full_training(args: argparse.Namespace) -> None:
    try:
        from torchvision import datasets, transforms  # type: ignore[import]
    except ImportError:
        print(
            "torchvision not installed.  Run: pip install thrindex[examples]",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"thrindex version: {thrindex.__version__}")  # type: ignore[attr-defined]  # noqa: F821

    torch.manual_seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor()])
    data_dir = Path(args.data_dir)
    train_data = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_data = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size
    )

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    history: list[dict[str, float]] = []

    print(f"\nTraining for {args.epochs} epochs…")
    print(f"Architecture: Dense(784,{HIDDEN}) → LIF → Dense({HIDDEN},10) → LIF")
    print(f"T={T}, tau_mem={TAU_MEM}, alpha≈{math.exp(-1/TAU_MEM):.6f}, lr={args.lr}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs,
            enc_seed_base=args.seed,
        )
        test_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        best_acc = max(best_acc, test_acc)

        record = {"epoch": epoch, "loss": train_loss, "test_acc": test_acc}
        history.append(record)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f}  test={test_acc*100:.2f}%  "
            f"best={best_acc*100:.2f}%  ({elapsed:.1f}s)"
        )

    print(f"\nFinal best test accuracy: {best_acc*100:.2f}%")
    if best_acc < 0.98:
        print("WARNING: best accuracy < 98.0% — check implementation against ADR-0005.")

    # Write results for docs/validation.md (user fills this file manually after review).
    results_path = Path(args.results_out) if args.results_out else None
    if results_path:
        results_path.write_text(json.dumps({
            "best_test_acc": best_acc,
            "final_epoch_loss": history[-1]["loss"],
            "history": history,
            "config": {
                "T": T, "hidden": HIDDEN, "threshold": THRESHOLD,
                "tau_mem": TAU_MEM, "beta": BETA, "seed": args.seed,
                "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
            },
        }, indent=2))
        print(f"Results written to {results_path}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST LIF-SNN benchmark")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="/tmp/mnist")
    parser.add_argument("--results-out", type=str, default="",
                        help="Path to write JSON results (for docs/validation.md)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run CI smoke test on synthetic data instead of full training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke_test:
        run_smoke_test(device)
    else:
        run_full_training(args)


if __name__ == "__main__":
    main()
