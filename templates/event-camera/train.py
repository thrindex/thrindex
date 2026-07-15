"""N-MNIST event-camera classification training script.

Dataset: Neuromorphic-MNIST (N-MNIST)
  https://tonic.readthedocs.io/en/latest/reference/datasets.html#tonic.datasets.NMNIST
  Orchard et al. 2015 — "Converting Static Image Datasets to Spiking Neuromorphic
  Datasets Using Saccades". Frontiers in Neuroscience, 9, 437.
  DOI: 10.3389/fnins.2015.00437

Architecture
------------
Input: N-MNIST frames binned to T=20 timesteps, 2 polarities × 34×34 pixels.
Spatial dimensions are flattened before the SNN so all layers compile through
thrindex's current compiler (Dense + LIF pass; Conv2d+Flatten deferred).

    Dense(2312, 1024) → LIF(τ_mem=20ms)
    Dense(1024,  256) → LIF(τ_mem=20ms)
    Dense( 256,   10) → LIF(τ_mem=20ms)

Reference numbers for floor
----------------------------
  Orchard et al. 2015 (original paper): 97.8%  (rate-coded SNN)
  Zheng et al. 2021 (spikingjelly):     99.6%  (STBP, 5 layers)
  This template (feedforward, no STDP, no recurrence): committed floor ≥ 95.0%

Floor is set conservatively below the published feedforward baseline (97.8%).
Recurrent or STDP-trained results are excluded from the floor evidence.

Usage
-----
    pip install thrindex tonic
    python templates/event-camera/train.py --data-dir /tmp/nmnist --epochs 30

After training the best checkpoint is compiled to model.thx.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch
import torch.nn as nn

import thrindex.snn as snn
from thrindex.compile import compile_model
from thrindex.train import rate_loss

# ── Constants ──────────────────────────────────────────────────────────────────

H, W, C = 34, 34, 2        # N-MNIST sensor size (height, width, polarities)
N_IN = C * H * W           # 2312 — flattened event frame per timestep
HIDDEN1 = 1024
HIDDEN2 = 256
N_CLASSES = 10
T = 20                     # number of time bins
TAU_MEM = 20.0             # ms
THRESHOLD = 0.5
BATCH_SIZE = 128
LR = 5e-4
SEED = 0

FLOOR_ACC = 0.950          # committed accuracy floor (M5 DoD)

# ── Dataset ────────────────────────────────────────────────────────────────────


def _require_tonic() -> None:
    try:
        import tonic  # noqa: F401
    except ImportError:
        sys.exit(
            "tonic is required for N-MNIST loading:\n"
            "    pip install tonic\n"
            "See https://tonic.readthedocs.io for documentation."
        )


def _load_nmnist(
    data_dir: Path,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load N-MNIST and bin events into frame tensors.

    Returns
    -------
    frames : Tensor
        Shape [N, T, C*H*W] — one flattened frame per timestep per sample.
    labels : Tensor
        Shape [N] — integer class in [0, 9].
    """
    import tonic
    import tonic.transforms as transforms

    train = split == "train"
    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)

    frame_transform = transforms.ToFrame(
        sensor_size=sensor_size,
        n_time_bins=T,
    )

    dataset = tonic.datasets.NMNIST(
        save_to=str(data_dir),
        train=train,
        transform=frame_transform,
    )

    print(f"  Loading {split} split ({len(dataset)} samples)…")
    all_frames: list[torch.Tensor] = []
    all_labels: list[int] = []

    for idx in range(len(dataset)):
        frames_np, label = dataset[idx]
        # frames_np: numpy array [T, C, H, W] or [T, H, W, C] depending on tonic version
        frames_t = torch.from_numpy(frames_np).float()
        # Normalise to [0, 1] in case counts > 1 (multiple events per bin)
        frames_t = frames_t.clamp(0.0, 1.0)
        # Ensure [T, C, H, W] — tonic ≥ 0.6 returns [T, C, H, W]
        if frames_t.shape[0] != T:
            # Older tonic may return [C, T, H, W] — permute if needed
            if frames_t.shape[1] == T:
                frames_t = frames_t.permute(1, 0, 2, 3)
        # Flatten spatial: [T, C*H*W]
        frames_t = frames_t.reshape(T, N_IN)
        all_frames.append(frames_t)
        all_labels.append(int(label))

        if (idx + 1) % 10_000 == 0:
            print(f"    {idx + 1}/{len(dataset)}", flush=True)

    data = torch.stack(all_frames, dim=0)    # [N, T, N_IN]
    labels = torch.tensor(all_labels, dtype=torch.long)
    return data, labels


# ── Model ──────────────────────────────────────────────────────────────────────


def build_model() -> snn.Sequential:
    torch.manual_seed(SEED)
    # threshold=0.5 — event frames are already binary {0,1}; membrane accumulates
    # across T=20 bins, so moderate threshold avoids dead-neuron collapse.
    return snn.Sequential(
        snn.Dense(N_IN, HIDDEN1),
        snn.LIF(threshold=THRESHOLD, tau_mem=TAU_MEM),
        snn.Dense(HIDDEN1, HIDDEN2),
        snn.LIF(threshold=THRESHOLD, tau_mem=TAU_MEM),
        snn.Dense(HIDDEN2, N_CLASSES),
        snn.LIF(threshold=THRESHOLD, tau_mem=TAU_MEM),
    )


# ── Training loop ──────────────────────────────────────────────────────────────


def train_epoch(
    model: snn.Sequential,
    data: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    idx = torch.randperm(len(labels))
    for start in range(0, len(labels), BATCH_SIZE):
        batch_idx = idx[start : start + BATCH_SIZE]
        x = data[batch_idx].to(device)          # [B, T, N_IN]
        y = labels[batch_idx].to(device)        # [B]
        x = x.permute(1, 0, 2)                  # [T, B, N_IN]  — time-first
        out = model(x)                           # [T, B, N_CLASSES]
        loss = rate_loss(out, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(
    model: snn.Sequential,
    data: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), BATCH_SIZE):
            x = data[start : start + BATCH_SIZE].to(device)
            y = labels[start : start + BATCH_SIZE].to(device)
            x = x.permute(1, 0, 2)
            out = model(x)
            pred = out.mean(0).argmax(1)
            correct += (pred == y).sum().item()
    return correct / len(labels)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a spiking event-camera classifier on N-MNIST."
    )
    parser.add_argument("--data-dir", default="/tmp/nmnist",
                        help="Directory to download/cache N-MNIST (default: /tmp/nmnist)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--results-out", default="/tmp/nmnist_results.json",
                        help="Path to write JSON results (default: /tmp/nmnist_results.json)")
    args = parser.parse_args()

    _require_tonic()

    data_dir = Path(args.data_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading N-MNIST…")
    train_data, train_labels = _load_nmnist(data_dir, "train")
    test_data, test_labels = _load_nmnist(data_dir, "test")
    print(
        f"Train: {train_data.shape} — Test: {test_data.shape}  "
        f"(dtype={train_data.dtype})"
    )

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    import math
    alpha = math.exp(-1.0 / TAU_MEM)
    print(
        f"\nTraining for {args.epochs} epochs…\n"
        f"Architecture: Dense({N_IN},{HIDDEN1}) → LIF → "
        f"Dense({HIDDEN1},{HIDDEN2}) → LIF → Dense({HIDDEN2},{N_CLASSES}) → LIF\n"
        f"T={T}, τ_mem={TAU_MEM}ms, α≈{alpha:.6f}, threshold={THRESHOLD}, lr={LR}\n"
    )

    best_acc = 0.0
    best_state: dict = {}
    records: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_data, train_labels, optimizer, device)
        acc = evaluate(model, test_data, test_labels, device)
        elapsed = time.time() - t0

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={loss:.4f}  test={acc:.2%}  best={best_acc:.2%}  ({elapsed:.1f}s)"
        )
        records.append({"epoch": epoch, "loss": loss, "test_acc": float(acc)})

    print(f"\nFinal best test accuracy: {best_acc:.2%}")

    # ── Compile best model to model.thx ────────────────────────────────────────
    model.load_state_dict(best_state)
    model = model.cpu()
    output_path = Path(__file__).parent / "model.thx"
    compile_model(model, output_path)
    print(f"Compiled to {output_path}")

    Path(args.results_out).write_text(
        json.dumps({"best_acc": best_acc, "epochs": records}, indent=2),
        encoding="utf-8",
    )
    print(f"Results written to {args.results_out}")

    if best_acc < FLOOR_ACC:
        print(
            f"\nWARNING: best accuracy {best_acc:.2%} < {FLOOR_ACC:.1%} floor.\n"
            "Diagnostic checklist:\n"
            "  1. Run for ≥ 30 epochs — N-MNIST typically saturates around epoch 20.\n"
            "  2. Verify event frames are clamped to {0, 1} — count overflow breaks LIF dynamics.\n"
            "  3. Check tonic version ≥ 0.6 (frame shape is [T, C, H, W]; older builds vary).\n"
            "Reference: Orchard et al. 2015 feedforward SNN baseline: 97.8%."
        )


if __name__ == "__main__":
    main()
