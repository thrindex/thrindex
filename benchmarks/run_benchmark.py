"""THRINDEX public benchmark harness — energy and latency.

Task
----
SHD keyword spotting: Dense(700→512) → LIF(τ=20ms) → Dense(512→20) → LIF(τ=20ms)
Input: 100 timesteps × 700 cochlea channels per sample.

What is measured
----------------
1. THRINDEX SNN (via CPU simulator)
   - Synaptic operations (syn-ops): the fundamental neuromorphic work unit.
   - Modeled energy: syn_ops × coefficient_pJ.
   - Simulator wall time (CPU, single sample, single thread).

2. Equivalent dense MLP baselines
   Two comparisons, clearly labeled:

   (a) Rate-collapsed MLP — Dense(700→512→20) on a single 700-dim vector
       (time-averaged spike rates). Same parameter count. Loses temporal structure.
       This is the minimum-cost MLP that processes any representation of the input.

   (b) Temporal MLP — Dense(700→512→20) run 100× (once per timestep).
       Same information as the SNN. This is the apples-to-apples energy comparison
       for a temporally-faithful baseline without a time-series architecture.

Energy model
------------
SNN:  E = syn_ops × coeff_pJ_per_syn_op
      Default coeff = 0.5 pJ/syn-op (consistent with published 7nm neuromorphic silicon).

MLP:  E = MACs × coeff_pJ_per_MAC
      Two reference points (both modeled, not measured):
      - GPU: 2.5 pJ/MAC  (NVIDIA A100 at 400W / 312 TFLOPS ≈ 1.28 pJ/FLOP = 2.56 pJ/MAC)
      - CPU: 1.0 pJ/MAC  (ARM Cortex-A55 published figure, conservative for modern 7nm)

Latency note
------------
The simulator is a SOFTWARE tool for development and validation — it is not the
deployment target. SNN wall time here reflects the CPU Python simulator, not
neuromorphic silicon. A hardware comparison requires the M6 silicon bring-up.
MLP wall time is measured on the same CPU for consistency.

Usage
-----
    pip install thrindex
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --output benchmarks/results/shd.json

The three SHD test samples in templates/keyword-spotting/samples/ are used.
No external dataset download is required.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))

# ── Constants ──────────────────────────────────────────────────────────────────

# Architecture (matches templates/keyword-spotting/model.thx)
N_IN = 700
HIDDEN = 512
N_CLASSES = 20
T = 100           # timesteps per SHD sample

# Energy coefficients (pJ per operation)
# pJ/syn-op — thrindex default, consistent with 7nm neuromorphic silicon
SNN_COEFF_PJ: float = 0.5
GPU_COEFF_PJ: float = 2.56   # pJ/MAC  — A100: 400W / (2 × 156 TMAC/s) ≈ 2.56 pJ/MAC
CPU_COEFF_PJ: float = 1.0    # pJ/MAC  — ARM reference (conservative)

# Maximum possible syn-ops for this architecture (100% firing, all T timesteps)
MAX_SYN_OPS = (N_IN * HIDDEN + HIDDEN * N_CLASSES) * T  # 36,864,000

# ── Helpers ────────────────────────────────────────────────────────────────────


def _require_thrindex_core() -> None:
    try:
        importlib.import_module("thrindex._core")
    except ImportError:
        sys.exit(
            "thrindex._core is not installed.\n"
            "Build with: maturin develop\n"
            "Or install: pip install thrindex"
        )


def _load_samples() -> list[list[list[float]]]:
    """Load the three bundled SHD samples from templates/keyword-spotting/samples/."""
    samples_dir = ROOT / "templates" / "keyword-spotting" / "samples"
    if not samples_dir.exists():
        sys.exit(
            f"Samples directory not found: {samples_dir}\n"
            "Ensure templates/keyword-spotting/samples/*.json are present."
        )
    samples = []
    for i in range(3):
        path = samples_dir / f"sample_{i:03d}.json"
        data = json.loads(path.read_text())
        samples.append(data["spikes"])
    return samples


def _build_mlp() -> nn.Sequential:
    """Dense MLP with the same architecture as the SNN (no temporal dimension)."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(N_IN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, N_CLASSES),
    )


# ── SNN benchmark ──────────────────────────────────────────────────────────────


def _run_snn(samples: list[list[list[float]]]) -> list[dict]:
    from thrindex._core import run_sim  # type: ignore[import-untyped]

    model_path = ROOT / "templates" / "keyword-spotting" / "model.thx"
    if not model_path.exists():
        sys.exit(
            f"Model artifact not found: {model_path}\n"
            "Run: python templates/keyword-spotting/train.py --epochs 100"
        )

    results = []
    for i, spikes in enumerate(samples):
        _, stats, _ = run_sim(str(model_path), [spikes], 1, 0)
        syn_ops = stats["synaptic_ops"]
        wall_s = stats["wall_secs"]
        energy_nj = syn_ops * SNN_COEFF_PJ / 1e3
        firing_rate = syn_ops / MAX_SYN_OPS
        results.append({
            "sample": i,
            "syn_ops": int(syn_ops),
            "firing_rate": round(firing_rate, 4),
            "energy_nj": round(energy_nj, 3),
            "wall_ms": round(wall_s * 1e3, 3),
        })
    return results


# ── MLP baselines ──────────────────────────────────────────────────────────────


def _macs_per_forward() -> int:
    """MACs for one Dense(700→512→20) forward pass."""
    return N_IN * HIDDEN + HIDDEN * N_CLASSES  # 368,640


def _run_mlp_rate_collapsed(samples: list[list[list[float]]]) -> list[dict]:
    """Rate-collapsed MLP: one forward pass on the mean spike rate vector."""
    mlp = _build_mlp()
    mlp.eval()
    macs = _macs_per_forward()

    results = []
    for i, spikes in enumerate(samples):
        # Collapse temporal dimension: mean spike rate per channel [N_IN]
        t = torch.tensor(spikes, dtype=torch.float32)  # [T, N_IN]
        x = t.mean(dim=0).unsqueeze(0)                 # [1, N_IN]

        # Warm up
        for _ in range(5):
            with torch.no_grad():
                mlp(x)

        # Time N_ITER iterations
        N_ITER = 1_000
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            with torch.no_grad():
                mlp(x)
        wall_ms = (time.perf_counter() - t0) / N_ITER * 1e3

        energy_gpu_nj = macs * GPU_COEFF_PJ / 1e3
        energy_cpu_nj = macs * CPU_COEFF_PJ / 1e3

        results.append({
            "sample": i,
            "macs": macs,
            "energy_gpu_nj": round(energy_gpu_nj, 3),
            "energy_cpu_nj": round(energy_cpu_nj, 3),
            "wall_ms": round(wall_ms, 4),
        })
    return results


def _run_mlp_temporal(samples: list[list[list[float]]]) -> list[dict]:
    """Temporal MLP: T sequential forward passes — same information as the SNN."""
    mlp = _build_mlp()
    mlp.eval()
    macs_per_step = _macs_per_forward()
    total_macs = macs_per_step * T

    results = []
    for i, spikes in enumerate(samples):
        frames = torch.tensor(spikes, dtype=torch.float32)  # [T, N_IN]

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                for t in range(T):
                    mlp(frames[t].unsqueeze(0))

        # Time N_ITER iterations
        N_ITER = 200
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            with torch.no_grad():
                for t in range(T):
                    mlp(frames[t].unsqueeze(0))
        wall_ms = (time.perf_counter() - t0) / N_ITER * 1e3

        energy_gpu_nj = total_macs * GPU_COEFF_PJ / 1e3
        energy_cpu_nj = total_macs * CPU_COEFF_PJ / 1e3

        results.append({
            "sample": i,
            "macs": total_macs,
            "energy_gpu_nj": round(energy_gpu_nj, 3),
            "energy_cpu_nj": round(energy_cpu_nj, 3),
            "wall_ms": round(wall_ms, 3),
        })
    return results


# ── Reporting ──────────────────────────────────────────────────────────────────


def _avg(vals: list[float]) -> float:
    return statistics.mean(vals)


def _pct(a: float, b: float) -> str:
    """Return 'a is X× b' or 'a/b = X'."""
    ratio = a / b if b > 0 else float("inf")
    return f"{ratio:.1f}×"


def _print_report(snn: list[dict], mlp_rate: list[dict], mlp_temp: list[dict]) -> None:
    snn_energy   = _avg([r["energy_nj"]     for r in snn])
    snn_fire     = _avg([r["firing_rate"]   for r in snn])
    snn_wall     = _avg([r["wall_ms"]       for r in snn])
    snn_syn_ops  = _avg([r["syn_ops"]       for r in snn])

    rate_gpu     = _avg([r["energy_gpu_nj"] for r in mlp_rate])
    rate_cpu     = _avg([r["energy_cpu_nj"] for r in mlp_rate])
    rate_wall    = _avg([r["wall_ms"]       for r in mlp_rate])

    temp_gpu     = _avg([r["energy_gpu_nj"] for r in mlp_temp])
    temp_cpu     = _avg([r["energy_cpu_nj"] for r in mlp_temp])
    temp_wall    = _avg([r["wall_ms"]       for r in mlp_temp])

    w = 60
    print("=" * w)
    print(" THRINDEX benchmark — SHD keyword spotting")
    print(f" Model: Dense({N_IN}→{HIDDEN}) → LIF → Dense({HIDDEN}→{N_CLASSES}) → LIF")
    print(f" T={T} timesteps  |  {len(snn)} samples  |  thrindex CPU simulator")
    print("=" * w)

    print(f"\n{'SNN (thrindex sim)':}")
    print(f"  Avg synaptic ops:   {snn_syn_ops:>14,.0f}")
    print(f"  Avg firing rate:    {snn_fire:>13.1%}")
    print(f"  Modeled energy:     {snn_energy:>13.1f} nJ  "
          f"({SNN_COEFF_PJ} pJ/syn-op — neuromorphic 7nm reference)")
    print(f"  Sim wall time:      {snn_wall:>13.2f} ms  "
          f"(CPU simulator — NOT the deployment target)")

    print(f"\n{'Baseline A — Rate-collapsed MLP (same params, 1 frame, loses temporal info)':}")
    print(f"  MACs:               {mlp_rate[0]['macs']:>14,}")
    print(f"  Modeled energy GPU: {rate_gpu:>13.1f} nJ  "
          f"({GPU_COEFF_PJ} pJ/MAC — A100 reference)")
    print(f"  Modeled energy CPU: {rate_cpu:>13.1f} nJ  "
          f"({CPU_COEFF_PJ} pJ/MAC — ARM reference)")
    print(f"  Wall time:          {rate_wall:>13.4f} ms  (same CPU)")

    print(f"\n{'Baseline B — Temporal MLP (T=100 forward passes, same info as SNN)':}")
    print(f"  MACs:               {mlp_temp[0]['macs']:>14,}")
    print(f"  Modeled energy GPU: {temp_gpu:>13.1f} nJ  "
          f"({GPU_COEFF_PJ} pJ/MAC — A100 reference)")
    print(f"  Modeled energy CPU: {temp_cpu:>13.1f} nJ  "
          f"({CPU_COEFF_PJ} pJ/MAC — ARM reference)")
    print(f"  Wall time:          {temp_wall:>13.2f} ms  (same CPU)")

    print(f"\nEnergy ratios (vs SNN at {SNN_COEFF_PJ} pJ/syn-op)")
    print(f"  Temporal MLP vs SNN (GPU):  {_pct(temp_gpu, snn_energy)}"
          f"  ← apples-to-apples (same temporal information)")
    print(f"  Temporal MLP vs SNN (CPU):  {_pct(temp_cpu, snn_energy)}")
    print(f"  Rate MLP vs SNN (GPU):      {_pct(rate_gpu, snn_energy)}"
          f"  ← not apples-to-apples (MLP discards temporal structure)")

    print(f"\n{'Methodology notes':}")
    print("  Energy values are MODELED from operation counts and published coefficients.")
    print("  Neither SNN nor MLP energy has been measured on physical hardware.")
    print("  Simulator wall time reflects thrindex CPU simulation, not silicon latency.")
    print("  See benchmarks/METHODOLOGY.md for coefficient provenance.")
    print("=" * w)


def _build_results(
    snn: list[dict],
    mlp_rate: list[dict],
    mlp_temp: list[dict],
    device_info: dict,
) -> dict:
    snn_energy  = _avg([r["energy_nj"]     for r in snn])
    snn_fire    = _avg([r["firing_rate"]   for r in snn])
    snn_wall    = _avg([r["wall_ms"]       for r in snn])
    rate_gpu    = _avg([r["energy_gpu_nj"] for r in mlp_rate])
    rate_cpu    = _avg([r["energy_cpu_nj"] for r in mlp_rate])
    temp_gpu    = _avg([r["energy_gpu_nj"] for r in mlp_temp])
    temp_cpu    = _avg([r["energy_cpu_nj"] for r in mlp_temp])

    return {
        "benchmark": "shd-keyword-spotting",
        "run_at": datetime.now(UTC).isoformat(),
        "model": {
            "architecture": (
                f"Dense({N_IN},{HIDDEN}) → LIF(τ=20ms) "
                f"→ Dense({HIDDEN},{N_CLASSES}) → LIF(τ=20ms)"
            ),
            "timesteps": T,
            "n_samples": len(snn),
        },
        "energy_coefficients": {
            "snn_pj_per_syn_op": SNN_COEFF_PJ,
            "mlp_gpu_pj_per_mac": GPU_COEFF_PJ,
            "mlp_cpu_pj_per_mac": CPU_COEFF_PJ,
            "snn_reference": "7nm neuromorphic silicon (consistent with published figures)",
            "gpu_reference": "NVIDIA A100 — 400W TDP / (2 × 156 TMAC/s) ≈ 2.56 pJ/MAC",
            "cpu_reference": "ARM Cortex class CPU — 1.0 pJ/MAC (conservative 7nm estimate)",
        },
        "snn": {
            "per_sample": snn,
            "avg": {
                "syn_ops": round(_avg([r["syn_ops"]    for r in snn])),
                "firing_rate": round(snn_fire, 4),
                "energy_nj": round(snn_energy, 3),
                "wall_ms": round(snn_wall, 3),
            },
        },
        "mlp_rate_collapsed": {
            "description": (
                "Single forward pass on mean spike rates — "
                "same params, loses temporal structure"
            ),
            "macs": mlp_rate[0]["macs"],
            "per_sample": mlp_rate,
            "avg": {
                "energy_gpu_nj": round(rate_gpu, 3),
                "energy_cpu_nj": round(rate_cpu, 3),
                "wall_ms": round(_avg([r["wall_ms"] for r in mlp_rate]), 4),
            },
        },
        "mlp_temporal": {
            "description": f"{T} sequential forward passes — same temporal information as SNN",
            "macs": mlp_temp[0]["macs"],
            "per_sample": mlp_temp,
            "avg": {
                "energy_gpu_nj": round(temp_gpu, 3),
                "energy_cpu_nj": round(temp_cpu, 3),
                "wall_ms": round(_avg([r["wall_ms"] for r in mlp_temp]), 3),
            },
        },
        "energy_ratios": {
            "temporal_mlp_gpu_vs_snn": round(temp_gpu / snn_energy, 2),
            "temporal_mlp_cpu_vs_snn": round(temp_cpu / snn_energy, 2),
            "rate_mlp_gpu_vs_snn": round(rate_gpu / snn_energy, 2),
            "note": (
                "temporal_mlp_*_vs_snn is the apples-to-apples comparison "
                "(same temporal information processed). rate_mlp_*_vs_snn "
                "is NOT apples-to-apples (MLP discards temporal structure)."
            ),
        },
        "device": device_info,
        "methodology": "see benchmarks/METHODOLOGY.md",
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="THRINDEX energy/latency benchmark — SHD keyword spotting."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON results to this path. Default: benchmarks/results/shd.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else ROOT / "benchmarks" / "results" / "shd.json"

    _require_thrindex_core()

    import platform
    device_info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "os": platform.platform(),
        "cpu": platform.processor() or "unknown",
    }

    print("Loading samples…")
    samples = _load_samples()
    print(f"  {len(samples)} samples loaded (SHD keyword-spotting, bundled in repo)")

    print("Running SNN (thrindex CPU simulator)…")
    snn_results = _run_snn(samples)

    print("Running rate-collapsed MLP baseline…")
    mlp_rate_results = _run_mlp_rate_collapsed(samples)

    print("Running temporal MLP baseline (T=100 forward passes)…")
    mlp_temp_results = _run_mlp_temporal(samples)

    print()
    _print_report(snn_results, mlp_rate_results, mlp_temp_results)

    results = _build_results(snn_results, mlp_rate_results, mlp_temp_results, device_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
