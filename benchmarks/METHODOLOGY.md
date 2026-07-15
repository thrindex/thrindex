# Benchmark Methodology

## What is benchmarked

**Task:** SHD keyword spotting — `Dense(700→512) → LIF(τ=20ms) → Dense(512→20) → LIF(τ=20ms)`, T=100 timesteps per sample.

**Samples:** Three real SHD test samples bundled in `templates/keyword-spotting/samples/` (CC BY 4.0, Zenke Lab). No external download required.

---

## Energy model

All energy values are **modeled from operation counts and published coefficients**, not measured on physical hardware.

### SNN energy

```
E_snn = syn_ops × coeff_pJ_per_syn_op
```

- `syn_ops` = actual synaptic operations triggered by spikes, measured by the thrindex simulator.
- `coeff_pJ_per_syn_op` = **0.5 pJ/syn-op** (default).

**Provenance of 0.5 pJ/syn-op:**
This figure is consistent with published results from digital neuromorphic chips fabricated in advanced CMOS nodes:
- Loihi 2 (Intel, Intel 4 process): 0.5–10 pJ/syn-op depending on workload
- TrueNorth (IBM, 28nm): ~26 pJ/syn-op at the chip level (higher due to older node)
- Academic TSMC 7nm designs: ~0.3–1.0 pJ/syn-op reported

0.5 pJ is a representative figure for a modern 7nm digital implementation.
It is not calibrated to any shipping chip. Use `thrindex run model.thx --energy-coeff X`
to apply a device-specific value.

### MLP energy

```
E_mlp = MACs × coeff_pJ_per_MAC
```

Two reference coefficients are reported:

| Reference | Coefficient | Source |
|---|---|---|
| GPU (NVIDIA A100) | 2.56 pJ/MAC | 400W TDP / (2 × 156 TMAC/s FP16) |
| CPU (ARM reference) | 1.0 pJ/MAC | ARM Cortex class, conservative 7nm estimate |

**Why GPU energy is higher per operation:** GPU efficiency peaks at large batch sizes due to memory bandwidth. For a single-sample inference (T=100 frames), the GPU spends the majority of its power budget on memory access and idle compute cycles, not computation. The per-MAC figure used here is a lower bound (bulk throughput), making the GPU comparison optimistic for the MLP.

---

## Baseline definitions

### Baseline A — Rate-collapsed MLP

One forward pass on the **mean spike rate** across T timesteps.

```python
x_rate = spikes.mean(dim=0)   # [N_IN=700] — collapse time
mlp(x_rate)                   # 1 forward pass, 368,640 MACs
```

**This is NOT apples-to-apples.** The MLP receives a single 700-dimensional vector (the temporal average); the SNN processes 100 timesteps sequentially. Any timing information in the spike pattern is discarded. This baseline establishes the minimum-cost alternative — a fair comparison only if temporal structure is irrelevant for the task.

### Baseline B — Temporal MLP

T=100 sequential forward passes, one per timestep, on the same spike frames the SNN sees.

```python
for t in range(T):
    mlp(spikes[t])    # 368,640 MACs per step → 36,864,000 total
```

**This is apples-to-apples.** Both systems process the same 100 timesteps of temporal data. The temporal MLP has no memory (each pass is stateless), so it discards inter-timestep dependencies — still an architectural disadvantage — but the information budget is identical.

---

## Latency note

The SNN simulator wall time measures **thrindex-sim on CPU** — a software tool for development and conformance validation, not the deployment target. Neuromorphic silicon (M6 milestone) achieves orders-of-magnitude lower latency than CPU simulation.

The MLP wall time is measured on the same CPU for a consistent baseline, but a production GPU deployment would be faster. Both latency figures should be read as "CPU software comparison" only.

**No latency claim about neuromorphic hardware is made here.**

---

## Reproducibility

```bash
pip install thrindex
python benchmarks/run_benchmark.py
```

Output is written to `benchmarks/results/shd.json`. The script is seeded (MLP weights at `torch.manual_seed(0)`) and fully deterministic given the same hardware and software versions.
