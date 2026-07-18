<div align="center">

<a href="https://thrindex.com">
  <img src="https://raw.githubusercontent.com/thrindex/thrindex/main/docs/assets/banner.png" width="800px" alt="Thrindex — The open compiler for spiking neuromorphic compute"/>
</a>

<br/>

**The open compiler for spiking neuromorphic compute.**  
Author spiking neural networks in Python. Compile them to neuromorphic hardware.

<br/>

<a href="https://thrindex.com">Website</a> · <a href="https://docs.thrindex.com">Docs</a> · <a href="https://github.com/thrindex/thrindex/discussions">Discussions</a>

<br/>
<br/>

[![CI](https://img.shields.io/github/actions/workflow/status/thrindex/thrindex/ci.yml?label=CI&color=0066FF)](https://github.com/thrindex/thrindex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/thrindex?color=0066FF&label=python)](https://pypi.org/project/thrindex/)
[![Downloads](https://img.shields.io/pepy/dt/thrindex?color=0066FF&label=downloads)](https://pepy.tech/projects/thrindex)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-0066FF.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-0066FF.svg)](https://www.python.org/)

</div>

---

## What is it?

thrindex is the full-stack toolkit for spiking neural networks (SNNs) — from model authoring to deployment on neuromorphic hardware.

It gives you a **PyTorch-native authoring API** for building and training SNNs with surrogate gradients, a **deterministic Rust simulation engine** for fast, reproducible model validation, and a **compiler + runtime** that targets real neuromorphic chips — all through one unified SDK.

```python
import thrindex as thx
import thrindex.snn as snn

model = snn.Sequential(
    snn.Dense(784, 1000),
    snn.LIF(threshold=1.0, tau_mem=5.0),
    snn.Dense(1000, 10),
    snn.LIF(threshold=1.0, tau_mem=5.0),
)

# train with surrogate gradients — standard PyTorch loop
loss = thx.train.rate_loss(model(encoded_input), labels)
loss.backward()
```

---

## Install

```bash
pip install thrindex
```

Requires Python 3.11+ and PyTorch 2.0+.

---

## Quickstart

```python
import torch
import thrindex as thx
import thrindex.snn as snn
from thrindex.encoders import rate
from thrindex.train import rate_loss

# Build a two-layer LIF network
model = snn.Sequential(
    snn.Dense(784, 1000),
    snn.LIF(threshold=1.0, tau_mem=5.0, reset="subtract"),
    snn.Dense(1000, 10),
    snn.LIF(threshold=1.0, tau_mem=5.0, reset="subtract"),
)

# Rate-encode your input (explicit generator — no global RNG state)
gen = torch.Generator()
gen.manual_seed(42)
spikes = rate(x_batch, T=25, generator=gen)   # [T, batch, 784]

# Forward + loss
spk_out = model(spikes)                        # [T, batch, 10]
loss = rate_loss(spk_out, labels)

# Standard PyTorch backward
loss.backward()
optimizer.step()
```

---

## Key features

| Feature | Detail |
|---|---|
| **SNN layers** | `LIF`, `Sequential`, `Dense`, `Conv2d` — all `nn.Module` subclasses |
| **Surrogate gradients** | Fast-sigmoid, analytically verified backward pass |
| **Spike encoders** | Rate (Bernoulli), latency (time-to-first-spike), delta (change coding) |
| **Deterministic numerics** | Parameterizable fixed-point `Q<INT, FRAC>` types; round-half-even; bit-identical across platforms |
| **Typed errors** | Stable `E####` error codes — every error tells you what happened, why, and how to fix it |
| **Hardware target** | Compile `.thx` artifacts and deploy to neuromorphic silicon (M2+) |

---

## LIF neuron

The `LIF` module follows the exact per-timestep update order:

```
(i)   mem_t    = α · mem_prev + x_t          # leak + integrate
(ii)  spk_t    = H(mem_t − threshold)         # fire  (Heaviside; surrogate in backward)
(iii) mem_out  = mem_t − spk_t · threshold   # reset (subtract mode)
```

where `α = exp(−dt / τ_mem)` (exact exponential discretization, `dt = 1 ms`).

Canonical parameters: `threshold`, `tau_mem`, `tau_syn`, `reset`, `delay`.

---

## Encoders

```python
from thrindex.encoders import rate, latency, delta

# Rate: Bernoulli-sample — always pass an explicit generator
gen = torch.Generator(); gen.manual_seed(0)
spikes = rate(x, T=25, generator=gen)          # [T, *x.shape]

# Latency: higher value spikes earlier — deterministic, no RNG
spikes = latency(x, T=25)

# Delta: fires on change above threshold — deterministic, no RNG
spikes = delta(x_sequence, T=25, threshold=0.1)
```

---

## Stack

```
python/thrindex/       Python SDK — strictly typed, zero logic
crates/thrindex-py/    PyO3 bridge (maturin)
crates/thrindex-sim/   Behavioral simulator — deterministic, seeded, CPU-parallel  (M2)
crates/thrindex-compiler/  Compiler: capture → validate → quantize              (M3)
crates/thrindex-numerics/  Fixed-point core — zero dependencies, consumed bit-for-bit
```

Rust engines are MIT/Apache-2.0. The Python SDK wraps them with no logic of its own.

---

## Status

M5 complete — templates, public docs, CI snippet tests, conformance suite, and energy benchmark harness are all shipped. M6 (first silicon bring-up) is next.

Issues and discussions are open.  
See [SECURITY.md](SECURITY.md) for responsible disclosure.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
