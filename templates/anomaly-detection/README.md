# Template: Anomaly Detection (Industrial Sensors)

Demonstrates spiking neural network–based anomaly detection on a 128-channel
synthetic sensor array.  The model is a binary classifier (normal vs. anomaly)
trained entirely in PyTorch with surrogate gradients and compiled to a sealed
`.thx` artifact for deployment on neuromorphic hardware.

**No external dataset download required** — the synthetic data generator is
self-contained and produces reproducible train/test splits seeded by `SEED = 0`.

## Problem

A sensor array monitors an industrial system (e.g. motor drive, airframe, reactor
coolant loop).  Normal operation occupies a compact region of the 128-dimensional
sensor space.  Faults shift specific sensor clusters in characteristic patterns.
The detector must run at the edge — low power, offline, no cloud round-trip.

## Synthetic dataset

Three fault modes are simulated, each perturbing a disjoint sensor cluster:

| Class | Label | Channels affected | Shift | Rate after tanh |
|---|---|---|---|---|
| Normal | 0 | — | Gaussian (σ=0.8) | ≈ 0.50 |
| Fault A — bearing wear | 1 | 0–31 | +3σ | ≈ 0.995 |
| Fault B — overheating | 1 | 48–79 | +5σ | ≈ 0.9999 |
| Fault C — supply fault | 1 | 96–127 | −5σ | ≈ 0.00003 |

- **Train:** 10,000 samples — 50% normal / 50% anomaly (balanced, for stable learning)
- **Test:** 2,000 samples — 85% normal / 15% anomaly (realistic operational distribution)
- Independent seeds; AUROC is distribution-invariant

Input normalisation: `tanh(x)` squashed to `[0, 1]`, then rate-encoded over
T = 100 timesteps.

> **Why −5σ for Fault C?**  Plain dropout-to-zero is undetectable with `tanh`
> normalisation: `tanh(0) = 0 → (0+1)/2 = 0.5 = normal mean`.  Shifting to
> −5σ produces a near-zero firing rate (≈ 0.00003), which is strongly distinctive.

## Architecture

```
rate_encode(128) → [T=100, batch, 128]

Dense(128 → 256) → LIF(τ_mem=10ms)
Dense(256 →  64) → LIF(τ_mem=10ms)
Dense( 64 →   2) → LIF(τ_mem=10ms)
```

- Input: 128 sensor channels, T = 100 timesteps, 1 ms each
- Output: 2 classes (0 = normal, 1 = anomaly)
- Committed floor: **AUROC ≥ 0.90**

**τ_mem = 10 ms** — shorter than the keyword-spotting template because sensor
readings are largely non-periodic; a shorter time constant lets each timestep
contribute more independently to the rate decision.

**Floor evidence:**

| Reference | AUROC | Notes |
|---|---|---|
| This template (seed 0, 30 epochs, measured) | 0.9999 | floor cleared by epoch 2 |
| Dense MLP (128→256→64→2) on same synthetic data | ~0.9999 | frame-based ceiling |
| Luo et al. 2022, CWRU bearing dataset | ~0.990 | SNN, real bearing-fault data |

Floor is set at 0.90 — well below measured performance — to allow headroom for
variation across random seeds and real industrial datasets.

## Quick demo (offline, no training needed)

```bash
pip install thrindex
thrindex run templates/anomaly-detection/model.thx --seed 0
```

> **Note:** The committed `model.thx` contains **placeholder weights** (Xavier
> initialisation, seed 0).  Run full training below to obtain a model that passes
> the AUROC floor.

## Full training

```bash
pip install thrindex
python templates/anomaly-detection/train.py --epochs 50
```

No additional libraries needed — the dataset is generated on-the-fly.
Training for 50 epochs on CPU takes roughly **5–8 minutes**;
on a GPU, under 30 seconds.

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `50` | Training epochs |
| `--results-out` | `/tmp/anomaly_results.json` | JSON results path |

## Bundled samples

`samples/` contains three pre-generated samples for offline demos:

| File | Label | Description |
|---|---|---|
| `sample_000.json` | 0 — normal | All channels within normal range |
| `sample_001.json` | 1 — anomaly | Fault A: channels 0–31 elevated (bearing wear) |
| `sample_002.json` | 1 — anomaly | Fault B: channels 48–79 elevated (overheating) |

Each JSON encodes both the raw normalised sensor reading (`[128]`) and the
rate-encoded spike train (`[T=50, 128]`).

## Pipeline

```python
import thrindex as thx
import thrindex.snn as snn
import thrindex.encoders as encoders
import torch

# Build the model
model = snn.Sequential(
    snn.Dense(128, 256),
    snn.LIF(threshold=1.0, tau_mem=10.0),
    snn.Dense(256, 64),
    snn.LIF(threshold=1.0, tau_mem=10.0),
    snn.Dense(64, 2),
    snn.LIF(threshold=1.0, tau_mem=10.0),
)

# Encode a sensor reading
gen = torch.Generator()
gen.manual_seed(0)
sensor_reading = torch.rand(1, 128)          # [batch, features]
spikes = encoders.rate(sensor_reading, T=100, generator=gen)  # [100, 1, 128]

# Inference
out = model(spikes)                          # [50, 1, 2]
anomaly_score = out.mean(0)[0, 1].item()    # mean firing rate of class-1 neuron

# After training:
thx.compile(model, "model.thx")
```

Then from the terminal:

```bash
thrindex run model.thx --seed 0
```

## Adapting to real datasets

To use this template on a real industrial dataset:

1. **Replace `_generate_dataset`** with your data loader.  Ensure sensor values
   are normalised to `[0, 1]` before calling `encoders.rate`.

2. **Adjust class balance** — most fault datasets are highly imbalanced (< 5%
   anomaly).  Add a weighted cross-entropy loss if your anomaly rate drops below
   ~10%:

   ```python
   weight = torch.tensor([1.0, pos_weight])
   loss = F.cross_entropy(rates, labels, weight=weight.to(device))
   ```

3. **Tune `tau_mem` and `T`** — sensor periodicity matters.  For vibration data
   sampled at 12 kHz, T should cover at least one full vibration cycle.

4. **Real-dataset references:**
   - [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter)
   - [MIMII Dataset](https://zenodo.org/record/3384388) (industrial machine sounds)
   - [SKAB](https://github.com/waico/SKAB) (Skoltech Anomaly Benchmark)

## Energy estimate

`thrindex run` prints a modelled energy estimate per inference.
See [docs/energy.md](../../docs/energy.md) for the synaptic-operation cost model.

At ~50% normal firing rate and Dense(128 → 256) as the dominant layer:
rate-coded anomaly detection uses roughly **2× fewer synaptic operations** than
a frame-based equivalent when normal activity dominates — the SNN is quiet when
the system is healthy.
