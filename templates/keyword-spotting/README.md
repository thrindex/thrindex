# Template: Keyword Spotting (SHD)

Demonstrates the full thrindex pipeline on the
[Spiking Heidelberg Digits (SHD)](https://zenkelab.org/resources/spiking-heidelberg-datasets/)
dataset.

**Reference:** Cramer et al. 2020 — [Heidelberg Spiking Datasets](https://arxiv.org/abs/1910.07407)

## Architecture

```
Dense(700 → 512) → LIF(τ_mem=20ms) → Dense(512 → 20) → LIF(τ_mem=20ms)
```

- Input: 700 cochlea channels (SHD format), 100 timesteps (1ms each)
- Output: 20 classes (spoken digits 0–9, English and German)
- Committed floor: **≥ 60%**

**Floor evidence (feedforward, non-recurrent references only):**

| Reference | Accuracy | Architecture |
|---|---|---|
| Cramer et al. 2020 ([doi](https://doi.org/10.1109/TNNLS.2020.3044364)) | 48.1 ± 1.6% | feedforward SNN, 1 hidden layer |
| Zheng et al. 2025 (OpenReview) | 69.0 ± 5.8% | feedforward LIF, no sparsity/init conditions |
| Zheng et al. 2025 (OpenReview) | 75.8 ± 3.1% | feedforward LIF, with conditions |

All recurrent-network SHD results (≥ 71.4%) are excluded — recurrent vs.
feedforward is not an apples-to-apples comparison. Our architecture (2-layer,
512 hidden, no recurrence, no delays) is comparable to the "without conditions"
Zheng et al. result. Floor is set with conservative margin below 69.0% − 1σ.

## Quick demo (offline, no training needed)

```bash
pip install thrindex
thrindex run templates/keyword-spotting/model.thx --seed 0
```

> **Note:** `model.thx` ships with **random (seeded) weights** as a pipeline demo.
> Predictions are not meaningful.  Run `train.py` for benchmark weights.

## Full training

```bash
pip install thrindex h5py
python templates/keyword-spotting/train.py --data-dir /tmp/shd --epochs 100
```

The script downloads SHD (~280 MB) automatically and writes the trained model to
`model.thx` when training is complete.

## Bundled samples

`samples/` contains three pre-generated SHD-format spike trains (JSON).
These let `thrindex run` work fully offline without downloading the dataset.

## Pipeline

```python
import thrindex as thx
import thrindex.snn as snn

model = snn.Sequential(
    snn.Dense(700, 512),
    snn.LIF(threshold=1.0, tau_mem=20.0),
    snn.Dense(512, 20),
    snn.LIF(threshold=1.0, tau_mem=20.0),
)

# After training:
thx.compile(model, "model.thx")
```

Then from the terminal:

```bash
thrindex run model.thx --seed 0
```

## Energy estimate

`thrindex run` prints a modeled energy estimate.  See [docs/energy.md](../../docs/energy.md)
for the formula and coefficient.
