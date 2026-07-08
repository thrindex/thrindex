# Modeled Energy Estimates

`thrindex run` prints a modeled energy estimate alongside the spike raster.
This document defines the formula, the default coefficient, and its provenance.

## Formula

```
E = Σ_layer  (pre_spikes_in_layer × fan_out_of_layer)  ×  coefficient
```

Where:

| Symbol | Meaning |
|--------|---------|
| `pre_spikes_in_layer` | Number of spike events from the pre-synaptic population across all timesteps |
| `fan_out_of_layer` | Number of post-synaptic connections per neuron (= output dimension for Dense layers) |
| `coefficient` | Energy per synaptic operation, in pJ (default: **0.5 pJ/syn-op**) |

The product `pre_spikes × fan_out` is the number of **synaptic operations** (multiply-accumulates) triggered by spikes.  This is the standard metric for neuromorphic energy.

## Default coefficient

The default of **0.5 pJ/syn-op** is a round figure consistent with published
neuromorphic silicon figures from multiple implementations.  It is not calibrated to
any specific device.

To override: `thrindex run model.thx --energy-coeff 1.2` (units: pJ/syn-op).

## Example calculation

A `Dense(700→512) → LIF → Dense(512→20) → LIF` model, 100 timesteps, 10% spike rate:

```
Layer 1 (Dense 700→512):
  pre_spikes  = 700 neurons × 100 T × 10% = 7,000
  fan_out     = 512
  ops         = 7,000 × 512 = 3,584,000

Layer 2 (LIF 512):  no synaptic ops

Layer 3 (Dense 512→20):
  pre_spikes  = 512 neurons × 100 T × 10% = 5,120
  fan_out     = 20
  ops         = 5,120 × 20 = 102,400

Layer 4 (LIF 20):  no synaptic ops

Total ops   = 3,584,000 + 102,400 = 3,686,400
Energy      = 3,686,400 × 0.5 pJ = 1,843,200 pJ ≈ 1.84 µJ
```

The CLI prints both the op count and the energy estimate so the coefficient's impact
is visible.

## Interpreting the estimate

This is a **model**, not a measurement.  It captures the spike-driven, event-based
efficiency advantage of neuromorphic computing: energy scales with spike count, not
with model parameter count.  A fully-dense ANN layer with the same parameter count
would perform all multiplications every timestep, consuming roughly `1/spike_rate`× more
energy for the same architecture.

For hardware-measured power, see the device datasheet for the specific target.
