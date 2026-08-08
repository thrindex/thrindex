#!/usr/bin/env python3
"""
crates/thrindex-backends/akida/python/spike_test.py

One-time probe: verify that a manually-built Akida Dense layer set via
set_variable() survives model.map(AKD1500(), hw_only=True).

Status: SPIKE_TEST_RESULT: SUCCESS (confirmed aarch64, Python 3.11.9, akida 2.19.2)

Run on a machine with the `akida` package installed (pip install akida).
Expected runtime: < 5 minutes.
Output: SPIKE_TEST_RESULT line to stdout.
Exit codes: 0 = success, 1 = failure, 2 = skip (akida not installed).
"""
import sys
import numpy as np

try:
    from akida import Model, AKD1500
    from akida.layers.input_data import InputData
    from akida.layers.fully_connected import FullyConnected
except ImportError as e:
    print(f"SPIKE_TEST_RESULT: SKIP - akida package not available: {e}")
    sys.exit(2)

# Construct the simplest possible Akida model manually.
# InputData layer is required first; declares spatial shape and 4-bit input encoding.
model = Model()
model.add(InputData(input_shape=(1, 1, 4), input_bits=4))

# FullyConnected layer with 4-bit weights and activations (confirmed constraint).
layer = FullyConnected(units=8, name="dense_0", weights_bits=4, act_bits=4)
model.add(layer)

# Set weights to a known-good value via set_variable().
# int8 container for 4-bit values; range [-7, 7]; shape (1, 1, in_features, out_features).
W = np.zeros((1, 1, 4, 8), dtype=np.int8)
try:
    layer.set_variable("weights", W)
except Exception as e:
    print(f"SPIKE_TEST_RESULT: FAILURE (set_variable) - {type(e).__name__}: {e}")
    print("NEXT_STEP: item 7 must use Model.from_json() with documented JSON schema")
    sys.exit(1)

# Attempt to map to virtual AKD1500.
try:
    model.map(AKD1500(), hw_only=True)
    buf = model.to_buffer()
    print(f"SPIKE_TEST_RESULT: SUCCESS - {len(buf)} bytes produced")
    print("NEXT_STEP: item 7 may use the layer.set_variable() path directly")
    sys.exit(0)
except Exception as e:
    print(f"SPIKE_TEST_RESULT: FAILURE (map) - {type(e).__name__}: {e}")
    print("NEXT_STEP: item 7 must use Model.from_json() with documented JSON schema")
    sys.exit(1)
