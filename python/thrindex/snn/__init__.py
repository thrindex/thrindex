"""``thrindex.snn`` — spiking neural network layers and containers.

Mirrors the ``torch.nn`` namespace so that an ML engineer's existing muscle
memory transfers immediately (Playbook §27, law 1).

Canonical import::

    import thrindex.snn as snn

    model = snn.Sequential(
        snn.Dense(784, 1000),
        snn.LIF(threshold=1.0, tau_mem=5.0),
        snn.Dense(1000, 10),
        snn.LIF(threshold=1.0, tau_mem=5.0),
    )
"""

from thrindex.snn.conv2d import Conv2d
from thrindex.snn.dense import Dense
from thrindex.snn.lif import LIF, LIFState
from thrindex.snn.sequential import Sequential
from thrindex.snn.surrogate import FastSigmoid

__all__ = ["Conv2d", "Dense", "FastSigmoid", "LIF", "LIFState", "Sequential"]
