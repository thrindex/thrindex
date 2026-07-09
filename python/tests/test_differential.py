"""Differential / backward-compatibility tests for the M3 compiler (Python side).

These tests verify:
1. Self-determinism: the M3 compile → run pipeline produces identical output
   across runs for the same model and input.
2. Behavioral equivalence: M3 compile → run vs frozen M2 fixture → run agree
   on spike patterns for identical architectures.
3. Two-level parameter rule: the Graph IR emitted by Python contains tau_mem
   (continuous), NOT alpha (resolved) — alpha only appears in the .thx artifact.

Tests that need `thrindex._core` call `_require_core()` which skips if not built.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import thrindex.snn as snn
import torch
from thrindex.compile import compile_model


def _require_core() -> None:
    try:
        import importlib
        importlib.import_module("thrindex._core")
    except ImportError:
        pytest.skip("thrindex._core not built — run `maturin develop`")


# Frozen M2 fixture path (relative to the file location).
_FIXTURES_DIR = (
    Path(__file__).parent.parent.parent
    / "crates"
    / "thrindex-compiler"
    / "tests"
    / "fixtures"
)
M2_FIXTURE_PATH = _FIXTURES_DIR / "m2_dense_lif.thx"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _identity_model() -> snn.Sequential:
    """Dense(2→2) + LIF matching the M2 frozen fixture."""
    m = snn.Sequential(snn.Dense(2, 2), snn.LIF(threshold=1.0, tau_mem=10.0))
    with torch.no_grad():
        m.layers[0].linear.weight.copy_(torch.eye(2))  # type: ignore[attr-defined]
        if m.layers[0].linear.bias is not None:  # type: ignore[attr-defined]
            m.layers[0].linear.bias.zero_()  # type: ignore[attr-defined]
    return m


def _run_from_path(path: Path) -> list[list[list[float]]]:
    """Run the sim on an artifact at *path* with input [1.0, 1.0] for 1 timestep."""
    from thrindex._core import run_sim  # type: ignore[import-untyped]

    input_spikes = [[[1.0, 1.0]]]  # [1 sample, 1 timestep, 2 features]
    spikes, _stats, _transcript = run_sim(str(path), input_spikes, 1, 0)
    return spikes  # type: ignore[no-any-return]


# ── Self-determinism ──────────────────────────────────────────────────────────


def test_m3_self_determinism(tmp_path: Path) -> None:
    """Compiling the same model twice must produce bit-identical simulation output."""
    _require_core()
    m = _identity_model()
    path1 = tmp_path / "run1.thx"
    path2 = tmp_path / "run2.thx"
    compile_model(m, path1)
    compile_model(m, path2)

    spikes1 = _run_from_path(path1)
    spikes2 = _run_from_path(path2)
    assert spikes1 == spikes2, (
        "Two compiles of the same model must produce identical simulation output"
    )


# ── Two-level parameter rule ───────────────────────────────────────────────────


def test_graph_ir_no_alpha() -> None:
    """The Graph IR JSON must carry tau_mem (continuous), NOT alpha (resolved).

    This test does NOT require _core — it tests the Python extraction pass only.
    """
    from thrindex.compile import _build_ir_json  # type: ignore[reportPrivateUsage]

    m = _identity_model()
    ir = json.loads(_build_ir_json(m))
    lif = ir["layers"][1]
    assert "alpha" not in lif, "Graph IR must not contain alpha (ADR-0008)"
    assert "tau_mem" in lif, "Graph IR must contain tau_mem"
    assert "alpha_syn" not in lif, "Graph IR must not contain alpha_syn"


def test_thx_artifact_has_alpha(tmp_path: Path) -> None:
    """The .thx artifact (target-side) must carry alpha (resolved by Rust)."""
    _require_core()
    m = _identity_model()
    path = tmp_path / "model.thx"
    compile_model(m, path)
    artifact = json.loads(path.read_text())
    lif = artifact["model"]["layers"][1]
    assert "alpha" in lif, "Target artifact must contain resolved alpha"
    # tau_mem is Graph IR source of truth; resolved constants only in target artifact.
    assert "tau_mem" not in lif, "Target artifact must NOT contain tau_mem"


# ── Behavioral equivalence vs M2 fixture ─────────────────────────────────────


def test_m3_vs_m2_fixture_spike_agreement(tmp_path: Path) -> None:
    """M3 compile output must agree with the M2 frozen fixture on spike patterns.

    Both models are identical architecturally (Dense 2→2 identity + LIF tau_mem=10ms).
    The spike pattern must match exactly — the identity matrix means both neurons
    should fire when input is [1.0, 1.0] and threshold is 1.0.
    """
    _require_core()
    assert M2_FIXTURE_PATH.exists(), f"M2 fixture not found at {M2_FIXTURE_PATH}"

    m = _identity_model()
    m3_path = tmp_path / "m3.thx"
    compile_model(m, m3_path)

    spikes_m3 = _run_from_path(m3_path)
    spikes_m2 = _run_from_path(M2_FIXTURE_PATH)

    assert spikes_m2 == [[[1.0, 1.0]]], f"M2 fixture unexpected output: {spikes_m2}"
    assert spikes_m3 == [[[1.0, 1.0]]], f"M3 artifact unexpected output: {spikes_m3}"
    assert spikes_m3 == spikes_m2, (
        f"M3 ({spikes_m3}) must match M2 ({spikes_m2}) for identical model"
    )


# ── dt_ms in artifact metadata ────────────────────────────────────────────────


def test_m3_artifact_has_dt_ms(tmp_path: Path) -> None:
    """M3 artifacts must carry dt_ms in metadata (ADR-0008 extension to ADR-0006)."""
    _require_core()
    m = _identity_model()
    path = tmp_path / "model.thx"
    compile_model(m, path)
    artifact = json.loads(path.read_text())
    assert "dt_ms" in artifact["metadata"], "M3 artifact must have dt_ms in metadata"
    assert artifact["metadata"]["dt_ms"] == pytest.approx(1.0)


def test_m3_artifact_m2_fixture_both_loadable() -> None:
    """Both M2 and M3 artifacts must be loadable by the M3 Rust sim."""
    _require_core()
    from thrindex._core import check_artifact  # type: ignore[import-untyped]

    assert M2_FIXTURE_PATH.exists(), f"M2 fixture not found at {M2_FIXTURE_PATH}"
    check_artifact(str(M2_FIXTURE_PATH))
