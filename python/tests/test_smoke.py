"""Smoke test: the thrindex package is importable and declares the correct version.

Version must be sourced from ``pyproject.toml`` (single source of truth, correction 6).
When this test fails on a version bump, update ONLY ``pyproject.toml [project] version``.
"""

import json
import re
import time
from pathlib import Path

import pytest
import thrindex

# Resolved at import time — tests that use the template bail if the path is absent.
_TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates" / "keyword-spotting"


def test_version_is_defined() -> None:
    """Package must declare a string version."""
    assert isinstance(thrindex.__version__, str)
    assert len(thrindex.__version__) > 0


def test_version_is_semver_shaped() -> None:
    """Version must look like MAJOR.MINOR.PATCH."""
    assert re.match(r"^\d+\.\d+\.\d+$", thrindex.__version__), (
        f"version {thrindex.__version__!r} is not semver-shaped"
    )


def test_version_is_pinned() -> None:
    """Pin the exact release version, proving single-source derivation from pyproject.toml."""
    assert thrindex.__version__ == "0.3.4", (
        f"Expected '0.3.4', got {thrindex.__version__!r}. "
        "Update pyproject.toml [project] version — that is the only place."
    )


def test_snn_importable() -> None:
    """thrindex.snn must be importable as of M1."""
    import thrindex.snn as snn  # noqa: PLC0415

    assert hasattr(snn, "LIF")
    assert hasattr(snn, "Sequential")
    assert hasattr(snn, "Dense")
    assert hasattr(snn, "Conv2d")


def test_train_importable() -> None:
    """thrindex.train must be importable as of M1."""
    from thrindex.train import rate_loss  # noqa: PLC0415

    assert callable(rate_loss)


def test_encoders_importable() -> None:
    """thrindex.encoders must be importable as of M1."""
    from thrindex import encoders  # noqa: PLC0415

    assert callable(encoders.rate)
    assert callable(encoders.latency)
    assert callable(encoders.delta)


def test_compile_importable() -> None:
    """thrindex.compile must be importable as of M2."""
    assert callable(thrindex.compile)


def test_cli_importable() -> None:
    """thrindex._cli.main must be importable as of M2."""
    from thrindex._cli import main  # noqa: PLC0415

    assert callable(main)


def test_golden_path_run_under_60s() -> None:
    """Golden path: thrindex run on the keyword-spotting template completes in < 60 s.

    Closes M2 box: 'Golden path timed — CI must confirm'.

    Uses the first bundled sample directly via run_sim to avoid subprocess overhead.
    The model carries placeholder (random) weights; the sim execution time is what
    is being measured here, not accuracy.
    """
    try:
        import importlib
        importlib.import_module("thrindex._core")
    except ImportError:
        pytest.skip("thrindex._core not built — run `maturin develop`")

    from thrindex._core import run_sim  # type: ignore[import-untyped]

    model_path = _TEMPLATE_DIR / "model.thx"
    sample_path = _TEMPLATE_DIR / "samples" / "sample_000.json"

    if not model_path.exists() or not sample_path.exists():
        pytest.skip("keyword-spotting template not present")

    sample = json.loads(sample_path.read_text())
    # run_sim expects [batch, timesteps, features] as f32; wrap in batch dim and cast.
    spikes_f32 = [[float(v) for v in row] for row in sample["spikes"]]
    input_spikes = [spikes_f32]

    t0 = time.perf_counter()
    _spikes, _stats, _transcript = run_sim(str(model_path), input_spikes, 1, 0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 60.0, (
        f"Golden path run took {elapsed:.2f}s — exceeds 60s budget. "
        "Check for regression in thrindex-sim."
    )
