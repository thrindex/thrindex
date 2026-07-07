"""Smoke test: the thrindex package is importable and declares the correct version.

Version must be sourced from ``pyproject.toml`` (single source of truth, correction 6).
When this test fails on a version bump, update ONLY ``pyproject.toml [project] version``.
"""

import re

import thrindex


def test_version_is_defined() -> None:
    """Package must declare a string version."""
    assert isinstance(thrindex.__version__, str)
    assert len(thrindex.__version__) > 0


def test_version_is_semver_shaped() -> None:
    """Version must look like MAJOR.MINOR.PATCH."""
    assert re.match(r"^\d+\.\d+\.\d+$", thrindex.__version__), (
        f"version {thrindex.__version__!r} is not semver-shaped"
    )


def test_version_is_0_1_0() -> None:
    """Pin the exact M1 version, proving single-source derivation from pyproject.toml."""
    assert thrindex.__version__ == "0.1.0", (
        f"Expected '0.1.0', got {thrindex.__version__!r}. "
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
