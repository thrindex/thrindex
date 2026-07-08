"""Integration tests: compile a model, run it through the Rust sim via PyO3.

These tests require ``thrindex._core`` to be built (``maturin develop``).
If the extension is not available, all tests are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import thrindex.snn as snn
import torch
from thrindex.compile import compile_model


def _require_core() -> None:
    try:
        import thrindex._core  # noqa: F401
    except ImportError:
        pytest.skip("thrindex._core not built — run `maturin develop --uv`")


def _simple_model() -> snn.Sequential:
    torch.manual_seed(0)
    return snn.Sequential(
        snn.Dense(8, 16),
        snn.LIF(threshold=1.0, tau_mem=10.0),
        snn.Dense(16, 4),
        snn.LIF(threshold=0.5, tau_mem=10.0),
    )


class TestCompileAndRun:
    def test_run_returns_spike_raster(self, tmp_path: Path) -> None:
        _require_core()
        from thrindex._core import run_sim  # type: ignore[import-untyped]

        model = _simple_model()
        artifact = str(tmp_path / "model.thx")
        compile_model(model, artifact)

        # Pre-generated input: [1 sample, 10 timesteps, 8 features]
        g = torch.Generator()
        g.manual_seed(0)
        spikes = torch.bernoulli(torch.full((1, 10, 8), 0.3), generator=g).tolist()

        result_spikes, stats, transcript = run_sim(artifact, spikes, 1, 0)
        assert len(result_spikes) == 1
        assert len(result_spikes[0]) == 10
        assert len(result_spikes[0][0]) == 4  # output neurons

    def test_transcript_contains_version(self, tmp_path: Path) -> None:
        _require_core()
        from thrindex._core import run_sim  # type: ignore[import-untyped]

        model = _simple_model()
        artifact = str(tmp_path / "model.thx")
        compile_model(model, artifact)
        g = torch.Generator()
        g.manual_seed(0)
        spikes = torch.bernoulli(torch.full((1, 10, 8), 0.3), generator=g).tolist()
        _, _, transcript = run_sim(artifact, spikes, 1, 0)
        assert "0.2.0" in transcript

    def test_transcript_contains_target_sim(self, tmp_path: Path) -> None:
        _require_core()
        from thrindex._core import run_sim  # type: ignore[import-untyped]

        model = _simple_model()
        artifact = str(tmp_path / "model.thx")
        compile_model(model, artifact)
        g = torch.Generator()
        g.manual_seed(0)
        spikes = torch.bernoulli(torch.full((1, 10, 8), 0.3), generator=g).tolist()
        _, _, transcript = run_sim(artifact, spikes, 1, 0)
        assert "sim" in transcript

    def test_stats_have_expected_keys(self, tmp_path: Path) -> None:
        _require_core()
        from thrindex._core import run_sim  # type: ignore[import-untyped]

        model = _simple_model()
        artifact = str(tmp_path / "model.thx")
        compile_model(model, artifact)
        g = torch.Generator()
        g.manual_seed(0)
        spikes = torch.bernoulli(torch.full((1, 10, 8), 0.3), generator=g).tolist()
        _, stats, _ = run_sim(artifact, spikes, 1, 0)
        assert "total_spikes" in stats
        assert "synaptic_ops" in stats
        assert "wall_secs" in stats

    def test_check_artifact_passes_for_valid(self, tmp_path: Path) -> None:
        _require_core()
        from thrindex._core import check_artifact  # type: ignore[import-untyped]

        model = _simple_model()
        artifact = str(tmp_path / "model.thx")
        compile_model(model, artifact)
        check_artifact(artifact)  # must not raise

    def test_check_artifact_fails_for_missing(self) -> None:
        _require_core()
        from thrindex._core import check_artifact  # type: ignore[import-untyped]

        with pytest.raises(ValueError, match="E0001"):
            check_artifact("/nonexistent/model.thx")

    def test_check_artifact_fails_for_corrupt(self, tmp_path: Path) -> None:
        _require_core()
        from thrindex._core import check_artifact  # type: ignore[import-untyped]

        bad = tmp_path / "bad.thx"
        bad.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="E0008"):
            check_artifact(str(bad))

    def test_determinism_repeated_calls(self, tmp_path: Path) -> None:
        """(a) Self-determinism via Python bridge: repeated calls → identical output."""
        _require_core()
        from thrindex._core import run_sim  # type: ignore[import-untyped]

        model = _simple_model()
        artifact = str(tmp_path / "model.thx")
        compile_model(model, artifact)
        g = torch.Generator()
        g.manual_seed(99)
        spikes = torch.bernoulli(torch.full((1, 15, 8), 0.25), generator=g).tolist()
        out_a, _, _ = run_sim(artifact, spikes, 1, 0)
        out_b, _, _ = run_sim(artifact, spikes, 1, 0)
        assert out_a == out_b
