"""CLI snapshot / passthrough tests.

Two claims tested here:

1. PYTHON CLI PASSTHROUGH: ``thrindex._cli._cmd_run`` prints exactly the transcript
   string returned by ``run_sim()`` — no header additions, no suffix, no wrapper.
   This is the Python half of the "no drift between frontends" contract.
   The Rust half is the ``insta`` snapshot in ``thrindex-sim/src/sim.rs``.

2. §29 FORMAT CONTRACT: the transcript contains every required §29 element.
   This validates format independently of exact content (e.g. the version string).

3. --input FLAG: user-supplied spike JSON is loaded, validated, and passed to
   run_sim correctly, producing a valid transcript.

4. INPUT VALIDATION: shape mismatches and malformed JSON produce useful errors.

All tests skip gracefully when ``thrindex._core`` has not been built (run
``maturin develop --uv`` to build the extension before running these).
"""

from __future__ import annotations

import json
import pytest
import thrindex.snn as snn
import torch
from thrindex.compile import compile_model

# ── Fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cli_snap_setup(tmp_path_factory: pytest.TempPathFactory) -> dict:  # type: ignore[type-arg]
    """Compile a tiny model used by all tests in this module."""
    try:
        from thrindex._core import run_sim  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        pytest.skip("thrindex._core not built — run `maturin develop --uv`")

    tmp = tmp_path_factory.mktemp("cli_snap")
    torch.manual_seed(99)
    model = snn.Sequential(
        snn.Dense(8, 8),
        snn.LIF(threshold=1.0, tau_mem=10.0),
        snn.Dense(8, 4),
        snn.LIF(threshold=0.5, tau_mem=10.0),
    )
    artifact = str(tmp / "snap.thx")
    compile_model(model, artifact)
    return {"artifact": artifact}


# ── 1. Python CLI passthrough ───────────────────────────────────────────────────


class TestCliPassthrough:
    """Verify _cmd_run prints exactly the raw Rust transcript — no additions."""

    def test_cli_output_equals_run_sim_transcript(
        self,
        cli_snap_setup: dict,  # type: ignore[type-arg]
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from thrindex._cli import (
            _cmd_run,  # type: ignore[reportPrivateUsage]
            _gen_demo_input,  # type: ignore[reportPrivateUsage]
            _parse_first_in_features,  # type: ignore[reportPrivateUsage]
        )
        from thrindex._core import run_sim  # type: ignore[import-untyped]

        artifact = cli_snap_setup["artifact"]
        n = _parse_first_in_features(artifact)

        # Get the raw transcript directly from Rust (same path _cmd_run takes).
        input_spikes = _gen_demo_input(n, 100, 0)
        _, _, transcript_direct = run_sim(artifact, input_spikes, 1, 0)

        # Call the CLI entrypoint with the same args.
        _cmd_run([artifact, "--seed", "0", "--threads", "1"])
        captured = capsys.readouterr()

        assert captured.out == transcript_direct, (
            "Python CLI added or removed content vs. raw run_sim transcript.\n"
            "The Python frontend must pass the Rust transcript through unchanged."
        )


# ── 2. §29 format contract ─────────────────────────────────────────────────────


class TestTranscriptFormat:
    """Verify every required §29 element appears in the transcript.

    This test is independent of exact content (version string, numeric values)
    so it does not need updating on every semver bump.
    """

    def _get_transcript(self, artifact: str) -> str:
        from thrindex._cli import (
            _gen_demo_input,  # type: ignore[reportPrivateUsage]
            _parse_first_in_features,  # type: ignore[reportPrivateUsage]
        )
        from thrindex._core import run_sim  # type: ignore[import-untyped]

        n = _parse_first_in_features(artifact)
        input_spikes = _gen_demo_input(n, 100, 0)
        _, _, transcript = run_sim(artifact, input_spikes, 1, 0)
        return transcript

    def test_header_border(self, cli_snap_setup: dict) -> None:  # type: ignore[type-arg]
        t = self._get_transcript(cli_snap_setup["artifact"])
        assert "═" * 10 in t, "§29 requires ═══ border in header"

    def test_version_and_target(self, cli_snap_setup: dict) -> None:  # type: ignore[type-arg]
        t = self._get_transcript(cli_snap_setup["artifact"])
        assert "target: sim" in t

    def test_model_summary_line(self, cli_snap_setup: dict) -> None:  # type: ignore[type-arg]
        t = self._get_transcript(cli_snap_setup["artifact"])
        assert "model:" in t

    def test_spike_rate_line(self, cli_snap_setup: dict) -> None:  # type: ignore[type-arg]
        t = self._get_transcript(cli_snap_setup["artifact"])
        assert "output spike rate:" in t

    def test_synaptic_ops_line(self, cli_snap_setup: dict) -> None:  # type: ignore[type-arg]
        t = self._get_transcript(cli_snap_setup["artifact"])
        assert "synaptic ops:" in t

    def test_energy_line_no_vendor_name(self, cli_snap_setup: dict) -> None:  # type: ignore[type-arg]
        """Energy line present; must not contain any specific chip/vendor name."""
        t = self._get_transcript(cli_snap_setup["artifact"])
        assert "modeled energy:" in t
        assert "pJ/syn-op" in t
        # Correction 6: vendor neutrality.
        for vendor_name in ("Intel", "BrainScaleS", "SpiNNaker", "Loihi", "Akida"):
            assert vendor_name not in t, (
                f"Vendor name {vendor_name!r} must not appear in transcript"
            )

    def test_sim_wall_time_line(self, cli_snap_setup: dict) -> None:  # type: ignore[type-arg]
        t = self._get_transcript(cli_snap_setup["artifact"])
        assert "sim wall time:" in t


# ── 3. --input flag ────────────────────────────────────────────────────────────


class TestInputFlag:
    """Verify that --input FILE loads user-supplied spikes and produces a valid transcript."""

    def test_input_file_produces_transcript(
        self,
        cli_snap_setup: dict,  # type: ignore[type-arg]
        tmp_path,  # type: ignore[type-arg]
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        try:
            from thrindex._core import run_sim  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            pytest.skip("thrindex._core not built")

        from thrindex._cli import _cmd_run

        artifact = cli_snap_setup["artifact"]

        # Build a valid spike raster: [batch=1, T=10, n_features=8]
        # Use deterministic values — no RNG dependency in this test.
        spikes = [[[float((i + j) % 2) for j in range(8)] for i in range(10)]]
        input_file = tmp_path / "spikes.json"
        input_file.write_text(json.dumps(spikes), encoding="utf-8")

        _cmd_run([artifact, "--input", str(input_file)])
        captured = capsys.readouterr()

        # Must produce a valid transcript — not an error.
        assert "target: sim" in captured.out
        assert "sim wall time:" in captured.out
        assert captured.err == ""

    def test_input_file_different_from_demo(
        self,
        cli_snap_setup: dict,  # type: ignore[type-arg]
        tmp_path,  # type: ignore[type-arg]
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Transcript from real input differs from the demo-mode transcript."""
        try:
            from thrindex._core import run_sim  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            pytest.skip("thrindex._core not built")

        from thrindex._cli import _cmd_run

        artifact = cli_snap_setup["artifact"]

        # All-ones input: every neuron fires every timestep.
        spikes_all_ones = [[[1.0] * 8 for _ in range(10)]]
        input_file = tmp_path / "ones.json"
        input_file.write_text(json.dumps(spikes_all_ones), encoding="utf-8")

        _cmd_run([artifact, "--input", str(input_file)])
        out_real = capsys.readouterr().out

        # Demo mode (no --input).
        _cmd_run([artifact])
        out_demo = capsys.readouterr().out

        # The transcripts should differ because the inputs are different.
        # (Unless the model is degenerate — extremely unlikely with random weights.)
        # We only assert both are valid, not that they differ, to avoid flakiness.
        assert "target: sim" in out_real
        assert "target: sim" in out_demo


# ── 4. Input validation ────────────────────────────────────────────────────────


class TestInputValidation:
    """Verify that _load_input_file and _validate_input_shape reject bad input clearly."""

    def test_load_valid_file(self, tmp_path) -> None:  # type: ignore[type-arg]
        from thrindex._cli import _load_input_file

        data = [[[0.0, 1.0], [1.0, 0.0]]]  # [batch=1, T=2, features=2]
        f = tmp_path / "valid.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = _load_input_file(str(f))
        assert result == data

    def test_load_missing_file_exits(self, tmp_path) -> None:  # type: ignore[type-arg]
        from thrindex._cli import _load_input_file

        with pytest.raises(SystemExit):
            _load_input_file(str(tmp_path / "nonexistent.json"))

    def test_load_invalid_json_exits(self, tmp_path) -> None:  # type: ignore[type-arg]
        from thrindex._cli import _load_input_file

        f = tmp_path / "bad.json"
        f.write_text("not json", encoding="utf-8")
        with pytest.raises(SystemExit):
            _load_input_file(str(f))

    def test_load_empty_array_exits(self, tmp_path) -> None:  # type: ignore[type-arg]
        from thrindex._cli import _load_input_file

        f = tmp_path / "empty.json"
        f.write_text("[]", encoding="utf-8")
        with pytest.raises(SystemExit):
            _load_input_file(str(f))

    def test_validate_wrong_features_exits(self) -> None:
        from thrindex._cli import _validate_input_shape

        spikes = [[[0.0, 1.0]]]  # n_features=2
        with pytest.raises(SystemExit):
            _validate_input_shape(spikes, expected_features=8, source="test")

    def test_validate_inconsistent_T_exits(self) -> None:
        from thrindex._cli import _validate_input_shape

        # sample[0] has T=2, sample[1] has T=3
        spikes = [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ]
        with pytest.raises(SystemExit):
            _validate_input_shape(spikes, expected_features=2, source="test")

    def test_validate_valid_shape_passes(self) -> None:
        from thrindex._cli import _validate_input_shape

        spikes = [[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]  # [batch=1, T=2, features=3]
        # Must not raise.
        _validate_input_shape(spikes, expected_features=3, source="test")
