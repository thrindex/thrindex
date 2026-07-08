"""CLI snapshot / passthrough tests.

Two claims tested here:

1. PYTHON CLI PASSTHROUGH: ``thrindex._cli._cmd_run`` prints exactly the transcript
   string returned by ``run_sim()`` — no header additions, no suffix, no wrapper.
   This is the Python half of the "no drift between frontends" contract.
   The Rust half is the ``insta`` snapshot in ``thrindex-sim/src/sim.rs``.

2. §29 FORMAT CONTRACT: the transcript contains every required §29 element.
   This validates format independently of exact content (e.g. the version string).

Both tests skip gracefully when ``thrindex._core`` has not been built (run
``maturin develop --uv`` to build the extension before running these).
"""

from __future__ import annotations

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
        from thrindex._cli import _cmd_run, _gen_demo_input, _parse_first_in_features
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
        from thrindex._cli import _gen_demo_input, _parse_first_in_features
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
