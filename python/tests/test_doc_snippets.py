"""CI guard for public documentation code snippets.

Every test in this file executes a snippet that appears verbatim (or structurally
equivalent) in the public thrindex-docs pages.  A test failure means the
corresponding doc page is incorrect and must be updated before merge.

Coverage map
------------
getting-started/quickstart.mdx          → TestQuickstart
api/thrindex-encoders.mdx               → TestEncoderSnippets
tutorials/keyword-spotting.mdx          → TestKeywordSpottingSnippets
tutorials/event-camera.mdx              → TestEventCameraSnippets
tutorials/anomaly-detection.mdx         → TestAnomalyDetectionSnippets

Tests that require the compiled Rust extension (thrindex._core) call
_require_core() at the top, which skips them if the extension is absent.
Run `maturin develop` or `pip install -e .` to build it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import thrindex
import thrindex.snn as snn
import torch

# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_core() -> None:
    try:
        import importlib
        importlib.import_module("thrindex._core")
    except ImportError:
        pytest.skip("thrindex._core not built — run `maturin develop`")


# ── getting-started/quickstart.mdx ────────────────────────────────────────────

class TestQuickstart:
    """Snippet: 'Write a model' and 'Compile' sections of quickstart.mdx."""

    def test_model_build(self) -> None:
        """The exact 4-layer quickstart model must construct without error."""
        model = snn.Sequential(
            snn.Dense(700, 512),
            snn.LIF(tau_mem=20.0, threshold=0.3),
            snn.Dense(512, 20),
            snn.LIF(tau_mem=20.0, threshold=0.3),
        )

        assert len(model.layers) == 4
        assert isinstance(model.layers[0], snn.Dense)
        assert isinstance(model.layers[1], snn.LIF)
        assert model.layers[1].tau_mem == pytest.approx(20.0)
        assert model.layers[1].threshold == pytest.approx(0.3)

    def test_compile(self, tmp_path: Path) -> None:
        """thx.compile(model, path) — the quickstart compile step."""
        _require_core()

        model = snn.Sequential(
            snn.Dense(700, 512),
            snn.LIF(tau_mem=20.0, threshold=0.3),
            snn.Dense(512, 20),
            snn.LIF(tau_mem=20.0, threshold=0.3),
        )

        output = tmp_path / "model.thx"
        thrindex.compile(model, output)

        assert output.exists()
        artifact = json.loads(output.read_text())
        assert artifact["target"] == "sim"
        assert len(artifact["model"]["layers"]) == 4

    def test_run(self, tmp_path: Path) -> None:
        """thrindex run — compile then execute via run_sim.

        Mirrors `thrindex run model.thx` from the quickstart CLI section.
        Uses run_sim directly (same code path as the CLI) to avoid subprocess
        overhead in CI.
        """
        _require_core()
        from thrindex._core import run_sim  # type: ignore[import-untyped]

        model = snn.Sequential(
            snn.Dense(700, 512),
            snn.LIF(tau_mem=20.0, threshold=0.3),
            snn.Dense(512, 20),
            snn.LIF(tau_mem=20.0, threshold=0.3),
        )
        output = tmp_path / "model.thx"
        thrindex.compile(model, output)

        # Provide one zero-spike input sample: [1 batch, T=100, N_IN=700]
        input_spikes = [[[0.0] * 700 for _ in range(100)]]
        _spikes, _stats, transcript = run_sim(str(output), input_spikes, 1, 0)

        assert isinstance(transcript, str)
        assert len(transcript) > 0
        # Transcript must contain key §29 fields — version token and target.
        assert "thrindex" in transcript
        assert "target: sim" in transcript
        assert "prediction:" in transcript
        assert "synaptic ops:" in transcript


# ── api/thrindex-encoders.mdx ─────────────────────────────────────────────────

class TestEncoderSnippets:
    """Snippets from api/thrindex-encoders.mdx."""

    def test_rate_encoder(self) -> None:
        """Rate encoder requires explicit Generator — snippet from encoders API doc."""
        from thrindex import encoders  # noqa: PLC0415

        gen = torch.Generator()
        gen.manual_seed(42)

        x = torch.rand(4, 128)                  # [batch, features]
        spikes = encoders.rate(x, T=25, generator=gen)

        assert spikes.shape == (25, 4, 128)
        assert spikes.dtype == torch.float32
        # Rate encoder output must be binary {0, 1}.
        assert ((spikes == 0.0) | (spikes == 1.0)).all()

    def test_latency_encoder(self) -> None:
        """Latency encoder — deterministic, fires once per feature."""
        from thrindex import encoders  # noqa: PLC0415

        x = torch.rand(4, 64)
        spikes = encoders.latency(x, T=50)

        assert spikes.shape == (50, 4, 64)
        # Each spatial location must fire at most once.
        assert (spikes.sum(dim=0) <= 1).all()

    def test_delta_encoder(self) -> None:
        """Delta encoder — fires on absolute change above threshold."""
        from thrindex import encoders  # noqa: PLC0415

        # Increasing ramp — difference is constant, above default threshold 0.1.
        x = torch.linspace(0.0, 1.0, steps=20).unsqueeze(1)  # [T=20, 1]
        x = x.unsqueeze(1).expand(20, 4, 1)                  # [T, batch, features]
        spikes = encoders.delta(x, T=20, threshold=0.05)

        assert spikes.shape == (20, 4, 1)
        assert spikes.dtype == torch.float32

    def test_rate_encoder_clamping(self) -> None:
        """Values outside [0, 1] must be clamped before sampling."""
        from thrindex import encoders  # noqa: PLC0415

        gen = torch.Generator()
        gen.manual_seed(0)
        x = torch.tensor([[2.0, -1.0]])  # out-of-range values
        spikes = encoders.rate(x, T=10, generator=gen)
        # Output must still be binary — clamped Bernoulli.
        assert ((spikes == 0.0) | (spikes == 1.0)).all()


# ── tutorials/keyword-spotting.mdx ────────────────────────────────────────────

class TestKeywordSpottingSnippets:
    """Snippet: model definition and training step from keyword-spotting.mdx."""

    def test_model_definition(self) -> None:
        """SHD model from the keyword-spotting tutorial."""
        from thrindex.train import rate_loss  # noqa: PLC0415

        model = snn.Sequential(
            snn.Dense(700, 512),
            snn.LIF(tau_mem=20.0, threshold=0.3),
            snn.Dense(512, 20),
            snn.LIF(tau_mem=20.0, threshold=0.3),
        )

        # Single forward pass with random binary input.
        x = torch.bernoulli(torch.full((100, 2, 700), 0.01))  # [T, batch, N]
        out = model(x)
        assert out.shape == (100, 2, 20)

        y = torch.randint(0, 20, (2,))
        loss = rate_loss(out, y)
        assert loss.item() > 0.0

    def test_compile(self, tmp_path: Path) -> None:
        """thx.compile(model, path) from keyword-spotting compile section."""
        _require_core()

        model = snn.Sequential(
            snn.Dense(700, 512),
            snn.LIF(tau_mem=20.0, threshold=0.3),
            snn.Dense(512, 20),
            snn.LIF(tau_mem=20.0, threshold=0.3),
        )
        path = tmp_path / "kws.thx"
        thrindex.compile(model, path)
        assert path.exists()


# ── tutorials/event-camera.mdx ────────────────────────────────────────────────

class TestEventCameraSnippets:
    """Snippets from tutorials/event-camera.mdx."""

    def test_model_definition(self) -> None:
        """N-MNIST event-camera model — Dense-based, all layers compile."""
        model = snn.Sequential(
            snn.Dense(2312, 1024),
            snn.LIF(tau_mem=20.0, threshold=0.5),
            snn.Dense(1024, 256),
            snn.LIF(tau_mem=20.0, threshold=0.5),
            snn.Dense(256, 10),
            snn.LIF(tau_mem=20.0, threshold=0.5),
        )

        assert len(model.layers) == 6
        # Verify input/output dimensions via a forward pass.
        x = torch.zeros(20, 2, 2312)  # [T=20, batch=2, N_IN=2312]
        out = model(x)
        assert out.shape == (20, 2, 10)

    def test_hyperparameters(self) -> None:
        """LIF parameters from event-camera.mdx must match train.py constants."""
        model = snn.Sequential(
            snn.Dense(2312, 1024),
            snn.LIF(tau_mem=20.0, threshold=0.5),
            snn.Dense(1024, 256),
            snn.LIF(tau_mem=20.0, threshold=0.5),
            snn.Dense(256, 10),
            snn.LIF(tau_mem=20.0, threshold=0.5),
        )
        for layer in model.layers:
            if isinstance(layer, snn.LIF):
                assert layer.tau_mem == pytest.approx(20.0)
                assert layer.threshold == pytest.approx(0.5)
                expected_alpha = math.exp(-1.0 / 20.0)
                assert layer.alpha == pytest.approx(expected_alpha, abs=1e-6)

    def test_compile(self, tmp_path: Path) -> None:
        """thx.compile from event-camera compile section."""
        _require_core()

        model = snn.Sequential(
            snn.Dense(2312, 1024),
            snn.LIF(tau_mem=20.0, threshold=0.5),
            snn.Dense(1024, 256),
            snn.LIF(tau_mem=20.0, threshold=0.5),
            snn.Dense(256, 10),
            snn.LIF(tau_mem=20.0, threshold=0.5),
        )
        path = tmp_path / "ec.thx"
        thrindex.compile(model, path)
        artifact = json.loads(path.read_text())
        # Verify layer structure: Dense LIF Dense LIF Dense LIF
        types = [lay["type"] for lay in artifact["model"]["layers"]]
        assert types == ["dense", "lif", "dense", "lif", "dense", "lif"]

    def test_preprocessing_flatten(self) -> None:
        """Preprocessing snippet: event frames flattened to [T, 2312]."""
        T, C, H, W = 20, 2, 34, 34
        # Simulate tonic ToFrame output: [T, C, H, W], binary {0, 1}
        frames = torch.zeros(T, C, H, W)
        frames[3, 0, 10, 15] = 1.0   # one ON event at t=3
        frames[7, 1, 20, 8] = 1.0    # one OFF event at t=7

        frames_flat = frames.reshape(T, C * H * W)  # [20, 2312]

        assert frames_flat.shape == (T, C * H * W)
        assert frames_flat[3, 0 * H * W + 10 * W + 15] == 1.0
        assert frames_flat[7, 1 * H * W + 20 * W + 8] == 1.0


# ── tutorials/anomaly-detection.mdx ──────────────────────────────────────────

class TestAnomalyDetectionSnippets:
    """Snippets from tutorials/anomaly-detection.mdx."""

    def test_model_definition(self) -> None:
        """Anomaly-detection binary classifier from the tutorial."""
        model = snn.Sequential(
            snn.Dense(128, 256),
            snn.LIF(tau_mem=10.0, threshold=1.0),
            snn.Dense(256, 64),
            snn.LIF(tau_mem=10.0, threshold=1.0),
            snn.Dense(64, 2),
            snn.LIF(tau_mem=10.0, threshold=1.0),
        )
        assert len(model.layers) == 6
        x = torch.zeros(100, 4, 128)  # [T=100, batch=4, N_IN=128]
        out = model(x)
        assert out.shape == (100, 4, 2)

    def test_tanh_encoding_snippet(self) -> None:
        """tanh normalisation + rate encoder from the tutorial."""
        from thrindex import encoders  # noqa: PLC0415

        x = torch.randn(4, 128)                     # raw sensor readings

        x_norm = (x.tanh() + 1.0) / 2.0            # squash to [0, 1]
        assert x_norm.min() >= 0.0
        assert x_norm.max() <= 1.0

        gen = torch.Generator()
        gen.manual_seed(0)
        spikes = encoders.rate(x_norm, T=100, generator=gen)  # [100, 4, 128]
        assert spikes.shape == (100, 4, 128)

    def test_fault_c_distinguishability(self) -> None:
        """tanh(-5σ) must be near zero — the doc explains why plain dropout fails."""
        # Fault C: channels suppressed to -5σ
        fault_c_raw = torch.full((32,), -5.0)
        fault_c_norm = (fault_c_raw.tanh() + 1.0) / 2.0
        # Near-zero rate — clearly distinct from normal mean of 0.50
        assert fault_c_norm.max() < 0.01

        # Plain dropout-to-zero maps to 0.50 — indistinguishable from normal mean
        zero_raw = torch.zeros(32)
        zero_norm = (zero_raw.tanh() + 1.0) / 2.0
        assert zero_norm.mean() == pytest.approx(0.5, abs=1e-6)

    def test_training_step_snippet(self) -> None:
        """One training step from the tutorial — loss + backward + step."""
        import torch.nn.functional as F  # noqa: PLC0415
        from thrindex import encoders  # noqa: PLC0415

        model = snn.Sequential(
            snn.Dense(128, 256),
            snn.LIF(tau_mem=10.0, threshold=1.0),
            snn.Dense(256, 64),
            snn.LIF(tau_mem=10.0, threshold=1.0),
            snn.Dense(64, 2),
            snn.LIF(tau_mem=10.0, threshold=1.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(8, 128)
        y = torch.randint(0, 2, (8,))

        x_norm = (x.tanh() + 1.0) / 2.0
        gen = torch.Generator()
        gen.manual_seed(0)
        spikes = encoders.rate(x_norm, T=100, generator=gen)  # [100, 8, 128]

        out = model(spikes)              # [100, 8, 2]
        rates = out.mean(dim=0)          # [8, 2]
        loss = F.cross_entropy(rates, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()

        assert loss.item() > 0.0

    def test_compile(self, tmp_path: Path) -> None:
        """thx.compile from anomaly-detection compile section."""
        _require_core()

        model = snn.Sequential(
            snn.Dense(128, 256),
            snn.LIF(tau_mem=10.0, threshold=1.0),
            snn.Dense(256, 64),
            snn.LIF(tau_mem=10.0, threshold=1.0),
            snn.Dense(64, 2),
            snn.LIF(tau_mem=10.0, threshold=1.0),
        )
        path = tmp_path / "ad.thx"
        thrindex.compile(model, path)
        artifact = json.loads(path.read_text())
        types = [lay["type"] for lay in artifact["model"]["layers"]]
        assert types == ["dense", "lif", "dense", "lif", "dense", "lif"]
        # First Dense must have correct dimensions
        d0 = artifact["model"]["layers"][0]
        assert d0["in_features"] == 128
        assert d0["out_features"] == 256
