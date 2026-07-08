"""Behavioral equivalence tests: Python snn.Sequential vs thrindex-sim (Rust).

Correction 1 — three distinct claims tested here:
(a) Self-determinism: same .thx + same input + threads=1 → same output on repeated calls.
    (Rust-side self-determinism is tested in thrindex-sim/src/sim.rs — this file tests
    the round-trip through the Python bridge.)
(b) Python↔Rust behavioral equivalence: ≥99% prediction agreement on a fixed batch.
    Residual divergence is expected (f32 reduction-order difference, documented).
(c) Bit-identity sim↔silicon is deferred to post-RFC-004 (ADR-0007).

Correction 2: input spike trains are pre-generated ONCE (as fixtures) and fed
identically to both sides.  The sim contains no RNG.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import thrindex.snn as snn
import torch
from thrindex.compile import compile_model

# ── Fixtures ───────────────────────────────────────────────────────────────────

def _build_model(in_f: int = 16, hidden: int = 32, out_f: int = 4) -> snn.Sequential:
    torch.manual_seed(42)
    return snn.Sequential(
        snn.Dense(in_f, hidden),
        snn.LIF(threshold=1.0, tau_mem=10.0),
        snn.Dense(hidden, out_f),
        snn.LIF(threshold=0.5, tau_mem=10.0),
    )


def _bernoulli_spikes(n_samples: int, t: int, n_features: int, seed: int) -> torch.Tensor:
    """Pre-generate a fixed spike tensor — called once, not during tests."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.bernoulli(torch.full((n_samples, t, n_features), 0.2), generator=g)


@pytest.fixture(scope="module")
def equivalence_setup(tmp_path_factory: pytest.TempPathFactory) -> dict:  # type: ignore[type-arg]
    tmp = tmp_path_factory.mktemp("equiv")
    model = _build_model()
    artifact = str(tmp / "model.thx")
    compile_model(model, artifact)
    # Pre-generate spike trains ONCE.
    n_samples, T, N = 50, 20, 16
    spikes_tensor = _bernoulli_spikes(n_samples, T, N, seed=1234)
    return {"model": model, "artifact": artifact, "spikes": spikes_tensor}


# ── (a) Self-determinism through Python bridge ────────────────────────────────

class TestSelfDeterminism:
    def test_repeated_calls_identical(
        self, equivalence_setup: dict, tmp_path: Path  # type: ignore[type-arg]
    ) -> None:
        try:
            from thrindex._core import run_sim  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("thrindex._core not built — run `maturin develop`")

        artifact = equivalence_setup["artifact"]
        spikes = equivalence_setup["spikes"]
        input_list = spikes[:2].tolist()

        spikes_a, _, _ = run_sim(artifact, input_list, 1, 0)
        spikes_b, _, _ = run_sim(artifact, input_list, 1, 0)
        assert spikes_a == spikes_b, "Repeated runs must produce identical output"

    def test_threads_1_vs_4_identical(
        self, equivalence_setup: dict, tmp_path: Path  # type: ignore[type-arg]
    ) -> None:
        try:
            from thrindex._core import run_sim  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("thrindex._core not built — run `maturin develop`")

        artifact = equivalence_setup["artifact"]
        spikes = equivalence_setup["spikes"]
        input_list = spikes[:4].tolist()

        spikes_1, _, _ = run_sim(artifact, input_list, 1, 0)
        spikes_4, _, _ = run_sim(artifact, input_list, 4, 0)
        assert spikes_1 == spikes_4, (
            "threads=1 and threads=4 must produce identical spike rasters (correction 8)"
        )


# ── (b) Python↔Rust behavioral equivalence ────────────────────────────────────

class TestBehavioralEquivalence:
    """≥99% prediction agreement between Python and Rust on the same inputs.

    Residual divergence is expected and documented: Python uses torch.mm (BLAS with
    vendor-specific reduction order); Rust uses a naive loop (different order).
    f32 is non-associative; both are computing the correct operation, with different
    rounding accumulation.  This is not a bug.
    """

    def _python_predict(self, model: snn.Sequential, sample: torch.Tensor) -> int:
        """Predict class index from mean firing rate using Python snn.Sequential."""
        with torch.no_grad():
            # sample: [T, N] → expand to [T, 1, N]
            out = model(sample.unsqueeze(1))  # [T, 1, out_features]
        rates = out.mean(dim=0).squeeze(0)  # [out_features]
        return int(rates.argmax().item())

    def _rust_predict(self, artifact: str, sample: torch.Tensor) -> int:
        from thrindex._core import run_sim  # type: ignore[import-untyped]

        # sample: [T, N] → batch of 1: [1, T, N]
        input_list = sample.unsqueeze(0).tolist()
        spikes, _, _ = run_sim(artifact, input_list, 1, 0)
        rates = torch.tensor(spikes[0]).mean(dim=0)  # [out_features]
        return int(rates.argmax().item())

    def test_prediction_agreement_geq_99pct(
        self, equivalence_setup: dict  # type: ignore[type-arg]
    ) -> None:
        try:
            from thrindex._core import run_sim  # noqa: F401
        except ImportError:
            pytest.skip("thrindex._core not built — run `maturin develop`")

        model = equivalence_setup["model"]
        artifact = equivalence_setup["artifact"]
        spikes = equivalence_setup["spikes"]  # [50, 20, 16]

        agreements = 0
        total = spikes.shape[0]
        spike_count_deltas: list[int] = []

        for i in range(total):
            sample = spikes[i]  # [T, N]
            py_pred = self._python_predict(model, sample)
            rust_pred = self._rust_predict(artifact, sample)
            if py_pred == rust_pred:
                agreements += 1

            # Spike count delta (for documentation).
            with torch.no_grad():
                py_spikes = model(sample.unsqueeze(1))
            py_count = int(py_spikes.sum().item())
            from thrindex._core import run_sim  # type: ignore[import-untyped]

            rust_spikes_raw, _, _ = run_sim(artifact, sample.unsqueeze(0).tolist(), 1, 0)
            rust_count = sum(
                sum(1 for v in frame if v > 0.5)
                for frame in rust_spikes_raw[0]
            )
            spike_count_deltas.append(abs(py_count - rust_count))

        agreement_rate = agreements / total
        max_delta = max(spike_count_deltas) if spike_count_deltas else 0
        avg_delta = sum(spike_count_deltas) / len(spike_count_deltas) if spike_count_deltas else 0.0

        # Document the residual divergence.
        print(
            f"\nPython↔Rust equivalence: {agreements}/{total} ({agreement_rate:.1%}) agree. "
            f"Spike-count delta: max={max_delta}, avg={avg_delta:.2f}. "
            "Residual divergence is f32 reduction-order; see ADR-0007."
        )

        assert agreement_rate >= 0.99, (
            f"Expected ≥99% prediction agreement, got {agreement_rate:.1%}. "
            "If this fails, check for a LIF dynamics bug, not just rounding."
        )
