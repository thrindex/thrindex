"""Unit tests for LIF neuron dynamics, encoders, Dense, Conv2d, Sequential, and rate_loss.

All LIF tests use hand-computed expected values.  The update order tested here is
(ADR-0005, correction 2):
    (i)   mem_t   = alpha * mem_prev + x_t
    (ii)  spk_t   = H(mem_t − threshold)
    (iii) mem_out = mem_t − spk_t * threshold      (subtract mode)
          OR
          mem_out = mem_t * (1 − spk_t.detach())   (zero mode)
where alpha = exp(−dt / tau_mem), dt = 1.0 ms.
"""

from __future__ import annotations

import math

import pytest
import thrindex.snn as snn
import torch
from thrindex.encoders import delta, latency, rate
from thrindex.snn.lif import LIF, LIFState
from thrindex.train import rate_loss

# ── LIF dynamics ───────────────────────────────────────────────────────────────

THRESHOLD = 2.0
TAU_MEM = 10.0
ALPHA = math.exp(-1.0 / TAU_MEM)  # ≈ 0.904837


class TestLIFDynamics:
    """Verify the per-timestep update order with hand-computed expected values."""

    def _lif(self, reset: str = "subtract") -> LIF:
        return LIF(threshold=THRESHOLD, tau_mem=TAU_MEM, reset=reset)

    def _zero_state(self, batch: int = 1, features: int = 1) -> LIFState:
        return LIFState(mem=torch.zeros(batch, features))

    # ── firing ──

    def test_fires_above_threshold(self) -> None:
        """x_t > threshold with zero initial mem → spike = 1."""
        lif = self._lif()
        x = torch.tensor([[3.0]])  # mem_t = alpha*0 + 3.0 = 3.0 > THRESHOLD=2.0
        spk, _ = lif(x, self._zero_state())
        assert spk.item() == 1.0

    def test_no_fire_at_threshold_minus_epsilon(self) -> None:
        """x_t just below threshold → no spike."""
        lif = self._lif()
        x = torch.tensor([[THRESHOLD - 1e-4]])
        spk, _ = lif(x, self._zero_state())
        assert spk.item() == 0.0

    def test_no_fire_with_zero_input(self) -> None:
        """Zero input, zero initial membrane → no spike ever."""
        lif = self._lif()
        state = self._zero_state()
        x = torch.zeros(1, 1)
        for _ in range(10):
            spk, state = lif(x, state)
            assert spk.item() == 0.0

    # ── subtract reset: hand-computed ──

    def test_reset_subtract_hand_computed(self) -> None:
        """Subtract reset: mem_out = mem_t − spk * threshold.

        Setup: zero initial mem, x_t = 3.0 (causes spike).
          (i)   mem_t   = alpha * 0 + 3.0 = 3.0
          (ii)  spk_t   = H(3.0 − 2.0) = 1.0
          (iii) mem_out = 3.0 − 1.0 * 2.0 = 1.0
        """
        lif = self._lif(reset="subtract")
        x = torch.tensor([[3.0]])
        expected_mem = ALPHA * 0.0 + 3.0 - 1.0 * THRESHOLD  # = 1.0
        spk, new_state = lif(x, self._zero_state())
        assert spk.item() == 1.0
        assert abs(new_state.mem.item() - expected_mem) < 1e-5, (
            f"mem_out={new_state.mem.item():.8f} != expected={expected_mem:.8f}"
        )

    def test_reset_subtract_second_step(self) -> None:
        """Two-step subtract-reset, fully hand-computed.

        Step 1: mem_prev=0, x=3.0 → mem_t=3.0, spk=1, mem_out=1.0
        Step 2: mem_prev=1.0, x=3.0
          (i)   mem_t   = alpha * 1.0 + 3.0 = 0.904837 + 3.0 = 3.904837
          (ii)  spk_t   = H(3.904837 − 2.0) = 1.0
          (iii) mem_out = 3.904837 − 1.0 * 2.0 = 1.904837
        """
        lif = self._lif(reset="subtract")
        x = torch.tensor([[3.0]])
        state = self._zero_state()

        spk1, state = lif(x, state)
        assert spk1.item() == 1.0
        assert abs(state.mem.item() - 1.0) < 1e-5

        expected_mem2 = ALPHA * 1.0 + 3.0 - THRESHOLD  # ≈ 1.904837
        spk2, state = lif(x, state)
        assert spk2.item() == 1.0
        assert abs(state.mem.item() - expected_mem2) < 1e-5, (
            f"step2 mem_out={state.mem.item():.8f} != expected={expected_mem2:.8f}"
        )

    # ── zero reset: hand-computed ──

    def test_reset_zero_hand_computed(self) -> None:
        """Zero reset: mem_out = mem_t * (1 − spk.detach()).

        Setup: zero initial mem, x_t = 3.0 → spike.
          (i)   mem_t   = 3.0
          (ii)  spk_t   = 1.0
          (iii) mem_out = 3.0 * (1 − 1.0) = 0.0
        """
        lif = self._lif(reset="zero")
        x = torch.tensor([[3.0]])
        spk, new_state = lif(x, self._zero_state())
        assert spk.item() == 1.0
        assert abs(new_state.mem.item() - 0.0) < 1e-5

    def test_reset_zero_no_spike_unchanged(self) -> None:
        """Zero reset: when no spike fires, mem_out = mem_t (unchanged by reset)."""
        lif = self._lif(reset="zero")
        x = torch.tensor([[0.5]])  # small input, well below threshold
        # After one step: mem_t = alpha*0 + 0.5 = 0.5, spk=0, mem_out = 0.5*(1-0) = 0.5
        spk, new_state = lif(x, self._zero_state())
        assert spk.item() == 0.0
        expected_mem = ALPHA * 0.0 + 0.5
        assert abs(new_state.mem.item() - expected_mem) < 1e-5

    # ── membrane decay ──

    def test_membrane_decays_without_input(self) -> None:
        """With zero input and no spikes, membrane decays by factor alpha each step."""
        lif = self._lif()
        # Initial membrane below threshold so no spikes occur.
        initial_mem = THRESHOLD * 0.5  # = 1.0 < 2.0
        state = LIFState(mem=torch.tensor([[initial_mem]]))
        x = torch.zeros(1, 1)

        T = 10
        for t in range(1, T + 1):
            spk, state = lif(x, state)
            assert spk.item() == 0.0, f"Unexpected spike at t={t}"

        expected_mem = initial_mem * ALPHA**T
        assert abs(state.mem.item() - expected_mem) < 1e-5, (
            f"After {T} decay steps: got {state.mem.item():.8f}, expected {expected_mem:.8f}"
        )

    # ── alpha value ──

    def test_alpha_matches_exact_exponential(self) -> None:
        """alpha = exp(−dt/tau_mem), not the Euler approximation 1 − dt/tau_mem."""
        lif = self._lif()
        assert abs(lif.alpha - math.exp(-1.0 / TAU_MEM)) < 1e-12
        euler_alpha = 1.0 - 1.0 / TAU_MEM
        assert abs(lif.alpha - euler_alpha) > 1e-5, (
            "alpha matches Euler approximation — check ADR-0005 implementation"
        )

    # ── gradient flows ──

    def test_gradient_flows_through_lif(self) -> None:
        """After backward, Dense weight gradients are nonzero (surrogate works)."""
        model = snn.Sequential(
            snn.Dense(4, 4),
            snn.LIF(threshold=1.0, tau_mem=5.0),
        )
        torch.manual_seed(0)
        gen = torch.Generator()
        gen.manual_seed(0)

        x = torch.rand(4, 4)
        spikes_in = rate(x, T=5, generator=gen)  # [5, 4, 4]
        spk_out = model(spikes_in)
        loss = spk_out.sum()
        loss.backward()

        linear_layer = None
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                linear_layer = m
                break
        assert linear_layer is not None
        assert linear_layer.weight.grad is not None
        assert linear_layer.weight.grad.abs().max() > 0, "All gradients are zero"

    # ── constructor guards ──

    def test_tau_mem_leq_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="tau_mem"):
            LIF(tau_mem=0.5)  # 0.5 <= dt=1.0

    def test_tau_mem_eq_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="tau_mem"):
            LIF(tau_mem=1.0)  # exactly dt — rejected

    def test_invalid_reset_raises(self) -> None:
        with pytest.raises(ValueError, match="reset"):
            LIF(reset="clamp")

    def test_nonzero_delay_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="RFC-002"):
            LIF(delay=1)


# ── Dense ──────────────────────────────────────────────────────────────────────


class TestDense:
    def test_forward_shape(self) -> None:
        layer = snn.Dense(8, 4)
        x = torch.randn(16, 8)
        out = layer(x)
        assert out.shape == (16, 4)

    def test_weight_gradient_after_backward(self) -> None:
        layer = snn.Dense(4, 2)
        x = torch.randn(8, 4)
        out = layer(x)
        out.sum().backward()
        assert layer.linear.weight.grad is not None
        assert layer.linear.weight.grad.abs().max() > 0


# ── Conv2d ─────────────────────────────────────────────────────────────────────


class TestConv2d:
    def test_forward_shape(self) -> None:
        layer = snn.Conv2d(1, 4, kernel_size=3, padding=1)
        x = torch.randn(8, 1, 28, 28)
        out = layer(x)
        assert out.shape == (8, 4, 28, 28)

    def test_weight_gradient_after_backward(self) -> None:
        layer = snn.Conv2d(1, 4, kernel_size=3)
        x = torch.randn(2, 1, 10, 10)
        out = layer(x)
        out.sum().backward()
        assert layer.conv.weight.grad is not None


# ── Sequential ─────────────────────────────────────────────────────────────────


class TestSequential:
    def test_output_shape_fc(self) -> None:
        """Dense → LIF: output shape [T, batch, out_features]."""
        model = snn.Sequential(
            snn.Dense(4, 8),
            snn.LIF(threshold=1.0, tau_mem=5.0),
        )
        torch.manual_seed(0)
        gen = torch.Generator()
        gen.manual_seed(0)
        x = torch.rand(4, 4)
        spikes_in = rate(x, T=5, generator=gen)  # [5, 4, 4]
        out = model(spikes_in)
        assert out.shape == (5, 4, 8)

    def test_output_shape_two_lif(self) -> None:
        """Dense → LIF → Dense → LIF: final output shape [T, batch, 2]."""
        model = snn.Sequential(
            snn.Dense(4, 8),
            snn.LIF(threshold=1.0, tau_mem=5.0),
            snn.Dense(8, 2),
            snn.LIF(threshold=1.0, tau_mem=5.0),
        )
        torch.manual_seed(0)
        gen = torch.Generator()
        gen.manual_seed(0)
        x = torch.rand(6, 4)
        spikes_in = rate(x, T=7, generator=gen)  # [7, 6, 4]
        out = model(spikes_in)
        assert out.shape == (7, 6, 2)

    def test_backward_propagates_to_dense_weights(self) -> None:
        """Gradients must reach the Dense layer weights through the surrogate."""
        model = snn.Sequential(
            snn.Dense(4, 4),
            snn.LIF(threshold=1.0, tau_mem=5.0),
            snn.Dense(4, 2),
            snn.LIF(threshold=1.0, tau_mem=5.0),
        )
        torch.manual_seed(2)
        gen = torch.Generator()
        gen.manual_seed(2)
        x = torch.rand(4, 4)
        spikes_in = rate(x, T=5, generator=gen)
        out = model(spikes_in)
        y = torch.zeros(4).long()
        loss = rate_loss(out, y)
        loss.backward()

        grads = [
            m.linear.weight.grad
            for m in model.modules()
            if isinstance(m, snn.Dense) and m.linear.weight.grad is not None
        ]
        assert len(grads) == 2, "Both Dense layers should have gradients"
        for g in grads:
            assert g.abs().max() > 0, "Dense gradient is all-zero"

    def test_state_resets_between_forward_calls(self) -> None:
        """Two identical forward calls must produce identical outputs (state does not leak)."""
        model = snn.Sequential(
            snn.Dense(2, 4),
            snn.LIF(threshold=1.0, tau_mem=5.0),
        )
        gen1 = torch.Generator()
        gen1.manual_seed(0)
        gen2 = torch.Generator()
        gen2.manual_seed(0)

        x = torch.rand(2, 2)
        sp1 = rate(x, T=3, generator=gen1)
        sp2 = rate(x, T=3, generator=gen2)

        with torch.no_grad():
            out1 = model(sp1)
            out2 = model(sp2)

        assert torch.allclose(out1, out2), "State leaked between forward calls"


# ── Rate encoder ───────────────────────────────────────────────────────────────


class TestRateEncoder:
    def test_output_is_binary(self) -> None:
        gen = torch.Generator()
        gen.manual_seed(0)
        x = torch.rand(4, 784)
        out = rate(x, T=25, generator=gen)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_output_shape(self) -> None:
        gen = torch.Generator()
        gen.manual_seed(0)
        x = torch.rand(8, 16)
        out = rate(x, T=10, generator=gen)
        assert out.shape == (10, 8, 16)

    def test_deterministic_with_same_generator_seed(self) -> None:
        x = torch.rand(4, 4)
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        out1 = rate(x, T=10, generator=gen1)
        out2 = rate(x, T=10, generator=gen2)
        assert torch.all(out1 == out2)

    def test_different_seeds_differ(self) -> None:
        x = torch.full((1, 100), 0.5)  # 50% firing rate
        gen1 = torch.Generator()
        gen1.manual_seed(0)
        gen2 = torch.Generator()
        gen2.manual_seed(1)
        out1 = rate(x, T=50, generator=gen1)
        out2 = rate(x, T=50, generator=gen2)
        # With overwhelming probability (p ≈ 1 − 2^-5000) outputs differ.
        assert not torch.all(out1 == out2)

    def test_does_not_modify_global_torch_state(self) -> None:
        """rate() must not touch global torch RNG (Playbook §23 no hidden state)."""
        torch.manual_seed(99)
        x = torch.rand(2, 2)  # consumes global RNG
        before = torch.rand(1).item()  # record global state

        torch.manual_seed(99)
        x = torch.rand(2, 2)  # reproduce same consumption
        gen = torch.Generator()
        gen.manual_seed(0)
        rate(x, T=5, generator=gen)  # should NOT advance global state
        after = torch.rand(1).item()  # should reproduce `before`

        assert abs(before - after) < 1e-9, (
            "rate() modified global torch RNG state"
        )

    def test_clamps_input_to_0_1(self) -> None:
        gen = torch.Generator()
        gen.manual_seed(0)
        x = torch.tensor([[2.0, -1.0]])  # out-of-range; should be clamped
        out = rate(x, T=20, generator=gen)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})


# ── Latency encoder ────────────────────────────────────────────────────────────


class TestLatencyEncoder:
    def test_higher_value_spikes_earlier(self) -> None:
        """Input 0.9 must spike before input 0.1."""
        T = 20
        x = torch.tensor([[0.9, 0.1]])  # [batch=1, features=2]
        out = latency(x, T=T)  # [T, 1, 2]
        spike_times = out.argmax(dim=0)  # [1, 2]: timestep of first (only) spike
        assert spike_times[0, 0] < spike_times[0, 1], (
            f"0.9 spiked at t={spike_times[0,0].item()}, 0.1 spiked at t={spike_times[0,1].item()}"
        )

    def test_deterministic(self) -> None:
        """Latency encoder has no RNG — identical calls must produce identical results."""
        x = torch.rand(4, 10)
        out1 = latency(x, T=15)
        out2 = latency(x, T=15)
        assert torch.all(out1 == out2)

    def test_output_shape(self) -> None:
        x = torch.rand(3, 8)
        out = latency(x, T=12)
        assert out.shape == (12, 3, 8)

    def test_output_is_binary(self) -> None:
        x = torch.rand(2, 5)
        out = latency(x, T=10)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})


# ── Delta encoder ──────────────────────────────────────────────────────────────


class TestDeltaEncoder:
    def test_fires_on_large_change(self) -> None:
        """Spike where |x[t] − x[t−1]| > threshold."""
        x = torch.zeros(5, 1, 2)
        x[2, 0, 0] = 1.0  # change of 1.0 at t=2, feature 0
        out = delta(x, T=5, threshold=0.5)
        assert out[2, 0, 0] == 1.0  # spike at t=2, feature 0
        assert out[1, 0, 0] == 0.0  # no spike at t=1

    def test_no_spike_on_small_change(self) -> None:
        x = torch.zeros(4, 1, 2)
        x[1, 0, 0] = 0.05  # change below default threshold 0.1
        out = delta(x, T=4)
        assert out[1, 0, 0] == 0.0

    def test_deterministic(self) -> None:
        x = torch.rand(5, 2, 4)
        out1 = delta(x, T=5)
        out2 = delta(x, T=5)
        assert torch.all(out1 == out2)

    def test_raises_on_shape_mismatch(self) -> None:
        x = torch.rand(5, 2, 4)
        with pytest.raises(ValueError, match="T=10"):
            delta(x, T=10)

    def test_output_shape(self) -> None:
        x = torch.rand(6, 3, 8)
        out = delta(x, T=6)
        assert out.shape == (6, 3, 8)


# ── rate_loss ──────────────────────────────────────────────────────────────────


class TestRateLoss:
    def test_returns_scalar(self) -> None:
        spikes = torch.rand(10, 8, 4)
        labels = torch.randint(0, 4, (8,))
        loss = rate_loss(spikes, labels)
        assert loss.shape == torch.Size([])

    def test_gradient_exists(self) -> None:
        spikes = torch.rand(5, 4, 3, requires_grad=True)
        labels = torch.randint(0, 3, (4,))
        loss = rate_loss(spikes, labels)
        loss.backward()
        assert spikes.grad is not None
        assert spikes.grad.abs().max() > 0

    def test_loss_finite_on_random_input(self) -> None:
        spikes = torch.randn(10, 16, 10)  # random logits
        labels = torch.randint(0, 10, (16,))
        loss = rate_loss(spikes, labels)
        assert torch.isfinite(loss)

    def test_training_step_reduces_loss(self) -> None:
        """5 Adam steps on a fixed seeded mini-batch → loss is lower at the end."""
        from thrindex.encoders import rate  # noqa: PLC0415

        torch.manual_seed(0)
        model = snn.Sequential(
            snn.Dense(4, 8),
            snn.LIF(threshold=1.0, tau_mem=5.0),
            snn.Dense(8, 3),
            snn.LIF(threshold=1.0, tau_mem=5.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        labels = torch.tensor([0, 1, 2, 0])
        x = torch.rand(4, 4)

        losses: list[float] = []
        for step in range(5):
            optimizer.zero_grad()
            gen = torch.Generator()
            gen.manual_seed(step)
            spikes_in = rate(x, T=10, generator=gen)
            out = model(spikes_in)
            loss = rate_loss(out, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        assert losses[-1] < losses[0], (
            f"Loss did not decrease over 5 steps: {[f'{v:.4f}' for v in losses]}"
        )


# ── accuracy ───────────────────────────────────────────────────────────────────


class TestAccuracy:
    """Unit tests for thrindex.train.accuracy."""

    def test_perfect_accuracy(self) -> None:
        from thrindex.train import accuracy

        # 3 classes, batch=3. Each sample's correct class neuron spikes; others silent.
        # spikes shape: [T=1, batch=3, n_classes=3]
        spikes = torch.zeros(1, 3, 3)
        spikes[0, 0, 0] = 10.0  # sample 0 → class 0
        spikes[0, 1, 1] = 10.0  # sample 1 → class 1
        spikes[0, 2, 2] = 10.0  # sample 2 → class 2
        labels = torch.tensor([0, 1, 2])
        assert accuracy(spikes, labels) == 1.0

    def test_zero_accuracy(self) -> None:
        from thrindex.train import accuracy

        # All predictions wrong.
        spikes = torch.zeros(1, 3, 3)
        spikes[0, 0, 1] = 10.0  # sample 0 predicts class 1, true = 0
        spikes[0, 1, 2] = 10.0  # sample 1 predicts class 2, true = 1
        spikes[0, 2, 0] = 10.0  # sample 2 predicts class 0, true = 2
        labels = torch.tensor([0, 1, 2])
        assert accuracy(spikes, labels) == 0.0

    def test_partial_accuracy(self) -> None:
        from thrindex.train import accuracy

        # 2 of 4 correct.
        spikes = torch.zeros(1, 4, 2)
        spikes[0, 0, 0] = 10.0  # correct (label 0)
        spikes[0, 1, 0] = 10.0  # correct (label 0)
        spikes[0, 2, 1] = 10.0  # wrong   (label 0, predicts 1)
        spikes[0, 3, 1] = 10.0  # wrong   (label 0, predicts 1)
        labels = torch.tensor([0, 0, 0, 0])
        assert accuracy(spikes, labels) == 0.5

    def test_returns_float(self) -> None:
        from thrindex.train import accuracy

        spikes = torch.zeros(5, 2, 3)
        labels = torch.tensor([0, 1])
        result = accuracy(spikes, labels)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_multi_timestep_uses_rate(self) -> None:
        from thrindex.train import accuracy

        # T=10: class 0 fires 8 times, class 1 fires 2 times → predicts class 0.
        spikes = torch.zeros(10, 1, 2)
        spikes[:8, 0, 0] = 1.0   # class 0: 8 spikes
        spikes[8:, 0, 1] = 1.0   # class 1: 2 spikes
        labels = torch.tensor([0])
        assert accuracy(spikes, labels) == 1.0
