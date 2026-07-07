"""Tests for the FastSigmoid surrogate gradient.

Correction 1 (mandatory): ``torch.autograd.gradcheck`` is invalid for surrogate
gradient functions.  The forward is a Heaviside step whose numerical gradient is
zero almost everywhere; gradcheck will always report a mismatch with the fast-sigmoid
backward.  This is expected by design — that is the definition of a surrogate.

Instead we test:
(a) Analytical backward: for a grid of membrane values, the backward output
    matches the closed-form β/(2(1+β|x|)²)·upstream_grad within float tolerance.
(b) End-to-end: a small network trained with optimizer steps strictly reduces loss
    on a seeded toy dataset, demonstrating that the surrogate provides a useful
    gradient signal.
"""

from __future__ import annotations

import math

import pytest
import torch
from thrindex.snn.surrogate import FastSigmoid

BETA = 25.0


# ── Helpers ────────────────────────────────────────────────────────────────────


def _expected_grad(x_val: float, beta: float, upstream: float = 1.0) -> float:
    """Closed-form fast-sigmoid surrogate gradient at a scalar x = U − threshold."""
    return beta / (2.0 * (1.0 + beta * abs(x_val)) ** 2) * upstream


def _call_backward(x_val: float, beta: float = BETA, upstream: float = 1.0) -> float:
    """Run forward + backward; return the computed gradient w.r.t. x."""
    x = torch.tensor([x_val], requires_grad=True)
    y = FastSigmoid.apply(x, beta)
    y.backward(torch.tensor([upstream]))
    assert x.grad is not None
    return float(x.grad.item())


# ── (a) Analytical backward on a grid ──────────────────────────────────────────


class TestAnalyticalBackward:
    """Assert backward matches β/(2(1+β|x|)²)·upstream for a grid of inputs."""

    TOLERANCE = 1e-6

    @pytest.mark.parametrize(
        "x_val",
        [
            0.0,          # at threshold
            1.0 / BETA,   # +1/β from threshold
            2.0 / BETA,   # +2/β
            -1.0 / BETA,  # −1/β (symmetric)
            -2.0 / BETA,  # −2/β
            10.0 / BETA,  # far positive
            -10.0 / BETA, # far negative
            100.0 / BETA, # very far: gradient should be tiny
        ],
    )
    def test_grad_matches_closed_form(self, x_val: float) -> None:
        computed = _call_backward(x_val)
        expected = _expected_grad(x_val, BETA)
        assert abs(computed - expected) < self.TOLERANCE, (
            f"At x={x_val:.6f}: computed={computed:.8f}, expected={expected:.8f}"
        )

    def test_grad_at_threshold_equals_beta_over_2(self) -> None:
        """At x=0 (U=threshold), gradient must equal β/2."""
        computed = _call_backward(0.0)
        assert abs(computed - BETA / 2.0) < self.TOLERANCE

    def test_grad_is_symmetric_around_threshold(self) -> None:
        """Gradient is symmetric: x and −x give the same surrogate value."""
        for k in [1, 2, 5, 10]:
            x_pos = k / BETA
            x_neg = -k / BETA
            grad_pos = _call_backward(x_pos)
            grad_neg = _call_backward(x_neg)
            assert abs(grad_pos - grad_neg) < self.TOLERANCE, (
                f"Asymmetry at ±{k}/β: pos={grad_pos:.8f}, neg={grad_neg:.8f}"
            )

    def test_grad_decays_with_distance(self) -> None:
        """Gradient must decrease as |x| increases from 0."""
        grads = [_call_backward(k / BETA) for k in [0, 1, 5, 20, 100]]
        for i in range(len(grads) - 1):
            assert grads[i] > grads[i + 1], (
                f"Gradient did not decrease: grads[{i}]={grads[i]}, grads[{i+1}]={grads[i+1]}"
            )

    @pytest.mark.parametrize("upstream", [0.5, 2.0, -1.0])
    def test_grad_scales_linearly_with_upstream(self, upstream: float) -> None:
        """Backward output must be linear in upstream gradient."""
        grad_1 = _call_backward(1.0 / BETA, upstream=1.0)
        grad_u = _call_backward(1.0 / BETA, upstream=upstream)
        assert abs(grad_u - grad_1 * upstream) < self.TOLERANCE

    def test_slope_parameter_scales_peak(self) -> None:
        """Peak gradient at x=0 must equal beta/2 for each beta."""
        for beta in [1.0, 10.0, 25.0, 100.0]:
            computed = _call_backward(0.0, beta=beta)
            assert abs(computed - beta / 2.0) < self.TOLERANCE, (
                f"beta={beta}: peak grad {computed:.6f} != beta/2 = {beta/2:.6f}"
            )


# ── (b) End-to-end: loss decreases with optimizer steps ────────────────────────


class TestEndToEndTraining:
    """A small SNN must train on seeded toy data; loss strictly decreases."""

    def _build_tiny_net(self) -> torch.nn.Module:
        import thrindex.snn as snn

        return snn.Sequential(
            snn.Dense(4, 8),
            snn.LIF(threshold=1.0, tau_mem=5.0),
            snn.Dense(8, 2),
            snn.LIF(threshold=1.0, tau_mem=5.0),
        )

    def test_loss_decreases_over_10_steps(self) -> None:
        """10 Adam optimizer steps on seeded 2-class toy data → strictly decreasing loss."""
        from thrindex.train import rate_loss

        torch.manual_seed(0)
        gen = torch.Generator()
        gen.manual_seed(0)

        # Seeded toy dataset: 16 samples, 4 features, 2 classes
        # Class 0: first two features hot; class 1: last two features hot
        x0 = torch.cat([torch.rand(8, 2) + 1.0, torch.zeros(8, 2)], dim=1)
        x1 = torch.cat([torch.zeros(8, 2), torch.rand(8, 2) + 1.0], dim=1)
        x_data = torch.cat([x0, x1], dim=0).clamp(0.0, 1.0)
        y_data = torch.cat([torch.zeros(8), torch.ones(8)], dim=0).long()

        model = self._build_tiny_net()
        torch.manual_seed(0)
        # Re-init weights for reproducibility
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        T = 10
        losses: list[float] = []
        for _ in range(10):
            optimizer.zero_grad()
            from thrindex.encoders import rate  # noqa: PLC0415
            step_gen = torch.Generator()
            step_gen.manual_seed(len(losses))  # deterministic per step
            spikes_in = rate(x_data, T=T, generator=step_gen)  # [T, 16, 4]
            spk_out = model(spikes_in)  # [T, 16, 2]
            loss = rate_loss(spk_out, y_data)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        assert losses[-1] < losses[0], (
            f"Loss did not decrease over 10 steps: {[f'{v:.4f}' for v in losses]}"
        )

    def test_gradients_are_nonzero_after_backward(self) -> None:
        """After a forward+backward pass, Dense layer weights must have nonzero gradients."""
        from thrindex.encoders import rate
        from thrindex.train import rate_loss

        torch.manual_seed(1)
        gen = torch.Generator()
        gen.manual_seed(1)

        model = self._build_tiny_net()
        x = torch.rand(4, 4)
        y = torch.tensor([0, 1, 0, 1])

        spikes_in = rate(x, T=5, generator=gen)
        spk_out = model(spikes_in)
        loss = rate_loss(spk_out, y)
        loss.backward()

        # At least one Dense layer must have a nonzero gradient.
        has_nonzero_grad = False
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.weight.grad is not None:
                if m.weight.grad.abs().max() > 0:
                    has_nonzero_grad = True
                    break
        assert has_nonzero_grad, "All gradients are zero — surrogate gradient not working"
