"""Conv, flatten, and delay survive compile and match the simulator."""

from __future__ import annotations

from pathlib import Path

import pytest
import thrindex.snn as snn
import torch
from thrindex.compile import compile_model


def _require_core() -> None:
    pytest.importorskip("thrindex._core")


def test_conv_flatten_sim_matches_pytorch(tmp_path: Path) -> None:
    _require_core()
    from thrindex._core import run_sim

    torch.manual_seed(0)
    conv = snn.Conv2d(
        1,
        2,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True,
        input_height=4,
        input_width=4,
    )
    model = snn.Sequential(conv, snn.Flatten(), snn.Dense(2 * 4 * 4, 3, bias=False))
    path = tmp_path / "conv.thx"
    compile_model(model, path)

    frame = torch.rand(1, 1, 4, 4)
    with torch.no_grad():
        expected = model(frame.unsqueeze(0)).reshape(-1)

    flat = frame.reshape(1, 1, -1).tolist()
    spikes, _stats, _transcript = run_sim(str(path), flat, 1, 0)
    got = torch.tensor(spikes[0][0])
    assert torch.allclose(got, expected, atol=1e-5)


def test_strided_dilated_conv_matches_pytorch(tmp_path: Path) -> None:
    _require_core()
    from thrindex._core import run_sim

    conv = snn.Conv2d(
        1,
        1,
        kernel_size=3,
        stride=2,
        dilation=2,
        padding=0,
        bias=True,
        input_height=7,
        input_width=7,
    )
    model = snn.Sequential(conv)
    path = tmp_path / "stride.thx"
    compile_model(model, path)
    frame = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
    with torch.no_grad():
        expected = model(frame.unsqueeze(0)).reshape(-1)
    spikes, _stats, _transcript = run_sim(str(path), frame.reshape(1, 1, -1).tolist(), 1, 0)
    assert torch.allclose(torch.tensor(spikes[0][0]), expected, atol=1e-5)


def test_conv_then_lif_matches_pytorch_across_time(tmp_path: Path) -> None:
    _require_core()
    from thrindex._core import run_sim

    torch.manual_seed(1)
    conv = snn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, input_height=5, input_width=5)
    with torch.no_grad():
        conv.conv.weight.fill_(0.25)
    lif = snn.LIF(threshold=0.5, tau_mem=5.0, reset="subtract")
    model = snn.Sequential(conv, lif)
    path = tmp_path / "convlif.thx"
    compile_model(model, path)
    frames = torch.rand(4, 1, 1, 5, 5)
    with torch.no_grad():
        expected = model(frames).reshape(4, -1)
    flat = frames.reshape(1, 4, -1).tolist()
    spikes, _stats, _transcript = run_sim(str(path), flat, 1, 0)
    assert torch.allclose(torch.tensor(spikes[0]), expected, atol=1e-5)


def test_grouped_conv_matches_pytorch(tmp_path: Path) -> None:
    _require_core()
    from thrindex._core import run_sim

    conv = snn.Conv2d(
        2,
        2,
        kernel_size=1,
        groups=2,
        bias=False,
        input_height=1,
        input_width=1,
    )
    with torch.no_grad():
        conv.conv.weight.zero_()
        conv.conv.weight[0, 0, 0, 0] = 3.0
        conv.conv.weight[1, 0, 0, 0] = 4.0
    model = snn.Sequential(conv)
    path = tmp_path / "groups.thx"
    compile_model(model, path)
    frame = torch.tensor([[[[1.0]], [[10.0]]]])  # [batch, C, H, W]
    with torch.no_grad():
        expected = model(frame.unsqueeze(0)).reshape(-1)
    spikes, _stats, _transcript = run_sim(str(path), frame.reshape(1, 1, -1).tolist(), 1, 0)
    assert torch.allclose(torch.tensor(spikes[0][0]), expected, atol=1e-6)


def test_compiled_delay_matches_pytorch(tmp_path: Path) -> None:
    _require_core()
    from thrindex._core import run_sim

    layer = snn.Dense(1, 1, bias=False, delay=2)
    with torch.no_grad():
        layer.linear.weight.fill_(1.0)
    model = snn.Sequential(layer)
    path = tmp_path / "delay.thx"
    compile_model(model, path)

    x = torch.tensor([[[1.0]], [[0.0]], [[0.0]], [[4.0]]])
    with torch.no_grad():
        expected = model(x)[:, 0, 0]

    spikes, _stats, _transcript = run_sim(str(path), x[:, 0, :].unsqueeze(0).tolist(), 1, 0)
    got = torch.tensor(spikes[0])[:, 0]
    assert torch.allclose(got, expected, atol=1e-6)
