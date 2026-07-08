"""Tests for thx.compile — .thx artifact serialisation.

Verifies ADR-0006:
- format_version, target, model structure present
- Weights encoded as base64 little-endian f32 (round-trippable)
- alpha resolved at compile time (never recomputed by sim)
- CRC32 stored and correct
- model_canonical stored (identical bytes for cross-language CRC verification)
- version_gate: unknown format_version → clear error
"""

from __future__ import annotations

import base64
import json
import math
import struct
import zlib
from pathlib import Path

import pytest
import thrindex.snn as snn
import torch
from thrindex.compile import compile_model


def _tiny_model() -> snn.Sequential:
    return snn.Sequential(
        snn.Dense(8, 16),
        snn.LIF(threshold=1.0, tau_mem=10.0, reset="subtract"),
        snn.Dense(16, 4),
        snn.LIF(threshold=0.5, tau_mem=20.0, reset="zero"),
    )


class TestCompileArtifactStructure:
    def test_format_version(self, tmp_path: Path) -> None:
        m = _tiny_model()
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        assert a["format_version"] == "m2-draft"

    def test_target_sim(self, tmp_path: Path) -> None:
        m = _tiny_model()
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        assert a["target"] == "sim"

    def test_layer_count(self, tmp_path: Path) -> None:
        m = _tiny_model()
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        assert len(a["model"]["layers"]) == 4

    def test_layer_types(self, tmp_path: Path) -> None:
        m = _tiny_model()
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        types = [layer["type"] for layer in a["model"]["layers"]]
        assert types == ["dense", "lif", "dense", "lif"]

    def test_metadata_fields_present(self, tmp_path: Path) -> None:
        m = _tiny_model()
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        meta = a["metadata"]
        assert "compiled_at" in meta
        assert "crc32" in meta
        assert "model_canonical" in meta


class TestWeightEncoding:
    def test_weights_are_base64(self, tmp_path: Path) -> None:
        m = _tiny_model()
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        dense0 = a["model"]["layers"][0]
        assert isinstance(dense0["weights_b64"], str)
        # Must decode without error.
        decoded = base64.b64decode(dense0["weights_b64"])
        assert len(decoded) % 4 == 0

    def test_weight_round_trip(self, tmp_path: Path) -> None:
        """Decoded base64 weights must equal original f32 values."""
        m = _tiny_model()
        original_w = m.layers[0].linear.weight.detach().clone()  # type: ignore[attr-defined]
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        raw = base64.b64decode(a["model"]["layers"][0]["weights_b64"])
        n = len(raw) // 4
        decoded = struct.unpack_from(f"<{n}f", raw)
        original_list = original_w.to(torch.float32).contiguous().numpy().flatten().tolist()
        assert list(decoded) == pytest.approx(original_list, abs=1e-7)

    def test_weights_little_endian(self, tmp_path: Path) -> None:
        """A single known weight value must decode to the expected LE bytes."""
        m = snn.Sequential(snn.Dense(1, 1), snn.LIF())
        with torch.no_grad():
            m.layers[0].linear.weight.fill_(1.0)  # type: ignore[attr-defined]
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        raw = base64.b64decode(a["model"]["layers"][0]["weights_b64"])
        # 1.0 as little-endian f32 = 0x3F800000 = bytes [0x00, 0x00, 0x80, 0x3F]
        assert raw == bytes([0x00, 0x00, 0x80, 0x3F])


class TestResolvedConstants:
    """alpha is resolved at compile time — never recomputed (ADR-0007 correction 4)."""

    def test_alpha_is_stored(self, tmp_path: Path) -> None:
        m = _tiny_model()
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        lif_layer = a["model"]["layers"][1]
        assert "alpha" in lif_layer
        assert isinstance(lif_layer["alpha"], float)

    def test_alpha_value_correct_tau10(self, tmp_path: Path) -> None:
        m = snn.Sequential(snn.Dense(4, 4), snn.LIF(tau_mem=10.0))
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        alpha = a["model"]["layers"][1]["alpha"]
        expected = math.exp(-1.0 / 10.0)
        assert abs(alpha - expected) < 1e-12

    def test_alpha_value_correct_tau20(self, tmp_path: Path) -> None:
        m = snn.Sequential(snn.Dense(4, 4), snn.LIF(tau_mem=20.0))
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        alpha = a["model"]["layers"][1]["alpha"]
        expected = math.exp(-1.0 / 20.0)
        assert abs(alpha - expected) < 1e-12

    def test_alpha_syn_none_when_no_tau_syn(self, tmp_path: Path) -> None:
        m = snn.Sequential(snn.Dense(4, 4), snn.LIF(tau_syn=None))
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        assert a["model"]["layers"][1]["alpha_syn"] is None

    def test_alpha_syn_stored_when_tau_syn_set(self, tmp_path: Path) -> None:
        m = snn.Sequential(snn.Dense(4, 4), snn.LIF(tau_syn=5.0))
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        alpha_syn = a["model"]["layers"][1]["alpha_syn"]
        expected = math.exp(-1.0 / 5.0)
        assert alpha_syn is not None
        assert abs(alpha_syn - expected) < 1e-12


class TestIntegrity:
    def test_crc32_is_correct(self, tmp_path: Path) -> None:
        m = _tiny_model()
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        canonical = a["metadata"]["model_canonical"]
        expected_crc = f"{zlib.crc32(canonical.encode('utf-8')) & 0xFFFF_FFFF:08x}"
        assert a["metadata"]["crc32"] == expected_crc

    def test_model_canonical_matches_model(self, tmp_path: Path) -> None:
        """model_canonical must be the sorted-key compact JSON of the model block."""
        m = _tiny_model()
        p = tmp_path / "model.thx"
        compile_model(m, p)
        a = json.loads(p.read_text())
        canonical = a["metadata"]["model_canonical"]
        reparsed = json.loads(canonical)
        assert reparsed == a["model"]


class TestValidation:
    def test_non_sequential_raises(self, tmp_path: Path) -> None:
        import torch.nn as nn

        with pytest.raises(TypeError, match="thrindex.snn.Sequential"):
            compile_model(nn.Linear(4, 4), tmp_path / "bad.thx")  # type: ignore[arg-type]
