# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.demos.gemma4.tt import shared_mlp


class FakeTensor:
    def __init__(self, name, shape=(1, 1, 32, 2560)):
        self.name = name
        self.shape = shape

    def deallocate(self, force):
        assert force is True


def test_decode_down_widths_enumerate_complete_bounded_surface():
    assert shared_mlp._down_in0_block_widths(10240) == (1, 2, 4, 5, 8, 10, 16, 20, 32)


@pytest.mark.parametrize("value", [1, "2", " 32 "])
def test_decode_down_selector_accepts_explicit_widths(value):
    assert shared_mlp._resolve_decode_down_in0_block_w(10240, value) == int(value)


@pytest.mark.parametrize("value", [0, "0"])
def test_decode_down_selector_zero_disables_program(value):
    assert shared_mlp._resolve_decode_down_in0_block_w(10240, value) is None


@pytest.mark.parametrize("value", [True, False, "auto", -1, 3, 40])
def test_decode_down_selector_rejects_implicit_or_invalid_values(expect_error, value):
    with expect_error(ValueError, shared_mlp.DECODE_DOWN_IN0_BLOCK_W_ENV):
        shared_mlp._resolve_decode_down_in0_block_w(10240, value)


def test_decode_down_selector_reads_environment(monkeypatch):
    monkeypatch.setenv(shared_mlp.DECODE_DOWN_IN0_BLOCK_W_ENV, "16")
    assert shared_mlp._resolve_decode_down_in0_block_w(10240) == 16


def test_prefill_down_widths_exclude_first_l1_unsafe_divisor():
    assert shared_mlp._prefill_down_in0_block_widths(10240) == (1, 2, 4, 5, 8, 10, 16, 20)


def test_prefill_down_widths_reject_unqualified_intermediate_size(expect_error):
    with expect_error(ValueError, "intermediate_size=10240"):
        shared_mlp._prefill_down_in0_block_widths(2112)


@pytest.mark.parametrize("value", [1, "2", " 20 "])
def test_prefill_down_selector_accepts_l1_safe_widths(value):
    assert shared_mlp._resolve_prefill_down_in0_block_w(10240, value) == int(value)


@pytest.mark.parametrize("value", [0, "0"])
def test_prefill_down_selector_zero_disables_program(value):
    assert shared_mlp._resolve_prefill_down_in0_block_w(10240, value) is None


@pytest.mark.parametrize("value", [True, False, "auto", -1, 3, 32])
def test_prefill_down_selector_rejects_implicit_or_l1_unsafe_values(expect_error, value):
    with expect_error(ValueError, shared_mlp.PREFILL_DOWN_IN0_BLOCK_W_ENV):
        shared_mlp._resolve_prefill_down_in0_block_w(10240, value)


def test_prefill_down_selector_reads_environment(monkeypatch):
    monkeypatch.setenv(shared_mlp.PREFILL_DOWN_IN0_BLOCK_W_ENV, "10")
    assert shared_mlp._resolve_prefill_down_in0_block_w(10240) == 10


def test_decode_down_config_matches_accepted_automatic_geometry(monkeypatch):
    captured = {}
    monkeypatch.setattr(shared_mlp.ttnn, "CoreCoord", lambda x, y: (x, y))
    monkeypatch.setattr(
        shared_mlp.ttnn,
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=9))

    config = shared_mlp._decode_down_program_config(
        mesh,
        intermediate_size=10240,
        hidden_size=2560,
        in0_block_w=16,
    )

    assert config == captured
    assert captured == {
        "compute_with_storage_grid_size": (8, 9),
        "in0_block_w": 16,
        "out_subblock_h": 1,
        "out_subblock_w": 2,
        "per_core_M": 1,
        "per_core_N": 2,
        "fuse_batch": False,
        "fused_activation": None,
        "mcast_in0": True,
    }


def test_prefill_down_config_matches_accepted_automatic_geometry(monkeypatch):
    captured = {}
    monkeypatch.setattr(shared_mlp.ttnn, "CoreCoord", lambda x, y: (x, y))
    monkeypatch.setattr(
        shared_mlp.ttnn,
        "MatmulMultiCoreReuseMultiCastProgramConfig",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=9))

    config = shared_mlp._prefill_down_program_config(
        mesh,
        intermediate_size=10240,
        hidden_size=2560,
        in0_block_w=10,
    )

    assert config == captured
    assert captured == {
        "compute_with_storage_grid_size": (8, 9),
        "in0_block_w": 10,
        "out_subblock_h": 4,
        "out_subblock_w": 2,
        "per_core_M": 4,
        "per_core_N": 10,
        "transpose_mcast": False,
        "fused_activation": None,
        "fuse_batch": False,
    }


def test_prefill_down_config_rejects_non_wormhole_grid(expect_error):
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=8))
    with expect_error(ValueError, "8x9 grid"):
        shared_mlp._prefill_down_program_config(
            mesh,
            intermediate_size=10240,
            hidden_size=2560,
            in0_block_w=10,
        )


@pytest.mark.parametrize(
    ("sequence_length", "expected_config"),
    [(32, "decode-down"), (1024, "prefill-down"), (128, None)],
)
def test_down_program_applies_only_to_target_phase(monkeypatch, sequence_length, expected_config):
    calls = []
    hidden_states = FakeTensor("input", shape=(1, 1, sequence_length, 2560))
    gate = FakeTensor("gate")
    up = FakeTensor("up")
    hidden = FakeTensor("hidden")
    output = FakeTensor("output")

    def linear(tensor, weight, program_config=None):
        calls.append((weight.name, program_config))
        return {"gate_weight": gate, "up_weight": up, "down_weight": output}[weight.name]

    monkeypatch.setattr(shared_mlp.ttnn, "linear", linear)
    monkeypatch.setattr(shared_mlp.ttnn, "gelu", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(shared_mlp.ttnn, "mul", lambda lhs, rhs, **kwargs: hidden)

    mlp = object.__new__(shared_mlp.SharedMLP)
    mlp.fuse_gate_gelu_mul = False
    mlp.decode_gate_up_program_config = None
    mlp.prefill_gate_up_program_config = None
    mlp.decode_down_program_config = "decode-down"
    mlp.prefill_down_program_config = "prefill-down"
    mlp.gate_proj = FakeTensor("gate_weight")
    mlp.up_proj = FakeTensor("up_weight")
    mlp.down_proj = FakeTensor("down_weight")
    mlp.mesh_config = SimpleNamespace(tp=1)

    assert mlp(hidden_states) is output
    assert calls == [("gate_weight", None), ("up_weight", None), ("down_weight", expected_config)]
