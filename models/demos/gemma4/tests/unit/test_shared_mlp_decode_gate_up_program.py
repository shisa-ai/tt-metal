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


@pytest.mark.parametrize("value", [True, "1", "true", "YES", "on"])
def test_bool_selector_accepts_true_values(value):
    assert shared_mlp._resolve_bool_env("TEST_SELECTOR", value) is True


@pytest.mark.parametrize("value", [False, "0", "false", "NO", "off"])
def test_bool_selector_accepts_false_values(value):
    assert shared_mlp._resolve_bool_env("TEST_SELECTOR", value) is False


def test_bool_selector_rejects_unknown_value(expect_error):
    with expect_error(ValueError, "TEST_SELECTOR"):
        shared_mlp._resolve_bool_env("TEST_SELECTOR", "sometimes")


def test_decode_widths_enumerate_all_safe_chotto_k_blocks():
    assert shared_mlp._decode_gate_up_in0_block_widths(2560) == (1, 2, 4, 5, 8, 10, 16, 20)


@pytest.mark.parametrize("value", [1, "2", " 20 "])
def test_decode_width_selector_accepts_safe_explicit_widths(value):
    assert shared_mlp._resolve_decode_gate_up_in0_block_w(2560, value) == int(value)


@pytest.mark.parametrize("value", [0, "0"])
def test_decode_width_selector_zero_disables_program(value):
    assert shared_mlp._resolve_decode_gate_up_in0_block_w(2560, value) is None


@pytest.mark.parametrize("value", [True, False, "auto", -1, 3, 32])
def test_decode_width_selector_rejects_implicit_or_unsafe_values(expect_error, value):
    with expect_error(ValueError, shared_mlp.DECODE_GATE_UP_IN0_BLOCK_W_ENV):
        shared_mlp._resolve_decode_gate_up_in0_block_w(2560, value)


def test_decode_width_selector_reads_environment(monkeypatch):
    monkeypatch.setenv(shared_mlp.DECODE_GATE_UP_IN0_BLOCK_W_ENV, "10")
    assert shared_mlp._resolve_decode_gate_up_in0_block_w(2560) == 10


def test_prefill_widths_exclude_chotto_k_blocks_that_exceed_l1():
    assert shared_mlp._prefill_gate_up_in0_block_widths(2560) == (1, 2, 4)


def test_prefill_widths_reject_unqualified_hidden_size(expect_error):
    with expect_error(ValueError, "hidden_size=2560"):
        shared_mlp._prefill_gate_up_in0_block_widths(2816)


@pytest.mark.parametrize("value", [1, "2", " 4 "])
def test_prefill_width_selector_accepts_safe_explicit_widths(value):
    assert shared_mlp._resolve_prefill_gate_up_in0_block_w(2560, value) == int(value)


@pytest.mark.parametrize("value", [0, "0"])
def test_prefill_width_selector_zero_disables_program(value):
    assert shared_mlp._resolve_prefill_gate_up_in0_block_w(2560, value) is None


@pytest.mark.parametrize("value", [True, False, "auto", -1, 3, 5, 8])
def test_prefill_width_selector_rejects_implicit_or_l1_unsafe_values(expect_error, value):
    with expect_error(ValueError, shared_mlp.PREFILL_GATE_UP_IN0_BLOCK_W_ENV):
        shared_mlp._resolve_prefill_gate_up_in0_block_w(2560, value)


def test_prefill_width_selector_reads_environment(monkeypatch):
    monkeypatch.setenv(shared_mlp.PREFILL_GATE_UP_IN0_BLOCK_W_ENV, "4")
    assert shared_mlp._resolve_prefill_gate_up_in0_block_w(2560) == 4


def test_decode_config_uses_explicit_safe_blocking(monkeypatch):
    captured = {}
    monkeypatch.setattr(shared_mlp.ttnn, "CoreCoord", lambda x, y: (x, y))
    monkeypatch.setattr(
        shared_mlp.ttnn,
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=9))

    config = shared_mlp._decode_gate_up_program_config(
        mesh,
        hidden_size=2560,
        intermediate_size=10240,
        in0_block_w=10,
    )

    assert config == captured
    assert captured == {
        "compute_with_storage_grid_size": (8, 9),
        "in0_block_w": 10,
        "out_subblock_h": 1,
        "out_subblock_w": 5,
        "per_core_M": 1,
        "per_core_N": 5,
        "fuse_batch": False,
        "fused_activation": None,
        "mcast_in0": True,
    }


def test_prefill_config_matches_accepted_automatic_geometry(monkeypatch):
    captured = {}
    monkeypatch.setattr(shared_mlp.ttnn, "CoreCoord", lambda x, y: (x, y))
    monkeypatch.setattr(
        shared_mlp.ttnn,
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=9))

    config = shared_mlp._prefill_gate_up_program_config(
        mesh,
        hidden_size=2560,
        intermediate_size=10240,
        in0_block_w=4,
    )

    assert config == captured
    assert captured == {
        "compute_with_storage_grid_size": (8, 9),
        "in0_block_w": 4,
        "out_subblock_h": 8,
        "out_subblock_w": 1,
        "per_core_M": 32,
        "per_core_N": 5,
        "fused_activation": None,
        "fuse_batch": False,
        "mcast_in0": True,
    }


def test_prefill_config_rejects_non_wormhole_grid(expect_error):
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=8))
    with expect_error(ValueError, "8x9 grid"):
        shared_mlp._prefill_gate_up_program_config(
            mesh,
            hidden_size=2560,
            intermediate_size=10240,
            in0_block_w=4,
        )


def test_prefill_config_rejects_unqualified_shape(expect_error):
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=9))
    with expect_error(ValueError, "intermediate_size=10240"):
        shared_mlp._prefill_gate_up_program_config(
            mesh,
            hidden_size=2560,
            intermediate_size=2112,
            in0_block_w=4,
        )


@pytest.mark.parametrize(
    ("sequence_length", "expected_config"),
    [(32, "decode"), (1024, "prefill"), (128, None)],
)
def test_program_config_applies_only_to_target_phase(monkeypatch, sequence_length, expected_config):
    calls = []
    shape = (1, 1, sequence_length, 2560)
    hidden_states = FakeTensor("input", shape=shape)
    gate = FakeTensor("gate")
    up = FakeTensor("up")
    hidden = FakeTensor("hidden")
    output = FakeTensor("output")
    decode_config = "decode"
    prefill_config = "prefill"

    def linear(tensor, weight, program_config=None):
        calls.append((weight.name, program_config))
        return {"gate_weight": gate, "up_weight": up, "down_weight": output}[weight.name]

    monkeypatch.setattr(shared_mlp.ttnn, "linear", linear)
    monkeypatch.setattr(shared_mlp.ttnn, "gelu", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(shared_mlp.ttnn, "mul", lambda lhs, rhs, **kwargs: hidden)

    mlp = object.__new__(shared_mlp.SharedMLP)
    mlp.fuse_gate_gelu_mul = False
    mlp.decode_gate_up_program_config = decode_config
    mlp.prefill_gate_up_program_config = prefill_config
    mlp.decode_down_program_config = None
    mlp.prefill_down_program_config = None
    mlp.gate_proj = FakeTensor("gate_weight")
    mlp.up_proj = FakeTensor("up_weight")
    mlp.down_proj = FakeTensor("down_weight")
    mlp.mesh_config = SimpleNamespace(tp=1)

    assert mlp(hidden_states) is output
    assert calls == [("gate_weight", expected_config), ("up_weight", expected_config), ("down_weight", None)]


def test_fused_accurate_gelu_mul_removes_standalone_gelu(monkeypatch):
    calls = []
    gate = FakeTensor("gate")
    up = FakeTensor("up")
    hidden = FakeTensor("hidden")
    output = FakeTensor("output")
    activation = object()

    monkeypatch.setattr(shared_mlp.ttnn, "UnaryWithParam", lambda op, value: activation)
    monkeypatch.setattr(
        shared_mlp.ttnn,
        "linear",
        lambda tensor, weight, program_config=None: {
            "gate_weight": gate,
            "up_weight": up,
            "down_weight": output,
        }[weight.name],
    )
    monkeypatch.setattr(shared_mlp.ttnn, "gelu", lambda *args, **kwargs: pytest.fail("standalone GELU called"))
    monkeypatch.setattr(
        shared_mlp.ttnn,
        "mul",
        lambda lhs, rhs, input_tensor_a_activations=None: calls.append(input_tensor_a_activations) or hidden,
    )

    mlp = object.__new__(shared_mlp.SharedMLP)
    mlp.fuse_gate_gelu_mul = True
    mlp.decode_gate_up_program_config = None
    mlp.prefill_gate_up_program_config = None
    mlp.decode_down_program_config = None
    mlp.prefill_down_program_config = None
    mlp.gate_proj = FakeTensor("gate_weight")
    mlp.up_proj = FakeTensor("up_weight")
    mlp.down_proj = FakeTensor("down_weight")
    mlp.mesh_config = SimpleNamespace(tp=1)

    assert mlp(FakeTensor("input")) is output
    assert calls == [[activation]]
