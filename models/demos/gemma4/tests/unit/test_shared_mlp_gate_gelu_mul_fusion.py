# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.demos.gemma4.tt import shared_mlp


class FakeTensor:
    def __init__(self, name):
        self.name = name
        self.deallocated = False

    def deallocate(self, force):
        assert force is True
        self.deallocated = True


@pytest.mark.parametrize("value", [True, "1", "true", "YES", "on"])
def test_fused_selector_accepts_true_values(value):
    assert shared_mlp._resolve_fused_gate_gelu_mul(value) is True


@pytest.mark.parametrize("value", [False, "0", "false", "NO", "off"])
def test_fused_selector_accepts_false_values(value):
    assert shared_mlp._resolve_fused_gate_gelu_mul(value) is False


def test_fused_selector_rejects_unknown_value(expect_error):
    with expect_error(ValueError, shared_mlp.FUSED_GATE_GELU_MUL_ENV):
        shared_mlp._resolve_fused_gate_gelu_mul("sometimes")


@pytest.mark.parametrize("fused", [False, True])
def test_forward_fuses_only_accurate_gelu_into_mul(monkeypatch, fused):
    calls = []
    hidden_states = FakeTensor("input")
    gate_before_gelu = FakeTensor("gate_before_gelu")
    gate_after_gelu = FakeTensor("gate_after_gelu")
    up = FakeTensor("up")
    hidden = FakeTensor("hidden")
    output = FakeTensor("output")

    accurate_activation = object()
    activation_args = []

    def unary_with_param(op, value):
        activation_args.append((op, value))
        return accurate_activation

    monkeypatch.setattr(shared_mlp.ttnn, "UnaryWithParam", unary_with_param)

    def linear(tensor, weight):
        calls.append(("linear", tensor.name, weight.name))
        if weight.name == "gate_weight":
            return gate_before_gelu
        if weight.name == "up_weight":
            return up
        return output

    def gelu(tensor, fast_and_approximate_mode):
        calls.append(("gelu", tensor.name, fast_and_approximate_mode))
        return gate_after_gelu

    def mul(lhs, rhs, input_tensor_a_activations=None):
        calls.append(("mul", lhs.name, rhs.name, input_tensor_a_activations))
        return hidden

    monkeypatch.setattr(shared_mlp.ttnn, "linear", linear)
    monkeypatch.setattr(shared_mlp.ttnn, "gelu", gelu)
    monkeypatch.setattr(shared_mlp.ttnn, "mul", mul)

    mlp = object.__new__(shared_mlp.SharedMLP)
    mlp.fuse_gate_gelu_mul = fused
    mlp.gate_proj = FakeTensor("gate_weight")
    mlp.up_proj = FakeTensor("up_weight")
    mlp.down_proj = FakeTensor("down_weight")
    mlp.mesh_config = SimpleNamespace(tp=1)

    assert mlp(hidden_states) is output
    if fused:
        assert activation_args == [(shared_mlp.ttnn.UnaryOpType.GELU, 0.0)]
        assert all(call[0] != "gelu" for call in calls)
        assert calls[2] == ("mul", "gate_before_gelu", "up", [accurate_activation])
    else:
        assert activation_args == []
        assert calls[2] == ("gelu", "gate_before_gelu", False)
        assert calls[3] == ("mul", "gate_after_gelu", "up", None)
    if fused:
        assert gate_before_gelu.deallocated
    else:
        assert gate_after_gelu.deallocated
    assert up.deallocated and hidden.deallocated


def test_fusion_rejects_tensor_parallel(monkeypatch, expect_error):
    monkeypatch.setattr(shared_mlp.ttnn, "as_tensor", lambda *args, **kwargs: FakeTensor("weight"))
    config = SimpleNamespace(hidden_size=2816, intermediate_size=2112)
    mesh_config = SimpleNamespace(tp=2)

    with expect_error(ValueError, "qualified only for TP1"):
        shared_mlp.SharedMLP(
            mesh_device=object(),
            hf_config=config,
            state_dict=None,
            mesh_config=mesh_config,
            fuse_gate_gelu_mul=True,
        )
