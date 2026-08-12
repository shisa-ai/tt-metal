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
    assert shared_mlp._resolve_fused_gate_gelu(value) is True


@pytest.mark.parametrize("value", [False, "0", "false", "NO", "off"])
def test_fused_selector_accepts_false_values(value):
    assert shared_mlp._resolve_fused_gate_gelu(value) is False


def test_fused_selector_rejects_unknown_value(expect_error):
    with expect_error(ValueError, shared_mlp.FUSED_GATE_GELU_ENV):
        shared_mlp._resolve_fused_gate_gelu("sometimes")


@pytest.mark.parametrize("fused", [False, True])
def test_forward_selects_only_accurate_fused_gelu(monkeypatch, fused):
    calls = []
    hidden_states = FakeTensor("input")
    gate_before_gelu = FakeTensor("gate_before_gelu")
    gate = FakeTensor("gate")
    up = FakeTensor("up")
    hidden = FakeTensor("hidden")
    output = FakeTensor("output")

    accurate_activation = object()
    activation_args = []

    def unary_with_param(op, value):
        activation_args.append((op, value))
        return accurate_activation

    monkeypatch.setattr(shared_mlp.ttnn, "UnaryWithParam", unary_with_param)

    def linear(tensor, weight, activation=None):
        calls.append(("linear", tensor.name, weight.name, activation))
        if weight.name == "gate_weight":
            return gate if activation is accurate_activation else gate_before_gelu
        if weight.name == "up_weight":
            return up
        return output

    def gelu(tensor, fast_and_approximate_mode):
        calls.append(("gelu", tensor.name, fast_and_approximate_mode))
        return gate

    def mul(lhs, rhs):
        calls.append(("mul", lhs.name, rhs.name))
        return hidden

    monkeypatch.setattr(shared_mlp.ttnn, "linear", linear)
    monkeypatch.setattr(shared_mlp.ttnn, "gelu", gelu)
    monkeypatch.setattr(shared_mlp.ttnn, "mul", mul)

    mlp = object.__new__(shared_mlp.SharedMLP)
    mlp.fuse_gate_gelu = fused
    mlp.gate_proj = FakeTensor("gate_weight")
    mlp.up_proj = FakeTensor("up_weight")
    mlp.down_proj = FakeTensor("down_weight")
    mlp.mesh_config = SimpleNamespace(tp=1)

    assert mlp(hidden_states) is output
    if fused:
        assert activation_args == [(shared_mlp.ttnn.UnaryOpType.GELU, 0.0)]
        assert calls[0] == ("linear", "input", "gate_weight", accurate_activation)
        assert all(call[0] != "gelu" for call in calls)
    else:
        assert activation_args == []
        assert calls[:2] == [
            ("linear", "input", "gate_weight", None),
            ("gelu", "gate_before_gelu", False),
        ]
    assert gate.deallocated and up.deallocated and hidden.deallocated
