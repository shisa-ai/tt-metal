# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt import shared_mlp


class FakeTensor:
    def __init__(self, name):
        self.name = name
        self.deallocated = False

    def deallocate(self, force):
        assert force is True
        self.deallocated = True


def test_prepare_up_gate_weight_preserves_geglu_order():
    gate = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    up = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    packed = shared_mlp._prepare_up_gate_weight({"gate_proj.weight": gate, "up_proj.weight": up})
    assert torch.equal(packed, torch.cat((up, gate), dim=0).T[None, None])


@pytest.mark.parametrize("value", [True, "1", "true", "YES", "on"])
def test_fused_selector_accepts_true_values(value):
    assert shared_mlp._resolve_fused_gate_up(value) is True


@pytest.mark.parametrize("value", [False, "0", "false", "NO", "off"])
def test_fused_selector_accepts_false_values(value):
    assert shared_mlp._resolve_fused_gate_up(value) is False


def test_fused_selector_rejects_unknown_value(expect_error):
    with expect_error(ValueError, shared_mlp.FUSED_GATE_UP_ENV):
        shared_mlp._resolve_fused_gate_up("sometimes")


def test_fused_forward_uses_one_projection_and_accurate_gelu(monkeypatch):
    calls = []
    hidden_states = FakeTensor("input")
    packed = FakeTensor("packed")
    up = FakeTensor("up")
    gate_before_gelu = FakeTensor("gate_before_gelu")
    gate = FakeTensor("gate")
    hidden = FakeTensor("hidden")
    output = FakeTensor("output")

    def linear(tensor, weight):
        calls.append(("linear", tensor.name, weight.name))
        return packed if weight.name == "up_gate_weight" else output

    def split(tensor, split_size, dim):
        calls.append(("split", tensor.name, split_size, dim))
        return up, gate_before_gelu

    def gelu(tensor, fast_and_approximate_mode):
        calls.append(("gelu", tensor.name, fast_and_approximate_mode))
        return gate

    def mul(lhs, rhs):
        calls.append(("mul", lhs.name, rhs.name))
        return hidden

    monkeypatch.setattr(shared_mlp.ttnn, "linear", linear)
    monkeypatch.setattr(shared_mlp.ttnn, "split", split)
    monkeypatch.setattr(shared_mlp.ttnn, "gelu", gelu)
    monkeypatch.setattr(shared_mlp.ttnn, "mul", mul)

    mlp = object.__new__(shared_mlp.SharedMLP)
    mlp.fuse_gate_up = True
    mlp.intermediate_size = 10240
    mlp.up_gate_proj = FakeTensor("up_gate_weight")
    mlp.down_proj = FakeTensor("down_weight")
    mlp.mesh_config = SimpleNamespace(tp=1)

    assert mlp(hidden_states) is output
    assert calls == [
        ("linear", "input", "up_gate_weight"),
        ("split", "packed", 10240, -1),
        ("gelu", "gate_before_gelu", False),
        ("mul", "gate", "up"),
        ("linear", "hidden", "down_weight"),
    ]
    assert packed.deallocated and gate.deallocated and up.deallocated and hidden.deallocated
