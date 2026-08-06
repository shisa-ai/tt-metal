# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import ttnn
from models.demos.gemma4.tt.attention import operations


class _FakeDevice:
    def arch(self):
        return "wormhole_b0"


class _FakeActivation:
    def __init__(self, device):
        self._device = device

    def device(self):
        return self._device


def test_qkv_compute_profile_unset_preserves_linear_defaults(monkeypatch):
    monkeypatch.delenv(operations.QKV_COMPUTE_PROFILE_ENV, raising=False)
    calls = []
    monkeypatch.setattr(ttnn, "linear", lambda *args, **kwargs: calls.append((args, kwargs)) or "output")

    activation = _FakeActivation(_FakeDevice())
    weights = SimpleNamespace(wqkv="qkv_weight")

    assert operations.apply_qkv_projection(activation, weights) == "output"
    assert calls == [((activation, "qkv_weight"), {"memory_config": None})]


def test_qkv_hifi3_fp32_profile_freezes_compute_semantics(monkeypatch):
    monkeypatch.setenv(operations.QKV_COMPUTE_PROFILE_ENV, operations.QKV_HIFI3_FP32_ACC_PROFILE)
    init_calls = []
    linear_calls = []
    monkeypatch.setattr(
        ttnn,
        "init_device_compute_kernel_config",
        lambda *args, **kwargs: init_calls.append((args, kwargs)) or "compute_config",
    )
    monkeypatch.setattr(ttnn, "linear", lambda *args, **kwargs: linear_calls.append((args, kwargs)) or "output")

    activation = _FakeActivation(_FakeDevice())
    weights = SimpleNamespace(wqkv="qkv_weight")

    assert operations.apply_qkv_projection(activation, weights, memory_config="l1") == "output"
    assert init_calls == [
        (
            ("wormhole_b0",),
            {
                "math_fidelity": ttnn.MathFidelity.HiFi3,
                "math_approx_mode": False,
                "fp32_dest_acc_en": True,
                "packer_l1_acc": False,
            },
        )
    ]
    assert linear_calls == [
        (
            (activation, "qkv_weight"),
            {"memory_config": "l1", "compute_kernel_config": "compute_config"},
        )
    ]


def test_qkv_compute_profile_rejects_unknown_value(monkeypatch, expect_error):
    monkeypatch.setenv(operations.QKV_COMPUTE_PROFILE_ENV, "hifi4_fp32_acc")

    with expect_error(ValueError, "GEMMA4_QKV_COMPUTE_PROFILE must be one of: hifi3_fp32_acc"):
        operations.apply_qkv_projection(_FakeActivation(_FakeDevice()), SimpleNamespace(wqkv="qkv_weight"))


def test_qkv_compute_profile_requires_device_activation(monkeypatch, expect_error):
    monkeypatch.setenv(operations.QKV_COMPUTE_PROFILE_ENV, operations.QKV_HIFI3_FP32_ACC_PROFILE)

    with expect_error(ValueError, "requires a device-resident activation"):
        operations.apply_qkv_projection(_FakeActivation(None), SimpleNamespace(wqkv="qkv_weight"))
