# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.gemma4.tt import shared_mlp


class _FakeDevice:
    def arch(self):
        return "wormhole_b0"


class _FakeActivation:
    def __init__(self, device):
        self._device = device

    def device(self):
        return self._device


def test_down_proj_compute_profile_unset_preserves_linear_defaults(monkeypatch):
    monkeypatch.delenv(shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV, raising=False)
    calls = []
    monkeypatch.setattr(
        ttnn,
        "linear",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "output",
    )

    activation = _FakeActivation(_FakeDevice())

    assert shared_mlp._apply_down_projection(activation, "weight", layer_idx=38) == "output"
    assert calls == [((activation, "weight"), {})]


def test_down_proj_hifi3_fp32_reload_profile_applies_only_to_layer38(monkeypatch):
    monkeypatch.setenv(
        shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV,
        shared_mlp.DOWN_PROJ_LAYER38_HIFI3_FP32_ACC_PROFILE,
    )
    init_calls = []
    linear_calls = []
    monkeypatch.setattr(
        ttnn,
        "init_device_compute_kernel_config",
        lambda *args, **kwargs: init_calls.append((args, kwargs)) or "compute_config",
    )
    monkeypatch.setattr(
        ttnn,
        "linear",
        lambda *args, **kwargs: linear_calls.append((args, kwargs)) or "output",
    )
    activation = _FakeActivation(_FakeDevice())

    assert shared_mlp._apply_down_projection(activation, "weight38", layer_idx=38) == "output"
    assert shared_mlp._apply_down_projection(activation, "weight37", layer_idx=37) == "output"
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
        ((activation, "weight38"), {"compute_kernel_config": "compute_config"}),
        ((activation, "weight37"), {}),
    ]


def test_down_proj_hifi3_fp32_l1_profile_enables_packer_l1(monkeypatch):
    monkeypatch.setenv(
        shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV,
        shared_mlp.DOWN_PROJ_LAYER38_HIFI3_FP32_L1_ACC_PROFILE,
    )
    init_calls = []
    monkeypatch.setattr(
        ttnn,
        "init_device_compute_kernel_config",
        lambda *args, **kwargs: init_calls.append((args, kwargs)) or "compute_config",
    )
    monkeypatch.setattr(ttnn, "linear", lambda *args, **kwargs: "output")

    assert shared_mlp._apply_down_projection(_FakeActivation(_FakeDevice()), "weight", layer_idx=38) == "output"
    assert init_calls == [
        (
            ("wormhole_b0",),
            {
                "math_fidelity": ttnn.MathFidelity.HiFi3,
                "math_approx_mode": False,
                "fp32_dest_acc_en": True,
                "packer_l1_acc": True,
            },
        )
    ]


def test_down_proj_compute_profile_rejects_unknown_value(monkeypatch, expect_error):
    monkeypatch.setenv(shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV, "all_layers_hifi4")

    with expect_error(
        ValueError,
        "GEMMA4_SHARED_MLP_DOWN_PROJ_COMPUTE_PROFILE must be one of: "
        "layer38_hifi3_fp32_acc, layer38_hifi3_fp32_l1_acc",
    ):
        shared_mlp._apply_down_projection(_FakeActivation(_FakeDevice()), "weight", layer_idx=38)


def test_down_proj_compute_profile_requires_device_activation(monkeypatch, expect_error):
    monkeypatch.setenv(
        shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV,
        shared_mlp.DOWN_PROJ_LAYER38_HIFI3_FP32_ACC_PROFILE,
    )

    with expect_error(ValueError, "requires a device-resident activation"):
        shared_mlp._apply_down_projection(_FakeActivation(None), "weight", layer_idx=38)
