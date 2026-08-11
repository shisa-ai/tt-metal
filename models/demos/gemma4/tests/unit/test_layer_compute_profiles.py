# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.gemma4.tt import layer


class _FakeDevice:
    def arch(self):
        return "wormhole_b0"


class _FakeActivation:
    def __init__(self, device):
        self._device = device

    def device(self):
        return self._device


def test_pli_projection_compute_profile_unset_preserves_defaults(monkeypatch):
    monkeypatch.delenv(layer.PLI_PROJECTION_COMPUTE_PROFILE_ENV, raising=False)

    assert layer._pli_projection_compute_kernel_config(_FakeActivation(_FakeDevice()), layer_idx=0) is None


def test_pli_projection_hifi3_fp32_profile_applies_only_to_layer0(monkeypatch):
    monkeypatch.setenv(
        layer.PLI_PROJECTION_COMPUTE_PROFILE_ENV,
        layer.PLI_PROJECTION_LAYER0_HIFI3_FP32_ACC_PROFILE,
    )
    init_calls = []
    monkeypatch.setattr(
        ttnn,
        "init_device_compute_kernel_config",
        lambda *args, **kwargs: init_calls.append((args, kwargs)) or "compute_config",
    )
    activation = _FakeActivation(_FakeDevice())

    assert layer._pli_projection_compute_kernel_config(activation, layer_idx=0) == "compute_config"
    assert layer._pli_projection_compute_kernel_config(activation, layer_idx=1) is None
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


def test_pli_projection_compute_profile_rejects_unknown_value(monkeypatch, expect_error):
    monkeypatch.setenv(layer.PLI_PROJECTION_COMPUTE_PROFILE_ENV, "all_layers_hifi4")

    with expect_error(
        ValueError,
        "GEMMA4_PLI_PROJECTION_COMPUTE_PROFILE must be one of: layer0_hifi3_fp32_acc",
    ):
        layer._pli_projection_compute_kernel_config(_FakeActivation(_FakeDevice()), layer_idx=0)


def test_pli_projection_compute_profile_requires_device_activation(monkeypatch, expect_error):
    monkeypatch.setenv(
        layer.PLI_PROJECTION_COMPUTE_PROFILE_ENV,
        layer.PLI_PROJECTION_LAYER0_HIFI3_FP32_ACC_PROFILE,
    )

    with expect_error(ValueError, "requires a device-resident activation"):
        layer._pli_projection_compute_kernel_config(_FakeActivation(None), layer_idx=0)
