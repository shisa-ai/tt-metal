# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

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


def test_unset_safe_profile_preserves_linear_defaults(monkeypatch):
    monkeypatch.delenv(shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV, raising=False)
    calls = []
    monkeypatch.setattr(ttnn, "linear", lambda *args, **kwargs: calls.append((args, kwargs)) or "output")
    activation = _FakeActivation(_FakeDevice())

    assert shared_mlp._apply_down_projection(activation, "weight", layer_idx=38) == "output"
    assert calls == [((activation, "weight"), {})]


def test_retained_safe_profile_specs_apply_only_to_layer38(monkeypatch):
    cases = (
        (shared_mlp.DOWN_PROJ_LAYER38_HIFI2_FP32_ACC_PROFILE, ttnn.MathFidelity.HiFi2, True, False),
        (shared_mlp.DOWN_PROJ_LAYER38_HIFI4_BF16_L1_ACC_PROFILE, ttnn.MathFidelity.HiFi4, False, True),
        (shared_mlp.DOWN_PROJ_LAYER38_HIFI3_BF16_L1_ACC_PROFILE, ttnn.MathFidelity.HiFi3, False, True),
        (shared_mlp.DOWN_PROJ_LAYER38_LOFI_FP32_ACC_PROFILE, ttnn.MathFidelity.LoFi, True, False),
        (shared_mlp.DOWN_PROJ_LAYER38_LOFI_BF16_L1_ACC_PROFILE, ttnn.MathFidelity.LoFi, False, True),
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

    for profile, fidelity, fp32, packer_l1 in cases:
        monkeypatch.setenv(shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV, profile)
        assert shared_mlp._apply_down_projection(activation, "weight38", layer_idx=38) == "output"
        assert shared_mlp._apply_down_projection(activation, "weight37", layer_idx=37) == "output"
        assert init_calls[-1] == (
            ("wormhole_b0",),
            {
                "math_fidelity": fidelity,
                "math_approx_mode": False,
                "fp32_dest_acc_en": fp32,
                "packer_l1_acc": packer_l1,
            },
        )
        assert linear_calls[-2:] == [
            ((activation, "weight38"), {"compute_kernel_config": "compute_config"}),
            ((activation, "weight37"), {}),
        ]


@pytest.mark.parametrize("removed_profile", ["layer38_hifi3_fp32_acc", "layer38_hifi3_fp32_l1_acc"])
def test_removed_profiles_are_not_reintroduced(monkeypatch, expect_error, removed_profile):
    monkeypatch.setenv(shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV, removed_profile)

    with expect_error(ValueError, "must be one of"):
        shared_mlp._apply_down_projection(_FakeActivation(_FakeDevice()), "weight", layer_idx=38)
