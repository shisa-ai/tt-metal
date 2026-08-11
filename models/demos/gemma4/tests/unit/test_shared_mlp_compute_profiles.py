# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.gemma4.tt import shared_mlp


class _FakeDevice:
    def arch(self):
        return "wormhole_b0"

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(8, 9)


class _FakeActivation:
    def __init__(self, device, shape=(1, 1, 1, 10240)):
        self._device = device
        self.shape = shape

    def device(self):
        return self._device


def test_down_proj_compute_profile_unset_preserves_linear_defaults(monkeypatch):
    monkeypatch.delenv(shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV, raising=False)
    monkeypatch.delenv(shared_mlp.DOWN_PROJ_PROGRAM_PROFILE_ENV, raising=False)
    calls = []
    monkeypatch.setattr(
        ttnn,
        "linear",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "output",
    )

    activation = _FakeActivation(_FakeDevice())

    assert shared_mlp._apply_down_projection(activation, "weight", layer_idx=38) == "output"
    assert calls == [((activation, "weight"), {})]


def test_down_proj_decode_program_profile_applies_only_to_exact_layer38_shape(monkeypatch):
    monkeypatch.delenv(shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV, raising=False)
    monkeypatch.setenv(
        shared_mlp.DOWN_PROJ_PROGRAM_PROFILE_ENV,
        shared_mlp.DOWN_PROJ_LAYER38_DECODE_MCAST1D_W8_PROFILE,
    )
    program_calls = []
    linear_calls = []
    monkeypatch.setattr(
        ttnn,
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        lambda **kwargs: program_calls.append(kwargs) or "program_config",
    )
    monkeypatch.setattr(
        ttnn,
        "linear",
        lambda *args, **kwargs: linear_calls.append((args, kwargs)) or "output",
    )
    activation = _FakeActivation(_FakeDevice())
    prefill_activation = _FakeActivation(_FakeDevice(), shape=(1, 1, 32, 10240))

    assert shared_mlp._apply_down_projection(activation, "weight38", layer_idx=38) == "output"
    assert shared_mlp._apply_down_projection(activation, "weight37", layer_idx=37) == "output"
    assert shared_mlp._apply_down_projection(prefill_activation, "weight38", layer_idx=38) == "output"
    assert program_calls == [
        {
            "compute_with_storage_grid_size": ttnn.CoreCoord(8, 9),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "out_block_h": 1,
            "out_block_w": 2,
            "per_core_M": 1,
            "per_core_N": 2,
            "fuse_batch": True,
            "fused_activation": None,
            "mcast_in0": True,
        }
    ]
    assert linear_calls == [
        ((activation, "weight38"), {"program_config": "program_config"}),
        ((activation, "weight37"), {}),
        ((prefill_activation, "weight38"), {}),
    ]


def test_down_proj_decode_program_profiles_select_block_width(monkeypatch):
    program_calls = []
    monkeypatch.setattr(
        ttnn,
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        lambda **kwargs: program_calls.append(kwargs) or "program_config",
    )
    monkeypatch.setattr(ttnn, "linear", lambda *args, **kwargs: "output")
    activation = _FakeActivation(_FakeDevice())

    cases = (
        (shared_mlp.DOWN_PROJ_LAYER38_DECODE_MCAST1D_W8_PROFILE, 8),
        (shared_mlp.DOWN_PROJ_LAYER38_DECODE_MCAST1D_W10_PROFILE, 10),
        (shared_mlp.DOWN_PROJ_LAYER38_DECODE_MCAST1D_W32_PROFILE, 32),
    )
    for profile, expected_width in cases:
        monkeypatch.setenv(shared_mlp.DOWN_PROJ_PROGRAM_PROFILE_ENV, profile)
        assert shared_mlp._apply_down_projection(activation, "weight", layer_idx=38) == "output"
        assert program_calls[-1]["in0_block_w"] == expected_width


def test_down_proj_program_profile_rejects_unknown_value(monkeypatch, expect_error):
    monkeypatch.setenv(shared_mlp.DOWN_PROJ_PROGRAM_PROFILE_ENV, "layer38_decode_w7")

    with expect_error(
        ValueError,
        "GEMMA4_SHARED_MLP_DOWN_PROJ_PROGRAM_PROFILE must be one of: "
        "layer38_decode_mcast1d_w10, layer38_decode_mcast1d_w32, layer38_decode_mcast1d_w8",
    ):
        shared_mlp._apply_down_projection(_FakeActivation(_FakeDevice()), "weight", layer_idx=38)


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


def test_down_proj_remaining_safe_profile_specs(monkeypatch):
    cases = (
        (shared_mlp.DOWN_PROJ_LAYER38_HIFI2_FP32_ACC_PROFILE, ttnn.MathFidelity.HiFi2, True, False),
        (shared_mlp.DOWN_PROJ_LAYER38_HIFI4_BF16_L1_ACC_PROFILE, ttnn.MathFidelity.HiFi4, False, True),
        (shared_mlp.DOWN_PROJ_LAYER38_HIFI3_BF16_L1_ACC_PROFILE, ttnn.MathFidelity.HiFi3, False, True),
        (shared_mlp.DOWN_PROJ_LAYER38_LOFI_FP32_ACC_PROFILE, ttnn.MathFidelity.LoFi, True, False),
        (shared_mlp.DOWN_PROJ_LAYER38_LOFI_BF16_L1_ACC_PROFILE, ttnn.MathFidelity.LoFi, False, True),
    )
    init_calls = []
    monkeypatch.setattr(
        ttnn,
        "init_device_compute_kernel_config",
        lambda *args, **kwargs: init_calls.append((args, kwargs)) or "compute_config",
    )
    monkeypatch.setattr(ttnn, "linear", lambda *args, **kwargs: "output")
    activation = _FakeActivation(_FakeDevice())

    for profile, fidelity, fp32, packer_l1 in cases:
        monkeypatch.setenv(shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV, profile)
        assert shared_mlp._apply_down_projection(activation, "weight", layer_idx=38) == "output"
        assert init_calls[-1] == (
            ("wormhole_b0",),
            {
                "math_fidelity": fidelity,
                "math_approx_mode": False,
                "fp32_dest_acc_en": fp32,
                "packer_l1_acc": packer_l1,
            },
        )


def test_down_proj_compute_profile_rejects_unknown_value(monkeypatch, expect_error):
    monkeypatch.setenv(shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV, "all_layers_hifi4")

    with expect_error(
        ValueError,
        "GEMMA4_SHARED_MLP_DOWN_PROJ_COMPUTE_PROFILE must be one of: "
        "layer38_hifi2_fp32_acc, layer38_hifi3_bf16_l1_acc, layer38_hifi3_fp32_acc, "
        "layer38_hifi3_fp32_l1_acc, layer38_hifi4_bf16_l1_acc, layer38_lofi_bf16_l1_acc, "
        "layer38_lofi_fp32_acc",
    ):
        shared_mlp._apply_down_projection(_FakeActivation(_FakeDevice()), "weight", layer_idx=38)


def test_down_proj_compute_profile_requires_device_activation(monkeypatch, expect_error):
    monkeypatch.setenv(
        shared_mlp.DOWN_PROJ_COMPUTE_PROFILE_ENV,
        shared_mlp.DOWN_PROJ_LAYER38_HIFI3_FP32_ACC_PROFILE,
    )

    with expect_error(ValueError, "requires a device-resident activation"):
        shared_mlp._apply_down_projection(_FakeActivation(None), "weight", layer_idx=38)
