# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import ttnn
from models.demos.gemma4.tt.attention import operations


class _FakeDevice:
    def arch(self):
        return "wormhole_b0"

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=8, y=9)


class _FakeActivation:
    def __init__(self, device, shape=(1, 1, 32, 2560)):
        self._device = device
        self.shape = shape

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


def test_prefill_sliding_qkv_program_matches_accepted_geometry(monkeypatch):
    monkeypatch.delenv(operations.QKV_COMPUTE_PROFILE_ENV, raising=False)
    monkeypatch.setenv(operations.PREFILL_SLIDING_QKV_IN0_BLOCK_W_ENV, "10")
    linear_calls = []
    monkeypatch.setattr(ttnn, "CoreCoord", lambda x, y: (x, y))
    monkeypatch.setattr(
        ttnn,
        "MatmulMultiCoreReuseMultiCastProgramConfig",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        ttnn,
        "linear",
        lambda *args, **kwargs: linear_calls.append((args, kwargs)) or "output",
    )

    activation = _FakeActivation(_FakeDevice(), shape=(1, 1, 1024, 2560))
    weights = SimpleNamespace(
        wqkv=SimpleNamespace(shape=(1, 1, 2560, 3072)),
        is_global=False,
    )

    assert operations.apply_qkv_projection(activation, weights) == "output"
    assert linear_calls == [
        (
            (activation, weights.wqkv),
            {
                "memory_config": None,
                "program_config": {
                    "compute_with_storage_grid_size": (8, 9),
                    "in0_block_w": 10,
                    "out_subblock_h": 4,
                    "out_subblock_w": 2,
                    "per_core_M": 4,
                    "per_core_N": 12,
                    "transpose_mcast": False,
                    "fused_activation": None,
                    "fuse_batch": False,
                },
            },
        )
    ]


def test_prefill_sliding_output_program_matches_accepted_geometry(monkeypatch):
    monkeypatch.setenv(operations.PREFILL_SLIDING_OUTPUT_IN0_BLOCK_W_ENV, "8")
    linear_calls = []
    monkeypatch.setattr(ttnn, "CoreCoord", lambda x, y: (x, y))
    monkeypatch.setattr(
        ttnn,
        "MatmulMultiCoreReuseMultiCastProgramConfig",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        ttnn,
        "linear",
        lambda *args, **kwargs: linear_calls.append((args, kwargs)) or "output",
    )

    activation = _FakeActivation(_FakeDevice(), shape=(1, 1, 1024, 2048))
    weights = SimpleNamespace(
        o_proj=SimpleNamespace(shape=(1, 1, 2048, 2560)),
        is_global=False,
    )

    assert operations.apply_output_projection(activation, weights) == "output"
    assert linear_calls == [
        (
            (activation, weights.o_proj),
            {
                "program_config": {
                    "compute_with_storage_grid_size": (8, 9),
                    "in0_block_w": 8,
                    "out_subblock_h": 4,
                    "out_subblock_w": 2,
                    "per_core_M": 4,
                    "per_core_N": 10,
                    "transpose_mcast": False,
                    "fused_activation": None,
                    "fuse_batch": False,
                }
            },
        )
    ]
