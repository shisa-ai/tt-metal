# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.gemma4.tt import model


class _FakeDevice:
    def arch(self):
        return "wormhole_b0"


def _capture_config(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ttnn,
        "init_device_compute_kernel_config",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "compute_config",
    )
    return calls


def test_lm_head_compute_profile_unset_preserves_stock_semantics(monkeypatch):
    monkeypatch.delenv(model.LM_HEAD_COMPUTE_PROFILE_ENV, raising=False)
    calls = _capture_config(monkeypatch)

    assert model._get_tp1_dram_sharded_lm_head_compute_kernel_config(_FakeDevice()) == "compute_config"
    assert calls == [
        (
            ("wormhole_b0",),
            {
                "math_fidelity": ttnn.MathFidelity.HiFi2,
                "math_approx_mode": False,
                "fp32_dest_acc_en": False,
                "packer_l1_acc": True,
            },
        )
    ]


def test_lm_head_hifi3_fp32_profile_freezes_candidate_semantics(monkeypatch):
    monkeypatch.setenv(model.LM_HEAD_COMPUTE_PROFILE_ENV, model.LM_HEAD_HIFI3_FP32_ACC_PROFILE)
    calls = _capture_config(monkeypatch)

    assert model._get_tp1_dram_sharded_lm_head_compute_kernel_config(_FakeDevice()) == "compute_config"
    assert calls == [
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


def test_lm_head_compute_profile_rejects_unknown_value(monkeypatch, expect_error):
    monkeypatch.setenv(model.LM_HEAD_COMPUTE_PROFILE_ENV, "hifi4_fp32_acc")

    with expect_error(ValueError, "GEMMA4_LM_HEAD_COMPUTE_PROFILE must be one of: hifi3_fp32_acc"):
        model._get_tp1_dram_sharded_lm_head_compute_kernel_config(_FakeDevice())
