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


def test_final_norm_compute_profile_unset_preserves_ttnn_default(monkeypatch):
    monkeypatch.delenv(model.FINAL_NORM_COMPUTE_PROFILE_ENV, raising=False)
    calls = _capture_config(monkeypatch)

    assert model._get_final_norm_compute_kernel_config(_FakeDevice()) is None
    assert calls == []


def test_final_norm_hifi3_fp32_profile_freezes_candidate_semantics(monkeypatch):
    monkeypatch.setenv(
        model.FINAL_NORM_COMPUTE_PROFILE_ENV,
        model.FINAL_NORM_HIFI3_FP32_ACC_PROFILE,
    )
    calls = _capture_config(monkeypatch)

    assert model._get_final_norm_compute_kernel_config(_FakeDevice()) == "compute_config"
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


def test_final_norm_compute_profile_rejects_unknown_value(monkeypatch, expect_error):
    monkeypatch.setenv(model.FINAL_NORM_COMPUTE_PROFILE_ENV, "hifi4_fp32_acc")

    with expect_error(ValueError, "GEMMA4_FINAL_NORM_COMPUTE_PROFILE must be one of: hifi3_fp32_acc"):
        model._get_final_norm_compute_kernel_config(_FakeDevice())
