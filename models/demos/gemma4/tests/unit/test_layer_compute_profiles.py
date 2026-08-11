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


class _FakeTemporary:
    def __init__(self, name):
        self.name = name
        self.deallocate_calls = []

    def deallocate(self, force):
        self.deallocate_calls.append(force)


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


def test_mlp_residual_add_profile_unset_preserves_add_defaults(monkeypatch):
    monkeypatch.delenv(layer.MLP_RESIDUAL_ADD_PROFILE_ENV, raising=False)
    add_calls = []
    monkeypatch.setattr(
        ttnn,
        "add",
        lambda *args, **kwargs: add_calls.append((args, kwargs)) or "stock_output",
    )

    assert layer._apply_mlp_residual_add("residual", "hidden", layer_idx=38, is_decode=True) == "stock_output"
    assert add_calls == [(("residual", "hidden"), {})]


def test_mlp_residual_add_profile_applies_only_to_layer38_decode(monkeypatch):
    monkeypatch.setenv(
        layer.MLP_RESIDUAL_ADD_PROFILE_ENV,
        layer.MLP_RESIDUAL_ADD_LAYER38_DECODE_FP32_INPUTS_BF16_OUTPUT_PROFILE,
    )
    temporaries = []
    typecast_calls = []
    add_calls = []

    def fake_typecast(value, dtype):
        typecast_calls.append((value, dtype))
        temporary = _FakeTemporary(f"{value}_fp32")
        temporaries.append(temporary)
        return temporary

    monkeypatch.setattr(ttnn, "typecast", fake_typecast)
    monkeypatch.setattr(
        ttnn,
        "add",
        lambda *args, **kwargs: add_calls.append((args, kwargs)) or "output",
    )

    assert layer._apply_mlp_residual_add("residual38", "hidden38", layer_idx=38, is_decode=True) == "output"
    assert layer._apply_mlp_residual_add("residual37", "hidden37", layer_idx=37, is_decode=True) == "output"
    assert layer._apply_mlp_residual_add("prefill38", "prefill_hidden38", layer_idx=38, is_decode=False) == "output"
    assert typecast_calls == [
        ("residual38", ttnn.float32),
        ("hidden38", ttnn.float32),
    ]
    assert add_calls == [
        ((temporaries[0], temporaries[1]), {"dtype": ttnn.bfloat16}),
        (("residual37", "hidden37"), {}),
        (("prefill38", "prefill_hidden38"), {}),
    ]
    assert [temporary.deallocate_calls for temporary in temporaries] == [[True], [True]]


def test_mlp_residual_add_profile_rejects_unknown_value(monkeypatch, expect_error):
    monkeypatch.setenv(layer.MLP_RESIDUAL_ADD_PROFILE_ENV, "all_layers_fp32")

    with expect_error(
        ValueError,
        "GEMMA4_MLP_RESIDUAL_ADD_COMPUTE_PROFILE must be one of: " "layer38_decode_fp32_inputs_bf16_output",
    ):
        layer._apply_mlp_residual_add("residual", "hidden", layer_idx=38, is_decode=True)


def test_mlp_residual_add_profile_releases_temporaries_on_add_error(monkeypatch, expect_error):
    monkeypatch.setenv(
        layer.MLP_RESIDUAL_ADD_PROFILE_ENV,
        layer.MLP_RESIDUAL_ADD_LAYER38_DECODE_FP32_INPUTS_BF16_OUTPUT_PROFILE,
    )
    temporaries = []

    def fake_typecast(value, _dtype):
        temporary = _FakeTemporary(f"{value}_fp32")
        temporaries.append(temporary)
        return temporary

    def fail_add(*_args, **_kwargs):
        raise RuntimeError("candidate add failed")

    monkeypatch.setattr(ttnn, "typecast", fake_typecast)
    monkeypatch.setattr(ttnn, "add", fail_add)

    with expect_error(RuntimeError, "candidate add failed"):
        layer._apply_mlp_residual_add("residual", "hidden", layer_idx=38, is_decode=True)
    assert [temporary.deallocate_calls for temporary in temporaries] == [[True], [True]]
