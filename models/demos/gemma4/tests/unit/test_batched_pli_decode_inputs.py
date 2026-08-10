"""Host-side shape tests for batched PLI decode inputs (Gemma4 E2B/E4B).

The driver is the native Parity / vLLM concurrency path: a device-free call to
``Gemma4Model.prepare_decode_inputs_host``. No accelerator is opened; the
ttnn tensors stay host-side because ``from_torch`` is only given a mesh
mapper, never a device. Single-user layout remains bit-identical.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ttnn", reason="TT runtime shim required to import the model")
import ttnn
from models.demos.gemma4.tt.model import Gemma4Model


class _FakeConfig:
    hidden_size = 8
    hidden_size_per_layer_input = 4
    rms_norm_eps = 1e-6


def _fake_per_layer_input_weights(hidden: int, pli: int, n_layers: int) -> dict:
    return {
        "embed_tokens_per_layer": torch.randn(2048, n_layers * pli),
        "per_layer_model_projection": torch.randn(n_layers * pli, hidden),
        "per_layer_projection_norm": torch.ones(pli),
    }


def _fake_model(n_layers: int = 3) -> Gemma4Model:
    m = Gemma4Model.__new__(Gemma4Model)
    m.mesh_device = None  # no device; ReplicateTensorToMesh skipped
    m.hf_config = _FakeConfig()
    m.hidden_size = _FakeConfig.hidden_size
    m.hidden_size_per_layer_input = _FakeConfig.hidden_size_per_layer_input
    m.per_layer_input_weights = _fake_per_layer_input_weights(
        _FakeConfig.hidden_size, _FakeConfig.hidden_size_per_layer_input, n_layers
    )
    m.per_layer_input_scale = 1.0
    m.per_layer_model_projection_scale = 1.0
    m.per_layer_embed_scale = 1.0
    m.embed_scale = 1.0
    m._embed_weight_cpu = torch.randn(2048, _FakeConfig.hidden_size, dtype=torch.float32)
    m.layers = [object()] * n_layers  # length only matters
    m.kv_shared_layer_map = {}
    return m


def test_batched_pli_layout_and_bit_identity_with_single_user() -> None:
    model = _fake_model(n_layers=3)
    tokens = torch.tensor([13, 1018, 1718], dtype=torch.long)
    pos = torch.tensor([0, 7, 3], dtype=torch.long)

    out = Gemma4Model.prepare_decode_inputs_host(model, tokens, pos)
    assert isinstance(out, tuple) and len(out) == 5
    _, _, _, _, pli_tt = out
    assert pli_tt is not None
    got = ttnn.to_torch(pli_tt).to(torch.float32)
    assert got.shape == (1, 3, 3, _FakeConfig.hidden_size_per_layer_input)

    # Compare per-user rows against the single-user reference path.
    for user in range(3):
        single_out = Gemma4Model.prepare_decode_inputs_host(model, tokens[user : user + 1], pos[user : user + 1])
        single = ttnn.to_torch(single_out[4]).to(torch.float32).reshape(3, -1)
        assert torch.equal(single, got[0, :, user, :]), f"user {user} differs"


def test_single_user_layout_unchanged() -> None:
    model = _fake_model(n_layers=3)
    tokens = torch.tensor([1071], dtype=torch.long)  # < 2048 vocab
    out = Gemma4Model.prepare_decode_inputs_host(model, tokens, torch.tensor([5]))
    pli_tt = out[4]
    assert ttnn.to_torch(pli_tt).shape == (
        1,
        1,
        3,
        _FakeConfig.hidden_size_per_layer_input,
    )


def test_per_layer_projection_fp32_conversion_is_cached_and_exact() -> None:
    model = _fake_model(n_layers=3)
    token = torch.tensor([[1071]], dtype=torch.int32)
    embeds = model._embed_weight_cpu[token.long()] * model.embed_scale

    # Direct reference matches the original implementation, including the
    # per-call BF16/FP32 conversion boundary.
    weights = model.per_layer_input_weights
    pli_size = model.hidden_size_per_layer_input
    full_n_layers = weights["embed_tokens_per_layer"].shape[-1] // pli_size
    pli_embed = torch.nn.functional.embedding(token.long(), weights["embed_tokens_per_layer"])
    pli_embed = (pli_embed * model.per_layer_embed_scale).reshape(1, 1, full_n_layers, pli_size)
    projection = (
        torch.nn.functional.linear(embeds.float(), weights["per_layer_model_projection"].float())
        * model.per_layer_model_projection_scale
    )
    projection = projection.reshape(1, 1, full_n_layers, pli_size)
    variance = projection.float().pow(2).mean(-1, keepdim=True)
    projection = projection.float() * torch.rsqrt(variance + model.hf_config.rms_norm_eps)
    projection = projection * weights["per_layer_projection_norm"].float()
    expected = (projection + pli_embed.float()) * model.per_layer_input_scale
    expected = [expected[:, :, i, :].to(torch.bfloat16) for i in range(len(model.layers))]

    first = model._compute_per_layer_inputs(token, embeds)
    cached = model._per_layer_model_projection_fp32
    assert cached is not None
    assert cached.dtype == torch.float32
    second = model._compute_per_layer_inputs(token, embeds)

    assert model._per_layer_model_projection_fp32 is cached
    assert all(torch.equal(got, want) for got, want in zip(first, expected))
    assert all(torch.equal(got, want) for got, want in zip(second, expected))


def test_batched_rejects_missing_pli(monkeypatch, expect_error) -> None:
    model = _fake_model(n_layers=3)
    monkeypatch.setattr(
        Gemma4Model,
        "compute_host_embeddings",
        lambda self, tok: (None, None),
    )
    tokens = torch.tensor([1, 2], dtype=torch.long)
    with expect_error(ValueError, "compute_host_embeddings"):
        Gemma4Model.prepare_decode_inputs_host(model, tokens, torch.tensor([0, 1]))
