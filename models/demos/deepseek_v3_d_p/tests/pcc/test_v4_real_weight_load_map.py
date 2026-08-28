# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The released checkpoint must map onto the reference model, name for name.

`v4_weight_prep` renames surfaces; this is the part that actually decides whether the
checkpoint can be *loaded*: 256 separate `w1/w2/w3` experts becoming one `[E, 2I, H]`
`gate_up_proj`, and a dozen names that are not renames at all (`attn_norm` →
`input_layernorm`, `attn.compressor.norm` → `self_attn.compressor.kv_norm`, `attn.indexer.*`
→ `self_attn.compressor.indexer.*`, `hc_attn_*` → `attn_hc.*`, `ffn.gate.bias` →
`mlp.gate.e_score_correction_bias`).

The main test derives shapes from shard headers only — no payload bytes — so it covers all
43 layers, all three attention classes and both MLP classes in seconds. It reports missing,
unexpected and shape-mismatched names separately, because the failure that matters most here
is the *silent* one: `load_state_dict` with a dropped expert reports one missing key and
leaves the rest of the 255-expert tensor at its uninitialised value.

Two payload-level tests follow. They read real bytes (~150 MB total) and check the two things
a header audit cannot see: that dequantization produces the dtype/shape the module wants, and
that stacking gate above up matches `DeepseekV4Experts._apply_gate`, which splits the packed
matrix with `chunk(2, dim=-1)`. Reversing that pair is silent and turns SwiGLU into a
different function.
"""

from __future__ import annotations

import types

import pytest
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig
from transformers.activations import ACT2FN

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
)
from models.demos.deepseek_v3_d_p.tt import v4_weight_load as load
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import V4Checkpoint, default_snapshot_dir

pytestmark = pytest.mark.skipif(default_snapshot_dir() is None, reason="no V4-Flash snapshot present")


@pytest.fixture(scope="module")
def snapshot():
    return default_snapshot_dir()


@pytest.fixture(scope="module")
def ckpt(snapshot):
    return V4Checkpoint(snapshot)


@pytest.fixture(scope="module")
def model(snapshot):
    cfg = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
    with init_empty_weights():
        built = DeepseekV4ForCausalLM(cfg)
    return built, cfg


def _layer_names(model, layer: int) -> dict[str, tuple[int, ...]]:
    node = model.model.layers[layer]
    return {n: tuple(p.shape) for n, p in list(node.named_parameters()) + list(node.named_buffers())}


def _top_names(model) -> dict[str, torch.Size]:
    return {
        n: tuple(p.shape)
        for n, p in list(model.named_parameters()) + list(model.named_buffers())
        if not n.startswith("model.layers.")
    }


def test_every_layer_maps_onto_the_reference_model(ckpt, model):
    """No missing, no extra, no shape drift — for all 43 layers, headers only."""
    built, cfg = model
    assert ckpt.bytes_read == 0, "audit must not read payload bytes to check shapes"
    problems = []
    for layer in range(cfg.num_hidden_layers):
        report = load.audit_layer(ckpt, layer, _layer_names(built, layer))
        if report["missing"] or report["extra"] or report["shape_mismatch"]:
            problems.append((layer, cfg.layer_types[layer], report))
    assert not problems, "checkpoint/reference mapping problems: " + "; ".join(
        f"layer {i} [{t}] missing={r['missing']} extra={r['extra']} mismatch={r['shape_mismatch']}"
        for i, t, r in problems[:4]
    )
    assert ckpt.bytes_read == 0, "shape audit silently read payload"


def test_model_level_tensors_map(ckpt, model):
    """Embeddings, final norm, mHC head and lm head. Rope tables are derived, not loaded."""
    built, _ = model
    report = load.audit_model(ckpt, _top_names(built))
    assert report["missing"] == [] and report["extra"] == [], report
    assert report["shape_mismatch"] == {}, report
    assert "model.rotary_emb.main_inv_freq" in load.DERIVED_MODEL_NAMES


def test_schedule_variety_is_covered_not_just_one_layer_type(ckpt, model):
    """The released schedule has sliding, CSA and HCA layers; a CSA-only audit proves little.

    CSA is the only class with an indexer, so it is the only place the indexer names get
    exercised — and sliding layers are the ones the preset used to omit entirely.
    """
    built, cfg = model
    by_type: dict[str, int] = {}
    for i, t in enumerate(cfg.layer_types):
        by_type.setdefault(t, i)
    assert set(by_type) == {
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    }, cfg.layer_types

    csa = by_type["compressed_sparse_attention"]
    offered = {load.reference_name(n) for n in ckpt.layer_names(csa) if not n.endswith(".scale")}
    assert "self_attn.compressor.indexer.kv_proj.weight" in offered
    assert "self_attn.compressor.indexer.scorer.weights_proj.weight" in offered

    hca = by_type["heavily_compressed_attention"]
    hca_offered = {load.reference_name(n) for n in ckpt.layer_names(hca) if not n.endswith(".scale")}
    assert not any(
        n.startswith("self_attn.compressor.indexer.") for n in hca_offered
    ), "HCA has no indexer (worklog 928348); an indexer here means the layer index is wrong"


def test_router_bias_is_the_score_correction_bias(ckpt):
    """Pins the one alias that is an *inverted* rename, not a new name."""
    layers = [i for i in range(43) if f"layers.{i}.ffn.gate.bias" in ckpt.index]
    assert layers, "no router bias tensors in the checkpoint"
    assert load.reference_name(f"layers.{layers[0]}.ffn.gate.bias") == "mlp.gate.e_score_correction_bias"


def test_non_expert_tensors_materialize_as_the_model_wants(ckpt, model):
    """Payload-level: dtype, shape and finiteness for one layer's non-expert weights."""
    built, _ = model
    layer = 0
    expected = _layer_names(built, layer)
    seen = {}
    for name in ckpt.layer_names(layer):
        suffix = name.split(f"layers.{layer}.", 1)[1]
        if suffix.endswith(".scale") or load.EXPERT_RE.match(suffix):
            continue
        ref_name, tensor = load.reference_name(name), ckpt.dequantized(name, torch.bfloat16)
        assert tensor.dtype is torch.bfloat16, f"{ref_name}: {tensor.dtype}"
        assert torch.isfinite(tensor).all(), f"{ref_name} has non-finite values after dequant"
        seen[ref_name] = tuple(tensor.shape)
    assert seen, "no tensors materialized"
    mismatched = {k: (v, tuple(expected[k])) for k, v in seen.items() if tuple(expected[k]) != v}
    assert not mismatched, f"loaded shapes differ from the module's: {mismatched}"


def test_gate_and_up_stack_in_the_reference_chunk_order(ckpt, model):
    """`gate_up_proj[e]` must put gate above up, because `_apply_gate` chunks on the last axis.

    Verified against the reference's own `_apply_gate` rather than a comment: stacked rows are
    fed through a linear and chunked, and the halves must reproduce the separately-loaded gate
    and up weights. Swapping the pair keeps every shape legal and changes the activation.
    """
    _, cfg = model
    layer, expert = 0, 0
    gate = ckpt.dequantized(f"layers.{layer}.ffn.experts.{expert}.w1.weight", torch.bfloat16)
    up = ckpt.dequantized(f"layers.{layer}.ffn.experts.{expert}.w3.weight", torch.bfloat16)
    down = ckpt.dequantized(f"layers.{layer}.ffn.experts.{expert}.w2.weight", torch.bfloat16)
    assert gate.shape == up.shape, "gate and up must be the same shape to stack"

    packed = torch.cat([gate, up], dim=0).unsqueeze(0)  # [1, 2I, H] as the module holds it
    probe = torch.randn(1, 4, gate.shape[1], dtype=torch.bfloat16)
    # Call the reference's real `_apply_gate` with a minimal `self`; instantiating the module
    # would mean allocating the 12.9 GiB expert buffers we are deliberately not touching here.
    fake = types.SimpleNamespace(act_fn=ACT2FN[cfg.hidden_act], limit=cfg.swiglu_limit)
    acted = DeepseekV4Experts._apply_gate(fake, torch.nn.functional.linear(probe, packed[0]))
    expected = fake.act_fn(torch.nn.functional.linear(probe, gate).clamp(max=cfg.swiglu_limit)) * (
        torch.nn.functional.linear(probe, up).clamp(min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
    )
    err = (acted.float() - expected.float()).abs().max().item()
    assert err < 1e-2, f"stacked gate/up disagrees with the reference chunk order: {err:.3e}"
    assert down.shape[0] * down.shape[1] == down.numel(), "down_proj should stay [H, I]"


def test_lazy_expert_stack_reproduces_what_the_loader_stacks(ckpt):
    """The lazy path must be the *same* composition, not a second interpretation of w1/w2/w3.

    Decode reads ``gate_up_proj[e]`` for the 6 experts a token hits, so a lazy stand-in can
    replace a ~6 GiB stack with ~100 MB. That is only safe if it composes identically: gate
    and up on the output axis in w1-then-w3 order. Getting that pair backwards is silent --
    SwiGLU still runs and produces a different function -- so the check is equality with the
    loader's own stacked tensor, on two experts, rather than a restatement of the rule.
    """
    import gc

    from models.demos.deepseek_v3_d_p.tt.v4_weight_load import LazyExpertStack, iter_reference_layer

    layer = next(i for i in range(43) if any("ffn.experts.0.w1.weight" in n for n in ckpt.layer_names(i)))
    stacked = {}
    for ref_name, tensor in iter_reference_layer(ckpt, layer, dtype=torch.float32):
        if ref_name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
            stacked[ref_name.split(".")[-1]] = tensor
    assert set(stacked) == {"gate_up_proj", "down_proj"}, f"loader offered {sorted(stacked)}"

    stacks = {
        "gate_up": LazyExpertStack(ckpt, layer, "gate_up"),
        "down": LazyExpertStack(ckpt, layer, "down"),
    }
    pairs = {"gate_up": stacked["gate_up_proj"], "down": stacked["down_proj"]}
    for role, full in pairs.items():
        for expert in (0, 3):
            piece = stacks[role][expert]
            assert tuple(piece.shape) == tuple(full[expert].shape), f"{role}[{expert}] shape"
            assert torch.equal(piece, full[expert]), f"{role}[{expert}] differs from the loader's stack"
        assert stacks[role].materializations == 2, "the one-entry cache must not materialise per lookup"
        assert stacks[role].bytes_read < full.numel(), "lazy reads must move far fewer bytes than the stack"
    del stacked, pairs
    gc.collect()
