# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Checkpoint tensors -> the reference/HF model's own parameter names.

``v4_weight_prep`` already renames the *surface* differences (``self_attn``↔``attn``,
``mlp``↔``ffn``, ``weight_scale_inv``↔``scale``). What it deliberately does **not** do is
the two things that actually block loading the released checkpoint into a model object:

* **Expert packing.** The checkpoint stores 256 separate experts as ``w1``/``w2``/``w3``;
  ``DeepseekV4Experts`` holds one ``gate_up_proj`` of ``[E, 2I, H]`` (gate and up stacked on
  dim 0) plus one ``down_proj`` of ``[E, H, I]``. Emitting 768 tensors into a 2-tensor module
  is where naive loaders silently drop experts, because ``load_state_dict`` reports the first
  missing key and stops caring.
* **Naming that is not a rename.** ``attn_norm`` is ``input_layernorm``, ``attn.attn_sink`` is
  ``self_attn.sinks``, ``attn.compressor.norm`` is ``self_attn.compressor.kv_norm``, the
  indexer lives under ``attn.indexer.*`` in the checkpoint but under
  ``self_attn.compressor.indexer.*`` in the model, and mHC is ``hc_attn_*`` outside versus
  ``attn_hc.*`` inside.

This module is also the shape of the loader the device path needs, so it is written once and
tested here rather than re-derived under hardware pressure.

Layer payloads are large: one layer's experts alone are 12.9 GiB in bfloat16, and all 43 do
not fit in host RAM (~559 GiB). So everything here is per-layer and iterator-shaped, and
:func:`audit_layer` derives shapes from shard headers without reading a single payload byte.
"""

from __future__ import annotations

import re
from typing import Iterator

import torch

from models.demos.deepseek_v3_d_p.tt import v4_weight_prep as prep
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import V4Checkpoint

#: Checkpoint suffix (after ``layers.<N>.``) -> reference parameter name. Covers every
#: non-expert tensor. Anything absent here is either derived at runtime (rope tables) or a
#: loading bug, and :func:`audit_layer` distinguishes the two.
LAYER_ALIASES: dict[str, str] = {
    # norms and mHC
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_scale": "ffn_hc.scale",
    # attention
    "attn.wq_a.weight": "self_attn.q_a_proj.weight",
    "attn.q_norm.weight": "self_attn.q_a_norm.weight",
    "attn.wq_b.weight": "self_attn.q_b_proj.weight",
    "attn.wkv.weight": "self_attn.kv_proj.weight",
    "attn.kv_norm.weight": "self_attn.kv_norm.weight",
    "attn.wo_a.weight": "self_attn.o_a_proj.weight",
    "attn.wo_b.weight": "self_attn.o_b_proj.weight",
    "attn.attn_sink": "self_attn.sinks",
    # compressor (CSA and HCA). Note ``norm`` -> ``kv_norm``: the checkpoint names the
    # compressor's own RMSNorm after the module, the model after what it normalises.
    "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
    "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
    "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
    "attn.compressor.ape": "self_attn.compressor.position_bias",
    # indexer (CSA only). The checkpoint nests it as ``attn.indexer.*``; the model puts it
    # under the compressor it belongs to.
    "attn.indexer.compressor.wkv.weight": "self_attn.compressor.indexer.kv_proj.weight",
    "attn.indexer.compressor.wgate.weight": "self_attn.compressor.indexer.gate_proj.weight",
    "attn.indexer.compressor.norm.weight": "self_attn.compressor.indexer.kv_norm.weight",
    "attn.indexer.compressor.ape": "self_attn.compressor.indexer.position_bias",
    "attn.indexer.wq_b.weight": "self_attn.compressor.indexer.q_b_proj.weight",
    "attn.indexer.weights_proj.weight": "self_attn.compressor.indexer.scorer.weights_proj.weight",
    # router
    "ffn.gate.weight": "mlp.gate.weight",
    # NAME_RENAMES maps ``e_score_correction_bias`` -> ``bias`` on the way out; the
    # model keeps the long name, so this alias inverts that rename.
    "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
    "ffn.gate.tid2eid": "mlp.gate.tid2eid",
    # shared expert (unpacked in the checkpoint, unpacked in the model)
    "ffn.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
    "ffn.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
}

#: Checkpoint suffix -> reference name for tensors the model computes itself. Loading these
#: would be wrong; the audit exempts them by name so an omission cannot hide behind them.
DERIVED_LAYER_SUFFIXES: tuple[str, ...] = ()  # nothing in the checkpoint feeds these

#: Model-level reference names built from config at construction time (see the per-layer set).
DERIVED_MODEL_NAMES: frozenset[str] = frozenset(
    f"model.rotary_emb.{which}_{suffix}"
    for which in ("main", "compress")
    for suffix in ("inv_freq", "original_inv_freq")
)

#: Reference names the model builds from config at construction time.
DERIVED_REFERENCE_NAMES: frozenset[str] = frozenset(
    {f"self_attn.compressor.rotary_emb.{which}_inv_freq" for which in ("main", "compress")}
    | {f"self_attn.compressor.rotary_emb.{which}_original_inv_freq" for which in ("main", "compress")}
    | {f"self_attn.compressor.indexer.rotary_emb.{which}_inv_freq" for which in ("main", "compress")}
    | {f"self_attn.compressor.indexer.rotary_emb.{which}_original_inv_freq" for which in ("main", "compress")}
)

MODEL_ALIASES: dict[str, str] = {
    "embed.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "head.weight": "lm_head.weight",
    # The mHC head prefixes its parameters `hc_`; the per-layer mHC modules do not, so the
    # two are not interchangeable patterns.
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_scale": "model.hc_head.hc_scale",
}

EXPERT_RE = re.compile(r"^ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")

EXPERT_WEIGHTS = ("w1", "w2", "w3")


def logical_shape(ref) -> list[int]:
    """Shape the tensor has *after* the publisher's packing is undone.

    MXFP4 experts are stored ``[out, in // 2]`` with two ``e2m1fn`` nibbles per byte (low
    nibble first), so the loaded tensor has twice the last axis.
    """
    shape = list(ref.shape)
    if ref.dtype == "I8":
        shape[-1] *= 2
    return shape


def reference_name(checkpoint_name: str) -> str | None:
    """Reference-model parameter name for a checkpoint tensor, or None if it is not a weight.

    Expert tensors return their per-expert checkpoint path; use :func:`iter_reference_layer`
    (or :func:`expert_shapes`) to get the packed names.
    """
    name = prep.port_name(checkpoint_name)
    if name.startswith("mtp."):
        return None
    if name.startswith("layers."):
        _, layer, suffix = name.split(".", 2)
        if EXPERT_RE.match(suffix):
            return suffix
        if suffix.endswith(".scale"):
            return None
        try:
            return LAYER_ALIASES[suffix]
        except KeyError:
            raise KeyError(f"no reference name for checkpoint tensor {checkpoint_name!r} (suffix {suffix!r})") from None
    if name.endswith(".scale"):
        return None
    try:
        return MODEL_ALIASES[name]
    except KeyError:
        raise KeyError(f"no reference name for checkpoint tensor {checkpoint_name!r}") from None


def expert_shapes(ckpt: V4Checkpoint, layer: int) -> dict[str, list[int]]:
    """Packed ``gate_up_proj`` / ``down_proj`` shapes derived from the shard headers alone."""
    n_expert, w1, w2 = 0, None, None
    for name in ckpt.layer_names(layer):
        suffix = name.split(f"layers.{layer}.", 1)[1]
        m = EXPERT_RE.match(suffix)
        if not m:
            continue
        n_expert = max(n_expert, int(m.group(1)) + 1)
        if m.group(2) == "w1":
            w1 = logical_shape(ckpt.index[name])  # [I, H]
        elif m.group(2) == "w2":
            w2 = logical_shape(ckpt.index[name])  # [H, I]
    if w1 is None or w2 is None:
        raise ValueError(f"layer {layer}: no expert w1/w2 in the checkpoint")
    return {
        "mlp.experts.gate_up_proj": [n_expert, 2 * w1[0], w1[1]],
        "mlp.experts.down_proj": [n_expert, w2[0], w2[1]],
    }


def audit_layer(ckpt: V4Checkpoint, layer: int, reference_names: dict[str, list[int]]) -> dict:
    """Compare what the checkpoint offers for one layer against what the model demands.

    Header-only: no payload bytes are read, so this can run over all 43 layers in seconds.
    ``reference_names`` maps parameter name -> expected shape for that layer (relative names).
    """
    offered: dict[str, list[int]] = {}
    for name in ckpt.layer_names(layer):
        suffix = name.split(f"layers.{layer}.", 1)[1]
        if suffix.endswith(".scale"):
            continue
        mapped = reference_name(name)
        if EXPERT_RE.match(suffix):
            continue  # handled as a packed pair below
        offered[mapped] = logical_shape(ckpt.index[name])
    offered.update(expert_shapes(ckpt, layer))

    expected = {k: list(v) for k, v in reference_names.items() if k not in DERIVED_REFERENCE_NAMES}
    missing = sorted(set(expected) - set(offered))
    extra = sorted(set(offered) - set(expected))
    mismatched = {k: (offered[k], expected[k]) for k in set(expected) & set(offered) if offered[k] != expected[k]}
    return {
        "layer": layer,
        "offered": len(offered),
        "expected": len(expected),
        "missing": missing,
        "extra": extra,
        "shape_mismatch": mismatched,
        "derived_exempt": sorted(set(reference_names) - set(expected)),
    }


def audit_model(ckpt: V4Checkpoint, reference_names: dict[str, list[int]]) -> dict:
    """Same contract as :func:`audit_layer` for the non-layer parameters (embed, norm, head)."""
    offered: dict[str, list[int]] = {}
    for name in ckpt.names():  # mtp.* excluded by names(); we do not build an MTP stack
        if name.startswith("layers.") or name.endswith(".scale"):
            continue
        offered[reference_name(name)] = logical_shape(ckpt.index[name])
    expected = {k: list(v) for k, v in reference_names.items() if k not in DERIVED_MODEL_NAMES}
    return {
        "offered": len(offered),
        "expected": len(expected),
        "missing": sorted(set(expected) - set(offered)),
        "extra": sorted(set(offered) - set(expected)),
        "shape_mismatch": {
            k: (offered[k], expected[k]) for k in set(expected) & set(offered) if offered[k] != expected[k]
        },
    }


def iter_reference_layer(
    ckpt: V4Checkpoint,
    layer: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    skip_experts: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield one layer's parameters under the reference model's names.

    Peak memory is bounded by the largest single expert tensor plus the two packed expert
    buffers; nothing accumulates across layers, because the whole model in bfloat16 is ~559
    GiB and does not fit in host RAM. Callers must consume each tensor before asking for the
    next layer (that is what makes this a stream rather than a slower way to run out of memory).

    ``skip_experts`` leaves the routed experts in the checkpoint entirely: nothing is read for
    them and no stack is yielded. Callers that answer ``gate_up_proj[e]`` on demand
    (:class:`LazyExpertStack`) must set it -- without it, installing a 43-layer model would
    dequantize all 1032 GiB of experts (256 per layer at 96 MiB each in fp32) to build stacks
    that are thrown away unread.
    """
    experts: dict[int, dict[str, torch.Tensor]] = {}
    for name in ckpt.layer_names(layer):
        suffix = name.split(f"layers.{layer}.", 1)[1]
        if suffix.endswith(".scale"):
            continue
        m = EXPERT_RE.match(suffix)
        if m:
            if skip_experts:
                continue
            experts.setdefault(int(m.group(1)), {})[m.group(2)] = ckpt.dequantized(name, dtype)
            continue
        yield reference_name(name), ckpt.dequantized(name, dtype)

    if skip_experts:
        return
    if experts:
        n = max(experts) + 1
        have = sorted(experts)
        if have != list(range(n)):
            raise ValueError(f"layer {layer}: expert ids {have[:4]}… are not contiguous 0..{n - 1}")
        gate_up = torch.empty((n, 2 * experts[0]["w1"].shape[0], experts[0]["w1"].shape[1]), dtype=dtype)
        down = torch.empty((n, experts[0]["w2"].shape[0], experts[0]["w2"].shape[1]), dtype=dtype)
        for e in range(n):
            parts = experts.pop(e)
            if set(parts) != set(EXPERT_WEIGHTS):
                raise ValueError(f"layer {layer} expert {e}: got {sorted(parts)}, want {list(EXPERT_WEIGHTS)}")
            # gate and up are stacked on the output axis, matching nn.Linear concatenation;
            # reversing this pair is silent and turns SwiGLU into a different function.
            gate_up[e, : parts["w1"].shape[0]] = parts["w1"]
            gate_up[e, parts["w1"].shape[0] :] = parts["w3"]
            down[e] = parts["w2"]
            del parts
        yield "mlp.experts.gate_up_proj", gate_up
        yield "mlp.experts.down_proj", down


MODEL_LEVEL = ("embed.weight", "norm.weight", "head.weight", "hc_head_base", "hc_head_fn", "hc_head_scale")


def set_param(model: torch.nn.Module, dotted: str, tensor: torch.Tensor) -> None:
    """Replace one meta parameter with the loaded one.

    Assigning ``param.data`` is refused across the meta boundary ("incompatible tensor type"),
    so the parameter object is swapped instead. Shapes are checked first: a wrong shape here
    would otherwise broadcast a weight into the module and train/serve a quietly transposed
    projection.
    """
    parent, _, attr = dotted.rpartition(".")
    module = model.get_submodule(parent) if parent else model
    current = getattr(module, attr)
    if tuple(current.shape) != tuple(tensor.shape):
        raise ValueError(
            f"{dotted}: model wants {tuple(current.shape)}, checkpoint gives {tuple(tensor.shape)} "
            f"(dtype {tensor.dtype}, requires_grad={current.requires_grad})"
        )
    setattr(module, attr, torch.nn.Parameter(tensor, requires_grad=False))


def free_module_params(module: torch.nn.Module) -> None:
    """Drop a finished layer's weights. Peak memory is bounded by one layer, not the stack."""
    for name, p in list(module.named_parameters()):
        parent, _, attr = name.rpartition(".")
        holder = module.get_submodule(parent) if parent else module
        setattr(
            holder,
            attr,
            torch.nn.Parameter(torch.empty(tuple(p.shape), dtype=p.dtype, device="meta"), requires_grad=False),
        )


def build(cfg, dtype: torch.dtype):
    """Model at real geometry with parameters on meta and *buffers* computed for real.

    ``include_buffers=False`` matters: the RoPE inverse-frequency tables are derived in
    ``__init__`` and would otherwise land on meta, and they are not in the checkpoint.

    Imported here rather than at module scope so that reading weights does not require
    building a reference model, and so this module stays importable without accelerate.
    """
    from accelerate import init_empty_weights

    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    with init_empty_weights(include_buffers=False):
        model = DeepseekV4ForCausalLM(cfg)
    model.to(dtype)
    model.eval()
    return model


class LazyExpertStack:
    """Stand-in for a stacked expert parameter that materialises one expert at a time.

    ``iter_reference_layer`` builds the whole ``[n_experts, ...]`` stack because a prefill
    touches most experts. Decode does not: ``Experts.forward`` hits the routed experts a
    token actually selected, and indexes the stack once per hit
    (``self.gate_up_proj[expert_idx]``), never as a matrix. So holding all ~6.4 GiB per layer
    to use 6 of them is what makes host decode look impossible when it is only wasteful.

    This object supplies exactly what that line asks for: ``stack[e]`` reads expert ``e`` and
    composes it with the same rule as :func:`iter_reference_layer` -- gate and up stacked on
    the output axis in ``w1``-then-``w3`` order, down from ``w2``. The order is not
    interchangeable (reversing the pair silently changes SwiGLU), so it is asserted against
    the loader's own output in ``tests/pcc/test_v4_real_weight_load_map.py`` rather than restated.

    A one-entry cache is kept because greedy decode revisits popular experts across steps;
    it is deliberately tiny, since a full cache is the thing being avoided.
    """

    def __init__(self, ckpt: V4Checkpoint, layer: int, role: str, *, dtype=torch.float32, cache: int = 1):
        if role not in ("gate_up", "down"):
            raise ValueError(f"role must be 'gate_up' or 'down', got {role!r}")
        self.ckpt, self.layer, self.role, self.dtype = ckpt, layer, role, dtype
        self.cache, self._cached_id, self._cached = cache, None, None
        self.materializations = 0
        self.bytes_read = 0

    def _expert_names(self, expert: int) -> tuple[str, ...]:
        prefix = f"layers.{self.layer}.ffn.experts.{expert}."
        names = tuple(prefix + w + ".weight" for w in (EXPERT_WEIGHTS))
        missing = [n for n in names if n not in self.ckpt.index]
        if missing:
            raise KeyError(f"layer {self.layer} expert {expert}: missing {missing}")
        return names

    def __getitem__(self, expert: int) -> torch.Tensor:
        if not isinstance(expert, int):
            expert = int(expert)
        if self._cached is not None and expert == self._cached_id:
            return self._cached
        before = self.ckpt.bytes_read
        w1, w2, w3 = (self.ckpt.dequantized(n, self.dtype) for n in self._expert_names(expert))
        self.materializations += 1
        self.bytes_read += self.ckpt.bytes_read - before
        # Same composition as iter_reference_layer: gate then up on the output axis, because
        # nn.Linear concatenates them in that order and the swap is silent.
        out = torch.cat([w1, w3], dim=0) if self.role == "gate_up" else w2
        if self.cache:
            self._cached_id, self._cached = expert, out
        return out

    def __repr__(self) -> str:
        return f"LazyExpertStack(layer={self.layer}, role={self.role!r}, reads={self.materializations})"


__all__ = [
    "MODEL_LEVEL",
    "LazyExpertStack",
    "build",
    "free_module_params",
    "set_param",
    "LAYER_ALIASES",
    "MODEL_ALIASES",
    "DERIVED_REFERENCE_NAMES",
    "logical_shape",
    "reference_name",
    "expert_shapes",
    "audit_model",
    "EXPERT_RE",
    "DERIVED_MODEL_NAMES",
    "audit_layer",
    "iter_reference_layer",
]
