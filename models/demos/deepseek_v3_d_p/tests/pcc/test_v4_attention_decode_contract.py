# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Decode contract for V4 sliding attention: what ``seq_len == 1`` must reproduce.

#71 must implement attention at ``seq_len == 1`` against a cache. The device module
can't be validated yet (no usable device), but the contract it has to satisfy is
measurable now, against the reference: **a sequence of single-token steps must
reproduce one prefill of the whole sequence.** That equality is the target for our
sliding window, and it turns "is decode correct?" from a design hope into a number.

The window bound is pinned too. A window that silently attended to everything would
still produce plausible logits, so a test that only checks outputs cannot detect it.
Here the boundary is measured from both sides: a token outside the window must not
move the result, and a token inside it must.

Construction facts this file depends on (all read from the reference, not inferred
from DeepSeek-V3, which differs structurally):

* ``DeepseekV4Attention(config, layer_idx)`` — two args, no layer-scaling ones.
* Every projection is ``bias=False``; **V4 has no ``use_mla``/``learned_rope``/
  ``q_proj`` at all.** Q is low-rank: ``q_a_proj -> q_a_norm (RMSNorm) -> q_b_proj
  -> q_b_norm (UnweightedRMSNorm)``. KV is a single shared head (MQA): ``kv_proj``
  emits one ``head_dim`` vector read as both key and value.
* ``compressor`` is ``None`` exactly when ``layer_types[i] == "sliding_attention"``
  — it is layer-type driven, not a separate flag.
* ``apply_rotary_pos_emb(x, cos, sin)`` rotates the **trailing** rope slice
  (``[nope | rope]`` layout) with interleaved pairs, and takes cos/sin **half-sized**
  (one entry per pair) since it does its own ``repeat_interleave(2)``.
* ``position_embeddings`` is a ``{"main": …, "compress": …}`` dict; sliding layers
  read ``"main"``.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
)
from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs

WINDOW = 4


def make_cfg(**overrides) -> DeepseekV4Config:
    preset = V4ModelArgs.tiny(1)
    merged = {
        **preset.drive_reference().__dict__,
        "layer_types": ["sliding_attention"],  # -> compressor is None
        "sliding_window": WINDOW,
        **overrides,
    }
    return DeepseekV4Config(**merged)


@pytest.fixture(scope="module")
def cfg():
    return make_cfg()


def _fill(module: torch.nn.Module, gen: torch.Generator) -> None:
    """Standalone sublayers bypass PreTrainedModel.init_weights, so weights must be
    set explicitly — a bare block otherwise reads uninitialised memory."""
    with torch.no_grad():
        for _, t in module.named_parameters():
            fan_in = t.shape[-1] if t.dim() > 0 else 1
            t.normal_(0.0, 1.0 / (fan_in**0.5), generator=gen)


@pytest.fixture(scope="module")
def attn(cfg):
    torch.manual_seed(11)
    module = DeepseekV4Attention(cfg, layer_idx=0).to(torch.float32)
    _fill(module, torch.Generator().manual_seed(11))
    return module.eval()


def rope(cfg, positions: torch.Tensor):
    """Half-sized cos/sin — [B, S, rope_dim/2] — as the reference expects.

    ``rope_parameters`` is keyed by **rope layer type** (``"main"`` / ``"compress"``),
    not by model layer type: a sliding layer reads ``"main"``. Geometry is read from
    the config rather than hardcoded — the tiny preset uses rope_theta=10000 and
    partial_rotary_factor=0.25 (rope slice 128 of head_dim 512), while V4-Flash uses
    0.125 (slice 64). V4 has no ``v_head_dim``; that is a DeepSeek-V3 field.
    """
    rp = cfg.rope_parameters["main"]
    theta = rp["rope_theta"]
    rd = int(round(rp["partial_rotary_factor"] * cfg.head_dim))
    inv = theta ** (-torch.arange(0, rd, 2, dtype=torch.float32) / rd)
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv  # [B, S, rd/2]
    return freqs.cos(), freqs.sin()


def run(attn, cfg, x, start, cache=None, mask=None):
    b, s, _ = x.shape
    positions = torch.arange(start, start + s).unsqueeze(0).expand(b, -1)
    cos, sin = rope(cfg, positions)
    out, _ = attn(
        x,
        {"main": (cos, sin)},
        positions,
        mask,
        past_key_values=cache,
        cache_position=torch.arange(start, start + s),
    )
    return out


NEG = torch.finfo(torch.float32).min


def window_mask(cfg, q_len, kv_len, q_offset=0):
    """Additive causal + sliding-window mask, built explicitly.

    ``allowed = (k <= q_abs) & (q_abs - k < window)`` — one place, so the semantics
    under test are stated rather than inherited from a helper whose exact convention
    would itself need a test. Built by hand because the installed transformers has
    no ``sliding_window_aware_common_mask_postprocess`` (a name I had assumed).
    """
    q = torch.arange(q_offset, q_offset + q_len).unsqueeze(1)
    k = torch.arange(kv_len).unsqueeze(0)
    allowed = (k <= q) & ((q - k) < cfg.sliding_window)
    mask = torch.where(allowed, torch.zeros_like(allowed, dtype=torch.float32), torch.full_like(allowed, NEG, dtype=torch.float32))
    return mask.view(1, 1, q_len, kv_len)


def test_decode_step_matches_prefill_at_the_last_position(cfg, attn):
    """The contract: one single-token step against a cache == the prefill row."""
    torch.manual_seed(4242)
    x = torch.randn(1, 8, cfg.hidden_size)

    with torch.no_grad():
        prefill = run(attn, cfg, x, 0, cache=None, mask=window_mask(cfg, 8, 8))[0, -1]

        cache = DynamicCache(config=cfg)
        _ = run(attn, cfg, x[:, :7], 0, cache=cache, mask=window_mask(cfg, 7, 7))

        # The reference's DynamicSlidingWindowLayer keeps exactly window-1 past keys,
        # so every retained key is already inside the window and the decode step needs
        # no mask at all. Asserted, because the device cache allocation follows from it.
        assert cache.layers[0].keys.shape[2] == cfg.sliding_window - 1, (
            f"sliding cache kept {cache.layers[0].keys.shape[2]} keys, expected "
            f"window-1 = {cfg.sliding_window - 1}; our device cache must match the "
            "depth the reference actually maintains"
        )
        step = run(attn, cfg, x[:, 7:], 7, cache=cache, mask=None)[0, -1]

    assert torch.isfinite(prefill).all() and torch.isfinite(step).all()
    # Scale-normalised: elementwise division by |value| explodes on near-zero
    # elements and would report a false divergence.
    err = float((prefill - step).abs().max())
    scale = float(step.abs().max())
    assert err / scale < 1e-5, (
        f"decode step diverged from prefill: max abs err {err:.3e} vs output scale "
        f"{scale:.3e} ({err / scale:.3e} relative)"
    )


def test_sliding_window_bounds_attention(cfg, attn):
    """Out-of-window must be inert and in-window must be live, or the test is vacuous."""
    torch.manual_seed(4243)
    x = torch.randn(1, WINDOW + 5, cfg.hidden_size)
    mask = window_mask(cfg, WINDOW + 5, WINDOW + 5)
    with torch.no_grad():
        base = run(attn, cfg, x, 0, mask=mask)[0, -1]

        far = x.clone()
        far[0, 0] += 10.0  # distance WINDOW+4 -> outside
        out_far = run(attn, cfg, far, 0, mask=mask)[0, -1]

        near = x.clone()
        near[0, -2] += 10.0  # distance 1 -> inside
        out_near = run(attn, cfg, near, 0, mask=mask)[0, -1]

    assert torch.allclose(base, out_far, atol=1e-6, rtol=1e-6), (
        "the last-position output moved when an out-of-window token changed; "
        "attention is not bounded by the window"
    )
    assert not torch.allclose(base, out_near, atol=1e-6, rtol=1e-6), (
        "the output ignores an in-window token; fixture is degenerate"
    )


def test_window_boundary_is_measured_not_assumed(cfg, attn):
    """Cache depth is a real design input: `window` vs `window-1` kept differs by one slot."""
    torch.manual_seed(4244)
    x = torch.randn(1, WINDOW + 1, cfg.hidden_size)
    mask = window_mask(cfg, WINDOW + 1, WINDOW + 1)
    with torch.no_grad():
        base = run(attn, cfg, x, 0, mask=mask)[0, -1]
        at_w = x.clone(); at_w[0, 0] += 10.0  # distance == WINDOW
        at_w_1 = x.clone(); at_w_1[0, 1] += 10.0  # distance == WINDOW-1
        out_w = run(attn, cfg, at_w, 0, mask=mask)[0, -1]
        out_w_1 = run(attn, cfg, at_w_1, 0, mask=mask)[0, -1]

    excluded = torch.allclose(base, out_w, atol=1e-6, rtol=1e-6)
    included = not torch.allclose(base, out_w_1, atol=1e-6, rtol=1e-6)
    assert included, "even a distance-(window-1) token is inert; the mask is not what we assume"
    # HF's sliding-window mask allows distances 0..window-1, so a token at distance
    # exactly `window` must be inert. If this flips, the convention is inclusive and
    # the device cache must keep window+1 keys rather than window — a real design
    # difference, caught here rather than in a silently-truncated cache.
    assert excluded, (
        f"a token at distance exactly {WINDOW} still influenced the output, so the "
        f"window is inclusive: the device cache must keep {WINDOW + 1} keys, not {WINDOW}"
    )


def test_un_rotation_before_the_grouped_projection_is_load_bearing(cfg, attn):
    """K==V means V carried rope, so the reference un-rotates the output first.

    If that step were dropped, results must differ measurably — i.e. #71 cannot skip it.
    """
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    torch.manual_seed(4245)
    x = torch.randn(1, 4, cfg.hidden_size)
    b, s, _ = x.shape
    positions = torch.arange(s).unsqueeze(0).expand(b, -1)
    cos, sin = rope(cfg, positions)
    hidden = (*x.shape[:2], -1, attn.head_dim)

    with torch.no_grad():
        q_r = attn.q_a_norm(attn.q_a_proj(x))
        q = attn.q_b_norm(attn.q_b_proj(q_r).view(*hidden).transpose(1, 2))
        q = apply_rotary_pos_emb(q, cos, sin)
        kv = attn.kv_norm(attn.kv_proj(x)).view(*hidden).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        raw, _ = eager_attention_forward(
            attn, q, kv, kv, None, scaling=attn.scaling,
            sliding_window=attn.sliding_window, s_aux=attn.sinks,
        )
        unrot = apply_rotary_pos_emb(raw.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = unrot.reshape(b, s, cfg.o_groups, -1)
        with_un = attn.o_b_proj(attn.o_a_proj(grouped).flatten(2))
        grouped_raw = raw.reshape(b, s, cfg.o_groups, -1)
        without = attn.o_b_proj(attn.o_a_proj(grouped_raw).flatten(2))

        module_out = run(attn, cfg, x, 0, mask=window_mask(cfg, 4, 4))[0, -1]

    assert not torch.allclose(with_un[0, -1], without[0, -1], atol=1e-6, rtol=1e-6), (
        "un-rotating the attention output changes nothing — either rope is inert here "
        "or the fixture is degenerate"
    )
    assert torch.allclose(module_out, with_un[0, -1], atol=1e-5, rtol=1e-5), (
        "the hand-composed op order does not reproduce the module: #71's transcription "
        "is wrong about where rope-undo, the grouped projection, or the norms go"
    )
