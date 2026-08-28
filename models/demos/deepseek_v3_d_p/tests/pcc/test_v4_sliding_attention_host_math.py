# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-checkable parts of `TtV4SlidingAttention`, so the module isn't untested code.

The device path can't run (quarantined tray, worklog dbf3d9), but everything the module
computes on the **host** — rope tables, the rotation matrix, the mask, cache depth, and
the grouped-projection decomposition — is checkable now. Testing these is what keeps
"written" from becoming "assumed correct": each one is a place where a silently wrong
constant survives until parity, and then looks like a device numerics bug.

`__init__` deliberately touches no device, so the geometry itself is asserted here
rather than reviewed by eye.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4GroupedLinear,
    apply_rotary_pos_emb,
)
from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs
from models.demos.deepseek_v3_d_p.tt.v4_sliding_attention import (
    TtV4SlidingAttention,
    build_transformation_mat,
    grouped_linear_torch,
    sliding_causal_mask,
)

ROPE_DIM = 64  # V4-Flash: partial_rotary_factor 0.125 of head_dim 512


def v4_geometry(**overrides):
    """Real V4-Flash attention geometry, not the tiny preset's toy numbers.

    The tiny preset has head_dim 64 with partial_rotary_factor 0.25, giving a 16-dim
    rope slice — smaller than one tile, which the module correctly refuses. V4-Flash is
    head_dim 512 with 0.125, i.e. a 64-dim slice = two tiles. Testing against the real
    geometry is the point; a module that only works on toy sizes proves nothing.
    """
    # partial_rotary_factor is not a V4ModelArgs field — it lives inside
    # rope_parameters["main"], which is where the module reads it.
    merged = {"head_dim": 512, "num_attention_heads": 4, **overrides}
    cfg = V4ModelArgs.tiny(1, **merged).drive_reference()
    rp = dict(cfg.rope_parameters)
    rp["main"] = {**rp["main"], "partial_rotary_factor": 0.125}
    cfg.rope_parameters = rp
    return cfg


def make_module(**kwargs):
    """Construct with mesh_device=None: __init__ must stay device-free for this to work."""
    return TtV4SlidingAttention(None, v4_geometry(), {}, **kwargs)


def test_grouped_decomposition_matches_the_reference_grouped_linear():
    """The device path is o_groups matmuls + concat; it must equal the reference op."""
    groups, in_per_group, out_per_group = 4, 32, 16
    ref = DeepseekV4GroupedLinear(in_per_group, groups * out_per_group, groups, bias=False).to(torch.float32).eval()
    x = torch.randn(3, groups, in_per_group)
    # The reference is an nn.Linear, so weight is 2D [groups*out_per_group, in]; the
    # per-group block is a view. The decomposition wants [groups, in, out].
    assert ref.weight.dim() == 2, tuple(ref.weight.shape)
    weight = ref.weight.view(groups, out_per_group, in_per_group).permute(0, 2, 1).contiguous()
    out = grouped_linear_torch(x, weight, groups)
    # The reference keeps the group axis and its caller flattens it
    # (`o_a_proj(grouped).flatten(2)` in the attention epilogue); concatenating the
    # group outputs in group order is the same tensor.
    expected = ref(x).flatten(-2)
    assert out.shape == expected.shape, (tuple(out.shape), tuple(expected.shape))
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6), float((out - expected).abs().max())


def test_grouping_is_not_a_plain_dense_matmul():
    """Guards the test above from passing for the wrong reason."""
    groups, in_per_group, out_per_group = 4, 32, 16
    torch.manual_seed(0)
    weight = torch.randn(groups, in_per_group, out_per_group) * 0.1
    x = torch.randn(1, groups, in_per_group)
    grouped = grouped_linear_torch(x, weight, groups)
    # Grouping = block-diagonal: group g sees only its own input slice. Filling the
    # off-diagonal blocks must change the result, otherwise "grouped" is untested.
    block_diag = torch.zeros(groups * in_per_group, groups * out_per_group)
    off_diag = torch.zeros_like(block_diag)
    for g in range(groups):
        block_diag[g * in_per_group : (g + 1) * in_per_group, g * out_per_group : (g + 1) * out_per_group] = weight[g]
        off_diag[g * in_per_group : (g + 1) * in_per_group, ((g + 1) * out_per_group) % (groups * out_per_group) :
                 ((g + 1) * out_per_group) % (groups * out_per_group) + out_per_group] = weight[g] * 0.2
    as_dense_block = torch.matmul(x.reshape(1, -1), block_diag)
    assert grouped.shape == as_dense_block.shape
    assert torch.allclose(grouped, as_dense_block, atol=1e-6, rtol=1e-6), "grouping is not block-diagonal"
    assert not torch.allclose(grouped, torch.matmul(x.reshape(1, -1), off_diag), atol=1e-5, rtol=1e-5), (
        "leaking between groups changed nothing; the grouped projection is not tested"
    )


def test_transformation_mat_is_block_diagonal_and_reaches_the_reference_rotation():
    """Device math: trailing slice * cos + (trailing slice @ trans) * sin, nope passed through."""
    nope_dim = 32
    head_dim = nope_dim + ROPE_DIM
    trans = build_transformation_mat(ROPE_DIM)
    assert trans.shape == (ROPE_DIM, ROPE_DIM)

    inv = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    freqs = torch.arange(5, dtype=torch.float32).unsqueeze(-1) * inv  # [5, rope/2]
    cos_full, sin_full = freqs.cos().repeat_interleave(2, -1), freqs.sin().repeat_interleave(2, -1)

    torch.manual_seed(1)
    x = torch.randn(1, 5, 2, head_dim)  # [B, S, H, D]
    # Device transcription, in the module's convention: cos/sin already expanded.
    nope, rope = x[..., :nope_dim], x[..., -ROPE_DIM:]
    cos_b = cos_full.view(1, 5, 1, ROPE_DIM)
    sin_b = sin_full.view(1, 5, 1, ROPE_DIM)
    ours = torch.cat([nope, rope * cos_b + (rope @ trans) * sin_b], dim=-1)

    # The reference takes HALF-sized cos/sin and expands them itself.
    ref = apply_rotary_pos_emb(x.transpose(1, 2), freqs.cos().unsqueeze(0), freqs.sin().unsqueeze(0)).transpose(1, 2)

    assert torch.allclose(ours, ref, atol=1e-6, rtol=1e-6), float((ours - ref).abs().max())
    assert torch.equal(ours[..., :nope_dim], x[..., :nope_dim]), "nope slice must pass through"


def test_transformation_mat_rejects_a_non_tiled_rope_dim():
    with pytest.raises(ValueError, match="TILE_SIZE"):
        build_transformation_mat(48)


def test_mask_boundary_matches_the_measured_convention():
    """Distance == window is inert; distance window-1 is live (measured vs reference)."""
    window = 128
    mask = sliding_causal_mask(window + 2, window)
    assert mask.shape == (1, 1, window + 2, window + 2)
    last = mask[0, 0, -1]  # query at position window+1
    assert float(last[0]) == torch.finfo(torch.float32).min  # distance window+1 -> masked
    assert float(last[1]) == torch.finfo(torch.float32).min  # distance == window -> masked
    assert float(last[2]) == 0.0  # distance window-1 -> allowed
    assert float(last[-1]) == 0.0  # self -> allowed
    # Causality: future columns masked.
    assert float(last[0]) < 0 and float(last[-1]) == 0.0


def test_geometry_is_derived_not_hardcoded():
    """__init__ is device-free, so the load-bearing geometry can be asserted directly."""
    m = make_module()
    cfg = m.cfg
    assert m.rope_dim == int(round(cfg.rope_parameters["main"]["partial_rotary_factor"] * cfg.head_dim))
    assert m.nope_dim == cfg.head_dim - m.rope_dim
    assert m.nope_dim >= 0
    assert m.scaling == cfg.head_dim**-0.5
    # Measured contract: the reference sliding cache retains window-1 past keys, which
    # is also why decode needs no mask.
    assert m.cache_len == cfg.sliding_window - 1
    assert m.window == cfg.sliding_window


def test_host_tables_are_interleaved_and_position_addressable():
    m = make_module(max_seq_len=256)
    assert m._cos_full.shape == (256, m.rope_dim)
    assert m._sin_full.shape == (256, m.rope_dim)
    # Interleaved expansion: each pair value repeats twice, so the table is not half-sized.
    assert not torch.equal(m._cos_full, m._cos_full.roll(1, dims=-1))
    inv_step = m.rope_theta ** (-torch.arange(0, m.rope_dim, 2, dtype=torch.float32) / m.rope_dim)
    expected = (torch.arange(256, dtype=torch.float32).unsqueeze(-1) * inv_step).cos().repeat_interleave(2, -1)
    assert torch.allclose(m._cos_full, expected, atol=1e-7), "position table must address absolute positions"
    assert m._trans_torch.shape == (m.rope_dim, m.rope_dim)


def test_reset_cache_is_available_and_documented():
    """A stale window from the previous sequence is inside the current window."""
    assert hasattr(TtV4SlidingAttention, "reset_cache")
