# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Partial RoPE: the reference's interleaved convention vs TTNN's trans_mat.

Completes the host-side numerics pinning for #71. This one is worth pinning in
isolation because the two conventions differ only in **which dims pair up** —
adjacent ``(0,1),(2,3),…`` versus half-split ``(i, i+rd/2)`` — so a mismatch keeps
every shape correct, and on the dot product it is even invisible (see the invariance
test). What it silently changes is the element values, which is what a KV cache and
any cross-stack comparison read.

Reference side, ``apply_rotary_pos_emb``: ``cos``/``sin`` arrive **half-sized** (one
value per pair), are widened by ``repeat_interleave(2, -1)``, and applied to the
**trailing** ``rope_dim`` channels as ``x*cos + rotate_half(x)*sin`` in fp32, with
the leading nope channels untouched. Layout is ``[nope | rope]``.

TTNN side, ``ttnn.experimental.rotary_embedding_llama`` plus
``get_rot_transformation_mat``: a single-tile matrix with ``+1`` on
``(even, odd)`` and ``-1`` on ``(odd, even)``, so ``x @ trans`` yields
``[-x1, x0, -x3, x2, …]``. That is exactly the reference's interleaved
``rotate_half``, which is what makes the pairing identical.

Verified, in this order: the transform matrix equals ``rotate_half``; the full
partial-rope application equals the reference; the half-split convention produces a
*different* tensor from the same cos/sin (so the suite can tell them apart); and
q·k scores are invariant under the dim permutation while element values are not —
the precise claim ``tt/mla/rope.py::interleaved_to_halfsplit_perm`` documents.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    apply_rotary_pos_emb,
    rotate_half,
)
from models.demos.deepseek_v3_d_p.tt.mla.rope import (
    get_rot_transformation_mat,
    interleaved_to_halfsplit_perm,
)

ROPE_DIM = 64  # V4-Flash qk_rope_head_dim
NOPE_DIM = 32  # small stand-in for head_dim - rope_head_dim


def _cos_sin(seq: int, half: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ang = torch.randn(seq, half, generator=g)
    return ang.cos(), ang.sin()


def ttnn_style_partial_rope(x, cos_half, sin_half, trans):
    """TTNN formulation: split trailing rope slice, ``x*cos + (x@trans)*sin``, concat.

    Mirrors rotary_embedding_llama's math with get_rot_transformation_mat, on the
    nope/rope split the ported HCA block performs with ttnn.slice/concat.
    """
    rope_dim = cos_half.shape[-1] * 2
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    cos = cos_half.repeat_interleave(2, dim=-1)
    sin = sin_half.repeat_interleave(2, dim=-1)
    rotated = rope * cos + (rope @ trans[:rope_dim, :rope_dim]) * sin
    return torch.cat([nope, rotated], dim=-1)


def halfsplit_style_partial_rope(x, cos_half, sin_half):
    """The WRONG convention for V4: pairs (i, i + rd/2) instead of adjacent pairs."""
    rope_dim = cos_half.shape[-1] * 2
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    cos = cos_half.repeat_interleave(2, dim=-1)
    sin = sin_half.repeat_interleave(2, dim=-1)
    x1, x2 = rope[..., : rope_dim // 2], rope[..., rope_dim // 2 :]
    r1 = x1 * cos[..., : rope_dim // 2] - x2 * sin[..., : rope_dim // 2]
    r2 = x2 * cos[..., rope_dim // 2 :] + x1 * sin[..., rope_dim // 2 :]
    return torch.cat([nope, torch.cat([r1, r2], dim=-1)], dim=-1)


def _reference(x_bsd, cos_half, sin_half):
    """Call the reference the way the model does: x is [B, H, S, D] and unsqueeze_dim=1
    turns cos/sin [1, S, rope] into the [1, 1, S, rope] broadcast head dim."""
    x = x_bsd.unsqueeze(1)  # [B, S, D] -> [B, 1, S, D]
    out = apply_rotary_pos_emb(x, cos_half.unsqueeze(0), sin_half.unsqueeze(0))
    return out.squeeze(1)


def test_trans_mat_equals_reference_rotate_half():
    """x @ rot_transformation_mat must be the reference's interleaved rotate_half."""
    trans = get_rot_transformation_mat().squeeze(0).squeeze(0)  # [32, 32] single tile
    x = torch.randn(4, 32)
    assert torch.allclose(x @ trans, rotate_half(x), atol=1e-7), "transform matrix is not rotate_half"


def test_partial_rope_matches_the_reference():
    """Full nope/rope split + interleaved rotation, vs the reference function."""
    trans = torch.zeros(ROPE_DIM, ROPE_DIM)
    base = get_rot_transformation_mat().squeeze(0).squeeze(0)  # tile
    trans[: base.shape[0], : base.shape[1]] = base
    # rope_dim 64 spans two tiles; the tile-local pattern repeats per tile.
    trans[32:, 32:] = base

    b, s = 1, 5
    x = torch.randn(b, s, NOPE_DIM + ROPE_DIM)
    cos_half, sin_half = _cos_sin(s, ROPE_DIM // 2)

    ours = ttnn_style_partial_rope(x, cos_half, sin_half, trans)
    ref = _reference(x, cos_half, sin_half)

    assert ours.shape == ref.shape, (tuple(ours.shape), tuple(ref.shape))
    assert torch.allclose(ours, ref, rtol=1e-6, atol=1e-7), float((ours - ref).abs().max())


def test_nope_channels_are_untouched():
    trans = torch.zeros(ROPE_DIM, ROPE_DIM)
    base = get_rot_transformation_mat().squeeze(0).squeeze(0)
    trans[:32, :32] = base
    trans[32:, 32:] = base
    cos_half, sin_half = _cos_sin(4, ROPE_DIM // 2)
    x = torch.randn(1, 4, NOPE_DIM + ROPE_DIM)
    out = ttnn_style_partial_rope(x, cos_half, sin_half, trans)
    assert torch.equal(out[..., :NOPE_DIM], x[..., :NOPE_DIM]), "nope slice must pass through unchanged"
    assert not torch.allclose(out[..., NOPE_DIM:], x[..., NOPE_DIM:], atol=1e-6), "rope slice did not rotate"


def test_halfsplit_convention_is_observably_different():
    """The suite must be able to tell the two pairings apart."""
    trans = torch.zeros(ROPE_DIM, ROPE_DIM)
    base = get_rot_transformation_mat().squeeze(0).squeeze(0)
    trans[:32, :32] = base
    trans[32:, 32:] = base
    cos_half, sin_half = _cos_sin(4, ROPE_DIM // 2)
    x = torch.randn(1, 4, NOPE_DIM + ROPE_DIM)

    interleaved = ttnn_style_partial_rope(x, cos_half, sin_half, trans)
    halfsplit = halfsplit_style_partial_rope(x, cos_half, sin_half)
    assert not torch.allclose(interleaved, halfsplit, rtol=1e-4, atol=1e-6), (
        "the two rope conventions are indistinguishable here, so this suite could "
        "not catch a convention mix-up"
    )


def test_dot_product_is_permutation_invariant_but_values_are_not():
    """Documents what upstream's interleaved_to_halfsplit_perm note claims.

    q.k sums over the rope dims, which the permutation only reorders, so scores are
    identical; the stored values differ element-wise. Measured here rather than
    taken on faith, because it is exactly the boundary between 'harmless within our
    stack' and 'silently wrong against a vLLM-written cache'.
    """
    perm = interleaved_to_halfsplit_perm(ROPE_DIM)
    g = torch.Generator().manual_seed(7)
    q = torch.randn(1, 2, 3, NOPE_DIM + ROPE_DIM, generator=g)
    k = torch.randn(1, 2, 3, NOPE_DIM + ROPE_DIM, generator=g)

    def rope_only(t):
        return t[..., NOPE_DIM:]

    def scores(a, b):
        return (a @ b.transpose(-1, -2))

    # Restrict to rope dims so only the permutation is under test.
    same = scores(rope_only(q), rope_only(k))
    permuted = scores(rope_only(q)[..., perm], rope_only(k)[..., perm])
    assert torch.allclose(same, permuted, rtol=1e-5, atol=1e-6), (
        "rope-dim scores should be invariant to a pure dim permutation"
    )
    assert not torch.allclose(rope_only(k), rope_only(k)[..., perm], atol=1e-6), (
        "permuted values are identical -- fixture does not exercise the layout"
    )
