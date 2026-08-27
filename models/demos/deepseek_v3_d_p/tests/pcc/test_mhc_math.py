# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-only parity check for the mHC transcription.

No device fixture and no hardware: this validates the *math transcription*, not
the TTNN lowering. It pins the parts of mHC that are easy to get subtly wrong
and that would change model output without ever raising an error:

* the ``[H, H, H*H]`` split order of ``fn``'s output,
* scale-before-bias order per output,
* ``pre`` adds ``eps`` after sigmoid but ``post`` is ``2 * sigmoid`` with **no** eps,
* ``comb`` is reshaped to ``[.., H, H]`` before scale/bias, then softmax, then
  Sinkhorn in the exact alternation column -> (row, column) * (iters-1),
* every Sinkhorn division carries ``+ eps``,
* the collapse is expressed as a ``[1,H] x [H,D]`` matmul (what the TTNN path
  does) rather than the reference's elementwise weighted sum,
* ``comb`` is consumed **transposed** in the residual mix.

The TTNN module in ``tt/mhc.py`` is written to perform exactly the operations in
``tt_mhc_math`` below. Equality here is therefore evidence about op order and
semantics; device numerics, tile padding of the ``H=4`` stream axis, and mesh
sharding remain unverified until a device run.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig

HC = DeepSeekV4FlashConfig.HC_MULT  # 4
SINKHORN_ITERS = DeepSeekV4FlashConfig.HC_SINKHORN_ITERS  # 20
HC_EPS = DeepSeekV4FlashConfig.HC_EPS  # 1e-6
NORM_EPS = DeepSeekV4FlashConfig.RMS_NORM_EPS
HIDDEN = DeepSeekV4FlashConfig.EMB_SIZE  # 4096


def _config() -> DeepseekV4Config:
    return DeepseekV4Config(
        hidden_size=HIDDEN,
        hc_mult=HC,
        hc_sinkhorn_iters=SINKHORN_ITERS,
        hc_eps=HC_EPS,
        rms_norm_eps=NORM_EPS,
    )


# --------------------------------------------------------------------------------------
# Mirror of the TTNN op sequence, expressed in torch.
# --------------------------------------------------------------------------------------


def unweighted_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """ttnn: multiply(x,x) -> mean(-1,keep) -> add(eps) -> rsqrt -> multiply(x, .)"""
    sq = x * x
    denom = torch.rsqrt(sq.mean(-1, keepdim=True) + eps)
    return x * denom


def sinkhorn(comb: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """ttnn: divide by (sum(axis, keepdim) + eps); column first, then (row, column)."""
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def tt_mhc_math(streams, fn, base, scale):
    """Exactly the operations ``TtHCMapping.forward`` performs."""
    b, s, hc, d = streams.shape
    mix = (2 + hc) * hc

    flat = streams.reshape(b, s, 1, hc * d)
    flat_norm = unweighted_rmsnorm(flat, NORM_EPS)

    wide = flat_norm @ fn.T.float()  # ttnn.linear
    pre_w, post_w, comb_w = wide[..., :hc], wide[..., hc : 2 * hc], wide[..., 2 * hc :]
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre_s, post_s, comb_s = scale.unbind(0)

    pre_logits = pre_w * pre_s + pre_b
    post_logits = post_w * post_s + post_b
    comb_logits = comb_w.reshape(b, s, hc, hc) * comb_s + comb_b.reshape(hc, hc)

    pre = torch.sigmoid(pre_logits) + HC_EPS
    post = torch.sigmoid(post_logits) * 2.0
    comb = sinkhorn(torch.softmax(comb_logits, dim=-1) + HC_EPS, SINKHORN_ITERS, HC_EPS)

    # collapse as a matmul (the TTNN form), not an elementwise weighted sum
    collapsed = torch.matmul(pre.reshape(b, s, 1, hc), streams.reshape(b, s, hc, d))
    return post, comb, collapsed.reshape(b, s, d)


def tt_apply_site(post, comb, streams, sublayer_out):
    """Exactly ``apply_mhc_site``."""
    b, s, hc, d = streams.shape
    placed = torch.matmul(post.reshape(b, s, 1, hc).transpose(2, 3), sublayer_out.reshape(b, s, 1, d))
    mixed = torch.matmul(comb.transpose(2, 3), streams)
    return placed + mixed


# --------------------------------------------------------------------------------------


def _init_mapping(module, seed=1234):
    """Give the mapping real weights.

    The reference classes declare ``nn.Parameter(torch.empty(...))`` with no
    init, so a freshly constructed module holds uninitialised memory (which can
    be NaN/inf). Filling them deterministically is required for the comparison
    to mean anything.
    """
    g = torch.Generator().manual_seed(seed)
    rows = module.fn.shape[0]
    cols = module.fn.shape[1]
    with torch.no_grad():
        module.fn.copy_(torch.randn(rows, cols, generator=g, dtype=torch.float32) / (cols ** 0.5))
        module.base.copy_(torch.randn(rows, generator=g, dtype=torch.float32) * 0.02)
        module.scale.copy_(torch.randn(module.scale.shape, generator=g, dtype=torch.float32) * 0.1)
    return module


def _init_head(module, seed=5678):
    g = torch.Generator().manual_seed(seed)
    rows, cols = module.hc_fn.shape
    with torch.no_grad():
        module.hc_fn.copy_(torch.randn(rows, cols, generator=g, dtype=torch.float32) / (cols ** 0.5))
        module.hc_base.copy_(torch.randn(rows, generator=g, dtype=torch.float32) * 0.02)
        module.hc_scale.copy_(torch.randn(module.hc_scale.shape, generator=g, dtype=torch.float32) * 0.1)
    return module


def _seeded_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    b, s = 1, 16
    streams = torch.randn(b, s, HC, HIDDEN, generator=g, dtype=torch.float32)
    return streams, g


def test_mapping_matches_reference():
    torch.manual_seed(0)
    ref = _init_mapping(DeepseekV4HyperConnection(_config()).float()).eval()
    streams, _ = _seeded_inputs()

    with torch.no_grad():
        ref_post, ref_comb, ref_collapsed = ref(streams.float())
        mine_post, mine_comb, mine_collapsed = tt_mhc_math(streams.float(), ref.fn, ref.base, ref.scale)

    # The TTNN path carries post as [B,S,1,H] (it feeds a matmul); the reference
    # returns [B,S,H]. Same numbers, different rank -- normalise before comparing.
    assert mine_post.shape == (1, 16, 1, HC), f"unexpected post rank: {mine_post.shape}"
    assert torch.allclose(mine_post.reshape(ref_post.shape), ref_post, atol=1e-6, rtol=1e-5), "post mismatch"
    assert torch.allclose(mine_comb, ref_comb, atol=1e-6, rtol=1e-5), "comb (Sinkhorn) mismatch"  # [B,S,H,H] both
    assert torch.allclose(mine_collapsed, ref_collapsed, atol=1e-4, rtol=1e-4), "collapsed mismatch"


def test_hyper_head_matches_reference():
    torch.manual_seed(0)
    ref = _init_head(DeepseekV4HyperHead(_config()).float()).eval()
    streams, _ = _seeded_inputs()

    with torch.no_grad():
        ref_out = ref(streams.float())
        flat = streams.float().reshape(1, 16, 1, HC * HIDDEN)
        mixes = unweighted_rmsnorm(flat, NORM_EPS) @ ref.hc_fn.T.float()
        pre = torch.sigmoid(mixes * ref.hc_scale.float() + ref.hc_base.float()) + HC_EPS
        mine = torch.matmul(pre.reshape(1, 16, 1, HC), streams.float().reshape(1, 16, HC, HIDDEN))

    assert torch.allclose(mine.reshape(ref_out.shape), ref_out, atol=1e-4, rtol=1e-4)


def test_sinkhorn_is_doubly_stochastic():
    """Post-condition of the projection: rows and columns both ~1 (up to eps)."""
    torch.manual_seed(0)
    ref = _init_mapping(DeepseekV4HyperConnection(_config()).float()).eval()
    streams, _ = _seeded_inputs()
    with torch.no_grad():
        _, _, ref_collapsed = ref(streams.float())
        _, comb, _ = tt_mhc_math(streams.float(), ref.fn, ref.base, ref.scale)
    assert torch.allclose(comb.sum(-1), torch.ones_like(comb.sum(-1)), atol=1e-4), "rows not ~1"
    assert torch.allclose(comb.sum(-2), torch.ones_like(comb.sum(-2)), atol=1e-4), "cols not ~1"


def test_residual_site_matches_reference_block_math():
    """The mix uses comb.transpose(-1,-2); a wrong transpose is silent, so pin it."""
    torch.manual_seed(0)
    ref = _init_mapping(DeepseekV4HyperConnection(_config()).float()).eval()
    streams, g = _seeded_inputs()
    sublayer_out = torch.randn(1, 16, HIDDEN, generator=g, dtype=torch.float32)

    with torch.no_grad():
        post, comb, _ = ref(streams.float())
        ref_new = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2), streams.float()
        )
        mine_new = tt_apply_site(post, comb, streams.float(), sublayer_out)

    assert torch.allclose(mine_new, ref_new, atol=1e-5, rtol=1e-5)

    # A non-symmetric comb means the transposed version must NOT match.
    untransposed = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2) + torch.matmul(comb, streams.float())
    assert not torch.allclose(untransposed, ref_new, atol=1e-3), "comb is symmetric; test lost its teeth"


def test_split_order_matters():
    """Swapping the pre/post slices must change the result (guards slice order)."""
    torch.manual_seed(0)
    ref = _init_mapping(DeepseekV4HyperConnection(_config()).float()).eval()
    streams, _ = _seeded_inputs()
    hc = HC
    with torch.no_grad():
        correct_post, _, _ = tt_mhc_math(streams.float(), ref.fn, ref.base, ref.scale)
        flat = streams.float().reshape(1, 16, 1, hc * HIDDEN)
        wide = unweighted_rmsnorm(flat, NORM_EPS) @ ref.fn.T.float()
        swapped_w = wide[..., hc : 2 * hc]  # post slice fed to the pre branch
        swapped_post = torch.sigmoid(swapped_w * ref.scale.unbind(0)[0] + ref.base.split([hc, hc, hc * hc])[0]) * 2.0
    assert not torch.allclose(correct_post, swapped_post, atol=1e-3)
