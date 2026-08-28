# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Manifold-Constrained Hyper-Connections (mHC) for DeepSeek-V4 (TTNN).

Ported-to-TTNN equivalent of ``DeepseekV4HyperConnection`` /
``DeepseekV4HyperHead`` in
``models/demos/deepseek_v3_d_p/reference/deepseek_v4/modeling_deepseek_v4.py``.
Written here (not ported from upstream) because no TT implementation exists on
either the pinned base or current upstream ``main``.

A V4 decoder block keeps ``hc_mult`` parallel residual streams in
``[B, S, hc_mult, D]`` instead of one residual. Each sublayer site (attention,
MLP) owns one mapping module that turns the current streams into three
quantities, then mixes the sublayer output back in:

    post    [B, S, H]        per-stream output placement, range [0, 2]
    comb    [B, S, H, H]     stream mixer, Sinkhorn-projected to doubly-stochastic
    collapsed [B, S, D]      the single sequence fed to the sublayer

    new_streams = post[..., None] * sublayer_out[..., None, :]
                  + matmul(comb.transpose(-1, -2), streams)

V4-Flash dims: ``hc_mult=4``, ``hc_sinkhorn_iters=20``, ``hc_eps=1e-6``,
``hidden_size=4096``, so ``fn`` is ``[(2+4)*4, 4*4096]`` per site.

Sharding
--------
Only the *mapping* needs the complete ``hc_mult*D`` activation, because ``fn``
mixes across the flattened stream axis. The collapse and the combine are local
in ``D``: ``collapsed`` is a weighted sum over the stream axis and the ``comb``
matmul contracts the stream axis only. So with ``hidden`` TP-sharded on ``D``,
the two per-site collectives can be dropped:

* mapping: feed a TP-complete (gathered or replicated) activation, or use a
  column-parallel ``fn`` (sharded on its input dim) plus an all-reduce;
* collapse / combine: run directly on the ``D``-sharded streams.

``TtHCMapping`` currently expects a TP-complete mapping input. The per-site
gather is the known cost of that choice -- two full-width collectives per layer
across 43 layers -- and is the obvious follow-up optimisation, not a
correctness issue.
"""

from __future__ import annotations

import torch

import ttnn


class _HCBase:
    """Shared pieces of the two mHC mappings.

    Both mappings start from the same unweighted RMSNorm over the flattened
    stream axis, so it lives here.
    """

    @staticmethod
    def unweighted_rmsnorm(x: ttnn.Tensor, eps: float, memory_config, compute_config) -> ttnn.Tensor:
        """``x * rsqrt(mean(x^2, -1) + eps)`` with **no** learned weight.

        Mirrors ``DeepseekV4UnweightedRMSNorm`` exactly: the mean is taken over
        the last dim in float, then applied to ``x``. The square is done as a
        multiply rather than ``ttnn.square`` so the operand dtype stays
        explicit.
        """
        sq = ttnn.multiply(x, x)
        mean = ttnn.mean(sq, dim=-1, keepdim=True, memory_config=memory_config)
        ttnn.deallocate(sq)
        denom = ttnn.rsqrt(ttnn.add(mean, eps, memory_config=memory_config), memory_config=memory_config)
        ttnn.deallocate(mean)
        out = ttnn.multiply(x, denom, memory_config=memory_config)
        ttnn.deallocate(denom)
        return out


class TtHCMapping(_HCBase):
    """One mHC mapping site (``attn_hc`` or ``ffn_hc``).

    Equivalent of ``DeepseekV4HyperConnection``. Holds ``fn`` ``[mix, H*D]``,
    ``base`` ``[mix]``, ``scale`` ``[3]`` where ``mix = (2 + H) * H``.
    """

    def __init__(
        self,
        device,
        hc_mult: int,
        hidden_size: int,
        sinkhorn_iters: int,
        hc_eps: float,
        norm_eps: float,
        weights: dict[str, torch.Tensor],
        compute_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hc_mult = int(hc_mult)
        self.hidden_size = int(hidden_size)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.hc_eps = float(hc_eps)
        self.norm_eps = float(norm_eps)
        self.memory_config = memory_config
        self.compute_config = compute_config

        hc = self.hc_mult
        mix = (2 + hc) * hc

        # Three separate weights rather than one [mix, H*D] weight plus a slice.
        # A single linear would produce 24 columns and splitting a *tiled* tensor
        # at 4/4/16 is not tile-aligned (TTNN tiles are 32 wide), so the slice
        # cannot be expressed. Splitting the weight on the host is mathematically
        # identical -- every output still contracts the full H*D -- and each
        # branch stays a legal tile shape. It costs two extra kernel launches.
        fn = weights["fn"].detach()
        parts = torch.split(fn, [hc, hc, hc * hc], dim=0)
        self.fn_pre, self.fn_post, self.fn_comb = (
            ttnn.from_torch(
                p.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0),
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for p in parts
        )
        # base / scale stay small and replicated; bias folds into the linear.
        self.base = weights["base"].detach().float().clone()
        self.scale = weights["scale"].detach().float().clone()

    def mapping(self, flat_norm: ttnn.Tensor):
        """Return ``(pre_logits, post_logits, comb_logits)`` from the normed flat.

        Three linears on the pre-split weights, so nothing has to slice a tiled
        tensor at a non-tile boundary. ``comb_logits`` comes back as
        ``[B, S, H, H]`` because the reference adds its bias only after viewing
        the logits as ``[..., hc, hc]``.
        """
        hc = self.hc_mult
        # ttnn Shape supports integer indexing but NOT slicing, so read the
        # extents by index rather than slicing the shape tuple.
        b, s = flat_norm.shape[0], flat_norm.shape[1]
        pre_w = ttnn.linear(
            flat_norm, self.fn_pre, compute_kernel_config=self.compute_config, memory_config=self.memory_config
        )
        post_w = ttnn.linear(
            flat_norm, self.fn_post, compute_kernel_config=self.compute_config, memory_config=self.memory_config
        )
        comb_w = ttnn.linear(
            flat_norm, self.fn_comb, compute_kernel_config=self.compute_config, memory_config=self.memory_config
        )

        # Each of the three outputs has its own learned scalar, then its own bias
        # slice -- reference applies `* scale` before `+ base`.
        pre_s, post_s, comb_s = self.scale.unbind(0)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])

        pre_logits = ttnn.add(
            ttnn.multiply(pre_w, float(pre_s), memory_config=self.memory_config),
            _as_bias(pre_b, self.device, self.memory_config),
            memory_config=self.memory_config,
        )
        post_logits = ttnn.add(
            ttnn.multiply(post_w, float(post_s), memory_config=self.memory_config),
            _as_bias(post_b, self.device, self.memory_config),
            memory_config=self.memory_config,
        )
        # Reference order: view as [..., hc, hc] -> scale -> bias. The reshape
        # must keep batch and sequence (volume b*s*H*H), so it is [b,s,H,H].
        comb_logits = ttnn.add(
            ttnn.multiply(
                ttnn.reshape(comb_w, [b, s, hc, hc]),
                float(comb_s),
                memory_config=self.memory_config,
            ),
            _as_bias(comb_b.reshape(hc, hc), self.device, self.memory_config),
            memory_config=self.memory_config,
        )
        return pre_logits, post_logits, comb_logits

    def forward(self, hidden_streams: ttnn.Tensor):
        """`(post, comb, collapsed)` from streams ``[B, S, H, D]`` (TP-complete).

        The reference computes the mapping in fp32 and only casts back to the
        activation dtype when mixing; we keep the mapping in fp32-ish
        (``bfloat16`` + fp32 dest accumulation) and return the collapsed tensor
        in the input layout.
        """
        hc = self.hc_mult
        batch, seq, _, dim = hidden_streams.shape

        # flatten(2): [B, S, H*D]
        flat = ttnn.reshape(hidden_streams, [batch, seq, 1, hc * dim])
        flat_norm = self.unweighted_rmsnorm(flat, self.norm_eps, self.memory_config, self.compute_config)
        ttnn.deallocate(flat)

        pre_logits, post_logits, comb_logits = self.mapping(flat_norm)
        ttnn.deallocate(flat_norm)

        eps = self.hc_eps

        # pre  = sigmoid(logits) + eps
        pre = ttnn.add(
            ttnn.sigmoid(pre_logits, memory_config=self.memory_config), eps, memory_config=self.memory_config
        )

        # post = 2 * sigmoid(logits)   (note: no eps on this branch)
        post = ttnn.multiply(
            ttnn.sigmoid(post_logits, memory_config=self.memory_config), 2.0, memory_config=self.memory_config
        )

        # comb = softmax(-1) + eps, then Sinkhorn-Knopp row/col normalisation.
        comb = ttnn.add(
            ttnn.softmax(comb_logits, dim=-1, memory_config=self.memory_config), eps, memory_config=self.memory_config
        )
        comb = self.sinkhorn(comb)

        # collapsed = sum_h pre[h] * streams[h]  -> [B, S, 1, D]
        collapsed = ttnn.matmul(
            ttnn.reshape(pre, [batch, seq, 1, hc]),
            ttnn.reshape(hidden_streams, [batch, seq, hc, dim]),
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_config,
        )
        return post, comb, collapsed, hidden_streams

    def sinkhorn(self, comb: ttnn.Tensor) -> ttnn.Tensor:
        """Sinkhorn-Knopp: alternate column then row normalisation.

        Order matters and is taken verbatim from the reference: one column
        normalisation first, then ``iters - 1`` pairs of (row, column). Every
        division carries ``+ eps``.
        """
        comb = _normalize_axis(comb, axis=-2, eps=self.hc_eps, device=self.device, mc=self.memory_config)
        for _ in range(self.sinkhorn_iters - 1):
            comb = _normalize_axis(comb, axis=-1, eps=self.hc_eps, device=self.device, mc=self.memory_config)
            comb = _normalize_axis(comb, axis=-2, eps=self.hc_eps, device=self.device, mc=self.memory_config)
        return comb


class TtHyperHead(_HCBase):
    """Final stream collapse before the shared RMSNorm (``DeepseekV4HyperHead``)."""

    def __init__(
        self,
        device,
        hc_mult: int,
        hidden_size: int,
        hc_eps: float,
        norm_eps: float,
        weights: dict[str, torch.Tensor],
        compute_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hc_mult = int(hc_mult)
        self.hidden_size = int(hidden_size)
        self.hc_eps = float(hc_eps)
        self.norm_eps = float(norm_eps)
        self.memory_config = memory_config
        self.compute_config = compute_config

        self.fn = ttnn.from_torch(
            weights["hc_fn"].detach().transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.base = weights["hc_base"].detach().float().clone()
        self.scale = float(weights["hc_scale"].detach().float().reshape(-1)[0])

    def forward(self, streams: ttnn.Tensor) -> ttnn.Tensor:
        hc = self.hc_mult
        batch, seq, _, dim = streams.shape
        flat = ttnn.reshape(streams, [batch, seq, 1, hc * dim])
        flat_norm = self.unweighted_rmsnorm(flat, self.norm_eps, self.memory_config, self.compute_config)
        ttnn.deallocate(flat)

        mixes = ttnn.linear(flat_norm, self.fn, compute_kernel_config=self.compute_config, memory_config=self.memory_config)
        ttnn.deallocate(flat_norm)

        pre = ttnn.add(
            ttnn.sigmoid(
                ttnn.add(
                    ttnn.multiply(mixes, self.scale, memory_config=self.memory_config),
                    _as_bias(self.base, self.device, self.memory_config),
                    memory_config=self.memory_config,
                ),
                memory_config=self.memory_config,
            ),
            self.hc_eps,
            memory_config=self.memory_config,
        )
        ttnn.deallocate(mixes)
        return ttnn.matmul(
            ttnn.reshape(pre, [batch, seq, 1, hc]),
            ttnn.reshape(streams, [batch, seq, hc, dim]),
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_config,
        )


def apply_mhc_site(
    post: ttnn.Tensor,
    comb: ttnn.Tensor,
    streams: ttnn.Tensor,
    sublayer_out: ttnn.Tensor,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    compute_config=None,
) -> ttnn.Tensor:
    """Mix one sublayer output back into the stream stack.

        new_streams = post.unsqueeze(-1) * out.unsqueeze(-2)
                      + matmul(comb.transpose(-1, -2), streams)

    ``comb`` is consumed **transposed** -- the reference sums over the *first*
    stream axis (``sum_j comb[j, k] * streams[j]``). Sinkhorn yields a
    doubly-stochastic but generally asymmetric matrix, so getting this
    transpose wrong is silent and changes the model.
    """
    batch, seq, hc, dim = streams.shape

    # post: [B, S, H] -> [B, S, H, 1]; out: [B, S, D] -> [B, S, 1, D]
    post4 = ttnn.reshape(post, [batch, seq, 1, hc])
    post4 = ttnn.transpose(post4, 2, 3)
    out4 = ttnn.reshape(sublayer_out, [batch, seq, 1, dim])

    placed = ttnn.matmul(post4, out4, memory_config=memory_config, compute_kernel_config=compute_config)
    ttnn.deallocate(post4)
    ttnn.deallocate(out4)

    mixed = ttnn.matmul(
        ttnn.transpose(comb, 2, 3),
        ttnn.reshape(streams, [batch, seq, hc, dim]),
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    return ttnn.add(placed, mixed, memory_config=memory_config)


def _normalize_axis(t: ttnn.Tensor, axis: int, eps: float, device, mc):
    """``t / (sum(t, axis, keepdim=True) + eps)``.

    Uses a real ``divide`` rather than ``multiply(t, reciprocal(total))``. The
    reciprocal form measured worse on device: Sinkhorn applies this 40 times per
    token (column, then 19 x (row, column)) and the two-roundings-per-step error
    compounds across all 40.
    """
    total = ttnn.sum(t, dim=axis, keepdim=True, memory_config=mc)
    return ttnn.divide(t, ttnn.add(total, eps, memory_config=mc), memory_config=mc)


def _as_bias(vec: torch.Tensor, device, mc) -> ttnn.Tensor:
    """A small parameter broadcast against a ``[..., n]`` activation."""
    shape = [1, 1, 1, -1] if vec.ndim == 1 else [1, 1, *vec.shape[:-1], vec.shape[-1]]
    return ttnn.from_torch(
        vec.reshape(shape).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
