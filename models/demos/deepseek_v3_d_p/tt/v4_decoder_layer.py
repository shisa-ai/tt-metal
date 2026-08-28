# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""One V4-Flash decoder block, as glue.

Reference: ``DeepseekV4DecoderLayer.forward`` in models/demos/deepseek_v3_d_p/
reference/deepseek_v4/modeling_deepseek_v4.py.

The block is *not* ``norm -> sublayer -> residual``. It threads a stack of
``hc_mult`` residual streams and mixes each sublayer back in through an
Sinkhorn-projected hyper-connection:

    post, comb, collapsed = attn_hc(streams)          # streams: [B,S,H,D]
    attn_out = attn(input_layernorm(collapsed))       # collapsed: [B,S,1,D]
    streams  = post[...,None]*attn_out[...,None,:] + comb^T @ streams

    post, comb, collapsed = ffn_hc(streams)
    mlp_out  = mlp(post_attention_layernorm(collapsed))
    streams  = post[...,None]*mlp_out[...,None,:] + comb^T @ streams

Both the stream-carrying and the ``comb`` transpose are load-bearing and silent
bugs; neither is exercised here, both are covered on host by
``tests/pcc/test_v4_decoder_layer_math.py``, which compares this transcription
against the reference block's own ``forward`` and asserts that the un-transposed
and collapsed-carry variants genuinely differ.

Why the sublayers are injected
------------------------------

``attn_fn`` / ``mlp_fn`` are callables, not constructed here. ``TtHCA`` and
``TtMoe`` take long, unverified constructor signatures (mesh/fabric config, weight
cache paths, dtypes per tensor group, CCL handles). Inventing those arguments here
would produce a file that looks finished and cannot run; they belong to the
assembly (#65), which is where the real construction happens against real APIs.
This module's job is only the composition, and keeping it injectable is what makes
that job reviewable.

Layout note
-----------

``TtHCMapping.forward`` yields ``collapsed`` as ``[B, S, 1, D]`` while ``TtHCA``
consumes ``[B, 1, S, D]``. The swap moves a singleton against ``S`` only, so the
element order is unchanged and the reshape is a view, not a transpose — and
``apply_mhc_site`` reshapes the sublayer output to ``[B, S, 1, D]`` internally, so
``[B, 1, S, D]`` can be handed to it directly. If a future change makes either
dim non-singleton, these reshapes stop being free and would silently scramble
order; ``_to_attn_layout`` is the single place to revisit.

Precision
---------

The reference declares ``input_layernorm``, ``post_attention_layernorm``,
``attn_hc``, ``ffn_hc``, ``sinks`` and ``e_score_correction_bias`` fp32-strict. The
norms here therefore default to ``ttnn.float32`` and take their weight in fp32.
Where a kernel cannot run fp32 on Wormhole the deviation must be declared at
construction and recorded, never discovered as an unexplained parity gap -- pass
``dtype`` explicitly and say so in the run record.
"""

from __future__ import annotations

from typing import Callable

import ttnn

from models.demos.deepseek_v3_d_p.tt.mhc import apply_mhc_site


def _to_attn_layout(x: ttnn.Tensor) -> ttnn.Tensor:
    """``[B, S, 1, D]`` -> ``[B, 1, S, D]``. Singleton/S swap, order preserved."""
    b, s, one, d = x.shape
    assert one == 1, f"expected a singleton stream dim, got shape {x.shape}"
    return ttnn.reshape(x, [b, 1, s, d])


class TtV4DecoderLayer:
    """Composes two mHC sites, two weighted RMSNorms, and two injected sublayers.

    Parameters
    ----------
    attn_hc, ffn_hc:
        :class:`~models.demos.deepseek_v3_d_p.tt.mhc.TtHCMapping` instances, one
        per sublayer (the reference names them ``attn_hc`` / ``ffn_hc``).
    attn_fn, mlp_fn:
        ``[B, 1, S, D] -> [B, 1, S, D]``. Stateful sublayers (KV/HCA state) close
        over their own state in the assembly.
    norm_in_weight, norm_out_weight:
        ``[1, 1, 1, D]`` fp32 tile tensors for the pre-attention and pre-MLP
        ``DeepseekV4RMSNorm``.
    rms_eps:
        ``config.rms_norm_eps``.
    dtype:
        Activation dtype for the norms. Defaults to fp32 because the reference
        pins these modules fp32-strict.
    """

    def __init__(
        self,
        *,
        attn_hc,
        ffn_hc,
        attn_fn: Callable[[ttnn.Tensor], ttnn.Tensor],
        mlp_fn: Callable[[ttnn.Tensor], ttnn.Tensor],
        norm_in_weight: ttnn.Tensor,
        norm_out_weight: ttnn.Tensor,
        rms_eps: float,
        dtype: ttnn.DataType = ttnn.float32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=None,
        name: str = "layer",
    ):
        self.attn_hc = attn_hc
        self.ffn_hc = ffn_hc
        self.attn_fn = attn_fn
        self.mlp_fn = mlp_fn
        self.norm_in_weight = norm_in_weight
        self.norm_out_weight = norm_out_weight
        self.rms_eps = float(rms_eps)
        self.dtype = dtype
        self.memory_config = memory_config
        self.compute_kernel_config = compute_kernel_config
        self.name = name

    def _norm(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.rms_eps,
            weight=weight,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _site(self, hc, streams: ttnn.Tensor, weight: ttnn.Tensor, sublayer) -> ttnn.Tensor:
        post, comb, collapsed, incoming = hc.forward(streams)
        normed = self._norm(collapsed, weight)
        out = sublayer(_to_attn_layout(normed))
        # `comb` is applied transposed inside apply_mhc_site; passing the *mixed*
        # stream (not hc's pass-through of `incoming`) is what makes the second
        # site see the first site's contribution.
        return apply_mhc_site(
            post,
            comb,
            streams,
            out,
            memory_config=self.memory_config,
            compute_config=self.compute_kernel_config,
        )

    def forward(self, streams: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, S, hc_mult, D]`` in and out.

        The returned tensor is the next block's stream stack, not a hidden state:
        collapsing it to ``[B, S, D]`` between blocks is the wrong-port failure the
        host tests assert against.
        """
        if streams.dim() != 4:
            raise ValueError(f"{self.name}: decoder block carries [B,S,hc_mult,D], got {tuple(streams.shape)}")
        streams = self._site(self.attn_hc, streams, self.norm_in_weight, self.attn_fn)
        return self._site(self.ffn_hc, streams, self.norm_out_weight, self.mlp_fn)
