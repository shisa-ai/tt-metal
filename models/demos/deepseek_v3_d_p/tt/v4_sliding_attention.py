# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""V4 sliding attention (``layer_types[i] == "sliding_attention"``) on TTNN.

Covers the pure-sliding layers: no compressor, shared-KV MQA, partial interleaved rope
on the trailing slice, learnable per-head sink, and the grouped output projection.
HCA/CSA layers use ``TtHCA`` in ``tt.mla`` instead.

Everything load-bearing here is pinned by a host test rather than assumed, because the
reference cannot be exercised at 1×1 by upstream and this pin's V4 demo is disabled:

* **Decode contract** (``tests/pcc/test_v4_attention_decode_contract.py``): a
  single-token step against a cache reproduces the prefill row to <1e-5, the sliding
  cache holds exactly ``window-1`` past keys, and because every retained key is
  already in-window, **the decode step needs no mask.**
* **Rope convention** (``test_v4_rope_layout_math.py``): ``x @ get_rot_transformation_mat()``
  *is* the reference's interleaved ``rotate_half``, applied to the **trailing** slice;
  the half-split convention differs observably, so it cannot be substituted silently.
* **Sink + K==V + grouped-O** (``test_v4_attn_epilogue_math.py``): ``s_aux`` is a true
  softmax denominator extra; K and V are the same tensor; the grouped output projection
  is reproduced here by per-group matmuls (see ``grouped_linear_torch``).

Two deliberate design decisions, each with a named risk:

1. **``is_decode_mode=False`` even at ``seq_len == 1``.** TTNN's decode-mode rotary
   comes from half-split-ropo models (llama/gemma), while V4 is interleaved; upstream's
   decode matrix is the *same* tile re-sharded per batch, so the flag changes layout
   handling, not the rotation pattern. Padded prefill-mode preserves the convention the
   host tests verified. Risk: the op may reject ``S == 1`` under its sharding, in which
   case queries are padded to a tile and the extra rows sliced away — the mask
   neutralises pad columns on the KV side already (the HCA module does the same).
   Smallest falsifying probe: one ``rotary_embedding_llama`` call with ``S=1`` padded to
   32, compared against the host transcription.
2. **Grouped output projection as ``o_groups`` matmuls plus adds**, because this demo
   tree contains **no** ``ttnn.bmm`` precedent for a batched grouped GEMM. The
   decomposition is verified against ``DeepseekV4GroupedLinear`` on host. Cost:
   ``o_groups`` (8) matmuls instead of one; revisit with bmm once the device is usable.

Not yet validated on hardware — the tray is quarantined (worklog dbf3d9). Nothing here
should be read as a device result.
"""

from __future__ import annotations

import torch
import ttnn

from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.tt_ccl import TT_CCL
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm


def _to_device(tensor: torch.Tensor, device, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
    """Ship a host tensor to the device.

    Written against ``ttnn.from_torch`` directly rather than a repo helper: on this pin
    ``torch_to_ttnn`` exists only under ``models/experimental/pi0``, so importing it
    here would couple production code to an experimental path.
    """
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )


def grouped_linear_torch(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """``DeepseekV4GroupedLinear`` as per-group matmuls — the device decomposition.

    ``x``: ``[..., groups, in_per_group]``; ``weight``: ``[groups, in_per_group, out_per_group]``.
    Returns ``[..., groups * out_per_group]``.

    Reference semantics (verified host-side): each group's slice multiplies only its own
    weight block, and the group results are concatenated along the feature axis — not
    summed. Summing would be a plain dense matmul over the concatenation and would mix
    groups, which is exactly what the grouping exists to prevent.
    """
    # Each group's slice multiplies only its own weight block; results are concatenated
    # along features in group order (concat, not sum — summing would mix groups, which
    # is exactly what the grouping exists to prevent).
    return torch.cat([torch.matmul(x[..., g, :], weight[g]) for g in range(groups)], dim=-1)


def build_transformation_mat(rope_dim: int) -> torch.Tensor:
    """Block-diagonal rotation matrix for a rope slice wider than one tile.

    ``get_rot_transformation_mat()`` returns a single ``32x32`` tile. V4-Flash's rope
    slice is 64 dims (``partial_rotary_factor`` 0.125 of ``head_dim`` 512), so the
    tile repeats twice on the diagonal — the interleaved pair pattern is tile-local.
    A slice that is not a whole number of tiles cannot be expressed this way, and
    failing loudly beats silently rotating half the pairs.
    """
    tile = ttnn.TILE_SIZE
    if rope_dim % tile:
        raise ValueError(
            f"rope_dim {rope_dim} is not a multiple of TILE_SIZE {tile}; the tile-local "
            "interleaved pattern cannot cover it — the rope layout assumption needs revisiting"
        )
    base = get_rot_transformation_mat()  # [1, 1, 32, 32]
    tile = base.squeeze(0).squeeze(0)
    out = torch.zeros(rope_dim, rope_dim, dtype=tile.dtype)
    for i in range(rope_dim // ttnn.TILE_SIZE):
        s = i * ttnn.TILE_SIZE
        out[s : s + ttnn.TILE_SIZE, s : s + ttnn.TILE_SIZE] = tile
    return out


def sliding_causal_mask(seq_len: int, window: int, dtype=torch.float32) -> torch.Tensor:
    """Additive causal+window mask ``[1, 1, seq, seq]`` for a fresh prefill.

    ``allowed = (k <= q) & (q - k < window)`` — the convention measured against the
    reference, where a token at distance exactly ``window`` is inert and the cache keeps
    ``window-1`` keys. Decode passes **no** mask: the cache is already window-trimmed,
    so every retained key is in-window (measured; asserted in the contract test).
    """
    q = torch.arange(seq_len).unsqueeze(1)
    k = torch.arange(seq_len).unsqueeze(0)
    allowed = (k <= q) & ((q - k) < window)
    neg = torch.finfo(dtype).min
    return torch.where(allowed, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype)).view(
        1, 1, seq_len, seq_len
    )


class TtV4SlidingAttention:
    """Sliding-window attention for one V4 layer, batch 1.

    Weight dict keys mirror the reference module, so a checkpoint adapter can hand them
    over directly:

        q_a_proj.weight, q_a_norm.weight, q_b_proj.weight,
        kv_proj.weight, kv_norm.weight,
        o_a_proj.weight [groups*out_per_group, in_per_group] (a 2D nn.Linear weight
            viewed per group), o_b_proj.weight,
        sinks [num_heads]

    ``q_b_norm`` is the reference's **unweighted** RMSNorm, so it is built with
    ``torch_weight=None``; giving it a weight would silently change the numerics.
    """

    def __init__(
        self,
        mesh_device,
        cfg,
        weights: dict,
        *,
        tt_ccl: TT_CCL | None = None,
        rope_theta: float = 10000.0,
        max_seq_len: int = 4096,
    ):
        self.device = mesh_device
        self.cfg = cfg
        self.w = weights
        self.tt_ccl = tt_ccl

        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        rope_factor = cfg.rope_parameters["main"]["partial_rotary_factor"]
        self.rope_theta = cfg.rope_parameters["main"]["rope_theta"]
        self.rope_dim = int(round(rope_factor * cfg.head_dim))
        if self.rope_dim % ttnn.TILE_SIZE:
            raise ValueError(
                f"rope_dim {self.rope_dim} not a multiple of TILE_SIZE; see build_transformation_mat"
            )
        self.nope_dim = cfg.head_dim - self.rope_dim
        self.window = cfg.sliding_window
        self.scaling = self.head_dim**-0.5
        self.o_groups = cfg.o_groups
        self.max_seq_len = max_seq_len

        # Cache depth is window-1 past keys, the depth the reference maintains (measured).
        self.cache_len = self.window - 1  # depth the reference actually maintains
        self._build_host_tables()

    # ---------------------------------------------------------------- host tables

    def _build_host_tables(self) -> None:
        """Everything buildable without a device, built once and never in forward."""
        inv = self.rope_theta ** (
            -torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim
        )
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = positions.unsqueeze(-1) * inv  # [max_seq, rope_dim/2]
        # Interleaved: one entry per pair, expanded by repeat_interleave(2) at use time,
        # matching the reference's apply_rotary_pos_emb.
        self._cos_full = freqs.cos().repeat_interleave(2, dim=-1)  # [max_seq, rope_dim]
        self._sin_full = freqs.sin().repeat_interleave(2, dim=-1)
        self._trans_torch = build_transformation_mat(self.rope_dim)

    # ---------------------------------------------------------------- construction

    def create_configs(self) -> None:
        """Device tensors and submodules. Call before forward; builds nothing per-call."""
        dev = self.device
        self.trans_mat = _to_device(self._trans_torch, dev)
        self.cos = _to_device(self._cos_full.unsqueeze(0).unsqueeze(0), dev, ttnn.bfloat32)
        self.sin = _to_device(self._sin_full.unsqueeze(0).unsqueeze(0), dev, ttnn.bfloat32)

        # Call convention copied from tests/pcc/test_rmsnorm.py and
        # tests/cache/test_rms_norm_cache.py, which pass torch_weight=None for the
        # unweighted case — so the reference's UnweightedRMSNorm has in-tree precedent
        # rather than one invented here.
        self.q_a_norm = TtDistributedRmsNorm(
            dev, self.cfg.q_lora_rank, torch_weight=self.w["q_a_norm.weight"], weight_cache_path=None
        )
        # The reference applies q_b_norm per HEAD over head_dim (it runs on [B, H, S, D]),
        # so emb_dim is head_dim and the wide tensor must be folded to [1, S*H, D] first.
        # Normalising over H*D instead would be a silent numerics change that only shows
        # up as an unexplained parity gap.
        self.q_b_norm = TtDistributedRmsNorm(
            dev, self.head_dim, torch_weight=None, weight_cache_path=None
        )
        self.kv_norm = TtDistributedRmsNorm(
            dev, self.head_dim, torch_weight=self.w["kv_norm.weight"], weight_cache_path=None
        )
        self.w_q_a = _to_device(self.w["q_a_proj.weight"].t(), dev)
        self.w_q_b = _to_device(self.w["q_b_proj.weight"].t(), dev)
        self.w_kv = _to_device(self.w["kv_proj.weight"].t(), dev)
        self.w_o_b = _to_device(self.w["o_b_proj.weight"].t(), dev)
        # DeepseekV4GroupedLinear extends nn.Linear, so its weight is 2D:
        # [groups*out_per_group, in_per_group]. The per-group block is a VIEW, and
        # indexing weight[g] would take one output row instead of a whole group.
        o_a = self.w["o_a_proj.weight"]
        out_per_group = o_a.shape[0] // self.o_groups
        self.w_o_a = [
            _to_device(o_a.view(self.o_groups, out_per_group, o_a.shape[1])[g].t().contiguous(), dev)
            for g in range(self.o_groups)
        ]
        self.sinks = _to_device(self.w["sinks"].reshape(1, 1, 1, self.num_heads).float(), dev, ttnn.bfloat32)
        # Rolling sliding cache: [1, 1, cache_len, head_dim]; K and V are the same tensor.
        self.cache = ttnn.zeros(
            [1, 1, self.cache_len, self.head_dim],
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

    # ---------------------------------------------------------------- forward

    def forward(self, x: ttnn.Tensor, position_start: int):
        """``x``: ``[1, seq, hidden]``. Returns ``[1, seq, hidden]``.

        ``position_start`` is the absolute position of ``x[:, 0]``; rope must use the
        absolute position, not the slice index, or chunked prefill and decode disagree.
        """
        seq = x.shape[1]
        hidden_shape = (1, seq, -1, self.head_dim)

        q = ttnn.matmul(x, self.w_q_a)
        q = self.q_a_norm(q)
        q = ttnn.matmul(q, self.w_q_b)  # [1, S, H*D]
        q = ttnn.reshape(q, (1, seq * self.num_heads, self.head_dim))  # fold heads into rows
        q = self.q_b_norm(q)  # per-head RMS over D, matching the reference
        q = ttnn.reshape(q, (*hidden_shape,))
        q = ttnn.transpose(q, 1, 2)  # [1, H, S, D]

        kv = ttnn.matmul(x, self.w_kv)
        kv = self.kv_norm(kv)
        kv = ttnn.reshape(kv, (1, seq, 1, self.head_dim))
        kv = ttnn.transpose(kv, 1, 2)  # [1, 1, S, D] — single shared head, K == V

        cos = ttnn.slice(self.cos, [0, 0, position_start, 0], [1, 1, position_start + seq, self.rope_dim])
        sin = ttnn.slice(self.sin, [0, 0, position_start, 0], [1, 1, position_start + seq, self.rope_dim])

        q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self.trans_mat, is_decode_mode=False)
        kv = ttnn.experimental.rotary_embedding_llama(kv, cos, sin, self.trans_mat, is_decode_mode=False)

        # Read the cache, concat, then write back the last cache_len keys. Order matters:
        # the new keys must not be trimmed before they can be attended to.
        keys = ttnn.concat([self.cache, kv], dim=2) if self.cache_len else kv
        if self.cache_len:
            keep_from = keys.shape[2] - self.cache_len
            self.cache = ttnn.slice(keys, [0, 0, keep_from, 0], [1, 1, keys.shape[2], self.head_dim])

        mask = None if seq == 1 else sliding_causal_mask(keys.shape[2], self.window)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            keys,
            keys,  # K == V in V4
            attn_mask=mask,
            is_causal=False,
            scale=self.scaling,
            attention_sink=self.sinks,
            program_config=self._sdpa_program_config,
        )

        # Undo V's rope on the rope slice (K==V means V carried it), at the QUERY
        # position, with sin negated. Layout fix-up only: the op wants [B, S, H, D].
        attn = ttnn.transpose(attn, 1, 2)  # [1, S, H, D]
        shape = list(attn.shape)
        shape[-1] = self.nope_dim
        nope = ttnn.slice(attn, [0, 0, 0, 0], [1, seq, self.num_heads, self.nope_dim])
        rope = ttnn.slice(attn, [0, 0, 0, self.nope_dim], [1, seq, self.num_heads, self.rope_dim])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, ttnn.neg(sin), self.trans_mat, is_decode_mode=False)
        attn = ttnn.concat([nope, rope], dim=-1)

        # Grouped output projection: per-group matmul then concat (verified decomposition;
        # no ttnn.bmm precedent in this demo tree).
        attn = ttnn.reshape(attn, (1, seq, 1, self.num_heads * self.head_dim))
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        parts = []
        for g in range(self.o_groups):
            lo, hi = g * in_per_group, (g + 1) * in_per_group
            piece = ttnn.reshape(attn, (1, seq, 1, self.num_heads * self.head_dim))
            piece = ttnn.slice(piece, [0, 0, 0, lo], [1, seq, 1, hi])
            parts.append(ttnn.matmul(piece, self.w_o_a[g]))
        grouped = ttnn.concat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return ttnn.matmul(grouped, self.w_o_b)

    def reset_cache(self) -> None:
        """Zero the rolling cache. Must be called between sequences: a stale window from
        the previous sequence is inside the current window and would be attended to."""
        ttnn.deallocate(self.cache)
        self.cache = ttnn.zeros(
            [1, 1, self.cache_len, self.head_dim],
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
