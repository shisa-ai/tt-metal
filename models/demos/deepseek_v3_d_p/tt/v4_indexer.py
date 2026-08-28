"""Lightning Indexer for V4-Flash compressed sparse attention.

Paper §2.3.1, eqs. 13-17 as implemented by ``DeepseekV4Indexer``: a scaled-down twin of the
CSA compressor runs over the same windows at ``index_head_dim`` (128), queries are projected
from the latent ``q_residual``, and each query keeps the top ``index_topk`` (512) compressed
entries by ``sum_h w[t,h] * ReLU(q[t,h] . K_s)``.

Two things are load-bearing and both are easy to get wrong:

* **The rope width is global to the rope group, not per module.** ``inv_freq`` is built from
  ``config.head_dim * partial_rotary_factor`` — 512 * 0.125 = 64 — so the indexer rotates
  trailing 64 of its 128-wide vectors, *not* ``int(index_head_dim * 0.125) = 16``. Deriving
  the width from the module's own head dim silently rotates a different subspace.
* **The indexer's two rope calls must share the compressor's theta.** Compressed keys are
  rotated at ``i * rate + first_window_position`` and queries at the live ``position_ids``,
  both from the *compress* group (YaRN, theta 160000). Mixed thetas leave a position-dependent
  skew in ``q . k``, which shows up as retrieval noise rather than an obvious error.

Host-only, like the rest of the V4 port: no device tensor is created here, and nothing in this
module has been executed on hardware.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.tt import v4_rope
from models.demos.deepseek_v3_d_p.tt.v4_compression import apply_v4_rope, compress_windows_csa


def index_scores(
    q: torch.Tensor, compressed_kv: torch.Tensor, weights: torch.Tensor, softmax_scale: float, weights_scaling: float
) -> torch.Tensor:
    """``[B, S, H, D]`` x ``[B, T, D]`` -> ``[B, S, T]`` retrieval scores.

    ``weights`` is already projected (``[B, S, H]``). ReLU before scaling matters: without it
    negative inner products would subtract and the ranking would change; the fp32 math is
    likewise part of the contract because the operands are bf16 in a real run.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be [B, S, H, D], got {tuple(q.shape)}")
    if compressed_kv.ndim != 3:
        raise ValueError(f"compressed_kv must be [B, T, D], got {tuple(compressed_kv.shape)}")
    scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = torch.relu(scores) * softmax_scale
    return (scores * (weights.float() * weights_scaling).unsqueeze(-1)).sum(dim=2)


def causal_entry_threshold(position_ids: torch.Tensor, compress_rate: int) -> torch.Tensor:
    """Query ``t`` may select entries ``< (t + 1) // rate``; the threshold itself is excluded.

    An entry at or past the threshold summarises tokens the query has not seen yet — with
    rate 4, query 2 cannot use the entry that compresses positions 12..16.
    """
    if position_ids.ndim != 2:
        raise ValueError(f"position_ids must be [B, S], got {tuple(position_ids.shape)}")
    if compress_rate <= 0:
        raise ValueError(f"compress_rate must be positive, got {compress_rate}")
    return (position_ids + 1) // compress_rate


def select_top_k(scores: torch.Tensor, position_ids: torch.Tensor, compress_rate: int, index_topk: int) -> torch.Tensor:
    """Top-``k`` entry indices per query, with ``-1`` for picks the causal rule forbids.

    ``k`` is clamped to the available entries, so early decode steps return fewer columns
    rather than padding. Entries past the threshold are masked to ``-inf`` before the ranking
    *and* re-checked afterwards: with too few ready entries, top-k has to return something,
    and that something must be discarded rather than attended.
    """
    if scores.ndim != 3:
        raise ValueError(f"scores must be [B, S, T], got {tuple(scores.shape)}")
    threshold = causal_entry_threshold(position_ids, compress_rate)
    compressed_len = scores.shape[-1]
    top_k = min(index_topk, compressed_len)
    if compressed_len == 0:
        return scores.new_zeros(scores.shape[:2] + (0,), dtype=torch.long)
    entries = torch.arange(compressed_len, device=scores.device)
    future = entries.view(1, 1, -1) >= threshold.unsqueeze(-1)
    masked = scores.masked_fill(future, float("-inf"))
    picked = masked.topk(top_k, dim=-1).indices
    invalid = picked >= threshold.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(picked, -1), picked)


class TtIndexer:
    """Port-side Lightning Indexer: index-side compressor, queries, scoring, top-k."""

    ROPE_GROUP = "compress"

    def __init__(self, cfg, weights: dict, *, max_windows: int = 8192):
        self.cfg = cfg
        self.w = weights
        self.compress_rate = int(cfg.compress_rates["compressed_sparse_attention"])
        self.num_heads = int(cfg.index_n_heads)
        self.head_dim = int(cfg.index_head_dim)
        self.index_topk = int(cfg.index_topk)
        self.q_lora_rank = int(cfg.q_lora_rank)
        self.eps = float(cfg.rms_norm_eps)
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scaling = self.num_heads**-0.5

        params = v4_rope.RopeParams.from_config(cfg, self.ROPE_GROUP)
        # Deliberately cfg.head_dim, not self.head_dim: the rope group defines the width.
        self.rope_dim = v4_rope.rope_dim(int(cfg.head_dim), params.partial_rotary_factor)
        if self.rope_dim > self.head_dim:
            raise ValueError(f"rope width {self.rope_dim} exceeds index head dim {self.head_dim}")
        inv_freq, attention_factor, self.rope_params = v4_rope.build_rope(cfg, self.ROPE_GROUP, int(cfg.head_dim))
        cos, sin = v4_rope.cos_sin_tables(inv_freq, attention_factor, max_windows * self.compress_rate, self.rope_dim)
        self._cos_by_offset, self._sin_by_offset = cos, sin
        self._inv_freq = inv_freq
        self._attention_factor = attention_factor

    # ------------------------------------------------------------------ rope

    def rope_for_positions(self, positions: torch.Tensor):
        """``(cos, sin)`` at arbitrary absolute positions, in the expanded layout.

        Keys use deterministic window slots, queries use the live ``position_ids``, so one
        table indexed by position serves both — but it must be the *same* table, which is how
        the shared theta is enforced structurally rather than by comment.
        """
        flat = positions.reshape(-1)
        if flat.numel() and int(flat.max()) >= self._cos_by_offset.shape[0]:
            raise ValueError(
                f"position {int(flat.max())} exceeds the precomputed table "
                f"({self._cos_by_offset.shape[0]} rows); raise max_windows at construction"
            )
        return self._cos_by_offset[flat].view(*positions.shape, self.rope_dim), self._sin_by_offset[flat].view(
            *positions.shape, self.rope_dim
        )

    def rope_for_entries(self, n_windows: int, first_window_position: int = 0):
        positions = (
            torch.arange(n_windows, device=self._cos_by_offset.device) * self.compress_rate + first_window_position
        )
        return self.rope_for_positions(positions.unsqueeze(0))

    # ------------------------------------------------------------- components

    def compress_keys(
        self, hidden_states: torch.Tensor, prior_kv=None, prior_gate=None, first_window_position: int = 0
    ) -> torch.Tensor:
        """Index-side compressed keys ``[B, T // rate, index_head_dim]``, roped."""
        kv = hidden_states @ self.w["kv_proj"].to(hidden_states.dtype).t()
        gate = hidden_states @ self.w["gate_proj"].to(hidden_states.dtype).t()
        entries = compress_windows_csa(
            kv,
            gate,
            self.w["position_bias"],
            self.w["kv_norm_weight"],
            self.eps,
            self.compress_rate,
            prior_kv,
            prior_gate,
        )
        cos, sin = self.rope_for_entries(entries.shape[1], first_window_position)
        return apply_v4_rope(entries.unsqueeze(1), cos, sin).squeeze(1)

    def queries(self, q_residual: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """``[B, S, H, index_head_dim]`` roped queries.

        The reference rotates in ``[B, H, S, D]`` layout and transposes afterwards. The layout
        is not cosmetic: ``apply_rotary_pos_emb`` unsqueezes its tables at ``dim=1``, which is
        the head axis in that order, so building ``[B, S, H, D]`` first and rotating there
        would broadcast the tables against the wrong axis.
        """
        batch, seq_len, _ = q_residual.shape
        cos_q, sin_q = self.rope_for_positions(position_ids)
        weight = self.w["q_b_proj"].to(q_residual.dtype)
        q = q_residual.reshape(-1, self.q_lora_rank) @ weight.t()
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        cos_q = cos_q.unsqueeze(1)
        sin_q = sin_q.unsqueeze(1)
        return apply_v4_rope(q, cos_q, sin_q).transpose(1, 2).contiguous()

    def score(self, q: torch.Tensor, compressed_kv: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = hidden_states @ self.w["weights_proj"].to(hidden_states.dtype).t()
        return index_scores(q, compressed_kv, weights, self.softmax_scale, self.weights_scaling)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_kv: torch.Tensor | None = None,
        prior_kv=None,
        prior_gate=None,
        first_window_position: int = 0,
    ):
        """Index selection for one forward, plus the scores and keys that produced it.

        ``compressed_kv`` may be supplied by the caller (the cache-owned, already-extended set
        of entries). When omitted, this call's own index keys are used, matching the
        reference's no-cache branch.
        """
        keys = self.compress_keys(hidden_states, prior_kv, prior_gate, first_window_position)
        if compressed_kv is None:
            compressed_kv = keys
        q = self.queries(q_residual, position_ids)
        scores = self.score(q, compressed_kv, hidden_states)
        return select_top_k(scores, position_ids, self.compress_rate, self.index_topk), scores, compressed_kv
