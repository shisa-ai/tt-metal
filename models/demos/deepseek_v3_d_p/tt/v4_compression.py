"""Heavily-Compressed Attention (HCA) compressor for V4-Flash.

Paper §2.3.2, eqs. 20-23 as implemented by ``DeepseekV4HCACompressor``: every
``compress_rate`` (128 for V4-Flash) source tokens collapse into **one** compressed KV
entry,

    C_comp[i] = RMSNorm( sum_{j in window_i} softmax_j(Z_j + B) * C_j )

where ``Z`` is a gate projection, ``B`` is a learned per-position bias inside the window,
and the softmax runs over the *window axis* (dim 2 of ``[B, n_windows, rate, head_dim]``)
in float32 before being cast back — the dtype and the axis are both load-bearing.

Two details decide correctness and are easy to lose:

* **RoPE position of a compressed entry is not a token position.** Entry ``i`` is rotated
  at absolute position ``i * compress_rate + first_window_position``, with the **compress**
  rope group (YaRN at theta 160000), not the main group and not the tokens' own positions.
  See :mod:`tt.v4_rope` for why the group matters. ``first_window_position`` comes from the
  cache, which is what keeps cross-call concatenation causality-correct during decode.
* **The causal rule is on compressed entries, not tokens.** Query token ``t`` may attend
  entry ``w`` iff ``w < (t + 1) // compress_rate``. With rate 128 a token at position 7 sees
  no compressed entries at all; a token at 200 sees entry 1 but not entry 2.

Host-only module. ``__init__`` touches no device and precomputes everything that is
deterministic, so the layout, causal mask and rope tables are testable on a machine with no
accelerator. The device decomposition is named in :meth:`create_configs`; it is **not
executed anywhere yet**, so nothing here is evidence that HCA runs on hardware.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.tt import v4_rope


def window_causal_bias(
    position_ids: torch.Tensor,
    compressed_len: int,
    compress_rate: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Additive mask ``[B, 1, S, compressed_len]`` over compressed entries.

    Entry ``w`` is visible to query ``t`` iff ``w < (t + 1) // rate``; everything else is
    ``-inf``. The strict ``<`` is what the reference does: the window containing token ``t``
    is not yet closed, so it must not be visible.
    """
    if position_ids.ndim != 2:
        raise ValueError(f"position_ids must be [B, S], got {tuple(position_ids.shape)}")
    if compress_rate <= 0:
        raise ValueError(f"compress_rate must be positive, got {compress_rate}")
    batch, seq = position_ids.shape
    entries = torch.arange(compressed_len, device=position_ids.device)
    threshold = (position_ids + 1) // compress_rate  # [B, S]
    bias = torch.zeros((batch, 1, seq, compressed_len), dtype=dtype, device=position_ids.device)
    return bias.masked_fill(entries.view(1, 1, 1, -1) >= threshold.unsqueeze(1).unsqueeze(-1), float("-inf"))


def compressed_positions(n_windows: int, compress_rate: int, first_window_position: int, device=None) -> torch.Tensor:
    """Absolute rope position of each compressed entry: ``i * rate + first``.

    A row vector ``[1, n_windows]`` broadcast over batch. Using the token positions here
    instead would rotate entries by their *contents'* positions rather than the window's
    canonical slot, which breaks cross-call concatenation once decode resumes.
    """
    if n_windows < 0:
        raise ValueError(f"n_windows must be non-negative, got {n_windows}")
    positions = torch.arange(n_windows, device=device) * compress_rate + first_window_position
    return positions.unsqueeze(0)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm as the reference applies it: variance over the last axis, fp32 math."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def compress_windows(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    compress_rate: int,
) -> torch.Tensor:
    """Collapse complete windows: ``[B, T, D]`` -> ``[B, T // rate, D]``.

    ``T`` need not be a multiple of ``rate``; the remainder is dropped exactly as the
    reference's stateless single-shot mode does. In cached decode the remainder is instead
    carried in the cache, which is the caller's job, not this function's.

    The softmax is over the **window** axis in float32 and then cast back to the operand
    dtype: softening in the operand dtype would change the weights, and softmaxing over
    ``head_dim`` would produce an entirely different (and much more wrong) object.
    """
    if kv.ndim != 3 or gate.shape != kv.shape:
        raise ValueError(f"kv and gate must both be [B, T, D]; got {tuple(kv.shape)} " f"and {tuple(gate.shape)}")
    if position_bias.shape != (compress_rate, kv.shape[-1]):
        raise ValueError(
            f"position_bias must be [rate, head_dim] = [{compress_rate}, {kv.shape[-1]}], "
            f"got {tuple(position_bias.shape)}"
        )
    batch, length, head_dim = kv.shape
    n_windows = length // compress_rate
    if n_windows == 0:
        return kv.new_zeros((batch, 0, head_dim))
    usable = n_windows * compress_rate
    w_kv = kv[:, :usable].view(batch, n_windows, compress_rate, head_dim)
    w_gate = gate[:, :usable].view(batch, n_windows, compress_rate, head_dim) + position_bias
    weights = w_gate.softmax(dim=2, dtype=torch.float32).to(w_gate.dtype)
    return rms_norm((w_kv * weights).sum(dim=2), weight, eps)


def compress_windows_csa(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    compress_rate: int,
    prior_kv: torch.Tensor | None = None,
    prior_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """CSA two-series compression: ``[B, T, 2*head_dim]`` -> ``[B, T // rate, head_dim]``.

    Each token emits **two** series in one projection: ``Ca = [..., :head_dim]`` feeds the
    *next* window's entry and ``Cb = [..., head_dim:]`` feeds the *current* one. Entry ``w``
    is therefore a convex combination over ``2 * rate`` slots - window ``w-1``'s Ca slice
    then window ``w``'s Cb slice - with a stride of only ``rate``. Getting the halves
    swapped, or using one series twice, is silent: the shapes still work.

    ``prior_kv`` / ``prior_gate`` carry window ``-1``'s Ca slice across calls. On the very
    first call that slot must stay **zero kv with -inf gate**, which gives it softmax weight
    exactly zero; filling it with zeros-but-finite-gate instead would dilute every entry of
    the first window and is the failure mode this function's shape guards exist for.
    """
    head_dim = kv.shape[-1] // 2
    if kv.ndim != 3 or gate.shape != kv.shape:
        raise ValueError(
            f"kv and gate must both be [B, T, 2*head_dim]; got {tuple(kv.shape)} " f"and {tuple(gate.shape)}"
        )
    if position_bias.shape != (compress_rate, 2 * head_dim):
        raise ValueError(
            f"position_bias must be [{compress_rate}, {2 * head_dim}], got " f"{tuple(position_bias.shape)}"
        )
    if (prior_kv is None) != (prior_gate is None):
        raise ValueError("prior_kv and prior_gate must be given together")

    batch, length, _ = kv.shape
    n_windows = length // compress_rate
    if n_windows == 0:
        return kv.new_zeros((batch, 0, head_dim))
    usable = n_windows * compress_rate
    chunk_kv = kv[:, :usable].view(batch, n_windows, compress_rate, 2 * head_dim)
    chunk_gate = gate[:, :usable].view(batch, n_windows, compress_rate, 2 * head_dim) + position_bias

    new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * compress_rate, head_dim))
    new_gate = chunk_gate.new_full((batch, n_windows, 2 * compress_rate, head_dim), float("-inf"))
    new_kv[:, :, compress_rate:] = chunk_kv[..., head_dim:]  # Cb: current window
    new_gate[:, :, compress_rate:] = chunk_gate[..., head_dim:]
    if n_windows > 1:  # Ca shifted by one window
        new_kv[:, 1:, :compress_rate] = chunk_kv[:, :-1, :, :head_dim]
        new_gate[:, 1:, :compress_rate] = chunk_gate[:, :-1, :, :head_dim]
    if prior_kv is not None:
        if prior_kv.shape != (batch, compress_rate, head_dim):
            raise ValueError(f"prior_kv must be [B, rate, head_dim], got {tuple(prior_kv.shape)}")
        new_kv[:, 0, :compress_rate] = prior_kv.to(new_kv.dtype)
        new_gate[:, 0, :compress_rate] = prior_gate.to(new_gate.dtype)

    weights = new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)
    return rms_norm((new_kv * weights).sum(dim=2), weight, eps)


def sparse_selection_bias(
    top_k_indices: torch.Tensor, compressed_len: int, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``[B, S, k]`` index tensor -> ``[B, 1, S, compressed_len]`` additive mask.

    The indexer marks slots it does not want with ``-1``; those are scattered into a
    throwaway ``compressed_len`` column and dropped by the final slice, so a raw scatter of
    ``-1`` (which Python-wraps to the last column) would corrupt the mask. Causal
    eligibility is already the indexer's responsibility, so this function only expresses
    "selected or not".
    """
    if top_k_indices.ndim != 3:
        raise ValueError(f"top_k_indices must be [B, S, k], got {tuple(top_k_indices.shape)}")
    batch, seq, _ = top_k_indices.shape
    valid = top_k_indices >= 0
    safe = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
    bias = top_k_indices.new_full((batch, 1, seq, compressed_len + 1), float("-inf"), dtype=dtype)
    bias.scatter_(-1, safe.unsqueeze(1), 0.0)
    return bias[..., :compressed_len]


class TtHCACompressor:
    """Port-side HCA compressor: weights, rope tables, and the compression math.

    Construction is device-free by design, matching ``TtV4SlidingAttention``.
    """

    #: Which rope parameter group the reference uses for compressed entries.
    ROPE_GROUP = "compress"

    def __init__(self, cfg, weights: dict, *, max_windows: int = 256):
        self.cfg = cfg
        self.w = weights
        self.compress_rate = int(cfg.compress_rates["heavily_compressed_attention"])
        if self.compress_rate <= 0:
            raise ValueError(f"compress rate must be positive, got {self.compress_rate}")
        self.head_dim = int(cfg.head_dim)
        self.hidden_size = int(cfg.hidden_size)
        self.eps = float(cfg.rms_norm_eps)
        self.max_windows = max_windows
        self._build_host_tables(cfg)

    # ---------------------------------------------------------------- host tables

    def _build_host_tables(self, cfg) -> None:
        """RoPE tables for compressed entries, built once, at the compress group's theta."""
        params = v4_rope.RopeParams.from_config(cfg, self.ROPE_GROUP)
        self.rope_dim = v4_rope.rope_dim(self.head_dim, params.partial_rotary_factor)
        inv_freq, attention_factor, self.rope_params = v4_rope.build_rope(cfg, self.ROPE_GROUP, self.head_dim)
        cos, sin = v4_rope.cos_sin_tables(
            inv_freq, attention_factor, self.max_windows * self.compress_rate, self.rope_dim
        )
        # Indexed by *entry*, not by token: entry i uses row i * rate + first_window_position.
        self._cos_by_offset = cos
        self._sin_by_offset = sin
        self._inv_freq = inv_freq

    def rope_for_entries(self, n_windows: int, first_window_position: int = 0):
        """``(cos, sin)`` as ``[1, n_windows, rope_dim]`` for the entry slots.

        Raises rather than silently wrapping, because reading past the table during decode
        would look like a numerical bug in the attention rather than a capacity limit here.
        """
        last = (n_windows - 1) * self.compress_rate + first_window_position
        if n_windows and last >= self._cos_by_offset.shape[0]:
            raise ValueError(
                f"entry slot {last} exceeds the precomputed table "
                f"({self._cos_by_offset.shape[0]} rows); raise max_windows at construction"
            )
        rows = compressed_positions(
            n_windows, self.compress_rate, first_window_position, device=self._cos_by_offset.device
        )
        cos = self._cos_by_offset[rows.squeeze(0)].unsqueeze(0)
        sin = self._sin_by_offset[rows.squeeze(0)].unsqueeze(0)
        return cos, sin

    # ------------------------------------------------------------- construction

    def create_configs(self, device=None) -> None:
        """Move weights and rope tables to ``device``.

        The intended device decomposition, none of it executed yet: ``kv_proj``/``gate_proj``
        are two ``[hidden, head_dim]`` matmuls (interleavable into one GEMM), the window
        softmax and the reduction are per-layer device ops over
        ``[B, n_windows, rate, head_dim]``, and the rope is an elementwise pair of tables
        rather than the half-split rotation the sliding path uses, because a compressed
        entry is a single ``head_dim`` vector with no heads to group.
        """
        if device is None:
            raise RuntimeError("create_configs needs a device; host tables are separate")
        self._device_tables = {
            name: tensor.to(device) for name, tensor in (("cos", self._cos_by_offset), ("sin", self._sin_by_offset))
        }

    # -------------------------------------------------------------------- forward

    def project(self, hidden_states: torch.Tensor):
        """``kv = W_kv h`` and ``gate = W_gate h``, the two projections the window needs."""
        kv = hidden_states @ self.w["kv_proj"].to(hidden_states.dtype).t()
        gate = hidden_states @ self.w["gate_proj"].to(hidden_states.dtype).t()
        return kv, gate

    def compress(self, hidden_states: torch.Tensor):
        """Compressed entries from one call's hidden states, stateless single-shot.

        Returns ``[B, T // rate, head_dim]``. The partial-window remainder is dropped, which
        is the reference's behaviour with no cache; decode keeps it in the cache instead.
        """
        kv, gate = self.project(hidden_states)
        return compress_windows(
            kv, gate, self.w["position_bias"], self.w["kv_norm_weight"], self.eps, self.compress_rate
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_len: int,
        first_window_position: int = 0,
    ):
        """Compressed entries (roped) plus the additive causal mask over them.

        ``seq_len == 1`` returns no mask, matching the reference: a single decode query
        cannot be masked against entries it is allowed to see, and the caller attends to all
        ``compressed_len`` entries.
        """
        entries = self.compress(hidden_states)
        cos, sin = self.rope_for_entries(entries.shape[1], first_window_position)
        entries = apply_v4_rope(entries.unsqueeze(1), cos, sin).squeeze(1)
        if position_ids.shape[1] == 1 or compressed_len == 0:
            return entries.unsqueeze(1), None
        return entries.unsqueeze(1), window_causal_bias(
            position_ids, compressed_len, self.compress_rate, dtype=entries.dtype
        )


class TtCSACompressor:
    """Port-side CSA compressor: two-series windows, compress-group rope, sparse selection.

    Same device-free construction contract as :class:`TtHCACompressor`. The Lightning
    Indexer is deliberately **not** reimplemented here; :meth:`forward` takes the index
    tensor as an argument so the compressor can be validated on its own and the device
    indexer can be slotted in without touching this math.
    """

    ROPE_GROUP = "compress"

    def __init__(self, cfg, weights: dict, *, max_windows: int = 8192):
        self.cfg = cfg
        self.w = weights
        self.compress_rate = int(cfg.compress_rates["compressed_sparse_attention"])
        if self.compress_rate <= 0:
            raise ValueError(f"CSA rate must be positive, got {self.compress_rate}")
        self.head_dim = int(cfg.head_dim)
        self.eps = float(cfg.rms_norm_eps)
        self.max_windows = max_windows
        params = v4_rope.RopeParams.from_config(cfg, self.ROPE_GROUP)
        self.rope_dim = v4_rope.rope_dim(self.head_dim, params.partial_rotary_factor)
        inv_freq, attention_factor, self.rope_params = v4_rope.build_rope(cfg, self.ROPE_GROUP, self.head_dim)
        cos, sin = v4_rope.cos_sin_tables(inv_freq, attention_factor, max_windows * self.compress_rate, self.rope_dim)
        self._cos_by_offset, self._sin_by_offset = cos, sin

    def rope_for_entries(self, n_windows: int, first_window_position: int = 0):
        """Entry rope tables; entry ``i`` sits at ``i * rate + first_window_position``."""
        last = (n_windows - 1) * self.compress_rate + first_window_position
        if n_windows and last >= self._cos_by_offset.shape[0]:
            raise ValueError(
                f"entry slot {last} exceeds the precomputed table "
                f"({self._cos_by_offset.shape[0]} rows); raise max_windows at construction"
            )
        rows = compressed_positions(
            n_windows, self.compress_rate, first_window_position, device=self._cos_by_offset.device
        )
        return (self._cos_by_offset[rows.squeeze(0)].unsqueeze(0), self._sin_by_offset[rows.squeeze(0)].unsqueeze(0))

    def project(self, hidden_states: torch.Tensor):
        """Both series come from one projection each: ``[B, T, 2 * head_dim]``."""
        kv = hidden_states @ self.w["kv_proj"].to(hidden_states.dtype).t()
        gate = hidden_states @ self.w["gate_proj"].to(hidden_states.dtype).t()
        return kv, gate

    def compress(self, hidden_states: torch.Tensor, prior_kv=None, prior_gate=None, first_window_position: int = 0):
        """Compressed entries (roped) for one call."""
        kv, gate = self.project(hidden_states)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        compressed_len: int,
        prior_kv=None,
        prior_gate=None,
        first_window_position: int = 0,
    ):
        """Roped entries plus the indexer's selection mask.

        ``top_k_indices`` is ``[B, S, k]`` with ``-1`` for rejected slots, i.e. exactly the
        indexer's output contract.
        """
        entries = self.compress(hidden_states, prior_kv, prior_gate, first_window_position)
        if top_k_indices is None or compressed_len == 0:
            return entries.unsqueeze(1), None
        return entries.unsqueeze(1), sparse_selection_bias(top_k_indices, compressed_len, dtype=entries.dtype)


def apply_v4_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the **trailing** rope slice of ``x``, leaving the leading nope channels alone.

    V4-Flash lays a head out as ``[nope | rope]``, which the reference expresses as
    ``x[..., :-rope_dim], x[..., -rope_dim:]``. Rotating the leading slice instead is a
    silent wrong-rope bug that short-context parity checks will not always catch.

    ``cos``/``sin`` are the **expanded** tables (width ``rope_dim``). The reference returns
    half-width tables and expands them with ``repeat_interleave(2)`` next to the rotation
    math, so comparing against it means expanding the reference's output, not shrinking
    ours. The arithmetic is fp32 with ``rotate_half`` over interleaved pairs
    (``x[..., 0::2]`` / ``x[..., 1::2]``), then cast back to the operand dtype.
    """
    rope_dim = cos.shape[-1]
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin must match shape; got {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if rope_dim > x.shape[-1] or rope_dim % 2:
        raise ValueError(f"rope_dim {rope_dim} invalid for vector width {x.shape[-1]}")
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    # rotate_half over interleaved pairs: (-x[2i+1], x[2i]) reinterleaved.
    rot_half = torch.stack((-rope[..., 1::2], rope[..., 0::2]), dim=-1).flatten(-2, -1)
    rotated = ((rope.float() * cos) + (rot_half.float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


def apply_interleaved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Deprecated alias kept only so an old import fails loudly rather than silently.

    :func:`apply_v4_rope` is the correct entry point; this name suggested the whole vector
    rotates, which it does not.
    """
    raise RuntimeError(
        "apply_interleaved_rope was replaced by apply_v4_rope, which rotates the trailing "
        "rope slice; see its docstring"
    )
