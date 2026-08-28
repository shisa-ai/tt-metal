"""Compression cache state for V4-Flash decode and chunked prefill.

Two classes mirror ``DeepseekV4HCACache`` / ``DeepseekV4CSACache``. Everything they decide is
decode-side correctness, and none of it is visible in a single-shot prefill test:

* ``first_window_position`` comes from ``entry_count * compress_rate``, **not** from
  ``position_ids``. That is what keeps prefill -> decode -> prefill consistent, and it means a
  compressed entry is rotated at its window *slot*, so tokens sitting in the partial-window
  buffer do not advance positions until their window closes.
* The partial-window remainder persists in ``buffer_kv`` / ``buffer_gate``. Feed 300 tokens at
  rate 128 and two entries close; the remaining 44 tokens are not lost and not compressed yet.
* CSA carries only the previous window's **Ca** slice across a call boundary — Cb was already
  folded into the emitted entry — so the overlap state is half the projected width, per
  ``name``.
* The sliding-window branch keeps ``sliding_window - 1`` tokens (127 for V4-Flash) and returns
  the untruncated concatenation, and shares one tensor for keys and values because V4 is
  shared-KV MQA.

This state is not optional plumbing. Asking the released reference to decode with a generic
``DynamicCache()`` dies on the first compressed layer with
``'DynamicLayer' object has no attribute 'store_compression_weights'`` -- the model builds its
own cache from its own config (``DynamicCache(config=self.config)``, which is what gives each
layer a ``DeepseekV4HCACache`` / ``DeepseekV4CSACache``), and the modelling file's own note
rules out ``StaticCache`` for the same reason. Measured with the released weights on host while
generating ``tools/v4_real_weight_generate.py``. So a device decoder has to carry this state
wherever its cache lives; there is no generic cache to inherit.

Host-only bookkeeping on torch tensors; the device layout is a separate decision. Nothing
here has run on hardware.
"""

from __future__ import annotations

import torch


class TtCompressionCache:
    """Per-layer compression state for one or more named producers.

    ``names`` is ``("compressor",)`` for HCA and ``("compressor", "indexer")`` for CSA. Each
    name keeps its own buffer, entries and count, because the indexer runs its own compressor
    at a different width and must not share a counter with the outer one.
    """

    def __init__(self, compress_rate: int, sliding_window: int, names=("compressor",), *, overlap: bool = False):
        if compress_rate <= 0:
            raise ValueError(f"compress_rate must be positive, got {compress_rate}")
        if sliding_window <= 1:
            raise ValueError(f"sliding_window must exceed 1, got {sliding_window}")
        if len(set(names)) != len(names) or not names:
            raise ValueError(f"names must be unique and non-empty, got {names}")
        self.compress_rate = compress_rate
        self.sliding_window = sliding_window
        self.names = tuple(names)
        self.overlap_enabled = overlap
        self.buffer_kv: dict[str, torch.Tensor | None] = {n: None for n in self.names}
        self.buffer_gate: dict[str, torch.Tensor | None] = {n: None for n in self.names}
        self.compressed_kv: dict[str, torch.Tensor | None] = {n: None for n in self.names}
        self.entry_count: dict[str, int] = {n: 0 for n in self.names}
        self.overlap_kv: dict[str, torch.Tensor | None] = {n: None for n in self.names}
        self.overlap_gate: dict[str, torch.Tensor | None] = {n: None for n in self.names}
        # Shared-KV MQA: one buffer stands in for both, like the reference's values = keys.
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.cumulative_length = 0

    # ------------------------------------------------------- sliding-window branch

    def update(self, key_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append to the sliding window and return the *untruncated* concatenation.

        Only the retained state is clipped, to ``sliding_window - 1`` tokens; the returned
        pair still covers the retained tail plus this call, which is what attention consumes.
        Clipping the return as well would drop the current call's tokens from the very result
        the caller is about to score.
        """
        if key_states.ndim != 4:
            raise ValueError(f"key_states must be [B, H, S, D], got {tuple(key_states.shape)}")
        if self.keys is None:
            # Start from an *empty* window, mirroring the reference's lazy initialisation.
            # Seeding with this call's keys and then concatenating them again doubles every
            # first-call token — a real bug, caught by parity, that looks like a duplicated
            # prefix rather than an error.
            self.keys = key_states.new_empty(key_states.shape[:2] + (0, key_states.shape[-1]))
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full

    # --------------------------------------------------------- compression branch

    def store_compression_weights(
        self,
        name: str,
        kv: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Buffer, then return the longest window-aligned prefix plus its slot position.

        ``first_window_position`` is ``entry_count * compress_rate``: the absolute position of
        the first source token of the first window being closed now.
        """
        self._check_name(name)
        if kv.shape != gate.shape:
            raise ValueError(f"kv/gate must match shape; got {tuple(kv.shape)} vs " f"{tuple(gate.shape)}")
        if kv.ndim != 3:
            raise ValueError(f"kv must be [B, T, F], got {tuple(kv.shape)}")
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        """Append emitted entries, bump the count, and return the running entries."""
        self._check_name(name)
        if compressed.ndim != 3:
            raise ValueError(f"compressed must be [B, N, D], got {tuple(compressed.shape)}")
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[1] > 0:
            if self.compressed_kv[name].shape[-1] != compressed.shape[-1]:
                raise ValueError(
                    f"entry width changed for {name!r}: {self.compressed_kv[name].shape[-1]} "
                    f"vs {compressed.shape[-1]}"
                )
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
        self.entry_count[name] += compressed.shape[1]
        return self.compressed_kv[name]

    def update_overlap_state(self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int):
        """Return the prior call's Ca slice and persist this call's last-window Ca slice.

        Only ``:head_dim`` is stored. Keeping the full ``2 * head_dim`` would not break
        correctness but would double the persistent state of 21 of 43 layers, and reading Cb
        back would be a bug: it was already folded into an emitted entry.
        """
        self._check_name(name)
        if not self.overlap_enabled:
            raise RuntimeError("overlap state belongs to the CSA path only; HCA windows do not overlap")
        if chunk_kv.ndim != 4 or chunk_kv.shape != chunk_gate.shape:
            raise ValueError(
                f"chunk kv/gate must both be [B, N, rate, F]; got "
                f"{tuple(chunk_kv.shape)} and {tuple(chunk_gate.shape)}"
            )
        if head_dim <= 0 or head_dim * 2 != chunk_kv.shape[-1]:
            raise ValueError(f"head_dim {head_dim} inconsistent with feature width {chunk_kv.shape[-1]}")
        prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
        # clone(): the slice aliases the caller's chunk, which is typically a view of a
        # projection buffer the caller is free to reuse on the next call.
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        return prior_kv, prior_gate

    # -------------------------------------------------------------------- helpers

    def _check_name(self, name: str) -> None:
        if name not in self.names:
            raise KeyError(f"unknown producer {name!r}; this cache tracks {list(self.names)}")

    def compressed_length(self, name: str = "compressor") -> int:
        entries = self.compressed_kv[name]
        return 0 if entries is None else entries.shape[1]

    def retained_length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[-2]


def hca_cache(cfg, **kwargs) -> TtCompressionCache:
    return TtCompressionCache(
        int(cfg.compress_rates["heavily_compressed_attention"]),
        int(cfg.sliding_window),
        names=("compressor",),
        **kwargs,
    )


def csa_cache(cfg, **kwargs) -> TtCompressionCache:
    return TtCompressionCache(
        int(cfg.compress_rates["compressed_sparse_attention"]),
        int(cfg.sliding_window),
        names=("compressor", "indexer"),
        overlap=True,
        **kwargs,
    )
