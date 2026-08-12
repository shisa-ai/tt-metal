# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure configuration helpers for the TP1 DRAM-sharded Gemma LM head."""

from __future__ import annotations

import os

TP1_LM_HEAD_DRAM_SHARD_CHUNK_SIZE_ENV = "GEMMA4_TP1_LM_HEAD_DRAM_SHARD_CHUNK_SIZE"
TP1_LM_HEAD_DRAM_SHARD_DEFAULT_CHUNK_SIZE = 8192
TP1_LM_HEAD_DRAM_SHARD_MIN_SCREEN_CHUNK_SIZE = 4096
CHOTTO_TP1_LM_HEAD_HIDDEN_SIZE = 2560
CHOTTO_TP1_LM_HEAD_VOCAB_SIZE = 262144
WORMHOLE_CB_USABLE_BYTES = 1391936
WORMHOLE_BF16_TILE_BYTES = 2048
WORMHOLE_DRAM_BANKS = 12
CHOTTO_TP1_LM_HEAD_WORKER_CORES = 40
CHOTTO_TP1_LM_HEAD_IN0_BLOCK_W = 2


def estimate_chotto_tp1_lm_head_cb_bytes(chunk_size: int) -> int:
    """Conservatively estimate DRAM-sharded LM-head circular-buffer bytes.

    This mirrors the pinned DRAM-sharded program factory for Chotto's M=32,
    K=2,560 BF16 shape: double-buffered in0, triple-buffered in1 over 12 DRAM
    banks, an output/intermediate CB padded to at most an eight-tile subblock,
    the 40-core output shard, and the two-tile input shard. There is no bias.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer; got {chunk_size!r}")
    if chunk_size % 32 != 0:
        raise ValueError(f"chunk_size must be tile aligned; got {chunk_size}")

    n_tiles = chunk_size // 32
    per_dram_bank_n = (n_tiles + WORMHOLE_DRAM_BANKS - 1) // WORMHOLE_DRAM_BANKS
    padded_compute_n = ((per_dram_bank_n + 7) // 8) * 8
    output_shard_n = (n_tiles + CHOTTO_TP1_LM_HEAD_WORKER_CORES - 1) // CHOTTO_TP1_LM_HEAD_WORKER_CORES
    fixed_tiles = 2 * CHOTTO_TP1_LM_HEAD_IN0_BLOCK_W + 2
    in1_tiles = 3 * CHOTTO_TP1_LM_HEAD_IN0_BLOCK_W * per_dram_bank_n
    total_tiles = fixed_tiles + in1_tiles + padded_compute_n + output_shard_n
    return total_tiles * WORMHOLE_BF16_TILE_BYTES


def tp1_lm_head_dram_shard_screen_chunk_sizes(hidden_size: int, vocab_size: int) -> tuple[int, ...]:
    """Return the bounded power-of-two chunk surface around the stock path.

    The screen starts at 4,096 columns (64 launches for Chotto's 262,144-word
    vocabulary) and stops at the largest source-derived Wormhole L1-safe width,
    32,768. Values must divide the vocabulary so every chunk has one frozen
    shape and program configuration.
    """
    if hidden_size != CHOTTO_TP1_LM_HEAD_HIDDEN_SIZE or vocab_size != CHOTTO_TP1_LM_HEAD_VOCAB_SIZE:
        raise ValueError(
            "TP1 LM-head chunk screening is qualified only for "
            f"hidden_size={CHOTTO_TP1_LM_HEAD_HIDDEN_SIZE}, vocab_size={CHOTTO_TP1_LM_HEAD_VOCAB_SIZE}; "
            f"got hidden_size={hidden_size!r}, vocab_size={vocab_size!r}"
        )

    values = []
    width = TP1_LM_HEAD_DRAM_SHARD_MIN_SCREEN_CHUNK_SIZE
    while width <= vocab_size:
        if estimate_chotto_tp1_lm_head_cb_bytes(width) >= WORMHOLE_CB_USABLE_BYTES:
            break
        if vocab_size % width == 0:
            values.append(width)
        width *= 2
    return tuple(values)


def resolve_tp1_lm_head_dram_shard_chunk_size(hidden_size: int, vocab_size: int, value=None) -> int:
    """Resolve an explicit screen width; zero/unset preserves stock 8,192."""
    if value is None:
        value = os.environ.get(TP1_LM_HEAD_DRAM_SHARD_CHUNK_SIZE_ENV, "0")
    if isinstance(value, bool):
        raise ValueError(f"{TP1_LM_HEAD_DRAM_SHARD_CHUNK_SIZE_ENV} must be an integer, not bool")
    try:
        width = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{TP1_LM_HEAD_DRAM_SHARD_CHUNK_SIZE_ENV} must be an integer; got {value!r}") from error

    if width == 0:
        width = TP1_LM_HEAD_DRAM_SHARD_DEFAULT_CHUNK_SIZE
    allowed = tp1_lm_head_dram_shard_screen_chunk_sizes(hidden_size, vocab_size)
    if width not in allowed:
        supported = ", ".join(str(candidate) for candidate in allowed)
        raise ValueError(
            f"{TP1_LM_HEAD_DRAM_SHARD_CHUNK_SIZE_ENV}={width} is outside the bounded surface; "
            f"supported values for vocab_size={vocab_size}: {supported}"
        )
    return width
