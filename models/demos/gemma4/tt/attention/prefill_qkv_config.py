# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure configuration helpers for the Chotto ISL-1024 sliding-QKV screen."""

from __future__ import annotations

import os

PREFILL_SLIDING_QKV_IN0_BLOCK_W_ENV = "GEMMA4_PREFILL_SLIDING_QKV_IN0_BLOCK_W"
PREFILL_SLIDING_QKV_SEQUENCE_LENGTH = 1024
PREFILL_SLIDING_QKV_HIDDEN_SIZE = 2560
PREFILL_SLIDING_QKV_OUTPUT_SIZE = 3072
PREFILL_SLIDING_QKV_IN0_BLOCK_WIDTHS = (1, 2, 4, 5, 8, 10, 16)


def resolve_prefill_sliding_qkv_in0_block_w(value=None):
    """Resolve zero/unset as stock and accept only the source-derived L1-safe set."""
    if value is None:
        value = os.environ.get(PREFILL_SLIDING_QKV_IN0_BLOCK_W_ENV, "0")
    if isinstance(value, bool):
        raise ValueError(f"{PREFILL_SLIDING_QKV_IN0_BLOCK_W_ENV} requires an explicit integer width, got {value!r}")
    try:
        width = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"{PREFILL_SLIDING_QKV_IN0_BLOCK_W_ENV} requires an integer width, got {value!r}") from error
    if width == 0:
        return None
    if width not in PREFILL_SLIDING_QKV_IN0_BLOCK_WIDTHS:
        raise ValueError(
            f"{PREFILL_SLIDING_QKV_IN0_BLOCK_W_ENV} must be 0 or one of "
            f"{PREFILL_SLIDING_QKV_IN0_BLOCK_WIDTHS}, got {width}"
        )
    return width
