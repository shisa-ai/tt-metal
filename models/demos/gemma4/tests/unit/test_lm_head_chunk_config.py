# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.gemma4.tt import lm_head_config

VOCAB_SIZE = 262144


def test_screen_chunk_sizes_bound_launch_count_and_l1_surface():
    assert lm_head_config.tp1_lm_head_dram_shard_screen_chunk_sizes(2560, VOCAB_SIZE) == (
        4096,
        8192,
        16384,
        32768,
    )


def test_cb_estimate_accepts_32768_and_rejects_65536():
    assert lm_head_config.estimate_chotto_tp1_lm_head_cb_bytes(32768) == 1302528
    assert lm_head_config.estimate_chotto_tp1_lm_head_cb_bytes(65536) == 2580480
    assert 1302528 < lm_head_config.WORMHOLE_CB_USABLE_BYTES < 2580480


@pytest.mark.parametrize("value", [4096, "8192", " 32768 "])
def test_selector_accepts_bounded_explicit_widths(value):
    assert lm_head_config.resolve_tp1_lm_head_dram_shard_chunk_size(2560, VOCAB_SIZE, value) == int(value)


@pytest.mark.parametrize("value", [0, "0"])
def test_selector_zero_preserves_stock_width(value):
    assert (
        lm_head_config.resolve_tp1_lm_head_dram_shard_chunk_size(2560, VOCAB_SIZE, value)
        == lm_head_config.TP1_LM_HEAD_DRAM_SHARD_DEFAULT_CHUNK_SIZE
    )


@pytest.mark.parametrize("value", [True, False, "auto", -1, 2048, 65536])
def test_selector_rejects_implicit_or_out_of_bounds_values(expect_error, value):
    with expect_error(ValueError, lm_head_config.TP1_LM_HEAD_DRAM_SHARD_CHUNK_SIZE_ENV):
        lm_head_config.resolve_tp1_lm_head_dram_shard_chunk_size(2560, VOCAB_SIZE, value)


def test_selector_reads_environment(monkeypatch):
    monkeypatch.setenv(lm_head_config.TP1_LM_HEAD_DRAM_SHARD_CHUNK_SIZE_ENV, "16384")
    assert lm_head_config.resolve_tp1_lm_head_dram_shard_chunk_size(2560, VOCAB_SIZE) == 16384


@pytest.mark.parametrize("shape", [(0, VOCAB_SIZE), (2560, 0), (2816, VOCAB_SIZE), (2560, 262208)])
def test_surface_rejects_unqualified_shape(expect_error, shape):
    with expect_error(ValueError, "qualified only"):
        lm_head_config.tp1_lm_head_dram_shard_screen_chunk_sizes(*shape)
