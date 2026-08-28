# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Where the cached and cache-free reference paths stop being the same model.

Observed while generating real completions with the released weights (worklog 6e5e0d): cached
decode and a cache-free forward over the *same* token prefix agree at cosine 1.0 for a
12-token prompt, then diverge hard (cosine 0.47-0.83, max|Δlogit| 19-33) as soon as the prefix
length stops being a multiple of the compressor rate. The explanation is eight lines of the
reference, not a numerical accident:

    if cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights(...)

``DeepseekV4CSACompressor`` and ``DeepseekV4HCACompressor`` both do this. With no cache, the
tokens past the last whole window are **dropped** -- they reach no entry and no attention. With
a cache they sit in ``buffer_kv``/``buffer_gate`` until their window closes. So "run it in one
forward instead" is not a control for cached decode: for a prefix of 13 tokens against rate 4 it
silently ignores the newest token, which is exactly the token decode depends on.

The second test is the good news, and it is the invariant a device cache has to satisfy: inside
the *cached* path, chunk size does not matter. 12+1+1 and 14-in-one-call reach the same entries,
the same remainder and the same ``first_window_position``. So a serving stack may re-chunk
freely; it may not go cache-free.

Pure cache arithmetic at the released configuration -- no weights, no device.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4CSACompressor,
    DeepseekV4HCACache,
    DeepseekV4HCACompressor,
)
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import default_snapshot_dir

pytestmark = pytest.mark.skipif(default_snapshot_dir() is None, reason="no V4-Flash snapshot present")


@pytest.fixture(scope="module")
def cfg():
    return AutoConfig.from_pretrained(default_snapshot_dir(), trust_remote_code=True)


def _rate(cfg, layer_type):
    return cfg.compress_rates[layer_type]


def _feed(cache, pieces, head_dim, source):
    """Push consecutive token-groups from ``source`` through the cache.

    Returns the total projected tokens handed back for compression, in order, so a test can
    check nothing was lost or reordered, not merely that a count looks plausible.
    """
    handed, cursor = [], 0
    for n in pieces:
        piece = source[:, cursor : cursor + n]
        cursor += n
        chunk_kv, chunk_gate, first_window_position = cache.store_compression_weights("compressor", piece, piece * 0.5)
        # Mirror what the compressor does with a window-aligned chunk: emit one entry per
        # window. The entry *value* is irrelevant here -- this checks cache bookkeeping.
        n_windows = chunk_kv.shape[1] // cache.compress_rate
        if n_windows:
            cache.update_compressor_states("compressor", torch.zeros(1, n_windows, chunk_kv.shape[-1]))
        handed.append((chunk_kv, first_window_position))
    total = torch.cat([c for c, _ in handed], dim=1) if handed else None
    return total, [p for c, p in handed]


def test_both_compressors_take_the_truncating_no_cache_branch(cfg):
    """The mechanism is quoted from the source rather than inferred from behaviour.

    ``co_consts`` introspection was the first attempt and proves nothing -- local variable names
    live in ``co_varnames``, so the assertion failed on correct code. Reading the source is the
    honest way to pin "this branch exists and drops the remainder", and it fails loudly if
    upstream ever changes the shape of it.
    """
    import inspect

    assert _rate(cfg, "compressed_sparse_attention") == 4, "m=4 in the paper's notation"
    assert _rate(cfg, "heavily_compressed_attention") > 4, "HCA windows are the wide ones"

    cache_src = inspect.getsource(DeepseekV4HCACache.store_compression_weights)
    assert "usable = (kv.shape[1] // self.compress_rate) * self.compress_rate" in cache_src
    assert "self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:]" in cache_src
    for cls in (DeepseekV4CSACompressor, DeepseekV4HCACompressor):
        src = inspect.getsource(cls.forward)
        assert "if cache_layer is None:" in src, f"{cls.__name__}: the no-cache branch moved"
        assert "usable" in src, f"{cls.__name__}: no truncating branch found in the no-cache path"


def test_no_cache_branch_drops_tokens_past_the_last_whole_window(cfg):
    """Reproduce the arithmetic of ``cache_layer is None``.

    A 13-token prefix at rate 4 gives ``usable == 12``: the thirteenth token -- the one decode
    just produced, in every real serving step -- contributes to nothing.
    """
    rate = _rate(cfg, "compressed_sparse_attention")
    for prefix, dropped in ((12, 0), (13, 1), (14, 2), (15, 3), (16, 0)):
        usable = (prefix // rate) * rate
        assert usable == prefix - dropped
        assert dropped == prefix % rate
    # Non-aligned prefixes are the common case, not an edge case: of the five prompts in the
    # retained prefill sweep, three had lengths 6, 6 and 13 -- each lost trailing tokens to this
    # branch, so those rows were scored on 4-, 4- and 12-token inputs.
    assert [p % 4 for p in (12, 8, 6, 13)] == [0, 0, 2, 1]


def test_cached_path_is_chunk_size_invariant_csa(cfg):
    """12+1+1 through the cache must equal 14 in one call, or the device cache is hopeless.

    Content, not just counts: the tokens handed back for compression must be the same tensor
    content in the same order, and the retained remainder must match.
    """
    rate, head_dim = _rate(cfg, "compressed_sparse_attention"), cfg.head_dim
    source = torch.randn(1, 14, 2 * head_dim, generator=torch.Generator().manual_seed(7))

    chunked = DeepseekV4CSACache(cfg)
    handed_a, positions_a = _feed(chunked, (12, 1, 1), head_dim, source)
    single = DeepseekV4CSACache(cfg)
    handed_b, positions_b = _feed(single, (14,), head_dim, source)

    expected = source[:, : (14 // rate) * rate]
    assert handed_a is not None and handed_b is not None
    torch.testing.assert_close(handed_a, expected, atol=0, rtol=0)
    torch.testing.assert_close(handed_b, expected, atol=0, rtol=0)
    assert chunked.entry_count["compressor"] == single.entry_count["compressor"] == 14 // rate
    assert chunked.buffer_kv["compressor"].shape[1] == single.buffer_kv["compressor"].shape[1] == 14 % rate
    assert positions_a[0] == positions_b[0] == 0, "first window position tracks entries, not position_ids"


def test_cached_path_keeps_what_the_no_cache_branch_throws_away(cfg):
    """After 13 tokens the cache holds a remainder; the branch has none."""
    head_dim, rate = cfg.head_dim, _rate(cfg, "compressed_sparse_attention")
    cache = DeepseekV4CSACache(cfg)
    source = torch.randn(1, 13, 2 * head_dim, generator=torch.Generator().manual_seed(11))
    _feed(cache, (13,), head_dim, source)
    assert cache.buffer_kv["compressor"].shape[1] == 13 % rate == 1, "the trailing token is retained"
    assert cache.entry_count["compressor"] == 13 // rate

    # The no-cache branch returns no remainder by construction: one path has consumed 12 tokens
    # and forgotten one, the other has consumed 12 and is holding one. That difference *is* the
    # cached-vs-forward divergence measured in worklog 6e5e0d, and it is why a cache-free rerun
    # cannot serve as its own control.
    assert (13 // rate) * rate == 12


def test_hca_agrees_under_the_same_chunking(cfg):
    """HCA is non-overlapping but keeps the same buffer discipline, so check it too.

    The rate is taken from the cache instance, not from the config key: the first version read
    ``compress_rates["heavily_compressed_attention"]`` while feeding an HCA-shaped expectation
    into a CSA cache, and the assertion computed entries for the wrong window size.
    """
    head_dim = cfg.head_dim
    rate = DeepseekV4HCACache(cfg).compress_rate
    assert rate == _rate(cfg, "heavily_compressed_attention")
    length = rate * 2 + 3
    source = torch.randn(1, length, 2 * head_dim, generator=torch.Generator().manual_seed(3))

    chunked = DeepseekV4HCACache(cfg)
    handed_a, _ = _feed(chunked, (rate * 2, 3), head_dim, source)
    single = DeepseekV4HCACache(cfg)
    handed_b, _ = _feed(single, (length,), head_dim, source)

    expected = source[:, : rate * 2]
    torch.testing.assert_close(handed_a, expected, atol=0, rtol=0)
    torch.testing.assert_close(handed_b, expected, atol=0, rtol=0)
    assert chunked.entry_count["compressor"] == single.entry_count["compressor"] == 2
    assert chunked.buffer_kv["compressor"].shape[1] == single.buffer_kv["compressor"].shape[1] == 3
