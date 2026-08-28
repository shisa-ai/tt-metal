"""Compression cache contracts, pinned against the reference cache objects.

Cache state is where decode correctness lives, and it is invisible to any single-shot test:
the interesting cases are call boundaries that split a window, and the counters that decide
where the next compressed entry gets rotated. Every test here drives **sequences of calls** and
compares state after each one.
"""

import pytest
import torch
from transformers import AutoConfig

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
)
from models.demos.deepseek_v3_d_p.tt.v4_cache import TtCompressionCache, csa_cache, hca_cache

CHECKPOINT = "/home/ubuntu/.cache/huggingface/hub/" "models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots"

HCA_RATE = 128
CSA_RATE = 4
SLIDING = 128


def checkpoint_config():
    import glob
    import os

    snapshots = sorted(glob.glob(os.path.join(CHECKPOINT, "*")))
    if not snapshots:
        pytest.skip("V4-Flash checkpoint config not present")
    return AutoConfig.from_pretrained(snapshots[-1], trust_remote_code=True)


def projections(tokens, feature, seed=0):
    g = torch.Generator().manual_seed(seed)
    kv = torch.randn(tokens, feature, generator=g, dtype=torch.float32)
    gate = torch.randn(tokens, feature, generator=g, dtype=torch.float32)
    return kv.unsqueeze(0), gate.unsqueeze(0)


def snapshot(ref, ours):
    """Comparable view of both caches' compression state."""

    def per_name(entries, count_attr):
        return {n: (None if entries[n] is None else tuple(entries[n].shape), count_attr[n]) for n in entries}

    out = {
        "per_name": per_name(ref.compressed_kv, ref.entry_count),
        "ours_per_name": per_name(ours.compressed_kv, ours.entry_count),
    }
    ref_buf = {n: (None if ref.buffer_kv[n] is None else ref.buffer_kv[n].shape[1]) for n in ref.buffer_kv}
    our_buf = {n: (None if ours.buffer_kv[n] is None else ours.buffer_kv[n].shape[1]) for n in ours.buffer_kv}
    out["buffer"] = (ref_buf, our_buf)
    return out


# ------------------------------------------------------------------- HCA path


def test_hca_chunked_calls_track_the_reference_state_exactly():
    """Prefill 300, decode 1, 1, then 44 and 128: every call boundary splits differently."""
    cfg = checkpoint_config()
    ref = DeepseekV4HCACache(cfg)
    ours = hca_cache(cfg)
    assert ours.compress_rate == HCA_RATE == ref.compress_rate
    feature = 8
    offset = 0
    for size in (300, 1, 1, 44, 128, 7):
        kv, gate = projections(size, feature, seed=offset)
        offset += size
        ref_chunk, ref_gate_chunk, ref_pos = ref.store_compression_weights("compressor", kv, gate)
        our_chunk, our_gate_chunk, our_pos = ours.store_compression_weights("compressor", kv, gate)
        assert our_chunk.shape == ref_chunk.shape, (size, our_chunk.shape, ref_chunk.shape)
        assert torch.equal(our_chunk, ref_chunk) and torch.equal(our_gate_chunk, ref_gate_chunk)
        assert our_pos == ref_pos, (size, our_pos, ref_pos)

        n_windows = ref_chunk.shape[1] // HCA_RATE
        emitted = ref_chunk.new_zeros((1, n_windows, feature))
        ref_running = ref.update_compressor_states("compressor", emitted)
        our_running = ours.update_compressor_states("compressor", emitted.clone())
        assert ref.entry_count["compressor"] == ours.entry_count["compressor"], size
        assert our_running.shape == ref_running.shape, size
        for name in ("compressor",):
            assert ours.buffer_kv[name].shape == ref.buffer_kv[name].shape, (size, name)


def test_first_window_position_is_entry_count_not_position_ids():
    """The cache never sees positions, so a resumed stream cannot desync its rope slots."""
    cfg = checkpoint_config()
    ours = hca_cache(cfg)
    kv, gate = projections(HCA_RATE * 2 + 40, 4, seed=1)
    chunk, _g, pos = ours.store_compression_weights("compressor", kv, gate)
    assert pos == 0 and chunk.shape[1] == 2 * HCA_RATE
    ours.update_compressor_states("compressor", torch.zeros(1, 2, 4))
    kv2, gate2 = projections(1, 4, seed=2)
    _c2, _g2, pos2 = ours.store_compression_weights("compressor", kv2, gate2)
    assert pos2 == 2 * HCA_RATE, pos2
    # 40 buffered + 1 new token closes nothing, and the count is unchanged.
    assert ours.entry_count["compressor"] == 2
    assert ours.buffer_kv["compressor"].shape[1] == 41


def test_partial_buffer_survives_until_its_window_closes():
    cfg = checkpoint_config()
    ref, ours = DeepseekV4HCACache(cfg), hca_cache(cfg)
    kv, gate = projections(130, 3, seed=3)
    ref.store_compression_weights("compressor", kv, gate)
    ours.store_compression_weights("compressor", kv, gate)
    assert ref.buffer_kv["compressor"].shape[1] == 2 == ours.buffer_kv["compressor"].shape[1]
    tail_kv, tail_gate = projections(126, 3, seed=4)
    r_c, _rg, r_pos = ref.store_compression_weights("compressor", tail_kv, tail_gate)
    o_c, _og, o_pos = ours.store_compression_weights("compressor", tail_kv, tail_gate)
    assert r_c.shape[1] == 128 == o_c.shape[1]
    assert o_pos == r_pos == 0, (o_pos, r_pos)
    # The closing window must begin with the two buffered tokens, not the new ones.
    assert torch.equal(o_c[0, :2], kv[0, 128:130])


def test_slot_position_advances_only_when_an_entry_is_emitted():
    """entry_count, not tokens-seen, drives the position; so a first window sits at 0."""
    cfg = checkpoint_config()
    ours = hca_cache(cfg)
    kv, gate = projections(130, 3, seed=5)
    _c, _g, pos = ours.store_compression_weights("compressor", kv, gate)
    assert pos == 0 and ours.entry_count["compressor"] == 0
    ours.update_compressor_states("compressor", torch.zeros(1, 1, 3))
    tail_kv, tail_gate = projections(126, 3, seed=6)
    _c2, _g2, pos2 = ours.store_compression_weights("compressor", tail_kv, tail_gate)
    assert pos2 == 128, pos2
    ours.update_compressor_states("compressor", torch.zeros(1, 1, 3))
    assert ours.entry_count["compressor"] == 2 and ours.compressed_length() == 2


# ------------------------------------------------------------------- CSA path


def test_csa_overlap_state_matches_the_reference_across_calls():
    cfg = checkpoint_config()
    ref = DeepseekV4CSACache(cfg)
    ours = csa_cache(cfg)
    head_dim, feature = 4, 8
    for name in ("compressor", "indexer"):
        assert ours.overlap_enabled and name in ours.names

    expected = None
    for call, size in enumerate((12, 4, 9)):
        kv, gate = projections(size, feature, seed=10 + call)
        ref_chunk, ref_gate, _ = ref.store_compression_weights("compressor", kv, gate)
        our_chunk, our_gate, _ = ours.store_compression_weights("compressor", kv, gate)
        n = ref_chunk.shape[1] // CSA_RATE
        rc, rgc = ref_chunk.view(1, n, CSA_RATE, feature), ref_gate.view(1, n, CSA_RATE, feature)
        oc, ogc = our_chunk.view(1, n, CSA_RATE, feature), our_gate.view(1, n, CSA_RATE, feature)
        r_prior, r_prior_g = ref.update_overlap_state("compressor", rc, rgc, head_dim)
        o_prior, o_prior_g = ours.update_overlap_state("compressor", oc, ogc, head_dim)
        if call == 0:
            assert r_prior is None and o_prior is None, "first call has no predecessor"
        else:
            assert o_prior.shape == r_prior.shape == (1, CSA_RATE, head_dim)
            assert torch.equal(o_prior, r_prior), call
            assert torch.equal(o_prior_g, r_prior_g), call
            assert torch.equal(o_prior, expected[0]), call
            assert torch.equal(o_prior_g, expected[1]), call
            assert o_prior.shape[-1] == feature // 2, "only Ca may be stored"
        expected = (oc.clone()[:, -1, :, :head_dim], ogc.clone()[:, -1, :, :head_dim])


def test_overlap_slice_does_not_alias_the_callers_buffer():
    cfg = checkpoint_config()
    ours = csa_cache(cfg)
    kv, gate = projections(CSA_RATE * 2, 8, seed=20)
    chunk = kv.view(1, 2, CSA_RATE, 8)
    gate_c = gate.view(1, 2, CSA_RATE, 8)
    ours.update_overlap_state("compressor", chunk, gate_c, 4)
    saved = ours.overlap_kv["compressor"].clone()
    chunk[:, -1, :, :4] = 0.0  # caller reuses its projection buffer
    assert torch.equal(ours.overlap_kv["compressor"], saved), "stored Ca was overwritten"


def test_indexer_and_compressor_counters_are_independent():
    cfg = checkpoint_config()
    ours = csa_cache(cfg)
    kv, gate = projections(CSA_RATE * 3, 6, seed=30)
    ours.store_compression_weights("compressor", kv, gate)
    ours.update_compressor_states("compressor", torch.zeros(1, 3, 6))
    ours.store_compression_weights("indexer", kv, gate)
    assert ours.entry_count == {"compressor": 3, "indexer": 0}, ours.entry_count
    assert ours.compressed_length("compressor") == 3
    assert ours.compressed_length("indexer") == 0


# ---------------------------------------------------------- sliding-window branch


def test_sliding_window_retains_window_minus_one_and_shares_kv_storage():
    cfg = checkpoint_config()
    ref = DeepseekV4HCACache(cfg)
    ours = hca_cache(cfg)
    assert ours.sliding_window == SLIDING == ref.sliding_window
    total = 0
    for size in (200, 1, 1, 300):
        keys = torch.randn(1, 1, size, 4, dtype=torch.float32)
        r_full, r_full_v = ref.update(keys, keys, 0, None)
        o_full, o_full_v = ours.update(keys)
        # The returned span is what the *previous* call retained plus this call, while the
        # stored window is clipped to sliding_window - 1 -- returning the clipped tensor
        # instead would hide this call's own tokens from the attention that asked for them.
        expected_span = min(total, SLIDING - 1) + size
        total += size
        assert o_full.shape == r_full.shape, (size, o_full.shape, r_full.shape)
        assert torch.equal(o_full, r_full)
        assert o_full.shape[-2] == expected_span, (size, o_full.shape, expected_span)
        assert ours.retained_length() == ref.keys.shape[-2] == min(total, SLIDING - 1)
        assert ours.values is ours.keys, "shared-KV MQA must not double the cache"
        assert ref.values is ref.keys
        assert ours.cumulative_length == ref.cumulative_length == total


def test_empty_appends_do_not_move_the_entry_count():
    cfg = checkpoint_config()
    ours = hca_cache(cfg)
    ours.update_compressor_states("compressor", torch.zeros(1, 0, 5))
    assert ours.entry_count["compressor"] == 0
    assert ours.compressed_length() == 0
    ours.update_compressor_states("compressor", torch.zeros(1, 2, 5))
    assert ours.entry_count["compressor"] == 2
    ours.update_compressor_states("compressor", torch.zeros(1, 0, 5))
    assert ours.entry_count["compressor"] == 2
    assert ours.compressed_length() == 2


# --------------------------------------------------------------------- guards


def test_guards_fire_before_a_wrong_shape_becomes_a_wrong_number(expect_error):
    cfg = checkpoint_config()
    ours = hca_cache(cfg)
    with expect_error(KeyError, "unknown producer"):
        ours.store_compression_weights("indexer", torch.zeros(1, 4, 2), torch.zeros(1, 4, 2))
    with expect_error(RuntimeError, "CSA path only"):
        ours.update_overlap_state("compressor", torch.zeros(1, 1, 4, 8), torch.zeros(1, 1, 4, 8), 4)
    csa = csa_cache(cfg)
    with expect_error(ValueError, "head_dim"):
        csa.update_overlap_state("compressor", torch.zeros(1, 1, 4, 8), torch.zeros(1, 1, 4, 8), 3)
    with expect_error(ValueError, "kv/gate must match"):
        ours.store_compression_weights("compressor", torch.zeros(1, 4, 2), torch.zeros(1, 4, 3))
    with expect_error(ValueError, r"key_states must be \[B, H, S, D\]"):
        ours.update(torch.zeros(1, 4, 2))
    ours.update_compressor_states("compressor", torch.zeros(1, 2, 5))
    with expect_error(ValueError, "entry width changed"):
        ours.update_compressor_states("compressor", torch.zeros(1, 1, 7))


def test_constructor_rejects_impossible_geometry(expect_error):
    with expect_error(ValueError, "compress_rate must be positive"):
        TtCompressionCache(0, 128)
    with expect_error(ValueError, "sliding_window must exceed 1"):
        TtCompressionCache(4, 1)
    with expect_error(ValueError, "unique and non-empty"):
        TtCompressionCache(4, 128, names=("compressor", "compressor"))
