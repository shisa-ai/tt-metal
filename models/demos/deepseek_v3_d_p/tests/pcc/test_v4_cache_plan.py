# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The per-layer allocation plan must match the released schedule, layer by layer.

`cache_plan()` is the seam between the config and everything that allocates: the assembly, the
KV/compression cache sizing, and the capacity math that decides whether 32 ASICs are enough.
It is therefore the place where the worklog 54d30a preset bug would finally become a wrong
number rather than a wrong string — the *counts* (2 sliding / 21 CSA / 20 HCA) survived that
bug, but the **positions** did not, and allocation is positional.

The plan is checked by consuming it, not by reading it: each entry is used to construct the
cache class it describes. A plan that is descriptive but unusable — CSA without an indexer,
HCA asking for CSA's overlap state, a sliding layer carrying a compressor rate — fails at
construction time, which is exactly when the assembly would have failed.
"""

from __future__ import annotations

import pytest

from models.demos.deepseek_v3_d_p.tt.v4_cache import TtCompressionCache
from models.demos.deepseek_v3_d_p.tt.v4_model_config import CSA, HCA, SLIDING, V4ModelArgs


@pytest.fixture(scope="module")
def args():
    return V4ModelArgs()


def test_plan_covers_every_layer_in_the_released_schedule(args):
    plan = args.cache_plan()
    types = args.layer_types()
    assert len(plan) == args.num_hidden_layers == 43
    assert [p.index for p in plan] == list(range(43)), "plan must be one entry per layer, in order"
    assert [p.attention for p in plan] == types
    assert [p.mlp for p in plan] == args.mlp_layer_types()


def test_plan_pins_positions_not_just_counts(args):
    """Counts alone would have passed the preset bug; positions will not."""
    plan = args.cache_plan()
    by_index = {p.index: p for p in plan}
    assert (
        by_index[0].attention == SLIDING and by_index[1].attention == SLIDING
    ), "layers 0 and 1 are sliding: this is the shape the preset used to miss entirely"
    assert by_index[2].attention == CSA
    assert by_index[3].attention == HCA
    assert by_index[42].attention == CSA, "the checkpoint's final ratio is 4 (CSA), not 128"
    counts = {k: sum(1 for p in plan if p.attention == k) for k in (SLIDING, CSA, HCA)}
    assert counts == {SLIDING: 2, CSA: 21, HCA: 20}, counts


def test_producers_match_what_each_attention_class_owns(args):
    plan = args.cache_plan()
    for p in plan:
        if p.attention == SLIDING:
            assert p.producers == (), f"layer {p.index}: sliding layers have no compressor"
            assert p.compress_rate is None, f"layer {p.index}: no rate without a compressor"
            assert not p.overlap
        elif p.attention == CSA:
            assert p.producers == ("compressor", "indexer"), f"layer {p.index}: CSA needs the indexer"
            assert p.compress_rate == args.compress_rates[CSA] == 4
            assert p.overlap, "CSA's two-series windows carry a slice across the call boundary"
        else:
            assert p.producers == ("compressor",), f"layer {p.index}: HCA has no indexer (worklog 928348)"
            assert p.compress_rate == args.compress_rates[HCA] == 128
            assert not p.overlap, "HCA windows do not overlap"


def test_every_layer_of_the_plan_builds_its_cache(args):
    """Consume the plan: the plan is only real if allocation succeeds for all 43 layers."""
    built = []
    for p in args.cache_plan():
        cache = TtCompressionCache(
            compress_rate=p.compress_rate or 1,
            sliding_window=p.sliding_window,
            names=p.producers or ("sliding",),
            overlap=p.overlap,
        )
        assert cache.names == tuple(p.producers or ("sliding",))
        assert cache.overlap_enabled is p.overlap
        assert cache.sliding_window == args.sliding_window == 128, "window must be the checkpoint's"
        built.append(cache)
    assert len(built) == 43


def test_hca_cannot_borrow_the_csa_overlap_path_through_the_plan(args, expect_error):
    """The guard exists in the cache; the plan must not route anyone into it."""
    hca = [p for p in args.cache_plan() if p.attention == HCA][0]
    cache = TtCompressionCache(hca.compress_rate, hca.sliding_window, hca.producers, overlap=False)
    with expect_error(RuntimeError, "overlap"):
        cache.update_overlap_state("compressor", None, None, head_dim=1)
