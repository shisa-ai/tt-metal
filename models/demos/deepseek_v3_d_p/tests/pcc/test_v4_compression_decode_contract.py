# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""HCA/CSA stateful compression must agree between one-shot and incremental decode.

This is the highest-risk piece of the port and it is the one thing upstream never
exercises: `TtHCA`/`TtIndexer` are never instantiated at 1×1 (`TtCSA` does not
exists at all — worklog 928348), and our pin's
whole V4 demo is disabled with `serialize_layernorms` unsupported (worklog 86ec0e).
The compressor is *stateful across calls* — it buffers partial windows, emits an
entry only when a window closes, and stamps each entry with a deterministic absolute
position so cross-call concatenation stays causally correct. A bug there produces
plausible output on a single prefill and wrong output during generation, i.e. exactly
the failure mode a prefill-only gate cannot see.

So the test is the contract our device cache must satisfy: **one-shot prefill and
incremental single-token decode must produce the same logits.** Measured agreement is
~5e-07 relative, with the compressor invoked once per decode step (asserted below —
without that, a vacuous test would pass by never compressing incrementally at all).

Also pins the compression *depth*, which feeds capacity math: one compressed entry per
`compress_rate` tokens, so HCA's rate-128 window is a 128x KV reduction.

Presets matter here: the tiny preset keeps HCA's real rate of **128** and overrides
only CSA to 4. At fewer than 128 tokens no HCA window ever closes, so a short test
would silently exercise nothing but CSA. This file uses 320 tokens for that reason.

One design constraint learned the hard way (worklog 974c2b): the parity assertion is
only meaningful when the CSA indexer's top-`k` selection is *deterministic*. The indexer
scores entries with `F.relu(...)`, which leaves a plateau of **exactly zero** scores; if
the `index_topk` boundary lands inside that plateau, `torch.topk` breaks the tie by
memory order, and the selected entry set can differ between two call chunkings even
though every stored tensor agrees to ~2e-7. Measured on the random-weight oracle at
`index_topk=16`: 34 of 320 positions select a different *set*, the indexer scores differ
by only ~1e-9, and the score gap at the selection boundary is *exactly 0*. That is index
selection, not cache state — so this file pins parity with a **non-binding** `index_topk`
(the checkpoint's own 512) and leaves the tie behaviour to its own test below.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs
from models.demos.deepseek_v3_d_p.tt.v4_oracle import build_reference_oracle

SEQ = 320  # > 128 so at least two HCA windows close
PREFIX = 300


@pytest.fixture(scope="module")
def args():
    # Released V4-Flash schedule for 4 layers is [sliding, sliding, CSA, HCA] (worklog
    # 974c2b); the old preset wrongly used the config class's V4-Pro default rule.
    # `index_topk` = the checkpoint's 512 > the 80 CSA entries a 320-token sequence
    # produces, so top-k selects *every* causally valid entry and cannot reshuffle.
    return V4ModelArgs.tiny(4, index_topk=V4ModelArgs().index_topk)


@pytest.fixture(scope="module")
def model_and_cfg(args):
    model, _ = build_reference_oracle(args, seed=0)
    return model, model.config


def _instrument(model):
    """Count compressor invocations per layer.

    An nn.Module attribute assignment of a plain callable falls through to
    ``object.__setattr__``, so this shadows the bound method without touching the
    class — no reference source is modified.
    """
    calls: dict[str, int] = {}

    def reset():
        calls.clear()

    for i, layer in enumerate(model.model.layers):
        comp = getattr(layer.self_attn, "compressor", None)
        if comp is None:
            continue
        original = comp.forward
        key = f"{type(comp).__name__}@{i}"

        def counting(key=key, original=original):
            def run(*a, **k):
                calls[key] = calls.get(key, 0) + 1
                return original(*a, **k)

            return run

        comp.forward = counting()
    return calls, reset


def test_hca_incremental_decode_matches_one_shot(model_and_cfg):
    model, cfg = model_and_cfg
    ids = torch.randint(0, cfg.vocab_size, (1, SEQ), generator=torch.Generator().manual_seed(4))
    calls, reset = _instrument(model)

    with torch.no_grad():
        reset()
        one_shot = model(ids).logits[0, -1].float()
        one_shot_calls = dict(calls)

        reset()
        cache = DynamicCache(config=cfg)
        model(ids[:, :PREFIX], past_key_values=cache, use_cache=True)
        reset()
        for step in range(PREFIX, SEQ):
            last = model(ids[:, step : step + 1], past_key_values=cache, use_cache=True).logits[0, -1].float()
        step_calls = dict(calls)

    assert one_shot_calls, "no compressor ever ran — the fixture is not an HCA/CSA model"
    # Every decode step must have invoked each compressor; otherwise the agreement
    # below would only prove that two identical no-ops agree.
    assert all(
        v == SEQ - PREFIX for v in step_calls.values()
    ), f"expected {SEQ - PREFIX} incremental compressor calls each, got {step_calls}"

    err = float((one_shot - last).abs().max())
    rel = err / float(one_shot.abs().max())
    assert rel < 1e-5, f"one-shot vs incremental diverged: max abs {err:.3e}, rel {rel:.3e}"
    assert int(one_shot.argmax()) == int(last.argmax()), "argmax diverged"


def test_compression_depth_is_one_entry_per_rate_tokens(model_and_cfg):
    """KV reduction factor, needed for capacity/placement decisions."""
    model, cfg = model_and_cfg
    ids = torch.randint(0, cfg.vocab_size, (1, SEQ), generator=torch.Generator().manual_seed(5))

    with torch.no_grad():
        cache = DynamicCache(config=cfg)
        model(ids, past_key_values=cache, use_cache=True)

    rates = cfg.compress_rates
    checked = 0
    for i, layer in enumerate(cache.layers):
        # `compressed_kv` is a dict keyed by component name — {"compressor": …} on
        # HCA layers, {"compressor": …, "indexer": …} on the CSA layer — not a
        # tensor. Indexer depth equals the window count, which is the quantity that
        # actually sets our device cache allocation.
        state = getattr(layer, "compressed_kv", None)
        if not state:
            continue
        layer_type = cfg.layer_types[i]
        rate = rates[layer_type]
        expected = SEQ // rate
        for component, tensor in state.items():
            entries = int(tensor.shape[-2])
            assert entries == expected, (
                f"layer {i} ({layer_type}, rate {rate}) component {component}: "
                f"{entries} entries, expected {expected} — window-closing semantics "
                "differ from floor(seq/rate)"
            )
            assert torch.isfinite(tensor).all(), f"layer {i} {component} has non-finite entries"
        checked += 1
    assert checked, "no layer reported compressed_kv; depth cannot be verified"


def test_hca_rate_is_the_real_128_not_a_toy_value(args):
    """Guards the SEQ choice above: at <128 tokens no HCA window closes at all."""
    assert args.compress_rates["heavily_compressed_attention"] == 128, (
        "HCA rate changed; this file's SEQ must stay above one HCA window or the " "test silently exercises only CSA"
    )
    assert SEQ > args.compress_rates["heavily_compressed_attention"], "SEQ too short"


def test_binding_topk_reshuffles_only_on_exact_ties():
    """The indexer's discrete top-k is the *only* chunk-boundary sensitivity.

    With `index_topk` binding (the tiny preset's 16 < the 80 CSA entries), the selected
    entry set can differ between one-shot prefill and chunked decode. This test does not
    assert that it does — torch may break ties differently across versions — it asserts
    the attribution: any position whose selected set differs MUST have an *exactly tied*
    score at the causal top-k boundary, and the underlying state must agree. If this ever
    fails, a real chunk-dependence has entered the compressor/cache, not a tie.
    """
    args = V4ModelArgs.tiny(4)  # tiny keeps index_topk=16, which binds at SEQ=320
    model, _ = build_reference_oracle(args, seed=0)
    model.eval()
    cfg = model.config
    csa_layers = [i for i, t in enumerate(cfg.layer_types) if t == "compressed_sparse_attention"]
    assert csa_layers, "fixture has no CSA layer, so indexer selection cannot be observed"
    layer = csa_layers[0]
    indexer = model.model.layers[layer].self_attn.compressor.indexer
    rate = cfg.compress_rates["compressed_sparse_attention"]
    top_k = cfg.index_topk
    ids = torch.randint(0, cfg.vocab_size, (1, SEQ), generator=torch.Generator().manual_seed(4))

    def run(chunked: bool):
        rows: dict[int, torch.Tensor] = {}
        picks: list[torch.Tensor] = []
        state = {"pos": 0}

        def score_hook(_mod, _inp, out):
            out = out.detach().float().cpu()
            for j in range(out.shape[1]):
                rows[state["pos"] + j] = out[0, j].clone()
            state["pos"] += out.shape[1]
            return out

        h1 = indexer.register_forward_hook(lambda _m, _i, o: picks.append(o.detach().cpu()))
        h2 = indexer.scorer.register_forward_hook(score_hook)
        cache = DynamicCache(config=cfg)
        with torch.no_grad():
            if not chunked:
                model(ids, past_key_values=cache, use_cache=True)
            else:
                model(ids[:, :PREFIX], past_key_values=cache, use_cache=True)
                for step in range(PREFIX, SEQ):
                    model(ids[:, step : step + 1], past_key_values=cache, use_cache=True)
        h1.remove()
        h2.remove()
        return rows, torch.cat(picks, dim=1)[0], cache

    one_rows, one_picks, one_cache = run(False)
    inc_rows, inc_picks, inc_cache = run(True)

    # 1. The *state* must be chunk-invariant: that is what our device cache must reproduce.
    for component, mine in one_cache.layers[layer].compressed_kv.items():
        theirs = inc_cache.layers[layer].compressed_kv[component]
        assert mine.shape == theirs.shape, f"{component}: {tuple(mine.shape)} vs {tuple(theirs.shape)}"
        delta = float((mine.float() - theirs.float()).abs().max())
        assert delta < 1e-5, f"{component} compressed state differs across chunkings: {delta:.3e}"

    # 2. Indexer scores must be chunk-invariant on the causally valid prefix.
    ties_seen = 0
    unexplained: list[int] = []
    for pos in range(SEQ):
        ready = (pos + 1) // rate  # entries this query may see at all
        if ready < 2:
            continue
        a, b = one_rows[pos][:ready], inc_rows[pos][:ready]
        score_delta = float((a - b).abs().max())
        assert score_delta < 1e-6, f"pos {pos}: indexer scores diverge by {score_delta:.3e}"
        if set(one_picks[pos].tolist()) == set(inc_picks[pos].tolist()):
            continue
        # Selection differed — it is only excusable if the boundary was an exact tie.
        srt = a.sort(descending=True).values
        kth = min(top_k, ready) - 1
        if kth + 1 >= len(srt):
            unexplained.append(pos)  # every ready entry fits: the sets cannot legitimately differ
            continue
        gap = abs(float(srt[kth]) - float(srt[kth + 1]))
        if gap > 0.0:
            unexplained.append(pos)
        else:
            ties_seen += 1

    assert not unexplained, f"top-k sets differ with no tie to explain it at positions {unexplained[:10]}"
    # Keep the test honest: the ReLU score plateau that causes the ties must really exist here.
    plateau = sum(1 for pos in range(SEQ) if float((one_rows[pos][: (pos + 1) // rate] == 0).any()))
    assert plateau > 0, "no exact-zero indexer scores in this fixture — the tie premise no longer holds"
