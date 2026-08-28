# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""HCA/CSA stateful compression must agree between one-shot and incremental decode.

This is the highest-risk piece of the port and it is the one thing upstream never
exercises: `TtHCA`/`TtCSA`/`TtIndexer` are never instantiated at 1×1, and our pin's
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
    return V4ModelArgs.tiny(4)  # HCA, HCA, HCA, CSA


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
    assert all(v == SEQ - PREFIX for v in step_calls.values()), (
        f"expected {SEQ - PREFIX} incremental compressor calls each, got {step_calls}"
    )

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
        "HCA rate changed; this file's SEQ must stay above one HCA window or the "
        "test silently exercises only CSA"
    )
    assert SEQ > args.compress_rates["heavily_compressed_attention"], "SEQ too short"
