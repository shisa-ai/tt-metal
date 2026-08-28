# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""V4 attention epilogue and grouped-O, checked against the reference on host.

Two pieces of V4 attention are small enough to verify here and dangerous enough to
be worth verifying before any device run, because in both cases a wrong
implementation produces a plausible tensor rather than an error:

1. **The attention sink.** Not "softmax, then add a sink weight". The reference
   concatenates the sink onto the logits, subtracts the max **of the combined set
   including the sink**, softmaxes over keys ∪ {sink}, and only then drops the sink
   column. So the sink sits in the softmax denominator and absorbs probability
   mass: attention output does not sum to 1 over the keys. Normalising over keys
   and mixing the sink afterwards gives a different number and looks identical in a
   shape dump.

2. **Grouped output projection.** `o_a_proj` splits heads into `o_groups`
   contiguous blocks and projects each independently. Group-major vs group-minor
   reshaping both produce the right shape and scramble head→group assignment
   silently.

Also recorded from reading, because it changes what to test: `eager_attention_forward`
accepts ``sliding_window`` and **ignores it** -- it only ever adds
``attention_mask``. So for ``sliding_attention`` layers the window comes entirely
from the mask the model builds, and a TTNN port that relies on an SDPA
``sliding_window`` argument while passing a mask that already encodes causality
will apply the constraint twice, or not at all, depending on the backend.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4GroupedLinear,
    eager_attention_forward,
)


# ---- 1. sink-augmented softmax ---------------------------------------------


def sink_attention(q, kv, scaling, sinks, attn_mask=None):
    """TTNN-oriented transcription: q,kv [B,H,S,D]; kv is used as BOTH key and value.

    Mirrors the reference epilogue exactly: scale, mask, concat sink, subtract the
    max over the *combined* row, softmax over keys+sink, drop sink column, then
    multiply by kv-as-value.
    """
    scores = torch.matmul(q, kv.transpose(2, 3)) * scaling
    if attn_mask is not None:
        scores = scores + attn_mask
    sink = sinks.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[2], -1)
    combined = torch.cat([scores, sink], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
    weights = probs[..., :-1]
    out = torch.matmul(weights, kv)
    return out.transpose(1, 2), weights


class _FakeModule:
    """Only ``sinks`` and ``num_key_value_groups`` are read by the reference path."""

    def __init__(self, sinks, groups):
        self.sinks = sinks
        self.num_key_value_groups = groups
        self.training = False


def test_sink_normalisation_matches_the_reference():
    b, h, s, d = 2, 4, 6, 8
    g = torch.Generator().manual_seed(0)
    q = torch.randn(b, h, s, d, generator=g)
    kv = torch.randn(b, 1, s, d, generator=g)  # single KV head, K == V
    sinks = torch.randn(h, generator=g) * 2.0  # large-ish so the sink can dominate
    scaling = d**-0.5
    mod = _FakeModule(sinks, h)

    ref_out, ref_w = eager_attention_forward(
        mod, q, kv, kv, attention_mask=None, scaling=scaling, dropout=0.0
    )
    ours_out, ours_w = sink_attention(q, kv, scaling, sinks)

    assert torch.allclose(ours_w, ref_w, rtol=1e-6, atol=1e-7)
    assert torch.allclose(ours_out, ref_out, rtol=1e-6, atol=1e-7)


def test_the_sink_actually_absorbs_probability_mass():
    """Prove this is not equivalent to key-only softmax, so the test above bites."""
    b, h, s, d = 1, 3, 5, 8
    g = torch.Generator().manual_seed(1)
    q = torch.randn(b, h, s, d, generator=g)
    kv = torch.randn(b, 1, s, d, generator=g)
    scaling = d**-0.5

    for sinks in (torch.full((h,), 6.0), torch.randn(h, generator=g) * 4.0):
        mod = _FakeModule(sinks, h)
        ref_out, ref_w = eager_attention_forward(mod, q, kv, kv, attention_mask=None, scaling=scaling, dropout=0.0)
        key_only = F.softmax(torch.matmul(q, kv.transpose(2, 3)) * scaling, dim=-1)
        assert not torch.allclose(ref_w, key_only, rtol=1e-4, atol=1e-6), (
            "sink had no effect on the weights -- the fixture is not exercising it"
        )
        # Rows must sum to less than 1: the sink keeps the remainder.
        assert float(ref_w.sum(dim=-1).max()) < 1.0 - 1e-6


def test_single_kv_head_is_reused_as_value():
    """K == V is a V4 contract; using a separate value tensor would be a silent model change."""
    b, h, s, d = 1, 4, 5, 8
    g = torch.Generator().manual_seed(2)
    q = torch.randn(b, h, s, d, generator=g)
    kv = torch.randn(b, 1, s, d, generator=g)
    other_v = torch.randn(b, 1, s, d, generator=g)
    sinks = torch.randn(h, generator=g)
    scaling = d**-0.5
    mod = _FakeModule(sinks, h)

    out_kv, _ = eager_attention_forward(mod, q, kv, kv, attention_mask=None, scaling=scaling, dropout=0.0)
    out_other, _ = eager_attention_forward(mod, q, kv, other_v, attention_mask=None, scaling=scaling, dropout=0.0)
    assert not torch.allclose(out_kv, out_other, rtol=1e-4, atol=1e-6), (
        "value tensor had no effect -- the fixture cannot detect a wrong V"
    )


# ---- 2. grouped output projection ------------------------------------------


def test_grouped_linear_matches_per_group_bmm():
    groups, rank_g, in_per_group, tokens = 4, 6, 8, 3
    g = torch.Generator().manual_seed(3)
    ref = DeepseekV4GroupedLinear(in_per_group, groups * rank_g, groups, bias=False).to(torch.float32).eval()
    # The reference is called as o_a_proj(attn_output.reshape(B, S, o_groups, -1)),
    # so the grouped axis is explicit and the last dim is one group's input width.
    x = torch.randn(1, tokens, groups, in_per_group, generator=g)

    with torch.no_grad():
        expected = ref(x)
        w = ref.weight  # [groups*rank_g, in_per_group]
        outs = [x[:, :, gi, :] @ w[gi * rank_g : (gi + 1) * rank_g].T for gi in range(groups)]
        ours = torch.stack(outs, dim=2)

    assert ours.shape == expected.shape, (tuple(ours.shape), tuple(expected.shape))
    assert torch.allclose(ours, expected, rtol=1e-6, atol=1e-7)


def test_group_ordering_is_discriminable():
    """A group-order mix-up keeps the shape and scrambles head->group assignment."""
    groups, rank_g, in_per_group, tokens = 4, 6, 8, 3
    g = torch.Generator().manual_seed(4)
    ref = DeepseekV4GroupedLinear(in_per_group, groups * rank_g, groups, bias=False).to(torch.float32).eval()
    x = torch.randn(1, tokens, groups, in_per_group, generator=g)

    with torch.no_grad():
        expected = ref(x)
        w = ref.weight
        outs = [
            x[:, :, gi, :] @ w[(groups - 1 - gi) * rank_g : (groups - gi) * rank_g].T for gi in range(groups)
        ]
        reversed_groups = torch.stack(outs, dim=2)

    assert reversed_groups.shape == expected.shape
    assert not torch.allclose(reversed_groups, expected, rtol=1e-3, atol=1e-5), (
        "reversed group order is indistinguishable -- fixture weights are too "
        "symmetric, so this suite would not catch a group-ordering bug"
    )
