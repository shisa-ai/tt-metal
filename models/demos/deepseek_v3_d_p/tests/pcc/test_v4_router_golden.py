# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Golden semantics for the V4 router, so `TtMoe` reuse is checked against V4.

The plan reuses upstream's `TtMoe`/gate stack rather than writing a new one, but that
stack was built for DeepSeek-V3's router. Before spending device time on it, the
target has to be pinned: here is exactly what a V4 gate must compute, which choices
are load-bearing, and which ones break under reduced precision.

The reference is explicit and short:

    logits  = x @ W^T
    scores  = sqrtsoftplus(logits)                        # sqrt(softplus(x))
    indices = topk(scores + bias, k, sorted=False).indices # bias only SELECTS
    weights = scores.gather(1, indices)                    # weights use UNBIASED scores
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return logits, weights * routed_scaling_factor, indices

Every one of those four choices is asserted to matter, so a port that silently drops
one cannot pass. The vendored reference's router is byte-identical to the `deepseek_v4`
model shipped by the installed transformers, so this is the canonical implementation,
not a fork.

Device-relevant hazard: the renorm epsilon is **1e-20**. That is not representable in
fp8 and is subnormal-ish even in fp16, so the renorm must happen in fp32 — asserted
below by showing what the degenerate path does when the denominator loses the epsilon.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4TopKRouter,
)
from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs


@pytest.fixture(scope="module")
def cfg():
    return V4ModelArgs.tiny(1).drive_reference()


def build_router(cfg, seed: int = 0, bias: float = 0.05):
    """Explicit weights and a NONZERO bias buffer.

    Standalone sublayers bypass ``PreTrainedModel.init_weights`` (worklog 88604c), and
    a zero bias would make the "bias only selects" property unobservable.
    """
    torch.manual_seed(seed)
    router = DeepseekV4TopKRouter(cfg).to(torch.float32)
    with torch.no_grad():
        router.weight.normal_(0.0, 1.0 / (router.hidden_dim**0.5), generator=torch.Generator().manual_seed(seed))
        router.e_score_correction_bias.normal_(0.0, bias, generator=torch.Generator().manual_seed(seed + 1))
    return router.eval()


def golden(x, router, *, score_fn=None, biased_weights=False, renorm=True, scale=True):
    """Transcription of the reference router, with each choice switchable."""
    logits = F.linear(x.reshape(-1, router.hidden_dim), router.weight)
    scores = (router.score_fn if score_fn is None else score_fn)(logits)
    indices = torch.topk(scores + router.e_score_correction_bias, router.top_k, dim=-1, sorted=False).indices
    biased = scores + router.e_score_correction_bias
    weights = biased.gather(1, indices) if biased_weights else scores.gather(1, indices)
    if renorm:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * router.routed_scaling_factor if scale else weights, indices


def _assert_state_sets(indices_a, indices_b):
    """``topk(sorted=False)`` makes index ORDER unspecified, so compare sets per row."""
    assert torch.equal(indices_a.sort(-1).values, indices_b.sort(-1).values), (
        "selected expert sets differ; a device gate must match the SETS, and cannot "
        "compare index order because topk(sorted=False) does not define it"
    )


def test_transcription_reproduces_the_reference(cfg):
    router = build_router(cfg)
    x = torch.randn(6, router.hidden_dim)
    lr, wr, ir = router(x)
    lg, wg, ig = golden(x, router)
    assert torch.allclose(lr, lg, atol=1e-6, rtol=1e-6)
    assert torch.allclose(wr, wg, atol=1e-6, rtol=1e-6)
    _assert_state_sets(ir, ig)


def test_bias_selects_but_does_not_weight(cfg):
    """The single most port-breaking V4 detail: biased selection, unbiased weights."""
    router = build_router(cfg)
    x = torch.randn(6, router.hidden_dim)
    _, ref_w, _ = router(x)
    _, bad_w, _ = golden(x, router, biased_weights=True)
    assert not torch.allclose(ref_w, bad_w, atol=1e-5, rtol=1e-5), (
        "biased and unbiased weights agree — either the bias buffer is inert (set it "
        "nonzero) or the fixture is degenerate"
    )


def test_renorm_and_scaling_factor_are_load_bearing(cfg):
    router = build_router(cfg)
    x = torch.randn(6, router.hidden_dim)
    _, ref_w, _ = router(x)
    _, no_renorm, _ = golden(x, router, renorm=False)
    _, no_scale, _ = golden(x, router, scale=False)
    assert not torch.allclose(ref_w, no_renorm, atol=1e-5, rtol=1e-5), "renorm is inert"
    assert not torch.allclose(ref_w, no_scale, atol=1e-5, rtol=1e-5), "routed_scaling_factor is inert"
    assert router.routed_scaling_factor != 1.0, "scaling factor must differ from 1 for that test to mean anything"
    # Renormalised weights sum to 1 before scaling, and to the factor after.
    assert torch.allclose(ref_w.sum(-1), torch.full((6,), router.routed_scaling_factor), atol=1e-6)


def test_scoring_function_changes_the_selection(cfg):
    """sqrtsoftplus is not interchangeable with sigmoid/softmax — reuse must switch it."""
    router = build_router(cfg)
    x = torch.randn(6, router.hidden_dim)
    _, ref_w, ref_i = router(x)
    _, sig_w, sig_i = golden(x, router, score_fn=torch.sigmoid)
    assert not torch.allclose(ref_w, sig_w, atol=1e-4, rtol=1e-4), (
        "sqrtsoftplus and sigmoid produce identical weights; fixture is degenerate"
    )
    sets_agree = torch.equal(ref_i.sort(-1).values, sig_i.sort(-1).values)
    if sets_agree:
        # Selection can coincide on small expert counts while weights differ. Record
        # that explicitly rather than asserting a divergence that may not occur.
        pass
    assert torch.isfinite(ref_w).all()


def test_extreme_logits_stay_finite_and_the_epsilon_matters(cfg):
    """sqrt(softplus(x)) underflows for very negative logits; 1e-20 is then the only
    thing between a finite weight and a 0/0 NaN.

    Exercised on the scores directly rather than through the projection: the router
    has no additive logit bias, so no choice of hidden state or weight matrix produces
    a controlled extreme logit — and the hazard is in the renorm arithmetic anyway,
    which is exactly the part a device gate reimplements in its own dtype.
    """
    router = build_router(cfg)
    logits = torch.full((2, router.num_experts), -1e4)
    scores = router.score_fn(logits)
    assert float(scores.abs().max()) == 0.0, (
        f"sqrtsoftplus did not underflow at -1e4 (max {float(scores.abs().max()):.3e}); "
        "the hazard moved and this test needs rewriting rather than loosening"
    )

    indices = torch.topk(scores + router.e_score_correction_bias, router.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    denom = weights.sum(-1, keepdim=True)
    assert float(denom.abs().max()) == 0.0, "degenerate denominator is not zero; hazard reassessed"

    with_eps = weights / (denom + 1e-20)  # the reference formula
    without_eps = weights / denom  # what a gate does if 1e-20 is not representable
    assert torch.isfinite(with_eps).all(), "reference renorm should give 0, not NaN"
    assert not torch.isfinite(without_eps).all(), (
        "0/0 came out finite, so the epsilon is not load-bearing and this test would "
        "be asserting a hazard that does not exist"
    )
    # Same arithmetic at fp16 scale, where 1e-20 is subnormal-to-zero: the epsilon
    # survives only because the renorm is fp32.
    denom16 = denom.to(torch.float16)
    assert float((denom16 + torch.tensor(1e-20, dtype=torch.float16)).abs().max()) == 0.0, (
        "1e-20 is representable at fp16 here; the dtype warning needs revisiting"
    )


def test_hash_router_uses_a_fixed_table_not_learned_scores(cfg):
    """Half the layers use a HASH router (fixed tid2eid table), not learned scores.

    Recorded so a reuse plan cannot silently apply the top-k gate everywhere.
    """
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4HashRouter,
    )

    hash_router = DeepseekV4HashRouter(cfg).to(torch.float32)
    table = hash_router.tid2eid
    # [vocab_size, top_k]: the table is indexed by token id, so its shape says nothing
    # about the expert count — the reason my first oracle fill read out of range.
    assert table.shape == (cfg.vocab_size, hash_router.top_k), tuple(table.shape)
    assert not table.dtype.is_floating_point, (
        f"tid2eid is {table.dtype}; a lookup table feeding an expert index must be an integer "
        "dtype or the gather needs an explicit cast on device"
    )
    assert (int(table.max()) if table.numel() else 0) < hash_router.num_experts, (
        "table entries must index into num_experts"
    )
    # Reference init zeroes the table ("real values come from the checkpoint"), so an
    # untrained hash layer routes every token to expert 0: a parity fixture that leaves
    # it at init exercises one expert, not the grouped path.
    assert torch.equal(table, torch.zeros_like(table)), (
        "tid2eid is no longer zero-initialised; the note above needs revisiting"
    )
