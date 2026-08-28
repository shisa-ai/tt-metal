# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Autoregressive decode-loop contract at the model level.

This is the property our decode loop must satisfy for "inference" to mean anything:
**greedy single-token generation must equal teacher-forced prefill at every step.**
Per-module contracts exist for attention (cc48c4) and compression (f6d531); this is the
whole-model version, over the real substituted tiny schedule — HCA x3 + CSA with
`hash_moe` on layer 0 and top-k `moe` elsewhere, so hash routing, learned routing,
partial compression and mHC all sit inside the loop being checked.

Why teacher-forcing is the right oracle: it needs no cache at all. Any disagreement
between stepping and full prefill is therefore attributable to state carried across calls
— the sliding window, the compressor's partial windows and entry offsets, cache
positions, or the rotary position — which is exactly the failure surface of a ported
decode loop. Parity here does not prove device numerics; it proves the *contract* the
device loop has to hit, with a deterministic golden token sequence keyed to the oracle's
weight fingerprint.

The whole suite runs on the fingerprinted oracle (explicit weights, worklog 88604c), so
the golden is reproducible rather than dependent on HF init, which is not reproducible
for standalone sublayers.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs
from models.demos.deepseek_v3_d_p.tt.v4_oracle import build_reference_oracle

PROMPT_LEN = 64
STEPS = 6


@pytest.fixture(scope="module")
def args():
    return V4ModelArgs.tiny(4)


def greedy(model, cfg, prompt: torch.Tensor, steps: int):
    """Greedy decoding with a live cache. Returns (tokens, per-step last logits)."""
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        out = model(prompt, past_key_values=cache, use_cache=True)
        token = int(out.logits[0, -1].argmax())
        tokens, step_logits = [token], [out.logits[0, -1].float()]
        for _ in range(steps - 1):
            nxt = torch.tensor([[token]], device=prompt.device)
            out = model(nxt, past_key_values=cache, use_cache=True)
            token = int(out.logits[0, -1].argmax())
            tokens.append(token)
            step_logits.append(out.logits[0, -1].float())
    return tokens, step_logits


def teacher_forced(model, prompt: torch.Tensor, generated: list[int]) -> torch.Tensor:
    """One cacheless forward over prompt + already-generated tokens; last-position logits."""
    seq = prompt.clone()
    if generated:
        extra = torch.tensor([generated], device=prompt.device, dtype=seq.dtype)
        seq = torch.cat([seq, extra], dim=1)
    with torch.no_grad():
        return model(seq).logits[0, -1].float()


def test_greedy_steps_match_teacher_forced_prefill_at_every_step(args):
    model, fingerprint = build_reference_oracle(args, seed=0)
    cfg = model.config
    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), generator=torch.Generator().manual_seed(9))

    tokens, step_logits = greedy(model, cfg, prompt, STEPS)
    assert len(tokens) == STEPS
    assert all(0 <= t < cfg.vocab_size for t in tokens)

    for k, step in enumerate(step_logits):
        tf = teacher_forced(model, prompt, tokens[:k])
        assert int(tf.argmax()) == tokens[k], (
            f"step {k}: teacher-forced argmax {int(tf.argmax())} != greedy token {tokens[k]} "
            "— cache/state carried across calls disagrees with a cacheless forward"
        )
        err = float((tf - step).abs().max())
        scale = float(tf.abs().max())
        assert err / scale < 1e-5, (
            f"step {k}: greedy vs teacher-forced max abs {err:.3e} vs scale {scale:.3e} "
            f"({err / scale:.3e} relative)"
        )


def test_positions_actually_advance_across_steps(args):
    """The regression this suite exists for.

    If the decode position fails to advance — a cache_position or rotary offset bug — every
    step produces an identical logit vector and generation silently repeats one token.
    Argmax-only checks can pass for a while on such a model, so the logits themselves must
    be shown to move.
    """
    model, _ = build_reference_oracle(args, seed=0)
    cfg = model.config
    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), generator=torch.Generator().manual_seed(9))
    _, step_logits = greedy(model, cfg, prompt, STEPS)

    assert all(torch.isfinite(v).all() for v in step_logits), "non-finite logits during decode"
    first = step_logits[0]
    moved = [float((v - first).abs().max()) / float(first.abs().max()) for v in step_logits[1:]]
    assert max(moved) > 1e-3, (
        f"per-step logits are effectively identical (max relative change {max(moved):.3e}); "
        "the decode position is not advancing"
    )


def test_golden_sequence_is_reproducible_from_the_weight_fingerprint(args):
    """Two builds at the same fingerprint must generate the same tokens.

    Without this, a golden token sequence is not usable as a parity target — it would be
    one sample from a random oracle rather than a fixed expectation.
    """
    prompt_tokens = None
    for seed in (0, 0):
        model, fingerprint = build_reference_oracle(args, seed=seed)
        cfg = model.config
        prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), generator=torch.Generator().manual_seed(11))
        tokens, _ = greedy(model, cfg, prompt, STEPS)
        if prompt_tokens is None:
            prompt_tokens, first_fingerprint = tokens, fingerprint
            assert fingerprint, "oracle must report a weight fingerprint"
        else:
            assert fingerprint == first_fingerprint, "same seed must reproduce the fingerprint"
            assert tokens == prompt_tokens, (
                f"same fingerprint produced different tokens: {tokens} vs {prompt_tokens}"
            )


def test_decode_loop_survives_a_longer_prompt(args):
    """Compression windows must close mid-decode, not just during the prompt.

    With HCA's real rate of 128, a 200-token prompt plus steps crosses a window boundary
    while stepping, so the compressor emits its first entry inside the decode loop rather
    than during the prompt — the ordering case most likely to be wrong in a ported cache.
    """
    model, _ = build_reference_oracle(args, seed=0)
    cfg = model.config
    rate = cfg.compress_rates["heavily_compressed_attention"]
    prompt_len = rate + 40  # one closed window plus slack
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), generator=torch.Generator().manual_seed(13))

    tokens, step_logits = greedy(model, cfg, prompt, 4)
    for k, step in enumerate(step_logits):
        tf = teacher_forced(model, prompt, tokens[:k])
        assert int(tf.argmax()) == tokens[k], f"long-prompt step {k}: argmax diverged"
        err = float((tf - step).abs().max())
        assert err / float(tf.abs().max()) < 1e-5, f"long-prompt step {k}: logits diverged ({err:.3e})"
