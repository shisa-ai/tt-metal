"""Lightning Indexer contracts, pinned against the reference indexer at the real config.

The indexer decides *which* compressed entries 21 of 43 layers ever see, so a wrong theta or
a wrong rope width is not a small numerical difference — it silently retrieves different
context. Parity here is on the retrieved indices as integer tensors, not on approximate
scores.

The reference exposes no seam for its intermediates, so ``indexer.scorer`` is wrapped in a
recording module that captures ``(q, compressed_kv, hidden_states)`` and forwards unchanged.
That makes the comparison three-way: keys, queries, and the final selection.
"""

import pytest
import torch
from transformers import AutoConfig

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer
from models.demos.deepseek_v3_d_p.tt.v4_indexer import TtIndexer, causal_entry_threshold, index_scores, select_top_k

CHECKPOINT = "/home/ubuntu/.cache/huggingface/hub/" "models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots"

RATE = 4
INDEX_HEAD_DIM = 128
INDEX_N_HEADS = 64
ROPE_DIM = 64
Q_LORA = 1024
HIDDEN = 4096


def checkpoint_config():
    import glob
    import os

    snapshots = sorted(glob.glob(os.path.join(CHECKPOINT, "*")))
    if not snapshots:
        pytest.skip("V4-Flash checkpoint config not present")
    return AutoConfig.from_pretrained(snapshots[-1], trust_remote_code=True)


class RecordingScorer(torch.nn.Module):
    """Capture the scorer's operands, forward them untouched."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = []

    def forward(self, q, compressed_kv, hidden_states):
        self.calls.append((q.detach().clone(), compressed_kv.detach().clone(), hidden_states.detach().clone()))
        return self.inner(q, compressed_kv, hidden_states)


def filled_reference(seed=21):
    cfg = checkpoint_config()
    torch.manual_seed(seed)
    idx = DeepseekV4Indexer(cfg).to(torch.float32).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in idx.named_parameters():
            if p.dtype is not torch.float32:
                continue
            if name.endswith("weight") and p.ndim == 1:
                p.normal_(1.0, 0.02, generator=g)
            elif name.endswith("position_bias"):
                p.normal_(0.0, 0.05, generator=g)
            else:
                p.normal_(0.0, (1.0 / p.shape[-1]) ** 0.5, generator=g)
    rec = RecordingScorer(idx.scorer)
    idx.scorer = rec
    return cfg, idx, rec


def port_weights(idx):
    return {
        "kv_proj": idx.kv_proj.weight.detach().clone(),
        "gate_proj": idx.gate_proj.weight.detach().clone(),
        "position_bias": idx.position_bias.detach().clone(),
        "kv_norm_weight": idx.kv_norm.weight.detach().clone(),
        "q_b_proj": idx.q_b_proj.weight.detach().clone(),
        "weights_proj": idx.scorer.inner.weights_proj.weight.detach().clone(),
    }


def inputs(batch=1, length=24, seed=4):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(batch, length, HIDDEN, generator=g, dtype=torch.float32)
    q_residual = torch.randn(batch, length, Q_LORA, generator=g, dtype=torch.float32)
    pos = torch.arange(length).unsqueeze(0).expand(batch, -1).contiguous()
    return hs, q_residual, pos


# -------------------------------------------------------------------- widths


def test_rope_width_is_global_to_the_group_not_the_index_head_dim():
    """64 from config.head_dim * 0.125, not int(index_head_dim * 0.125) == 16."""
    cfg, idx, _rec = filled_reference()
    ours = TtIndexer(cfg, port_weights(idx))
    assert ours.head_dim == INDEX_HEAD_DIM and ours.num_heads == INDEX_N_HEADS
    assert ours.rope_dim == ROPE_DIM, ours.rope_dim
    assert ours.rope_params.rope_type == "yarn" and ours.rope_params.rope_theta == 160000.0
    assert ours.index_topk == 512 and ours.compress_rate == RATE


def test_queries_and_index_keys_match_the_reference():
    cfg, idx, rec = filled_reference()
    ours = TtIndexer(cfg, port_weights(idx))
    hs, q_residual, pos = inputs()
    with torch.no_grad():
        ref_out = idx(hs, q_residual, pos, None, 0)
    assert len(rec.calls) == 1
    ref_q, ref_ckv, _hs = rec.calls[0]

    our_keys = ours.compress_keys(hs)
    our_q = ours.queries(q_residual, pos)
    assert our_q.shape == ref_q.shape, (our_q.shape, ref_q.shape)
    assert our_keys.shape == ref_ckv.shape, (our_keys.shape, ref_ckv.shape)
    assert torch.allclose(our_q, ref_q, rtol=1e-5, atol=1e-6), f"q max diff {(our_q - ref_q).abs().max():.3e}"
    assert torch.allclose(
        our_keys, ref_ckv, rtol=1e-5, atol=1e-6
    ), f"keys max diff {(our_keys - ref_ckv).abs().max():.3e}"
    assert ref_out.shape[-1] == min(ours.index_topk, ref_ckv.shape[1])


def test_selected_indices_match_the_reference_exactly():
    cfg, idx, _rec = filled_reference()
    ours = TtIndexer(cfg, port_weights(idx))
    for length in (8, 24, 40):
        hs, q_residual, pos = inputs(length=length, seed=length)
        with torch.no_grad():
            ref = idx(hs, q_residual, pos, None, 0)
            ours_idx, _scores, _keys = ours.forward(hs, q_residual, pos)
        assert ours_idx.shape == ref.shape, (length, ours_idx.shape, ref.shape)
        assert torch.equal(ours_idx, ref), (length, "first mismatch", (ours_idx != ref).nonzero()[:4].tolist())


def test_scores_match_the_reference_scorer():
    cfg, idx, rec = filled_reference()
    ours = TtIndexer(cfg, port_weights(idx))
    hs, q_residual, pos = inputs()
    with torch.no_grad():
        idx(hs, q_residual, pos, None, 0)
    ref_q, ref_ckv, _ = rec.calls[0]
    ref_scores = idx.scorer.inner(ref_q, ref_ckv, hs)
    our_scores = ours.score(ours.queries(q_residual, pos), ours.compress_keys(hs), hs)
    assert torch.allclose(
        our_scores, ref_scores, rtol=1e-5, atol=1e-6
    ), f"max {(our_scores - ref_scores).abs().max():.3e}"


# ------------------------------------------------------------- causal selection


def test_early_queries_get_the_sentinel_not_a_future_entry():
    """Rate 4: query 0..2 have fewer ready entries than top-k, and must return -1 there."""
    pos = torch.tensor([[0, 1, 2, 3, 7, 8, 100]])
    scores = torch.randn(1, 7, 32)
    picked = select_top_k(scores, pos, RATE, index_topk=8)
    assert picked.shape == (1, 7, 8), picked.shape
    threshold = causal_entry_threshold(pos, RATE)
    for col, token in enumerate(pos[0].tolist()):
        limit = int(threshold[0, col])
        for value in picked[0, col].tolist():
            assert value == -1 or value < limit, (token, value, limit)
    assert (picked[0, 0] == -1).all(), "query 0 has no ready entry at all"
    assert (picked[0, 2] == -1).all(), "query 2 still has none (3 // 4 == 0)"
    # Token 7 -> threshold 2, so entries 0 and 1 are eligible and the remaining 6 slots of
    # the k=8 request must come back as sentinels rather than as repeated or future entries.
    ready = [v for v in picked[0, 4].tolist() if v != -1]
    assert sorted(ready) == [0, 1], picked[0, 4].tolist()
    assert sum(v == -1 for v in picked[0, 4].tolist()) == 6


def test_no_entry_can_be_selected_twice_or_beyond_the_threshold():
    pos = torch.tensor([[63, 64, 65]])
    scores = torch.full((1, 3, 32), 1.0)  # all ties: top-k order is arbitrary
    picked = select_top_k(scores, pos, RATE, index_topk=16)
    for col in range(3):
        valid = [v for v in picked[0, col].tolist() if v != -1]
        assert len(valid) == len(set(valid)), "duplicate entries would double-count context"
        limit = int(causal_entry_threshold(pos, RATE)[0, col])
        assert all(v < limit for v in valid), (col, valid, limit)
        assert len(valid) == min(16, limit), "exactly the available entries are kept"


def test_top_k_is_clamped_to_available_entries():
    pos = torch.arange(16).unsqueeze(0)
    scores = torch.randn(1, 16, 4)
    picked = select_top_k(scores, pos, RATE, index_topk=512)
    assert picked.shape == (1, 16, 4), picked.shape


def test_zero_entries_returns_an_empty_index_axis():
    pos = torch.arange(5).unsqueeze(0)
    picked = select_top_k(torch.randn(1, 5, 0), pos, RATE, index_topk=512)
    assert picked.shape == (1, 5, 0), picked.shape


# ---------------------------------------------------------------- scoring math


def test_scoring_is_relu_weighted_head_sum():
    """Hand-written: ReLU before scaling, weights scaled by heads**-0.5, summed over heads."""
    b, s, h, d, t = 1, 2, 3, 4, 5
    torch.manual_seed(1)
    q = torch.randn(b, s, h, d)
    ckv = torch.randn(b, t, d)
    w = torch.randn(b, s, h)
    got = index_scores(q, ckv, w, d**-0.5, h**-0.5)
    manual = torch.zeros(b, s, t)
    for bi in range(b):
        for si in range(s):
            for ti in range(t):
                acc = 0.0
                for hi in range(h):
                    dot = float((q[bi, si, hi] * ckv[bi, ti]).sum())
                    acc += max(dot, 0.0) * (d**-0.5) * float(w[bi, si, hi]) * (h**-0.5)
                manual[bi, si, ti] = acc
    assert torch.allclose(got, manual, rtol=1e-4, atol=1e-6), (got - manual).abs().max()

    # ReLU's actual consequence: an anti-correlated query contributes *exactly zero*.
    # (The score itself stays signed, because the per-head weights are signed -- asserting
    # non-negativity overall would be a math error, and was the first version of this test.)
    anti_q = torch.zeros(b, s, h, d)
    anti_q[..., 0] = -1.0
    aligned_kv = torch.zeros(b, t, d)
    aligned_kv[..., 0] = 1.0
    positive_weights = torch.rand(b, s, h)
    anti = index_scores(anti_q, aligned_kv, positive_weights, d**-0.5, h**-0.5)
    assert torch.equal(anti, torch.zeros_like(anti)), "negative dots must be clamped, not summed"
    aligned = index_scores(-anti_q, aligned_kv, positive_weights, d**-0.5, h**-0.5)
    assert torch.all(aligned > 0.0), "aligned dots must score positive"


def test_shared_theta_is_enforced_by_one_table():
    """Keys and queries come from the same table, so the thetas cannot diverge."""
    cfg, idx, _rec = filled_reference()
    ours = TtIndexer(cfg, port_weights(idx))
    key_cos, _ = ours.rope_for_entries(2, 0)
    q_cos, _ = ours.rope_for_positions(torch.tensor([[0, 4]]))
    assert torch.equal(key_cos[0, 0], q_cos[0, 0]), "entry 0 sits at position 0"
    assert torch.equal(key_cos[0, 1], q_cos[0, 1]), "entry 1 sits at position rate"
    assert not torch.equal(key_cos[0, 0], key_cos[0, 1]), "positions must differ"


def test_table_overrun_raises_on_both_paths(expect_error):
    cfg, idx, _rec = filled_reference()
    ours = TtIndexer(cfg, port_weights(idx), max_windows=2)
    with expect_error(ValueError, "exceeds the precomputed table"):
        ours.rope_for_entries(3, 0)
    with expect_error(ValueError, "exceeds the precomputed table"):
        ours.rope_for_positions(torch.tensor([[0, 2 * RATE]]))


def test_shape_contracts_are_enforced(expect_error):
    pos = torch.arange(4)
    with expect_error(ValueError, r"position_ids must be \[B, S\]"):
        causal_entry_threshold(pos, RATE)
    with expect_error(ValueError, "scores must be"):
        select_top_k(torch.randn(4, 4), torch.arange(4).unsqueeze(0), RATE, 2)
    with expect_error(ValueError, "q must be"):
        index_scores(torch.randn(2, 3, 4), torch.randn(1, 5, 4), torch.randn(1, 2, 3), 1.0, 1.0)
    with expect_error(ValueError, "compressed_kv must be"):
        index_scores(torch.randn(1, 2, 3, 4), torch.randn(5, 4), torch.randn(1, 2, 3), 1.0, 1.0)


def test_threshold_formula_matches_the_causal_rule():
    pos = torch.tensor([[0, 3, 4, 7, 8]])
    assert causal_entry_threshold(pos, RATE).tolist() == [[0, 1, 1, 2, 2]]
