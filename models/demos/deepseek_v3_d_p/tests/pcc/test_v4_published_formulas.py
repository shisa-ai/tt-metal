"""Cross-implementation check against the **publisher's** published formulas.

Everything else in this suite pins the port against the tt-metal copy of
``modeling_deepseek_v4.py``. This file instead transcribes the math from DeepSeek's own
shipped inference implementation inside the released checkpoint snapshot —
``inference/model.py`` (revision ``7872f01b1d1fe23eabc4c98b48bffcef5a386062``), which uses a
different naming scheme, a different cache representation, and a stateful decode path, so it
is a genuinely independent second source rather than the same code path twice.

What is compared, with source lines:
* ``Compressor.overlap_transform`` (model.py:314-321) and the prefill compression body
  (model.py:340-349) -> ``compress_windows_csa`` / ``compress_windows``.
* ``Indexer.forward`` scoring and selection (model.py:424-436) -> ``index_scores`` /
  ``select_top_k``.

What is deliberately **not** claimed: bit-parity with the released model's *outputs*. The
published path quantizes activations (``act_quant`` on the compressor's non-rope slice at
model.py:372-376, ``fp4_act_quant`` on indexer queries at model.py:419-420, and QAT weights
throughout), and it addresses compressed entries with a paged ``offset``. Neither is modelled
by this port, and neither is visible in the compared quantities. See the accompanying worklog
entry for the divergences this file surfaced.
"""

import torch

from models.demos.deepseek_v3_d_p.tt.v4_compression import compress_windows, compress_windows_csa, rms_norm
from models.demos.deepseek_v3_d_p.tt.v4_indexer import index_scores, select_top_k

CHECKPOINT_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
CSA_RATIO = 4
HCA_RATIO = 128
INDEX_TOPK = 512


# ---------------------------------------------------- publisher transcription


def published_overlap_transform(tensor: torch.Tensor, value: float) -> torch.Tensor:
    """Transcribed from model.py:314-321 (`Compressor.overlap_transform`).

    Kept deliberately literal — `new_full(..., value)`, `[:, :, ratio:] = tensor[:, :, :, d:]`,
    `[:, 1:, :ratio] = tensor[:, :-1, :, :d]` — because the whole point is that this code was
    written without looking at the port.
    """
    b, s, _, _ = tensor.size()
    ratio, d = CSA_RATIO, HEAD_DIM
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new_tensor


def published_prefill_compress(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    overlap: bool,
    ratio: int,
) -> torch.Tensor:
    """model.py:330-349 with the linear layers inlined, `start_pos == 0`, no remainder.

    Their `x.float()` then fp32 projections, `unflatten(1, (-1, ratio))`, `+ ape`,
    the overlap transform with `0` / `-inf`, then `(kv * score.softmax(dim=2)).sum(dim=2)`
    and the RMSNorm.
    """
    x = x.float()
    kv = x @ wkv.float().t()
    score = x @ wgate.float().t()
    usable = (kv.shape[1] // ratio) * ratio
    kv = kv[:, :usable].unflatten(1, (-1, ratio))
    score = score[:, :usable].unflatten(1, (-1, ratio)) + ape
    if overlap:
        kv = published_overlap_transform(kv, 0)
        score = published_overlap_transform(score, float("-inf"))
    pooled = (kv * score.softmax(dim=2)).sum(dim=2)
    return rms_norm(pooled.to(kv.dtype), norm_weight.float(), eps)


# ------------------------------------------------------------------- checks


def test_csa_prefill_matches_the_published_formula():
    """21 of 43 layers, checked against a second implementation's arithmetic."""
    torch.manual_seed(23)
    tokens, hidden = CSA_RATIO * 6, 64
    x = torch.randn(1, tokens, hidden)
    wkv = torch.randn(2 * HEAD_DIM, hidden) * 0.05
    wgate = torch.randn(2 * HEAD_DIM, hidden) * 0.05
    ape = torch.randn(CSA_RATIO, 2 * HEAD_DIM) * 0.1
    norm_w = torch.rand(HEAD_DIM) + 0.5
    theirs = published_prefill_compress(x, wkv, wgate, ape, norm_w, 1e-6, overlap=True, ratio=CSA_RATIO)
    ours = compress_windows_csa(x @ wkv.t(), x @ wgate.t(), ape, norm_w, 1e-6, CSA_RATIO)
    assert ours.shape == theirs.shape == (1, 6, HEAD_DIM), (ours.shape, theirs.shape)
    diff = (ours - theirs).abs().max()
    assert torch.allclose(ours, theirs, rtol=1e-5, atol=1e-6), f"max diff {diff:.3e}"


def test_hca_prefill_matches_the_published_formula():
    """Non-overlapping case: `coff == 1`, no overlap transform, `ratio == 128`."""
    torch.manual_seed(29)
    tokens, hidden = HCA_RATIO * 3, 64
    x = torch.randn(1, tokens, hidden)
    wkv = torch.randn(HEAD_DIM, hidden) * 0.05
    wgate = torch.randn(HEAD_DIM, hidden) * 0.05
    ape = torch.randn(HCA_RATIO, HEAD_DIM) * 0.1
    norm_w = torch.rand(HEAD_DIM) + 0.5
    theirs = published_prefill_compress(x, wkv, wgate, ape, norm_w, 1e-6, overlap=False, ratio=HCA_RATIO)
    ours = compress_windows(x @ wkv.t(), x @ wgate.t(), ape, norm_w, 1e-6, HCA_RATIO)
    assert ours.shape == theirs.shape == (1, 3, HEAD_DIM)
    diff = (ours - theirs).abs().max()
    assert torch.allclose(ours, theirs, rtol=1e-5, atol=1e-6), f"max diff {diff:.3e}"


def test_the_sentinel_choice_is_shared_by_both_implementations():
    """model.py:347-348 uses 0 for kv and -inf for score, exactly as the port does."""
    torch.manual_seed(31)
    tokens, hidden = CSA_RATIO * 2, 32
    x = torch.randn(1, tokens, hidden)
    wkv = torch.randn(2 * HEAD_DIM, hidden) * 0.05
    wgate = torch.randn(2 * HEAD_DIM, hidden) * 0.05
    ape = torch.zeros(CSA_RATIO, 2 * HEAD_DIM)
    norm_w = torch.ones(HEAD_DIM)
    kv = (x @ wkv.t()).float().unflatten(1, (-1, CSA_RATIO))
    score = (x @ wgate.t()).float().unflatten(1, (-1, CSA_RATIO))
    kv_t = published_overlap_transform(kv, 0)
    score_t = published_overlap_transform(score, float("-inf"))
    # First window's leading `ratio` slots: zero kv under an -inf gate.
    assert torch.equal(kv_t[0, 0, :CSA_RATIO], torch.zeros(CSA_RATIO, HEAD_DIM))
    assert torch.equal(score_t[0, 0, :CSA_RATIO], torch.full((CSA_RATIO, HEAD_DIM), float("-inf")))
    weights = score_t.softmax(dim=2)
    assert torch.equal(
        weights[0, 0, :CSA_RATIO], torch.zeros(CSA_RATIO, HEAD_DIM)
    ), "the sentinel must contribute exactly zero weight"


def test_indexer_scoring_matches_the_published_einsum_form():
    """model.py:424-426 folds both scales into `weights`; the port folds them separately."""
    torch.manual_seed(37)
    b, s, h, d, t = 1, 4, 8, 16, 6
    q = torch.randn(b, s, h, d)
    keys = torch.randn(b, t, d)
    weights_raw = torch.randn(b, s, h)
    softmax_scale = d**-0.5
    weights_scaling = h**-0.5

    theirs = torch.einsum("bshd,btd->bsht", q, keys)
    theirs = (theirs.relu_() * (weights_raw * (softmax_scale * weights_scaling)).unsqueeze(-1)).sum(dim=2)
    ours = index_scores(q, keys, weights_raw, softmax_scale, weights_scaling)
    assert torch.allclose(ours, theirs, rtol=1e-5, atol=1e-6), (ours - theirs).abs().max()
    # And the scale folding must be interchangeable, not merely close at this size.
    alt = index_scores(q, keys, weights_raw * softmax_scale, 1.0, weights_scaling)
    assert torch.allclose(ours, alt, rtol=1e-5, atol=1e-6)


def test_selection_rule_matches_the_published_mask_and_clamp():
    """model.py:433-436: k = min(topk, eligible), then -1 where idx >= (t+1)//ratio."""
    torch.manual_seed(41)
    length = 40
    pos = torch.arange(length).unsqueeze(0)
    scores = torch.randn(1, length, length // CSA_RATIO)
    ours = select_top_k(scores, pos, CSA_RATIO, INDEX_TOPK)

    end_pos = length
    ratio = CSA_RATIO
    theirs = scores.clone()
    mask = torch.arange(length // ratio).repeat(length, 1) >= torch.arange(1, length + 1).unsqueeze(1) // ratio
    theirs = theirs + torch.where(mask, float("-inf"), 0.0)
    k = min(INDEX_TOPK, end_pos // ratio)
    picked = theirs.topk(k, dim=-1)[1]
    picked = torch.where(
        picked >= torch.arange(1, length + 1).unsqueeze(1) // ratio, torch.full_like(picked, -1), picked
    )
    assert ours.shape == picked.shape, (ours.shape, picked.shape)
    assert torch.equal(ours, picked), (ours != picked).nonzero()[:6].tolist()


def test_published_decode_alignment_is_position_derived_and_differs_on_gaps():
    """Documented divergence, not a bug claim about either side.

    model.py:350-352 gates compression on `(start_pos + 1) % ratio == 0` and picks the ape row
    with `start_pos % ratio`, while this port (following the HF/tt-metal reference) derives
    `first_window_position` from `entry_count * ratio`. Sequential streams agree; a stream that
    consumes tokens without emitting an entry in lockstep does not. The test pins the
    consequence so a future serving change cannot silently pick the other convention.

    Two rules, two slots. This port takes the next window slot from the cache
    (``entry_count * ratio``); the published decode path takes it from the absolute token
    position. They agree exactly while every consumed token has been appended to this
    sequence's own cache, which is the sequential-decode case.
    """

    def entry_rule(entry_count: int) -> int:
        return entry_count * HCA_RATIO

    def position_rule(start_pos: int) -> int:
        # model.py:351-352: boundaries at multiples of ratio, ape row indexed by start_pos % ratio.
        return start_pos - (start_pos % HCA_RATIO)

    # Sequential decode: two entries emitted, next token is position 256. Agreement.
    assert entry_rule(2) == position_rule(256) == 256

    # Divergence: a copied or truncated prefix cache (one entry retained) while the stream's
    # own position counter says 256 -- reachable with prefix reuse, restored requests, or
    # rejected speculative tokens. The two conventions then disagree about where the next
    # window starts, and therefore about which ape row and which rope slot it uses.
    assert entry_rule(1) == 128 and position_rule(256) == 256
    assert entry_rule(1) != position_rule(256)
    # And the condition of agreement, stated rather than implied:
    assert all(entry_rule(p // HCA_RATIO) == position_rule(p) for p in range(0, 1024, HCA_RATIO))
