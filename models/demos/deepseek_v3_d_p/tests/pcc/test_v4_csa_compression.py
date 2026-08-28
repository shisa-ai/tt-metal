"""CSA compressor contracts, pinned against the reference at the **real checkpoint config**.

CSA is the largest layer class in V4-Flash (21 of 43 layers), and its two-series window
trick is the easiest thing here to get subtly wrong: the shapes work whether or not Ca and
Cb are in the right halves, and whether or not the first window's missing predecessor is
sentinelled with ``-inf``.

The reference ``DeepseekV4CSACompressor.forward`` always calls its Lightning Indexer, and the
indexer only affects the returned **mask**, never ``compressed_kv``. Every compressor-parity
test here therefore replaces ``comp.indexer`` with a deterministic stub so the compression
math is compared in isolation; the mask is compared separately with hand-written index
tensors that include the indexer's ``-1`` rejections.
"""

import collections

import pytest
import torch
from transformers import AutoConfig

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACompressor
from models.demos.deepseek_v3_d_p.tt.v4_compression import TtCSACompressor, compress_windows_csa, sparse_selection_bias
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import default_snapshot_dir

SNAP = default_snapshot_dir()

HEAD_DIM = 512
RATE = 4
HIDDEN = 4096


def checkpoint_config():
    """The real released config, not a hand-built lookalike.

    CSA builds a full Lightning Indexer in ``__init__``, so a synthetic config either fails
    to construct or silently changes what is being tested; the released config also pins
    ``compress_rates`` and the rope groups to the shipped values.
    """

    if SNAP is None:
        pytest.skip("V4-Flash checkpoint not present")
    return AutoConfig.from_pretrained(SNAP, trust_remote_code=True)


def filled_reference(seed=13):
    cfg = checkpoint_config()
    torch.manual_seed(seed)
    comp = DeepseekV4CSACompressor(cfg).to(torch.float32).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in comp.named_parameters():
            if p.dtype is not torch.float32:
                continue
            if name.endswith("weight") and p.ndim == 1:
                p.normal_(1.0, 0.02, generator=g)
            elif name.endswith("position_bias"):
                p.normal_(0.0, 0.05, generator=g)
            else:
                p.normal_(0.0, (1.0 / p.shape[-1]) ** 0.5, generator=g)
    return cfg, comp


def port_weights(comp):
    return {
        "kv_proj": comp.kv_proj.weight.detach().clone(),
        "gate_proj": comp.gate_proj.weight.detach().clone(),
        "position_bias": comp.position_bias.detach().clone(),
        "kv_norm_weight": comp.kv_norm.weight.detach().clone(),
    }


class StubIndexer(torch.nn.Module):
    """Stand-in with the indexer's output contract, so nn.Module assignment is legal.

    ``compressed_kv`` never depends on the indexer, so replacing it isolates the compression
    math from the selection mask without touching the reference's own window code.
    """

    def __init__(self, indices: torch.Tensor):
        super().__init__()
        self.indices = indices

    def forward(self, *args, **kwargs):
        return self.indices


# --------------------------------------------------------------------- parity


def test_checkpoint_layer_classes_and_rate_are_what_we_claim():
    cfg = checkpoint_config()
    counts = collections.Counter(cfg.layer_types)
    assert counts["compressed_sparse_attention"] == 21, counts
    assert counts["heavily_compressed_attention"] == 20, counts
    assert counts["sliding_attention"] == 2, counts
    assert cfg.compress_rates["compressed_sparse_attention"] == RATE
    assert cfg.compress_rates["heavily_compressed_attention"] == 128
    assert cfg.index_topk == 512 and cfg.index_n_heads == 64


def test_compressed_entries_match_the_reference_compressor():
    cfg, comp = filled_reference()
    ours = TtCSACompressor(cfg, port_weights(comp))
    assert ours.compress_rate == RATE and ours.rope_dim == 64
    assert ours.rope_params.rope_type == "yarn" and ours.rope_params.rope_theta == 160000.0

    length = 8 * RATE  # 8 closed windows -> 8 entries
    torch.manual_seed(3)
    hs = torch.randn(1, length, HIDDEN, dtype=torch.float32)
    pos = torch.arange(length).unsqueeze(0)
    comp.indexer = StubIndexer(torch.arange(length).view(1, length, 1) // RATE)
    with torch.no_grad():
        ref_kv, ref_bias = comp(hs, None, pos, None, 0)
        our_kv, _ = ours.forward(hs, None, compressed_len=0)

    assert ref_kv.shape == (1, 1, 8, HEAD_DIM), ref_kv.shape
    assert our_kv.shape == ref_kv.shape, (our_kv.shape, ref_kv.shape)
    diff = (our_kv - ref_kv).abs().max()
    assert torch.allclose(our_kv, ref_kv, rtol=1e-5, atol=1e-6), f"max {diff:.3e}"


def test_selection_mask_matches_the_reference_scatter():
    cfg, comp = filled_reference()
    ours = TtCSACompressor(cfg, port_weights(comp))
    length = 6 * RATE
    pos = torch.arange(length).unsqueeze(0)
    hs = torch.randn(1, length, HIDDEN, dtype=torch.float32)
    # Three selected entries per query plus one rejected (-1) slot, exercising the clamp.
    idx = torch.stack(
        [torch.stack([torch.full((length,), 0), torch.full((length,), 1), torch.full((length,), -1)], dim=-1)]
    )
    idx[0, 3:, 2] = 2
    comp.indexer = StubIndexer(idx)
    with torch.no_grad():
        ref_kv, ref_bias = comp(hs, None, pos, None, 0)
    compressed_len = ref_kv.shape[2]
    our_bias = sparse_selection_bias(idx, compressed_len, dtype=ref_bias.dtype)
    assert our_bias.shape == ref_bias.shape, (our_bias.shape, ref_bias.shape)
    assert torch.equal(torch.isfinite(our_bias), torch.isfinite(ref_bias))


def test_rejected_index_does_not_wrap_to_the_last_column():
    """Scattering -1 unwrapped would mark the *last* entry visible for every query."""
    idx = torch.full((1, 2, 3), -1, dtype=torch.long)
    bias = sparse_selection_bias(idx, 4)
    assert not torch.isfinite(bias).any(), "every slot is rejected, so nothing may be visible"

    idx2 = idx.clone()
    idx2[0, 0, 0] = 3  # query 0 selects entry 3; query 1 selects nothing
    bias2 = sparse_selection_bias(idx2, 4)
    assert torch.isfinite(bias2[0, 0, 0, 3]), "the selected entry must be the visible column"
    assert not torch.isfinite(bias2[0, 0, 0, :3]).any(), "the other columns stay masked"
    assert not torch.isfinite(bias2[0, 0, 1]).any(), "a fully rejected query sees nothing"


# ------------------------------------------------------- two-series window math


def test_first_window_carries_no_prior_contribution():
    """Sentinelling window -1 with -inf gate is not cosmetic; a finite gate dilutes it."""
    head_dim, rate = 4, 2
    torch.manual_seed(5)
    kv = torch.randn(1, 2 * rate, 2 * head_dim)
    gate = torch.randn(1, 2 * rate, 2 * head_dim)
    bias = torch.zeros(rate, 2 * head_dim)
    w = torch.ones(head_dim)
    stateless = compress_windows_csa(kv, gate, bias, w, 1e-6, rate)
    zero_prior_kv = torch.zeros(1, rate, head_dim)
    inf_prior_gate = torch.full((1, rate, head_dim), float("-inf"))
    same = compress_windows_csa(kv, gate, bias, w, 1e-6, rate, zero_prior_kv, inf_prior_gate)
    assert torch.allclose(stateless, same, atol=0, rtol=0), "an -inf prior must be a no-op"

    # The failure this guards: zero kv with a *finite* gate still pulls weight away from
    # the real slots, so the entry shrinks even though no data was added.
    finite_gate = torch.zeros(1, rate, head_dim)
    diluted = compress_windows_csa(kv, gate, bias, w, 1e-6, rate, zero_prior_kv, finite_gate)
    assert not torch.allclose(stateless, diluted), "a finite prior must change the entry"
    assert diluted[0, 0].abs().sum() < stateless[0, 0].abs().sum()


def test_the_two_series_reach_different_entries():
    """Cb feeds the current entry, Ca feeds the next one. Both directions, separately.

    These two assertions are what detect a halves swap or an off-by-one window shift; either
    error keeps every shape valid and only moves which entry reacts to which token.
    """
    head_dim, rate = 4, 2
    torch.manual_seed(7)
    kv = torch.randn(1, 4 * rate, 2 * head_dim)
    gate = torch.randn(1, 4 * rate, 2 * head_dim)
    bias, w = torch.zeros(rate, 2 * head_dim), torch.ones(head_dim)
    base = compress_windows_csa(kv, gate, bias, w, 1e-6, rate)
    token = rate + 1  # second token of window 1

    cb = kv.clone()
    cb[0, token, head_dim:] += 3.0  # feeds entry 1
    from_cb = compress_windows_csa(cb, gate, bias, w, 1e-6, rate)
    assert not torch.allclose(from_cb[0, 1], base[0, 1]), "Cb must move its own entry"
    assert torch.allclose(from_cb[0, 2], base[0, 2], atol=1e-6), "Cb must not leak forward"
    assert torch.allclose(from_cb[0, 3], base[0, 3], atol=1e-6)

    ca = kv.clone()
    ca[0, token, :head_dim] += 3.0  # feeds entry 2
    from_ca = compress_windows_csa(ca, gate, bias, w, 1e-6, rate)
    assert not torch.allclose(from_ca[0, 2], base[0, 2]), "Ca must reach the next entry"
    assert torch.allclose(from_ca[0, 1], base[0, 1], atol=1e-6), "Ca must not hit its own entry"
    assert torch.allclose(from_ca[0, 3], base[0, 3], atol=1e-6), "and must not skip ahead"


def test_two_series_layout_matches_a_hand_written_formula():
    """Independent per-slot computation, written without consulting the implementation."""
    head_dim, rate, n_win = 3, 2, 3
    torch.manual_seed(9)
    kv = torch.randn(1, n_win * rate, 2 * head_dim)
    gate = torch.randn(1, n_win * rate, 2 * head_dim)
    bias = torch.randn(rate, 2 * head_dim)
    weight = torch.rand(head_dim) + 0.5
    got = compress_windows_csa(kv, gate, bias, weight, 1e-6, rate)

    def entry(w):
        slots_kv, slots_gate = [], []
        if w > 0:  # previous window's Ca
            for j in range(rate):
                slots_kv.append(kv[0, (w - 1) * rate + j, :head_dim])
                slots_gate.append(gate[0, (w - 1) * rate + j, :head_dim] + bias[j, :head_dim])
        for j in range(rate):  # this window's Cb
            slots_kv.append(kv[0, w * rate + j, head_dim:])
            slots_gate.append(gate[0, w * rate + j, head_dim:] + bias[j, head_dim:])
        k = torch.stack(slots_kv)  # [2*rate, head_dim]
        g = torch.stack(slots_gate)  # [2*rate, head_dim]
        soft = torch.softmax(g.float(), dim=0).to(k.dtype)
        summed = (k * soft).sum(dim=0)
        var = summed.pow(2).mean()
        return summed * torch.rsqrt(var + 1e-6) * weight

    for w in range(n_win):
        assert torch.allclose(got[0, w], entry(w), rtol=1e-5, atol=1e-6), (w, (got[0, w] - entry(w)).abs().max())


def test_swapping_the_two_series_is_detected():
    head_dim, rate = 4, 2
    torch.manual_seed(17)
    kv = torch.randn(1, 4 * rate, 2 * head_dim)
    bias = torch.zeros(rate, 2 * head_dim)
    gate = torch.randn(1, 4 * rate, 2 * head_dim)
    weight = torch.ones(head_dim)
    correct = compress_windows_csa(kv, gate, bias, weight, 1e-6, rate)
    swapped_kv = torch.cat([kv[..., head_dim:], kv[..., :head_dim]], dim=-1)
    swapped = compress_windows_csa(swapped_kv, gate, bias, weight, 1e-6, rate)
    assert not torch.allclose(correct, swapped), "layout must be observable"


# ----------------------------------------------------------------- boundaries


def test_partial_window_is_dropped():
    head_dim, rate = 4, 2
    kv = torch.randn(1, 5 * rate + 1, 2 * head_dim)
    gate = torch.randn(1, 5 * rate + 1, 2 * head_dim)
    out = compress_windows_csa(kv, gate, torch.zeros(rate, 2 * head_dim), torch.ones(head_dim), 1e-6, rate)
    assert out.shape == (1, 5, head_dim), out.shape


def test_prior_shapes_are_validated(expect_error):
    head_dim, rate = 4, 2
    kv = torch.randn(1, 2 * rate, 2 * head_dim)
    gate = torch.randn(1, 2 * rate, 2 * head_dim)
    with expect_error(ValueError, "given together"):
        compress_windows_csa(
            kv,
            gate,
            torch.zeros(rate, 2 * head_dim),
            torch.ones(head_dim),
            1e-6,
            rate,
            prior_kv=torch.zeros(1, rate, head_dim),
            prior_gate=None,
        )
    with expect_error(ValueError, "prior_kv must be"):
        compress_windows_csa(
            kv,
            gate,
            torch.zeros(rate, 2 * head_dim),
            torch.ones(head_dim),
            1e-6,
            rate,
            prior_kv=torch.zeros(1, rate + 1, head_dim),
            prior_gate=torch.zeros(1, rate + 1, head_dim),
        )


def test_entry_table_overrun_raises(expect_error):
    cfg, comp = filled_reference()
    ours = TtCSACompressor(cfg, port_weights(comp), max_windows=2)
    assert ours._cos_by_offset.shape[0] == 2 * RATE
    with expect_error(ValueError, "exceeds the precomputed table"):
        ours.rope_for_entries(3, 0)
    # A resumed decode offset counts against the table too.
    with expect_error(ValueError, "exceeds the precomputed table"):
        ours.rope_for_entries(1, first_window_position=2 * RATE)


def test_decode_resume_offset_resumes_the_window_slots():
    cfg, comp = filled_reference()
    ours = TtCSACompressor(cfg, port_weights(comp))
    cos_a, _ = ours.rope_for_entries(1, 0)
    cos_b, _ = ours.rope_for_entries(1, RATE * 3)
    fourth, _ = ours.rope_for_entries(4, 0)
    assert torch.equal(fourth[:, -1:, :], cos_b), "entry 3 sits at slot 3*rate"
    assert not torch.equal(cos_a, cos_b), "position must actually change the rotation"
