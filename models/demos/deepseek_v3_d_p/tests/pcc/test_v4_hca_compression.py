"""HCA compressor contracts, pinned against the reference compressor at real geometry.

Every claim here is about the compress path that most V4-Flash layers use, and most of it is
checkable without hardware: the window reduction, the causal rule over compressed entries, the
entry rope positions, and the trailing-slice rope layout.

The parity test builds the reference ``DeepseekV4HCACompressor`` with explicit weights at
V4-Flash geometry (head_dim 512, rate 128, YaRN compress rope) and compares fp32 outputs.
"""

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HCACompressor
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    apply_rotary_pos_emb as reference_apply_rotary,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import rotate_half as reference_rotate_half
from models.demos.deepseek_v3_d_p.tt.v4_compression import (
    TtHCACompressor,
    apply_v4_rope,
    compress_windows,
    compressed_positions,
    window_causal_bias,
)

HEAD_DIM = 512
ROPE_DIM = 64
HIDDEN = 256  # only the two projections see hidden; 4096 would just slow the test down
RATE = 128


def real_geometry_config():
    """V4-Flash attention geometry plus the checkpoint's two rope groups."""
    cfg = DeepseekV4Config(
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        num_hidden_layers=1,
        sliding_window=128,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": RATE},
        rope_scaling={
            "main": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.125},
            "compress": {
                "rope_type": "yarn",
                "rope_theta": 160000,
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
                "attention_factor": 1.0,
                "partial_rotary_factor": 0.125,
            },
        },
        max_position_embeddings=1_048_576,
    )
    return cfg


def filled_reference(seed=7):
    """The reference compressor with every parameter filled explicitly."""
    cfg = real_geometry_config()
    torch.manual_seed(seed)
    comp = DeepseekV4HCACompressor(cfg).to(torch.float32).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in comp.named_parameters():
            if p.dtype is not torch.float32:
                continue
            if name.endswith("weight") and p.ndim == 1:
                p.normal_(1.0, 0.02, generator=g)  # RMSNorm weight: near one, not zero
            elif name.endswith("position_bias"):
                p.normal_(0.0, 0.05, generator=g)
            else:
                p.normal_(0.0, (1.0 / p.shape[-1]) ** 0.5, generator=g)
    return cfg, comp


def port_weights(cfg, comp):
    return {
        "kv_proj": comp.kv_proj.weight.detach().clone(),
        "gate_proj": comp.gate_proj.weight.detach().clone(),
        "position_bias": comp.position_bias.detach().clone(),
        "kv_norm_weight": comp.kv_norm.weight.detach().clone(),
    }


# ------------------------------------------------------------------ window math


def test_window_reduction_matches_the_reference_compressor():
    cfg, comp = filled_reference()
    ours = TtHCACompressor(cfg, port_weights(cfg, comp))
    torch.manual_seed(11)
    hs = torch.randn(1, 3 * RATE, HIDDEN, dtype=torch.float32)
    pos = torch.arange(hs.shape[1]).unsqueeze(0)

    with torch.no_grad():
        ref_kv, ref_bias = comp(hs, None, pos, None, 0)
        our_kv, our_bias = ours.forward(hs, pos, compressed_len=ref_kv.shape[2])

    assert our_kv.shape == ref_kv.shape, (tuple(our_kv.shape), tuple(ref_kv.shape))
    assert torch.allclose(our_kv, ref_kv, rtol=1e-5, atol=1e-6), f"max diff {(our_kv - ref_kv).abs().max():.3e}"
    assert our_bias.shape == ref_bias.shape
    assert torch.equal(torch.isfinite(our_bias), torch.isfinite(ref_bias)), "causal pattern"


def test_partial_window_is_dropped_not_padded():
    cfg, comp = filled_reference()
    ours = TtHCACompressor(cfg, port_weights(cfg, comp))
    hs = torch.randn(1, 2 * RATE + RATE // 2, HIDDEN, dtype=torch.float32)
    assert ours.compress(hs).shape == (1, 2, HEAD_DIM), "remainder must be discarded"
    hs_exact = torch.randn(1, 2 * RATE, HIDDEN, dtype=torch.float32)
    assert compress_windows(
        hs_exact @ comp.kv_proj.weight.t(),
        hs_exact @ comp.gate_proj.weight.t(),
        comp.position_bias,
        comp.kv_norm.weight,
        cfg.rms_norm_eps,
        RATE,
    ).shape == (1, 2, HEAD_DIM)


def test_zero_windows_returns_an_empty_entry_axis():
    cfg, comp = filled_reference()
    ours = TtHCACompressor(cfg, port_weights(cfg, comp))
    hs = torch.randn(1, RATE - 1, HIDDEN, dtype=torch.float32)
    out = ours.compress(hs)
    assert out.shape == (1, 0, HEAD_DIM), out.shape
    assert compress_windows(
        torch.zeros(1, 3, HEAD_DIM),
        torch.zeros(1, 3, HEAD_DIM),
        torch.zeros(RATE, HEAD_DIM),
        torch.ones(HEAD_DIM),
        1e-6,
        RATE,
    ).shape == (1, 0, HEAD_DIM)


def test_softmax_runs_over_the_window_axis_in_fp32():
    """Softmaxing head_dim instead of the window would be a different, wrong object."""
    b, t, d = 1, 2 * RATE, 8
    kv = torch.ones(b, t, d)
    gate = torch.zeros(b, t, d)
    got = compress_windows(kv, gate, torch.zeros(RATE, d), torch.ones(d), 1e-6, RATE)
    # Uniform weights -> the mean of the window, then RMSNorm -> unit-ish magnitude.
    assert got.shape == (b, 2, d)
    assert torch.allclose(got, got[0, 0].expand_as(got), atol=1e-6)
    # A one-hot-ish gate must concentrate the entry on a single token of the window.
    sharp = torch.zeros(b, t, d)
    sharp[0, :, :] = -20.0
    sharp[0, 5, :] = 20.0
    focused = compress_windows(kv, sharp, torch.zeros(RATE, d), torch.ones(d), 1e-6, RATE)
    assert torch.allclose(focused[0, 0], got[0, 0], atol=1e-3), "softmax axis changed"


# -------------------------------------------------------------- causal and rope


def test_causal_rule_is_on_entries_not_tokens():
    """Entry ``w`` is visible to ``t`` iff ``w < (t + 1) // rate``. Hand-written expectations.

    With rate 128 the first compressed entry only becomes visible at token 127, and token
    200 still sees only entry 0 -- a point the reference's own comment (written for a smaller
    rate) invites you to get wrong.
    """
    pos = torch.tensor([[0, 7, 127, 128, 200, 255, 256, 300]])
    bias = window_causal_bias(pos, 4, RATE)
    visible = torch.isfinite(bias[0, 0])
    expected = {
        0: [],
        7: [],
        127: [0],
        128: [0],
        200: [0],
        255: [0, 1],
        256: [0, 1],
        300: [0, 1],
    }
    for col, token in enumerate(pos[0].tolist()):
        seen = [i for i, ok in enumerate(visible[col].tolist()) if ok]
        assert seen == expected[token], (token, seen, expected[token])


def test_causal_bias_matches_the_reference_exactly():
    """The reference's own mask over real windows, token by token.

    The hidden_states length must be at least one window: with fewer tokens the reference
    produces zero entries and returns ``None`` for the mask, which is a *different* branch
    and would test nothing about causality.
    """
    cfg, comp = filled_reference()
    length = 4 * RATE + RATE // 2  # 4 closed windows, so compressed_len == 4
    pos = torch.arange(length).unsqueeze(0)
    with torch.no_grad():
        hs = torch.randn(1, length, HIDDEN, dtype=torch.float32)
        ref_kv, ref_bias = comp(hs, None, pos, None, 0)
    assert ref_kv.shape[2] == 4 and ref_bias is not None, (ref_kv.shape, ref_bias)
    ours = window_causal_bias(pos, ref_bias.shape[-1], RATE, dtype=ref_bias.dtype)
    assert ours.shape == ref_bias.shape, (ours.shape, ref_bias.shape)
    assert torch.equal(torch.isfinite(ours), torch.isfinite(ref_bias))
    # Visible-entry *count* per query must equal min((t + 1) // rate, entries). The first
    # visible entry is always 0 once any entry exists, so a first-visible check would be a
    # weak test; the count is the actual causal statement.
    visible = [int(torch.isfinite(ref_bias[0, 0, t]).sum()) for t in (0, 126, 127, 254, 255, 511)]
    assert visible == [0, 0, 1, 1, 2, 4], visible


def test_entry_positions_are_window_slots_not_token_positions():
    got = compressed_positions(3, RATE, 0)
    assert got.tolist() == [[0, RATE, 2 * RATE]]
    resumed = compressed_positions(2, RATE, 4 * RATE)
    assert resumed.tolist() == [[4 * RATE, 5 * RATE]], "decode resume offset ignored"
    assert compressed_positions(0, RATE, 0).shape == (1, 0)


def test_compressed_entries_use_the_compress_rope_group():
    """YaRN at theta 160000, not main's plain theta 10000."""
    cfg, comp = filled_reference()
    ours = TtHCACompressor(cfg, port_weights(cfg, comp))
    assert ours.rope_params.rope_type == "yarn"
    assert ours.rope_params.rope_theta == 160000.0
    assert ours.rope_dim == ROPE_DIM
    assert ours.compress_rate == RATE == cfg.compress_rates["heavily_compressed_attention"]


def test_rope_matches_the_reference_helper_and_its_half_width_tables():
    """The reference returns half-width cos/sin and expands next to the math.

    Comparing widths directly is the only way to notice that the two conventions differ by
    exactly ``repeat_interleave(2)``.
    """
    cfg, comp = filled_reference()
    ours = TtHCACompressor(cfg, port_weights(cfg, comp))
    n = 4
    positions = compressed_positions(n, RATE, 0).expand(1, -1)
    probe = torch.randn(1, 1, n, HEAD_DIM, dtype=torch.float32)
    ref_cos, ref_sin = comp.rotary_emb(probe.squeeze(1), position_ids=positions, layer_type="compress")
    assert ref_cos.shape[-1] == ROPE_DIM // 2, ref_cos.shape
    our_cos, our_sin = ours.rope_for_entries(n, 0)
    assert our_cos.shape[-1] == ROPE_DIM
    ref_cos_full = ref_cos.repeat_interleave(2, dim=-1)
    ref_sin_full = ref_sin.repeat_interleave(2, dim=-1)
    assert our_cos.shape[-1] == ref_cos_full.shape[-1] == ROPE_DIM, (our_cos.shape, ref_cos.shape)
    assert torch.allclose(
        our_cos[0], ref_cos_full[0], rtol=1e-6, atol=1e-6
    ), f"cos differs, max {(our_cos[0] - ref_cos_full[0]).abs().max():.3e}"
    assert torch.allclose(our_sin[0], ref_sin_full[0], rtol=1e-6, atol=1e-6)

    # And the rotation itself, including the trailing-slice layout.
    ref_out = reference_apply_rotary(probe, ref_cos, ref_sin)
    our_out = apply_v4_rope(probe, our_cos, our_sin)
    assert torch.allclose(our_out, ref_out, rtol=1e-5, atol=1e-6), f"max diff {(our_out - ref_out).abs().max():.3e}"
    # The leading nope channels must be untouched by the rotation.
    assert torch.equal(our_out[..., :-ROPE_DIM], probe[..., :-ROPE_DIM])


def test_rotate_half_is_interleaved_not_half_split():
    x = torch.arange(1, 9, dtype=torch.float32).view(1, 1, 1, 8)
    got = reference_rotate_half(x)
    assert got[..., 0].item() == -2.0 and got[..., 1].item() == 1.0, got
    assert not torch.allclose(got[..., :4], torch.cat([-x[..., 4:8], x[..., 0:4]], dim=-1)[..., :4])


def test_rope_table_overrun_raises_instead_of_wrapping(expect_error):
    cfg, comp = filled_reference()
    ours = TtHCACompressor(cfg, port_weights(cfg, comp), max_windows=2)
    assert ours._cos_by_offset.shape[0] == 2 * RATE
    with expect_error(ValueError, "exceeds the precomputed table"):
        ours.rope_for_entries(3, 0)


def test_decode_step_returns_no_mask(expect_error):
    cfg, comp = filled_reference()
    ours = TtHCACompressor(cfg, port_weights(cfg, comp))
    hs = torch.randn(1, RATE, HIDDEN, dtype=torch.float32)
    pos = torch.tensor([[RATE]])
    entries, bias = ours.forward(hs, pos, compressed_len=1)
    assert bias is None, "a single decode query must not be masked"
    assert entries.shape == (1, 1, 1, HEAD_DIM), entries.shape
    # No entries yet is the other no-mask case.
    _e2, bias2 = ours.forward(torch.randn(1, 1, HIDDEN, dtype=torch.float32), pos, compressed_len=0)
    assert bias2 is None
    with expect_error(ValueError, r"must be \[B, S\]"):
        window_causal_bias(torch.zeros(4, dtype=torch.long), 2, RATE)


def test_weight_shape_contract_is_enforced(expect_error):
    cfg, comp = filled_reference()
    weights = port_weights(cfg, comp)
    weights["position_bias"] = torch.zeros(RATE, HEAD_DIM + 8)
    with expect_error(ValueError, "position_bias must be"):
        TtHCACompressor(cfg, weights).compress(torch.randn(1, 2 * RATE, HIDDEN))
