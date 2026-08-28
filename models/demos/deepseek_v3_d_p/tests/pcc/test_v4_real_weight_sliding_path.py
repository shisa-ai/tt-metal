# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The sliding path driven by the **released** config, not a hand-patched geometry.

`v4_sliding_attention` exists because V4-Flash has sliding layers. It also exists at all only
since worklog 54d30a: the preset was emitting a schedule with no sliding layers, so layers 0
and 1 — the only layers this module serves — were not in the model the port thought it was
building. Its own host-math suite is thorough, but it drives the module through
`V4ModelArgs.tiny(...)` with `partial_rotary_factor` patched in by hand, and its rope tables
are checked against a hand-written `10000 ** (-arange…)` expression. Both are fine as unit
tests and neither is a *parity* test: they compare the port to a restatement of the same
formula, at geometry somebody typed.

Here the module is constructed from the config the checkpoint demands, and its tables are
compared against the model's own rotary module — the code that will actually produce cos/sin
in any real run. If the port's rope convention (interleaved expansion, trailing-slice
position, absolute addressing) drifts from the reference's, this is where it shows.

Snapshot-gated: skipped when no checkpoint is present.
"""

from __future__ import annotations

import pytest
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs
from models.demos.deepseek_v3_d_p.tt.v4_sliding_attention import TtV4SlidingAttention, sliding_causal_mask
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import default_snapshot_dir

pytestmark = pytest.mark.skipif(default_snapshot_dir() is None, reason="no V4-Flash snapshot present")

POSITIONS = 64


@pytest.fixture(scope="module")
def released():
    """The real config plus the model it describes. Buffers for real, parameters on meta."""
    snap = default_snapshot_dir()
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    with init_empty_weights(include_buffers=False):
        model = DeepseekV4ForCausalLM(cfg)
    return cfg, model


@pytest.fixture(scope="module")
def port_module():
    """The port's sliding module driven by the released preset, with no patched geometry."""
    args = V4ModelArgs()
    cfg = args.drive_reference()
    return TtV4SlidingAttention(None, cfg, {}, max_seq_len=POSITIONS), cfg


def test_released_config_drives_the_port_geometry(port_module):
    """head_dim 512 with partial_rotary_factor 0.125 is a 64-wide rope slice = two tiles."""
    mod, cfg = port_module
    assert cfg.head_dim == 512, "this file is about the released head geometry"
    assert cfg.rope_parameters["main"]["rope_type"] == "default", "sliding layers use the MAIN group"
    assert cfg.rope_parameters["compress"]["rope_type"] == "yarn", "…while compress layers are YaRN"
    assert mod.rope_dim == 64, f"expected the real 64-dim slice, got {mod.rope_dim}"
    assert mod.nope_dim == 512 - 64
    assert mod.rope_theta == 10000.0, "the main group theta, not compress_rope_theta"
    assert mod.window == 128 == cfg.sliding_window
    assert mod.cache_len == 127, "the reference retains window-1, and a stale window is in-window"


def test_port_rope_tables_match_the_models_own_rotary_module(released, port_module):
    """Parity against the code that will produce cos/sin in a real run.

    Both sides are asked for positions 0..63. The reference returns half-width tables and
    expands them itself; the port keeps the expanded table. Comparing them is the whole point:
    an interleaving or half-width mistake is invisible inside either implementation alone.
    """
    cfg, model = released
    mod, port_cfg = port_module
    assert port_cfg.rope_parameters["main"]["rope_theta"] == cfg.rope_parameters["main"]["rope_theta"]

    # HF convention: position_ids is [B, S]; the module returns [B, 1, S, rope_dim] expanded.
    positions = torch.arange(POSITIONS).unsqueeze(0)
    probe = torch.zeros(1, POSITIONS, dtype=torch.float32)
    ref_cos, ref_sin = model.model.rotary_emb(probe, position_ids=positions, layer_type="main")
    # The reference returns HALF-width cos/sin and expands them inside apply_rotary_pos_emb,
    # while the port keeps the expanded table. That difference is the convention under test,
    # so it is asserted rather than silently hidden by reshaping both sides.
    assert (
        ref_cos.shape[-1] * 2 == mod.rope_dim
    ), f"reference half-width {ref_cos.shape[-1]} should be half of {mod.rope_dim}"
    ref_cos = ref_cos.repeat_interleave(2, dim=-1).reshape(POSITIONS, -1).float()
    ref_sin = ref_sin.repeat_interleave(2, dim=-1).reshape(POSITIONS, -1).float()
    cos_delta = float((mod._cos_full[:POSITIONS] - ref_cos).abs().max())
    sin_delta = float((mod._sin_full[:POSITIONS] - ref_sin).abs().max())
    assert cos_delta < 1e-6, f"cos table differs from the model's own: {cos_delta:.3e}"
    assert sin_delta < 1e-6, f"sin table differs from the model's own: {sin_delta:.3e}"


def test_rotation_agrees_with_the_reference_for_a_real_shaped_query(released, port_module):
    """The port's `nope passthrough + trailing slice` rotation vs `apply_rotary_pos_emb`."""
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    cfg, model = released
    mod, _ = port_module
    heads, seq = cfg.num_attention_heads, 8
    torch.manual_seed(0)
    x = torch.randn(1, seq, heads, cfg.head_dim, dtype=torch.float32)

    positions = torch.arange(seq).unsqueeze(0)
    probe = torch.zeros(1, seq, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(probe, position_ids=positions, layer_type="main")
    ref = apply_rotary_pos_emb(x.transpose(1, 2), cos, sin).transpose(1, 2)

    cos_b = mod._cos_full[:seq].reshape(1, seq, 1, mod.rope_dim)
    sin_b = mod._sin_full[:seq].reshape(1, seq, 1, mod.rope_dim)
    nope, rope = x[..., : mod.nope_dim], x[..., -mod.rope_dim :]
    ours = torch.cat([nope, rope * cos_b + (rope @ mod._trans_torch) * sin_b], dim=-1)

    delta = float((ours - ref).abs().max())
    assert delta < 1e-5, f"port rotation diverges from the reference: {delta:.3e}"
    assert torch.equal(ours[..., : mod.nope_dim], x[..., : mod.nope_dim]), "nope slice must pass through"


def test_window_mask_at_the_real_128_boundary(released, port_module):
    """Boundary behaviour at the checkpoint's real window, at the positions that matter."""
    cfg, _ = released
    mod, _ = port_module
    window = cfg.sliding_window
    mask = sliding_causal_mask(window * 2 + 1, window)  # 257 positions, window 128
    assert tuple(mask.shape) == (1, 1, window * 2 + 1, window * 2 + 1), "additive [1,1,S,S] prefill mask"

    # Excluded entries are torch.finfo(dtype).min, which is *finite* — so the visible set is
    # "exactly 0.0", not "isfinite". Using isfinite here would silently accept every position.
    allowed = mask[0, 0] == 0.0
    # Diagonal always visible; the token exactly `window` back is not; `window-1` back is.
    assert allowed[128, 128]
    assert not allowed[128, 128 - window], "one past the window must be masked"
    assert allowed[128, 128 - (window - 1)], "the oldest in-window token must survive"
    assert not allowed[128, 128 - window - 1]
    # Future is always masked, and the row width is exactly `window` for mature queries.
    assert not allowed[128, 129]
    assert int(allowed[128].sum()) == window, f"row 128 sees {int(allowed[128].sum())}, want {window}"
    assert int(allowed[256].sum()) == window
    assert mod.window == window
    # Additive, not boolean: an in-window entry is exactly 0.0 and an excluded one is the
    # dtype minimum, which is what lets the same tensor be added to attention scores.
    assert mask[0, 0, 128, 128] == 0.0
    assert mask[0, 0, 128, 0] < -1e30
