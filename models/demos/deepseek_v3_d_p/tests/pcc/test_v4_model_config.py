# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-only guards on the V4 bring-up config.

Everything in the TTNN glue is sized and scheduled from ``V4ModelArgs``, so a
drift here silently changes which model we are building. These checks are cheap
and need no device.

The most important ones are that our schedule and rope groups agree with what the
released checkpoint tells the reference to build. If they ever disagree, every parity
number measured through the preset is describing a different architecture than the
model — which is exactly the bug found in worklog 974c2b: the preset reproduced
``DeepseekV4Config``'s *default* rule (a V4-Pro shape: two HCA bootstrap layers, then
HCA/CSA interleaved, **no sliding layers**) while V4-Flash ships a per-layer
``compress_ratios`` list in ``config.json`` that yields ``S S C H C H …``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v4_model_config import CSA, HCA, SLIDING, V4ModelArgs
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import default_snapshot_dir


def _reference_from_checkpoint(n_layers: int | None = None) -> DeepseekV4Config:
    """The reference config built the way the checkpoint says to build it."""
    src = V4ModelArgs()
    kwargs = dict(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=n_layers or src.num_hidden_layers,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        num_experts_per_tok=2,
        n_routed_experts=4,
        compress_ratios=src.compress_ratios,
    )
    return DeepseekV4Config(**kwargs)


def test_our_schedule_matches_the_checkpoint_driven_reference():
    """`flash` must reproduce what DeepseekV4Config derives from the checkpoint's ratios."""
    for n in (1, 2, 3, 4, 8, 43):
        ours = V4ModelArgs.tiny(n).layer_types()
        theirs = _reference_from_checkpoint(n).layer_types
        assert ours == theirs, f"n={n}: ours={ours[:4]} reference={theirs[:4]}"


def test_the_released_schedule_is_not_the_reference_default_rule():
    """Names the trap: the config class's own fallback rule is a V4-Pro shape, not V4-Flash's.

    `V4ModelArgs` only reproduces the released model because it carries `compress_ratios`.
    With them emptied it degrades to the class default rule — which is *not* V4-Flash. If
    the fallback ever stops matching the class default, the plumbing deserves a review.
    """
    default_rule = _reference_from_checkpoint(43)
    default_rule.compress_ratios = None
    default_rule.layer_types = None
    default_rule.__post_init__()

    released = V4ModelArgs.tiny(43).layer_types()
    fallback = V4ModelArgs(**{**V4ModelArgs.tiny(43).__dict__, "compress_ratios": []}).layer_types()

    assert fallback == default_rule.layer_types, "the no-ratios fallback no longer matches the class rule"
    assert released != default_rule.layer_types, "released and default schedules must differ"
    assert SLIDING in released, "the released schedule must contain sliding layers"
    assert SLIDING not in default_rule.layer_types, "the default rule has no sliding layers (V4-Pro shape)"
    assert default_rule.layer_types[:2] == [HCA, HCA], "the default rule leads with two HCA bootstrap layers"
    assert released[:4] == [SLIDING, SLIDING, CSA, HCA], f"released schedule head changed: {released[:4]}"


def test_preset_reproduces_the_released_checkpoint_config():
    """The gate that would have caught worklog 974c2b: preset == what the weights demand.

    Field-by-field against ``AutoConfig`` on the real snapshot, so a preset that quietly
    matches *some* config rather than *this* one cannot pass.
    """
    snap = default_snapshot_dir()
    if snap is None:
        pytest.skip("no V4-Flash snapshot available; DS4_V4_FLASH_DIR unset and hub cache empty")
    from transformers import AutoConfig

    released = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    ours = V4ModelArgs().drive_reference()

    assert list(ours.layer_types) == list(released.layer_types), "layer schedule differs"
    assert ours.compress_rates == released.compress_rates, "compress rates differ"
    assert ours.num_hidden_layers == released.num_hidden_layers
    assert ours.hidden_size == released.hidden_size
    assert ours.head_dim == released.head_dim
    assert ours.index_topk == released.index_topk
    assert ours.index_n_heads == released.index_n_heads
    assert ours.index_head_dim == released.index_head_dim
    assert ours.rope_theta == released.rope_theta
    for group in ("main", "compress"):
        assert (
            ours.rope_parameters[group] == released.rope_parameters[group]
        ), f"rope group {group}: ours={ours.rope_parameters[group]} released={released.rope_parameters[group]}"
    # Both rope facts that the checkpoint carries and that a preset can silently drop.
    assert ours.rope_parameters["compress"]["rope_type"] == "yarn", "compress rope lost YaRN"
    assert ours.rope_parameters["main"]["rope_type"] == "default", "main rope must stay plain"


def test_substituted_schedules_clear_the_only_blocked_op():
    """CSA is what needs Blackhole-only ops; the substitutions must remove it."""
    for sched, want_csa in (("flash", True), ("hca_only", False), ("sliding_only", False)):
        args = V4ModelArgs(**{**V4ModelArgs.tiny(6).__dict__, "schedule": sched})
        types = args.layer_types()
        assert (CSA in types) is want_csa, sched
        assert bool(args.requires_unavailable_ops()) is want_csa, sched


def test_hash_moe_layer_is_never_substituted():
    """Hash routing is a distinct code path; keep it in every schedule."""
    for sched in ("flash", "hca_only", "sliding_only"):
        args = V4ModelArgs(**{**V4ModelArgs.tiny(6).__dict__, "schedule": sched})
        assert args.mlp_layer_types()[: args.num_hash_layers] == ["hash_moe"] * args.num_hash_layers
        assert args.mlp_layer_types()[args.num_hash_layers :] == ["moe"] * (6 - args.num_hash_layers)


def test_real_flash_dims_are_frozen_not_editable():
    """The non-tiny preset must equal the published contract."""
    a = V4ModelArgs()
    assert (a.hidden_size, a.num_hidden_layers, a.hc_mult) == (4096, 43, 4)
    assert (a.n_routed_experts, a.num_experts_per_tok) == (256, 6)
    assert a.vocab_size == 129280


def test_tiny_preset_builds_a_generating_oracle():
    """The whole parity strategy rests on this reference constructing and decoding."""
    args = V4ModelArgs(**{**V4ModelArgs.tiny(4).__dict__, "schedule": "sliding_only"})
    hf = args.drive_reference()
    assert hf.layer_types == [SLIDING] * 4 and HCA not in hf.layer_types

    model = DeepseekV4ForCausalLM(hf).float().eval()
    assert sum(p.numel() for p in model.parameters()) < 50_000_000, "tiny preset grew past one chip"
    # The mHC modules leave torch.empty params when built standalone; going
    # through PreTrainedModel must initialise them.
    assert not [k for k, v in model.named_parameters() if not torch.isfinite(v).all()]

    ids = torch.randint(0, hf.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    assert out.logits.shape == (1, 8, hf.vocab_size)
    assert torch.isfinite(out.logits).all()
    # Intermediates carry the mHC stream axis [B, S, hc_mult, hidden]; the LAST
    # one is [B, S, hidden], i.e. already through HyperHead + final norm. That
    # asymmetry pins where the stream collapse happens, so assert it rather than
    # assuming a uniform shape -- the TTNN model must match it exactly.
    streams = out.hidden_states[:-1]
    assert streams[0].shape == (1, 8, args.hc_mult, args.hidden_size)
    assert all(h.shape == (1, 8, args.hc_mult, args.hidden_size) for h in streams)
    assert out.hidden_states[-1].shape == (1, 8, args.hidden_size)
    assert len(out.hidden_states) == args.num_hidden_layers + 1

    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert gen.shape[1] == 16, "greedy 8-in/8-out loop did not run"
    assert torch.equal(gen[0, :8], ids[0]), "generate did not preserve the prompt prefix"


def test_unknown_schedule_is_rejected_not_silently_defaulted(expect_error):
    args = V4ModelArgs(**{**V4ModelArgs.tiny(2).__dict__, "schedule": "sliding"})
    with expect_error(ValueError, "sliding"):
        args.layer_types()


# ---- mapping onto upstream's MoE gate -------------------------------------- #

GATE_SOURCE = Path(__file__).resolve().parents[2] / "tt/moe/tt_moe_gate_prefill.py"
"""Absolute on purpose. A repo-root-relative path here makes the test pass only when
pytest is invoked from the tt-metal root and fail from the demo directory, which is a
trap for anyone running the suite from here."""


def test_gate_cfg_covers_every_key_the_gate_actually_reads():
    """Derived from the gate source, not from a list we maintain by hand.

    If upstream starts reading another ``model_cfg.X``, this fails rather than
    letting the MoE silently fall back to a DeepSeek-V3 default at graph-build
    time -- which is the kind of failure that only shows up on hardware.
    """
    import re

    text = Path(GATE_SOURCE).read_text()
    reads = set(re.findall(r"model_cfg\.([A-Z][A-Z0-9_]*)", text))
    assert reads, "no model_cfg reads found -- did the gate's config style change?"
    missing = reads - set(V4ModelArgs().moe_gate_cfg())
    assert not missing, f"gate reads keys we do not supply: {sorted(missing)}"


def test_gate_cfg_disables_group_routing_because_v4_has_no_group_stage():
    """n_expert_groups == 1 is the gate's ungrouped branch; V3 defaults would
    enable grouped routing that the V4 reference never performs."""
    cfg = V4ModelArgs().moe_gate_cfg()
    assert cfg["NUM_EXPERT_GROUPS"] == 1
    assert cfg["NUM_LIMITED_GROUPS"] == 1


def test_gate_cfg_carries_v4_router_semantics():
    cfg = V4ModelArgs().moe_gate_cfg()
    assert cfg["SCORE_FUNC"] == "sqrtsoftplus", "V4 scores with sqrt(softplus(x)), not sigmoid/softmax"
    assert cfg["ROUTE_SCALE"] == 1.5
    assert cfg["NUM_ROUTED_EXPERTS"] == 256
    # 6, not the 8 that DeepSeek-V3-family models use -- taken from the frozen
    # inventory and pinned by test_real_flash_dims_are_frozen_not_editable.
    assert cfg["NUM_EXPERTS_PER_TOKEN"] == 6
    assert cfg["NUM_SHARED_EXPERTS"] == 1
    assert cfg["EMB_SIZE"] == 4096


def test_gate_cfg_tracks_config_instead_of_hardcoding():
    args = replace(V4ModelArgs(), num_experts_per_tok=4, n_routed_experts=64, route_scale=2.0)
    cfg = args.moe_gate_cfg()
    assert cfg["NUM_EXPERTS_PER_TOKEN"] == 4
    assert cfg["NUM_ROUTED_EXPERTS"] == 64
    assert cfg["ROUTE_SCALE"] == 2.0
    # ...but the group fields stay pinned regardless of model size.
    assert cfg["NUM_EXPERT_GROUPS"] == cfg["NUM_LIMITED_GROUPS"] == 1
