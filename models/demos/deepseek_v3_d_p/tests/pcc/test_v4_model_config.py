# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-only guards on the V4 bring-up config.

Everything in the TTNN glue is sized and scheduled from ``V4ModelArgs``, so a
drift here silently changes which model we are building. These checks are cheap
and need no device.

The most important one is that our reproduction of the reference's default layer
schedule agrees with the reference itself. If that ever disagrees, every parity
number measured through the substituted schedule is describing a different
architecture than the model.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v4_model_config import CSA, HCA, SLIDING, V4ModelArgs


def test_our_schedule_matches_the_reference_default_rule():
    """`flash` must reproduce what DeepseekV4Config derives on its own."""
    for n in (1, 2, 3, 4, 8, 43):
        ours = V4ModelArgs.tiny(n).layer_types()
        theirs = DeepseekV4Config(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=n,
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
        ).layer_types
        assert ours == theirs, f"n={n}: ours={ours[:4]} reference={theirs[:4]}"


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


def test_unknown_schedule_is_rejected_not_silently_defaulted():
    args = V4ModelArgs(**{**V4ModelArgs.tiny(2).__dict__, "schedule": "sliding"})
    with pytest.raises(ValueError, match="sliding"):
        args.layer_types()
