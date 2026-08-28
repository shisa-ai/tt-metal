"""Two rope groups, not one: the compress path is YaRN at a different theta.

The recorded real values are asserted against ``config.json`` when the snapshot is present,
so the inline copy used everywhere else cannot silently drift from the checkpoint.

Tier 1 runs anywhere: it pins the YaRN reimplementation against ``transformers``' own
function, pins that the main group is still bit-identical to what
``TtV4SlidingAttention`` builds, and shows numerically that YaRN moves positions long
before the 65536-token boundary it was fitted to.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.tt import v4_rope
from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs
from models.demos.deepseek_v3_d_p.tt.v4_sliding_attention import TtV4SlidingAttention

SNAP = os.environ.get("DS4_V4_FLASH_DIR")
needs_ckpt = pytest.mark.skipif(
    not SNAP or not os.path.isdir(SNAP), reason="set DS4_V4_FLASH_DIR to the V4-Flash snapshot"
)

HEAD_DIM = 512
ROPE_DIM = 64  # partial_rotary_factor 0.125 of head_dim 512 == qk_rope_head_dim

#: Recorded from the checkpoint's config.json on 2026-08-28; tier 2 re-checks this.
#: The per-group ``rope_theta`` values are **not in config.json** -- ``DeepseekV4Config``
#: injects them, taking ``main`` from ``rope_theta`` and ``compress`` from
#: ``compress_rope_theta``. They are spelled out here because HF's rope functions require a
#: group-level ``rope_theta`` and raise a bare ``KeyError: 'rope_theta'`` without one, even
#: though the docstring claims a default.
REAL_ROPE_PARAMETERS = {
    "main": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.125},
    "compress": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
        "rope_type": "yarn",
        "partial_rotary_factor": 0.125,
        "attention_factor": 1.0,
        # Injected by DeepseekV4Config from compress_rope_theta; tier 2 asserts the real
        # config really does derive this value rather than trusting the note.
        "rope_theta": 160000,
    },
}


def real_rope_config(**overrides):
    """A config carrying the checkpoint's rope parameters at the checkpoint's head geometry.

    ``rope_parameters`` is passed to the constructor as *grouped* values with explicit
    per-group thetas. Two tempting shortcuts both fail, and both failed during development of
    this file: patching ``cfg.rope_parameters`` on a live config merges the groups into one
    flat dict (main's ``rope_theta`` ends up attached to compress's YaRN keys), and passing
    ``rope_scaling`` alongside the preset's derived ``rope_parameters`` yields a hybrid HF's
    YaRN cannot read. Passing grouped values with thetas present is what the config class
    itself produces for the real checkpoint, so that is what the fixture mirrors.
    """
    scaling = overrides.pop("rope_parameters", REAL_ROPE_PARAMETERS)
    merged = {"head_dim": HEAD_DIM, "num_attention_heads": 4, **overrides}
    args = V4ModelArgs.tiny(1, **merged)
    kwargs = args.drive_reference().__dict__
    kwargs["rope_parameters"] = {k: dict(v) for k, v in scaling.items()}
    kwargs["max_position_embeddings"] = 1_048_576
    # head_dim arrives through the preset. Assigning it *after* construction is not a
    # no-op: transformers re-derives rope_parameters from the config's own fields, which
    # discards the grouped values passed above and leaves the yarn group without a theta.
    return DeepseekV4Config(**kwargs)


# ------------------------------------------------------- tier 1: runs anywhere


def test_the_two_groups_really_do_differ():
    cfg = real_rope_config()
    main_inv, main_factor, main_p = v4_rope.build_rope(cfg, "main", HEAD_DIM)
    comp_inv, comp_factor, comp_p = v4_rope.build_rope(cfg, "compress", HEAD_DIM)

    assert main_p.rope_type == "default" and comp_p.rope_type == "yarn"
    assert main_p.rope_theta == 10000.0 and comp_p.rope_theta == 160000.0, "two thetas, not one"
    assert main_factor == comp_factor == 1.0
    assert main_inv.shape == comp_inv.shape == (ROPE_DIM // 2,)
    assert not torch.allclose(main_inv, comp_inv), "compress must not collapse onto main"


def test_yarn_reimplementation_matches_transformers_bitwise():
    """The dependency-free path must not be allowed to drift from the oracle."""
    cfg = real_rope_config()
    delegated, factor, params = v4_rope.build_rope(cfg, "compress", HEAD_DIM)
    explicit = v4_rope.yarn_inv_freq(
        params.rope_theta,
        ROPE_DIM,
        params.factor,
        params.original_max_position_embeddings,
        params.beta_fast,
        params.beta_slow,
        params.truncate,
    )
    assert torch.equal(delegated, explicit), f"max diff {(delegated - explicit).abs().max():.3e}"
    assert factor == params.attention_factor == 1.0


def test_main_group_is_still_what_the_attention_module_builds():
    """No-regression guard: adopting this module must not change the device-side tables."""
    cfg = real_rope_config()
    mod = TtV4SlidingAttention(None, cfg, {})
    inv, factor, params = v4_rope.build_rope(cfg, "main", HEAD_DIM)
    assert params.rope_theta == mod.rope_theta
    assert mod.rope_dim == ROPE_DIM == v4_rope.rope_dim(HEAD_DIM, params.partial_rotary_factor)
    cos, sin = v4_rope.cos_sin_tables(inv, factor, mod.max_seq_len, mod.rope_dim)
    assert torch.equal(cos, mod._cos_full), "main-rope cos table changed"
    assert torch.equal(sin, mod._sin_full), "main-rope sin table changed"


def test_yarn_moves_position_one_too_so_short_context_is_not_exempt():
    """YaRN rescales frequencies, not positions, so even pos 1 rotates differently.

    If this ever passes, the implementation became position-gated and the compress path
    would disagree with the reference at every short decode step.
    """
    cfg = real_rope_config()
    _, _, p = v4_rope.build_rope(cfg, "compress", HEAD_DIM)
    plain = v4_rope.plain_inv_freq(p.rope_theta, ROPE_DIM)
    yarn = v4_rope.yarn_inv_freq(
        p.rope_theta, ROPE_DIM, p.factor, p.original_max_position_embeddings, p.beta_fast, p.beta_slow
    )
    rows = 65_537  # one past the pre-training window, which is the interesting boundary
    cos_plain, sin_plain = v4_rope.cos_sin_tables(plain, 1.0, rows, ROPE_DIM)
    cos_yarn, sin_yarn = v4_rope.cos_sin_tables(yarn, 1.0, rows, ROPE_DIM)

    # Frequencies differ by up to 16x, so no position can agree exactly. Measured ladder of
    # max|cos_yarn - cos_plain|: ~5e-7 at 1, 2.3e-3 at 64, 0.32 at 1000, 1.87 at 65536,
    # ~1.9 beyond. Disagreement therefore grows with context but NOT monotonically at small
    # positions (an earlier draft of this test asserted growth from 1 to 7 and failed:
    # different frequency bands peak at different places in the small-angle regime).
    ratio = yarn / plain
    assert abs(float(ratio.max()) - 1.0) < 1e-6, float(ratio.max())
    assert abs(float(ratio.min()) - 1.0 / 16.0) < 1e-5, float(ratio.min())

    ladder = []
    for pos in (1, 64, 1000, 65536):
        delta = float(max((cos_yarn[pos] - cos_plain[pos]).abs().max(), (sin_yarn[pos] - sin_plain[pos]).abs().max()))
        ladder.append(delta)
        assert delta > 0.0, f"position {pos} agrees exactly"
    assert ladder == sorted(ladder) and ladder[0] < ladder[-1], ladder
    # Past the pre-training window the two are essentially decorrelated, which is the whole
    # reason no long-context claim is admissible without YaRN.
    assert ladder[-1] > 1.0, ladder


def test_correction_band_is_interpolated_above_and_extrapolated_below():
    """The inverted version of this was the first draft's bug; pin the direction."""
    yarn = v4_rope.yarn_inv_freq(160000.0, ROPE_DIM, 16.0, 65536)
    plain = v4_rope.plain_inv_freq(160000.0, ROPE_DIM)
    ratio = (yarn / plain).tolist()
    # Low indices keep ratio 1.0 (extrapolated), high indices approach 1/16 (interpolated).
    assert abs(ratio[0] - 1.0) < 1e-6, f"lowest dim scaled: {ratio[0]}"
    assert ratio[-1] < ratio[0], f"band direction inverted: {ratio[:3]} ... {ratio[-3:]}"
    assert 1.0 / 16.0 - 1e-6 <= ratio[-1] <= 1.0 + 1e-6


def test_truncation_changes_the_ramp_start(expect_error):
    trunc = v4_rope.yarn_inv_freq(160000.0, ROPE_DIM, 16.0, 65536, truncate=True)
    exact = v4_rope.yarn_inv_freq(160000.0, ROPE_DIM, 16.0, 65536, truncate=False)
    assert not torch.equal(trunc, exact), "truncate flag is not wired through"
    low, high = v4_rope._correction_range(32, 1, ROPE_DIM, 160000.0, 65536, True)
    assert 0 <= low < high <= ROPE_DIM - 1, (low, high)
    with expect_error(ValueError, "positive even"):
        v4_rope.plain_inv_freq(10000.0, 15)


def test_unknown_rope_type_is_refused_not_silently_plain(expect_error):
    cfg = real_rope_config(
        rope_parameters={
            "main": {"rope_type": "something_novel", "rope_theta": 10000, "partial_rotary_factor": 0.125},
            "compress": dict(REAL_ROPE_PARAMETERS["compress"]),
        }
    )
    with expect_error(NotImplementedError, "is not implemented"):
        v4_rope.build_rope(cfg, "main", HEAD_DIM)


def test_group_theta_falls_back_to_the_named_field_never_another_groups_theta():
    """Popping the injected theta must reproduce the same numbers, not main's theta.

    Inheriting the top-level ``rope_theta`` for the compress group would rotate it with
    10000 instead of 160000 -- a rope change that no short-context test would notice.
    """
    full = real_rope_config()
    broken = {k: dict(v) for k, v in REAL_ROPE_PARAMETERS.items()}
    broken["compress"].pop("rope_theta")
    cfg = real_rope_config(rope_parameters=broken)

    inv_full, _af, _p = v4_rope.build_rope(full, "compress", HEAD_DIM)
    inv_broken, _af2, params = v4_rope.build_rope(cfg, "compress", HEAD_DIM)
    assert params.rope_theta == 160000.0, params.rope_theta
    assert torch.equal(inv_full, inv_broken), "fallback theta changed the compress rope"

    # main may legitimately use config.rope_theta: it is main's own field.
    broken_main = {k: dict(v) for k, v in REAL_ROPE_PARAMETERS.items()}
    broken_main["main"].pop("rope_theta")
    cfg2 = real_rope_config(rope_parameters=broken_main)
    _inv, _af, params2 = v4_rope.build_rope(cfg2, "main", HEAD_DIM)
    assert params2.rope_theta == 10000.0


def test_no_theta_at_all_refuses_instead_of_inheriting(expect_error):
    stub = SimpleNamespace(rope_parameters={"compress": {"rope_type": "yarn"}}, rope_theta=10000)
    with expect_error(ValueError, "refusing to inherit"):
        v4_rope.RopeParams.from_config(stub, "compress")


def test_attention_factor_is_derived_when_the_config_omits_it():
    """V4-Flash ships 1.0 explicitly; other checkpoints may not."""
    no_factor = {
        k: {kk: vv for kk, vv in v.items() if kk != "attention_factor"} for k, v in REAL_ROPE_PARAMETERS.items()
    }
    cfg = real_rope_config(rope_parameters=no_factor)
    _, factor, params = v4_rope.build_rope(cfg, "compress", HEAD_DIM)
    assert abs(params.attention_factor - v4_rope._mscale(16.0)) < 1e-9
    assert abs(factor - v4_rope._mscale(16.0)) < 1e-6
    assert v4_rope._mscale(1.0) == 1.0


def test_cos_sin_tables_reject_a_count_mismatch(expect_error):
    inv = v4_rope.plain_inv_freq(10000.0, 64)
    with expect_error(ValueError, "cannot cover rotating dim"):
        v4_rope.cos_sin_tables(inv, 1.0, 4, 32)


# ------------------------------------------------- tier 2: the recorded values


@needs_ckpt
def test_recorded_rope_parameters_match_the_checkpoint():
    with open(os.path.join(SNAP, "config.json")) as fh:
        cfg = json.load(fh)
    built = DeepseekV4Config(**cfg)
    rp = built.rope_parameters
    assert set(rp) == set(REAL_ROPE_PARAMETERS), sorted(rp)
    for group, expected in REAL_ROPE_PARAMETERS.items():
        for key, want in expected.items():
            if key == "type":  # legacy alias kept alongside the normalised rope_type
                continue
            got = rp[group].get(key)
            assert got is not None, (group, key)
            if isinstance(want, str):
                assert got == want, (group, key, got, want)
            else:
                assert abs(float(got) - float(want)) < 1e-9, (group, key, got, want)
    assert (
        rp["main"]["rope_theta"] != rp["compress"]["rope_theta"]
    ), "the two-group theta difference is the whole point of this file"


@needs_ckpt
def test_real_config_rope_matches_the_recorded_geometry_bitwise():
    """End to end on the real config: same inv_freq as the inline recorded values."""
    with open(os.path.join(SNAP, "config.json")) as fh:
        real = DeepseekV4Config(**json.load(fh))
    real_head = real.head_dim or real.hidden_size // real.num_attention_heads
    assert real_head == HEAD_DIM, real_head
    recorded = real_rope_config()
    for group in ("main", "compress"):
        a, af, _ = v4_rope.build_rope(real, group, real_head)
        b, bf, _ = v4_rope.build_rope(recorded, group, HEAD_DIM)
        assert torch.equal(a, b), f"{group} inv_freq differs from the recorded values"
        assert af == bf, (group, af, bf)
