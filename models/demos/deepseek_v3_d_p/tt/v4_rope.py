"""RoPE tables for V4-Flash: two parameter groups, not one.

``DeepseekV4Config`` exposes ``rope_parameters`` keyed by **group**, and the checkpoint's
real values are:

``main``     ``rope_type=default``, ``rope_theta=10000``,  ``partial_rotary_factor=0.125``
``compress`` ``rope_type=yarn``,    ``rope_theta=160000``, ``partial_rotary_factor=0.125``,
             ``factor=16``, ``original_max_position_embeddings=65536``, ``beta_fast=32``,
             ``beta_slow=1``, ``attention_factor=1.0``

Two things are easy to get wrong here, and both were wrong in an earlier note of mine:

* Reading ``rope_theta`` and ``rope_scaling`` off the top level of ``config.json`` makes the
  model look like one plain-rope model with a YaRN tweak. There are **two** thetas, 10000
  and 160000, and YaRN applies **only** to the compress group. Going through the config
  class rather than the raw JSON is what reveals the structure.
* YaRN's correction band is the opposite way round from intuition. In
  ``_compute_yarn_parameters`` the frequencies *below* the band keep their extrapolated
  (unscaled) value and the ones *above* it are interpolated (divided by ``factor``); the
  first version of this module had it inverted, which the pinned cross-check against
  ``transformers`` now prevents.

``partial_rotary_factor`` 0.125 of ``head_dim`` 512 is 64, which is exactly
``qk_rope_head_dim``: only the rope half of the head rotates, the nope half never does.

``attention_factor`` is 1.0 here, so no cos/sin amplitude scaling is needed, but it is
returned rather than hardcoded because it is a config input and a future checkpoint may not
ship 1.0.

The parity requirement is to match ``transformers``' rope functions, because the reference
``deepseek_v4`` module reaches them through ``rope_init_fn(config, layer_type=...)`` and is
our oracle. :func:`build_rope` therefore **delegates** to them. :func:`yarn_inv_freq` is a
self-contained reimplementation for paths that must not import ``transformers`` (and for
reading the algorithm), and its equality with HF is a pinned test rather than an assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RopeParams:
    """One rope parameter group, as it appears in ``config.rope_parameters``."""

    rope_type: str
    rope_theta: float
    partial_rotary_factor: float
    factor: float = 1.0
    original_max_position_embeddings: int = 0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float = 1.0
    truncate: bool = True

    @staticmethod
    def from_config(config, group: str) -> "RopeParams":
        params = config.rope_parameters[group]
        # A group without its own theta must NOT inherit the top-level one: for V4-Flash the
        # two thetas are 10000 and 160000, so inheriting is a silent rope change. The only
        # acceptable defaults are the ones named for the group -- main <- rope_theta,
        # compress <- compress_rope_theta -- which is how DeepseekV4Config injects them for
        # the real checkpoint, whose config.json groups carry no theta at all.
        theta = params.get("rope_theta")
        if theta is None:
            theta = (
                getattr(config, "rope_theta", None) if group == "main" else getattr(config, f"{group}_rope_theta", None)
            )
        if theta is None:
            raise ValueError(
                f"rope group {group!r} has no rope_theta and config exposes no "
                f"{'rope_theta' if group == 'main' else f'{group}_rope_theta'} to stand in; "
                "refusing to inherit a different group's theta"
            )
        return RopeParams(
            rope_type=params.get("rope_type") or "default",
            rope_theta=float(theta),
            partial_rotary_factor=float(params.get("partial_rotary_factor", 1.0)),
            factor=float(params["factor"]) if params.get("factor") is not None else 1.0,
            original_max_position_embeddings=int(params["original_max_position_embeddings"])
            if params.get("original_max_position_embeddings") is not None
            else 0,
            beta_fast=float(params.get("beta_fast") or 32.0),
            beta_slow=float(params.get("beta_slow") or 1.0),
            attention_factor=float(params["attention_factor"])
            if params.get("attention_factor") is not None
            else _mscale(float(params["factor"]) if params.get("factor") else 1.0),
            truncate=bool(params.get("truncate", True)),
        )


def _mscale(scale: float, mscale: float = 1.0) -> float:
    """The paper's attention scaling, used only when config omits ``attention_factor``."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def rope_dim(head_dim: int, partial_rotary_factor: float) -> int:
    """Rotating width. ``int(head_dim * factor)`` per the reference module, not rounding."""
    return int(head_dim * partial_rotary_factor)


def plain_inv_freq(theta: float, dim: int) -> torch.Tensor:
    """Un-scaled inverse frequencies, Float32 ``[dim // 2]``.

    Same expression the reference uses and the one ``TtV4SlidingAttention`` already builds,
    so the main-rope path is unchanged by this module existing.
    """
    if dim <= 0 or dim % 2:
        raise ValueError(f"rope dim must be a positive even number, got {dim}")
    return theta ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)


def _correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: float):
    """Dimension at which a position window admits ``num_rotations`` full rotations."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _correction_range(beta_fast: float, beta_slow: float, dim: int, base: float, original_max_pos: int, truncate: bool):
    """Band of dimensions YaRN interpolates, clamped to ``[0, dim - 1]``.

    ``beta_fast`` is the **low** bound and ``beta_slow`` the high one: many rotations happen
    at low dimension indices. With ``truncate`` the bounds are floored/ceil'd, matching the
    reference; without it the bounds stay fractional and the ramp starts mid-dimension.
    """
    low = _correction_dim(beta_fast, dim, base, original_max_pos)
    high = _correction_dim(beta_slow, dim, base, original_max_pos)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def yarn_inv_freq(
    theta: float,
    dim: int,
    factor: float,
    original_max_pos: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    truncate: bool = True,
) -> torch.Tensor:
    """YaRN-scaled inverse frequencies, Float32 ``[dim // 2]``.

    Above the correction band the frequencies are interpolated (divided by ``factor``);
    below it they keep their extrapolated value; inside the band they interpolate between
    the two. The result is **not** position-gated, so it changes cos/sin at position 1 as
    much as at 100k -- which is why "short context is unaffected" is tested numerically
    rather than asserted.
    """
    if dim <= 0 or dim % 2:
        raise ValueError(f"rope dim must be a positive even number, got {dim}")
    pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_extrapolation = 1.0 / pos_freqs
    inv_interpolation = 1.0 / (factor * pos_freqs)

    low, high = _correction_range(beta_fast, beta_slow, dim, theta, original_max_pos, truncate)
    if low == high:  # reference guards the degenerate ramp the same way
        high = low + 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0.0, 1.0)
    extrapolation_factor = 1.0 - ramp
    return inv_interpolation * (1.0 - extrapolation_factor) + inv_extrapolation * extrapolation_factor


def build_rope(config, group: str, head_dim: int):
    """``(inv_freq, attention_factor, params)`` for one parameter group.

    Delegates to ``transformers``' registered rope functions with ``layer_type=group``,
    exactly as the reference module does, so the oracle and this module cannot drift.
    Unknown rope types raise rather than silently falling back to plain rope: a silent
    fallback is how a rope change becomes a correctness bug that only shows up long after
    the 65536-token boundary the scaling was fitted to.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    params = RopeParams.from_config(config, group)
    dim = rope_dim(head_dim, params.partial_rotary_factor)
    if params.rope_type == "default":
        return plain_inv_freq(params.rope_theta, dim), params.attention_factor, params
    if params.rope_type not in ROPE_INIT_FUNCTIONS:
        raise NotImplementedError(
            f"rope_type {params.rope_type!r} for group {group!r} is not implemented; "
            "refusing to substitute plain rope"
        )
    if config.rope_parameters[group].get("rope_theta") is None:
        # DeepseekV4Config normally injects per-group thetas, but HF's own rope functions
        # require one and raise a bare KeyError without it. Since we resolved the theta from
        # the config's named field, compute it here instead of handing HF something it cannot
        # read. yarn_inv_freq is pinned equal to HF's function elsewhere in the suite, so this
        # is the same arithmetic either way.
        if params.rope_type == "yarn":
            return (
                yarn_inv_freq(
                    params.rope_theta,
                    dim,
                    params.factor,
                    params.original_max_position_embeddings,
                    params.beta_fast,
                    params.beta_slow,
                    params.truncate,
                ),
                params.attention_factor,
                params,
            )
        raise ValueError(
            f"rope group {group!r} carries no rope_theta and {params.rope_type!r} scaling "
            "cannot be computed locally; refusing to guess"
        )
    init = ROPE_INIT_FUNCTIONS[params.rope_type]
    inv_freq, attention_factor = init(config, layer_type=group)
    inv_freq = inv_freq.to(torch.float32).cpu()
    if inv_freq.numel() != dim // 2:
        raise ValueError(f"group {group!r}: rope produced {inv_freq.numel()} freqs for rotating dim {dim}")
    return inv_freq, float(attention_factor), params


def cos_sin_tables(inv_freq: torch.Tensor, attention_factor: float, max_seq_len: int, rotating_dim: int):
    """Interleaved cos/sin tables ``[max_seq, rotating_dim]`` in the reference's layout.

    One entry per pair, expanded by ``repeat_interleave(2)`` at use time, because the
    reference's ``apply_rotary_pos_emb`` pairs adjacent elements instead of half-splitting.
    """
    if inv_freq.numel() * 2 != rotating_dim:
        raise ValueError(f"{inv_freq.numel()} frequencies cannot cover rotating dim {rotating_dim}")
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = positions.unsqueeze(-1) * inv_freq
    cos = (freqs.cos() * attention_factor).repeat_interleave(2, dim=-1)
    sin = (freqs.sin() * attention_factor).repeat_interleave(2, dim=-1)
    return cos, sin
