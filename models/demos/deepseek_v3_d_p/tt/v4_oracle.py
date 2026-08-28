# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Reproducible reference oracle for V4-Flash parity work.

Why this exists at all — and the honest reason, after a first explanation of mine
turned out to be wrong.

The **full model** path is reproducible: ``DeepseekV4ForCausalLM(cfg)`` under a
fixed seed twice gives bit-identical parameters (measured max delta 0.000e+00),
because ``PreTrainedModel.__init__`` runs ``post_init()`` -> ``init_weights()``,
whose ``DeepseekV4HyperConnection`` branch sets ``fn ~ N(0, std)``, ``base = 0``,
``scale = 1``.

What is **not** reproducible is constructing a bare sublayer.
``DeepseekV4DecoderLayer`` is a plain ``GradientCheckpointingLayer``, not a
``PreTrainedModel``, so building it standalone never triggers ``init_weights``;
its ``torch.empty`` parameters stay uninitialised — two same-seed bare-layer builds
differed by **1.86e+37** in ``attn_hc.base``. Any test that exercises a sublayer in
isolation is therefore running against garbage unless it sets weights itself.

So this module serves two purposes: it gives parity work a **fingerprinted** weight
set (the evidence policy asks for a resolved fingerprint anyway), and its explicit
fill makes standalone-layer testing safe by construction.

Deliberate deviation from reference init: this fills ``base`` and ``scale`` with
non-degenerate values instead of ``0``/``1``. Reference init leaves the residual
mixing neutral, which is a weak test of mHC wiring; a randomised ``base`` makes a
mis-wired mix change the numbers. Say which was used in any comparison.

So this builds the model, then **overwrites every floating-point parameter and
buffer from one seeded generator**, and returns a fingerprint of the resulting
weights. The fingerprint is what makes parity evidence self-describing — a run
record can say which weights it compared against, which the evidence policy asks
for anyway ("resolved revision/fingerprint").

Notes on the fill:

* Scale is ``config.initializer_range`` (0.02 default), applied per-tensor with
  fan-in scaling for square-ish projections, so logits stay sane without resembling
  a trained model — this is a **structural** oracle: correct shapes, correct
  dataflow, reproducible numbers, not a quality assessment.
* ``e_score_correction_bias`` stays zero (the reference registers it as a zeros
  buffer and the router semantics depend on that default), and ``tid2eid`` gets
  seeded indices so hash layers route somewhere non-degenerate — a zeros table
  sends every token to expert 0 and would leave most experts untested.
* Integer buffers are filled with seeded integers rather than left as ``empty``.
* fp32 throughout: the reference declares mHC, all RMSNorms, ``sinks`` and the
  router bias fp32-strict (worklog 2d029b). Any device-side dtype deviation must
  be declared at the comparison, not discovered as an unexplained gap.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs


def _fill_tensor(t: torch.Tensor, gen: torch.Generator, name: str, *, num_experts: int = 8) -> None:
    """Overwrite one tensor deterministically, respecting its dtype and role."""
    with torch.no_grad():
        if t.dtype in (torch.int64, torch.int32, torch.long):
            # Small range: keeps index-like buffers in range for small presets
            # without pretending to know each table's real bound.
            t.random_(0, 4, generator=gen)
            return
        if t.dtype is torch.bool:
            t.bernoulli_(0.5, generator=gen)
            return

        if name.endswith("e_score_correction_bias"):
            t.zero_()  # the router's semantics assume the documented zeros default
            return
        if name.endswith("tid2eid"):
            # Bounded by the expert count, NOT by any tensor dim: the table is
            # [vocab, top_k] so its shape says nothing about the valid index range,
            # and an out-of-range entry is an index error at forward time.
            t.random_(0, max(1, num_experts), generator=gen)
            return
        if name.endswith("inv_freq") or name.endswith("positions"):
            t.zero_()  # rotary tables are computed downstream; keep them inert
            return

        # Fan-in scaled so a deep stack neither vanishes nor explodes.
        fan_in = t.shape[-1] if t.dim() > 0 else 1
        std = (1.0 / (fan_in**0.5)) if fan_in > 1 else 0.02
        t.normal_(0.0, std, generator=gen)


def _walk_state(model, prefix: str = ""):
    """Every parameter and buffer, including non-persistent buffers.

    ``named_buffers()`` and ``state_dict()`` can both skip non-persistent buffers
    depending on the torch version, and rotary/``inv_freq`` style buffers are exactly
    where an unfilled value would hide. Recursing over the module tree is the only
    form that does not depend on that.
    """
    for name, param in model._parameters.items():
        if param is not None:
            yield f"{prefix}{name}", param
    for name, buf in model._buffers.items():
        if buf is not None:
            yield f"{prefix}{name}", buf
    for name, child in model._modules.items():
        if child is not None:
            yield from _walk_state(child, f"{prefix}{name}.")


def weight_fingerprint(model: torch.nn.Module) -> str:
    """SHA-256 over name-sorted parameter+buffer bytes, cast to fp32.

    Deterministic across processes; catches any uninitialised-memory difference,
    which is exactly the failure this module exists to prevent.
    """
    h = hashlib.sha256()
    entries = {name: t for name, t in _walk_state(model)}
    for name in sorted(entries.keys()):
        t = entries[name]
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def build_reference_oracle(args: "V4ModelArgs", seed: int = 0, dtype: torch.dtype = torch.float32):
    """Build the tiny/preset reference model with fully explicit weights.

    Returns ``(model, fingerprint)``. Same ``args`` + same ``seed`` must give the
    same fingerprint; assert it in the run record.
    """
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4ForCausalLM,
    )

    cfg = DeepseekV4Config(**args.drive_reference().__dict__)
    gen = torch.Generator().manual_seed(seed)

    # Materialise the module tree with no RNG dependence at all, then fill
    # everything ourselves; HF init is not trusted to be complete.
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(dtype).eval()

    for name, t in _walk_state(model):
        _fill_tensor(t, gen, name, num_experts=cfg.num_local_experts)

    return model, weight_fingerprint(model)


def prefill_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Last-position logits, in fp32, with the forward repeated for determinism.

    Returns only the final result; the discarded first pass exists so that
    "identical across calls" is a meaningful claim rather than a property of a
    cold-cache first call.
    """
    with torch.no_grad():
        model(input_ids)
        out = model(input_ids)
    return out.logits[0, -1].to(torch.float32)
