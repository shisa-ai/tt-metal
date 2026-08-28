# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Reproducibility of the V4 reference oracle, and where it breaks.

Tests written because an earlier claim of mine got the *mechanism* wrong while the
observation was real: a same-seed rebuild differing by 1.86e+37 looked like "the
reference oracle is not reproducible". It is — for the full model. What is not
reproducible is a **barely-constructed sublayer**, because ``DeepseekV4DecoderLayer``
is a plain module, not a ``PreTrainedModel``, so building it alone never runs
``init_weights()`` and its ``torch.empty`` parameters stay uninitialised.

That distinction decides whether parity numbers are trustworthy, so it is asserted
here in both directions rather than asserted once.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
)
from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs
from models.demos.deepseek_v3_d_p.tt.v4_oracle import (
    build_reference_oracle,
    prefill_logits,
    weight_fingerprint,
)


@pytest.fixture(scope="module")
def args():
    return V4ModelArgs.tiny(1)


@pytest.fixture(scope="module")
def cfg(args):
    return DeepseekV4Config(**args.drive_reference().__dict__)


def _max_delta(a: torch.nn.Module, b: torch.nn.Module) -> float:
    return max(float((x.detach() - y.detach()).abs().max()) for x, y in zip(a.parameters(), b.parameters()))


def _build_layer(cfg, seed: int):
    torch.manual_seed(seed)
    return DeepseekV4DecoderLayer(cfg, layer_idx=0).to(torch.float32).eval()


def _build_model(cfg, seed: int):
    torch.manual_seed(seed)
    return DeepseekV4ForCausalLM(cfg).to(torch.float32).eval()


def test_bare_sublayer_construction_is_not_reproducible(cfg):
    """The hazard: a standalone block runs against uninitialised mHC parameters."""
    delta = _max_delta(_build_layer(cfg, 0), _build_layer(cfg, 0))
    assert delta > 1e6, (
        f"expected a large same-seed delta from uninitialised parameters, got {delta:.3e}; "
        "if this is now zero, whatever fills them changed and the note in v4_oracle.py "
        "should be revisited"
    )


def test_full_model_construction_is_reproducible(cfg):
    """The correction: PreTrainedModel runs init_weights, so the model is stable."""
    assert _max_delta(_build_model(cfg, 0), _build_model(cfg, 0)) == 0.0


def test_reference_init_leaves_mhc_mixing_neutral(cfg):
    """base=0 / scale=1 is why the oracle deviates on purpose."""
    model = _build_model(cfg, 0)
    params = dict(model.named_parameters())
    names = [n for n in params if n.endswith("attn_hc.base")]
    assert names, "no attn_hc.base found -- parameter naming changed"
    assert all(bool((params[n] == 0).all()) for n in names)
    scales = [n for n in params if n.endswith("attn_hc.scale")]
    assert all(bool((params[n] == 1).all()) for n in scales)


def test_oracle_is_reproducible_and_seed_sensitive(args):
    m1, f1 = build_reference_oracle(args, seed=0)
    m2, f2 = build_reference_oracle(args, seed=0)
    _, f3 = build_reference_oracle(args, seed=1)
    assert f1 == f2, "same seed must reproduce the fingerprint"
    assert f1 != f3, "different seeds must differ, else the fingerprint is not observing weights"
    assert weight_fingerprint(m1) == weight_fingerprint(m2)


def test_oracle_delegates_its_mhc_fill_away_from_neutral(args):
    """The deliberate deviation, asserted rather than assumed."""
    model, _ = build_reference_oracle(args, seed=0)
    params = dict(model.named_parameters())
    bases = [params[n] for n in params if n.endswith("attn_hc.base")]
    assert bases and any(not bool((b == 0).all()) for b in bases), (
        "oracle left attn_hc.base at the reference's neutral zero, so a mis-wired "
        "mHC mix could go undetected -- the whole point of the explicit fill"
    )


def test_oracle_logits_are_reproducible(args):
    model, _ = build_reference_oracle(args, seed=0)
    ids = torch.randint(0, args.vocab_size, (1, 32), generator=torch.Generator().manual_seed(5))
    first = prefill_logits(model, ids)
    second = prefill_logits(model, ids)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second), "same model + same ids must give identical logits"
    assert float(first.std()) > 1e-4, "logits are degenerate; the fixture is not exercising the head"


def test_oracle_survives_a_longer_prefill(args):
    """>=128 tokens, the length the HCA port requires; guards against a fixture that
    only works on toy lengths the device path cannot use."""
    model, _ = build_reference_oracle(args, seed=0)
    ids = torch.randint(0, args.vocab_size, (1, 128), generator=torch.Generator().manual_seed(6))
    logits = prefill_logits(model, ids)
    assert logits.shape[-1] == args.vocab_size
    assert torch.isfinite(logits).all()
