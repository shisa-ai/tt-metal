# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Composition math for the V4 decoder layer, settled before any device code.

The device layer is glue around ``TtHCMapping``, ``apply_mhc_site``, ``TtHCA`` and
``TtMoe``. Those calls need hardware, but the *composition* around them -- what
flows between sublayers, and how a sublayer output re-enters the stream stack --
is plain tensor algebra and can be settled on host now. Both of its plausible
failure modes are silent:

* carrying ``[B, S, D]`` between blocks and re-expanding, instead of threading the
  ``[B, S, hc_mult, D]`` stack the reference actually carries; and
* applying ``comb`` un-transposed. Sinkhorn output is doubly stochastic but
  generally **asymmetric**, so ``comb`` and ``combᵀ`` both look valid and only one
  is V4.

Method: the reference block's own ``forward`` is the oracle. Its sublayers are
replaced with stubs that accept its kwargs and return a deterministic function of
their input, which reduces ``DeepseekV4DecoderLayer.forward`` to pure composition
code that is *the reference's*, not mine. The mirror here transcribes what
``tt/mhc.py`` plus the layer will do. Non-composition bugs cannot flatter the
comparison, and the discriminating tests below assert that wrong variants
genuinely differ -- a mirror test that passes for every variant measures nothing.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer
from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs


# ---- transcription of tt/mhc.py::apply_mhc_site -----------------------------


def mix_in(post: torch.Tensor, comb: torch.Tensor, streams: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``new_streams = post[...,None] * out[...,None,:] + combᵀ @ streams``.

    Shapes: post [B,S,H], comb [B,S,H,H], streams [B,S,H,D], out [B,S,D].
    """
    return post.unsqueeze(-1) * out.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), streams)


def mix_in_untransposed(post, comb, streams, out):
    """Wrong variant: comb applied without the transpose."""
    return post.unsqueeze(-1) * out.unsqueeze(-2) + torch.matmul(comb, streams)


def block_body(hc_in, hc_out, norm_in, norm_out, attn, mlp, streams):
    """The block, transcribed the way the reference threads it.

    Returns the next stream stack -- not a [B,S,D] hidden state -- which is the
    entire point of an mHC block.
    """
    post, comb, collapsed = hc_in(streams)
    streams = mix_in(post, comb, streams, attn(norm_in(collapsed)))

    post, comb, collapsed = hc_out(streams)
    return mix_in(post, comb, streams, mlp(norm_out(collapsed)))


def block_body_flat_carry(hc_in, hc_out, norm_in, norm_out, attn, mlp, streams):
    """Wrong variant: collapse to one stream between the two sublayers."""
    post, comb, collapsed = hc_in(streams)
    mixed = mix_in(post, comb, streams, attn(norm_in(collapsed)))
    h = mixed.mean(dim=-2)  # [B,S,D], as a naive port would carry
    wide = h.unsqueeze(-2).expand_as(mixed)
    post2, comb2, collapsed2 = hc_out(wide)
    return mix_in(post2, comb2, wide, mlp(norm_out(collapsed2)))


class _Stub:
    """Deterministic sublayer that swallows the reference's kwargs.

    Nonlinear and input-dependent, so it cannot be mistaken for an identity that
    would hide a wiring error.
    """

    def __init__(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.w = None
        self._g = g
        self._seed = seed

    def _weight(self, d: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self._seed)
        return torch.randn(d, d, generator=g) / d**0.5

    def __call__(self, x: torch.Tensor, **_kwargs):
        w = self._weight(x.shape[-1]).to(x.device, x.dtype)
        return torch.tanh(x @ w)


@pytest.fixture(scope="module")
def block():
    args = V4ModelArgs.tiny(1)
    cfg = DeepseekV4Config(**args.drive_reference().__dict__)
    torch.manual_seed(0)
    lay = DeepseekV4DecoderLayer(cfg, layer_idx=0).to(torch.float32).eval()
    b, s, hc, d = 1, 4, cfg.hc_mult, cfg.hidden_size
    torch.manual_seed(1)
    streams = torch.randn(b, s, hc, d)
    return cfg, lay, streams


def _reference_block(lay, streams, seed_attn=11, seed_mlp=12):
    """Run the reference's own forward with sublayers stubbed out."""
    class Stub(torch.nn.Module):
        def __init__(self, seed, as_tuple):
            super().__init__()
            self.s = Stub._mk(seed)
            self.as_tuple = as_tuple

        @staticmethod
        def _mk(seed):
            return seed

        def forward(self, x, **_kwargs):
            d = x.shape[-1]
            g = torch.Generator().manual_seed(self.s)
            w = torch.randn(d, d, generator=g).to(x.device, x.dtype) / d**0.5
            out = torch.tanh(x @ w)
            return (out, None) if self.as_tuple else out

    real_attn, real_mlp = lay.self_attn, lay.mlp
    lay.self_attn = Stub(seed_attn, as_tuple=True)
    lay.mlp = Stub(seed_mlp, as_tuple=False)
    try:
        return lay.forward(streams)
    finally:
        lay.self_attn, lay.mlp = real_attn, real_mlp


def _mirror_block(lay, streams, seed_attn=11, seed_mlp=12):
    a = _Stub(seed_attn)
    m = _Stub(seed_mlp)
    # stubs must see the normalised collapsed stream and return the same tensor
    attn = lambda x: a(x)
    mlp = lambda x: m(x)
    return block_body(lay.attn_hc, lay.ffn_hc, lay.input_layernorm, lay.post_attention_layernorm, attn, mlp, streams)


def test_mirror_reproduces_the_reference_block_composition(block):
    cfg, lay, streams = block
    with torch.no_grad():
        reference = _reference_block(lay, streams)
        ours = _mirror_block(lay, streams)
    assert ours.shape == reference.shape, (tuple(ours.shape), tuple(reference.shape))
    diff = float((ours - reference).abs().max())
    assert torch.allclose(ours, reference, rtol=1e-5, atol=1e-6), f"max abs diff {diff:.3e}"


def test_block_returns_a_stream_stack_not_a_single_hidden_state(block):
    cfg, lay, streams = block
    with torch.no_grad():
        out = _mirror_block(lay, streams)
    assert out.dim() == 4, tuple(out.shape)
    assert out.shape == streams.shape, (
        f"block must carry [B,S,hc_mult,D] = {tuple(streams.shape)}, got {tuple(out.shape)}"
    )


def test_comb_transpose_is_load_bearing():
    """A direct math test of mix_in, with comb built deliberately asymmetric.

    Deliberately NOT driven by model init: at initialisation the reference's
    Sinkhorn output is uniform/symmetric, so ``comb^T @ streams`` equals
    ``comb @ streams`` and no init-time parity check could detect an un-transposed
    comb. That is asserted separately below, and it is the reason parity must use
    asymmetric comb matrices rather than untouched initialisation.
    """
    torch.manual_seed(0)
    b, s, h, d = 1, 3, 4, 8
    streams = torch.randn(b, s, h, d)
    out = torch.randn(b, s, d)
    post = torch.rand(b, s, h) + 0.25
    # Random row-stochastic -> asymmetric with probability 1.
    comb = torch.softmax(torch.randn(b, s, h, h) * 3.0, dim=-1)
    assert not torch.allclose(comb, comb.transpose(-1, -2), atol=1e-6), "fixture must be asymmetric"

    right = mix_in(post, comb, streams, out)
    wrong = mix_in_untransposed(post, comb, streams, out)
    assert right.shape == wrong.shape
    assert not torch.allclose(right, wrong, rtol=1e-4, atol=1e-6), (
        "mix_in cannot distinguish comb from comb^T -- the transpose is not "
        "actually applied"
    )


def _fresh_layer(seed: int):
    """Build the block with RNG pinned immediately before construction.

    The module-scoped `block` fixture is NOT reproducible for this question: three
    identical runs of this file gave skip / pass / skip, because global RNG state at
    fixture creation depends on what ran before it. Anything that asks "is comb
    symmetric at init?" must seed at the point of construction.
    """
    args = V4ModelArgs.tiny(1)
    cfg = DeepseekV4Config(**args.drive_reference().__dict__)
    torch.manual_seed(seed)
    lay = DeepseekV4DecoderLayer(cfg, layer_idx=0).to(torch.float32).eval()
    # The reference's _init_weights does NOT fill mhc.base (nor several others):
    # two builds under the same seed differed by 1.86e+37 in attn_hc.base, i.e. the
    # mHC residual mixing came from uninitialised memory. Anything comparing against
    # this block must set them, or the oracle is not reproducible run to run.
    g = torch.Generator().manual_seed(seed)
    for name, mod in (("attn_hc", lay.attn_hc), ("ffn_hc", lay.ffn_hc)):
        with torch.no_grad():
            for pname in ("base", "scale", "fn"):
                param = getattr(mod, pname)
                if pname == "base":
                    param.normal_(0.0, 0.02, generator=g)
                elif pname == "scale":
                    param.fill_(1.0)
                else:
                    param.normal_(0.0, 0.02, generator=g)
    return cfg, lay


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_init_comb_may_be_symmetric_so_init_parity_is_not_a_reliable_guard(seed):
    """An un-transposed comb can be invisible at initialisation -- deterministically checked.

    Not "skip if symmetric": the hazard IS that detection depends on initialisation,
    so this measures symmetry per seed and records it. The load-bearing assertion
    lives in the synthetic asymmetric-comb test, which is reproducible.
    """
    cfg, lay = _fresh_layer(seed)
    b, s, hc, d = 1, 4, cfg.hc_mult, cfg.hidden_size
    torch.manual_seed(seed + 100)
    streams = torch.randn(b, s, hc, d)
    with torch.no_grad():
        _, comb, _ = lay.attn_hc(streams)
    symmetric = bool(torch.allclose(comb, comb.transpose(-1, -2), atol=1e-6))

    # Whatever the answer, it must be reproducible for this seed.
    cfg2, lay2 = _fresh_layer(seed)
    with torch.no_grad():
        _, comb2, _ = lay2.attn_hc(streams)
    assert torch.equal(comb, comb2), "comb is not reproducible for a fixed seed -- fixture is unsafe"
    assert bool(torch.allclose(comb2, comb2.transpose(-1, -2), atol=1e-6)) == symmetric

    if symmetric:
        import warnings
        warnings.warn(
            f"seed {seed}: comb is symmetric at init, so init-time parity cannot "
            "detect an un-transposed comb for this initialisation",
            stacklevel=2,
        )


def test_flat_carry_port_is_observably_wrong():
    """Collapsing the stream stack between blocks must change the result.

    Synthetic and asymmetric for the same reason as the transpose test -- with a
    uniform init ``comb`` and identical streams the two coincide, which is exactly
    the case a real run must not be validated on.
    """
    torch.manual_seed(1)
    b, s, h, d = 1, 3, 4, 8
    eps = 1e-6

    class FakeHC:
        """Stands in for TtHCMapping: returns (post, comb, collapsed, streams)."""

        def __init__(self, seed):
            g = torch.Generator().manual_seed(seed)
            self.post = torch.rand(b, s, h, generator=g) + 0.2
            self.comb = torch.softmax(torch.randn(b, s, h, h, generator=g) * 3.0, dim=-1)
            self.pre = torch.softmax(torch.randn(b, s, 1, h, generator=g) * 2.0, dim=-1)

        def __call__(self, streams):
            # 3-tuple, matching the reference HyperConnection that block_body
            # transcribes; TtHCMapping adds a 4th pass-through value.
            collapsed = torch.matmul(self.pre, streams)
            return self.post, self.comb, collapsed

    def sublayer(x):
        return torch.tanh(x * 1.7 + 0.1)

    streams = torch.randn(b, s, h, d)
    right = block_body(FakeHC(101), FakeHC(102), lambda x: x, lambda x: x, sublayer, sublayer, streams)
    wrong = block_body_flat_carry(FakeHC(101), FakeHC(102), lambda x: x, lambda x: x, sublayer, sublayer, streams)
    assert right.shape == wrong.shape
    assert not torch.allclose(right, wrong, rtol=1e-3, atol=1e-5), (
        "the flat-carry variant is indistinguishable, so this suite would not "
        "catch collapsing the stream stack between blocks"
    )
