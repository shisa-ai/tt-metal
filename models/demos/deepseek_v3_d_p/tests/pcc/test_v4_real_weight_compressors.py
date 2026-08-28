"""Real checkpoint weights through the ported compressors and indexer.

Everything else in the V4 parity suite uses explicitly-filled synthetic parameters, which is
the right way to pin a contract but says nothing about the shipped numbers. This file loads the
actual released tensors for one CSA layer and one HCA layer and checks that the port and the
reference agree on them, which exercises things synthetic weights hide:

* the publisher's dtype choices — compressor `wkv`/`wgate` are **BF16** while `ape` is
  **float32**, so a port that assumes one dtype for a module gets it wrong here;
* an FP8-carried tensor on the indexer path (`indexer.wq_b` is `F8_E4M3` with an
  `F8_E8M0` block scale), dequantized through the port's own format rules;
* the two-series coefficient, visible as real shapes: CSA `ape` is `[4, 1024]` (= `2 *
  head_dim`) while HCA `ape` is `[128, 512]` (= `head_dim`), and the indexer's is
  `[4, 256]` (= `2 * index_head_dim`).

Snapshot resolution prefers ``DS4_V4_FLASH_DIR`` (the repo convention used by
``test_v4_weight_stream.py``) and falls back to the Hugging Face cache. The fallback is
deliberate: a real-weight check that only runs when someone remembers an environment variable
is a check that silently does not run, and that is how 11 tests in the sibling file sit
skipped on a machine that has the weights.

No device is involved. Parity is fp32 on CPU, with a bf16 agreement check as an observation.
"""

import os

import pytest
import torch
from transformers import AutoConfig

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACompressor,
    DeepseekV4HCACompressor,
    DeepseekV4Indexer,
)
from models.demos.deepseek_v3_d_p.tt.v4_compression import TtCSACompressor, TtHCACompressor
from models.demos.deepseek_v3_d_p.tt.v4_indexer import TtIndexer
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import V4Checkpoint

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots")


def resolve_snapshot():
    explicit = os.environ.get("DS4_V4_FLASH_DIR")
    if explicit and os.path.isdir(explicit):
        return explicit
    if os.path.isdir(HF_CACHE):
        snaps = sorted(
            os.path.join(HF_CACHE, d) for d in os.listdir(HF_CACHE) if os.path.isdir(os.path.join(HF_CACHE, d))
        )
        if snaps:
            return snaps[-1]
    return None


SNAP = resolve_snapshot()
pytestmark = pytest.mark.skipif(SNAP is None, reason="V4-Flash snapshot not present")

HIDDEN = 4096
HEAD_DIM = 512
INDEX_HEAD_DIM = 128
Q_LORA = 1024


@pytest.fixture(scope="module")
def ck():
    return V4Checkpoint(SNAP)


@pytest.fixture(scope="module")
def cfg():
    return AutoConfig.from_pretrained(SNAP, trust_remote_code=True)


@pytest.fixture(scope="module")
def layer_of(cfg):
    """First layer index per class, derived the way the model derives it."""
    first = {}
    for i, t in enumerate(cfg.layer_types):
        first.setdefault(t, i)
    return first


class StubIndexer(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def forward(self, *args, **kwargs):
        return self.indices


def compressor_tensors(ck, layer):
    base = f"layers.{layer}.attn.compressor."
    return {
        "kv_proj": ck.dequantized(base + "wkv.weight", torch.float32),
        "gate_proj": ck.dequantized(base + "wgate.weight", torch.float32),
        "position_bias": ck.dequantized(base + "ape", torch.float32),
        "kv_norm_weight": ck.dequantized(base + "norm.weight", torch.float32),
    }


def activations(tokens, seed=0):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(1, tokens, HIDDEN, generator=g, dtype=torch.float32)
    q_residual = torch.randn(1, tokens, Q_LORA, generator=g, dtype=torch.float32)
    pos = torch.arange(tokens).unsqueeze(0)
    return hs, q_residual, pos


# ---------------------------------------------------------------- shapes are facts


def test_checkpoint_shapes_fix_the_two_series_coefficients(ck, layer_of):
    csa, hca = layer_of["compressed_sparse_attention"], layer_of["heavily_compressed_attention"]
    ape_csa = ck.index[f"layers.{csa}.attn.compressor.ape"]
    ape_hca = ck.index[f"layers.{hca}.attn.compressor.ape"]
    assert ape_csa.shape == (4, 2 * HEAD_DIM), ape_csa.shape
    assert ape_hca.shape == (128, HEAD_DIM), ape_hca.shape
    assert ck.index[f"layers.{csa}.attn.compressor.wkv.weight"].shape == (2 * HEAD_DIM, HIDDEN)
    assert ck.index[f"layers.{hca}.attn.compressor.wkv.weight"].shape == (HEAD_DIM, HIDDEN)
    ape_idx = ck.index[f"layers.{csa}.attn.indexer.compressor.ape"]
    assert ape_idx.shape == (4, 2 * INDEX_HEAD_DIM), ape_idx.shape
    # Dtypes are mixed on purpose: the gates are BF16, the position bias is float32.
    assert ape_csa.dtype == "F32" and ape_hca.dtype == "F32"
    assert ck.index[f"layers.{csa}.attn.compressor.wkv.weight"].dtype == "BF16"
    assert ck.index[f"layers.{csa}.attn.indexer.wq_b.weight"].dtype == "F8_E4M3"


def test_reader_dequantizes_the_fp8_indexer_projection(ck, layer_of):
    csa = layer_of["compressed_sparse_attention"]
    w = ck.dequantized(f"layers.{csa}.attn.indexer.wq_b.weight", torch.float32)
    assert w.shape == (64 * INDEX_HEAD_DIM, Q_LORA), w.shape
    assert torch.isfinite(w).all(), "FP8 dequant produced a non-finite value"
    assert w.abs().max() < 50.0, w.abs().max()  # a wrong scale layout blows up here
    assert w.dtype is torch.float32


# --------------------------------------------------------------------- parity


def test_hca_compressor_with_real_weights_matches_the_reference(cfg, ck, layer_of):
    layer = layer_of["heavily_compressed_attention"]
    weights = compressor_tensors(ck, layer)
    torch.manual_seed(0)
    ref = DeepseekV4HCACompressor(cfg).to(torch.float32).eval()
    with torch.no_grad():
        ref.kv_proj.weight.copy_(weights["kv_proj"])
        ref.gate_proj.weight.copy_(weights["gate_proj"])
        ref.position_bias.copy_(weights["position_bias"])
        ref.kv_norm.weight.copy_(weights["kv_norm_weight"])
    ours = TtHCACompressor(cfg, weights)

    hs, _qr, pos = activations(3 * 128, seed=11)
    with torch.no_grad():
        ref_kv, ref_bias = ref(hs, None, pos, None, 0)
        our_kv, our_bias = ours.forward(hs, pos, compressed_len=ref_kv.shape[2])
    assert our_kv.shape == ref_kv.shape == (1, 1, 3, HEAD_DIM), (our_kv.shape, ref_kv.shape)
    diff = (our_kv - ref_kv).abs().max()
    assert torch.allclose(our_kv, ref_kv, rtol=1e-5, atol=1e-6), f"max diff {diff:.3e}"
    assert torch.equal(torch.isfinite(our_bias), torch.isfinite(ref_bias))
    # Scale sanity, expressed against the norm weight rather than a magic constant: the
    # compressor ends in RMSNorm-with-weight, so entries must stay the same order as that
    # weight. A swapped weight layout shows up as an order-of-magnitude blow-up, not a crash.
    scale = weights["kv_norm_weight"].abs().mean().item()
    observed = our_kv.float().abs().mean().item()
    assert 0.2 * scale < observed < 5.0 * scale, (observed, scale)


def test_csa_compressor_with_real_weights_matches_the_reference(cfg, ck, layer_of):
    layer = layer_of["compressed_sparse_attention"]
    weights = compressor_tensors(ck, layer)
    torch.manual_seed(0)
    ref = DeepseekV4CSACompressor(cfg).to(torch.float32).eval()
    with torch.no_grad():
        ref.kv_proj.weight.copy_(weights["kv_proj"])
        ref.gate_proj.weight.copy_(weights["gate_proj"])
        ref.position_bias.copy_(weights["position_bias"])
        ref.kv_norm.weight.copy_(weights["kv_norm_weight"])
    ref.indexer = StubIndexer(torch.zeros(1, 32, 1, dtype=torch.long))
    ours = TtCSACompressor(cfg, weights)

    hs, _qr, _pos = activations(32, seed=13)
    with torch.no_grad():
        ref_kv, _ = ref(hs, None, torch.arange(32).unsqueeze(0), None, 0)
        our_kv, _ = ours.forward(hs, None, compressed_len=0)
    assert our_kv.shape == ref_kv.shape == (1, 1, 8, HEAD_DIM), (our_kv.shape, ref_kv.shape)
    diff = (our_kv - ref_kv).abs().max()
    assert torch.allclose(our_kv, ref_kv, rtol=1e-5, atol=1e-6), f"max diff {diff:.3e}"
    assert torch.isfinite(our_kv).all()


def test_indexer_with_real_weights_selects_like_the_reference(cfg, ck, layer_of):
    layer = layer_of["compressed_sparse_attention"]
    base = f"layers.{layer}.attn.indexer."
    weights = {
        "kv_proj": ck.dequantized(base + "compressor.wkv.weight", torch.float32),
        "gate_proj": ck.dequantized(base + "compressor.wgate.weight", torch.float32),
        "position_bias": ck.dequantized(base + "compressor.ape", torch.float32),
        "kv_norm_weight": ck.dequantized(base + "compressor.norm.weight", torch.float32),
        "q_b_proj": ck.dequantized(base + "wq_b.weight", torch.float32),
        "weights_proj": ck.dequantized(base + "weights_proj.weight", torch.float32),
    }
    torch.manual_seed(0)
    ref = DeepseekV4Indexer(cfg).to(torch.float32).eval()
    with torch.no_grad():
        ref.kv_proj.weight.copy_(weights["kv_proj"])
        ref.gate_proj.weight.copy_(weights["gate_proj"])
        ref.position_bias.copy_(weights["position_bias"])
        ref.kv_norm.weight.copy_(weights["kv_norm_weight"])
        ref.q_b_proj.weight.copy_(weights["q_b_proj"])
        ref.scorer.weights_proj.weight.copy_(weights["weights_proj"])
    ours = TtIndexer(cfg, weights)

    hs, q_residual, pos = activations(64, seed=17)
    with torch.no_grad():
        ref_idx = ref(hs, q_residual, pos, None, 0)
        our_idx, scores, keys = ours.forward(hs, q_residual, pos)
    assert our_idx.shape == ref_idx.shape, (our_idx.shape, ref_idx.shape)
    assert torch.equal(our_idx, ref_idx), (our_idx != ref_idx).nonzero()[:6].tolist()
    assert torch.isfinite(scores).all() and torch.isfinite(keys).all()
    # Real weights must not collapse retrieval: with 16 entries and k=512 clamped, most
    # selections are legal, and they must not all be the same entry.
    valid = our_idx[0, -1][our_idx[0, -1] >= 0]
    assert valid.numel() > 1, valid
    assert valid.unique().numel() > 1, "retrieval degenerated to one entry"


def test_bf16_agreement_is_looser_but_still_close(cfg, ck, layer_of):
    """An observation about dtype risk, not a device claim.

    A real run loads the module itself in bf16; feeding bf16 activations to an fp32 module is
    not a configuration that exists (torch refuses the matmul outright), so both sides are
    cast the way a deployment would cast them.
    """
    layer = layer_of["heavily_compressed_attention"]
    weights = compressor_tensors(ck, layer)

    def build(dtype):
        torch.manual_seed(0)
        ref = DeepseekV4HCACompressor(cfg).to(dtype).eval()
        with torch.no_grad():
            ref.kv_proj.weight.copy_(weights["kv_proj"])
            ref.gate_proj.weight.copy_(weights["gate_proj"])
            ref.position_bias.copy_(weights["position_bias"])
            ref.kv_norm.weight.copy_(weights["kv_norm_weight"])
        ours = TtHCACompressor(cfg, {k: v.to(dtype) for k, v in weights.items()})
        return ref, ours

    hs, _qr, pos = activations(2 * 128, seed=19)
    ref32, ours32 = build(torch.float32)
    ref_bf, ours_bf = build(torch.bfloat16)
    with torch.no_grad():
        r32, _ = ref32(hs, None, pos, None, 0)
        o32, _ = ours32.forward(hs, pos, compressed_len=r32.shape[2])
        hs_bf = hs.to(torch.bfloat16)
        rbf, _ = ref_bf(hs_bf, None, pos, None, 0)
        obf, _ = ours_bf.forward(hs_bf, pos, compressed_len=rbf.shape[2])
    fp32_gap = (o32 - r32).abs().max().item()
    cross_gap = (obf.float() - rbf.float()).abs().max().item()
    assert fp32_gap < 1e-5, fp32_gap
    assert cross_gap < 5e-2, cross_gap
    assert torch.isfinite(obf.float()).all()
    # The dtype gap must be visible. Pretending bf16 is exact would be its own lie, and the
    # real device dtype behaviour still has to be measured on hardware, not inferred here.
    assert cross_gap > fp32_gap, (cross_gap, fp32_gap)
