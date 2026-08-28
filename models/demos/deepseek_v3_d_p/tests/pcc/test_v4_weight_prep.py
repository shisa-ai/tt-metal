"""Weight-preparation contracts for V4-Flash, pinned against the publisher's converter.

Two tiers:

* Format and sharding rules pinned with no external dependency, so they run everywhere.
* Parity against the publisher's own ``inference/convert.py`` and against the real
  checkpoint, enabled by pointing ``DS4_V4_FLASH_DIR`` at the snapshot directory. Those
  are the tests that make "lineage copy" a checked claim rather than a comment.

The publisher file is the oracle here deliberately: it is the primary source for the
format, and when it changes these tests fail loudly instead of our loader quietly
drifting from the checkpoint.
"""

import glob as globmod
import importlib.util
import math
import os
import struct

import pytest
import torch
from tt import v4_weight_prep as prep

SNAP = os.environ.get("DS4_V4_FLASH_DIR")
needs_ckpt = pytest.mark.skipif(
    not SNAP or not os.path.isdir(SNAP), reason="set DS4_V4_FLASH_DIR to the V4-Flash snapshot"
)


def _publisher_convert():
    path = os.path.join(SNAP, "inference", "convert.py")
    if not os.path.isfile(path):
        pytest.skip(f"publisher converter not found at {path}")
    spec = importlib.util.spec_from_file_location("dsv4_convert", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- format, no dependencies


def test_low_nibble_is_the_even_column():
    def i8(*bytes_):
        return torch.tensor([list(bytes_)], dtype=torch.uint8).view(torch.int8)

    # 0x21: low nibble 1 -> +0.5 at the even column, high nibble 2 -> +1.0 at the odd one.
    got = prep.unpack_e2m1fn(i8(0x21))
    assert got.tolist() == [[0.5, 1.0]], f"nibble order changed: {got.tolist()}"
    # 0xF8: low 8 -> +0.0, high 15 -> -6.0. Sign lives in bit 3, so the table is not
    # monotonic in the code and no (-1)**b * magnitude formula reproduces it.
    assert prep.unpack_e2m1fn(i8(0xF8)).tolist() == [[0.0, -6.0]]
    assert math.copysign(1.0, prep.unpack_e2m1fn(i8(0xF8))[0, 0].item()) == 1.0


def test_decode_table_is_not_monotonic_in_the_code():
    vals = list(prep.E2M1FN_VALUES)
    assert vals[:8] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert vals[8] == 0.0 and vals[9] == -0.5, "the negative half must reuse code 8..15"
    # ``-0.0 == 0.0`` in Python, so equality cannot pin this. The publisher's table writes
    # code 8 as +0.0, and their fp8 cast therefore emits 0x00 where a -0.0 entry emits
    # 0x80. Bitwise parity with their converted files depends on the sign bit.
    assert math.copysign(1.0, vals[8]) == 1.0, "code 8 must decode to +0.0 for parity"
    assert math.copysign(1.0, vals[0]) == 1.0
    assert math.copysign(1.0, vals[15]) == -1.0
    assert len(vals) == 16


def test_dequant_mxfp4_exact_for_a_hand_case():
    # One scale column covers FP4_BLOCK=32 elements, so K = 64 needs 32 packed bytes.
    packed = torch.zeros(1, 32, dtype=torch.int8)
    packed[0, 0] = 0x10  # -> [0.0, 0.5] in the first 32-group
    packed[0, 16] = 0x32  # low nibble 2 -> 1.0, high nibble 3 -> 1.5 (second group)
    scale = torch.tensor([[127 + 3, 127 - 1]], dtype=torch.uint8)  # x8, x0.5
    got = prep.dequant_mxfp4(packed, scale)
    assert got.shape == (1, 64)
    assert got[0, :2].tolist() == [0.0, 4.0]
    assert got[0, 32:34].tolist() == [0.5, 0.75]
    assert (got[0, 2:32] == 0.0).all() and (got[0, 34:] == 0.0).all()


def test_e8m0_carriers_agree():
    from_uint8 = prep.e8m0_to_float(torch.tensor([127, 130, 126], dtype=torch.uint8))
    from_native = prep.e8m0_to_float(torch.tensor([1.0, 8.0, 0.5], dtype=torch.float32).to(torch.float8_e8m0fnu))
    assert from_uint8.tolist() == [1.0, 8.0, 0.5]
    assert from_native.tolist() == [1.0, 8.0, 0.5]


def test_group_sizes_are_the_32_and_128_the_publisher_uses():
    assert (prep.FP4_BLOCK, prep.FP8_BLOCK, prep.E8M0_BIAS) == (32, 128, 127)


def test_shard_rule_is_not_uniform_across_tensors():
    # wq_b/wo_a output-parallel, wo_b INPUT-parallel. A blanket dim=0 rule is wrong.
    assert prep.shard_dim("layers.3.attn.wq_b.weight") == 0
    assert prep.shard_dim("layers.3.attn.wo_a.weight") == 0
    assert prep.shard_dim("layers.3.attn.wo_b.weight") == 1
    assert prep.shard_dim("layers.3.attn.wq_a.weight") is None
    assert prep.shard_dim("layers.3.ffn.experts.7.w1.weight") is None  # expert-parallel


def test_owner_rank_blocks_experts_not_interleaves_them(expect_error):
    owners = [prep.owner_rank(f"layers.0.ffn.experts.{i}.w1.weight", 256, 32) for i in range(256)]
    assert owners == [i // 8 for i in range(256)], "ranks must own consecutive blocks"
    assert sorted(set(owners)) == list(range(32))
    assert prep.owner_rank("layers.0.ffn.shared_experts.w1.weight", 256, 32) is None
    assert prep.owner_rank("layers.0.attn.wq_a.weight", 256, 32) is None
    with expect_error(ValueError, "not divisible"):
        prep.owner_rank("layers.0.ffn.experts.1.w1.weight", 256, 3)


def test_port_name_renames_are_idempotent():
    for name in (
        "model.layers.1.self_attn.q_b_proj.weight",
        "layers.1.self_attn.q_b_proj.weight",
        "layers.1.attn.wq_b.weight",
    ):
        once = prep.port_name(name)
        assert once.startswith("layers.1.attn."), once
        assert prep.port_name(once) == once, "renames must be idempotent"
    assert prep.port_name("model.layers.2.mlp.gate.weight") == "layers.2.ffn.gate.weight"


def test_scale_shape_contract_is_enforced(expect_error):
    packed = torch.zeros(2, 4, dtype=torch.int8)  # logical 2 x 8
    with expect_error(ValueError, "does not match"):
        prep.dequant_mxfp4(packed, torch.zeros(2, 4, dtype=torch.uint8))  # wants 2 x 8/32
    with expect_error(ValueError, "is not a multiple"):
        prep.dequant_block_fp8(torch.zeros(100, 128, dtype=torch.float8_e4m3fn), torch.ones(1, 1, dtype=torch.uint8))
    with expect_error(TypeError, "must be int8"):
        prep.unpack_e2m1fn(torch.zeros(2, 2, dtype=torch.uint8))


def test_torch_float4_view_cannot_be_cast_which_is_why_the_table_exists():
    """Pins the reason ``E2M1FN_VALUES`` is load-bearing rather than convenience."""
    raw = torch.tensor([[0x21]], dtype=torch.int8)
    try:
        raw.view(torch.float4_e2m1fn_x2).to(torch.float32)
    except NotImplementedError:
        return
    pytest.skip("torch gained Float4_e2m1fn_x2 copy support; the table may be redundant")


# ------------------------------------------------------- publisher parity (needs source)


@needs_ckpt
def test_cast_matches_the_publishers_implementation_bitwise():
    pub = _publisher_convert()
    g = torch.Generator().manual_seed(20260828)
    for out_dim, in_dim in ((128, 128), (256, 384), (128, 1024)):
        x = torch.randint(-128, 128, (out_dim, in_dim // 2), generator=g, dtype=torch.int8)
        # Valid e8m0 bytes; keep exponents in a range where the fp8 cast cannot overflow.
        s = torch.randint(120, 132, (out_dim, in_dim // prep.FP4_BLOCK), generator=g, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        mine_w, mine_s = prep.cast_e2m1fn_to_e4m3fn(x, s)
        ref_w, ref_s = pub.cast_e2m1fn_to_e4m3fn(x, s.clone())
        assert torch.equal(mine_w.view(torch.uint8), ref_w.view(torch.uint8)), f"fp8 {out_dim}x{in_dim}"
        assert torch.equal(mine_s.view(torch.uint8), ref_s.view(torch.uint8)), f"e8m0 {out_dim}x{in_dim}"


@needs_ckpt
def test_publishers_table_and_ours_agree():
    pub = _publisher_convert()
    assert torch.equal(pub.FP4_TABLE.float(), prep.e2m1fn_table())


# ------------------------------------------- real checkpoint (needs the 156 GB snapshot)


def _read_tensor(path, name):
    """Read one tensor by parsing the shard header, so no payload is loaded twice."""
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json_loads(fh.read(hlen))
        meta = header[name]
        lo, hi = meta["data_offsets"]
        fh.seek(8 + hlen + lo)
        buf = fh.read(hi - lo)
    dtype = {
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "I8": torch.int8,
        "I64": torch.int64,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
    }[meta["dtype"]]
    t = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    if dtype is torch.int8:
        t = t.view(torch.int8)
    elif dtype is torch.int64:
        t = t.view(torch.int64)
    elif dtype is torch.float8_e4m3fn:
        t = t.view(torch.float8_e4m3fn)
    elif dtype is torch.float8_e8m0fnu:
        t = t.view(torch.float8_e8m0fnu)
    elif dtype is torch.bfloat16:
        t = t.view(torch.bfloat16)
    return t.reshape(meta["shape"]), dtype


def json_loads(payload):
    import json

    return json.loads(payload)


def _shards():
    return sorted(globmod.glob(os.path.join(SNAP, "model-*.safetensors")))


_HEADERS = None


def _headers():
    """Parse every shard header once. Shards do **not** hold consecutive layers, so a
    tensor cannot be found by guessing an index."""
    global _HEADERS
    if _HEADERS is None:
        _HEADERS = {}
        for path in _shards():
            with open(path, "rb") as fh:
                (hlen,) = struct.unpack("<Q", fh.read(8))
                _HEADERS[path] = json_loads(fh.read(hlen))
    return _HEADERS


def _read(name):
    """Read one tensor by name from whichever shard holds it."""
    for path, header in _headers().items():
        if name in header:
            return _read_tensor(path, name)
    pytest.skip(f"{name} not present in this snapshot")


@needs_ckpt
def test_real_expert_dequant_is_exact_and_the_fp8_cast_is_lossless():
    """The strongest format claim in the port, checked on real bytes.

    ``dequant_mxfp4`` must be bit-exact (e2m1fn values times a power of two is binary
    exact), and the publisher's fp8 upcast must preserve the dequantized values exactly,
    which is what makes "lossless" more than the publisher's word for it.
    """
    w, _ = _read("layers.0.ffn.experts.0.w1.weight")
    s, _ = _read("layers.0.ffn.experts.0.w1.scale")
    assert w.shape == (2048, 2048) and s.shape == (2048, 128)

    deq = prep.dequant_mxfp4(w, s)
    assert deq.shape == (2048, 4096)
    assert torch.isfinite(deq).all()
    # Independent dequant: nibble table lookup with an explicit per-32 multiplier.
    table = prep.e2m1fn_table()
    raw = w.view(torch.uint8)
    indep = torch.stack(
        [
            table[(raw & 0xF).long()],
            table[((raw >> 4) & 0xF).long()],
        ],
        dim=-1,
    ).flatten(-2)
    indep = indep * prep.e8m0_to_float(s).repeat_interleave(32, dim=-1)
    assert torch.equal(deq, indep), "dequant must match an independent nibble path"
    # Values are table magnitudes scaled by powers of two, so the dequantization is exact.
    allowed = {v * 2.0**e for v in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) for e in range(-20, 20)} | {0.0}
    assert set(deq.abs().unique().tolist()) <= allowed

    fp8, block = prep.cast_e2m1fn_to_e4m3fn(w, s)
    back = fp8.to(torch.float32) * prep.e8m0_to_float(block).repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.equal(back, deq), "the fp8 upcast must be lossless, not approximately so"


@needs_ckpt
def test_real_attention_tensor_dequantizes_with_128_block_scales():
    w, dtype = _read("layers.0.attn.wo_a.weight")
    s, _ = _read("layers.0.attn.wo_a.scale")
    assert dtype is torch.float8_e4m3fn, dtype
    assert w.shape == (8192, 4096) and s.shape == (64, 32)
    got = prep.dequant_block_fp8(w, s)
    assert got.shape == (8192, 4096) and torch.isfinite(got).all()
    # Spot-check one block by hand against a scalar multiply.
    blk = got[0, :128] / (w[0, :128].to(torch.float32) * prep.e8m0_to_float(s)[0, 0])
    assert torch.allclose(blk[torch.isfinite(blk)], torch.ones_like(blk[torch.isfinite(blk)]))


@needs_ckpt
def test_real_names_are_covered_and_experts_partition_once_across_32_ranks():
    counts = {}
    seen = set()
    for path, header in _headers().items():
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            pn = prep.port_name(name)
            assert pn.startswith(("layers.", "embed", "head", "norm", "hc_head", "mtp")), pn
            if name.startswith("mtp."):
                continue
            seen.add(pn)
            if pn.endswith(".weight") and meta["dtype"] in ("I8", "F8_E4M3"):
                base = pn[: -len(".weight")]
                counts[base] = counts.get(base, 0) + 1
    # Every quantized weight we would load must have exactly one scale sibling.
    missing = {k for k, v in counts.items() if v != 1}
    assert not missing, f"quantized weights without exactly one scale: {sorted(missing)[:5]}"
    assert len(counts) > 30000, len(counts)

    owners = {}
    for name in seen:
        rank = prep.owner_rank(name, 256, 32)
        if rank is None:
            continue
        owners.setdefault(rank, set()).add(name)
    assert sorted(owners) == list(range(32))
    sizes = {len(v) for v in owners.values()}
    assert len(sizes) == 1, f"ranks got unequal expert tensors: {sizes}"
    all_names = [n for v in owners.values() for n in v]
    assert len(all_names) == len(set(all_names)), "an expert tensor is owned twice"


@needs_ckpt
def test_publishers_converter_end_to_end_on_one_rank_reproduces_our_path(tmp_path):
    """Run the publisher's own ``main`` for mp=1 and compare rank 0 against our path.

    Scoped to a single layer by writing a mini checkpoint first, because the real one is
    156 GB and the disk here has ~104 GB free -- a full conversion cannot be written,
    which is itself a constraint worth pinning.
    """
    pub = _publisher_convert()
    import safetensors.torch as st

    mini = {}
    for suffix in (
        "attn.wq_a.weight",
        "attn.wq_a.scale",
        "attn.wo_a.weight",
        "attn.wo_a.scale",
        "attn.wo_b.weight",
        "attn.wo_b.scale",
        "attn.wq_b.weight",
        "attn.wq_b.scale",
        "ffn.experts.0.w1.weight",
        "ffn.experts.0.w1.scale",
        "ffn.experts.255.w1.weight",
        "ffn.experts.255.w1.scale",
    ):
        name = f"layers.0.{suffix}"
        mini[name] = _read(name)[0]
    src = tmp_path / "src"
    src.mkdir()
    st.save_file(mini, str(src / "model-00001-of-00001.safetensors"))
    out = tmp_path / "out"
    pub.main(str(src), str(out), n_experts=256, mp=1, expert_dtype="fp8")

    produced = {}
    for path in globmod.glob(str(out / "*.safetensors")):
        with open(path, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json_loads(fh.read(hlen))
        for name in header:
            produced[name] = _read_tensor(path, name)[0]

    # experts upcast to fp8, wo_a dequantized to bfloat16, names in port style
    assert "layers.0.attn.wo_a.weight" in produced
    assert produced["layers.0.attn.wo_a.weight"].dtype is torch.bfloat16
    assert produced["layers.0.ffn.experts.0.w1.weight"].dtype is torch.float8_e4m3fn
    ours = prep.dequant_block_fp8(mini["layers.0.attn.wo_a.weight"], mini["layers.0.attn.wo_a.scale"]).bfloat16()
    assert torch.equal(produced["layers.0.attn.wo_a.weight"], ours), "our wo_a dequant must agree with the publisher's"
