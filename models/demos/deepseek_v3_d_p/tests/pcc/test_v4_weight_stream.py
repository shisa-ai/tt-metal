"""Streaming-reader contracts for the real V4-Flash checkpoint.

The reader exists because the conversion cannot be written to disk here, so the tests have
to check the properties that make that viable: bounded per-tensor IO, an exact rank
partition, shard-and-convert equivalence, and a peak-memory bound. Everything that can be
checked without the 156 GB snapshot runs unconditionally; the rest is gated on
``DS4_V4_FLASH_DIR``.
"""

import gc
import importlib.util
import os
import tracemalloc

import pytest
import torch

from models.demos.deepseek_v3_d_p.tt import v4_weight_prep as prep
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import V4Checkpoint, _narrow_scale, scale_sibling

SNAP = os.environ.get("DS4_V4_FLASH_DIR")
needs_ckpt = pytest.mark.skipif(
    not SNAP or not os.path.isdir(SNAP), reason="set DS4_V4_FLASH_DIR to the V4-Flash snapshot"
)

#: Measured fact from the header inventory, so index drift is a loud failure.
EXPECTED_TENSORS = 72317
EXPECTED_BYTES = 166_878_536_440
N_EXPERTS = 256
EP = 32


@pytest.fixture(scope="module")
def ck():
    if not SNAP or not os.path.isdir(SNAP):
        pytest.skip("no snapshot")
    return V4Checkpoint(SNAP)


def _publisher_convert():
    path = os.path.join(SNAP, "inference", "convert.py")
    if not os.path.isfile(path):
        pytest.skip(f"publisher converter not found at {path}")
    spec = importlib.util.spec_from_file_location("dsv4_convert", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------- snapshot-free contracts


def test_scale_sibling_name_rule():
    assert scale_sibling("layers.3.attn.wq_a.weight") == "layers.3.attn.wq_a.scale"
    assert scale_sibling("layers.3.attn.attn_sink") == ""
    assert scale_sibling("norm.weight") == "norm.scale"


def test_narrow_scale_derives_granularity_instead_of_assuming_it(expect_error):
    # fp4: one scale per row along the output axis -> ratio 1.
    s = torch.arange(8).view(8, 1).to(torch.uint8)
    got = _narrow_scale(s, 0, 8, 4, "I8", "x.w", 1)
    assert got.tolist() == [[4], [5], [6], [7]]
    # fp8: one scale per 128 block on both axes -> ratio 128.
    blk = torch.arange(2 * 4).view(2, 4).to(torch.uint8)
    got = _narrow_scale(blk, 0, 256, 128, "F8_E4M3", "x.w", 1)
    assert got.shape == (1, 4) and got[0].tolist() == [4, 5, 6, 7]
    # A scale whose axis matches neither layout must raise, not silently descale.
    with expect_error(ValueError, "matches neither"):
        _narrow_scale(torch.zeros(3, 3, dtype=torch.uint8), 0, 256, 128, "F8_E4M3", "x.w", 0)
    # A shard whose extent is finer than the scale granularity must raise too. Here the
    # axis does resolve (2 scales x 128 = 256 rows) but 64 rows is finer than one block.
    with expect_error(ValueError, "splits scale groups"):
        _narrow_scale(torch.zeros(2, 2, dtype=torch.uint8), 0, 256, 64, "F8_E4M3", "x.w", 0)
    assert _narrow_scale(None, 0, 8, 4, "I8", "x.w", 0) is None


# ------------------------------------------------- snapshot-gated contracts


@needs_ckpt
def test_index_matches_the_measured_inventory(ck):
    assert len(ck) == EXPECTED_TENSORS, len(ck)
    assert ck.total_bytes() == EXPECTED_BYTES, ck.total_bytes()
    assert len(ck.shards) == 48, len(ck.shards)
    assert not any(n.startswith("mtp.") for n in ck.names())
    assert any(n.startswith("mtp.") for n in ck.names(include_mtp=True))
    # The stream skips *.scale names in its driver loop, which is only safe if every
    # scale has a weight to ride along with.
    assert ck.orphan_scales() == [], ck.orphan_scales()[:5]


@needs_ckpt
def test_one_tensor_read_touches_only_its_own_bytes(ck):
    """The premise of the whole module, measured rather than asserted in prose."""
    name = "layers.0.ffn.experts.0.w1.weight"
    ref = ck.index[name]
    shard_bytes = os.path.getsize(ref.shard)
    assert shard_bytes > 2**31, f"shard unexpectedly small: {shard_bytes}"
    before = ck.bytes_read
    t = ck.read(name)
    fetched = ck.bytes_read - before
    assert t.shape == ref.shape and t.dtype is torch.int8
    assert fetched == ref.nbytes == 2048 * 2048, fetched
    assert fetched < shard_bytes / 100, f"read {fetched} of a {shard_bytes} shard"


@needs_ckpt
def test_raw_stream_equals_direct_reads(ck):
    direct = {}
    for name in ck.layer_names(0):
        if name.endswith(".scale"):
            continue  # reached through its weight
        w, s = ck.read_with_scale(name)
        direct[prep.port_name(name)] = (w, s)
    streamed = dict(ck.iter_rank_layer(0, mode="raw"))
    expected = {prep.port_name(n) for n in ck.layer_names(0)}
    assert set(streamed) == expected, expected ^ set(streamed)
    assert set(streamed) >= {k for k, (w, _) in direct.items() if w.dtype in (torch.int8, torch.float8_e4m3fn)}
    for name, (w, s) in direct.items():
        assert torch.equal(streamed[name], w), name
        if s is not None:
            assert torch.equal(streamed[scale_sibling(name)], s), name


@needs_ckpt
def test_f32_stream_dequantizes_to_the_logical_shapes(ck):
    """Declared shapes are packed; the stream must hand out logical ones."""
    seen = {}
    for name, t in ck.iter_rank_layer(0, mode="f32"):
        seen[name] = t
    assert seen["layers.0.ffn.experts.0.w1.weight"].shape == (2048, 4096)
    assert seen["layers.0.ffn.experts.0.w2.weight"].shape == (4096, 2048)
    assert seen["layers.0.attn.wq_b.weight"].shape == (32768, 1024)
    assert seen["layers.0.attn.attn_sink"].shape == (64,)
    for name, t in seen.items():
        assert t.dtype is torch.float32, name
        assert torch.isfinite(t).all(), f"{name} has non-finite dequantized values"


@needs_ckpt
def test_expert_names_partition_exactly_once_across_32_ranks(ck):
    """Index-level over all layers: no expert lost, none owned twice."""
    per_rank = {r: [] for r in range(EP)}
    for layer in range(43):
        names = [n for n in ck.layer_names(layer) if ".experts." in n and "shared" not in n]
        for r in range(EP):
            owned = [prep.port_name(n) for n in names if prep.owner_rank(prep.port_name(n), N_EXPERTS, EP) == r]
            per_rank[r].extend(owned)
    flat = [n for r in per_rank for n in per_rank[r]]
    assert len(flat) == len(set(flat)), "an expert tensor is owned by two ranks"
    all_experts = [
        prep.port_name(n)
        for layer in range(43)
        for n in ck.layer_names(layer)
        if ".experts." in n and "shared" not in n
    ]
    assert set(flat) == set(all_experts), "expert tensors missing from the partition"
    sizes = {len(v) for v in per_rank.values()}
    assert len(sizes) == 1, f"ranks hold different numbers of expert tensors: {sizes}"


@needs_ckpt
def test_sharded_attention_reassembles_to_the_full_tensor(ck):
    """wo_b is input-parallel and wq_b output-parallel; both must reassemble exactly."""
    for name, dim in (("layers.0.attn.wo_b.weight", 1), ("layers.0.attn.wq_b.weight", 0)):
        full = ck.read(name)
        parts = []
        for r in range(EP):
            got = dict(ck.iter_rank_layer(0, rank=r, ep_size=EP, mode="raw"))
            assert name in got, f"{name} vanished for rank {r}"
            slice_ = got[name]
            assert slice_.shape[dim] == full.shape[dim] // EP, (name, tuple(slice_.shape))
            assert slice_.shape[1 - dim] == full.shape[1 - dim], (name, tuple(slice_.shape))
            parts.append(slice_)
        assert torch.equal(torch.cat(parts, dim=dim), full), name


@needs_ckpt
def test_shard_then_convert_equals_convert_then_shard(ck):
    """Order-independence is what makes the block-alignment guard meaningful."""
    name = "layers.0.attn.wq_b.weight"
    full = dict(ck.iter_rank_layer(0, mode="fp8"))
    for r in range(EP):
        shard = dict(ck.iter_rank_layer(0, rank=r, ep_size=EP, mode="fp8"))
        rows = full[name].shape[0] // EP
        assert torch.equal(shard[name], full[name][r * rows : (r + 1) * rows]), f"rank {r}"
        scale = scale_sibling(name)
        assert torch.equal(shard[scale], full[scale][r * (rows // 128) : (r + 1) * (rows // 128)]), f"rank {r} scale"


@needs_ckpt
def test_a_shard_that_would_cut_block_scales_is_rejected(ck, expect_error):
    """128-block FP8 scales cannot be split; better a raise than descaled weights."""
    # ep_size must divide the 256 experts, so the largest legal value whose attention
    # shard is finer than a block is 128: wo_a is [8192, 4096], so 8192/128 = 64 rows.
    with expect_error(ValueError, "cuts 128-block scales"):
        list(ck.iter_rank_layer(0, rank=0, ep_size=128, mode="raw"))


@needs_ckpt
def test_fp8_stream_matches_the_publishers_converter(ck, tmp_path):
    """Same input, same rank: our stream and their main() must agree byte for byte."""
    import safetensors.torch as st

    pub = _publisher_convert()
    names = [
        n
        for n in ck.layer_names(0)
        if n.endswith(
            (
                "wq_a.weight",
                "wq_a.scale",
                "wo_a.weight",
                "wo_a.scale",
                "wo_b.weight",
                "wo_b.scale",
                "wq_b.weight",
                "wq_b.scale",
                "experts.0.w1.weight",
                "experts.0.w1.scale",
                "experts.255.w1.weight",
                "experts.255.w1.scale",
            )
        )
    ]
    src = tmp_path / "src"
    src.mkdir()
    st.save_file({n: ck.read(n) for n in names}, str(src / "model-00001-of-00001.safetensors"))
    out = tmp_path / "out"
    pub.main(str(src), str(out), n_experts=N_EXPERTS, mp=1, expert_dtype="fp8")

    theirs = {}
    for path in ((out / "model0-mp1.safetensors"),):
        theirs.update({n: t for n, t in st.load_file(str(path)).items()})

    ours = dict(ck.iter_rank_layer(0, mode="fp8"))
    checked = 0
    for name, tensor in theirs.items():
        if not name.startswith("layers.0."):
            continue
        assert name in ours, f"{name} produced by the publisher but not by us"
        assert ours[name].dtype == tensor.dtype, (name, ours[name].dtype, tensor.dtype)
        assert ours[name].shape == tensor.shape, (name, ours[name].shape, tensor.shape)
        assert torch.equal(ours[name].contiguous(), tensor.contiguous()), name
        checked += 1
    assert checked >= 8, checked
    assert ours["layers.0.attn.wo_a.weight"].dtype is torch.bfloat16


@needs_ckpt
def test_peak_memory_stays_bounded_while_streaming_a_layer(ck):
    """'Streaming' means a bound, so measure it: rank 0 of 32 owns 8 of 256 experts."""
    gc.collect()
    ck.bytes_read = 0
    tracemalloc.start()
    peak = 0
    n = 0
    for _name, t in ck.iter_rank_layer(0, rank=0, ep_size=EP, mode="f32"):
        n += 1
        del t
        peak = max(peak, tracemalloc.get_traced_memory()[0])
    current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # A rank at ep_size=32 legitimately sees few tensors: 248 of 256 experts are skipped
    # without ever being read, which is exactly where the IO and memory win comes from.
    assert 30 < n < 200, n
    assert peak < 2**30, f"peak allocation {peak / 2**20:.0f} MiB while streaming one layer"
    # A whole layer at ep_size=1 is ~3.3 GiB; the win must be visible in bytes read.
    whole = sum(ck.index[x].nbytes for x in ck.layer_names(0))
    assert ck.bytes_read < whole / 2, f"read {ck.bytes_read} of {whole} bytes for one layer"
    assert traced_peak >= peak


@needs_ckpt
def test_reader_rejects_a_snapshot_without_shards(tmp_path, expect_error):
    with expect_error(FileNotFoundError, "no model-"):
        V4Checkpoint(str(tmp_path))
