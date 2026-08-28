"""Memory-bounded streaming reader for the real V4-Flash checkpoint.

Why this exists rather than ``convert.py``: the publisher's converter writes one output
file per model-parallel rank for the *whole* model, and the FP8 path is ~285 GiB while
this host has ~104 GB free. It also cannot be loaded wholesale for a host-side oracle —
the float model is ~566 GiB in bfloat16 and ~1.1 TiB in float32, against ~427 GB of
available RAM. So the working style has to be "index everything once, then hold one
tensor at a time", and no step of it may require the whole model in memory.

Consequences encoded here:

* Only safetensors **headers** are scanned at construction; payload bytes are read by
  positioned read, one tensor at a time. A full 3.3 GiB shard is never materialised.
* Rank selection happens per tensor, so a rank's weights can be streamed to a device
  without an intermediate file.
* Conversion follows the publisher's converted (per-expert) layout, **not** HuggingFace's
  fused ``mlp.experts.gate_up_proj`` / ``down_proj`` bmm layout. That is deliberate: the
  device path consumes per-expert tensors, and bridging to the fused layout would mean
  materialising the whole model as floats, which does not fit here. If a fused-layout
  consumer ever appears, add it explicitly rather than by accident.

Format semantics come from :mod:`tt.v4_weight_prep`, which is pinned against the
publisher's own converter; this module only decides *what to read and in what order*.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Iterator

import torch

from models.demos.deepseek_v3_d_p.tt import v4_weight_prep as prep

#: safetensors dtype strings this checkpoint actually uses, plus the adjacent ones.
SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


@dataclass(frozen=True)
class TensorRef:
    """Where one tensor lives, and how to interpret its bytes."""

    shard: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


def scale_sibling(weight_name: str) -> str:
    """The scale tensor for a ``*.weight`` tensor, per the checkpoint's convention.

    Empty string when the name is not a weight, so callers can test truthiness.
    """
    return weight_name[: -len(".weight")] + ".scale" if weight_name.endswith(".weight") else ""


class V4Checkpoint:
    """Header-indexed, payload-lazy view of an unpacked V4-Flash snapshot.

    ``snapshot_dir`` is the directory holding ``config.json`` and ``model-*.safetensors``
    (an HF cache ``snapshots/<rev>`` directory works directly, since its entries are
    symlinks into ``blobs/``).
    """

    def __init__(self, snapshot_dir: str):
        self.snapshot_dir = os.path.abspath(snapshot_dir)
        #: Payload bytes fetched by :meth:`read`, so "streaming" stays a measured claim.
        self.bytes_read = 0
        shards = sorted(
            os.path.join(self.snapshot_dir, p)
            for p in os.listdir(self.snapshot_dir)
            if p.startswith("model-") and p.endswith(".safetensors")
        )
        if not shards:
            raise FileNotFoundError(f"no model-*.safetensors under {self.snapshot_dir}")
        self.shards: list[str] = shards
        self.index: dict[str, TensorRef] = {}
        for path in shards:
            for name, meta in _read_header(path).items():
                if name == "__metadata__":
                    continue
                lo, hi = meta["data_offsets"]
                if name in self.index:
                    raise ValueError(f"{name} appears in two shards; refusing to guess")
                self.index[name] = TensorRef(path, meta["dtype"], tuple(meta["shape"]), lo, hi)

    # -------------------------------------------------------------- indexing

    def __len__(self) -> int:
        return len(self.index)

    def names(self, *, include_mtp: bool = False) -> list[str]:
        """Checkpoint tensor names, optionally including the ``mtp.*`` tree."""
        if include_mtp:
            return sorted(self.index)
        return sorted(n for n in self.index if not n.startswith("mtp."))

    def layer_names(self, layer: int, *, prefix: str = "layers") -> list[str]:
        head = f"{prefix}.{layer}."
        return sorted(n for n in self.index if n.startswith(head))

    def total_bytes(self) -> int:
        return sum(ref.nbytes for ref in self.index.values())

    def orphan_scales(self) -> list[str]:
        """``*.scale`` tensors with no matching ``*.weight``.

        The stream skips scale names in its driver loop because it emits them next to
        their weight, which would silently drop any orphan. This makes that invisible
        loss impossible to assume: assert it is empty, or name what came back.
        """
        return sorted(
            n for n in self.index if n.endswith(".scale") and n[: -len(".scale")] + ".weight" not in self.index
        )

    # ------------------------------------------------------------ payload IO

    def read(self, name: str) -> torch.Tensor:
        """Read exactly one tensor's bytes. Nothing else from its shard is touched."""
        ref = self.index[name]
        dtype = SAFETENSORS_DTYPES.get(ref.dtype)
        if dtype is None:
            raise KeyError(f"unmapped safetensors dtype {ref.dtype!r} for {name}")
        with open(ref.shard, "rb") as fh:
            fh.seek(8 + _header_length(fh) + ref.begin)
            buf = fh.read(ref.nbytes)
        if len(buf) != ref.nbytes:
            raise IOError(f"short read for {name}: {len(buf)} of {ref.nbytes} bytes")
        self.bytes_read += ref.nbytes
        # frombuffer aliases the bytes, so copy into an owned tensor before fh closes.
        flat = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
        return _interpret(flat, dtype).reshape(ref.shape)

    def read_with_scale(self, name: str):
        """``(weight, scale)`` where scale is ``None`` for unquantized tensors."""
        sibling = scale_sibling(name)
        scale = self.read(sibling) if sibling and sibling in self.index else None
        return self.read(name), scale

    def dequantized(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Float value of one tensor, using the publisher's format rules.

        MXFP4 experts and FP8 attention are both exact into binary32; converting straight
        to bfloat16 is not necessarily exact and so is an explicit caller choice.
        """
        ref = self.index[name]
        weight, scale = self.read_with_scale(name)
        if scale is None:
            return weight.to(dtype)
        if ref.dtype == "I8":
            return prep.dequant_mxfp4(weight, scale).to(dtype)
        if ref.dtype == "F8_E4M3":
            return prep.dequant_block_fp8(weight, scale).to(dtype)
        raise ValueError(f"{name}: unexpected quantized carrier {ref.dtype}")

    # ------------------------------------------------- per-rank streaming

    def iter_rank_layer(
        self,
        layer: int,
        *,
        rank: int = 0,
        ep_size: int = 1,
        n_experts: int = 256,
        mode: str = "fp8",
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield one layer's tensors for one rank, in port naming.

        ``mode``:

        * ``"raw"``  stored payload plus its scale sibling, no arithmetic.
        * ``"fp8"``  the publisher's converted form: MXFP4 experts upcast losslessly to
          ``float8_e4m3fn`` with an e8m0 per 128x128 block, ``attn.wo_a`` dequantized to
          bfloat16, everything else as stored.
        * ``"f32"``  fully dequantized floats. One layer at a time only; do not build a
          whole model this way, it does not fit in RAM.

        Tensors are yielded in ascending name order so peak memory is bounded by the
        largest single tensor rather than by the layer.
        """
        if mode not in ("raw", "fp8", "f32"):
            raise ValueError(f"unknown mode {mode!r}")
        for name in self.layer_names(layer):
            if name.endswith(".scale"):
                # Scale tensors are emitted alongside their weight, narrowed with it.
                # Driving them here as if they were weights would shard them twice, and
                # the second pass would see a scale-shaped axis and reject a legal shard.
                continue
            out_name = prep.port_name(name)
            ref = self.index[name]

            if ref.dtype in ("I8", "F8_E4M3") and not _rank_owns(name, rank, ep_size, n_experts):
                # Read nothing at all for experts this rank does not own: that is where
                # the memory and IO win actually comes from (255/256 of a layer).
                continue

            weight, scale = self.read_with_scale(name)
            dim = prep.shard_dim(out_name)
            if dim is not None and ep_size > 1:
                total = weight.shape[dim]
                if total % ep_size:
                    raise ValueError(f"{name}: dim {dim} size {total} not divisible by {ep_size}")
                per = total // ep_size
                if ref.dtype == "F8_E4M3" and per % prep.FP8_BLOCK:
                    raise ValueError(f"{name}: shard of {per} along dim {dim} cuts {prep.FP8_BLOCK}-block scales")
                weight = weight.narrow(dim, rank * per, per).contiguous()
                scale = _narrow_scale(scale, dim, total, per, ref.dtype, name, rank)

            if mode == "raw":
                yield out_name, weight
                if scale is not None:
                    yield _scale_name(out_name), scale
            elif mode == "f32":
                yield out_name, _to_float(ref.dtype, weight, scale)
            else:
                for pair in _converted(out_name, ref.dtype, weight, scale):
                    yield pair


# ------------------------------------------------------------------- helpers


def _header_length(fh) -> int:
    fh.seek(0)
    (hlen,) = struct.unpack("<Q", fh.read(8))
    return hlen


def _read_header(path: str) -> dict:
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(hlen))


def _interpret(flat: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype is torch.uint8 or dtype is torch.bool:
        return flat.view(dtype) if dtype is torch.bool else flat
    if dtype is torch.int8:
        return flat.view(torch.int8)
    if dtype is torch.int64:
        return flat.view(torch.int64)
    if dtype is torch.int32:
        return flat.view(torch.int32)
    if dtype is torch.int16:
        return flat.view(torch.int16)
    if dtype is torch.float8_e4m3fn:
        return flat.view(torch.float8_e4m3fn)
    if dtype is torch.float8_e8m0fnu:
        return flat.view(torch.float8_e8m0fnu)
    if dtype is torch.bfloat16:
        return flat.view(torch.bfloat16)
    if dtype is torch.float16:
        return flat.view(torch.float16)
    if dtype is torch.float64:
        return flat.view(torch.float64)
    return flat.view(torch.float32)


def _scale_name(weight_name: str) -> str:
    return scale_sibling(weight_name)


def _narrow_scale(scale, dim: int, weight_total: int, per: int, dtype: str, name: str, rank: int):
    """Narrow a scale tensor alongside its weight, respecting the scale's granularity.

    Two layouts exist: MXFP4 keeps one scale **per element group of 32 along K** (so the
    scale's K axis is ``in/32`` while its output axis is one-per-row), and FP8 keeps one
    scale **per 128x128 block** on both axes. Narrowing either one as if it were a plain
    per-row array silently descales the weights, so the ratio is derived, never assumed.
    """
    if scale is None:
        return None
    n = scale.shape[dim]
    block = prep.FP4_BLOCK if dtype == "I8" else prep.FP8_BLOCK
    if n == weight_total:
        ratio = 1
    elif n * block == weight_total:
        ratio = block
    else:
        raise ValueError(
            f"{name}: scale axis {n} on dim {dim} matches neither per-element nor "
            f"per-{block}-block granularity for weight extent {weight_total}"
        )
    if per % ratio:
        raise ValueError(f"{name}: shard of {per} along dim {dim} splits scale groups of {ratio}")
    return scale.narrow(dim, rank * (per // ratio), per // ratio).contiguous()


def _rank_owns(name: str, rank: int, ep_size: int, n_experts: int) -> bool:
    if ep_size <= 1:
        return True
    owner = prep.owner_rank(prep.port_name(name), n_experts, ep_size)
    return owner is None or owner == rank


def _to_float(dtype: str, weight: torch.Tensor, scale) -> torch.Tensor:
    if scale is None:
        return weight.to(torch.float32)
    if dtype == "I8":
        return prep.dequant_mxfp4(weight, scale)
    if dtype == "F8_E4M3":
        return prep.dequant_block_fp8(weight, scale)
    raise ValueError(f"unexpected quantized carrier {dtype}")


def _converted(name: str, dtype: str, weight: torch.Tensor, scale):
    """The publisher's converted form for one tensor."""
    if dtype == "I8":
        if scale is None:
            raise ValueError(f"{name}: MXFP4 payload without its scale sibling")
        fp8, block = prep.cast_e2m1fn_to_e4m3fn(weight, scale)
        yield name, fp8
        yield _scale_name(name), block
    elif dtype == "F8_E4M3":
        if name.endswith("attn.wo_a.weight"):
            # The publisher's one attention dequant, per inference/convert.py.
            yield name, prep.dequant_block_fp8(weight, scale).bfloat16()
        else:
            yield name, weight
            if scale is not None:
                yield _scale_name(name), scale
    else:
        yield name, weight
