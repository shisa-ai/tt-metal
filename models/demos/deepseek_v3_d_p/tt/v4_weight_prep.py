"""V4-Flash checkpoint -> device weight preparation: names, sharding, formats.

V4-Flash ships **quantized** weights and the HuggingFace ``deepseek_v4`` modelling code
has **no** dequantization path at all (no ``float4``/``e8m0``/block-scale handling in
``reference/deepseek_v4/modeling_deepseek_v4.py``), so anything that wants float weights
must produce them itself. This module is that path, and it is also the parity basis for
whatever the device loads.

Everything format-level here is **lineage-copied from the publisher's own converter**, not
inferred: ``inference/convert.py`` in the ``deepseek-ai/DeepSeek-V4-Flash-0731`` snapshot
(local snapshot ``7872f01b1d1fe23eabc4c98b48bffcef5a386062``, blob
``ab0b3cb7a122c785bad0f82babde50515be954ed``, accessed 2026-08-28). Where this file adds
behaviour the publisher does not have, it says so.

Measured checkpoint facts this encodes (headers of all 48 shards, see
``projects/05-deepseek-v4-flash/docs/WEIGHTS-AND-CAPACITY.md`` in the tenstorrent-testing
repo): expert weights are int8 storing two ``e2m1fn`` values per byte with one power-of-two
``e8m0`` scale per 32 elements; attention weights are ``float8_e4m3fn`` with one ``e8m0``
scale per 128x128 block; ``attn.wo_a`` is the one attention tensor the publisher
dequantizes to bfloat16.
"""

from __future__ import annotations

import torch

#: Elements sharing one e8m0 scale in the packed expert (fp4) layout.
FP4_BLOCK = 32
#: Elements sharing one e8m0 scale in the fp8 attention layout (128 x 128 blocks).
FP8_BLOCK = 128
#: e8m0 exponent bias.
E8M0_BIAS = 127

#: The e2m1fn value table, publisher's ``FP4_TABLE`` **verbatim**. The low nibble selects
#: index ``code`` and the high nibble ``code + 8``; the negative half is a sign bit, so the
#: table is not monotonic in the code and cannot be replaced by ``(-1)**b * value``.
#: Index 8 is ``+0.0``, not ``-0.0``: that is what the publisher writes, and it is
#: load-bearing for bitwise parity. Their e2m1fn->fp8 cast turns a code-8 nibble into
#: ``0x00``; a ``-0.0`` here produces ``0x80`` instead, which is numerically equivalent for
#: a matmul but breaks byte-for-byte comparison against their converted files. Pinned by
#: ``test_cast_matches_the_publishers_implementation_bitwise``.
E2M1FN_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

#: Which axis the publisher model-parallel-shards each tensor on. ``wq_b`` and ``wo_a``
#: are output-parallel; ``wo_b`` is **input**-parallel (``dim=1``), which is why a
#: blanket ``dim=0`` shard rule is wrong. Anything absent is not sharded.
SHARD_DIMS = {"wq_b": 0, "wo_a": 0, "wo_b": 1}

#: HF module names -> checkpoint/port names, from the publisher's ``mapping`` plus its
#: ``str.replace`` rules. Applied to checkpoint names, which already use port style.
NAME_RENAMES = (
    ("self_attn", "attn"),
    ("mlp", "ffn"),
    ("weight_scale_inv", "scale"),
    ("e_score_correction_bias", "bias"),
)


def e2m1fn_table(device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The 16-entry decode table as a tensor."""
    return torch.tensor(E2M1FN_VALUES, dtype=dtype, device=device)


def unpack_e2m1fn(packed: torch.Tensor) -> torch.Tensor:
    """``int8[out, in // 2]`` -> Float32``[out, in]``, low nibble first.

    Interleaving matters: the publisher takes ``low = x & 0xF`` and
    ``high = (x >> 4) & 0xF`` and stacks them along a new last axis before flattening, so
    the **low nibble is the even column** and the high nibble the odd one. Swapping them
    silently transposes the mantissa across the K axis.

    ``torch.float4_e2m1fn_x2`` is *not* a substitute: it exists in torch 2.11 but
    ``copy_kernel`` is unimplemented for it, so it cannot be cast to a float dtype. The
    explicit table is required, which is also why the publisher keeps one.
    """
    if packed.dtype is not torch.int8:
        raise TypeError(f"packed must be int8, got {packed.dtype}")
    if packed.ndim != 2:
        raise ValueError(f"expected [out, in // 2], got {tuple(packed.shape)}")
    table = e2m1fn_table(device=packed.device, dtype=torch.float32)
    raw = packed.view(torch.uint8)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()
    return torch.stack([table[low], table[high]], dim=-1).flatten(-2, -1)


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Power-of-two block scales to Float32.

    Accepts ``float8_e8m0fnu`` (already the right values) or the raw ``uint8``/``int8``
    bytes, where the value is ``2 ** (byte - 127)``.
    """
    if scale.dtype in (
        torch.float8_e8m0fnu,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float16,
    ):
        return scale.to(torch.float32)
    if scale.dtype in (torch.uint8, torch.int8):
        return torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=scale.device),
            scale.to(torch.float32).remainder(256.0) - E8M0_BIAS,
        )
    raise TypeError(f"unsupported e8m0 carrier dtype {scale.dtype}")


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 ``e2m1fn`` with 32-element ``e8m0`` scales -> Float32 ``[out, in]``.

    Exact: the decoded values and the power-of-two multiplier are both representable in
    binary32, so no rounding happens here (asserted in the tests).
    """
    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError("expected 2-D packed and scale")
    out, packed_in = packed.shape
    in_dim = packed_in * 2
    if scale.shape != (out, in_dim // FP4_BLOCK):
        raise ValueError(
            f"scale {tuple(scale.shape)} does not match [out, in/{FP4_BLOCK}] " f"for packed {tuple(packed.shape)}"
        )
    values = unpack_e2m1fn(packed)
    mult = e8m0_to_float(scale).repeat_interleave(FP4_BLOCK, dim=-1)
    return values * mult


def dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``float8_e4m3fn`` with 128x128 ``e8m0`` block scales -> Float32 ``[out, in]``.

    This is the publisher's ``wo_a`` dequantization written as a general function; the
    publisher applies it to ``wo_a`` only and keeps the other attention tensors in fp8.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected 2-D weight, got {tuple(weight.shape)}")
    out, in_dim = weight.shape
    if out % FP8_BLOCK or in_dim % FP8_BLOCK:
        raise ValueError(f"{out}x{in_dim} is not a multiple of {FP8_BLOCK}x{FP8_BLOCK}")
    if scale.shape != (out // FP8_BLOCK, in_dim // FP8_BLOCK):
        raise ValueError(
            f"scale {tuple(scale.shape)} does not match blocks of {FP8_BLOCK} " f"for weight {tuple(weight.shape)}"
        )
    w = weight.to(torch.float32)
    mult = e8m0_to_float(scale).repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    return w * mult


def cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor):
    """MXFP4 -> ``float8_e4m3fn`` + per-128-block ``e8m0``, publisher's function.

    Lineage copy of ``inference/convert.py::cast_e2m1fn_to_e4m3fn``, kept structurally
    close so a publisher update is diffable. It is **lossless** because the e2m1fn values
    are a subset of e4m3's: inside each 128x128 block the largest scale exponent is pulled
    out into the block scale and the 32-element scales become integer offsets.

    The fp8 path is what makes the realistic weight footprint ~287 GiB rather than the
    stored 155.42 GiB, so the FP4-vs-FP8 choice is capacity-dependent even though BF16 is
    not. See the capacity doc.
    """
    if x.dtype is not torch.int8:
        raise TypeError(f"x must be int8, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"expected 2-D x, got {tuple(x.shape)}")
    out_dim, in_dim = x.shape
    in_dim *= 2
    if in_dim % FP8_BLOCK or out_dim % FP8_BLOCK:
        raise ValueError(f"{out_dim}x{in_dim} must be a multiple of {FP8_BLOCK}")
    if scale.shape != (out_dim, in_dim // FP4_BLOCK):
        raise ValueError(f"scale {tuple(scale.shape)} does not match [out, in/{FP4_BLOCK}]")

    # max_fp4 (6.0) * 2**6 = 384 < 448 (e4m3 max) but 6.0 * 2**7 = 768 > 448.
    max_offset_bits = 6
    values = unpack_e2m1fn(x)
    b_out, b_in = out_dim // FP8_BLOCK, in_dim // FP8_BLOCK
    values = values.view(b_out, FP8_BLOCK, b_in, FP8_BLOCK).transpose(1, 2)
    scales = e8m0_to_float(scale).view(b_out, FP8_BLOCK, b_in, -1).transpose(1, 2).flatten(2)
    block_scale = scales.amax(dim=-1, keepdim=True) / (2**max_offset_bits)
    offset = scales / block_scale
    offset = offset.unflatten(-1, (FP8_BLOCK, -1)).repeat_interleave(FP4_BLOCK, dim=-1)
    values = (values * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return (values.to(torch.float8_e4m3fn), block_scale.squeeze(-1).to(torch.float8_e8m0fnu))


def port_name(name: str) -> str:
    """Checkpoint name -> port/device name, applying the publisher's renames.

    Idempotent on already-converted names, which is what the publisher relies on too,
    since its input may arrive in either HF or port style.
    """
    out = name[len("model.") :] if name.startswith("model.") else name
    for old, new in NAME_RENAMES:
        out = out.replace(old, new)
    return out


def shard_dim(name: str):
    """The axis to shard on, or ``None`` when the publisher does not shard it."""
    leaf = name.replace(".weight", "").replace(".scale", "").split(".")[-1]
    return SHARD_DIMS.get(leaf)


def owner_rank(name: str, n_experts: int, ep_size: int):
    """Expert-parallel owner for an expert tensor, or ``None`` if it is replicated.

    The publisher groups whole experts per rank (``n_local_experts = n_experts // ep`` and
    rank ``i`` keeps experts ``[i*n_local, (i+1)*n_local)``), so with 256 experts and
    ``ep_size = 32`` each rank owns 8 consecutive experts. A weight laid out by interleave
    rather than by block would not match this and would silently mix experts.
    """
    if n_experts % ep_size:
        raise ValueError(f"{n_experts} experts not divisible by ep_size {ep_size}")
    parts = name.split(".")
    if "experts" not in parts or "shared_experts" in parts:
        return None
    idx = int(parts[parts.index("experts") + 1])
    return idx // (n_experts // ep_size)
