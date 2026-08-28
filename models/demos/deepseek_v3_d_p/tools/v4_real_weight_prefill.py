# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced prefill over the **real** V4-Flash checkpoint, on host, no device.

Why this exists: every result so far came from the fingerprinted random-weight oracle, which
proves contracts but says nothing about the model. This script runs the reference
implementation with the released 155 GiB weights, so the port gets its first measurement
against actual model behaviour — and the first parity target that a device run can be
compared against as *tokens*, not just module outputs.

Why it streams. The released stack in bfloat16 is ~559 GiB (256 experts/layer dominates) and
this host has 421 GiB available, so a normal `from_pretrained` cannot work. Each layer is
materialised immediately before its forward and freed immediately after, which bounds peak
memory at roughly one layer plus the embedding/lm-head pair. Costs follow from that choice and
are printed per layer: one pass reads the whole checkpoint (155 GiB of stored payload, page
cached after the first token) and dequantizes ~148 G MXFP4 nibbles.

Deliberately cache-free: the whole prompt goes through in one forward, so this measures a
prefill, not an autoregressive loop. Multi-token generation would re-stream the stack per
step, so it is opt-in via --steps and each step is a fresh teacher-forced forward over the
growing prefix. That is slower than a cache but exercises no cache-position/rope-offset
assumption, which is the right thing to keep out of a first real-weights result.

Usage:
    python models/demos/deepseek_v3_d_p/tools/v4_real_weight_prefill.py \\
        --prompt-file /tmp/p.txt --out /home/ubuntu/tt-runs/<run>/result.json

Precision: everything is loaded into binary32. The publisher's MXFP4 and FP8 formats are both
exact into float32, so no rounding is introduced by the loader; the reference's own fp32 mHC
scalars stay fp32. Running in bfloat16 would halve the streamed-layer footprint but would also
silently cast the mHC/sinkhorn path, which is not a change to smuggle into a first result.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
from models.demos.deepseek_v3_d_p.tt import v4_weight_load as load
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import V4Checkpoint, default_snapshot_dir

MODEL_LEVEL = ("embed.weight", "norm.weight", "head.weight", "hc_head_base", "hc_head_fn", "hc_head_scale")


def set_param(model: torch.nn.Module, dotted: str, tensor: torch.Tensor) -> None:
    """Replace one meta parameter with the loaded one.

    Assigning ``param.data`` is refused across the meta boundary ("incompatible tensor type"),
    so the parameter object is swapped instead. Shapes are checked first: a wrong shape here
    would otherwise broadcast a weight into the module and train/serve a quietly transposed
    projection.
    """
    parent, _, attr = dotted.rpartition(".")
    module = model.get_submodule(parent) if parent else model
    current = getattr(module, attr)
    if tuple(current.shape) != tuple(tensor.shape):
        raise ValueError(
            f"{dotted}: model wants {tuple(current.shape)}, checkpoint gives {tuple(tensor.shape)} "
            f"(dtype {tensor.dtype}, requires_grad={current.requires_grad})"
        )
    setattr(module, attr, torch.nn.Parameter(tensor, requires_grad=False))


def free_module_params(module: torch.nn.Module) -> None:
    """Drop a finished layer's weights. Peak memory is bounded by one layer, not the stack."""
    for name, p in list(module.named_parameters()):
        parent, _, attr = name.rpartition(".")
        holder = module.get_submodule(parent) if parent else module
        setattr(
            holder,
            attr,
            torch.nn.Parameter(torch.empty(tuple(p.shape), dtype=p.dtype, device="meta"), requires_grad=False),
        )


def build(cfg, dtype: torch.dtype):
    """Model at real geometry with parameters on meta and *buffers* computed for real.

    ``include_buffers=False`` matters: the RoPE inverse-frequency tables are derived in
    ``__init__`` and would otherwise land on meta, and they are not in the checkpoint.
    """
    with init_empty_weights(include_buffers=False):
        model = DeepseekV4ForCausalLM(cfg)
    model.to(dtype)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--prompt-file", default=None, help="text file; stdin if omitted")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--steps", type=int, default=0, help="extra greedy tokens, teacher-forced")
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (plumbing smoke only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-prompt-tokens", type=int, default=256)
    args = ap.parse_args()

    snap = args.snapshot or default_snapshot_dir()
    if not snap:
        raise SystemExit("no V4-Flash snapshot; set DS4_V4_FLASH_DIR")
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    if args.layers:
        cfg.num_hidden_layers = args.layers
    ckpt = V4Checkpoint(snap)

    text = args.prompt if args.prompt is not None else (open(args.prompt_file).read() if args.prompt_file else "")
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens)
    ids = enc["input_ids"]
    print(f"prompt: {ids.shape[1]} tokens, vocab sample {ids[0, :6].tolist()}", flush=True)

    t0 = time.time()
    model = build(cfg, torch.float32)
    print(
        f"empty model built at real geometry in {time.time() - t0:.1f} s " f"({cfg.num_hidden_layers} layers)",
        flush=True,
    )

    for name in MODEL_LEVEL:
        if name in ckpt.index:
            set_param(model, load.reference_name(name), ckpt.dequantized(name, torch.float32))
    print(f"model-level tensors loaded ({time.time() - t0:.0f} s)", flush=True)

    timing, peak = [], 0
    for i, layer in enumerate(model.model.layers):

        def pre(mod, a, kw, i=i):
            t = time.time()
            for ref, tensor in load.iter_reference_layer(ckpt, i, dtype=torch.float32):
                set_param(mod, ref, tensor)
            timing.append({"layer": i, "load_s": round(time.time() - t, 2), "bytes_read": ckpt.bytes_read})
            print(
                f"  layer {i:2d} loaded in {time.time() - t:5.1f} s " f"({ckpt.bytes_read / 2**30:.1f} GiB read)",
                flush=True,
            )

        def post(mod, a, kw, out, i=i):
            free_module_params(mod)
            gc.collect()

        layer.register_forward_pre_hook(pre, with_kwargs=True)
        layer.register_forward_hook(post, with_kwargs=True)

    result = {"snapshot": os.path.basename(snap.rstrip("/")), "prompt_tokens": int(ids.shape[1]), "layers": []}
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0, -1].float()
        top = torch.topk(logits, 5)
        first = {
            "argmax_token": int(logits.argmax()),
            "argmax_text": tok.decode([int(logits.argmax())]),
            "top5": [
                {"id": int(t), "text": tok.decode([int(t)]), "logit": round(float(l), 3)}
                for t, l in zip(top.indices.tolist(), top.values.tolist())
            ],
            "logits_finite": bool(torch.isfinite(logits).all()),
            "logit_absmax": round(float(logits.abs().max()), 3),
        }
        result["prefill"] = first
        print(
            f"prefill done after {time.time() - t0:.0f} s: argmax={first['argmax_token']} "
            f"({first['argmax_text']!r}) finite={first['logits_finite']}",
            flush=True,
        )

        generated = ids
        for step in range(args.steps):
            out = model(input_ids=generated, use_cache=False)
            nxt = int(out.logits[0, -1].argmax())
            generated = torch.cat([generated, torch.tensor([[nxt]])], dim=1)
            result["layers"].append({"step": step, "token": nxt, "text": tok.decode([nxt])})
            print(f"step {step}: {nxt} {tok.decode([nxt])!r} ({time.time() - t0:.0f} s)", flush=True)

    result["timing_s"] = round(time.time() - t0, 1)
    result["per_layer_load_s"] = [t["load_s"] for t in timing]
    result["bytes_read_gib"] = round(ckpt.bytes_read / 2**30, 2)
    print(f"total {result['timing_s']} s, read {result['bytes_read_gib']} GiB", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
