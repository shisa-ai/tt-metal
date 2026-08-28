# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Greedy decode of the **released** V4-Flash weights on host, one expert at a time.

Every earlier real-weight result here was a single prefill: one forward, one next token, then
the whole 155 GiB stack thrown away. That was not timidity about decode, it was arithmetic --
streaming all 43 layers per generated token costs ~1400 s/token, so eight tokens is three
hours and the run is a nightly job nobody can iterate on.

The cost was in the wrong place. A decode step needs each layer's attention weights (always)
and the routed experts the router actually selects -- `Experts.forward` indexes
``gate_up_proj[expert_idx]`` per hit, six of them. Holding a ~6 GiB stack per layer to use a
few hundred MB is what made decode expensive, so this tool keeps the non-expert weights
resident and materialises experts on demand (`LazyExpertStack`). The bytes still come from
page cache, which the prefill pass warmed.

What this can show and what it cannot:

* It exercises the reference model's **cache** path (``use_cache=True``, ``seq_len == 1``
  decode) with real weights for the first time here. The port has to mirror exactly that
  behaviour on device, including the sliding layers' window-trimmed cache, so a failure here
  is information about the contract the port must meet -- not a port bug.
* It is CPU/host math in fp32 against the publisher's own modules. Nothing in this file
  executes on a Tenstorrent accelerator, and no result here says anything about device
  numerics, throughput, or the ported TTNN modules.
* Correctness evidence is a greedy continuation plus a guard: the first token must match the
  id the streaming prefill harness measured for the same prompt with cache disabled
  (17575 `` Berlin`` for the France/Germany prompt). Two different code paths agreeing on the
  first token is a real check; a plausible-looking sentence is still just a sentence.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))]

from models.demos.deepseek_v3_d_p.tt.v4_weight_load import (  # noqa: E402
    MODEL_LEVEL,
    LazyExpertStack,
    build,
    iter_reference_layer,
    reference_name,
    set_param,
)
from models.demos.deepseek_v3_d_p.tt.v4_weight_stream import V4Checkpoint, default_snapshot_dir  # noqa: E402

EXPERT_ATTRS = {"mlp.experts.gate_up_proj": "gate_up", "mlp.experts.down_proj": "down"}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def install_layer(ckpt: V4Checkpoint, layer, index: int, dtype, stats: list, lazy: bool = True) -> int:
    """Load one layer's non-expert weights and stand in lazy stacks for its experts."""
    t0 = time.time()
    experts = 0
    # skip_experts keeps the routed experts in the file: a lazy stack reads them per hit, and
    # building 256-expert stacks here would dequantize ~24 GiB per layer to throw it away.
    for ref, tensor in iter_reference_layer(ckpt, index, dtype=dtype, skip_experts=lazy):
        set_param(layer, ref, tensor)
        del tensor
    if not lazy:
        return 0
    for ref, role in EXPERT_ATTRS.items():
        parent, _, attr = ref.rpartition(".")
        module = layer.get_submodule(parent)
        if not hasattr(module, attr):
            continue  # dense layers have no routed experts
        before = ckpt.bytes_read
        delattr(module, attr)  # a registered Parameter refuses a non-tensor assignment
        setattr(module, attr, LazyExpertStack(ckpt, index, role, dtype=dtype))
        experts += 1
        del before
    stats.append({"layer": index, "load_s": round(time.time() - t0, 2)})
    print(
        f"  layer {index:2d} resident in {time.time() - t0:5.1f} s "
        f"(RSS {rss_gib():.1f} GiB, {ckpt.bytes_read / 2**30:.1f} GiB read)",
        flush=True,
    )
    return experts


def decode(model, ids, position, tokens, on_token, collect=None):
    """Prefill with cache, then greedy ``seq_len == 1`` steps.

    The cache is whatever the model builds for itself. Passing a stock ``DynamicCache()`` fails
    with ``'DynamicLayer' object has no attribute 'store_compression_weights'``: V4 keeps
    compressor state on the *per-layer* cache (``DeepseekV4HCACache`` / ``DeepseekV4CSACache``),
    and the model only constructs those when it creates the cache from its own config
    (``DynamicCache(config=self.config)``). That per-layer compression state is precisely what
    the device port has to reproduce, so it is worth knowing the reference will not accept a
    generic cache -- and that ``StaticCache`` is ruled out by the same comment in the modelling
    file.
    """
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0, -1].float()
        nxt = int(logits.argmax())
        first = {"logits_finite": bool(torch.isfinite(logits).all()), "top5": torch.topk(logits, 5).indices.tolist()}
        on_token(nxt, first)
        if collect is not None:
            collect.append(logits.clone())
        for step in range(1, tokens):
            pos = torch.full((1, 1), position + step, dtype=torch.long)
            out = model(input_ids=torch.tensor([[nxt]]), use_cache=True, past_key_values=cache, position_ids=pos)
            logits = out.logits[0, -1].float()
            nxt = int(logits.argmax())
            on_token(
                nxt,
                {"logits_finite": bool(torch.isfinite(logits).all()), "top5": torch.topk(logits, 5).indices.tolist()},
            )
            if collect is not None:
                collect.append(logits.clone())
    return first


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument(
        "--teacher-force-ids",
        default=None,
        help="comma-separated token ids: skip decoding entirely and run ONE cache-free forward "
        "over them, writing per-position argmax/top-5 for offline comparison against a cached "
        "run. Separate invocation on purpose -- calling the cache-free path on a model that "
        "has already decoded with a cache dies with \"'DynamicCache' object is not "
        'subscriptable", so sharing a process would contaminate the control.',
    )
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (smoke only)")
    ap.add_argument("--expect-first-token", type=int, default=None, help="guard: prefill argmax must equal this")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--teacher-forcing-check",
        type=int,
        default=0,
        help="after decoding N tokens, re-run ONE cache-free forward over prompt+generated "
        "and compare the prediction at every position -- the standard cache-equivalence check",
    )
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument(
        "--experts-implementation",
        default="eager",
        help="'eager' runs the module's own per-expert loop and is what makes a lazy stack "
        "possible; 'grouped_mm' needs the whole stack materialised (transformers' fused path "
        "hands the *stack* to grouped_mm, not per-expert slices)",
    )
    args = ap.parse_args()

    snap = default_snapshot_dir()
    if snap is None:
        print("no snapshot; set DS4_V4_FLASH_DIR", file=sys.stderr)
        return 2
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    if args.layers:
        # Same truncation the prefill harness uses, so the two harnesses describe the same
        # (sub)model: the module list is sized by num_hidden_layers and indexes the released
        # schedules per layer, so lowering the count is enough. compress_ratios is not even
        # kept as an attribute on the driven config -- it is consumed to derive layer_types.
        cfg.num_hidden_layers = args.layers
    dtype = getattr(torch, args.dtype)

    t0 = time.time()
    model = build(cfg, dtype)
    print(
        f"empty model at {cfg.num_hidden_layers} layers in {time.time() - t0:.1f} s (RSS {rss_gib():.1f} GiB)",
        flush=True,
    )
    ck = V4Checkpoint(snap)
    for name in MODEL_LEVEL:  # checkpoint-side names; the target path is the reference's
        if name in ck.index:
            set_param(model, reference_name(name), ck.dequantized(name, dtype))

    stats, stacks = [], 0
    # Set it explicitly, including for "eager": the default dispatch resolves to transformers'
    # fused grouped_mm path, which hands the *whole stack* to grouped_mm and so is incompatible
    # with a lazy per-expert stand-in by construction.
    model.set_experts_implementation(args.experts_implementation)
    lazy = args.experts_implementation == "eager"
    for i, layer in enumerate(model.model.layers):
        stacks += install_layer(ck, layer, i, dtype, stats, lazy=lazy)
    print(
        f"model resident: {stacks} expert stacks lazy, RSS {rss_gib():.1f} GiB, " f"{time.time() - t0:.0f} s elapsed",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    if args.teacher_force_ids:
        seq = [int(x) for x in args.teacher_force_ids.split(",")]
        print(f"teacher-forcing {len(seq)} ids, cache off", flush=True)
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([seq]), use_cache=False).logits[0].float()
        rows = []
        for pos in range(len(seq)):
            top = torch.topk(lg[pos], 5)
            rows.append(
                {
                    "position": pos,
                    "input_id": seq[pos],
                    "argmax": int(lg[pos].argmax()),
                    "argmax_text": tok.decode([int(lg[pos].argmax())]),
                    "margin_top2": round(float(top.values[0] - top.values[1]), 4),
                    "top5": [
                        {"id": int(i), "text": tok.decode([int(i)]), "logit": round(float(v), 4)}
                        for i, v in zip(top.indices.tolist(), top.values.tolist())
                    ],
                }
            )
            print(
                f"  pos {pos:2d} argmax {rows[-1]['argmax']:6d} {rows[-1]['argmax_text']!r:12s} "
                f"margin={rows[-1]['margin_top2']}",
                flush=True,
            )
        out = {
            "mode": "teacher_forced_cache_free",
            "layers": cfg.num_hidden_layers,
            "dtype": args.dtype,
            "seq": seq,
            "rows": rows,
            "bytes_read_gib": round(ck.bytes_read / 2**30, 2),
            "elapsed_s": round(time.time() - t0, 1),
        }
        if args.out:
            json.dump(out, open(args.out, "w"), indent=1)
            print("wrote", args.out, flush=True)
        return 0

    ids = tok(args.prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"]
    prompt_tokens = int(ids.shape[1])
    print(f"prompt: {prompt_tokens} tokens {args.prompt!r}", flush=True)

    generated, guard = [], {"expected": args.expect_first_token, "passed": None}
    result = {
        "snapshot": os.path.basename(snap.rstrip("/")),
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "layers": cfg.num_hidden_layers,
        "dtype": args.dtype,
        "lazy_expert_stacks": stacks,
        "experts_implementation": args.experts_implementation,
        "resident_load_s": round(time.time() - t0, 1),
        "peak_rss_gib": None,
        "bytes_read_gib": None,
        "tokens": [],
        "decode_note": "host CPU fp32 reference modules; no Tenstorrent device involved",
    }

    def on_token(nxt, extra):
        step = len(generated)
        generated.append({"id": nxt, "text": tok.decode([nxt])})
        result["tokens"] = generated
        result["peak_rss_gib"] = round(rss_gib(), 1)
        result["bytes_read_gib"] = round(ck.bytes_read / 2**30, 2)
        if step == 0 and args.expect_first_token is not None:
            guard["actual"] = nxt
            guard["passed"] = nxt == args.expect_first_token
            print(
                f"  first token {nxt} {generated[0]['text']!r} "
                f"guard(expect {args.expect_first_token}) = {guard['passed']}",
                flush=True,
            )
        print(
            f"  +[{step:2d}] {nxt:6d} {generated[-1]['text']!r:12s} "
            f"finite={extra['logits_finite']} RSS {rss_gib():.1f} GiB "
            f"({time.time() - t0:.0f} s)",
            flush=True,
        )
        if args.out:
            json.dump(result, open(args.out + ".partial", "w"), indent=1)

    cached_logits: list[torch.Tensor] = []
    try:
        decode(model, ids, prompt_tokens - 1, args.tokens, on_token, collect=cached_logits)
    finally:
        result["guard"] = guard
        result["generated_text"] = args.prompt + "".join(t["text"] for t in generated)
        result["expert_materializations"] = sum(
            m.materializations
            for m in (getattr(l.mlp.experts, "gate_up_proj", None) for l in model.model.layers)
            if isinstance(m, LazyExpertStack)
        )
        if args.out:
            # Written fresh rather than only renamed: a run that dies before the first token
            # has nothing on disk to rename, and losing the resident-load timings to a crash
            # in the forward is exactly how an 18-minute pass becomes an empty directory.
            json.dump(result, open(args.out, "w"), indent=1)
            print("wrote", args.out, flush=True)
        print("continuation:", repr(result["generated_text"][-160:]), flush=True)
    if args.teacher_forcing_check:
        n = min(args.teacher_forcing_check, len(generated))
        seq = ids[0].tolist() + [g["id"] for g in generated[:n]]
        print(f"teacher-forcing {n} generated tokens over {len(seq)} positions (cache off)", flush=True)
        before = ck.bytes_read
        with torch.no_grad():
            ref = model(input_ids=torch.tensor([seq]), use_cache=False).logits[0].float()
        rows, agree = [], 0
        for step in range(n):
            # Position prompt_tokens+step-1 predicts generated[step]; the cached run produced
            # it from that same prefix, so agreement at every step is cache equivalence.
            pos = prompt_tokens + step - 1
            predicted = int(ref[pos].argmax())
            match = predicted == generated[step]["id"]
            agree += match
            rows.append(
                {
                    "step": step,
                    "position": pos,
                    "cached_id": generated[step]["id"],
                    "teacher_forced_id": predicted,
                    "match": match,
                    "cached_text": generated[step]["text"],
                    "teacher_forced_text": tok.decode([predicted]),
                }
            )
            if step < len(cached_logits):
                # Argmax agreement alone cannot tell a broken cache from a near-tie: a truncated
                # stack produces almost-degenerate logits, and fp32 reduction order differs
                # between a [1, S] prefill and a [1, 1] decode. Compare the vectors -- a cache
                # error moves the whole distribution, a tie flips one id among near-equals.
                a, b = cached_logits[step], ref[pos]
                top2 = torch.topk(b, 2)
                rows[-1].update(
                    forced_margin=round(float(top2.values[0] - top2.values[1]), 4),
                    logit_max_abs_delta=round(float((a - b).abs().max()), 5),
                    logit_cosine=round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 6),
                    cached_top5=torch.topk(a, 5).indices.tolist(),
                    forced_top5=top2.indices.tolist(),
                )
            print(
                f"  step {step}: cached {generated[step]['id']} {generated[step]['text']!r} vs "
                f"forced {predicted} {tok.decode([predicted])!r} {'OK' if match else 'MISMATCH'} "
                f"margin={rows[-1].get('forced_margin')} dlogit={rows[-1].get('logit_max_abs_delta')} "
                f"cos={rows[-1].get('logit_cosine')}",
                flush=True,
            )
        result["teacher_forcing"] = {
            "checked": n,
            "agreements": agree,
            "all_match": agree == n,
            "bytes_read_gib": round((ck.bytes_read - before) / 2**30, 2),
            "rows": rows,
        }
        print(f"teacher-forcing agreement: {agree}/{n}", flush=True)
        if args.out:
            json.dump(result, open(args.out, "w"), indent=1)

    return 0 if guard["passed"] is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
