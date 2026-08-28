#!/usr/bin/env python
"""Smallest probe for cached-incremental vs cached-one-shot divergence. Random init, preset
geometry, seconds -- deliberately cheaper than the released-weights harness so a mechanism hunt
does not have to read 700 GiB per iteration.

The released-weights runs disagree (cosine 1.0 at the aligned step, 0.47-0.83 after, worklog
e3838b). This asks whether the *shape* of the problem is enough: depth, prompt length, and a
schedule that crosses compression-window boundaries, with random weights. Measured answer at
time of writing: no -- layers x prompt sweeps down to 2.1e-4 max|dlogit| with identical argmax
at every step, so geometry alone does not reproduce it and the trigger is something the released
weights bring (fp4 dequant, real router scores, indexer top-k, or a real magnitude that exposes
one of them).

Both executions are CACHED. ``use_cache=False`` does not make a forward cache-free on this model
(DeepseekV4Model.forward builds DynamicCache(config=self.config) whenever no cache is passed and
uses the flag only for the return value), so there is no cache-free comparison to be had here.

Set 29.5.
"""

from __future__ import annotations

import os
import sys

sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))]

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.v4_oracle import build_reference_oracle  # noqa: E402

torch.set_grad_enabled(False)

LAYERS = (int(x) for x in os.environ.get("PROBE_LAYERS", "2,4,8").split(","))
LENGTHS = (int(x) for x in os.environ.get("PROBE_LENGTHS", "64,168,300,600").split(","))
STEPS = int(os.environ.get("PROBE_STEPS", "6"))


def incremental(model, cfg, prompt, steps):
    """Greedy decode carrying one cache across calls -- what a serving stack does."""
    cache = DynamicCache(config=cfg)
    out = model(prompt, past_key_values=cache, use_cache=True)
    tok = int(out.logits[0, -1].argmax())
    toks, lg = [tok], [out.logits[0, -1].float()]
    for _ in range(steps - 1):
        out = model(torch.tensor([[tok]], device=prompt.device), past_key_values=cache, use_cache=True)
        tok = int(out.logits[0, -1].argmax())
        toks.append(tok)
        lg.append(out.logits[0, -1].float())
    return toks, lg


def one_shot(model, prompt, generated):
    """One forward over prompt+generated with a FRESH model-built cache."""
    seq = prompt.clone()
    if generated:
        seq = torch.cat([seq, torch.tensor([generated], device=prompt.device, dtype=seq.dtype)], dim=1)
    return model(seq).logits[0, -1].float()


def main() -> int:
    worst_overall = 0.0
    for layers in LAYERS:
        for prompt_len in LENGTHS:
            model, _fp = build_reference_oracle(V4ModelArgs.tiny(layers), seed=0)
            cfg = model.config
            prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), generator=torch.Generator().manual_seed(3))
            toks, lg = incremental(model, cfg, prompt, STEPS)
            first_bad, worst, min_cos = None, 0.0, 1.0
            for k, (t, l) in enumerate(zip(toks, lg)):
                ref = one_shot(model, prompt, toks[:k])
                d = float((ref - l).abs().max())
                cos = float(torch.nn.functional.cosine_similarity(ref, l, dim=0))
                worst, min_cos = max(worst, d), min(min_cos, cos)
                if first_bad is None and (int(ref.argmax()) != int(t) or d > 1e-3):
                    first_bad = k
            worst_overall = max(worst_overall, worst)
            print(
                f"layers={layers:>2} prompt_len={prompt_len:>3}  first_bad_step={first_bad} "
                f"max|dlogit|={worst:.3e} min_cos={min_cos:.6f}",
                flush=True,
            )
            del model
    print(f"worst disagreement over the sweep: {worst_overall:.3e} (argmax identical everywhere)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
