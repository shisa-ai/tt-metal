"""Last unflipped axis: RELEASED widths with RANDOM values, 2 layers, no checkpoint.

tiny() carries the released schedules on shrunk widths (hidden 256, vocab 512, 8 experts,
index_topk 16), so every "geometry does not reproduce it" claim so far was unsupported. This
overrides the widths from the released config.json and keeps random init, so the only thing
present is released size -- no released values.

Read the result at LOGIT level only. With random values the logits are structureless over
129280 tokens, so any 1e-4 gap flips argmax; argmax agreement here would manufacture a false
positive. Baseline for comparison: released values at 43 layers gave cosine 0.47-0.83 and
max|dlogit| 19-33 between the same two schedules.
"""

import dataclasses
import json
import os
import sys

sys.path[:0] = ["/home/ubuntu/tt-metal-ds4-port", "/home/ubuntu/tt-metal-ds4-port/models/demos/deepseek_v3_d_p"]

import torch
from transformers import DynamicCache

from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs
from models.demos.deepseek_v3_d_p.tt.v4_oracle import build_reference_oracle

torch.set_grad_enabled(False)
SNAP = os.environ["DS4_V4_FLASH_DIR"].rstrip("/") + "/"
cfg = json.load(open(SNAP + "config.json"))
fields = {f.name for f in dataclasses.fields(V4ModelArgs)}
over = {k: v for k, v in cfg.items() if k in fields and isinstance(v, (int, float)) and k != "num_hidden_layers"}
print("released widths applied:", {k: v for k, v in sorted(over.items()) if k != "vocab_size"})

model, fp = build_reference_oracle(V4ModelArgs.tiny(2, **over), seed=0)
c = model.config
print(
    f"built: hidden={c.hidden_size} vocab={c.vocab_size} experts={getattr(c,'n_routed_experts','?')} "
    f"topk={getattr(c,'num_experts_per_tok','?')} index_topk={getattr(c,'index_topk','?')} "
    f"layers={c.num_hidden_layers}  RSS check: {os.popen('ps -o rss= -p %d' % os.getpid()).read().strip()}"
)

prompt = torch.randint(0, c.vocab_size, (1, 64), generator=torch.Generator().manual_seed(3))
cache = DynamicCache(config=c)
out = model(prompt, past_key_values=cache, use_cache=True)
tok = int(out.logits[0, -1].argmax())
toks, lg = [tok], [out.logits[0, -1].float()]
for _ in range(5):
    out = model(torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
    tok = int(out.logits[0, -1].argmax())
    toks.append(tok)
    lg.append(out.logits[0, -1].float())

print("\nstep  cos(incremental, one-shot)   max|dlogit|   mean|d|/mean|l|   argmax same")
for k, l in enumerate(lg):
    seq = torch.cat([prompt, torch.tensor([toks[:k]], dtype=torch.long)], dim=1)
    ref = model(seq).logits[0, -1].float()
    cos = float(torch.nn.functional.cosine_similarity(ref, l, dim=0))
    d = (ref - l).abs()
    print(
        f"{k:>4}  {cos:>10.6f}            {float(d.max()):>8.3f}      "
        f"{float(d.mean()) / float(l.abs().mean()):>10.3e}        {int(ref.argmax()) == toks[k]}"
    )
