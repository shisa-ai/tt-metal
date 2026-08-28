"""The cached decode path must carry state across forwards, using the harness's own calling pattern.

Regression guard for P05-SW-002. Letting the model build the cache and reusing
``out.past_key_values`` returns a cache of the right *type* -- it satisfies the per-layer
compression-state requirement, so nothing raises -- yet it fails to carry per-layer state between
forwards. Every decode step then attends only its own token. The symptom is invisible in a loss
value, and a suite that passes its own cache in never sees it: the reference generate contract
passed throughout the day this bug was live.

Runs at preset scale with random weights in seconds and needs no checkpoint.
"""

import pytest
import torch
from transformers import DynamicCache

from models.demos.deepseek_v3_d_p.tt.v4_model_config import V4ModelArgs
from models.demos.deepseek_v3_d_p.tt.v4_oracle import build_reference_oracle

STEPS = 5
PROMPT_LEN = 64


def _decode(model, prompt, *, caller_owns_cache: bool):
    """Greedy decode with one of the two calling patterns; returns (last token, per-step logits)."""
    if caller_owns_cache:
        cache = DynamicCache(config=model.config)
        out = model(prompt, use_cache=True, past_key_values=cache)
    else:
        out = model(prompt, use_cache=True)
        cache = out.past_key_values
    logits, token = [out.logits[0, -1].float()], None
    for _ in range(STEPS):
        token = int(logits[-1].argmax())
        out = model(torch.tensor([[token]]), use_cache=True, past_key_values=cache)
        logits.append(out.logits[0, -1].float())
    return token, logits


@pytest.mark.parametrize(
    "caller_owns_cache",
    [True, False],
    ids=["caller-owned", "from-model-output"],
)
def test_cached_decode_matches_recompute(caller_owns_cache):
    torch.set_grad_enabled(False)
    model, _ = build_reference_oracle(V4ModelArgs.tiny(4), seed=0)
    prompt = torch.randint(0, model.config.vocab_size, (1, PROMPT_LEN), generator=torch.Generator().manual_seed(9))

    tokens, logits = _decode(model, prompt, caller_owns_cache=caller_owns_cache)

    # ``logits[k]`` was produced after consuming the prompt plus tokens 0..k-1, so the comparison
    # is against a recompute of the *same* token stream. That isolates the question to "does a
    # cached step predict like a full forward" rather than "do two greedy trajectories agree".
    chosen = [int(l.argmax()) for l in logits]
    cosines = []
    for step in range(len(logits)):
        consumed = chosen[:step]
        seq = torch.cat([prompt, torch.tensor([consumed], dtype=torch.long)], dim=1) if consumed else prompt
        ref = model(seq).logits[0, -1].float()
        cosines.append(float(torch.nn.functional.cosine_similarity(ref, logits[step], dim=0)))

    assert cosines[0] == pytest.approx(1.0), "prefill itself disagrees with a recompute"
    if caller_owns_cache:
        assert min(cosines[1:]) > 0.999, f"cached decode drifted from the recompute: {cosines}"
    else:
        # Documented, not desired: the known-bad pattern is measurably wrong, and this assertion is
        # what proves the test can fail. If this ever stops holding, the reference fixed itself and
        # P05-SW-002 can be closed.
        assert min(cosines[1:]) < 0.9, f"model-built cache unexpectedly started round-tripping: {cosines}"
