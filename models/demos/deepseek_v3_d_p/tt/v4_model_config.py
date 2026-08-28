# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Model dimensions and layer schedule for DeepSeek-V4 bring-up.

One source of truth feeding BOTH the in-repo torch reference (the parity oracle)
and the TTNN modules. Duplicating dimensions across the two is how glue work
drifts into a model that quietly stops being V4.

Schedules
---------
The real V4-Flash schedule is produced by the config's own default rule: two
``heavily_compressed_attention`` bootstrap layers, then a ``compressed_sparse``
/ ``heavily_compressed`` interleave. Of those, only HCA has any TT
implementation (upstream ``tt/mla/heavily_compressed_attention.py``); CSA needs
the Lightning indexer plus ``indexer_score``/``sparse_sdpa``, which are
``TT_FATAL(arch == BLACKHOLE)`` and therefore unavailable on Wormhole.

So bring-up runs a **substituted** schedule. That is a real architectural
change and is named as one:

  ``"flash"``        the real 43-layer schedule. Not runnable end-to-end today.
  ``"hca_only"``     every compressive layer becomes HCA. Exercises the
                     compressor/KV path; still no indexer.
  ``"sliding_only"`` every layer becomes a plain sliding-window causal
                     attention. Fewest moving parts, so this is stage 1: it
                     proves embedding -> mHC -> attention -> MoE -> LM head ->
                     decode looping before any compressor is in play.

``mlp_layer_types`` is never substituted: the first ``num_hash_layers`` layers
stay ``hash_moe`` because hash routing is a distinct code path, and it depends on
``input_ids`` -- see ``V4ModelArgs.drive_reference``.

Sizes
-----
The published model is 304,180,418,494 parameters over 166,886,535,336 bytes of
weight shards. That does not fit one ASIC, so ``tiny`` exists: reduced
dimensions, random weights, same code paths. Nothing produced from ``tiny`` is a
statement about the real model's quality or speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig as F

if TYPE_CHECKING:  # pragma: no cover
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

SLIDING = "sliding_attention"
CSA = "compressed_sparse_attention"
HCA = "heavily_compressed_attention"

MOE = "moe"
HASH_MOE = "hash_moe"


def _interleave(n_layers: int, first_two_hca: bool = True) -> list[str]:
    """The config's own default rule, reproduced so a mismatch is loud.

    ``DeepseekV4Config`` builds ``[HCA]*min(n,2) + interleave`` where the
    interleave is HCA at even i and CSA at odd i. Verified against a 4-layer
    build, which yields ``[HCA, HCA, HCA, CSA]``.
    """
    head = [HCA] * min(n_layers, 2) if first_two_hca else []
    body = [CSA if i % 2 else HCA for i in range(max(n_layers - (2 if first_two_hca else 0), 0))]
    return (head + body)[:n_layers]


@dataclass(frozen=True)
class V4ModelArgs:
    """Dimensions for one V4 build. Defaults are the frozen V4-Flash values."""

    # core
    vocab_size: int = F.VOCAB_SIZE
    hidden_size: int = F.EMB_SIZE
    moe_intermediate_size: int = F.MOE_INTERMEDIATE_SIZE
    head_dim: int = F.HEAD_DIM
    num_hidden_layers: int = F.NUM_LAYERS

    # attention
    num_attention_heads: int = F.NUM_ATTENTION_HEADS
    num_key_value_heads: int = F.NUM_KEY_VALUE_HEADS
    q_lora_rank: int = F.Q_LORA_RANK
    o_lora_rank: int = F.O_LORA_RANK
    o_groups: int = F.O_GROUPS
    qk_rope_head_dim: int = F.QK_ROPE_HEAD_DIM
    sliding_window: int = F.SLIDING_WINDOW

    # indexer / compression
    index_n_heads: int = F.INDEX_N_HEADS
    index_head_dim: int = F.INDEX_HEAD_DIM
    index_topk: int = F.INDEX_TOPK
    compress_rates: dict = field(default_factory=lambda: dict(F.COMPRESS_RATES))

    # mHC
    hc_mult: int = F.HC_MULT
    hc_sinkhorn_iters: int = F.HC_SINKHORN_ITERS
    hc_eps: float = F.HC_EPS

    # MoE
    n_routed_experts: int = F.NUM_ROUTED_EXPERTS
    num_experts_per_tok: int = F.NUM_EXPERTS_PER_TOKEN
    n_shared_experts: int = F.NUM_SHARED_EXPERTS
    score_func: str = F.SCORE_FUNC
    route_scale: float = F.ROUTE_SCALE
    swiglu_limit: float = F.SWIGLU_LIMIT
    num_hash_layers: int = F.NUM_HASH_LAYERS

    # misc
    rms_norm_eps: float = F.RMS_NORM_EPS
    rope_theta: float = F.ROPE_THETA
    compress_rope_theta: float = F.COMPRESS_ROPE_THETA
    max_position_embeddings: int = F.MAX_POSITION_EMBEDDINGS

    schedule: str = "flash"

    # ---- presets ---------------------------------------------------------------

    @classmethod
    def tiny(cls, num_hidden_layers: int = 4, **overrides) -> "V4ModelArgs":
        """A V4-shaped model small enough for one ASIC with random weights.

        Brings every width down but keeps the *structural* features that make the
        glue non-trivial: ``hc_mult=4`` streams, grouped-O (``o_groups`` > 1), a
        single KV head, MoE with a shared expert, and at least one ``hash_moe``
        layer. Shrinking any of those would make the parity check prove less.
        """
        args = dict(
            vocab_size=512,
            hidden_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=64,
            q_lora_rank=64,
            o_lora_rank=64,
            o_groups=2,
            qk_rope_head_dim=16,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=16,
            num_hash_layers=1,
            max_position_embeddings=4096,
            # Experts dominate the parameter count, so they must shrink too or
            # "tiny" is not tiny (256 experts alone is ~190 M params here). The
            # top-k and shared-expert structure is kept, only the population is
            # reduced.
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
        )
        args.update(overrides)
        return cls(**args)

    # ---- schedules -------------------------------------------------------------

    def layer_types(self) -> list[str]:
        n = self.num_hidden_layers
        if self.schedule == "flash":
            return _interleave(n)
        if self.schedule == "hca_only":
            return [HCA] * n
        if self.schedule == "sliding_only":
            return [SLIDING] * n
        raise ValueError(f"unknown schedule {self.schedule!r}")

    def mlp_layer_types(self) -> list[str]:
        return [HASH_MOE] * min(self.num_hidden_layers, self.num_hash_layers) + [MOE] * max(
            0, self.num_hidden_layers - self.num_hash_layers
        )

    def requires_unavailable_ops(self) -> list[str]:
        """CSA layers need ops that do not exist on Wormhole. Empty == runnable."""
        return ["indexer_score / sparse_sdpa (BLACKHOLE-only)"] if CSA in self.layer_types() else []

    # ---- drive the reference ---------------------------------------------------

    def drive_reference(self) -> "DeepseekV4Config":
        """Build the reference config from these same dimensions.

        Going through ``DeepseekV4Config`` (rather than constructing modules
        directly) matters for two reasons: ``_init_weights`` fills the ``mhc``
        parameters, which the module classes leave as uninitialised
        ``torch.empty``; and it is what derives ``layer_types`` /
        ``mlp_layer_types`` when we do not pass them explicitly.
        """
        from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import (
            DeepseekV4Config,
        )

        kwargs = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"schedule"} and not k.startswith("_") and v is not None
        }
        kwargs["intermediate_size"] = self.moe_intermediate_size * 2
        kwargs["layer_types"] = self.layer_types()
        kwargs["mlp_layer_types"] = self.mlp_layer_types()
        kwargs["partial_rotary_factor"] = self.qk_rope_head_dim / self.head_dim
        return DeepseekV4Config(**kwargs)

    def with_layers(self, n: int) -> "V4ModelArgs":
        return replace(self, num_hidden_layers=n)

    # ---- mapping onto upstream's MoE gate --------------------------------- #

    def moe_gate_cfg(self) -> dict:
        """Keys ``TtMoEGatePrefill`` reads off a model config, filled for V4-Flash.

        Upstream's MoE gate already covers the V4 router, so the MoE half is a
        wiring job rather than a rewrite -- verified against the implementation,
        not inferred from its name:

        * ``tt_moe_gate_prefill.py:151`` -- "V4 variants ship
          SCORE_FUNC=\"sqrtsoftplus\"; V3/Kimi omit it and keep the sigmoid
          default", read via ``getattr(model_cfg, "SCORE_FUNC", cls.score_func)``.
        * ``topk_method="noaux_tc"`` selects on ``scores + e_score_correction_bias``
          and takes weights from the **un-biased** scores, then normalises and
          scales -- which is what ``DeepseekV4TopKRouter`` does.
        * hash routing is a first-class mode: ``RoutingType.HASH_HOST`` ("Host-first
          implementation reusing the V4 reference HashRouter") and
          ``HASH_DEVICE`` ("matmul device, moe_hash_gate device").

        The one mapping that needs care is group routing. V4's reference router has
        **no** expert-group stage (no ``group_score``/``topk_group`` semantics at
        all), whereas the gate branches on ``n_expert_groups``: ``== 1`` is the
        ungrouped path (commented "e.g. Kimi"), anything else takes grouped
        routing. So both group fields must be 1 to reproduce V4 numerics; leaving
        DeepSeek-V3 defaults in place would silently enable routing V4 does not do.

        Not verified here and to be checked at parity: the gate's normalisation
        epsilon versus the reference's literal ``1e-20``, and whether normalisation
        happens before or after the gather. Either difference is small but real.
        """
        return {
            "EMB_SIZE": self.hidden_size,
            "NUM_ROUTED_EXPERTS": self.n_routed_experts,
            "NUM_SHARED_EXPERTS": self.n_shared_experts,
            "NUM_EXPERTS_PER_TOKEN": self.num_experts_per_tok,
            # 1 disables the grouped-routing branch; V4 has no group stage at all.
            "NUM_EXPERT_GROUPS": 1,
            "NUM_LIMITED_GROUPS": 1,
            "ROUTE_SCALE": self.route_scale,
            "SCORE_FUNC": self.score_func,
        }

    def describe(self) -> str:
        types = self.layer_types()
        mlps = self.mlp_layer_types()
        counts = ", ".join(f"{t}={types.count(t)}" for t in dict.fromkeys(types))
        mcounts = ", ".join(f"{t}={mlps.count(t)}" for t in dict.fromkeys(mlps))
        blocked = self.requires_unavailable_ops()
        return (
            f"{self.num_hidden_layers} layers [{counts}] / mlp [{mcounts}] "
            f"| hidden={self.hidden_size} heads={self.num_attention_heads} hc={self.hc_mult} "
            f"experts={self.n_routed_experts}/{self.num_experts_per_tok}+{self.n_shared_experts}"
            f" | {'BLOCKED: ' + '; '.join(blocked) if blocked else 'all ops available'}"
        )
