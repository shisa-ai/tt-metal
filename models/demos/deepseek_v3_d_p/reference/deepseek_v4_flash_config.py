# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V4 Flash Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for DeepSeek-V4-Flash.
"""


class DeepSeekV4FlashConfig:
    """DeepSeek V4 Flash model dimensions."""

    # Core dimensions
    EMB_SIZE = 4096  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    HEAD_DIM = 512

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 6
    NUM_SHARED_EXPERTS = 1
    # V4 drops V3's expert-group routing: a single group means the gate collapses to a plain top-k.
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    # V4 replaces V3/Kimi's sigmoid router affinity with sqrt(softplus(.)).
    SCORE_FUNC = "sqrtsoftplus"

    # Model architecture
    NUM_LAYERS = 43
    NUM_HASH_LAYERS = 3
    VOCAB_SIZE = 129280
    SLIDING_WINDOW = 128

    # MLA dimensions
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 1
    Q_LORA_RANK = 1024
    O_LORA_RANK = 1024
    O_GROUPS = 8
    QK_ROPE_HEAD_DIM = 64

    # Indexer / sparse attention (NSA-style)
    INDEX_N_HEADS = 64
    INDEX_HEAD_DIM = 128
    INDEX_TOPK = 512
    # Compressed attention config
    COMPRESS_RATES = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
    COMPRESS_ROPE_THETA = 160000.0
    # Per-layer schedule, verbatim from the checkpoint's legacy `compress_ratios` key:
    # 0 = sliding, 4 = CSA, 128 = HCA. This IS a checkpoint fact, and it is what makes
    # V4-Flash differ from the config class's own V4-Pro default (2x HCA bootstrap, then
    # HCA/CSA interleave, no sliding layers). `DeepseekV4Config` maps it through
    # `_COMPRESS_RATIO_TO_LAYER_TYPE` and truncates to `num_hidden_layers`; the list is
    # 46 entries long for a 43-layer model, so truncation is part of the contract.
    # Written as the pattern it actually is: two sliding layers, then 20 (CSA, HCA)
    # pairs, one trailing CSA, then the three MTP-ish zeros the checkpoint carries.
    # Truncation to num_hidden_layers happens in the config class.
    COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4] + [0] * 3
    # YaRN block from the checkpoint's `rope_scaling`. It applies to the *compress*
    # rope group only; `DeepseekV4Config.__post_init__` splits it and injects
    # `attention_factor=1.0`, because the V4 reference does not apply YaRN's mscale.
    ROPE_SCALING = {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    HC_MULT = 4
    HC_SINKHORN_ITERS = 20
    HC_EPS = 1.0e-6

    # Other
    RMS_NORM_EPS = 1e-6
    ROUTE_SCALE = 1.5
    ROPE_THETA = 10000
    SWIGLU_LIMIT = 10.0
    MAX_POSITION_EMBEDDINGS = 1048576
