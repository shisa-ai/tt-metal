# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Decoder Layer.

Each layer has 7 RMSNorms + layer_scalar:
  - input_layernorm: before attention
  - post_attention_layernorm: after attention, before residual add
  - pre_feedforward_layernorm: before shared MLP
  - post_feedforward_layernorm: after combined MLP+MoE, before final residual add
  - post_feedforward_layernorm_1: after shared MLP output (MoE path only)
  - pre_feedforward_layernorm_2: before expert input (MoE path only)
  - post_feedforward_layernorm_2: after expert output (MoE path only)
  - layer_scalar: learned per-layer scalar

Forward flow (matching HF exactly):
  residual = x
  x = input_layernorm(x)
  x = self_attn(x)
  x = post_attention_layernorm(x)
  x = residual + x

  residual = x
  x = pre_feedforward_layernorm(x)
  x = mlp(x)

  if enable_moe_block:
    x_1 = post_feedforward_layernorm_1(x)
    x_flat = residual.reshape(-1, H)     # router input = pre-norm residual
    _, top_k_w, top_k_idx = router(x_flat)
    x_2 = pre_feedforward_layernorm_2(x_flat)
    x_2 = experts(x_2, top_k_idx, top_k_w)
    x_2 = post_feedforward_layernorm_2(x_2)
    x = x_1 + x_2

  x = post_feedforward_layernorm(x)
  x = residual + x
  x *= layer_scalar
"""

import os

import torch

import ttnn
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.gemma4_attention_config import get_attention_program_config
from models.demos.gemma4.tt.moe import MoEBlock
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.tt.shared_mlp import SharedMLP
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate

PLI_PROJECTION_COMPUTE_PROFILE_ENV = "GEMMA4_PLI_PROJECTION_COMPUTE_PROFILE"
PLI_PROJECTION_LAYER0_HIFI3_FP32_ACC_PROFILE = "layer0_hifi3_fp32_acc"
PLI_PROJECTION_COMPUTE_PROFILES = frozenset({PLI_PROJECTION_LAYER0_HIFI3_FP32_ACC_PROFILE})


def _pli_projection_compute_kernel_config(hidden_states, layer_idx):
    """Return the opt-in layer-0 PLI projection compute profile, if selected."""
    profile = os.environ.get(PLI_PROJECTION_COMPUTE_PROFILE_ENV) or None
    if profile is None:
        return None
    if profile not in PLI_PROJECTION_COMPUTE_PROFILES:
        supported = ", ".join(sorted(PLI_PROJECTION_COMPUTE_PROFILES))
        raise ValueError(f"{PLI_PROJECTION_COMPUTE_PROFILE_ENV} must be one of: " f"{supported}; got {profile!r}")
    if layer_idx != 0:
        return None

    device = hidden_states.device()
    if device is None:
        raise ValueError(f"{PLI_PROJECTION_COMPUTE_PROFILE_ENV}={profile} requires a " "device-resident activation")
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class Gemma4DecoderLayer:
    BOUNDARY_CAPTURE_NAMES = (
        "layer_input",
        "post_attention_residual",
        "post_mlp_residual",
        "post_pli_residual",
        "pre_layer_scalar",
        "layer_output",
    )
    MLP_BOUNDARY_CAPTURE_NAMES = (
        "pre_mlp_norm",
        *SharedMLP.BOUNDARY_CAPTURE_NAMES,
        "post_feedforward_norm",
        "post_mlp_residual",
    )

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        dtype,
        tensor_cache_path,
        mesh_config,
        max_seq_len,
        max_local_batch_size,
        shared_mlp_dtype=None,
        attention_dtype=None,
        experts_dtype=None,
        router_dtype=None,
        bounded_sliding_kv_cache: bool = False,
        transformation_mats=None,  # Legacy — ignored (HF-style RoPE needs no transformation mats)
    ):
        # Per-module dtype overrides default to the model-wide ``dtype`` so
        # callers that don't care about precision config see no change.
        if shared_mlp_dtype is None:
            shared_mlp_dtype = dtype
        if attention_dtype is None:
            attention_dtype = dtype
        if experts_dtype is None:
            experts_dtype = dtype
        if router_dtype is None:
            router_dtype = dtype
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.hidden_size = hf_config.hidden_size
        self.layer_type = hf_config.layer_types[layer_idx]
        self.enable_moe_block = hf_config.enable_moe_block
        self.hidden_size_per_layer_input = getattr(hf_config, "hidden_size_per_layer_input", 0) or 0

        # Try both key formats (HF uses "model.language_model.layers", tests use "model.layers")
        layer_state = {}
        if state_dict:
            for prefix in [f"model.language_model.layers.{layer_idx}", f"model.layers.{layer_idx}"]:
                layer_state = substate(state_dict, prefix)
                if layer_state:
                    break

        def _norm(name, with_scale=True):
            return RMSNorm(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=substate(layer_state, name) if layer_state else {},
                tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/{name}" if tensor_cache_path else None,
                mesh_config=mesh_config,
                with_scale=with_scale,
            )

        # 4 norms present on every layer
        self.input_layernorm = _norm("input_layernorm")
        self.post_attention_layernorm = _norm("post_attention_layernorm")
        self.pre_feedforward_layernorm = _norm("pre_feedforward_layernorm")
        self.post_feedforward_layernorm = _norm("post_feedforward_layernorm")

        # 3 additional norms for MoE layers
        if self.enable_moe_block:
            self.post_feedforward_layernorm_1 = _norm("post_feedforward_layernorm_1")
            self.pre_feedforward_layernorm_2 = _norm("pre_feedforward_layernorm_2")
            self.post_feedforward_layernorm_2 = _norm("post_feedforward_layernorm_2")

        # Layer scalar
        if layer_state and "layer_scalar" in layer_state:
            self.layer_scalar = layer_state["layer_scalar"].item()
        else:
            self.layer_scalar = 1.0

        # Attention
        attn_config = Gemma4AttentionConfig(hf_config, layer_idx)
        attn_program_config = get_attention_program_config(attn_config, mesh_config, is_decode=True)
        self.self_attn = Gemma4Attention(
            mesh_device=mesh_device,
            config=attn_config,
            state_dict=substate(layer_state, "self_attn") if layer_state else {},
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=attn_program_config,
            layer_idx=layer_idx,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/self_attn" if tensor_cache_path else None,
            weight_dtype=attention_dtype,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )

        # Shared/dense MLP (HF key: "mlp")
        self.shared_mlp = SharedMLP(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=substate(layer_state, "mlp") if layer_state else {},
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            dtype=shared_mlp_dtype,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/mlp" if tensor_cache_path else None,
        )

        # MoE block (router + routed experts) — split dtypes between the two
        if self.enable_moe_block:
            self.moe = MoEBlock(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=layer_state,  # MoE expects "router.*" and "experts.*" keys
                ccl_manager=ccl_manager,
                mesh_config=mesh_config,
                dtype=experts_dtype,
                router_dtype=router_dtype,
                tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/moe" if tensor_cache_path else None,
            )

        # Per-layer input embeddings (E2B/E4B feature)
        if self.hidden_size_per_layer_input:
            pli_prefix = f"{tensor_cache_path}/layer_{layer_idx}" if tensor_cache_path else None

            if layer_state and "per_layer_input_gate.weight" in layer_state:
                gate_w = layer_state["per_layer_input_gate.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
                proj_w = layer_state["per_layer_projection.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            else:
                gate_w = None
                proj_w = None

            self.per_layer_input_gate = ttnn.as_tensor(
                gate_w,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=get_cache_file_name(pli_prefix, "per_layer_input_gate"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.per_layer_projection = ttnn.as_tensor(
                proj_w,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=get_cache_file_name(pli_prefix, "per_layer_projection"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.post_per_layer_input_norm = _norm("post_per_layer_input_norm")

    def _capture_boundary(self, boundary_name, hidden_states):
        """Synchronously expose an opt-in diagnostic boundary.

        Normal execution never installs ``_boundary_capture_callback`` and
        therefore only pays the attribute lookup. A diagnostic callback must
        copy any tensor it needs before returning because later operations may
        reuse or deallocate the underlying device storage.
        """
        callback = getattr(self, "_boundary_capture_callback", None)
        if callback is not None:
            callback(self.layer_idx, boundary_name, hidden_states)

    def _capture_mlp_boundary(self, boundary_name, value):
        """Synchronously expose an opt-in layer-local MLP boundary."""
        callback = getattr(self, "_mlp_boundary_capture_callback", None)
        if callback is not None:
            callback(self.layer_idx, boundary_name, value)

    def __call__(
        self,
        hidden_states,
        rope_mats,
        position_idx,
        page_table,
        kv_cache,
        is_decode,
        token_index=None,
        per_layer_input=None,
        shared_kv=None,
        keep_kv=False,
        is_kv_shared=False,
        position_idx_cache=None,
        batch_size=1,
        user_id=0,
        valid_seq_len=None,
        sequential_kv_write=False,
        rope_presliced=False,
        packed=None,
        chunk_start_idx=None,
        chunk_page_table=None,
    ):
        """
        Decoder layer forward pass.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            rope_mats: precomputed RoPE matrices
            position_idx: current position index
            page_table: paged attention page table
            kv_cache: KV cache for this layer
            is_decode: True for decode mode
            shared_kv: optional (tt_k, tt_v) from source layer for KV sharing (prefill only)
            keep_kv: if True, keep K/V alive for sharing with later layers (prefill only)
            is_kv_shared: if True, this layer shares KV from source (skip K/V proj + cache update)

        Returns:
            hidden_states: [1, 1, seq_len, hidden_size] on device
        """
        self._capture_boundary("layer_input", hidden_states)

        # 1. Attention block: norm -> attn -> post_attn_norm -> residual add
        residual = hidden_states
        normed = self.input_layernorm.forward(hidden_states)
        if not is_decode and batch_size > 1:
            attn_in = ttnn.reshape(normed, [batch_size, 1, normed.shape[-2] // batch_size, -1])
        else:
            attn_in = normed
        attn_output = self.self_attn(
            attn_in,
            rope_mats=rope_mats,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            token_index=token_index,
            shared_kv=shared_kv,
            keep_kv=keep_kv,
            is_kv_shared=is_kv_shared,
            position_idx_cache=position_idx_cache,
            batch_size=batch_size,
            user_id=user_id,
            valid_seq_len=valid_seq_len,
            sequential_kv_write=sequential_kv_write,
            rope_presliced=rope_presliced,
            packed=packed,
            chunk_start_idx=chunk_start_idx,
            chunk_page_table=chunk_page_table,
        )

        if isinstance(attn_output, torch.Tensor):
            hidden_states = residual
        else:
            attn_output = self.post_attention_layernorm.forward(attn_output)
            if not is_decode and batch_size > 1:
                residual = ttnn.reshape(
                    residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1]
                )
            hidden_states = ttnn.add(residual, attn_output)
            residual.deallocate(True)
            attn_output.deallocate(True)

        self._capture_boundary("post_attention_residual", hidden_states)

        # 2. MLP + MoE block
        residual = hidden_states
        normed = self.pre_feedforward_layernorm.forward(hidden_states)
        self._capture_mlp_boundary("pre_mlp_norm", normed)
        mlp_output = self.shared_mlp(normed)
        normed.deallocate(True)

        if self.enable_moe_block:
            # post_feedforward_layernorm_1 on MLP output
            mlp_normed = self.post_feedforward_layernorm_1.forward(mlp_output)
            mlp_output.deallocate(True)

            # Router input = pre-MLP residual, expert input = normed residual
            # All on device — no CPU round-trip
            residual_for_router = residual
            expert_input = self.pre_feedforward_layernorm_2.forward(residual_for_router)

            # MoE: router(residual) → dense_routing → experts(normed_input, routing)
            expert_output = self.moe(residual_for_router, expert_input)
            expert_input.deallocate(True)

            # post_feedforward_layernorm_2 on expert output
            expert_normed = self.post_feedforward_layernorm_2.forward(expert_output)
            expert_output.deallocate(True)

            # Combine: mlp_normed + expert_normed
            hidden_states = ttnn.add(mlp_normed, expert_normed)
            mlp_normed.deallocate(True)
            expert_normed.deallocate(True)
        else:
            hidden_states = mlp_output

        # post_feedforward_layernorm -> residual add
        hidden_states = self.post_feedforward_layernorm.forward(hidden_states)
        self._capture_mlp_boundary("post_feedforward_norm", hidden_states)
        combined = ttnn.add(residual, hidden_states)
        residual.deallocate(True)
        hidden_states.deallocate(True)

        hidden_states = combined
        self._capture_mlp_boundary("post_mlp_residual", hidden_states)
        self._capture_boundary("post_mlp_residual", hidden_states)

        # Per-layer input embeddings (E2B/E4B) — BEFORE layer_scalar (matching HF order)
        if self.hidden_size_per_layer_input and per_layer_input is not None and hasattr(self, "per_layer_input_gate"):
            residual_pli = hidden_states
            gated = ttnn.linear(hidden_states, self.per_layer_input_gate)
            gated = ttnn.gelu(gated, fast_and_approximate_mode=True)
            gated = ttnn.mul(gated, per_layer_input)
            projection_kwargs = {}
            projection_compute_kernel_config = _pli_projection_compute_kernel_config(gated, self.layer_idx)
            if projection_compute_kernel_config is not None:
                projection_kwargs["compute_kernel_config"] = projection_compute_kernel_config
            projected = ttnn.linear(gated, self.per_layer_projection, **projection_kwargs)
            normed_pli = self.post_per_layer_input_norm.forward(projected)
            hidden_states = ttnn.add(residual_pli, normed_pli)
            if len(hidden_states.shape) > 4:
                hidden_states = ttnn.reshape(hidden_states, (1, 1, hidden_states.shape[-2], self.hidden_size))

        self._capture_boundary("post_pli_residual", hidden_states)
        self._capture_boundary("pre_layer_scalar", hidden_states)

        # Layer scalar — AFTER PLI (matching HF order)
        if self.layer_scalar != 1.0:
            hidden_states = ttnn.mul(hidden_states, self.layer_scalar)

        self._capture_boundary("layer_output", hidden_states)

        return hidden_states
