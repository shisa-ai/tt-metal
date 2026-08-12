# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Shared/Dense MLP with GeGLU activation.

Each decoder layer has BOTH a shared MLP and routed MoE experts.
Architecture: down_proj(GELU(gate_proj(x)) * up_proj(x))
intermediate_size = 2112, no bias.

HF weight shapes:
  gate_proj.weight: [intermediate_size, hidden_size] = [2112, 2816]
  up_proj.weight:   [intermediate_size, hidden_size] = [2112, 2816]
  down_proj.weight: [hidden_size, intermediate_size] = [2816, 2112]
"""

import os

import torch

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.utils.general_utils import get_cache_file_name

FUSED_GATE_UP_ENV = "GEMMA4_FUSED_SHARED_MLP_GATE_UP"


def _resolve_fused_gate_up(value=None):
    """Resolve the bounded gate/up fusion selector without changing the default."""
    if value is None:
        value = os.environ.get(FUSED_GATE_UP_ENV, "0")
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("0", "false", "no", "off"):
        return False
    if normalized in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{FUSED_GATE_UP_ENV} must be a boolean value, got {value!r}")


def _prepare_up_gate_weight(state_dict):
    """Pack [up, gate] so an accurate split/GELU/multiply implements GeGLU."""
    if not state_dict:
        return None
    return (
        torch.cat((state_dict["up_proj.weight"], state_dict["gate_proj.weight"]), dim=-2)
        .transpose(-2, -1)
        .unsqueeze(0)
        .unsqueeze(0)
    )


class SharedMLP:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        fuse_gate_up=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""
        self.fuse_gate_up = _resolve_fused_gate_up(fuse_gate_up)
        if self.fuse_gate_up and tp != 1:
            raise ValueError(f"{FUSED_GATE_UP_ENV} is currently qualified only for TP1")

        # Tag the cache filenames with the weight dtype so that flipping a
        # SharedMLP weight's dtype (e.g. bf16 → bfp8 for DRAM-pressure relief)
        # doesn't collide with a previously-cached file that holds the same
        # logical weight at a different dtype. The rest of the model's cache
        # entries are unaffected and stay reusable across runs.
        _dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]
        dtype_suffix = f"_{_dtype_str}"

        if tp > 1:
            col_mapper = mesh_config.column_parallel(mesh_device)
            row_mapper = mesh_config.row_parallel(mesh_device)
        else:
            col_mapper = None
            row_mapper = None

        if self.fuse_gate_up:
            up_gate_proj_weight = _prepare_up_gate_weight(state_dict)
            gate_proj_weight = None
            up_proj_weight = None
        elif state_dict:
            up_gate_proj_weight = None
            gate_proj_weight = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            up_proj_weight = state_dict["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            up_gate_proj_weight = None
            gate_proj_weight = None
            up_proj_weight = None
        if state_dict:
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            down_proj_weight = None

        # gate/up: column-parallel (shard output dim across TP devices)
        if self.fuse_gate_up:
            self.up_gate_proj = ttnn.as_tensor(
                up_gate_proj_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(
                    tensor_cache_path, f"up_gate_proj.weight_fused{tp_suffix}{dtype_suffix}"
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.gate_proj = None
            self.up_proj = None
        else:
            self.up_gate_proj = None
            self.gate_proj = ttnn.as_tensor(
                gate_proj_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj.weight{tp_suffix}{dtype_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.up_proj = ttnn.as_tensor(
                up_proj_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj.weight{tp_suffix}{dtype_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        # down: row-parallel (shard input dim, allreduce after)
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        if self.fuse_gate_up:
            # A single projection removes one matmul program per layer. Keep
            # the [up, gate] order explicit: ttnn.geglu currently hard-codes
            # approximate GELU, while Gemma 4 requires the accurate path.
            up_gate = ttnn.linear(hidden_states, self.up_gate_proj)
            up, gate = ttnn.split(up_gate, self.intermediate_size, dim=-1)
            up_gate.deallocate(True)
            gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
        else:
            # gate = GELU(x @ gate_proj)
            gate = ttnn.linear(hidden_states, self.gate_proj)
            # Gemma 4's GeGLU is numerically sensitive across layers and decode
            # steps; FastLut drift can change greedy token selection.
            gate = ttnn.gelu(gate, fast_and_approximate_mode=False)

            # up = x @ up_proj
            up = ttnn.linear(hidden_states, self.up_proj)

        # hidden = gate * up
        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        # output = hidden @ down_proj
        output = ttnn.linear(hidden, self.down_proj)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
