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

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.utils.general_utils import get_cache_file_name

DOWN_PROJ_COMPUTE_PROFILE_ENV = "GEMMA4_SHARED_MLP_DOWN_PROJ_COMPUTE_PROFILE"
DOWN_PROJ_PROGRAM_PROFILE_ENV = "GEMMA4_SHARED_MLP_DOWN_PROJ_PROGRAM_PROFILE"
DOWN_PROJ_LAYER38_HIFI2_FP32_ACC_PROFILE = "layer38_hifi2_fp32_acc"
DOWN_PROJ_LAYER38_HIFI2_FP32_L1_ACC_PROFILE = "layer38_hifi2_fp32_l1_acc"
DOWN_PROJ_LAYER38_HIFI3_FP32_ACC_PROFILE = "layer38_hifi3_fp32_acc"
DOWN_PROJ_LAYER38_HIFI3_FP32_L1_ACC_PROFILE = "layer38_hifi3_fp32_l1_acc"
DOWN_PROJ_LAYER38_HIFI4_BF16_L1_ACC_PROFILE = "layer38_hifi4_bf16_l1_acc"
DOWN_PROJ_LAYER38_HIFI3_BF16_L1_ACC_PROFILE = "layer38_hifi3_bf16_l1_acc"
DOWN_PROJ_LAYER38_LOFI_FP32_ACC_PROFILE = "layer38_lofi_fp32_acc"
DOWN_PROJ_LAYER38_LOFI_BF16_L1_ACC_PROFILE = "layer38_lofi_bf16_l1_acc"
DOWN_PROJ_COMPUTE_PROFILE_SPECS = {
    DOWN_PROJ_LAYER38_HIFI2_FP32_ACC_PROFILE: (ttnn.MathFidelity.HiFi2, True, False),
    DOWN_PROJ_LAYER38_HIFI2_FP32_L1_ACC_PROFILE: (ttnn.MathFidelity.HiFi2, True, True),
    DOWN_PROJ_LAYER38_HIFI3_FP32_ACC_PROFILE: (ttnn.MathFidelity.HiFi3, True, False),
    DOWN_PROJ_LAYER38_HIFI3_FP32_L1_ACC_PROFILE: (ttnn.MathFidelity.HiFi3, True, True),
    DOWN_PROJ_LAYER38_HIFI4_BF16_L1_ACC_PROFILE: (ttnn.MathFidelity.HiFi4, False, True),
    DOWN_PROJ_LAYER38_HIFI3_BF16_L1_ACC_PROFILE: (ttnn.MathFidelity.HiFi3, False, True),
    DOWN_PROJ_LAYER38_LOFI_FP32_ACC_PROFILE: (ttnn.MathFidelity.LoFi, True, False),
    DOWN_PROJ_LAYER38_LOFI_BF16_L1_ACC_PROFILE: (ttnn.MathFidelity.LoFi, False, True),
}
DOWN_PROJ_COMPUTE_PROFILES = frozenset(DOWN_PROJ_COMPUTE_PROFILE_SPECS)
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W1_PROFILE = "layer38_decode_mcast1d_w1"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W4_PROFILE = "layer38_decode_mcast1d_w4"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W5_PROFILE = "layer38_decode_mcast1d_w5"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W8_PROFILE = "layer38_decode_mcast1d_w8"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W10_PROFILE = "layer38_decode_mcast1d_w10"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W16_PROFILE = "layer38_decode_mcast1d_w16"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W20_PROFILE = "layer38_decode_mcast1d_w20"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W32_PROFILE = "layer38_decode_mcast1d_w32"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W40_PROFILE = "layer38_decode_mcast1d_w40"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W64_PROFILE = "layer38_decode_mcast1d_w64"
DOWN_PROJ_LAYER38_DECODE_MCAST1D_W80_PROFILE = "layer38_decode_mcast1d_w80"
DOWN_PROJ_PROGRAM_PROFILE_WIDTHS = {
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W1_PROFILE: 1,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W4_PROFILE: 4,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W5_PROFILE: 5,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W8_PROFILE: 8,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W10_PROFILE: 10,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W16_PROFILE: 16,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W20_PROFILE: 20,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W32_PROFILE: 32,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W40_PROFILE: 40,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W64_PROFILE: 64,
    DOWN_PROJ_LAYER38_DECODE_MCAST1D_W80_PROFILE: 80,
}
DOWN_PROJ_PROGRAM_PROFILES = frozenset(DOWN_PROJ_PROGRAM_PROFILE_WIDTHS)
DOWN_PROJ_LAYER38_DECODE_INPUT_SHAPE = (1, 1, 1, 10240)


def _down_proj_compute_kernel_config(hidden_states, layer_idx):
    """Return the opt-in layer-38 down-projection profile, if selected."""
    profile = os.environ.get(DOWN_PROJ_COMPUTE_PROFILE_ENV) or None
    if profile is None:
        return None
    if profile not in DOWN_PROJ_COMPUTE_PROFILES:
        supported = ", ".join(sorted(DOWN_PROJ_COMPUTE_PROFILES))
        raise ValueError(f"{DOWN_PROJ_COMPUTE_PROFILE_ENV} must be one of: {supported}")
    if layer_idx != 38:
        return None
    device = hidden_states.device()
    if device is None:
        raise ValueError(f"{DOWN_PROJ_COMPUTE_PROFILE_ENV}={profile} requires a " "device-resident activation")
    math_fidelity, fp32_dest_acc_en, packer_l1_acc = DOWN_PROJ_COMPUTE_PROFILE_SPECS[profile]
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


def _down_proj_program_config(hidden_states, layer_idx):
    """Return the exact-shape opt-in layer-38 decode program, if selected."""
    profile = os.environ.get(DOWN_PROJ_PROGRAM_PROFILE_ENV) or None
    if profile is None:
        return None
    if profile not in DOWN_PROJ_PROGRAM_PROFILES:
        supported = ", ".join(sorted(DOWN_PROJ_PROGRAM_PROFILES))
        raise ValueError(f"{DOWN_PROJ_PROGRAM_PROFILE_ENV} must be one of: {supported}")
    if layer_idx != 38 or tuple(hidden_states.shape) != DOWN_PROJ_LAYER38_DECODE_INPUT_SHAPE:
        return None
    device = hidden_states.device()
    if device is None:
        raise ValueError(f"{DOWN_PROJ_PROGRAM_PROFILE_ENV}={profile} requires a device-resident activation")
    grid = device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (8, 9):
        raise ValueError(f"{DOWN_PROJ_PROGRAM_PROFILE_ENV}={profile} requires an 8x9 compute grid")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=DOWN_PROJ_PROGRAM_PROFILE_WIDTHS[profile],
        out_subblock_h=1,
        out_subblock_w=2,
        out_block_h=1,
        out_block_w=2,
        per_core_M=1,
        per_core_N=2,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


FUSED_GATE_GELU_MUL_ENV = "GEMMA4_FUSED_SHARED_MLP_GATE_GELU_MUL"
DECODE_GATE_UP_PROGRAM_ENV = "GEMMA4_DECODE_SHARED_MLP_GATE_UP_PROGRAM"


def _resolve_bool_env(name, value=None):
    if value is None:
        value = os.environ.get(name, "0")
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("0", "false", "no", "off"):
        return False
    if normalized in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _decode_gate_up_program_config(mesh_device, hidden_size, intermediate_size):
    """Use the widest K block that divides the TP1 Chotto decode shape."""
    grid = mesh_device.compute_with_storage_grid_size()
    k_tiles = hidden_size // ttnn.TILE_SIZE
    n_tiles = intermediate_size // ttnn.TILE_SIZE
    core_count = grid.x * grid.y
    per_core_n = (n_tiles + core_count - 1) // core_count
    in0_block_w = max(width for width in range(1, 33) if k_tiles % width == 0)
    out_subblock_w = max(width for width in range(1, min(per_core_n, 8) + 1) if per_core_n % width == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _apply_down_projection(hidden_states, weight, layer_idx):
    kwargs = {}
    compute_kernel_config = _down_proj_compute_kernel_config(hidden_states, layer_idx)
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config
    program_config = _down_proj_program_config(hidden_states, layer_idx)
    if program_config is not None:
        kwargs["program_config"] = program_config
    return ttnn.linear(hidden_states, weight, **kwargs)


class SharedMLP:
    BOUNDARY_CAPTURE_NAMES = (
        "gate_projection",
        "gate_gelu",
        "up_projection",
        "gated_product",
        "down_projection",
    )

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        layer_idx=None,
        fuse_gate_gelu_mul=None,
        decode_gate_up_program=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.layer_idx = layer_idx
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""
        self.fuse_gate_gelu_mul = _resolve_bool_env(FUSED_GATE_GELU_MUL_ENV, fuse_gate_gelu_mul)
        self.decode_gate_up_program = _resolve_bool_env(DECODE_GATE_UP_PROGRAM_ENV, decode_gate_up_program)
        if (self.fuse_gate_gelu_mul or self.decode_gate_up_program) and tp != 1:
            raise ValueError("experimental shared-MLP selectors are currently qualified only for TP1")
        self.decode_gate_up_program_config = (
            _decode_gate_up_program_config(mesh_device, self.hidden_size, self.intermediate_size)
            if self.decode_gate_up_program
            else None
        )

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

        if state_dict:
            gate_proj_weight = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            up_proj_weight = state_dict["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            gate_proj_weight = None
            up_proj_weight = None
            down_proj_weight = None

        # gate/up: column-parallel (shard output dim across TP devices)
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

    def _capture_boundary(self, boundary_name, value):
        """Synchronously expose an opt-in dense-MLP diagnostic boundary."""
        callback = getattr(self, "_boundary_capture_callback", None)
        if callback is not None:
            callback(boundary_name, value)

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        # gate = GELU(x @ gate_proj)
        is_decode = hidden_states.shape[-2] == ttnn.TILE_SIZE
        gate_up_program_config = self.decode_gate_up_program_config if is_decode else None

        if gate_up_program_config is None:
            gate = ttnn.linear(hidden_states, self.gate_proj)
        else:
            gate = ttnn.linear(
                hidden_states,
                self.gate_proj,
                program_config=gate_up_program_config,
            )
        self._capture_boundary("gate_projection", gate)

        # up = x @ up_proj
        if gate_up_program_config is None:
            up = ttnn.linear(hidden_states, self.up_proj)
        else:
            up = ttnn.linear(
                hidden_states,
                self.up_proj,
                program_config=gate_up_program_config,
            )
        self._capture_boundary("up_projection", up)

        if self.fuse_gate_gelu_mul:
            hidden = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)],
            )
        else:
            # Gemma 4's GeGLU is numerically sensitive across layers and decode
            # steps; FastLut drift can change greedy token selection.
            gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
            self._capture_boundary("gate_gelu", gate)
            hidden = ttnn.mul(gate, up)
        self._capture_boundary("gated_product", hidden)
        gate.deallocate(True)
        up.deallocate(True)

        # output = hidden @ down_proj
        output = _apply_down_projection(hidden, self.down_proj, self.layer_idx)
        self._capture_boundary("down_projection", output)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
