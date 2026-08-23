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

FUSED_GATE_GELU_MUL_ENV = "GEMMA4_FUSED_SHARED_MLP_GATE_GELU_MUL"
DECODE_GATE_UP_IN0_BLOCK_W_ENV = "GEMMA4_DECODE_SHARED_MLP_GATE_UP_IN0_BLOCK_W"
PREFILL_GATE_UP_IN0_BLOCK_W_ENV = "GEMMA4_PREFILL_SHARED_MLP_GATE_UP_IN0_BLOCK_W"
DECODE_DOWN_IN0_BLOCK_W_ENV = "GEMMA4_DECODE_SHARED_MLP_DOWN_IN0_BLOCK_W"
PREFILL_DOWN_IN0_BLOCK_W_ENV = "GEMMA4_PREFILL_SHARED_MLP_DOWN_IN0_BLOCK_W"
MAX_EXPERIMENTAL_IN0_BLOCK_W = 32
MAX_PREFILL_IN0_BLOCK_W = 4
MAX_PREFILL_DOWN_IN0_BLOCK_W = 20
PREFILL_PROGRAM_HIDDEN_SIZE = 2560
PREFILL_PROGRAM_INTERMEDIATE_SIZE = 10240
PREFILL_PROGRAM_SEQUENCE_LENGTH = 1024


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


def _decode_gate_up_in0_block_widths(hidden_size):
    if hidden_size % ttnn.TILE_SIZE != 0:
        raise ValueError(f"hidden_size must be tile-aligned, got {hidden_size}")
    k_tiles = hidden_size // ttnn.TILE_SIZE
    return tuple(width for width in range(1, min(k_tiles, MAX_EXPERIMENTAL_IN0_BLOCK_W) + 1) if k_tiles % width == 0)


def _resolve_decode_gate_up_in0_block_w(hidden_size, value=None):
    if value is None:
        value = os.environ.get(DECODE_GATE_UP_IN0_BLOCK_W_ENV, "0")
    if isinstance(value, bool):
        raise ValueError(f"{DECODE_GATE_UP_IN0_BLOCK_W_ENV} requires an explicit integer width, got {value!r}")
    try:
        width = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"{DECODE_GATE_UP_IN0_BLOCK_W_ENV} requires an integer width, got {value!r}") from error
    if width == 0:
        return None
    widths = _decode_gate_up_in0_block_widths(hidden_size)
    if width not in widths:
        raise ValueError(
            f"{DECODE_GATE_UP_IN0_BLOCK_W_ENV} must be 0 or one of {widths} for hidden_size={hidden_size}, got {width}"
        )
    return width


def _prefill_gate_up_in0_block_widths(hidden_size):
    """Return shape-legal widths that fit the frozen Wormhole prefill L1 budget."""
    if hidden_size != PREFILL_PROGRAM_HIDDEN_SIZE:
        raise ValueError(
            f"experimental prefill gate/up program requires hidden_size={PREFILL_PROGRAM_HIDDEN_SIZE}, "
            f"got {hidden_size}"
        )
    # For the frozen M=32/N=5 geometry, width 4 consumes 1,261,568 of
    # 1,391,936 CB-usable bytes; the next divisor, width 5, needs 1,413,120.
    return tuple(width for width in _decode_gate_up_in0_block_widths(hidden_size) if width <= MAX_PREFILL_IN0_BLOCK_W)


def _resolve_prefill_gate_up_in0_block_w(hidden_size, value=None):
    if value is None:
        value = os.environ.get(PREFILL_GATE_UP_IN0_BLOCK_W_ENV, "0")
    if isinstance(value, bool):
        raise ValueError(f"{PREFILL_GATE_UP_IN0_BLOCK_W_ENV} requires an explicit integer width, got {value!r}")
    try:
        width = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"{PREFILL_GATE_UP_IN0_BLOCK_W_ENV} requires an integer width, got {value!r}") from error
    if width == 0:
        return None
    widths = _prefill_gate_up_in0_block_widths(hidden_size)
    if width not in widths:
        raise ValueError(
            f"{PREFILL_GATE_UP_IN0_BLOCK_W_ENV} must be 0 or one of {widths} for hidden_size={hidden_size}, got {width}"
        )
    return width


def _down_in0_block_widths(intermediate_size):
    if intermediate_size % ttnn.TILE_SIZE != 0:
        raise ValueError(f"intermediate_size must be tile-aligned, got {intermediate_size}")
    k_tiles = intermediate_size // ttnn.TILE_SIZE
    return tuple(width for width in range(1, min(k_tiles, MAX_EXPERIMENTAL_IN0_BLOCK_W) + 1) if k_tiles % width == 0)


def _resolve_decode_down_in0_block_w(intermediate_size, value=None):
    if value is None:
        value = os.environ.get(DECODE_DOWN_IN0_BLOCK_W_ENV, "0")
    if isinstance(value, bool):
        raise ValueError(f"{DECODE_DOWN_IN0_BLOCK_W_ENV} requires an explicit integer width, got {value!r}")
    try:
        width = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"{DECODE_DOWN_IN0_BLOCK_W_ENV} requires an integer width, got {value!r}") from error
    if width == 0:
        return None
    widths = _down_in0_block_widths(intermediate_size)
    if width not in widths:
        raise ValueError(
            f"{DECODE_DOWN_IN0_BLOCK_W_ENV} must be 0 or one of {widths} "
            f"for intermediate_size={intermediate_size}, got {width}"
        )
    return width


def _prefill_down_in0_block_widths(intermediate_size):
    """Return shape-legal widths that fit the frozen Wormhole prefill L1 budget."""
    if intermediate_size != PREFILL_PROGRAM_INTERMEDIATE_SIZE:
        raise ValueError(
            "experimental prefill down program requires "
            f"intermediate_size={PREFILL_PROGRAM_INTERMEDIATE_SIZE}, got {intermediate_size}"
        )
    # For the frozen M=4/N=10 geometry, width 20 consumes 1,310,720 of
    # 1,391,936 CB-usable bytes; the next divisor, width 32, needs 1,998,848.
    return tuple(width for width in _down_in0_block_widths(intermediate_size) if width <= MAX_PREFILL_DOWN_IN0_BLOCK_W)


def _resolve_prefill_down_in0_block_w(intermediate_size, value=None):
    if value is None:
        value = os.environ.get(PREFILL_DOWN_IN0_BLOCK_W_ENV, "0")
    if isinstance(value, bool):
        raise ValueError(f"{PREFILL_DOWN_IN0_BLOCK_W_ENV} requires an explicit integer width, got {value!r}")
    try:
        width = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"{PREFILL_DOWN_IN0_BLOCK_W_ENV} requires an integer width, got {value!r}") from error
    if width == 0:
        return None
    widths = _prefill_down_in0_block_widths(intermediate_size)
    if width not in widths:
        raise ValueError(
            f"{PREFILL_DOWN_IN0_BLOCK_W_ENV} must be 0 or one of {widths} "
            f"for intermediate_size={intermediate_size}, got {width}"
        )
    return width


def _decode_gate_up_program_config(mesh_device, hidden_size, intermediate_size, in0_block_w):
    """Build an explicit TP1 decode program for a validated K-block width."""
    grid = mesh_device.compute_with_storage_grid_size()
    n_tiles = intermediate_size // ttnn.TILE_SIZE
    core_count = grid.x * grid.y
    per_core_n = (n_tiles + core_count - 1) // core_count
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


def _prefill_gate_up_program_config(mesh_device, hidden_size, intermediate_size, in0_block_w):
    """Build the accepted automatic geometry with an explicit prefill K block."""
    grid = mesh_device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (8, 9):
        raise ValueError(f"experimental prefill gate/up program requires an 8x9 grid, got {grid.x}x{grid.y}")
    if (hidden_size, intermediate_size) != (PREFILL_PROGRAM_HIDDEN_SIZE, PREFILL_PROGRAM_INTERMEDIATE_SIZE):
        raise ValueError(
            "experimental prefill gate/up program requires "
            f"hidden_size={PREFILL_PROGRAM_HIDDEN_SIZE} and intermediate_size={PREFILL_PROGRAM_INTERMEDIATE_SIZE}, "
            f"got hidden_size={hidden_size} and intermediate_size={intermediate_size}"
        )
    n_tiles = intermediate_size // ttnn.TILE_SIZE
    core_count = grid.x * grid.y
    per_core_n = (n_tiles + core_count - 1) // core_count
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=8,
        out_subblock_w=1,
        per_core_M=PREFILL_PROGRAM_SEQUENCE_LENGTH // ttnn.TILE_SIZE,
        per_core_N=per_core_n,
        fused_activation=None,
        fuse_batch=False,
        mcast_in0=True,
    )


def _decode_down_program_config(mesh_device, intermediate_size, hidden_size, in0_block_w):
    """Build an explicit TP1 decode down-projection program."""
    grid = mesh_device.compute_with_storage_grid_size()
    n_tiles = hidden_size // ttnn.TILE_SIZE
    core_count = grid.x * grid.y
    per_core_n = (n_tiles + core_count - 1) // core_count
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


def _prefill_down_program_config(mesh_device, intermediate_size, hidden_size, in0_block_w):
    """Build the accepted automatic prefill down geometry with an explicit K block."""
    grid = mesh_device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (8, 9):
        raise ValueError(f"experimental prefill down program requires an 8x9 grid, got {grid.x}x{grid.y}")
    if (hidden_size, intermediate_size) != (PREFILL_PROGRAM_HIDDEN_SIZE, PREFILL_PROGRAM_INTERMEDIATE_SIZE):
        raise ValueError(
            "experimental prefill down program requires "
            f"hidden_size={PREFILL_PROGRAM_HIDDEN_SIZE} and intermediate_size={PREFILL_PROGRAM_INTERMEDIATE_SIZE}, "
            f"got hidden_size={hidden_size} and intermediate_size={intermediate_size}"
        )
    m_tiles = PREFILL_PROGRAM_SEQUENCE_LENGTH // ttnn.TILE_SIZE
    n_tiles = hidden_size // ttnn.TILE_SIZE
    per_core_m = (m_tiles + grid.y - 1) // grid.y
    per_core_n = (n_tiles + grid.x - 1) // grid.x
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=4,
        out_subblock_w=2,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
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
        fuse_gate_gelu_mul=None,
        decode_gate_up_in0_block_w=None,
        prefill_gate_up_in0_block_w=None,
        decode_down_in0_block_w=None,
        prefill_down_in0_block_w=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""
        # Column-parallel (gate/up output) and row-parallel (down input)
        # shard the intermediate dimension evenly across TP ranks, so the
        # per-rank intermediate width is intermediate_size // tp. The decode
        # program configs below must be sized to the per-rank shard, not the
        # full-width TP1 value.
        per_rank_intermediate = self.intermediate_size // tp if tp > 1 else self.intermediate_size
        self.fuse_gate_gelu_mul = _resolve_bool_env(FUSED_GATE_GELU_MUL_ENV, fuse_gate_gelu_mul)
        self.decode_gate_up_in0_block_w = _resolve_decode_gate_up_in0_block_w(
            self.hidden_size, decode_gate_up_in0_block_w
        )
        self.prefill_gate_up_in0_block_w = _resolve_prefill_gate_up_in0_block_w(
            self.hidden_size, prefill_gate_up_in0_block_w
        )
        self.decode_down_in0_block_w = _resolve_decode_down_in0_block_w(self.intermediate_size, decode_down_in0_block_w)
        self.prefill_down_in0_block_w = _resolve_prefill_down_in0_block_w(
            self.intermediate_size, prefill_down_in0_block_w
        )
        if (
            self.fuse_gate_gelu_mul
            or self.decode_gate_up_in0_block_w is not None
            or self.prefill_gate_up_in0_block_w is not None
            or self.decode_down_in0_block_w is not None
            or self.prefill_down_in0_block_w is not None
        ) and tp > 2:
            # TP1/TP2 are re-qualified for the decode selectors; TP4+ keeps
            # the guard (wider shards shrink per-rank N to widths the tuned
            # 1D-mcast configs have not been validated for). The prefill
            # program functions still hard-code the full TP1 geometry, so
            # prefill widths remain TP1-only and raise there if set at TP2.
            raise ValueError("experimental shared-MLP selectors are currently qualified only for TP1/TP2")
        self.decode_gate_up_program_config = (
            _decode_gate_up_program_config(
                mesh_device,
                self.hidden_size,
                per_rank_intermediate,
                self.decode_gate_up_in0_block_w,
            )
            if self.decode_gate_up_in0_block_w is not None
            else None
        )
        self.prefill_gate_up_program_config = (
            _prefill_gate_up_program_config(
                mesh_device,
                self.hidden_size,
                self.intermediate_size,
                self.prefill_gate_up_in0_block_w,
            )
            if self.prefill_gate_up_in0_block_w is not None
            else None
        )
        self.decode_down_program_config = (
            _decode_down_program_config(
                mesh_device,
                per_rank_intermediate,
                self.hidden_size,
                self.decode_down_in0_block_w,
            )
            if self.decode_down_in0_block_w is not None
            else None
        )
        self.prefill_down_program_config = (
            _prefill_down_program_config(
                mesh_device,
                self.intermediate_size,
                self.hidden_size,
                self.prefill_down_in0_block_w,
            )
            if self.prefill_down_in0_block_w is not None
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

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        # gate = GELU(x @ gate_proj)
        is_decode = hidden_states.shape[-2] == ttnn.TILE_SIZE
        is_target_prefill = hidden_states.shape[-2] == PREFILL_PROGRAM_SEQUENCE_LENGTH
        if is_decode:
            gate_up_program_config = self.decode_gate_up_program_config
            down_program_config = self.decode_down_program_config
        elif is_target_prefill:
            gate_up_program_config = self.prefill_gate_up_program_config
            down_program_config = self.prefill_down_program_config
        else:
            gate_up_program_config = None
            down_program_config = None

        if gate_up_program_config is None:
            gate = ttnn.linear(hidden_states, self.gate_proj)
        else:
            gate = ttnn.linear(
                hidden_states,
                self.gate_proj,
                program_config=gate_up_program_config,
            )

        # up = x @ up_proj
        if gate_up_program_config is None:
            up = ttnn.linear(hidden_states, self.up_proj)
        else:
            up = ttnn.linear(
                hidden_states,
                self.up_proj,
                program_config=gate_up_program_config,
            )

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
            hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        # output = hidden @ down_proj
        if down_program_config is None:
            output = ttnn.linear(hidden, self.down_proj)
        else:
            output = ttnn.linear(hidden, self.down_proj, program_config=down_program_config)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
