# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure host helpers for Gemma4 per-layer-input (PLI) paths."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

PLI_PREFILL_TRACE_ENV = "GEMMA4_PLI_PREFILL_TRACE"


def resolve_pli_prefill_trace_enabled(value=None) -> bool:
    """Resolve the opt-in PLI prefill-trace selector without importing TTNN."""
    if value is None:
        value = os.environ.get(PLI_PREFILL_TRACE_ENV, "0")
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("0", "false", "no", "off"):
        return False
    if normalized in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{PLI_PREFILL_TRACE_ENV} must be a boolean value, got {value!r}")


def pack_prefill_per_layer_inputs(per_layer_inputs: Sequence[torch.Tensor], expected_layers: int) -> torch.Tensor:
    """Pack ``[batch, seq, pli]`` layer tensors as ``[1, layer, batch*seq, pli]``.

    The packed layout lets a captured prefill trace bind one persistent device
    buffer and select one layer with ``packed[:, layer:layer+1, :, :]``.  The
    flattened token order matches the existing eager path's per-layer reshape.
    """
    if isinstance(expected_layers, bool) or not isinstance(expected_layers, int) or expected_layers < 1:
        raise ValueError(f"expected_layers must be a positive integer, got {expected_layers!r}")
    if len(per_layer_inputs) != expected_layers:
        raise ValueError(f"expected {expected_layers} PLI tensors, got {len(per_layer_inputs)}")

    reference = per_layer_inputs[0]
    if not isinstance(reference, torch.Tensor) or reference.dim() != 3:
        raise ValueError("PLI tensor 0 must have shape [batch, seq, pli]")
    expected_shape = tuple(reference.shape)
    expected_dtype = reference.dtype
    flattened = []
    for index, tensor in enumerate(per_layer_inputs):
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
            raise ValueError(f"PLI tensor {index} must have shape [batch, seq, pli]")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"PLI tensor {index} shape {tuple(tensor.shape)} does not match {expected_shape}")
        if tensor.dtype != expected_dtype:
            raise ValueError(f"PLI tensor {index} dtype {tensor.dtype} does not match {expected_dtype}")
        flattened.append(tensor.reshape(-1, tensor.shape[-1]))
    return torch.stack(flattened, dim=0).unsqueeze(0).contiguous()
