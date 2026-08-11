# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""No-device tests for opt-in Gemma decoder boundary capture."""

import inspect

from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.shared_mlp import SharedMLP


def test_decoder_boundary_capture_is_inert_until_callback_is_installed():
    layer = object.__new__(Gemma4DecoderLayer)
    layer.layer_idx = 38
    marker = object()

    layer._capture_boundary("layer_input", marker)

    captured = []
    layer._boundary_capture_callback = lambda *values: captured.append(values)
    layer._capture_boundary("post_attention_residual", marker)

    assert captured == [(38, "post_attention_residual", marker)]


def test_decoder_call_exposes_boundaries_in_execution_order():
    source = inspect.getsource(Gemma4DecoderLayer.__call__)
    offsets = [source.index(f'self._capture_boundary("{name}"') for name in Gemma4DecoderLayer.BOUNDARY_CAPTURE_NAMES]

    assert offsets == sorted(offsets)


def test_decoder_mlp_boundary_capture_is_inert_until_callback_is_installed():
    layer = object.__new__(Gemma4DecoderLayer)
    layer.layer_idx = 38
    marker = object()

    layer._capture_mlp_boundary("pre_mlp_norm", marker)

    captured = []
    layer._mlp_boundary_capture_callback = lambda *values: captured.append(values)
    layer._capture_mlp_boundary("post_feedforward_norm", marker)

    assert captured == [(38, "post_feedforward_norm", marker)]


def test_shared_mlp_boundary_capture_is_inert_until_callback_is_installed():
    mlp = object.__new__(SharedMLP)
    marker = object()

    mlp._capture_boundary("gate_projection", marker)

    captured = []
    mlp._boundary_capture_callback = lambda *values: captured.append(values)
    mlp._capture_boundary("gated_product", marker)

    assert captured == [("gated_product", marker)]


def test_mlp_call_exposes_boundaries_in_execution_order():
    layer_source = inspect.getsource(Gemma4DecoderLayer.__call__)
    shared_source = inspect.getsource(SharedMLP.__call__)
    source = (
        layer_source[
            layer_source.index('self._capture_mlp_boundary("pre_mlp_norm"') : layer_source.index(
                'self._capture_mlp_boundary("post_feedforward_norm"'
            )
        ]
        + shared_source
        + layer_source[layer_source.index('self._capture_mlp_boundary("post_feedforward_norm"') :]
    )
    offsets = []
    for name in Gemma4DecoderLayer.MLP_BOUNDARY_CAPTURE_NAMES:
        method = "self._capture_boundary" if name in SharedMLP.BOUNDARY_CAPTURE_NAMES else "self._capture_mlp_boundary"
        offsets.append(source.index(f'{method}("{name}"'))

    assert offsets == sorted(offsets)
