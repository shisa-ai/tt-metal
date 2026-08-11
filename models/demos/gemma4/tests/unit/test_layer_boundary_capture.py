# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""No-device tests for opt-in Gemma decoder boundary capture."""

import inspect

from models.demos.gemma4.tt.layer import Gemma4DecoderLayer


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
