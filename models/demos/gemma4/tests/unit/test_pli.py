# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
from pathlib import Path

import pytest
import torch

PLI = Path(__file__).resolve().parents[2] / "tt/pli.py"
spec = importlib.util.spec_from_file_location("gemma4_pli", PLI)
pli = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pli)


@pytest.mark.parametrize("value", [True, "1", "true", "YES", "on"])
def test_pli_prefill_trace_selector_accepts_true_values(value):
    assert pli.resolve_pli_prefill_trace_enabled(value) is True


@pytest.mark.parametrize("value", [False, "0", "false", "NO", "off"])
def test_pli_prefill_trace_selector_accepts_false_values(value):
    assert pli.resolve_pli_prefill_trace_enabled(value) is False


def test_pli_prefill_trace_selector_rejects_unknown_value():
    with pytest.raises(ValueError, match=pli.PLI_PREFILL_TRACE_ENV):  # allow-pytest.raises: TTNN-free conftest
        pli.resolve_pli_prefill_trace_enabled("sometimes")


def test_pli_prefill_trace_selector_reads_environment(monkeypatch):
    monkeypatch.setenv(pli.PLI_PREFILL_TRACE_ENV, "1")
    assert pli.resolve_pli_prefill_trace_enabled() is True


def test_pli_prefill_trace_selector_defaults_off(monkeypatch):
    monkeypatch.delenv(pli.PLI_PREFILL_TRACE_ENV, raising=False)
    assert pli.resolve_pli_prefill_trace_enabled() is False


def test_pack_prefill_pli_preserves_single_user_layer_values():
    layers = [torch.full((1, 4, 2), fill_value=index, dtype=torch.bfloat16) for index in range(3)]
    packed = pli.pack_prefill_per_layer_inputs(layers, expected_layers=3)
    assert packed.shape == (1, 3, 4, 2)
    assert packed.dtype == torch.bfloat16
    for index, layer in enumerate(layers):
        torch.testing.assert_close(packed[:, index : index + 1], layer.unsqueeze(1))


def test_pack_prefill_pli_flattens_batch_in_existing_eager_order():
    layers = [torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2) + 100 * index for index in range(2)]
    packed = pli.pack_prefill_per_layer_inputs(layers, expected_layers=2)
    assert packed.shape == (1, 2, 6, 2)
    for index, layer in enumerate(layers):
        torch.testing.assert_close(packed[0, index], layer.reshape(6, 2))


def test_pack_prefill_pli_rejects_missing_layer():
    with pytest.raises(ValueError, match="expected 2 PLI tensors"):  # allow-pytest.raises: TTNN-free conftest
        pli.pack_prefill_per_layer_inputs([torch.zeros(1, 1, 1)], expected_layers=2)


@pytest.mark.parametrize("expected_layers", [True, 0, -1])
def test_pack_prefill_pli_rejects_invalid_layer_count(expected_layers):
    with pytest.raises(ValueError, match="positive integer"):  # allow-pytest.raises: TTNN-free conftest
        pli.pack_prefill_per_layer_inputs([], expected_layers=expected_layers)


@pytest.mark.parametrize(
    "layers, message",
    [
        ([torch.zeros(1, 1)], "shape [batch, seq, pli]"),
        ([torch.zeros(1, 2, 3), torch.zeros(1, 3, 3)], "does not match"),
        ([torch.zeros(1, 2, 3), torch.zeros(1, 2, 3, dtype=torch.bfloat16)], "dtype"),
    ],
)
def test_pack_prefill_pli_rejects_incompatible_tensors(layers, message):
    with pytest.raises(ValueError, match=re.escape(message)):  # allow-pytest.raises: TTNN-free conftest
        pli.pack_prefill_per_layer_inputs(layers, expected_layers=len(layers))
