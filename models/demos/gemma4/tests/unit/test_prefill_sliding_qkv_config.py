# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[2] / "tt/attention/prefill_qkv_config.py"
spec = importlib.util.spec_from_file_location("gemma4_prefill_qkv_config", CONFIG)
prefill_qkv_config = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prefill_qkv_config)


def test_prefill_sliding_qkv_widths_are_complete_l1_safe_surface():
    assert prefill_qkv_config.PREFILL_SLIDING_QKV_IN0_BLOCK_WIDTHS == (
        1,
        2,
        4,
        5,
        8,
        10,
        16,
    )


@pytest.mark.parametrize("value", [1, "2", " 16 "])
def test_prefill_sliding_qkv_selector_accepts_explicit_widths(value):
    assert prefill_qkv_config.resolve_prefill_sliding_qkv_in0_block_w(value) == int(value)


@pytest.mark.parametrize("value", [0, "0"])
def test_prefill_sliding_qkv_selector_zero_disables_program(value):
    assert prefill_qkv_config.resolve_prefill_sliding_qkv_in0_block_w(value) is None


@pytest.mark.parametrize("value", [True, False, "auto", -1, 3, 20, 32])
def test_prefill_sliding_qkv_selector_rejects_invalid_or_l1_unsafe_values(expect_error, value):
    with expect_error(ValueError, prefill_qkv_config.PREFILL_SLIDING_QKV_IN0_BLOCK_W_ENV):
        prefill_qkv_config.resolve_prefill_sliding_qkv_in0_block_w(value)


def test_prefill_sliding_qkv_selector_reads_environment(monkeypatch):
    monkeypatch.setenv(prefill_qkv_config.PREFILL_SLIDING_QKV_IN0_BLOCK_W_ENV, "10")
    assert prefill_qkv_config.resolve_prefill_sliding_qkv_in0_block_w() == 10
