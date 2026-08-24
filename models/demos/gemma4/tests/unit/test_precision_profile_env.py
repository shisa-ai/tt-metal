# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path

import ttnn
from models.demos.gemma4.tt import precision

MODEL_KEY = "e6cca0128d4fe1fcafbbcdb62dcbaf458b420af2"


def _restore_default_module():
    return importlib.reload(precision)


def test_external_precision_profile_env_reaches_direct_model_import(monkeypatch, tmp_path):
    profile = tmp_path / "repo-owned-precision.json"
    profile.write_text(json.dumps({MODEL_KEY: {"1x1": {"shared_mlp": "bfp8", "lm_head": "bfp8"}}}))

    try:
        with monkeypatch.context() as context:
            context.setenv("GEMMA4_PRECISION_PROFILE", str(profile))
            module = importlib.reload(precision)
            assert Path(module._PATH) == profile
            resolved = module.Gemma4Precision.load(Path("/models") / MODEL_KEY, (1, 1))
            assert resolved.get("shared_mlp") == ttnn.bfloat8_b
            assert resolved.get("lm_head") == ttnn.bfloat8_b
            assert resolved.get("attention") == ttnn.bfloat16
    finally:
        _restore_default_module()


def test_missing_external_precision_profile_fails_instead_of_using_bf16(monkeypatch, tmp_path, expect_error):
    missing = tmp_path / "missing.json"
    try:
        with monkeypatch.context() as context:
            context.setenv("GEMMA4_PRECISION_PROFILE", str(missing))
            module = importlib.reload(precision)
            with expect_error(
                FileNotFoundError,
                "configured Gemma4 precision profile does not exist",
            ):
                module.Gemma4Precision.load(Path("/models") / MODEL_KEY, (1, 1))
    finally:
        _restore_default_module()
