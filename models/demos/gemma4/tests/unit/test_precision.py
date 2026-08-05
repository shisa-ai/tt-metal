# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json

import ttnn
from models.demos.gemma4.tt import precision


def test_bfp4_dtype_aliases_and_cache_suffix(tmp_path, monkeypatch):
    table = {
        "model": {
            "1x1": {
                "shared_mlp": "bfp4",
                "attention": "bfloat4_b",
            }
        }
    }
    path = tmp_path / "precision.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setattr(precision, "_PATH", str(path))

    resolved = precision.Gemma4Precision.load(tmp_path / "model", (1, 1))

    assert resolved.get("shared_mlp") == ttnn.bfloat4_b
    assert resolved.get("attention") == ttnn.bfloat4_b
    assert precision.dtype_to_str(ttnn.bfloat4_b) == "bfp4"
