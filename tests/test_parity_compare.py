# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("parity_compare.py")


def _dump(
    path: Path, image_embeds: torch.Tensor, layers: dict[str, torch.Tensor] | None = None
) -> None:
    with open(path, "wb") as f:
        payload = {"image_embeds": image_embeds, "deepstack": None}
        if layers is not None:
            payload["layers"] = layers
        pickle.dump(payload, f)


def test_tp_gate_passes_directional_match(tmp_path: Path) -> None:
    left = torch.eye(4, dtype=torch.float16)
    right = left.clone()
    right[0, 0] += 1e-4
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    _dump(left_path, left)
    _dump(right_path, right)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(left_path), str(right_path), "--tp"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "TP parity: PASS" in result.stdout


def test_tp_gate_fails_low_cosine(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    _dump(left_path, torch.eye(4, dtype=torch.float16))
    _dump(right_path, -torch.eye(4, dtype=torch.float16))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(left_path), str(right_path), "--tp"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "TP parity: FAIL" in result.stdout


def test_layer_compare_skips_shard_local_shape_mismatch(tmp_path: Path) -> None:
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"
    _dump(
        left_path,
        torch.eye(4, dtype=torch.float16),
        layers={
            "blk_00": torch.zeros((2, 4), dtype=torch.float16),
            "blk_00.attn.qkv_proj": torch.zeros((2, 8), dtype=torch.float16),
        },
    )
    _dump(
        right_path,
        torch.eye(4, dtype=torch.float16),
        layers={
            "blk_00": torch.zeros((2, 4), dtype=torch.float16),
            "blk_00.attn.qkv_proj": torch.zeros((2, 4), dtype=torch.float16),
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(left_path),
            str(right_path),
            "--layers",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped shape-mismatched shard-local tensors" in result.stdout
    assert "blk_00.attn.qkv_proj" in result.stdout
