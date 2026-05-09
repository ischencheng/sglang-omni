# SPDX-License-Identifier: Apache-2.0
"""Compare ``image_embeds`` pickles from the parity harness.

Reports per-element diff stats, per-token cosine similarity buckets,
and the strict RFC tolerance verdict (``atol=1e-3 rtol=1e-3`` on fp16).
"""
from __future__ import annotations

import argparse
import pickle
import sys

import torch


def _summarize(L: dict, S: dict, label: str) -> None:
    le = L[label].to(torch.float32) if not isinstance(L[label], list) else None
    se = S[label].to(torch.float32) if not isinstance(S[label], list) else None
    if le is None:
        return
    print(f"\n{label}: shape={le.shape}")
    print(f"  local stats:  abs_max={le.abs().max():.3f} mean={le.mean():.4f} std={le.std():.4f}")
    print(f"  sglang stats: abs_max={se.abs().max():.3f} mean={se.mean():.4f} std={se.std():.4f}")

    abs_diff = (le - se).abs()
    print(f"  abs diff: max={abs_diff.max().item():.4f} mean={abs_diff.mean().item():.4f}")
    cos = torch.nn.functional.cosine_similarity(le, se, dim=-1)
    print(f"  per-token cosine_sim: min={cos.min().item():.4f} mean={cos.mean().item():.4f}")

    for atol, rtol in [(1e-3, 1e-3), (1e-2, 1e-2), (5e-2, 5e-2)]:
        ok = torch.allclose(le, se, atol=atol, rtol=rtol)
        marker = "  <-- RFC" if (atol, rtol) == (1e-3, 1e-3) else ""
        print(f"  allclose(atol={atol}, rtol={rtol}): {ok}{marker}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("local_pkl")
    p.add_argument("sglang_pkl")
    args = p.parse_args()

    L = pickle.load(open(args.local_pkl, "rb"))
    S = pickle.load(open(args.sglang_pkl, "rb"))

    _summarize(L, S, "image_embeds")
    if L.get("deepstack") and S.get("deepstack"):
        for i, (lds, sds) in enumerate(zip(L["deepstack"], S["deepstack"])):
            _summarize({f"ds[{i}]": lds}, {f"ds[{i}]": sds}, f"ds[{i}]")


if __name__ == "__main__":
    main()
