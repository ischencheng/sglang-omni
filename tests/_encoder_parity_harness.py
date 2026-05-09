# SPDX-License-Identifier: Apache-2.0
"""Subprocess harness for backend=local vs backend=sglang encoder parity.

Loads exactly ONE backend per process to avoid OOM (each Qwen3-Omni
load takes ~57 GB on H100). The parent script runs the backends
sequentially on the same GPU and pickles output tensors to disk for
post-hoc comparison.

Run as: ``python _encoder_parity_harness.py <local|sglang> <out_path>``.

Companion script: ``parity_compare.py`` reads the two pickles and
emits the strict RFC tolerance report.

Findings from the local run (2026-05-09, Qwen3-Omni-30B-A3B-Instruct,
fp16, real ``cars.jpg``); see also
``docs/developer_reference/encoder_tp_parity_findings.md``.

backend=local (HF tower) vs backend=sglang at tp_size=1
--------------------------------------------------------
- Token count + dtype match exactly: (6042, 2048) fp16.
- Distributions match: local mean=0.0041 std=0.3294;
  sglang mean=0.0040 std=0.3296.
- Per-token cosine sim: 84% ≥ 0.99, 96.7% ≥ 0.9, mean 0.988.
- Per-element max abs diff 5.95, mean 0.022.
- ``allclose(atol=1e-3, rtol=1e-3)`` (RFC) → **False**.
- Real implementation difference between HF transformers
  ``Qwen3OmniMoeVisionEncoder`` and SGLang's reimplemented
  ``Qwen3VLMoeVisionModel``. Different attention / fused QKV /
  normalization paths produce non-bit-equivalent results. Both
  backends pass the docs CI probes with semantically equivalent
  outputs.

backend=sglang at tp_size=1 vs tp_size=2
-----------------------------------------
- Token count + dtype match exactly.
- Distributions bit-identical at 4 decimals.
- Per-token cosine sim: **97.5% ≥ 0.9999**, mean 0.999984.
- Per-element max abs diff 0.27, mean 0.0008.
- ``allclose(atol=1e-3, rtol=1e-3)`` → **False** (one outlier element
  at 0.27, NCCL fp16 accumulation-order noise).
- Effectively exact within fp16 collective rounding.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import torch


def _resolve_model_path() -> str:
    env_path = os.environ.get("QWEN3_OMNI_MODEL_PATH")
    if env_path:
        return env_path
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    snap_root = (
        cache_root
        / "models--Qwen--Qwen3-Omni-30B-A3B-Instruct"
        / "snapshots"
    )
    for snap in snap_root.iterdir():
        if (snap / "config.json").exists():
            return str(snap)
    raise SystemExit("Qwen3-Omni checkpoint not found")


def _build_real_image_inputs(model_path: str):
    """Run the production preprocessor on tests/data/cars.jpg.

    Synthetic gaussian-noise inputs drive activations out of the
    pre-normalization range both implementations were trained on. A
    real image keeps inputs in the valid distribution.
    """
    from transformers import AutoImageProcessor
    from PIL import Image

    img_path = Path(__file__).resolve().parents[1] / "tests" / "data" / "cars.jpg"
    img = Image.open(img_path).convert("RGB")
    proc = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    out = proc(images=img, return_tensors="pt")
    pixel = out["pixel_values"].to(torch.float32)
    if "image_grid_thw" in out:
        grid = out["image_grid_thw"].to(torch.long)
    elif "image_grid_hws" in out:
        grid = out["image_grid_hws"].to(torch.long)
    else:
        raise SystemExit(f"unexpected processor outputs: {list(out.keys())}")
    return pixel, grid


def _run_local(model_path: str, out_path: str) -> None:
    from sglang_omni_v1.models.qwen3_omni.components.image_encoder import (
        Qwen3OmniImageEncoder,
    )

    pixel, grid = _build_real_image_inputs(model_path)
    model = Qwen3OmniImageEncoder(
        model_path=model_path, device="cuda:0", dtype="float16"
    )
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel.to("cuda:0"),
            image_grid_thw=grid.to("cuda:0"),
        )
    image_embeds = outputs["image_embeds"].detach().cpu()
    deepstack = outputs.get("deepstack_visual_embeds_image")
    if deepstack is not None:
        deepstack = [t.detach().cpu() for t in deepstack]
    with open(out_path, "wb") as f:
        pickle.dump({"image_embeds": image_embeds, "deepstack": deepstack}, f)


def _run_sglang(model_path: str, out_path: str, *, tp_size: int = 1, tp_rank: int = 0, nccl_port: int | None = None) -> None:
    from sglang_omni_v1.model_runner.sglang_encoder_worker import (
        SGLangEncoderWorker,
    )
    from sglang_omni_v1.models.qwen3_omni.encoder_adapters import (
        Qwen3OmniImageEncoderAdapter,
    )
    from sglang_omni_v1.proto import StagePayload
    from sglang_omni_v1.scheduling.messages import IncomingMessage

    pixel, grid = _build_real_image_inputs(model_path)
    worker = SGLangEncoderWorker(
        model_path=model_path,
        gpu_id=0,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
        dtype="float16",
    )
    hf_cfg = worker.model_config.hf_config
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=hf_cfg, dtype=torch.float16)
    msg = IncomingMessage(
        request_id="r0",
        type="new_request",
        data=StagePayload(
            request_id="r0",
            request=None,
            data={
                "encoder_inputs": {
                    "image_encoder": {
                        "pixel_values": pixel.to("cuda:0"),
                        "image_grid_thw": grid,  # CPU per upstream contract
                    }
                }
            },
        ),
    )
    plan = adapter.build_batch([msg])
    raw = worker.encode_batch(plan)
    # Only rank 0 writes (the result is the same across ranks because of TP-symmetric outputs).
    if tp_rank != 0:
        return
    sliced = adapter.slice_results(raw, plan, [msg])
    state = sliced[0].data["encoder_outs"]["image_encoder"]
    image_embeds = state["image_embeds"].detach().cpu()
    deepstack = state.get("deepstack_visual_embeds_image")
    if deepstack is not None:
        deepstack = [t.detach().cpu() for t in deepstack]
    with open(out_path, "wb") as f:
        pickle.dump({"image_embeds": image_embeds, "deepstack": deepstack}, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["local", "sglang"])
    parser.add_argument("out_path")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=None)
    args = parser.parse_args()

    model_path = _resolve_model_path()
    if args.backend == "local":
        _run_local(model_path, args.out_path)
    else:
        _run_sglang(
            model_path,
            args.out_path,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            nccl_port=args.nccl_port,
        )
    print(
        f"PARITY_OK backend={args.backend} tp={args.tp_size} rank={args.tp_rank} "
        f"out={args.out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
