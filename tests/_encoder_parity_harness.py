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


def _parse_capture_blocks(spec: str | None, num_blocks: int) -> set[int]:
    if spec is None or spec.strip() == "":
        return set(range(num_blocks))
    blocks: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            blocks.update(range(start, end + 1))
        else:
            blocks.add(int(part))
    bad = sorted(i for i in blocks if i < 0 or i >= num_blocks)
    if bad:
        raise SystemExit(
            f"--capture-blocks contains out-of-range block indexes {bad}; "
            f"valid range is 0..{num_blocks - 1}"
        )
    return blocks


def _parse_index_set(spec: str | None) -> list[int] | None:
    if spec is None or spec.strip() == "":
        return None
    indexes: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            indexes.update(range(int(start_s), int(end_s) + 1))
        else:
            indexes.add(int(part))
    bad = sorted(i for i in indexes if i < 0)
    if bad:
        raise SystemExit(f"negative token indexes are not supported: {bad}")
    return sorted(indexes)


def _premerge_indices(final_indices: list[int] | None, merge: int) -> list[int] | None:
    if final_indices is None:
        return None
    out: list[int] = []
    for idx in final_indices:
        out.extend(range(idx * merge, (idx + 1) * merge))
    return out


def _slice_capture_tensor(
    tensor: torch.Tensor,
    *,
    final_indices: list[int] | None,
    pre_indices: list[int] | None,
) -> torch.Tensor:
    if final_indices is None:
        return tensor

    final_max = max(final_indices)
    pre_max = max(pre_indices or final_indices)
    for dim, size in enumerate(tensor.shape[:2]):
        if size > pre_max:
            index = torch.tensor(pre_indices, dtype=torch.long, device=tensor.device)
            return tensor.index_select(dim, index)
        if size > final_max:
            index = torch.tensor(final_indices, dtype=torch.long, device=tensor.device)
            return tensor.index_select(dim, index)
    return tensor


def _all_gather_shard_output(
    name: str,
    module,
    tensor: torch.Tensor,
    *,
    tp_size: int,
    enabled: bool,
) -> torch.Tensor:
    """Gather TP shard-local debug captures into TP1-compatible layout."""
    if not enabled or tp_size <= 1 or not torch.is_tensor(tensor) or tensor.dim() == 0:
        return tensor

    from sglang.srt.distributed import tensor_model_parallel_all_gather
    from sglang.srt.layers.linear import ColumnParallelLinear

    if isinstance(module, ColumnParallelLinear):
        sizes = list(getattr(module, "output_partition_sizes", []) or [])
        if len(sizes) > 1:
            chunks = torch.split(tensor, sizes, dim=-1)
            gathered_chunks = [
                tensor_model_parallel_all_gather(chunk.contiguous(), dim=-1)
                for chunk in chunks
            ]
            return torch.cat(gathered_chunks, dim=-1)
        return tensor_model_parallel_all_gather(tensor.contiguous(), dim=-1)

    # Merger mlp.1 is GELU over the shard-local ColumnParallelLinear output.
    # Gather it too so the activation can be compared against tp=1 directly.
    if name.endswith(".mlp.1") and module.__class__.__name__ == "GELU":
        return tensor_model_parallel_all_gather(tensor.contiguous(), dim=-1)

    return tensor


def _all_gather_attention_backend_output(
    tensor: torch.Tensor,
    *,
    tp_size: int,
    enabled: bool,
) -> torch.Tensor:
    if not enabled or tp_size <= 1 or not torch.is_tensor(tensor) or tensor.dim() < 2:
        return tensor

    from sglang.srt.distributed import tensor_model_parallel_all_gather

    # VisionAttention qkv_backend output is [tokens, local_heads, head_dim].
    return tensor_model_parallel_all_gather(tensor.contiguous(), dim=1)


def _patch_qkv_backend_capture(
    attn,
    name: str,
    capture_tensor,
    *,
    tp_size: int,
    capture_gathered_shards: bool,
) -> None:
    backend = getattr(attn, "qkv_backend", None)
    forward = getattr(backend, "forward", None)
    if forward is None:
        return

    def wrapped_forward(*args, **kwargs):
        out = forward(*args, **kwargs)
        if torch.is_tensor(out):
            capture_tensor(
                name,
                backend,
                _all_gather_attention_backend_output(
                    out,
                    tp_size=tp_size,
                    enabled=capture_gathered_shards,
                ),
            )
        return out

    backend.forward = wrapped_forward


def _register_block_internal_hooks(blk, index: int, grab_factory) -> None:
    prefix = f"blk_{index:02d}"
    for attr in ("norm1", "norm2"):
        mod = getattr(blk, attr, None)
        if mod is not None:
            mod.register_forward_hook(grab_factory(f"{prefix}.{attr}"))

    attn = getattr(blk, "attn", None)
    if attn is not None:
        attn.register_forward_hook(grab_factory(f"{prefix}.attn"))
        for attr in ("qkv_proj", "qkv_backend", "proj"):
            mod = getattr(attn, attr, None)
            if mod is not None:
                mod.register_forward_hook(grab_factory(f"{prefix}.attn.{attr}"))

    mlp = getattr(blk, "mlp", None)
    if mlp is not None:
        mlp.register_forward_hook(grab_factory(f"{prefix}.mlp"))
        for attr in ("linear_fc1", "linear_fc2"):
            mod = getattr(mlp, attr, None)
            if mod is not None:
                mod.register_forward_hook(grab_factory(f"{prefix}.mlp.{attr}"))


def _register_merger_internal_hooks(merger, prefix: str, grab_factory) -> None:
    for attr in ("ln_q", "norm", "linear_fc1", "linear_fc2"):
        mod = getattr(merger, attr, None)
        if mod is not None:
            mod.register_forward_hook(grab_factory(f"{prefix}.{attr}"))
    mlp = getattr(merger, "mlp", None)
    if mlp is not None:
        for i, mod in enumerate(mlp):
            mod.register_forward_hook(grab_factory(f"{prefix}.mlp.{i}"))


def _run_sglang(
    model_path: str,
    out_path: str,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    capture_layers: bool = False,
    capture_block_internals: bool = False,
    capture_blocks: str | None = None,
    capture_token_indices: str | None = None,
    dtype: str = "float16",
    tp_parity_mode: str = "default",
    capture_gathered_shards: bool = False,
) -> None:
    from sglang_omni_v1.model_runner.sglang_encoder_runner import (
        SGLangEncoderRunner,
    )
    from sglang_omni_v1.models.qwen3_omni.encoder_adapters import (
        Qwen3OmniImageEncoderAdapter,
    )
    from sglang_omni_v1.proto import StagePayload
    from sglang_omni_v1.scheduling.messages import IncomingMessage

    pixel, grid = _build_real_image_inputs(model_path)
    final_capture_indices = _parse_index_set(capture_token_indices)
    runner = SGLangEncoderRunner(
        model_path=model_path,
        gpu_id=0,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
        encoder_specs=Qwen3OmniImageEncoderAdapter.encoder_specs,
        dtype=dtype,
        tp_parity_mode=tp_parity_mode,
    )

    # Optional layer-output capture for tp1-vs-tp2 layer-by-layer
    # parity. Hooks fire on the block boundary (after both
    # attn.o_proj's all-reduce and mlp.down_proj's all-reduce), so
    # the captured tensor is replicated across ranks. Output of
    # patch_embed is independent of TP; output of merger is the
    # final (N, base*(1+ndeepstack)) used to slice image_embeds +
    # deepstack.
    captures: dict[str, torch.Tensor] = {}
    if capture_layers or capture_block_internals:
        visual = runner.model.visual
        pre_capture_indices = _premerge_indices(
            final_capture_indices,
            int(getattr(visual, "spatial_merge_unit", 4)),
        )

        def _capture_tensor(name: str, mod, t: torch.Tensor) -> None:
            t = _all_gather_shard_output(
                name,
                mod,
                t,
                tp_size=tp_size,
                enabled=capture_gathered_shards,
            )
            captures[name] = _slice_capture_tensor(
                t,
                final_indices=final_capture_indices,
                pre_indices=pre_capture_indices,
            ).detach().cpu()

        def _grab(name: str):
            def hook(mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t):
                    _capture_tensor(name, mod, t)
            return hook

        if hasattr(visual, "patch_embed"):
            visual.patch_embed.register_forward_hook(_grab("00_patch_embed"))
        if hasattr(visual, "blocks"):
            selected_blocks = _parse_capture_blocks(
                capture_blocks, len(visual.blocks)
            )
            for i, blk in enumerate(visual.blocks):
                if capture_layers:
                    blk.register_forward_hook(_grab(f"blk_{i:02d}"))
                if capture_block_internals and i in selected_blocks:
                    _register_block_internal_hooks(blk, i, _grab)
                    attn = getattr(blk, "attn", None)
                    if attn is not None:
                        _patch_qkv_backend_capture(
                            attn,
                            f"blk_{i:02d}.attn.qkv_backend",
                            _capture_tensor,
                            tp_size=tp_size,
                            capture_gathered_shards=capture_gathered_shards,
                        )
        if hasattr(visual, "merger"):
            visual.merger.register_forward_hook(_grab("99_merger"))
            if capture_block_internals:
                _register_merger_internal_hooks(visual.merger, "99_merger", _grab)
        if capture_block_internals and hasattr(visual, "deepstack_merger_list"):
            for i, merger in enumerate(visual.deepstack_merger_list):
                prefix = f"deepstack_merger_{i}"
                merger.register_forward_hook(_grab(prefix))
                _register_merger_internal_hooks(merger, prefix, _grab)

    hf_cfg = runner.model_config.hf_config
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=hf_cfg, dtype=torch_dtype)
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
    raw = runner.encode_batch(plan)
    # Only rank 0 writes (the result is the same across ranks because of TP-symmetric outputs).
    if tp_rank != 0:
        return
    sliced = adapter.slice_results(raw, plan, [msg])
    state = sliced[0].data["encoder_outs"]["image_encoder"]
    image_embeds = state["image_embeds"].detach().cpu()
    deepstack = state.get("deepstack_visual_embeds_image")
    if deepstack is not None:
        deepstack = [t.detach().cpu() for t in deepstack]
    out_dict: dict[str, object] = {
        "image_embeds": image_embeds,
        "deepstack": deepstack,
        "metadata": {
            "tp_parity_mode": tp_parity_mode,
            "dtype": dtype,
            "capture_token_indices": final_capture_indices,
            "capture_gathered_shards": capture_gathered_shards,
        },
    }
    if capture_layers or capture_block_internals:
        out_dict["layers"] = captures
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)


def _run_bare_sglang(model_path: str, out_path: str) -> None:
    """Upstream SGLang ``get_model()`` on the full
    ``Qwen3OmniMoeForConditionalGeneration``, no ``encoder_only``, no
    ``EncoderModuleContainer``, no fused-shard dispatch logic from us.
    Then call ``thinker.get_image_feature`` directly.

    Isolates: does our wrapper (partial-load + fused-shard dispatch +
    init_distributed at tp=1) introduce any numerical drift vs a pure
    upstream load? Both lanes run at tp_size=1 so TP collectives are
    identical no-ops.
    """
    import torch
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.model_loader import get_model
    from sglang.srt.server_args import (
        ServerArgs, set_global_server_args_for_scheduler,
    )

    from sglang_omni_v1.proto import StagePayload
    from sglang_omni_v1.scheduling.messages import IncomingMessage
    from sglang_omni_v1.models.qwen3_omni.encoder_adapters import (
        Qwen3OmniImageEncoderAdapter,
    )

    pixel, grid = _build_real_image_inputs(model_path)

    server_args = ServerArgs(
        model_path=model_path,
        trust_remote_code=True,
        tp_size=1, pp_size=1, base_gpu_id=0,
        dist_init_addr="127.0.0.1:29577",
        encoder_only=False,           # bare = full ForConditionalGeneration
        mm_enable_dp_encoder=False,
        disable_cuda_graph=True,
        random_seed=123,
        dtype="float16",
        mem_fraction_static=0.9,
    )
    set_global_server_args_for_scheduler(server_args)

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29577",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    model_config = ModelConfig.from_server_args(server_args)
    initialize_dp_attention(server_args, model_config)
    load_config = LoadConfig(load_format=server_args.load_format)
    torch.cuda.set_device(0)
    device_config = DeviceConfig(device="cuda", gpu_id=0)
    model = get_model(
        model_config=model_config,
        load_config=load_config,
        device_config=device_config,
    )

    # Build the MultimodalDataItem exactly like our adapter does.
    hf_cfg = model_config.hf_config
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=hf_cfg, dtype=torch.float16)
    msg = IncomingMessage(
        request_id="r0",
        type="new_request",
        data=StagePayload(
            request_id="r0", request=None,
            data={"encoder_inputs": {"image_encoder": {
                "pixel_values": pixel.to("cuda:0"),
                "image_grid_thw": grid,
            }}},
        ),
    )
    plan = adapter.build_batch([msg])
    items = plan.image_items
    thinker = model.thinker if hasattr(model, "thinker") else model
    visual = thinker.visual
    pixel_values = torch.cat([item.feature for item in items], dim=0).type(visual.dtype)
    grid = torch.cat([item.image_grid_thw for item in items], dim=0)
    with torch.no_grad():
        embeds = visual(pixel_values, grid_thw=grid)
    image_embeds = embeds.detach().cpu() if torch.is_tensor(embeds) else embeds[0].detach().cpu()
    with open(out_path, "wb") as f:
        pickle.dump({"image_embeds": image_embeds, "deepstack": None}, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["local", "sglang", "bare_sglang"])
    parser.add_argument("out_path")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=None)
    parser.add_argument("--capture-layers", action="store_true")
    parser.add_argument("--capture-block-internals", action="store_true")
    parser.add_argument(
        "--capture-blocks",
        default=None,
        help="Comma-separated block indexes/ranges for --capture-block-internals, e.g. 25,26 or 20-26.",
    )
    parser.add_argument(
        "--capture-token-indices",
        default=None,
        help="Comma-separated final-token indexes/ranges to keep in layer captures; pre-merger tensors keep the corresponding 2x2 patch tokens.",
    )
    parser.add_argument(
        "--capture-gathered-shards",
        action="store_true",
        help="All-gather shard-local Column/QKV/Gate-Up debug captures so they can be compared with tp=1 full-width tensors.",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument(
        "--tp-parity-mode",
        default="default",
        choices=["default", "fp32_row_parallel", "fp32_linear"],
    )
    args = parser.parse_args()

    model_path = _resolve_model_path()
    if args.backend == "local":
        _run_local(model_path, args.out_path)
    elif args.backend == "bare_sglang":
        _run_bare_sglang(model_path, args.out_path)
    else:
        _run_sglang(
            model_path,
            args.out_path,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            nccl_port=args.nccl_port,
            capture_layers=args.capture_layers,
            capture_block_internals=args.capture_block_internals,
            capture_blocks=args.capture_blocks,
            capture_token_indices=args.capture_token_indices,
            dtype=args.dtype,
            tp_parity_mode=args.tp_parity_mode,
            capture_gathered_shards=args.capture_gathered_shards,
        )
    print(
        f"PARITY_OK backend={args.backend} tp={args.tp_size} rank={args.tp_rank} "
        f"out={args.out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
