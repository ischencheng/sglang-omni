# SPDX-License-Identifier: Apache-2.0
"""Shared ServerArgs construction for SGLang AR engines and encoder workers."""
from __future__ import annotations

from typing import Any

from sglang.srt.server_args import ServerArgs


def build_sglang_server_args(
    model_path: str,
    context_length: int,
    *,
    chunked_prefill_size: int | None = None,
    max_prefill_tokens: int = 16384,
    max_running_requests: int = 16,
    mem_fraction_static: float = 0.7,
    **overrides: Any,
) -> ServerArgs:
    """Build ServerArgs with shared defaults for all SGLang AR engines."""
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": 1,
        "pp_size": 1,
        "chunked_prefill_size": chunked_prefill_size,
        "max_prefill_tokens": max_prefill_tokens,
        "max_running_requests": max_running_requests,
        "mem_fraction_static": mem_fraction_static,
        "random_seed": 123,
        "context_length": context_length,
    }
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


# Worker invariants that must NOT be reachable from server_args_overrides.
# Mutating these would either invalidate GPU placement, change the
# parallelism axis we promised, or flip the encoder-vs-language-only
# fork. See encoder_tp_path_b_design.md "GPU placement..." and
# "AR-only knob protection".
_ENCODER_PROTECTED_KEYS = frozenset({
    # Parallelism / placement / rank topology.
    # For encoder stages, rank topology has exactly one source of truth:
    # StageConfig.tp_size + StageConfig.gpu, materialized by the runner
    # into per-rank kwargs. server_args_overrides is only for safe
    # SGLang loading/runtime knobs (quantization, attention backend,
    # remote weight loading); it must not be able to change rank
    # topology or GPU placement after the runner has already spawned
    # processes.
    "tp_size",
    "pp_size",
    "dp_size",
    "ep_size",
    "moe_dense_tp_size",
    "nnodes",
    "node_rank",
    "rank",
    "world_size",
    "tp_rank",
    "gpu_id",
    "base_gpu_id",
    "nccl_port",
    "dist_init_addr",
    # Encoder-only fork
    "encoder_only",
    "language_only",
    "mm_enable_dp_encoder",
    "enable_dp_attention",
    "enable_dp_lm_head",
    "disable_cuda_graph",
    "device",
    # AR-only knobs that have no meaning for an encoder-only worker.
    # Locked so users cannot reintroduce SGLang AR memory semantics
    # through server_args_overrides — the encoder path explicitly does
    # not own a KV pool, an AR running queue, or a chunked-prefill
    # budget.
    "mem_fraction_static",
    "max_running_requests",
    "max_prefill_tokens",
    "chunked_prefill_size",
    "context_length",
})


def build_sglang_encoder_server_args(
    model_path: str,
    *,
    tp_size: int,
    base_gpu_id: int,
    dist_init_addr: str,
    dtype: str | None = None,
    load_format: str | None = None,
    **overrides: Any,
) -> ServerArgs:
    """ServerArgs configured for an encoder-only SGLang worker.

    Distinct from :func:`build_sglang_server_args` because encoder
    stages do not have a meaningful ``context_length`` /
    ``mem_fraction_static`` / running-queue budget.

    Raises:
        ValueError: ``overrides`` tries to mutate a protected invariant
            (parallelism shape, GPU placement, encoder-only fork, or AR-only
            memory knobs). Such fields are decided by the worker / pipeline
            runner and must be passed through ``StageConfig`` (``tp_size``,
            ``gpu``) instead.
    """
    bad = sorted(_ENCODER_PROTECTED_KEYS & overrides.keys())
    if bad:
        raise ValueError(
            f"server_args_overrides cannot override protected keys: {bad}. "
            f"These are decided by the worker / pipeline runner; pass them "
            f"through StageConfig (tp_size, gpu) instead."
        )

    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": tp_size,
        "pp_size": 1,
        "base_gpu_id": base_gpu_id,
        "dist_init_addr": dist_init_addr,
        "encoder_only": True,
        "language_only": False,
        "mm_enable_dp_encoder": False,    # MVP: TP only
        "disable_cuda_graph": True,        # variable shapes; piecewise CG lands later
        "random_seed": 123,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if load_format is not None:
        kwargs["load_format"] = load_format
    kwargs.update(overrides)
    return ServerArgs(**kwargs)
