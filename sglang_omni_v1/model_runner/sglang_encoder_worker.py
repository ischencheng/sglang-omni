# SPDX-License-Identifier: Apache-2.0
"""Minimal SGLang-native encoder worker.

Owns SGLang's distributed init, encoder-only ``ServerArgs`` /
``ModelConfig`` / ``LoadConfig``, the loaded upstream encoder model,
and exposes ``encode_batch()``. Does not start the upstream
``MMEncoder`` (no ZMQ, no transfer engine, no embedding cache) — v1
already owns request lifecycle, control plane, and relay.

See ``docs/developer_reference/encoder_tp_path_b_design.md`` for the
full RFC. The init sequence mirrors
``sglang/python/sglang/srt/disaggregation/encode_server.py`` so it
inherits SGLang's TP + native encoder implementations.
"""
from __future__ import annotations

import logging
import socket
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.model_loader import get_model
from sglang.srt.server_args import set_global_server_args_for_scheduler

from sglang_omni_v1.scheduling.sglang_backend import build_sglang_encoder_server_args

if TYPE_CHECKING:
    from sglang_omni_v1.models.qwen3_omni.encoder_adapters import BatchPlan

logger = logging.getLogger(__name__)


# Worker-managed kwargs that the helper consumes as direct keyword args.
# Reject these in `server_args_overrides` BEFORE the **splat — otherwise
# Python raises TypeError "got multiple values for keyword argument" before
# the helper's protected-key reject can fire.
_WORKER_MANAGED_KEYS = frozenset({
    "model_path", "tp_size", "base_gpu_id", "dist_init_addr",
    "dtype", "load_format",
})


def _pick_free_port() -> int:
    """Return an available loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class SGLangEncoderWorker:
    """SGLang-native encoder model wrapper.

    The worker process is invoked by ``MultiProcessPipelineRunner``,
    which has remapped ``CUDA_VISIBLE_DEVICES`` to a single physical GPU
    before importing torch (see ``encoder_tp_path_b_design.md`` "GPU
    placement"). Therefore ``cuda:0`` is the only visible device in this
    process, regardless of ``tp_size`` and the configured physical
    ``gpu``. The worker pins ``cuda_device=0`` and forwards
    ``dist_local_rank=tp_rank`` so SGLang's local-master / shard-index
    callsites observe a unique ``[0, world_size)`` rank.

    Args:
        model_path: HF model id or local checkpoint path.
        gpu_id: Physical GPU id (informational; the launcher remap means
            the worker actually sees this as ``cuda:0``).
        tp_rank: TP rank in ``[0, tp_size)``.
        tp_size: TP world size.
        nccl_port: Loopback NCCL bootstrap port allocated by the runner.
            Optional; the worker picks a free port when unset
            (single-rank case).
        dtype: Optional dtype override forwarded to ``ServerArgs``.
        load_format: Optional load-format override forwarded to ``ServerArgs``.
        server_args_overrides: Extra ``ServerArgs`` kwargs (loader / processor
            knobs only — protected keys raise ``ValueError``).
    """

    def __init__(
        self,
        *,
        model_path: str,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int | None,
        dtype: str | None = None,
        load_format: str | None = None,
        server_args_overrides: dict[str, Any] | None = None,
    ) -> None:
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1 (got {tp_size})")
        if not (0 <= tp_rank < tp_size):
            raise ValueError(
                f"tp_rank={tp_rank} must satisfy 0 <= tp_rank < tp_size={tp_size}"
            )

        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.is_entry_rank = self.tp_rank == 0

        # Forwarded for telemetry / debugging only — the worker hard-pins
        # cuda_device=0 because the launcher remap leaves only one device
        # visible.
        self.physical_gpu_id = int(gpu_id)

        # ServerArgs.dist_init_addr expects "host:port", torch.distributed
        # expects the full "tcp://host:port" URL. Keep them separate.
        port = int(nccl_port) if nccl_port is not None else _pick_free_port()
        dist_addr = f"127.0.0.1:{port}"
        dist_init_method = f"tcp://{dist_addr}"

        cuda_device = 0
        dist_local_rank = self.tp_rank
        self.device = torch.device(f"cuda:{cuda_device}")

        overrides = dict(server_args_overrides or {})
        clobbered = sorted(_WORKER_MANAGED_KEYS & overrides.keys())
        if clobbered:
            raise ValueError(
                f"server_args_overrides cannot set worker-managed keys "
                f"{clobbered}. These are derived from StageConfig and "
                f"factory parameters; pass them through StageConfig "
                f"(model_path, tp_size, gpu, dtype, load_format) instead."
            )

        server_args = build_sglang_encoder_server_args(
            model_path=model_path,
            tp_size=self.tp_size,
            base_gpu_id=cuda_device,
            dist_init_addr=dist_addr,
            dtype=dtype,
            load_format=load_format,
            **overrides,
        )
        set_global_server_args_for_scheduler(server_args)

        self.server_args = server_args
        self.model_config = ModelConfig.from_server_args(server_args)
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=(
                server_args.remote_instance_weight_loader_seed_instance_ip
            ),
            remote_instance_weight_loader_seed_instance_service_port=(
                server_args.remote_instance_weight_loader_seed_instance_service_port
            ),
            remote_instance_weight_loader_send_weights_group_ports=(
                server_args.remote_instance_weight_loader_send_weights_group_ports
            ),
        )

        torch.cuda.set_device(cuda_device)

        # Always run init_distributed_environment + initialize_model_parallel,
        # including tp_size==1: SGLang's parallel layers
        # (ColumnParallelLinear / RowParallelLinear) call get_tp_group()
        # during their own __init__ and will assert if the group has not
        # been initialized.
        init_distributed_environment(
            world_size=self.tp_size,
            rank=self.tp_rank,
            distributed_init_method=dist_init_method,
            local_rank=dist_local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        initialize_dp_attention(server_args, self.model_config)
        self.tp_group = get_tp_group()

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(device="cuda", gpu_id=cuda_device),
        )
        logger.info(
            "SGLangEncoderWorker ready (tp_rank=%d/%d, physical_gpu=%d, "
            "model=%s, dist_addr=%s)",
            self.tp_rank, self.tp_size, self.physical_gpu_id,
            model_path, dist_addr,
        )

    @torch.no_grad()
    def encode_batch(self, plan: "BatchPlan") -> dict[str, torch.Tensor | None]:
        """Run the encoder forward for one ``BatchPlan``.

        Determinism guarantee: this method is deterministic given identical
        ``plan`` inputs across ranks, which the EncoderScheduler ensures
        through the two-channel broadcast.
        """
        return plan.adapter.run_feature(self.model, plan)
