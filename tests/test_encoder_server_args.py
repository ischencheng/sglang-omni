# SPDX-License-Identifier: Apache-2.0
"""Tests for ``build_sglang_encoder_server_args`` and the protected-keys reject.

We avoid actually constructing ``ServerArgs`` (which fetches the model
config from HF Hub) by monkey-patching it to a recording stand-in. The
helper must reject protected keys *before* delegating, so the stand-in
never sees a bad call.
"""
from __future__ import annotations

import pytest

from sglang_omni_v1.scheduling.sglang_backend import server_args_builder
from sglang_omni_v1.scheduling.sglang_backend import (
    build_sglang_encoder_server_args,
)


class _FakeServerArgs:
    """Records the kwargs the builder passed without doing HF lookup."""

    captured: dict | None = None

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _stub_server_args(monkeypatch):
    monkeypatch.setattr(server_args_builder, "ServerArgs", _FakeServerArgs)
    _FakeServerArgs.captured = None
    yield


# ---------------------------------------------------------------------------
# Happy path: helper produces ServerArgs kwargs with encoder-only fork on
# ---------------------------------------------------------------------------


def test_helper_sets_encoder_only_fork():
    args = build_sglang_encoder_server_args(
        model_path="dummy/model",
        tp_size=2,
        base_gpu_id=0,
        dist_init_addr="127.0.0.1:29500",
    )
    assert args.encoder_only is True
    assert args.language_only is False
    assert args.mm_enable_dp_encoder is False
    assert args.disable_cuda_graph is True
    assert args.tp_size == 2
    assert args.pp_size == 1
    assert args.base_gpu_id == 0
    assert args.dist_init_addr == "127.0.0.1:29500"
    assert args.trust_remote_code is True


def test_helper_passes_dtype_load_format():
    args = build_sglang_encoder_server_args(
        model_path="dummy/model",
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr="127.0.0.1:29500",
        dtype="float16",
        load_format="auto",
    )
    assert args.dtype == "float16"
    assert args.load_format == "auto"


def test_helper_forwards_loader_overrides():
    """Loader / processor knobs are NOT protected — pass through."""
    args = build_sglang_encoder_server_args(
        model_path="dummy/model",
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr="127.0.0.1:29500",
        model_loader_extra_config={"key": "value"},
        download_dir="/tmp/cache",
    )
    assert args.model_loader_extra_config == {"key": "value"}
    assert args.download_dir == "/tmp/cache"


# ---------------------------------------------------------------------------
# Protected-key rejects (helper level — go through `**overrides`)
# ---------------------------------------------------------------------------

# AR-only knobs that have no meaning for an encoder-only worker.
_AR_ONLY = [
    ("mem_fraction_static", 0.5),
    ("max_running_requests", 32),
    ("max_prefill_tokens", 1024),
    ("chunked_prefill_size", 512),
    ("context_length", 4096),
]

# Parallelism / placement keys that pass through ``**overrides``.
# (``tp_size``, ``base_gpu_id``, ``dist_init_addr`` are direct kwargs of
# the helper; passing them again via `**overrides` is a Python-level
# TypeError before our reject can fire — the worker's
# ``_WORKER_MANAGED_KEYS`` reject catches users who try this through the
# worker's ``server_args_overrides`` dict.)
_PARALLELISM = [
    ("pp_size", 2),
    ("dp_size", 2),
    ("ep_size", 4),
    ("moe_dense_tp_size", 2),
    ("nnodes", 2),
    ("node_rank", 1),
]

# Encoder-only fork knobs.
_ENCODER_FORK = [
    ("encoder_only", False),
    ("language_only", True),
    ("mm_enable_dp_encoder", True),
    ("enable_dp_attention", True),
    ("enable_dp_lm_head", True),
    ("disable_cuda_graph", False),
    ("device", "cpu"),
]


@pytest.mark.parametrize("key,value", _AR_ONLY)
def test_helper_rejects_ar_only_knobs(key, value):
    with pytest.raises(ValueError, match=key):
        build_sglang_encoder_server_args(
            model_path="dummy/model",
            tp_size=1,
            base_gpu_id=0,
            dist_init_addr="127.0.0.1:29500",
            **{key: value},
        )


@pytest.mark.parametrize("key,value", _PARALLELISM)
def test_helper_rejects_parallelism_keys(key, value):
    with pytest.raises(ValueError, match=key):
        build_sglang_encoder_server_args(
            model_path="dummy/model",
            tp_size=1,
            base_gpu_id=0,
            dist_init_addr="127.0.0.1:29500",
            **{key: value},
        )


@pytest.mark.parametrize("key,value", _ENCODER_FORK)
def test_helper_rejects_encoder_fork_keys(key, value):
    with pytest.raises(ValueError, match=key):
        build_sglang_encoder_server_args(
            model_path="dummy/model",
            tp_size=1,
            base_gpu_id=0,
            dist_init_addr="127.0.0.1:29500",
            **{key: value},
        )


# ---------------------------------------------------------------------------
# Worker-level reject — protected keys reachable only through
# ``server_args_overrides`` dict on the worker
# ---------------------------------------------------------------------------


def test_worker_rejects_worker_managed_keys_in_overrides_dict():
    """SGLangEncoderWorker rejects worker-managed keys *before* the
    helper-level **splat, otherwise Python raises a confusing
    "got multiple values" TypeError. Tested without actually
    instantiating the worker (which would require a real model)."""
    from sglang_omni_v1.model_runner import sglang_encoder_worker as sew

    # The reject runs before any heavy initialization, so we can hit it
    # directly by calling __init__ with a stub server_args path that
    # raises before init_distributed_environment.
    cls = sew.SGLangEncoderWorker
    # Build a dummy instance bypassing __init__ to hit the validation.
    inst = cls.__new__(cls)
    # Re-run only the early validation by calling __init__ partially
    # via a minimal stub: pass overrides with a worker-managed key.
    with pytest.raises(ValueError, match="worker-managed keys"):
        cls.__init__(
            inst,
            model_path="dummy/model",
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            nccl_port=None,
            server_args_overrides={"tp_size": 99},
        )


def test_worker_rejects_invalid_tp_rank():
    from sglang_omni_v1.model_runner.sglang_encoder_worker import SGLangEncoderWorker

    inst = SGLangEncoderWorker.__new__(SGLangEncoderWorker)
    with pytest.raises(ValueError, match="tp_rank"):
        SGLangEncoderWorker.__init__(
            inst,
            model_path="dummy/model",
            gpu_id=0,
            tp_rank=2,
            tp_size=2,  # tp_rank must be < tp_size, so 2 is invalid
            nccl_port=None,
        )


def test_worker_rejects_invalid_tp_size():
    from sglang_omni_v1.model_runner.sglang_encoder_worker import SGLangEncoderWorker

    inst = SGLangEncoderWorker.__new__(SGLangEncoderWorker)
    with pytest.raises(ValueError, match="tp_size"):
        SGLangEncoderWorker.__init__(
            inst,
            model_path="dummy/model",
            gpu_id=0,
            tp_rank=0,
            tp_size=0,
            nccl_port=None,
        )
