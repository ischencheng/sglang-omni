# SPDX-License-Identifier: Apache-2.0
"""Phase-0 launcher contract tests for encoder TP (Plan B).

These tests cover only the launcher / config layer — they do not
spawn child processes or load any model. They lock the contracts the
RFC calls "load-bearing":

- ``StageProcessSpec.single_visible_device`` flips when a stage's
  resolved factory backend is ``sglang`` / ``auto``, including when the
  flip is driven by ``runtime_overrides`` (signature-default trap).
- ``get_stage_process_env`` honors the flag at ``tp_size=1``.
- ``serve/launcher.py`` ``needs_mp`` predicate routes single-stage,
  single-GPU, ``tp_size=1`` ``backend="sglang"`` configs through the
  multi-process runner.
- ``compile_pipeline`` rejects ``backend in {"sglang","auto"}`` and
  ``tp_size > 1`` directly.
- The TP preflight in ``mp_runner._build_stage_groups`` rejects:
   * Layer 1 — TP stage whose factory does not accept
     ``tp_rank/tp_size/nccl_port``;
   * Layer 2 — encoder TP stage whose resolved backend is not
     ``sglang``.
- Production factory signature defaults stay ``"local"`` forever
  (real-factory signature lock).
"""
from __future__ import annotations

import inspect
import multiprocessing
import os
from typing import Any

import pytest

from sglang_omni_v1.config.compiler import compile_pipeline
from sglang_omni_v1.config.schema import (
    EndpointsConfig,
    PipelineConfig,
    StageConfig,
)
from sglang_omni_v1.pipeline.mp_runner import _build_stage_groups
from sglang_omni_v1.pipeline.stage_process import (
    StageProcessSpec,
    get_stage_process_env,
)


# ---------------------------------------------------------------------------
# Test factories — registered as importable callables so ``import_string``
# inside the runner can resolve them. Live module-level (not inside a
# fixture) so dotted-path import works.
# ---------------------------------------------------------------------------


def fake_factory_sglang(
    model_path: str,
    *,
    backend: str = "local",
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
):
    """Encoder-shaped factory: accepts both backend and TP launch params."""
    return ("scheduler", model_path, backend, gpu_id, tp_rank, tp_size, nccl_port)


def fake_factory_thinker(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
):
    """Thinker-shaped factory: TP-capable but no `backend` param."""
    return ("thinker", model_path, gpu_id, tp_rank, tp_size, nccl_port)


def fake_factory_simple(model_path: str, *, gpu_id: int = 0):
    """SimpleScheduler-shaped factory: no TP params."""
    return ("simple", model_path, gpu_id)


def fake_factory_signature_auto_default(
    model_path: str,
    *,
    backend: str = "auto",  # signature default flipped to "auto"
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
):
    """Factory whose signature default for `backend` is "auto".

    Used to lock the [Backend resolution contract]: launcher decisions
    must NOT introspect the signature default. A StageConfig that omits
    ``factory_args["backend"]`` should resolve to ``"local"`` here, not
    ``"auto"``, even though the factory body would default to ``"auto"``.
    """
    return ("auto-default", model_path, backend, gpu_id, tp_rank, tp_size, nccl_port)


# Module-level dotted paths for the helpers above.
_F_SGLANG = f"{__name__}.fake_factory_sglang"
_F_THINKER = f"{__name__}.fake_factory_thinker"
_F_SIMPLE = f"{__name__}.fake_factory_simple"
_F_AUTO_DEFAULT = f"{__name__}.fake_factory_signature_auto_default"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    factory: str,
    factory_args: dict[str, Any] | None = None,
    runtime_overrides: dict[str, dict[str, Any]] | None = None,
    tp_size: int = 1,
    gpu: int | list[int] | None = 0,
) -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy/model",
        stages=[
            StageConfig(
                name="image_encoder",
                factory=factory,
                factory_args=dict(factory_args or {}),
                tp_size=tp_size,
                gpu=gpu,
                terminal=True,
            ),
        ],
        runtime_overrides=runtime_overrides or {},
        endpoints=EndpointsConfig(scheme="ipc", base_path="/tmp/encoder_tp_test"),
    )


# ---------------------------------------------------------------------------
# 1. single_visible_device flag (mp_runner._build_stage_groups)
# ---------------------------------------------------------------------------


def test_single_visible_device_flag_flips_for_sglang_backend(tmp_path):
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "sglang"},
    )
    ctx = multiprocessing.get_context("spawn")
    groups = _build_stage_groups(cfg, ctx=ctx)
    assert len(groups) == 1
    spec = groups[0].specs[0]
    assert spec.single_visible_device is True


def test_single_visible_device_flag_flips_for_auto_backend():
    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "auto"})
    ctx = multiprocessing.get_context("spawn")
    spec = _build_stage_groups(cfg, ctx=ctx)[0].specs[0]
    assert spec.single_visible_device is True


def test_single_visible_device_flag_off_for_local_backend():
    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "local"})
    ctx = multiprocessing.get_context("spawn")
    spec = _build_stage_groups(cfg, ctx=ctx)[0].specs[0]
    assert spec.single_visible_device is False


def test_single_visible_device_flag_flipped_via_runtime_overrides():
    """Locks the [Backend resolution contract]: runtime_overrides drives the flip."""
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "local"},  # default
        runtime_overrides={"image_encoder": {"backend": "sglang"}},
    )
    ctx = multiprocessing.get_context("spawn")
    spec = _build_stage_groups(cfg, ctx=ctx)[0].specs[0]
    assert spec.single_visible_device is True


def test_single_visible_device_signature_default_trap():
    """Launcher must NOT read factory signature defaults.

    Factory body defaults to ``backend="auto"`` but ``factory_args``
    omits the key — the launcher should resolve to ``"local"``.
    """
    cfg = _make_config(factory=_F_AUTO_DEFAULT, factory_args={})
    ctx = multiprocessing.get_context("spawn")
    spec = _build_stage_groups(cfg, ctx=ctx)[0].specs[0]
    assert spec.single_visible_device is False


# ---------------------------------------------------------------------------
# 2. get_stage_process_env early return
# ---------------------------------------------------------------------------


def _spec(*, tp_size: int = 1, single_visible_device: bool = False, gpu_id: int = 0):
    return StageProcessSpec(
        stage_name="image_encoder",
        tp_size=tp_size,
        gpu_id=gpu_id,
        single_visible_device=single_visible_device,
    )


def test_get_stage_process_env_returns_empty_for_plain_single_process_stage():
    assert get_stage_process_env(_spec()) == {}


def test_get_stage_process_env_remaps_when_single_visible_device_set():
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    overrides = get_stage_process_env(
        _spec(single_visible_device=True, gpu_id=4),
        env=env,
    )
    assert overrides["CUDA_VISIBLE_DEVICES"] == "4"
    assert overrides["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"


def test_get_stage_process_env_remaps_for_tp_size_2():
    """Existing tp_size>1 lane keeps working."""
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    overrides = get_stage_process_env(
        _spec(tp_size=2, gpu_id=2),
        env=env,
    )
    assert overrides["CUDA_VISIBLE_DEVICES"] == "2"


# ---------------------------------------------------------------------------
# 3. serve/launcher.py needs_mp predicate
# ---------------------------------------------------------------------------


def test_any_sglang_backend_stage_detects_sglang():
    from sglang_omni_v1.pipeline.mp_runner import any_sglang_backend_stage

    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "sglang"})
    assert any_sglang_backend_stage(cfg) is True


def test_any_sglang_backend_stage_detects_auto():
    from sglang_omni_v1.pipeline.mp_runner import any_sglang_backend_stage

    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "auto"})
    assert any_sglang_backend_stage(cfg) is True


def test_any_sglang_backend_stage_false_for_local():
    from sglang_omni_v1.pipeline.mp_runner import any_sglang_backend_stage

    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "local"})
    assert any_sglang_backend_stage(cfg) is False


def test_any_sglang_backend_stage_ignores_signature_default():
    """signature default = "auto" but factory_args is empty → resolves to local."""
    from sglang_omni_v1.pipeline.mp_runner import any_sglang_backend_stage

    cfg = _make_config(factory=_F_AUTO_DEFAULT, factory_args={})
    assert any_sglang_backend_stage(cfg) is False


# ---------------------------------------------------------------------------
# 4. compile_pipeline rejects sglang/auto + tp_size>1
# ---------------------------------------------------------------------------


def test_compile_pipeline_rejects_sglang_backend():
    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "sglang"})
    with pytest.raises(ValueError, match="MultiProcessPipelineRunner"):
        compile_pipeline(cfg)


def test_compile_pipeline_rejects_auto_backend():
    cfg = _make_config(factory=_F_SGLANG, factory_args={"backend": "auto"})
    with pytest.raises(ValueError, match="MultiProcessPipelineRunner"):
        compile_pipeline(cfg)


def test_compile_pipeline_rejects_tp_size_gt_1():
    cfg = _make_config(
        factory=_F_THINKER,
        factory_args={},
        tp_size=2,
        gpu=[0, 1],
    )
    with pytest.raises(ValueError, match="MultiProcessPipelineRunner"):
        compile_pipeline(cfg)


# ---------------------------------------------------------------------------
# 5. TP preflight Layers 1 & 2
# ---------------------------------------------------------------------------


def test_tp_preflight_layer1_rejects_factory_without_tp_params():
    cfg = _make_config(factory=_F_SIMPLE, tp_size=2, gpu=[0, 1])
    ctx = multiprocessing.get_context("spawn")
    with pytest.raises(ValueError, match="not TP-capable"):
        _build_stage_groups(cfg, ctx=ctx)


def test_tp_preflight_layer2_rejects_encoder_with_local_backend():
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "local"},
        tp_size=2,
        gpu=[0, 1],
    )
    ctx = multiprocessing.get_context("spawn")
    with pytest.raises(ValueError, match="requires backend='sglang'"):
        _build_stage_groups(cfg, ctx=ctx)


def test_tp_preflight_layer2_rejects_encoder_with_auto_backend():
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "auto"},
        tp_size=2,
        gpu=[0, 1],
    )
    ctx = multiprocessing.get_context("spawn")
    with pytest.raises(ValueError, match="requires backend='sglang'"):
        _build_stage_groups(cfg, ctx=ctx)


def test_tp_preflight_layer2_rejects_encoder_without_explicit_backend():
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={},
        tp_size=2,
        gpu=[0, 1],
    )
    ctx = multiprocessing.get_context("spawn")
    with pytest.raises(ValueError, match="requires backend='sglang'"):
        _build_stage_groups(cfg, ctx=ctx)


def test_tp_preflight_does_not_regress_thinker_tp():
    cfg = _make_config(factory=_F_THINKER, tp_size=2, gpu=[0, 1])
    ctx = multiprocessing.get_context("spawn")
    # Should succeed.
    groups = _build_stage_groups(cfg, ctx=ctx)
    assert len(groups[0].specs) == 2


def test_tp_preflight_passes_encoder_with_sglang_backend():
    cfg = _make_config(
        factory=_F_SGLANG,
        factory_args={"backend": "sglang"},
        tp_size=2,
        gpu=[0, 1],
    )
    ctx = multiprocessing.get_context("spawn")
    groups = _build_stage_groups(cfg, ctx=ctx)
    assert len(groups[0].specs) == 2
    for spec in groups[0].specs:
        assert spec.single_visible_device is True


# ---------------------------------------------------------------------------
# 6. Real-factory signature lock
# ---------------------------------------------------------------------------


def test_real_factory_signature_lock_image_encoder():
    """The production image-encoder factory's signature default must stay "local".

    Bumping it to "auto" / "sglang" would silently desync launcher
    (sees "local") from factory body. See [Backend resolution contract].
    """
    from sglang_omni_v1.models.qwen3_omni.stages import create_image_encoder_executor

    sig = inspect.signature(create_image_encoder_executor)
    assert sig.parameters["backend"].default == "local"


def test_real_factory_signature_lock_audio_encoder():
    from sglang_omni_v1.models.qwen3_omni.stages import create_audio_encoder_executor

    sig = inspect.signature(create_audio_encoder_executor)
    assert sig.parameters["backend"].default == "local"
