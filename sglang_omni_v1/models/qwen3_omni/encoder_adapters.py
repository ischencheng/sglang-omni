# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni encoder adapters between v1 :class:`PipelineState` and SGLang.

For each encoder stage there is one adapter implementing
``build_batch`` / ``run_feature`` / ``slice_results``:

- ``build_batch`` materializes a :class:`BatchPlan` from a flat list of
  :class:`IncomingMessage`. Skip requests (preprocessor-stamped
  ``_skip``) contribute no upstream items; active requests yield one or
  more :class:`MultimodalDataItem` per request, with per-request
  bookkeeping captured in :class:`RequestSpan`.
- ``run_feature`` calls the upstream encoder methods
  (``thinker.get_image_feature``, ``...get_video_feature``,
  ``...get_audio_feature``). Empty plans short-circuit and never call
  the model.
- ``slice_results`` un-batches the upstream output back to per-request
  ``encoder_outs`` dicts using the captured spans.

Cache (``EncoderRequestData.cache_key``) is intentionally not consumed
in this Phase 0 path — wiring cache into a TP-broadcasting scheduler
requires shipping the hit/miss decision before the device broadcast
(see RFC Open Question 6). The local backend keeps its cache; the
SGLang backend always forwards every non-skip request.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn.functional as F

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_omni_moe import _get_feat_extract_output_lengths

from sglang_omni_v1.models.qwen3_omni.payload_types import PipelineState
from sglang_omni_v1.models.qwen3_omni.request_builders import (
    apply_encoder_result,
    build_encoder_request,
)
from sglang_omni_v1.proto import StagePayload
from sglang_omni_v1.scheduling.messages import IncomingMessage

if TYPE_CHECKING:
    from sglang_omni_v1.models.qwen3_omni.request_builders import EncoderRequestData

logger = logging.getLogger(__name__)

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"

# Sourced from sglang_omni.models.qwen3_omni.pipeline.visual_budget — duplicated
# here to avoid pulling that module's heavyweight import chain (preprocessor,
# transformers video utils) into the lean adapter module.
QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER = 5


# ---------------------------------------------------------------------------
# BatchPlan / RequestSpan
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class RequestSpan:
    """Per-request slice of one BatchPlan.

    Exactly one of ``skip_result`` or the active counters is meaningful.
    """

    request_id: str
    skip_result: dict | None = None
    image_rows: int = 0
    video_rows: int = 0
    audio_rows: int = 0
    image_token_count: int = 0
    video_token_count: int = 0
    # Audio splitting: keep both the unpadded per-request feature lengths
    # (passed back to merge_for_thinker as audio_feature_lengths) and the
    # downsampled output lengths used to slice the encoder output along
    # the token axis.
    audio_feature_lengths: torch.Tensor | None = None
    audio_output_lengths: torch.Tensor | None = None


@dataclasses.dataclass(slots=True)
class BatchPlan:
    """Bundle passed from ``build_batch`` to ``encode_batch`` to ``slice_results``."""

    adapter: "EncoderAdapter"
    image_items: list[MultimodalDataItem]
    video_items: list[MultimodalDataItem]
    audio_items: list[MultimodalDataItem]
    spans: list[RequestSpan]

    @property
    def is_empty(self) -> bool:
        return not (self.image_items or self.video_items or self.audio_items)


class EncoderAdapter(Protocol):
    """Adapter contract.

    Implementers should keep ``build_batch`` and ``slice_results`` cheap
    (no GPU work) — they run on every rank.
    """

    stage_name: str

    def build_batch(self, messages: list[IncomingMessage]) -> BatchPlan: ...

    def run_feature(
        self, model: Any, plan: BatchPlan
    ) -> dict[str, torch.Tensor | None]: ...

    def slice_results(
        self,
        raw: dict[str, torch.Tensor | None],
        plan: BatchPlan,
        messages: list[IncomingMessage],
    ) -> list[StagePayload]: ...


# ---------------------------------------------------------------------------
# Shared helpers (moved from stages.py with one device-aware fix)
# ---------------------------------------------------------------------------


def _payload_with_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def _grid_visual_tokens(grid: Any, merge: int) -> int:
    if not isinstance(grid, torch.Tensor) or grid.numel() == 0:
        return 0
    return int((grid.to(dtype=torch.long).prod(dim=-1) // merge).sum().item())


def _split_visual_features(
    tensor: torch.Tensor | None,
    *,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[start:end]


def _split_visual_multiscale(
    tensors: list[torch.Tensor] | None,
    *,
    start: int,
    end: int,
) -> list[torch.Tensor] | None:
    if tensors is None:
        return None
    return [tensor[start:end] for tensor in tensors]


def _normalize_audio_request_tensors(
    request: "EncoderRequestData",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(features [B, mel, time], mask [B, time] bool, lengths [B] long)``.

    The synthesized fallback mask uses ``arange`` on ``lengths.device``
    so that this helper is safe in both the local v1 path (CPU) and the
    SGLang Plan B path (cuda:0 after ``_strip_and_lift``). See RFC
    "Audio adapter / mandatory device-aware fix".
    """
    input_dict = request.model_inputs
    features = input_dict["input_features"]
    if features.ndim == 2:
        features = features.unsqueeze(0)

    lengths = input_dict.get("audio_feature_lengths")
    mask = input_dict.get("feature_attention_mask")
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.to(dtype=torch.long).view(-1)
    elif isinstance(mask, torch.Tensor):
        lengths = mask.to(dtype=torch.long).sum(dim=1).view(-1)
    else:
        raise ValueError(
            "audio_feature_lengths or feature_attention_mask is required"
        )

    time_dim = features.shape[-1]
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=torch.bool)
    else:
        steps = torch.arange(
            time_dim,
            dtype=torch.long,
            device=lengths.device,
        ).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)

    return features, mask, lengths


def _pad_audio_features(features: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(features.shape[-1])
    if pad <= 0:
        return features
    return F.pad(features, (0, pad))


def _pad_audio_mask(mask: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(mask.shape[-1])
    if pad <= 0:
        return mask
    return F.pad(mask, (0, pad), value=False)


# ---------------------------------------------------------------------------
# Image / video adapter
# ---------------------------------------------------------------------------


class Qwen3OmniImageEncoderAdapter:
    """Adapter for the visual stage (images and videos)."""

    stage_name = IMAGE_STAGE

    def __init__(
        self,
        *,
        hf_config: Any,
        dtype: torch.dtype,
    ) -> None:
        thinker_cfg = getattr(hf_config, "thinker_config", hf_config)
        vision_cfg = thinker_cfg.vision_config
        self._merge = int(vision_cfg.spatial_merge_size) ** 2
        self._base_hidden = int(vision_cfg.out_hidden_size)
        self._output_layers = 1 + len(vision_cfg.deepstack_visual_indexes)
        self._dtype_bytes = torch.empty((), dtype=dtype).element_size()

    # -- request_cost_fn ------------------------------------------------

    def request_cost_fn(self, payload: StagePayload) -> int:
        """Cheap byte estimator used for batch-cost admission control.

        Mirrors the local-path arithmetic at ``stages.py:144-166``:
        ``(raw_input_bytes + token_count * base_hidden * dtype_bytes *
        output_layers) * activation_multiplier``. Reads metadata from
        the HF ``vision_config`` rather than the SGLang model wrapper to
        avoid the deepstack double-count trap (the wrapper already folds
        deepstack into ``out_hidden_size``; the local config does not).
        """
        state = PipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)
        if request.skip_result is not None:
            return 0
        inputs = request.model_inputs
        raw_bytes = _tensor_bytes(inputs.get("pixel_values"))
        raw_bytes += _tensor_bytes(inputs.get("pixel_values_videos"))
        visual_tokens = _grid_visual_tokens(inputs.get("image_grid_thw"), self._merge)
        visual_tokens += _grid_visual_tokens(
            inputs.get("video_grid_thw"), self._merge
        )
        output_bytes = (
            visual_tokens * self._base_hidden * self._dtype_bytes * self._output_layers
        )
        return (raw_bytes + output_bytes) * QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER

    # -- build / run / slice -------------------------------------------

    def build_batch(self, messages: list[IncomingMessage]) -> BatchPlan:
        images: list[MultimodalDataItem] = []
        videos: list[MultimodalDataItem] = []
        spans: list[RequestSpan] = []

        for msg in messages:
            payload = msg.data
            state = PipelineState.from_dict(payload.data)
            request = build_encoder_request(state, stage_name=IMAGE_STAGE)

            if request.skip_result is not None:
                spans.append(
                    RequestSpan(
                        request_id=msg.request_id,
                        skip_result=request.skip_result,
                    )
                )
                continue

            inputs = request.model_inputs
            n_img = n_vid = 0
            img_tokens = vid_tokens = 0
            if isinstance(inputs.get("pixel_values"), torch.Tensor):
                # _strip_and_lift moved every tensor in the payload tree to
                # cuda:0. For grid_thw that is wrong: upstream
                # ``compute_cu_seqlens_from_grid_numpy`` asserts a CPU tensor
                # (sglang/python/sglang/srt/models/utils.py:154). Move it
                # back to CPU so the upstream model never sees a CUDA grid.
                grid = inputs["image_grid_thw"].to(dtype=torch.long, device="cpu")
                it = MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=inputs["pixel_values"],
                )
                it.image_grid_thw = grid
                images.append(it)
                n_img = int(grid.shape[0])
                img_tokens = int(
                    (grid.prod(-1) // self._merge).sum().item()
                )
            if isinstance(inputs.get("pixel_values_videos"), torch.Tensor):
                grid = inputs["video_grid_thw"].to(dtype=torch.long, device="cpu")
                v = MultimodalDataItem(
                    modality=Modality.VIDEO,
                    feature=inputs["pixel_values_videos"],
                )
                v.video_grid_thw = grid
                videos.append(v)
                n_vid = int(grid.shape[0])
                vid_tokens = int(
                    (grid.prod(-1) // self._merge).sum().item()
                )
            spans.append(
                RequestSpan(
                    request_id=msg.request_id,
                    image_rows=n_img,
                    video_rows=n_vid,
                    image_token_count=img_tokens,
                    video_token_count=vid_tokens,
                )
            )
        return BatchPlan(self, images, videos, [], spans)

    def run_feature(
        self,
        model: Any,
        plan: BatchPlan,
    ) -> dict[str, torch.Tensor | None]:
        if plan.is_empty:
            return {"image": None, "video": None, "audio": None}
        thinker = getattr(model, "thinker", model)
        # get_video_feature() upstream needs the same MultimodalDataItem
        # contract that get_image_feature does: items[i].image_grid_thw
        # is read by the parent helper. Upstream Qwen3-VL uses the same
        # method name for video-grid tensors but it dispatches by
        # modality internally — we set image_grid_thw on video items too
        # so the wrapped concat works (qwen3_vl.py:1212).
        for vid in plan.video_items:
            if not hasattr(vid, "image_grid_thw") and hasattr(vid, "video_grid_thw"):
                vid.image_grid_thw = vid.video_grid_thw
        image_embed = (
            thinker.get_image_feature(plan.image_items)
            if plan.image_items
            else None
        )
        video_embed = (
            thinker.get_video_feature(plan.video_items)
            if plan.video_items
            else None
        )
        return {"image": image_embed, "video": video_embed, "audio": None}

    def slice_results(
        self,
        raw: dict[str, torch.Tensor | None],
        plan: BatchPlan,
        messages: list[IncomingMessage],
    ) -> list[StagePayload]:
        out: list[StagePayload] = []
        img_row = img_tok = vid_row = vid_tok = 0
        image_feat = raw.get("image")
        video_feat = raw.get("video")

        for span, msg in zip(plan.spans, messages):
            payload = msg.data
            state = PipelineState.from_dict(payload.data)

            if span.skip_result is not None:
                apply_encoder_result(
                    state, stage_name=IMAGE_STAGE, result=span.skip_result
                )
                out.append(_payload_with_state(payload, state))
                continue

            result: dict[str, Any] = {}
            if span.image_rows > 0 and image_feat is not None:
                tok_end = img_tok + span.image_token_count
                base, deepstack = _split_with_deepstack(
                    image_feat, img_tok, tok_end, self._base_hidden
                )
                result["image_embeds"] = base
                result["deepstack_visual_embeds_image"] = deepstack
                # image_grid_thw / token counts come from the original inputs
                request = build_encoder_request(state, stage_name=IMAGE_STAGE)
                grid = request.model_inputs.get("image_grid_thw")
                if isinstance(grid, torch.Tensor):
                    result["image_grid_thw"] = grid.to(dtype=torch.long)
                    result["image_token_counts"] = (
                        grid.to(dtype=torch.long).prod(-1) // self._merge
                    )
                img_row += span.image_rows
                img_tok = tok_end

            if span.video_rows > 0 and video_feat is not None:
                tok_end = vid_tok + span.video_token_count
                base, deepstack = _split_with_deepstack(
                    video_feat, vid_tok, tok_end, self._base_hidden
                )
                result["video_embeds"] = base
                result["deepstack_visual_embeds_video"] = deepstack
                request = build_encoder_request(state, stage_name=IMAGE_STAGE)
                grid = request.model_inputs.get("video_grid_thw")
                if isinstance(grid, torch.Tensor):
                    result["video_grid_thw"] = grid.to(dtype=torch.long)
                    result["video_token_counts"] = (
                        grid.to(dtype=torch.long).prod(-1) // self._merge
                    )
                vid_row += span.video_rows
                vid_tok = tok_end

            apply_encoder_result(state, stage_name=IMAGE_STAGE, result=result)
            out.append(_payload_with_state(payload, state))
        return out


def _split_with_deepstack(
    feature: torch.Tensor,
    start: int,
    end: int,
    base_hidden: int,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Split the upstream visual return tensor into base + deepstack parts.

    Upstream visual return tensor has last-dim
    ``vision_config.out_hidden_size * (1 + len(deepstack_visual_indexes))``
    (confirmed via ``disaggregation/encode_server.py:_infer_embedding_dims``).
    """
    sliced = feature[start:end]
    parts = list(torch.split(sliced, base_hidden, dim=-1))
    base = parts[0]
    deepstack = parts[1:] if len(parts) > 1 else None
    return base, deepstack


# ---------------------------------------------------------------------------
# Audio adapter
# ---------------------------------------------------------------------------


class Qwen3OmniAudioEncoderAdapter:
    """Adapter for the audio stage."""

    stage_name = AUDIO_STAGE

    def __init__(self, *, hf_config: Any, dtype: torch.dtype) -> None:
        del hf_config, dtype  # not currently consumed

    def request_cost_fn(self, payload: StagePayload) -> int:
        # Phase 0: no audio cost model in v1 yet. Returning 0 keeps the
        # admission shape the same as the local audio path today (no cap).
        del payload
        return 0

    def build_batch(self, messages: list[IncomingMessage]) -> BatchPlan:
        spans: list[RequestSpan] = []
        normalized: list[dict[str, torch.Tensor]] = []

        for msg in messages:
            payload = msg.data
            state = PipelineState.from_dict(payload.data)
            request = build_encoder_request(state, stage_name=AUDIO_STAGE)

            if request.skip_result is not None:
                spans.append(
                    RequestSpan(
                        request_id=msg.request_id,
                        skip_result=request.skip_result,
                    )
                )
                continue

            features, mask, lengths = _normalize_audio_request_tensors(request)
            out_lens = _get_feat_extract_output_lengths(lengths)
            spans.append(
                RequestSpan(
                    request_id=msg.request_id,
                    audio_rows=int(lengths.shape[0]),
                    audio_feature_lengths=lengths,
                    audio_output_lengths=out_lens,
                )
            )
            normalized.append(
                {"features": features, "mask": mask, "lengths": lengths}
            )

        if not normalized:
            return BatchPlan(self, [], [], [], spans)

        max_time = max(int(item["features"].shape[-1]) for item in normalized)
        audios: list[MultimodalDataItem] = []
        for item in normalized:
            feat = _pad_audio_features(item["features"], max_time)
            m = _pad_audio_mask(item["mask"], max_time)
            mm = MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=feat,
            )
            mm.feature_attention_mask = m
            audios.append(mm)
        return BatchPlan(self, [], [], audios, spans)

    def run_feature(
        self,
        model: Any,
        plan: BatchPlan,
    ) -> dict[str, torch.Tensor | None]:
        if plan.is_empty:
            return {"image": None, "video": None, "audio": None}
        thinker = getattr(model, "thinker", model)
        embed = thinker.get_audio_feature(plan.audio_items)
        return {"image": None, "video": None, "audio": embed}

    def slice_results(
        self,
        raw: dict[str, torch.Tensor | None],
        plan: BatchPlan,
        messages: list[IncomingMessage],
    ) -> list[StagePayload]:
        out: list[StagePayload] = []
        tok = 0
        audio_feat = raw.get("audio")

        for span, msg in zip(plan.spans, messages):
            payload = msg.data
            state = PipelineState.from_dict(payload.data)

            if span.skip_result is not None:
                apply_encoder_result(
                    state, stage_name=AUDIO_STAGE, result=span.skip_result
                )
                out.append(_payload_with_state(payload, state))
                continue

            assert audio_feat is not None
            assert span.audio_output_lengths is not None
            assert span.audio_feature_lengths is not None
            tok_end = tok + int(span.audio_output_lengths.sum().item())
            result = {
                "audio_embeds": audio_feat[tok:tok_end],
                # Unpadded lengths preserved at build_batch time.
                "audio_feature_lengths": span.audio_feature_lengths,
                "audio_output_lengths": span.audio_output_lengths,
            }
            apply_encoder_result(state, stage_name=AUDIO_STAGE, result=result)
            out.append(_payload_with_state(payload, state))
            tok = tok_end
        return out
