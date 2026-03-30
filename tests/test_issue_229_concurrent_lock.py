# SPDX-License-Identifier: Apache-2.0
"""Regression test for issue #229: concurrent requests cause CUDA illegal memory access.

Root cause: _CodePredictorStreamingExecutor and _Code2WavStreamingExecutor both call
run_in_executor on shared model instances without any lock, allowing two threads to
run GPU inference simultaneously.

This test uses CPU-only mocks to detect the race condition without real GPUs:
- A fake model sleeps briefly inside forward() to make overlap detectable.
- Two requests are submitted concurrently.
- On main (unfixed): the model is called from two threads simultaneously -> bug detected.
- On fix branch:   a lock serializes calls  -> no overlap -> test passes.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.proto.request import OmniRequest, StagePayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="test"),
        data=None,
    )


def _make_stream_item(request_id: str, chunk_id: int, data: Any, metadata: dict | None = None) -> StreamItem:
    return StreamItem(chunk_id=chunk_id, data=data, from_stage="talker_ar", metadata=metadata)


class _ConcurrencyDetector:
    """Tracks whether a callable was ever entered by two threads simultaneously."""

    def __init__(self, sleep_s: float = 0.05):
        self._sleep_s = sleep_s
        self._lock = threading.Lock()
        self._inside = 0
        self.max_concurrent = 0

    def __enter__(self):
        with self._lock:
            self._inside += 1
            if self._inside > self.max_concurrent:
                self.max_concurrent = self._inside
        time.sleep(self._sleep_s)  # yield so the other thread can enter

    def __exit__(self, *_):
        with self._lock:
            self._inside -= 1


# ---------------------------------------------------------------------------
# Test: _CodePredictorStreamingExecutor
# ---------------------------------------------------------------------------

class _FakeCodePredictorModel(nn.Module):
    """Fake code predictor that records concurrent access."""

    def __init__(self, detector: _ConcurrencyDetector):
        super().__init__()
        self._detector = detector

    def forward(self, talker_hidden: torch.Tensor, layer0_code: torch.Tensor):
        with self._detector:
            pass
        return {
            "codes": torch.zeros(16, dtype=torch.long),
            "summed_embeddings": torch.zeros(talker_hidden.shape[-1]),
        }


def test_code_predictor_no_concurrent_gpu_access() -> None:
    """Two concurrent requests must never call the model simultaneously."""
    from sglang_omni.models.qwen3_omni.components.code_predictor_executor import (
        _CodePredictorStreamingExecutor,
    )

    detector = _ConcurrencyDetector(sleep_s=0.05)
    model = _FakeCodePredictorModel(detector)
    executor = _CodePredictorStreamingExecutor(model=model, device="cpu")

    # Wire up stream infrastructure
    stream_queue = StreamQueue()
    emitted: list[Any] = []

    def stream_fn(request_id, data, target_stage, metadata=None):
        emitted.append((request_id, target_stage))

    executor._stream_queue = stream_queue
    executor._stream_fn = stream_fn

    NUM_CHUNKS = 3

    async def run():
        # Open queues for both requests
        for rid in ["req-A", "req-B"]:
            stream_queue.open(rid)

        # Feed chunks for both requests
        for rid in ["req-A", "req-B"]:
            for i in range(NUM_CHUNKS):
                stream_queue.put(
                    rid,
                    _make_stream_item(
                        rid,
                        chunk_id=i,
                        data=torch.zeros(64),
                        metadata={"codec_code": i},
                    ),
                )
            stream_queue.put_done(rid)

        payload_a = _make_payload("req-A")
        payload_b = _make_payload("req-B")

        # Submit both requests concurrently
        await asyncio.gather(
            executor.add_request(payload_a),
            executor.add_request(payload_b),
        )

    asyncio.run(run())

    assert detector.max_concurrent == 1, (
        f"BUG: model was called from {detector.max_concurrent} threads simultaneously "
        f"(expected 1). This is the race condition reported in issue #229."
    )
    # Each request produces NUM_CHUNKS * 2 stream events (codes + embeddings)
    assert len(emitted) == NUM_CHUNKS * 2 * 2


# ---------------------------------------------------------------------------
# Test: _Code2WavStreamingExecutor
# ---------------------------------------------------------------------------

class _FakeCode2WavModel(nn.Module):
    """Fake codec decoder that records concurrent access."""

    total_upsample = 1

    def __init__(self, detector: _ConcurrencyDetector):
        super().__init__()
        self._detector = detector

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        with self._detector:
            pass
        # Return dummy waveform: (batch=1, channels=1, samples)
        num_tokens = codes.shape[-1]
        return torch.zeros(1, 1, num_tokens * self.total_upsample)


def test_code2wav_no_concurrent_gpu_access() -> None:
    """Two concurrent requests must never call the decoder simultaneously.

    On CPU, _decode_async short-circuits run_in_executor, so we subclass to force
    the GPU code path (always run through the thread pool) for testing purposes.
    """
    from sglang_omni.models.qwen3_omni.components.code2wav_executor import (
        _Code2WavStreamingExecutor,
    )

    class _ForceThreadPoolExecutor(_Code2WavStreamingExecutor):
        """Skip the CPU shortcut to simulate GPU threading path.

        If _gpu_lock is present (fix branch), use it; otherwise run without lock
        (main branch) so the race condition is still detectable.
        """

        async def _decode_async(self, loop, code_chunks, start_index, end_index):
            lock = getattr(self, "_gpu_lock", None)
            if lock is not None:
                async with lock:
                    return await loop.run_in_executor(
                        None, self._decode_incremental, code_chunks, start_index, end_index
                    )
            return await loop.run_in_executor(
                None, self._decode_incremental, code_chunks, start_index, end_index
            )

    detector = _ConcurrencyDetector(sleep_s=0.05)
    model = _FakeCode2WavModel(detector)

    # stream_chunk_size=1 so every chunk triggers a decode immediately
    executor = _ForceThreadPoolExecutor(
        model,
        device="cpu",
        stream_chunk_size=1,
        left_context_size=0,
    )

    stream_queue = StreamQueue()
    executor._stream_queue = stream_queue

    NUM_CHUNKS = 3

    async def run():
        for rid in ["req-A", "req-B"]:
            stream_queue.open(rid)

        payload_a = _make_payload("req-A")
        payload_b = _make_payload("req-B")

        # add_request launches background tasks; feed data after
        await executor.add_request(payload_a)
        await executor.add_request(payload_b)

        # Feed codec codes (layer-0 code = 1, not EOS=2150)
        for rid in ["req-A", "req-B"]:
            for i in range(NUM_CHUNKS):
                codes = torch.ones(16, dtype=torch.long)  # 16 code groups
                stream_queue.put(rid, _make_stream_item(rid, chunk_id=i, data=codes))
            stream_queue.put_done(rid)

        # Collect both results
        result_a = await executor.get_result()
        result_b = await executor.get_result()
        return result_a, result_b

    asyncio.run(run())

    assert detector.max_concurrent == 1, (
        f"BUG: decoder was called from {detector.max_concurrent} threads simultaneously "
        f"(expected 1). This is the race condition reported in issue #229."
    )
