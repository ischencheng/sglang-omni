# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import gc
import queue
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

from sglang_omni.models.moss_transcribe_diarize import encoder_service
from sglang_omni.models.moss_transcribe_diarize.encoder_service import (
    BatchedAudioEncoderService,
)
from sglang_omni.scheduling.pre_lm_encoder import QueueEntry
from sglang_omni.scheduling.stage_cache import StageOutputCache


def test_drain_batch_respects_gpu_microbatch_limit() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service.max_batch_size = 2
    service.queue = queue.Queue()
    entries = [
        QueueEntry(object(), concurrent.futures.Future())
        for _ in range(service.max_batch_size + 2)
    ]
    for entry in entries:
        service.queue.put(entry)

    assert service.drain_batch() == entries[:2]
    assert service.queue.qsize() == 2


def test_encoder_microbatch_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
        BatchedAudioEncoderService(object(), max_batch_size=0)


class FailingStream:
    def synchronize(self) -> None:
        raise torch.OutOfMemoryError("test encoder OOM")


class EncoderIntermediate:
    pass


class StopWorker(BaseException):
    pass


def test_encode_batch_commits_item_state_only_after_stream_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service.stream = FailingStream()
    service.dtype = torch.float32
    service.hidden_size = 3
    service.model = SimpleNamespace(
        get_audio_feature_uncached=lambda items, forward_batch: torch.ones(2, 3)
    )
    monkeypatch.setattr(
        encoder_service.torch.cuda,
        "stream",
        lambda stream: contextlib.nullcontext(),
    )
    features = [torch.ones(1), torch.ones(1)]
    items = [
        MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=feature,
            model_specific_data={"audio_feature_lengths": torch.tensor([1])},
        )
        for feature in features
    ]

    with pytest.raises(torch.OutOfMemoryError, match="test encoder OOM"):
        service.execute_batch(items)

    for item, feature in zip(items, features):
        assert item.feature is feature
        assert item.precomputed_embeddings is None


def test_singleton_oom_is_request_scoped_and_worker_processes_next_item(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service.max_batch_size = 1
    service.queue = queue.Queue()
    service.worker_state_lock = threading.Lock()
    service.worker_error = None
    service.lifecycle_lock = threading.Lock()
    service.closed = False
    service.batch_count = 0
    service.item_count = 0
    service.device = "cuda:7"
    cleanup_steps: list[str] = []
    selected_devices: list[str] = []
    calls: list[list[object]] = []
    retained_intermediates: list[weakref.ReferenceType[EncoderIntermediate]] = []
    poisoned = False
    failed_item = MultimodalDataItem(modality=Modality.AUDIO, feature=object())
    healthy_item = MultimodalDataItem(modality=Modality.AUDIO, feature=object())
    stop_item = object()

    def execute_batch(items: list[object]) -> list[object]:
        nonlocal poisoned
        if items == [stop_item]:
            raise StopWorker
        calls.append(items)
        if items == [failed_item]:
            intermediate = EncoderIntermediate()
            retained_intermediates.append(weakref.ref(intermediate))
            poisoned = True
            raise torch.OutOfMemoryError("test encoder OOM")
        if poisoned:
            raise RuntimeError("allocator remained poisoned after OOM")
        items[0].precomputed_embeddings = object()
        items[0].feature = None
        return [items[0].precomputed_embeddings]

    def cuda_device(device: str) -> contextlib.AbstractContextManager:
        selected_devices.append(device)
        return contextlib.nullcontext()

    def empty_cache() -> None:
        nonlocal poisoned
        cleanup_steps.append("empty_cache")
        poisoned = False

    service.stream = SimpleNamespace(
        synchronize=lambda: cleanup_steps.append("synchronize")
    )
    monkeypatch.setattr(service, "execute_batch", execute_batch)
    monkeypatch.setattr(encoder_service.torch.cuda, "device", cuda_device)
    monkeypatch.setattr(encoder_service.torch.cuda, "empty_cache", empty_cache)

    def run_worker() -> None:
        try:
            service.worker()
        except StopWorker:
            pass

    service.thread = threading.Thread(target=run_worker, daemon=True)
    service.thread.start()

    try:
        with pytest.raises(torch.OutOfMemoryError, match="test encoder OOM"):
            service.encode_item(failed_item)
        service.encode_item(healthy_item)
    finally:
        service.queue.put(QueueEntry(stop_item, concurrent.futures.Future()))
        service.thread.join(timeout=1)

    gc.collect()
    assert not service.thread.is_alive()
    assert calls == [[failed_item], [healthy_item]]
    assert cleanup_steps == ["synchronize", "empty_cache"]
    assert selected_devices == ["cuda:7"]
    assert healthy_item.feature is None
    assert service.batch_count == 1
    assert service.item_count == 1
    assert retained_intermediates[0]() is None
    assert all(record.exc_info is None for record in caplog.records)
    assert all(
        not any(isinstance(arg, BaseException) for arg in record.args)
        for record in caplog.records
    )


def test_batched_oom_falls_back_to_per_item_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service.batch_count = 0
    service.item_count = 0
    service.worker_state_lock = threading.Lock()
    service.worker_error = None
    service.device = "cuda:5"
    cleanup_steps: list[str] = []
    selected_devices: list[str] = []
    service.stream = SimpleNamespace(
        synchronize=lambda: cleanup_steps.append("synchronize")
    )
    poisoned = False
    calls: list[list[object]] = []
    items = [object(), object()]

    def execute_batch(batch: list[object]) -> list[object]:
        nonlocal poisoned
        calls.append(batch)
        if len(batch) > 1:
            poisoned = True
            raise torch.OutOfMemoryError("aggregate batch is too large")
        if poisoned:
            raise RuntimeError("allocator remained poisoned after OOM")
        return [object()]

    def cuda_device(device: str) -> contextlib.AbstractContextManager:
        selected_devices.append(device)
        return contextlib.nullcontext()

    def empty_cache() -> None:
        nonlocal poisoned
        cleanup_steps.append("empty_cache")
        poisoned = False

    monkeypatch.setattr(service, "execute_batch", execute_batch)
    monkeypatch.setattr(encoder_service.torch.cuda, "device", cuda_device)
    monkeypatch.setattr(encoder_service.torch.cuda, "empty_cache", empty_cache)
    entries = [QueueEntry(item, concurrent.futures.Future()) for item in items]
    batches = iter([(entries, False), ([], True)])
    service.next_batch = lambda: next(batches)

    service.worker()

    assert all(entry.future.result() is not None for entry in entries)
    assert calls == [items, [items[0]], [items[1]]]
    assert service.batch_count == 2
    assert cleanup_steps == ["synchronize", "empty_cache"]
    assert selected_devices == ["cuda:5"]
    assert service.item_count == 2


def test_non_oom_failure_logs_traceback_without_retaining_exception_state(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service.worker_state_lock = threading.Lock()
    service.worker_error = None
    retained_intermediates: list[weakref.ReferenceType[EncoderIntermediate]] = []

    def raise_non_oom_encoder_failure(_items: list[object]) -> list[object]:
        intermediate = EncoderIntermediate()
        retained_intermediates.append(weakref.ref(intermediate))
        raise ValueError("unexpected encoder shape")

    monkeypatch.setattr(service, "execute_batch", raise_non_oom_encoder_failure)
    entry = QueueEntry(object(), concurrent.futures.Future())
    batches = iter([([entry], False), ([], True)])
    service.next_batch = lambda: next(batches)

    service.worker()

    failure = entry.future.exception()
    assert isinstance(failure, ValueError)
    assert failure.__traceback__ is None
    assert failure.__cause__ is None
    assert failure.__context__ is None

    gc.collect()
    assert retained_intermediates[0]() is None
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "Traceback (most recent call last):" in message
    assert "raise_non_oom_encoder_failure" in message
    assert "ValueError: unexpected encoder shape" in message
    assert all(record.exc_info is None for record in caplog.records)
    assert all(
        not any(isinstance(arg, BaseException) for arg in record.args)
        for record in caplog.records
    )


def test_encode_item_rechecks_cache_after_preprocessing() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service.device = torch.device("cpu")
    service.lifecycle_lock = threading.Lock()
    service.closed = False
    service.dtype = torch.float32
    service.hidden_size = 3
    service.cache = StageOutputCache(max_size=4, max_bytes=1024, cache_device="cpu")
    cached = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    service.cache.put("fingerprint", cached)
    item = MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=7,
        feature=object(),
        model_specific_data={
            "audio_fingerprint": "fingerprint",
            "audio_feature_lengths": torch.tensor([2]),
        },
    )
    service.submit = lambda item: pytest.fail("cached item must not be submitted")

    service.encode_item(item)

    assert torch.equal(item.precomputed_embeddings, cached)
    assert item.feature is None


@pytest.mark.parametrize(
    "cached",
    [
        torch.ones(3, 3),
        torch.ones(2, 4),
        torch.ones(2, 3, dtype=torch.float64),
    ],
)
def test_lookup_cached_embedding_evicts_invalid_entries(cached: torch.Tensor) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service.dtype = torch.float32
    service.hidden_size = 3
    service.cache = StageOutputCache(max_size=4, max_bytes=1024, cache_device="cpu")
    service.cache.put("fingerprint", cached)

    assert service.lookup_cached_embedding("fingerprint", 2) is None
    assert len(service.cache) == 0


def test_lookup_cached_embedding_returns_valid_entry() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service.dtype = torch.float32
    service.hidden_size = 3
    service.cache = StageOutputCache(max_size=4, max_bytes=1024, cache_device="cpu")
    cached = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    service.cache.put("fingerprint", cached)

    result = service.lookup_cached_embedding("fingerprint", 2)

    assert result is not None
    assert torch.equal(result, cached)


def test_batch_failure_retries_moss_items_with_failure_isolation() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service.batch_count = 0
    service.item_count = 0
    service.worker_state_lock = threading.Lock()
    service.worker_error = None
    service.device = torch.device("cpu")
    service.dtype = torch.float32
    service.hidden_size = 3
    service.cache = StageOutputCache(max_size=4, max_bytes=1024, cache_device="cpu")
    synchronized: list[None] = []
    service.stream = SimpleNamespace(synchronize=lambda: synchronized.append(None))

    good = MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=1,
        feature=object(),
        model_specific_data={
            "audio_fingerprint": "good",
            "audio_feature_lengths": torch.tensor([2]),
            "fail": False,
        },
    )
    bad = MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=2,
        feature=object(),
        model_specific_data={
            "audio_fingerprint": "bad",
            "audio_feature_lengths": torch.tensor([1]),
            "fail": True,
        },
    )

    def encode(items, _unused):  # noqa: ANN001, ANN202
        if len(items) > 1:
            raise RuntimeError("batch failed")
        if items[0].fail:
            raise RuntimeError("item failed")
        rows = int(items[0].audio_feature_lengths.sum())
        return torch.ones(rows, 3)

    service.model = SimpleNamespace(get_audio_feature_uncached=encode)
    service.batch_context = contextlib.nullcontext
    good_entry = QueueEntry(good, concurrent.futures.Future())
    bad_entry = QueueEntry(bad, concurrent.futures.Future())
    batches = iter([([good_entry, bad_entry], False), ([], True)])
    service.next_batch = lambda: next(batches)

    service.worker()

    assert torch.equal(good_entry.future.result(timeout=0), good.precomputed_embeddings)
    with pytest.raises(RuntimeError, match="item failed"):
        bad_entry.future.result(timeout=0)
    assert good.precomputed_embeddings.shape == (2, 3)
    assert good.feature is None
    assert bad.precomputed_embeddings is None
    assert torch.equal(service.cache.get("good"), good.precomputed_embeddings)
    assert service.cache.get("bad") is None
    assert len(synchronized) == 1


@pytest.fixture
def controlled_service():  # noqa: ANN201
    service = object.__new__(BatchedAudioEncoderService)
    service.device = torch.device("cpu")
    service.dtype = torch.float32
    service.hidden_size = 3
    service.cache = StageOutputCache(max_size=4, max_bytes=1024, cache_device="cpu")
    service.lock = threading.Lock()
    service.lifecycle_lock = threading.Lock()
    service.closed = False
    service.inflight = {}
    service.ENCODE_TIMEOUT_S = 2
    submitted: queue.Queue[QueueEntry[MultimodalDataItem]] = queue.Queue()
    waiting: queue.Queue[None] = queue.Queue()

    def submit(
        item: MultimodalDataItem,
        future: concurrent.futures.Future[torch.Tensor] | None = None,
    ) -> concurrent.futures.Future[torch.Tensor]:
        if future is None:
            future = concurrent.futures.Future()
        original_result = future.result

        def result(timeout: float | None = None) -> torch.Tensor:
            waiting.put(None)
            return original_result(timeout)

        future.result = result
        submitted.put(QueueEntry(item, future))
        return future

    service.submit = submit
    return service, submitted, waiting


def audio_item(fingerprint: str | None = "clip", tokens: int = 2) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=7,
        feature=torch.ones(1),
        model_specific_data={
            "audio_fingerprint": fingerprint,
            "audio_feature_lengths": torch.tensor([tokens]),
        },
    )


def finish_encoding(
    service: BatchedAudioEncoderService,
    entry: QueueEntry[MultimodalDataItem],
) -> torch.Tensor:
    embedding = torch.ones(int(entry.item.audio_feature_lengths.sum()), 3)
    service.attach_embedding(entry.item, embedding)
    service.cache_embedding(entry.item, embedding)
    entry.future.set_result(embedding)
    return embedding


def test_cold_requests_share_one_encode_without_blocking_other_keys(
    controlled_service,
) -> None:
    service, submitted, waiting = controlled_service
    items = [audio_item() for _ in range(4)] + [audio_item("other")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        calls = [pool.submit(service.encode_item, item) for item in items]
        for _ in items:
            waiting.get(timeout=1)
        entries = [submitted.get_nowait(), submitted.get_nowait()]
        assert submitted.empty()
        assert {entry.item.audio_fingerprint for entry in entries} == {"clip", "other"}
        for entry in entries:
            finish_encoding(service, entry)
        for call in calls:
            assert call.result(timeout=1) is None

    assert not service.inflight
    assert all(item.feature is None for item in items)
    assert all(
        item.precomputed_embeddings is items[0].precomputed_embeddings
        for item in items[:4]
    )
    assert items[4].precomputed_embeddings is not items[0].precomputed_embeddings


def test_cache_fill_between_initial_miss_and_leader_creation(
    controlled_service, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, submitted, waiting = controlled_service
    original_lookup = (
        service._lookup_cached_embedding
    )  # noqa: leading-underscore  # inject the cache-publication race
    embedding = torch.ones(2, 3)

    def lookup(key: str, expected_tokens: int) -> torch.Tensor | None:
        cached = original_lookup(key, expected_tokens)
        service.cache.put(key, embedding)
        return cached

    monkeypatch.setattr(service, "_lookup_cached_embedding", lookup)
    item = audio_item()
    service.encode_item(item)
    assert torch.equal(item.precomputed_embeddings, embedding)
    assert submitted.empty()
    assert waiting.empty()


def test_failed_encode_reaches_all_waiters_and_next_request_retries(
    controlled_service,
) -> None:
    service, submitted, waiting = controlled_service
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(service.encode_item, audio_item()) for _ in range(2)]
        for _ in calls:
            waiting.get(timeout=1)
        entry = submitted.get_nowait()
        assert submitted.empty()
        entry.future.set_exception(ValueError("encoder failed"))
        for call in calls:
            with pytest.raises(ValueError, match="encoder failed"):
                call.result(timeout=1)
        retry_item = audio_item()
        retry = pool.submit(service.encode_item, retry_item)
        waiting.get(timeout=1)
        finish_encoding(service, submitted.get_nowait())
        retry.result(timeout=1)
    assert retry_item.feature is None
    assert not service.inflight


def test_follower_timeout_does_not_cancel_shared_encoding(controlled_service) -> None:
    service, submitted, waiting = controlled_service
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(service.encode_item, audio_item())
        waiting.get(timeout=1)
        entry = submitted.get_nowait()
        service.ENCODE_TIMEOUT_S = 0.01
        follower_item = audio_item()
        follower = pool.submit(service.encode_item, follower_item)
        with pytest.raises(TimeoutError):
            follower.result(timeout=1)
        assert not entry.future.cancelled()
        assert follower_item.feature is not None
        finish_encoding(service, entry)
        leader.result(timeout=1)
    assert submitted.empty()
    assert not service.inflight


def test_submission_failure_does_not_leave_a_stuck_leader(
    controlled_service, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, submitted, waiting = controlled_service
    original_submit = service.submit

    def reject(
        item: MultimodalDataItem,
        future: concurrent.futures.Future[torch.Tensor],
    ) -> concurrent.futures.Future[torch.Tensor]:
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(service, "submit", reject)
    with pytest.raises(RuntimeError, match="worker unavailable"):
        service.encode_item(audio_item())
    assert not service.inflight
    monkeypatch.setattr(service, "submit", original_submit)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        retry = pool.submit(service.encode_item, audio_item())
        waiting.get(timeout=1)
        finish_encoding(service, submitted.get_nowait())
        retry.result(timeout=1)


@pytest.mark.asyncio
async def test_cancelled_follower_does_not_cancel_leader_or_other_follower(
    controlled_service,
) -> None:
    service, submitted, waiting = controlled_service
    items = [audio_item() for _ in range(3)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        calls = [pool.submit(service.encode_item, items[0])]
        waiting.get(timeout=1)
        entry = submitted.get_nowait()
        calls.extend(pool.submit(service.encode_item, item) for item in items[1:])
        for _ in items[1:]:
            waiting.get(timeout=1)
        cancelled = asyncio.wrap_future(calls[1])
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert not entry.future.cancelled()
        finish_encoding(service, entry)
        for call in calls:
            call.result(timeout=1)
    assert submitted.empty()
    assert all(item.feature is None for item in items)


@pytest.mark.parametrize("fingerprint", [None, "clip"])
def test_missing_fingerprint_or_different_token_count_does_not_merge(
    controlled_service, fingerprint: str | None
) -> None:
    service, submitted, waiting = controlled_service
    items = [audio_item(fingerprint, tokens) for tokens in (1, 2)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(service.encode_item, item) for item in items]
        for _ in calls:
            waiting.get(timeout=1)
        for _ in calls:
            finish_encoding(service, submitted.get_nowait())
        for call in calls:
            call.result(timeout=1)
    assert [item.precomputed_embeddings.shape[0] for item in items] == [1, 2]
    if fingerprint is None:
        assert len(service.cache) == 0


def test_stale_completion_cannot_remove_replacement_encode(
    controlled_service, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, submitted, waiting = controlled_service
    callback_started = threading.Event()
    release_callback = threading.Event()
    original_clear = service.clear_inflight
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(service.encode_item, audio_item())
        waiting.get(timeout=1)
        old = submitted.get_nowait()

        def delayed_clear(
            key: tuple[str, int], future: concurrent.futures.Future[torch.Tensor]
        ) -> None:
            if future is old.future:
                callback_started.set()
                assert release_callback.wait(timeout=2)
            original_clear(key, future)

        monkeypatch.setattr(service, "clear_inflight", delayed_clear)
        completion = pool.submit(
            old.future.set_exception, ValueError("old generation failed")
        )
        assert callback_started.wait(timeout=1)
        with pytest.raises(ValueError, match="old generation failed"):
            first.result(timeout=1)
        second = pool.submit(service.encode_item, audio_item())
        waiting.get(timeout=1)
        replacement = submitted.get_nowait()
        release_callback.set()
        completion.result(timeout=1)
        follower = pool.submit(service.encode_item, audio_item())
        waiting.get(timeout=1)
        assert submitted.empty()
        finish_encoding(service, replacement)
        second.result(timeout=1)
        follower.result(timeout=1)
    assert not service.inflight


@pytest.mark.parametrize(
    "embedding",
    [torch.ones(1, 3), torch.ones(2, 4), torch.ones(2, 3, dtype=torch.float64)],
)
def test_invalid_encoded_shape_or_dtype_is_not_attached_or_cached(
    controlled_service, embedding: torch.Tensor
) -> None:
    service, _, _ = controlled_service
    service.model = SimpleNamespace(
        get_audio_feature_uncached=lambda items, forward_batch: embedding
    )
    service.batch_context = contextlib.nullcontext
    service.stream = SimpleNamespace(synchronize=lambda: None)
    item = audio_item()
    with pytest.raises(RuntimeError, match="encoder output"):
        service.execute_batch([item])
    assert item.feature is not None
    assert item.precomputed_embeddings is None
    assert len(service.cache) == 0


def test_close_drains_pending_work_and_rejects_later_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def encode(items: list[MultimodalDataItem], forward_batch: None) -> torch.Tensor:
        started.set()
        assert release.wait(timeout=2)
        return torch.ones(
            sum(int(item.audio_feature_lengths.sum()) for item in items), 3
        )

    model = SimpleNamespace(
        whisper_encoder=torch.nn.Linear(3, 3),
        vq_adaptor=torch.nn.Linear(3, 3),
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=3)),
        get_audio_feature_uncached=encode,
    )
    monkeypatch.setattr(
        encoder_service.torch.cuda,
        "Stream",
        lambda device: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(
        encoder_service.torch.cuda, "stream", lambda stream: contextlib.nullcontext()
    )
    service = BatchedAudioEncoderService(model)
    service.ENCODE_TIMEOUT_S = 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(service.encode_item, audio_item())
        assert started.wait(timeout=1)
        pending = service.submit(audio_item("queued"))
        closing = pool.submit(service.close)
        while not service.closed:
            threading.Event().wait(0.001)
        with pytest.raises(RuntimeError, match="service is closed"):
            service.encode_item(audio_item())
        release.set()
        first.result(timeout=1)
        pending.result(timeout=1)
        closing.result(timeout=1)
    service.close()
    assert not service.thread.is_alive()
    assert not service.inflight
    assert service.item_count == 2

    model.get_audio_feature_uncached = lambda items, forward_batch: torch.full(
        (2, 3), 2.0
    )
    replacement = BatchedAudioEncoderService(model)
    replacement.ENCODE_TIMEOUT_S = 2
    try:
        item = audio_item()
        replacement.encode_item(item)
        assert torch.equal(item.precomputed_embeddings, torch.full((2, 3), 2.0))
        assert replacement.item_count == 1
    finally:
        replacement.close()
