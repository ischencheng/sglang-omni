# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import gc
import queue
import threading
import time
import weakref
from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

from sglang_omni.models.qwen3_asr.deferred_cache import (
    CachePublication as DeferredPublication,
)
from sglang_omni.models.qwen3_asr.deferred_cache import DeferredEmbeddingCache
from sglang_omni.models.qwen3_asr.encoder_service import (
    Qwen3ASRPreLMEncoderService,
    build_cache_namespace,
    expected_audio_tokens,
)
from sglang_omni.scheduling.stage_cache import StageOutputCache

_HIDDEN_SIZE = 4
if torch.cuda.is_available():
    _DEVICE = "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    _DEVICE = "xpu"
else:
    _DEVICE = "cpu"
requires_accelerator = pytest.mark.skipif(
    _DEVICE == "cpu",
    reason="requires cuda or xpu",
)
_NAMESPACE = "testns"
_SERVICES: list[Qwen3ASRPreLMEncoderService] = []


@pytest.fixture(autouse=True)
def _close_services() -> Iterator[None]:
    yield
    for service in _SERVICES:
        service.close()
    _SERVICES.clear()


class _StubModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.audio_tower = torch.nn.Linear(2, 2).to(dtype)
        self.config = SimpleNamespace(
            thinker_config=SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE)
            )
        )
        self.dtype = dtype
        self.encode_calls = 0
        self.encode_batch_sizes: list[int] = []
        self.fail = False
        self.fail_oom = False
        self.fail_multi_item = False
        self.packed_3d_output = True
        self.encode_gate: threading.Event | None = None
        self.encode_started: threading.Event | None = None
        self.row_offset = 0
        self.encode_delay_s = 0.0
        self.grad_enabled_during_encode: bool | None = None

    def get_audio_feature(self, items):  # noqa: ANN001
        self.grad_enabled_during_encode = torch.is_grad_enabled()
        self.encode_calls += 1
        self.encode_batch_sizes.append(len(items))
        if self.encode_started is not None:
            self.encode_started.set()
        gate = self.encode_gate
        if gate is not None:
            self.encode_gate = None
            gate.wait(timeout=10)
        if self.encode_delay_s:
            time.sleep(self.encode_delay_s)
        if self.fail_oom:
            raise torch.OutOfMemoryError("encoder OOM")
        if self.fail:
            raise RuntimeError("boom")
        if self.fail_multi_item and len(items) > 1:
            raise RuntimeError("multi-item boom")
        parts = []
        for item in items:
            rows = expected_audio_tokens(item) + self.row_offset
            fill = float((getattr(item, "hash", None) or 0) % 97 + 1)
            parts.append(torch.full((rows, _HIDDEN_SIZE), fill, dtype=self.dtype))
        packed = torch.cat(parts, dim=0)
        if self.packed_3d_output:
            # Mirrors the real audio tower: one packed frame stream in, so
            # last_hidden_state is [1, total_tokens, hidden].
            return packed.unsqueeze(0)
        return packed


def _make_service(
    model: _StubModel | None = None,
    *,
    cache_max_entries: int = 16,
    cache_max_bytes: int = 1 << 20,
    max_batch_size: int = 8,
    defer_cache_copy: bool = False,
) -> Qwen3ASRPreLMEncoderService:
    service = Qwen3ASRPreLMEncoderService(
        model or _StubModel(),
        cache_namespace=_NAMESPACE,
        cache_max_entries=cache_max_entries,
        cache_max_bytes=cache_max_bytes,
        max_batch_size=max_batch_size,
        defer_cache_copy=defer_cache_copy,
        cache_pending_max_entries=8,
        cache_pending_max_bytes=1 << 20,
    )
    _SERVICES.append(service)
    return service


def _item(
    audio_hash: int | None,
    num_audio_tokens: int,
    *,
    with_feature: bool = True,
) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=audio_hash,
        feature=torch.zeros(1, 128, 300) if with_feature else None,
        model_specific_data={
            "audio_fingerprint": str(audio_hash) if audio_hash is not None else None,
            "num_audio_tokens": num_audio_tokens,
        },
    )


@dataclass(kw_only=True)
class CachePublication:
    key: str
    embedding: torch.Tensor
    completion: Future[None]


class ControlledDeferredCache:
    def __init__(self, cache: StageOutputCache) -> None:
        self.cache = cache
        self.max_pending_bytes = 1 << 20
        self.submitted: queue.Queue[CachePublication] = queue.Queue()
        self.publications: list[CachePublication] = []
        self.submit_gate = threading.Event()
        self.submit_gate.set()
        self.close_gate = threading.Event()
        self.close_gate.set()
        self.close_started = threading.Event()
        self.publish_during_submit = False

    def submit(
        self, key: str, embedding: torch.Tensor, completion: Future[None]
    ) -> None:
        publication = CachePublication(
            key=key, embedding=embedding.detach().clone(), completion=completion
        )
        self.publications.append(publication)
        if self.publish_during_submit:
            self.publish(publication)
        else:
            pass
        self.submitted.put(publication)
        assert self.submit_gate.wait(timeout=10)

    def publish(self, publication: CachePublication) -> None:
        self.cache.put(publication.key, publication.embedding)
        publication.completion.set_result(None)

    def close(self) -> None:
        self.close_started.set()
        assert self.close_gate.wait(timeout=10)
        for publication in self.publications:
            if not publication.completion.done():
                self.publish(publication)
            else:
                pass

    def stats(self) -> dict[str, int]:
        return {}


class FakeCopyStream:
    def __init__(self, device: torch.device) -> None:
        self.device = device


def test_deferred_cache_opt_in_preserves_synchronous_cpu_cache() -> None:
    service = _make_service(defer_cache_copy=True)
    audio_item = _item(123, 3)

    service.submit_item(audio_item).result(timeout=2)

    assert service.deferred_cache is None
    assert torch.equal(
        service.lookup_cached_embedding("123", 3), audio_item.precomputed_embeddings
    )


@pytest.mark.parametrize(
    ("max_entries", "max_bytes"), [(1, 1 << 20), (8, 3 * _HIDDEN_SIZE * 4)]
)
def test_deferred_cache_bounds_include_active_publication(
    monkeypatch: pytest.MonkeyPatch, max_entries: int, max_bytes: int
) -> None:
    monkeypatch.setattr(torch.cuda, "Stream", FakeCopyStream)
    cache = StageOutputCache(cache_device="cpu")
    publisher = DeferredEmbeddingCache(
        cache,
        device=torch.device("cpu"),
        max_pending_entries=max_entries,
        max_pending_bytes=max_bytes,
    )
    publication_started = threading.Event()
    release_publication = threading.Event()
    second_started = threading.Event()
    second_accepted = threading.Event()
    embedding = torch.ones(3, _HIDDEN_SIZE)
    first_completion: Future[None] = Future()
    second_completion: Future[None] = Future()

    def controlled_publish(publication: DeferredPublication) -> None:
        publication_started.set()
        assert release_publication.wait(timeout=10)
        cache.put(publication.key, publication.embedding)

    def submit_second() -> None:
        second_started.set()
        publisher.submit("second", embedding, second_completion)
        second_accepted.set()

    monkeypatch.setattr(publisher, "publish", controlled_publish)
    submit_thread = threading.Thread(target=submit_second)
    try:
        publisher.submit("first", embedding, first_completion)
        assert publication_started.wait(timeout=2)
        submit_thread.start()
        assert second_started.wait(timeout=2)
        assert not second_accepted.wait(timeout=0.05)
        stats = publisher.stats()
        assert stats["cache_pending_entries"] == 1
        assert (
            stats["cache_pending_bytes"] == embedding.numel() * embedding.element_size()
        )
        assert cache.get("first") is None
        assert not first_completion.done()

        release_publication.set()
        first_completion.result(timeout=2)
        second_completion.result(timeout=2)
        submit_thread.join(timeout=2)
        assert second_accepted.is_set()
        assert torch.equal(cache.get("first"), embedding)
        assert torch.equal(cache.get("second"), embedding)
        publisher.close()
        assert publisher.stats()["cache_pending_entries"] == 0
        assert publisher.stats()["cache_pending_bytes"] == 0
    finally:
        release_publication.set()
        if submit_thread.ident is not None:
            submit_thread.join(timeout=2)
        else:
            pass
        publisher.close()


def test_deferred_cache_close_rejects_blocked_submit_and_drains_active_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "Stream", FakeCopyStream)
    cache = StageOutputCache(cache_device="cpu")
    publisher = DeferredEmbeddingCache(
        cache, device=torch.device("cpu"), max_pending_entries=1, max_pending_bytes=1024
    )
    publication_started = threading.Event()
    release_publication = threading.Event()
    submit_started = threading.Event()
    submit_rejected = threading.Event()
    embedding = torch.ones(3, _HIDDEN_SIZE)
    completion: Future[None] = Future()

    def controlled_publish(publication: DeferredPublication) -> None:
        publication_started.set()
        assert release_publication.wait(timeout=10)
        cache.put(publication.key, publication.embedding)

    def submit_second() -> None:
        submit_started.set()
        with pytest.raises(RuntimeError, match="closed"):
            publisher.submit("second", embedding, Future())
        submit_rejected.set()

    monkeypatch.setattr(publisher, "publish", controlled_publish)
    submit_thread = threading.Thread(target=submit_second)
    close_thread = threading.Thread(target=publisher.close)
    try:
        publisher.submit("first", embedding, completion)
        assert publication_started.wait(timeout=2)
        submit_thread.start()
        assert submit_started.wait(timeout=2)
        close_thread.start()
        assert submit_rejected.wait(timeout=2)
        assert close_thread.is_alive()
        assert not completion.done()
        release_publication.set()
        close_thread.join(timeout=2)
        assert not close_thread.is_alive()
        completion.result(timeout=2)
        assert torch.equal(cache.get("first"), embedding)
        assert cache.get("second") is None
    finally:
        release_publication.set()
        if submit_thread.ident is not None:
            submit_thread.join(timeout=2)
        else:
            pass
        if close_thread.ident is not None:
            close_thread.join(timeout=2)
        else:
            pass
        publisher.close()


@pytest.mark.parametrize("cache_max_entries", [0, 16])
def test_deferred_cache_disabled_or_oversized_fallback_does_not_queue(
    cache_max_entries: int,
) -> None:
    model = _StubModel()
    model.encode_gate = threading.Event()
    encode_gate = model.encode_gate
    service = _make_service(model, cache_max_entries=cache_max_entries)
    publisher = ControlledDeferredCache(service.cache)
    publisher.max_pending_bytes = 1
    service.deferred_cache = publisher

    first = service.submit_item(_item(123, 3))
    duplicate = service.submit_item(_item(123, 3))
    encode_gate.set()
    first.result(timeout=2)
    duplicate.result(timeout=2)
    assert model.encode_calls == 1
    assert publisher.submitted.empty()

    service.submit_item(_item(123, 3)).result(timeout=2)
    assert model.encode_calls == (2 if cache_max_entries == 0 else 1)
    assert len(service.cache) == (0 if cache_max_entries == 0 else 1)


def test_pending_cache_publication_does_not_delay_lm_or_duplicate_requests() -> None:
    model = _StubModel()
    model.encode_gate = threading.Event()
    encode_gate = model.encode_gate
    service = _make_service(model)
    publisher = ControlledDeferredCache(service.cache)
    service.deferred_cache = publisher
    leader, follower, late_follower = [_item(123, 3) for _ in range(3)]

    leader_future = service.submit_item(leader)
    follower_future = service.submit_item(follower)
    encode_gate.set()
    publication = publisher.submitted.get(timeout=2)
    leader_future.result(timeout=2)
    follower_future.result(timeout=2)

    assert not publication.completion.done()
    assert service.lookup_cached_embedding("123", 3) is None
    service.submit_item(late_follower).result(timeout=2)
    assert model.encode_calls == 1
    for audio_item in (leader, follower, late_follower):
        assert torch.equal(audio_item.precomputed_embeddings, publication.embedding)
        assert audio_item.feature is None

    publisher.publish(publication)
    assert torch.equal(service.lookup_cached_embedding("123", 3), publication.embedding)
    service.submit_item(_item(123, 3)).result(timeout=2)
    assert model.encode_calls == 1


def test_publication_before_lm_completion_preserves_single_flight() -> None:
    model = _StubModel()
    service = _make_service(model)
    publisher = ControlledDeferredCache(service.cache)
    publisher.publish_during_submit = True
    publisher.submit_gate.clear()
    service.deferred_cache = publisher

    try:
        leader_future = service.submit_item(_item(123, 3))
        publication = publisher.submitted.get(timeout=2)
        assert publication.completion.done()
        assert not leader_future.done()
        service.cache.clear()
        follower_future = service.submit_item(_item(123, 3))
        publisher.submit_gate.set()
        leader_future.result(timeout=2)
        follower_future.result(timeout=2)
        assert model.encode_calls == 1

        service.submit_item(_item(123, 3)).result(timeout=2)
        assert model.encode_calls == 2
    finally:
        publisher.submit_gate.set()


def test_cache_publication_failure_keeps_lm_result_and_allows_retry() -> None:
    model = _StubModel()
    service = _make_service(model)
    publisher = ControlledDeferredCache(service.cache)
    service.deferred_cache = publisher
    first_item = _item(123, 3)

    first_future = service.submit_item(first_item)
    publication = publisher.submitted.get(timeout=2)
    embedding = first_future.result(timeout=2)
    publication.completion.set_exception(RuntimeError("cache publication failed"))

    assert first_future.result() is embedding
    assert torch.equal(first_item.precomputed_embeddings, embedding)
    assert service.lookup_cached_embedding("123", 3) is None

    retry_item = _item(123, 3)
    service.submit_item(retry_item).result(timeout=2)
    assert model.encode_calls == 2
    retry_publication = publisher.submitted.get(timeout=2)
    publisher.publish(retry_publication)
    assert torch.equal(
        service.lookup_cached_embedding("123", 3), retry_item.precomputed_embeddings
    )


def test_batch_retry_reuses_an_already_pending_cache_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel()
    model.encode_gate = threading.Event()
    model.encode_started = threading.Event()
    encode_gate = model.encode_gate
    service = _make_service(model)
    publisher = ControlledDeferredCache(service.cache)
    publisher.max_pending_bytes = 3 * _HIDDEN_SIZE * 4
    service.deferred_cache = publisher
    deferred_item = _item(123, 3)
    oversized_item = _item(456, 5)
    oversized_key = service.cache_key(oversized_item)
    original_put = service.cache.put
    failed_once = False

    def fail_first_oversized_copy(key: str | None, embedding: torch.Tensor) -> None:
        nonlocal failed_once
        if key == oversized_key and not failed_once:
            failed_once = True
            raise RuntimeError("synchronous cache publication failed")
        else:
            original_put(key, embedding)

    monkeypatch.setattr(service.cache, "put", fail_first_oversized_copy)
    initial_future = service.submit_item(_item(1, 3))
    assert model.encode_started.wait(timeout=2)
    deferred_future = service.submit_item(deferred_item)
    oversized_future = service.submit_item(oversized_item)
    encode_gate.set()
    for completion in (initial_future, deferred_future, oversized_future):
        completion.result(timeout=2)

    assert failed_once
    assert model.encode_batch_sizes == [1, 2, 1, 1]
    deferred_key = service.cache_key(deferred_item)
    assert sum(record.key == deferred_key for record in publisher.publications) == 1
    for publication in publisher.publications:
        publisher.publish(publication)
    assert torch.equal(
        service.lookup_cached_embedding("123", 3), deferred_item.precomputed_embeddings
    )
    assert torch.equal(
        service.lookup_cached_embedding("456", 5), oversized_item.precomputed_embeddings
    )


def test_cancelled_request_futures_leave_shared_encode_and_cache_usable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _StubModel()
    model.encode_gate = threading.Event()
    encode_gate = model.encode_gate
    service = _make_service(model)
    publisher = ControlledDeferredCache(service.cache)
    service.deferred_cache = publisher

    leader_future = service.submit_item(_item(123, 3))
    follower_future = service.submit_item(_item(123, 3))
    survivor = _item(123, 3)
    survivor_future = service.submit_item(survivor)
    assert leader_future.cancel()
    assert follower_future.cancel()
    encode_gate.set()
    survivor_future.result(timeout=2)
    publication = publisher.submitted.get(timeout=2)
    publisher.publish(publication)

    assert model.encode_calls == 1
    assert leader_future.cancelled()
    assert follower_future.cancelled()
    assert not survivor_future.cancel()
    assert torch.equal(
        service.lookup_cached_embedding("123", 3), survivor.precomputed_embeddings
    )
    service.submit_item(_item(456, 3)).result(timeout=2)
    assert model.encode_calls == 2
    assert "exception calling callback" not in caplog.text


def test_close_drains_cache_publication_and_rejects_all_new_requests() -> None:
    service = _make_service()
    publisher = ControlledDeferredCache(service.cache)
    publisher.close_gate.clear()
    service.deferred_cache = publisher
    service.submit_item(_item(123, 3)).result(timeout=2)
    publication = publisher.submitted.get(timeout=2)
    service.cache.put(service.cache_key(_item(456, 3)), torch.ones(3, _HIDDEN_SIZE))
    close_thread = threading.Thread(target=service.close)

    try:
        close_thread.start()
        assert publisher.close_started.wait(timeout=2)
        assert close_thread.is_alive()
        assert not service.thread.is_alive()
        assert not publication.completion.done()
        for audio_hash in (123, 456, 789):
            with pytest.raises(RuntimeError, match="closed"):
                service.submit_item(_item(audio_hash, 3))
        publisher.close_gate.set()
        close_thread.join(timeout=2)
        assert not close_thread.is_alive()
        assert publication.completion.done()
        assert torch.equal(
            service.lookup_cached_embedding("123", 3), publication.embedding
        )
    finally:
        publisher.close_gate.set()
        close_thread.join(timeout=2)


def test_encode_attaches_lm_ready_embedding_and_clears_feature() -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _item(7, 3)

    service.encode_item(item)

    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
    assert item.precomputed_embeddings.dtype == model.dtype
    assert (
        item.precomputed_embeddings.device
        == next(model.audio_tower.parameters()).device
    )
    assert item.feature is None
    assert item.format.name == "PRECOMPUTED_EMBEDDING"
    assert model.encode_calls == 1
    assert model.grad_enabled_during_encode is False
    assert service.stats()["misses"] == 1


def test_submit_returns_before_encoding_completes() -> None:
    model = _StubModel()
    gate = threading.Event()
    model.encode_gate = gate
    service = _make_service(model)
    item = _item(7, 3)

    future = service.submit_item(item)

    assert not future.done()
    assert item.precomputed_embeddings is None
    gate.set()
    future.result(timeout=2)
    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


def test_async_submissions_form_full_batch_without_blocked_callers() -> None:
    model = _StubModel()
    gate = threading.Event()
    encode_started = threading.Event()
    model.encode_gate = gate
    model.encode_started = encode_started
    service = _make_service(model, max_batch_size=8)
    items = [_item(audio_hash, 3) for audio_hash in range(9)]

    futures = [service.submit_item(items[0])]
    assert encode_started.wait(timeout=2)
    assert model.encode_calls == 1
    futures.extend(service.submit_item(item) for item in items[1:])
    gate.set()
    for future in futures:
        future.result(timeout=2)

    assert model.encode_batch_sizes == [1, 8]


def test_async_single_flight_completes_each_item_future() -> None:
    model = _StubModel()
    gate = threading.Event()
    model.encode_gate = gate
    service = _make_service(model)
    items = [_item(123, 3) for _ in range(3)]

    futures = [service.submit_item(item) for item in items]
    assert all(not future.done() for future in futures)
    gate.set()
    for future in futures:
        future.result(timeout=2)

    assert model.encode_calls == 1
    assert service.stats()["merged"] == 2
    assert all(item.precomputed_embeddings is not None for item in items)


def test_close_stops_worker() -> None:
    service = _make_service()

    service.close()

    assert not service.thread.is_alive()


def test_batch_context_unwinds_inference_mode_when_stream_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(Qwen3ASRPreLMEncoderService)
    service.stream = SimpleNamespace(device=torch.device("cuda", 0))

    class _FakeDeviceModule:
        def __init__(self) -> None:
            self.stream_calls: list[object] = []

        @contextlib.contextmanager
        def stream(self, stream):  # noqa: ANN001, ANN202
            self.stream_calls.append(stream)
            raise RuntimeError("stream context failed")
            yield

    device_module = _FakeDeviceModule()
    monkeypatch.setattr(torch, "get_device_module", lambda _device=None: device_module)

    assert not torch.is_inference_mode_enabled()
    with pytest.raises(RuntimeError, match="stream context failed"):
        with service.batch_context():
            pass
    assert not torch.is_inference_mode_enabled()
    assert device_module.stream_calls == [service.stream]


def test_cache_hit_skips_reencode() -> None:
    model = _StubModel()
    service = _make_service(model)

    first = _item(11, 3)
    second = _item(11, 3)
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 1
    assert torch.equal(first.precomputed_embeddings, second.precomputed_embeddings)
    assert second.feature is None
    assert service.stats()["hits"] == 1


def test_lookup_cached_embedding_returns_only_valid_entries() -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _item(11, 3)
    service.encode_item(item)

    cached = service.lookup_cached_embedding(item.audio_fingerprint, 3)

    assert cached is not None
    assert torch.equal(cached, item.precomputed_embeddings.cpu())
    assert service.stats()["hits"] == 1

    assert service.lookup_cached_embedding(item.audio_fingerprint, 4) is None
    assert len(service.cache) == 0


def test_extended_audio_never_reuses_prefix_embedding() -> None:
    model = _StubModel()
    service = _make_service(model)

    short = _item(111, 3)
    extended = _item(222, 5)
    service.encode_item(short)
    service.encode_item(extended)

    assert model.encode_calls == 2
    assert extended.precomputed_embeddings.shape == (5, _HIDDEN_SIZE)
    assert len(service.cache) == 2
    assert not torch.equal(
        short.precomputed_embeddings[0], extended.precomputed_embeddings[0]
    )


def test_cache_key_prefers_full_waveform_fingerprint() -> None:
    model = _StubModel()
    service = _make_service(model)
    first = _item(7, 3)
    second = _item(7, 3)
    first.audio_fingerprint = "full-hash-a"
    second.audio_fingerprint = "full-hash-b"

    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert len(service.cache) == 2


def test_concurrent_identical_requests_encode_once() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.05
    service = _make_service(model)
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    items = [_item(123, 3) for _ in range(n_threads)]
    errors: list[Exception] = []

    def worker(item: MultimodalDataItem) -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(item)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert model.encode_calls == 1
    for item in items:
        assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
        assert torch.equal(item.precomputed_embeddings, items[0].precomputed_embeddings)
    stats = service.stats()
    assert stats["merged"] + stats["hits"] == n_threads - 1


def test_stale_cache_miss_rechecks_before_starting_duplicate_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel()
    service = _make_service(model)
    stale_miss = threading.Event()
    release_stale_reader = threading.Event()
    original_get = service.cache.get

    def controlled_get(key: str | None):  # noqa: ANN202
        cached = original_get(key)
        if (
            threading.current_thread().name == "stale-cache-reader"
            and not stale_miss.is_set()
        ):
            assert cached is None
            stale_miss.set()
            assert release_stale_reader.wait(timeout=10)
        return cached

    monkeypatch.setattr(service.cache, "get", controlled_get)
    follower_item = _item(123, 3)
    errors: list[Exception] = []

    def follower() -> None:
        try:
            service.encode_item(follower_item)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=follower, name="stale-cache-reader")
    thread.start()
    assert stale_miss.wait(timeout=10)

    leader_item = _item(123, 3)
    service.encode_item(leader_item)
    release_stale_reader.set()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert not errors, errors
    assert model.encode_calls == 1
    assert torch.equal(
        leader_item.precomputed_embeddings,
        follower_item.precomputed_embeddings,
    )
    assert service.stats()["hits"] == 1


def test_concurrent_identical_requests_deduplicate_without_cache() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.05
    service = _make_service(model, cache_max_entries=0)
    barrier = threading.Barrier(2)
    items = [_item(123, 3) for _ in range(2)]
    errors: list[Exception] = []

    def worker(item: MultimodalDataItem) -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(item)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert model.encode_calls == 1
    assert len(service.cache) == 0
    assert torch.equal(items[0].precomputed_embeddings, items[1].precomputed_embeddings)


def test_submit_item_failure_counts_failed() -> None:
    model = _StubModel()
    model.fail = True
    service = _make_service(model)

    future = service.submit_item(_item(88, 3))

    with pytest.raises(RuntimeError, match="boom"):
        future.result(timeout=2)
    assert service.stats()["failed"] == 1


def test_encode_failure_propagates_without_poisoning_cache() -> None:
    model = _StubModel()
    model.fail = True
    model.encode_delay_s = 0.05
    service = _make_service(model)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(_item(55, 3))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(errors) == 2
    assert all(isinstance(exc, RuntimeError) and "boom" in str(exc) for exc in errors)
    assert len(service.cache) == 0
    assert service.stats()["failed"] == 2

    model.fail = False
    item = _item(55, 3)
    service.encode_item(item)
    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


def test_oom_failure_detaches_traceback_and_recovers() -> None:
    model = _StubModel()
    model.fail_oom = True
    service = _make_service(model)

    with pytest.raises(torch.OutOfMemoryError, match="encoder OOM") as excinfo:
        service.encode_item(_item(77, 3))

    # note (luojiaxuan): the propagated exception must not pin encoder frames.
    assert excinfo.value.__traceback__ is not None  # pytest's own raise site
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert service.stats()["failed"] == 1

    model.fail_oom = False
    item = _item(77, 3)
    service.encode_item(item)
    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


def test_merged_follower_token_mismatch_raises_and_counts_failed() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.2
    service = _make_service(model)
    leader_item = _item(321, 3)
    follower_item = _item(321, 5)
    errors: list[Exception] = []

    def leader() -> None:
        try:
            service.encode_item(leader_item)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=leader)
    thread.start()
    deadline = time.monotonic() + 5
    while not service.inflight and time.monotonic() < deadline:
        time.sleep(0.005)
    assert service.inflight, "leader never registered in-flight"

    with pytest.raises(RuntimeError, match="returned an invalid"):
        service.encode_item(follower_item)
    thread.join(timeout=30)

    assert not errors, errors
    assert leader_item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
    assert follower_item.precomputed_embeddings is None
    stats = service.stats()
    assert stats["merged"] == 1
    assert stats["failed"] == 1


def test_multi_item_batch_failure_retries_per_item_and_counts_stats() -> None:
    model = _StubModel()
    model.fail_multi_item = True
    gate = threading.Event()
    model.encode_gate = gate
    service = _make_service(model)
    items = [_item(31, 3), _item(32, 3), _item(33, 4)]
    errors: list[Exception] = []

    def worker(item: MultimodalDataItem) -> None:
        try:
            service.encode_item(item)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    # note (luojiaxuan): queue every leader before releasing the gate so the
    # next drain exercises the multi-item retry path.
    deadline = time.monotonic() + 5
    while len(service.inflight) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(service.inflight) == 3, "items never queued"
    gate.set()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    for item in items:
        assert item.precomputed_embeddings.shape == (
            item.num_audio_tokens,
            _HIDDEN_SIZE,
        )
    stats = service.stats()
    assert stats["failed"] == 0
    assert stats["items"] == 3
    assert stats["batches"] == 3
    assert model.encode_calls == 4
    assert len(service.cache) == 3


def test_eviction_under_byte_budget_triggers_reencode() -> None:
    model = _StubModel()
    service = _make_service(model, cache_max_bytes=100)

    for audio_hash in (1, 2, 3):
        service.encode_item(_item(audio_hash, 3))
    assert model.encode_calls == 3
    assert service.cache.eviction_count >= 1
    assert len(service.cache) == 2

    service.encode_item(_item(1, 3))
    assert model.encode_calls == 4


def test_invalid_cache_entry_is_evicted_and_reencoded() -> None:
    model = _StubModel()
    service = _make_service(model)
    probe = _item(42, 3)
    service.encode_item(probe)
    assert model.encode_calls == 1
    key = service.cache_key(probe)

    for poison in (
        torch.zeros(5, _HIDDEN_SIZE),
        torch.zeros(3, _HIDDEN_SIZE + 1),
        torch.zeros(3, _HIDDEN_SIZE, dtype=torch.float64),
    ):
        service.cache.put(key, poison)
        item = _item(42, 3)
        service.encode_item(item)
        assert model.encode_calls == 2
        assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
        assert torch.equal(item.precomputed_embeddings, probe.precomputed_embeddings)
        model.encode_calls = 1


def test_token_count_mismatch_fails_loudly() -> None:
    model = _StubModel()
    model.row_offset = 1
    service = _make_service(model)
    item = _item(9, 3)

    with pytest.raises(RuntimeError, match="!= expected rows"):
        service.encode_item(item)

    assert item.precomputed_embeddings is None
    assert len(service.cache) == 0


def test_missing_token_count_raises() -> None:
    service = _make_service()
    item = MultimodalDataItem(modality=Modality.AUDIO, hash=1)

    with pytest.raises(RuntimeError, match="num_audio_tokens"):
        service.encode_item(item)


def test_item_without_fingerprint_encodes_without_caching() -> None:
    model = _StubModel()
    service = _make_service(model)

    first = _item(1, 2)
    second = _item(1, 2)
    first.audio_fingerprint = None
    second.audio_fingerprint = None
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert first.feature is None
    assert first.precomputed_embeddings.shape == (2, _HIDDEN_SIZE)
    assert len(service.cache) == 0


def test_expected_audio_tokens_uses_request_metadata() -> None:
    explicit = MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=torch.zeros(1, 128, 300),
        model_specific_data={"num_audio_tokens": 5},
    )
    assert expected_audio_tokens(explicit) == 5
    assert expected_audio_tokens(MultimodalDataItem(modality=Modality.AUDIO)) is None


def test_build_cache_namespace_is_stable_and_scoped() -> None:
    model = _StubModel()
    frontend = SimpleNamespace(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        nb_max_frames=3000,
        padding_value=0.0,
    )
    base = dict(
        model_path="Qwen/Qwen3-ASR-1.7B",
        feature_extractor=frontend,
        mm_attention_backend=None,
    )

    namespace = build_cache_namespace(model, **base)
    assert namespace == build_cache_namespace(model, **base)
    assert namespace != build_cache_namespace(
        model, **{**base, "model_path": "other/revision"}
    )
    assert namespace != build_cache_namespace(
        model, **{**base, "mm_attention_backend": "triton_attn"}
    )
    assert namespace != build_cache_namespace(_StubModel(dtype=torch.bfloat16), **base)
    changed_frontend = SimpleNamespace(**{**vars(frontend), "hop_length": 320})
    assert namespace != build_cache_namespace(
        model, **{**base, "feature_extractor": changed_frontend}
    )
    changed_config = _StubModel()
    changed_config.config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE)
        ),
        marker="other",
    )
    assert namespace != build_cache_namespace(changed_config, **base)


def test_flat_2d_encoder_output_is_also_accepted() -> None:
    model = _StubModel()
    model.packed_3d_output = False
    service = _make_service(model)
    item = _item(8, 3)

    service.encode_item(item)

    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_deferred_cache_cuda_copy_retains_source_until_complete(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    cache = StageOutputCache(cache_device="cpu")
    publisher = DeferredEmbeddingCache(
        cache,
        device=device,
        max_pending_entries=2,
        max_pending_bytes=4 << 20,
    )
    producer_stream = torch.cuda.Stream(device=device)
    publication_started = threading.Event()
    release_publication = threading.Event()
    original_publish = publisher.publish
    completion: Future[None] = Future()

    def controlled_publish(publication: DeferredPublication) -> Exception | None:
        publication_started.set()
        assert release_publication.wait(timeout=10)
        return original_publish(publication)

    monkeypatch.setattr(publisher, "publish", controlled_publish)
    try:
        with torch.cuda.stream(producer_stream):
            embedding = (
                torch.arange(513 * 1024, device=device).remainder(251).to(dtype)
            ).reshape(513, 1024)
        producer_stream.synchronize()
        expected = embedding.cpu()
        source_reference = weakref.ref(embedding)
        publisher.submit("audio", embedding, completion)
        assert publication_started.wait(timeout=2)
        del embedding
        gc.collect()

        assert source_reference() is not None
        assert cache.get("audio") is None
        assert not completion.done()
        with torch.cuda.stream(producer_stream):
            scratch = torch.empty(513, 1024, dtype=dtype, device=device)
            scratch.fill_(99)
        release_publication.set()
        completion.result(timeout=10)
        cached = cache.get("audio")
        assert cached.device.type == "cpu"
        assert cached.is_pinned()
        assert torch.equal(cached, expected)
        publisher.close()
        gc.collect()
        assert source_reference() is None
        assert publisher.stats()["cache_pending_bytes"] == 0
    finally:
        release_publication.set()
        publisher.close()


@pytest.mark.accelerator
@requires_accelerator
def test_the_device_cache_is_really_reclaimed_after_an_oom() -> None:
    """The behaviour the fix exists for, on the live accelerator."""
    device_module = torch.get_device_module(torch.device(_DEVICE))
    service = _make_service(_StubModel().to(_DEVICE))

    with device_module.device(_DEVICE):
        device_module.synchronize()
        device_module.empty_cache()
        floor = device_module.memory_reserved()
        allocated = device_module.memory_allocated()
        cached_slack = max(0, floor - allocated)
        block = torch.empty(
            cached_slack + 64 * 1024 * 1024,
            dtype=torch.uint8,
            device=_DEVICE,
        )
        device_module.synchronize()
        reserved_with_block = device_module.memory_reserved()
        assert reserved_with_block > floor

        del block
        device_module.synchronize()
        reserved_before = device_module.memory_reserved()
        assert reserved_before > floor

        service.recover_after_failure(torch.OutOfMemoryError("encoder OOM"))

        device_module.synchronize()
        assert device_module.memory_reserved() < reserved_before
