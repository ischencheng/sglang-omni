# SPDX-License-Identifier: Apache-2.0
"""Bounded publication of completed Qwen3-ASR embeddings to the CPU cache."""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
from dataclasses import dataclass

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CachePublication:
    key: str
    embedding: torch.Tensor
    completion: concurrent.futures.Future[None]
    size_bytes: int


class DeferredEmbeddingCache:
    """Publish CPU copies without holding up ready encoder requests."""

    def __init__(
        self,
        cache: StageOutputCache,
        *,
        device: torch.device,
        max_pending_entries: int,
        max_pending_bytes: int,
    ) -> None:
        if max_pending_entries < 1 or max_pending_bytes < 1:
            raise ValueError("Deferred cache pending limits must be positive")
        else:
            pass
        self.cache = cache
        self.device = device
        self.max_pending_entries = max_pending_entries
        self.max_pending_bytes = max_pending_bytes
        self.stream = torch.cuda.Stream(device=device)
        self.queue: queue.Queue[CachePublication | None] = queue.Queue()
        self.condition = threading.Condition()
        self.pending_entries = 0
        self.pending_bytes = 0
        self.failures = 0
        self.backpressure_waits = 0
        self.closed = False
        self.thread = threading.Thread(
            target=self.worker, name="qwen3-asr-cache-copy", daemon=True
        )
        self.thread.start()

    def submit(
        self,
        key: str,
        embedding: torch.Tensor,
        completion: concurrent.futures.Future[None],
    ) -> None:
        size_bytes = embedding.numel() * embedding.element_size()
        if size_bytes > self.max_pending_bytes:
            raise ValueError("Embedding exceeds the deferred cache pending budget")
        else:
            pass
        with self.condition:
            waited = False
            while (
                self.pending_entries >= self.max_pending_entries
                or self.pending_bytes + size_bytes > self.max_pending_bytes
            ) and not self.closed:
                if not waited:
                    self.backpressure_waits += 1
                    waited = True
                else:
                    pass
                self.condition.wait()
            if self.closed:
                raise RuntimeError("Deferred embedding cache is closed")
            else:
                pass
            self.pending_entries += 1
            self.pending_bytes += size_bytes
            try:
                self.queue.put(
                    CachePublication(
                        key=key,
                        embedding=embedding,
                        completion=completion,
                        size_bytes=size_bytes,
                    )
                )
            except Exception:
                self.pending_entries -= 1
                self.pending_bytes -= size_bytes
                self.condition.notify_all()
                raise

    def publish(self, publication: CachePublication) -> Exception | None:
        try:
            with torch.inference_mode(), torch.cuda.device(self.device):
                try:
                    host_embedding = torch.empty_like(
                        publication.embedding, device="cpu", pin_memory=True
                    )
                except RuntimeError as exc:
                    logger.warning(
                        f"Qwen3-ASR cache pinning failed; using pageable memory: {exc}"
                    )
                    host_embedding = publication.embedding.to(device="cpu")
                else:
                    with torch.cuda.stream(self.stream):
                        publication.embedding.record_stream(self.stream)
                        host_embedding.copy_(publication.embedding, non_blocking=True)
                        copied = torch.cuda.Event()
                        copied.record(self.stream)
                    copied.synchronize()
                self.cache.put(publication.key, host_embedding)
        except Exception as exc:
            # note (Andrew Cheng): keep copy buffers alive through stream cleanup.
            try:
                self.stream.synchronize()
            except Exception:
                logger.exception("Qwen3-ASR cache stream cleanup failed")
            logger.exception("Qwen3-ASR deferred cache publication failed")
            return RuntimeError(f"{type(exc).__name__}: {exc}")
        else:
            return None

    def worker(self) -> None:
        while True:
            publication = self.queue.get()
            if publication is None:
                return
            else:
                pass
            failure = self.publish(publication)
            completion = publication.completion
            size_bytes = publication.size_bytes
            publication = None
            if failure is None:
                completion.set_result(None)
            else:
                completion.set_exception(failure)
            completion = None
            with self.condition:
                self.pending_entries -= 1
                self.pending_bytes -= size_bytes
                self.failures += int(failure is not None)
                self.condition.notify_all()

    def stats(self) -> dict[str, int]:
        with self.condition:
            return {
                "cache_pending_entries": self.pending_entries,
                "cache_pending_bytes": self.pending_bytes,
                "cache_publication_failures": self.failures,
                "cache_backpressure_waits": self.backpressure_waits,
            }

    def close(self) -> None:
        with self.condition:
            if not self.closed:
                self.closed = True
                self.queue.put(None)
                self.condition.notify_all()
            else:
                pass
        self.thread.join()
