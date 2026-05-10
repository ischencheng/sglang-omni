# SPDX-License-Identifier: Apache-2.0
"""TP-aware scheduler for SGLang-backed encoder stages.

Same public shape as :class:`SimpleScheduler` (``inbox`` / ``outbox`` /
``start`` / ``stop`` / ``abort``) so :class:`Stage` does not branch on
scheduler type. Adds an explicit two-channel broadcast in
``_recv_messages``:

1. **Metadata** over the SGLang TP CPU group via ``broadcast_pyobj`` —
   pickles only the dict skeleton, never the tensor payload bytes.
2. **Tensors** over the SGLang TP device group via ``dist.broadcast``
   on cuda:0.

A small ``all_gather_object`` "alloc-ok?" handshake between the two
broadcasts prevents a non-entry-rank OOM mid-allocation from leaving
the entry rank stuck in ``dist.broadcast``.

The scheduler is correct in both lanes:

- ``tp_size == 1``: skip the broadcasts; still strip-and-lift CPU shm
  tensors to ``cuda:0`` because the upstream ``get_image_feature`` /
  ``get_video_feature`` call ``.type(dtype)`` only — they do not move
  tensors to the model's device.
- ``tp_size  > 1``: drain the inbox on the entry rank only; broadcast
  inputs to non-entry ranks; run the same ``build_batch`` /
  ``encode_batch`` / ``slice_results`` pipeline on every rank;
  emit results from the entry rank only.

Naming note: this module uses ``entry_rank`` / ``non-entry rank`` to
describe the rank-0-vs-rest asymmetry. The asymmetry is just "who
owns external IO" — there's no leader election or failover. The
Stage-level ``single/leader/follower`` role split is a separate
abstraction layer that this scheduler doesn't touch.

See ``docs/developer_reference/encoder_tp_path_b_design.md`` for the
load-bearing design notes.
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import os
import queue as _queue_mod
import time
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist

from sglang.srt.utils import broadcast_pyobj

from sglang_omni_v1.pipeline.relay_io import extract_tensors, restore_tensors
from sglang_omni_v1.scheduling.messages import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from sglang_omni_v1.model_runner.sglang_encoder_runner import SGLangEncoderRunner
    from sglang_omni_v1.models.qwen3_omni.encoder_adapters import (
        BatchPlan,
        EncoderAdapter,
    )

logger = logging.getLogger(__name__)

# Tagged-dict sentinel used to ship recv-time errors over the same
# CPU-group ``broadcast_pyobj`` slot the success path uses. Identity
# sentinels (e.g. ``object()``) do not survive the pickle round-trip
# inside ``broadcast_pyobj``; a kind-string does.
_RECV_ERROR_KIND = "encoder_recv_error"


@dataclasses.dataclass(slots=True)
class _TensorSpec:
    """Lightweight description of a tensor for the metadata broadcast.

    Carries the typed ``torch.dtype`` (not the stringified form
    ``relay_io.extract_tensors`` produces, which would force a parser on
    the non-entry-rank side).
    """
    path: str
    shape: tuple[int, ...]
    dtype: torch.dtype


class BatchCollectError(RuntimeError):
    """Raised when batch admission fails after draining one or more messages.

    Carries the drained ``messages`` so the scheduler can emit one
    request-level error per request instead of silently dropping them.
    """

    def __init__(
        self,
        messages: list[IncomingMessage],
        error: BaseException,
    ) -> None:
        super().__init__(str(error))
        self.messages = messages
        self.error = error


class EncoderScheduler:
    """Inbox -> two-channel broadcast -> encoder forward -> outbox.

    Public contract identical to :class:`SimpleScheduler`. ``Stage`` is
    unaware of the TP shape — it just hands inputs to ``inbox`` and
    drains ``outbox``.
    """

    def __init__(
        self,
        runner: "SGLangEncoderRunner",
        adapter: "EncoderAdapter",
        *,
        max_batch_size: int = 32,
        max_batch_wait_ms: int = 50,
        request_cost_fn: Callable[[Any], int] | None = None,
        max_batch_cost: int | None = None,
    ) -> None:
        self.runner = runner
        self.adapter = adapter
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._request_cost_fn = request_cost_fn
        self._max_batch_cost = (
            max(int(max_batch_cost), 0) if max_batch_cost is not None else None
        )
        self._pending_messages: collections.deque[IncomingMessage] = (
            collections.deque()
        )
        self._running = False
        self._aborted_request_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the scheduler loop on the current thread.

        Three error domains, not one catch-all:

        1. **Recoverable pre-forward** (``_recv_messages``,
           ``build_batch``): both synchronize through the TP CPU group
           via :meth:`_gather_pre_forward_error` *before* any model
           collective starts. On any rank failure, the entry rank emits
           one ``OutgoingMessage(type="error")`` per drained request and
           every rank ``continue``s into the next loop iteration.

        2. **Fatal forward** (``encode_batch``): once we've entered
           upstream SGLang TP collectives (``ColumnParallelLinear``,
           ``RowParallelLinear``, NCCL), one rank cannot safely recover
           with a CPU gather — peers may still be blocked in NCCL. The
           rank that observes the exception calls
           :meth:`_fatal_tp_forward_error`, which exits the process
           non-zero so ``StageGroup`` / ``MultiProcessPipelineRunner``
           tears down the whole TP group and the runner's monitor fails
           outstanding Coordinator futures.

        3. **Recoverable post-forward** (``slice_results``): runs only
           on the entry rank after ``encode_batch`` returned on every
           rank. It can emit per-request errors locally and continue.
        """
        self._running = True
        while self._running:
            messages, recv_err = self._recv_messages()
            if self._gather_pre_forward_error(recv_err):
                if self.runner.is_entry_rank:
                    self._emit_error(
                        messages,
                        recv_err
                        if recv_err is not None
                        else RuntimeError("peer-rank encoder recv failed"),
                    )
                continue
            if not messages:
                continue

            plan = None
            build_err: BaseException | None = None
            try:
                plan = self.adapter.build_batch(messages)
            except Exception as exc:  # noqa: BLE001
                build_err = exc

            if self._gather_pre_forward_error(build_err):
                if self.runner.is_entry_rank:
                    self._emit_error(
                        messages,
                        build_err
                        if build_err is not None
                        else RuntimeError("peer-rank encoder build_batch failed"),
                    )
                continue

            try:
                raw = self.runner.encode_batch(plan)
            except Exception as exc:  # noqa: BLE001
                # Production: _fatal_tp_forward_error calls os._exit(1)
                # and never returns; the runner's _monitor_children
                # then fails outstanding Coordinator futures.
                # Tests: monkey-patched fatal handler returns. In that
                # case we exit the loop cleanly without re-raising —
                # re-raising would crash the scheduler thread, which
                # under Stage._handle_scheduler_crash would tear down
                # the stage even in tests where we want to assert state
                # post-fault.
                self._fatal_tp_forward_error(exc)
                self._running = False
                return

            if not self.runner.is_entry_rank:
                continue

            try:
                results = self.adapter.slice_results(raw, plan, messages)
            except Exception as exc:  # noqa: BLE001
                logger.exception("EncoderScheduler slice_results failed")
                self._emit_error(messages, exc)
                continue

            for msg, out in zip(messages, results):
                if msg.request_id in self._aborted_request_ids:
                    continue
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="result",
                        data=out,
                    )
                )

    def _gather_pre_forward_error(
        self,
        local_err: BaseException | None,
    ) -> bool:
        """Synchronize recoverable recv/build errors before model collectives.

        Returns True iff *any* rank reported a failure. The TP CPU group
        gather marshals only a picklable boolean — the exception object
        stays local, so each rank emits its own request-level error.
        """
        if self.runner.tp_size <= 1:
            return local_err is not None
        err_flags: list[bool] = [False] * self.runner.tp_size
        dist.all_gather_object(
            err_flags,
            local_err is not None,
            group=self.runner.tp_group.cpu_group,
        )
        return any(err_flags)

    def _fatal_tp_forward_error(self, error: BaseException) -> None:
        """Exit non-zero after a TP forward fault.

        Once ``encode_batch`` has entered upstream SGLang TP collectives,
        a rank-local exception cannot be recovered through a post-hoc CPU
        gather: peers may still be blocked in NCCL and never reach the
        gather. Force a child-process failure so
        :class:`MultiProcessPipelineRunner` tears down the whole TP group
        and fails outstanding Coordinator futures from the parent side.

        Overridable by tests via monkey-patch (the test stub records the
        error and *does* return so the test runner can assert on it).
        """
        logger.exception(
            "Fatal TP encoder forward failure on rank %d/%d: %r",
            self.runner.tp_rank, self.runner.tp_size, error,
        )
        os._exit(1)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        """Mark a request as aborted; results emitted later will be dropped."""
        self._aborted_request_ids.add(request_id)
        # bound the set so it cannot grow forever in long-running servers
        if len(self._aborted_request_ids) > 10000:
            keep = list(self._aborted_request_ids)[-5000:]
            self._aborted_request_ids = set(keep)

    # ------------------------------------------------------------------
    # Recv path: inbox drain + two-channel broadcast
    # ------------------------------------------------------------------

    def _next_message(self) -> IncomingMessage | None:
        if self._pending_messages:
            return self._pending_messages.popleft()
        try:
            return self.inbox.get(timeout=0.1)
        except _queue_mod.Empty:
            return None

    def _message_cost(self, msg: IncomingMessage) -> int:
        if self._request_cost_fn is None or msg.type != "new_request":
            return 0
        return max(int(self._request_cost_fn(msg.data)), 0)

    def _collect_batch_from_inbox(self) -> list[IncomingMessage]:
        """Drain the inbox into a cost-bounded batch (entry rank only).

        Mirrors :func:`SimpleScheduler._collect_batch` so the SGLang lane
        inherits the same admission control as the local lane.

        Raises:
            BatchCollectError: ``request_cost_fn`` (adapter / model code)
                raised. Carries the drained list so the caller can emit
                one error per request instead of silently dropping them.
        """
        first = self._next_message()
        if first is None:
            return []

        if first.type != "new_request":
            return [first]

        batch: list[IncomingMessage] = [first]
        try:
            batch_cost = self._message_cost(first)
        except Exception as exc:  # noqa: BLE001
            raise BatchCollectError(batch, exc) from exc

        deadline = time.monotonic() + self._max_batch_wait_s
        while len(batch) < self._max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if msg.type != "new_request":
                self._pending_messages.append(msg)
                continue

            if self._max_batch_cost is not None:
                try:
                    cost = self._message_cost(msg)
                except Exception as exc:  # noqa: BLE001
                    batch.append(msg)  # so the failed request gets an error
                    raise BatchCollectError(batch, exc) from exc
                if batch_cost + cost > self._max_batch_cost:
                    self._pending_messages.appendleft(msg)
                    break
                batch_cost += cost
            batch.append(msg)
        return batch

    def _collect_batch_or_error(
        self,
    ) -> tuple[list[IncomingMessage], BaseException | None]:
        try:
            return self._collect_batch_from_inbox(), None
        except BatchCollectError as exc:
            return exc.messages, exc.error
        except Exception as exc:  # noqa: BLE001
            return [], exc

    def _strip_and_lift(
        self,
        messages: list[IncomingMessage],
    ) -> tuple[
        list[IncomingMessage],
        list[list[torch.Tensor]],
        list[list[_TensorSpec]],
    ]:
        """Extract tensors from each message, lift them to the runner device.

        Returns three parallel lists indexed by message:

        - ``meta_msgs``: deep-copied IncomingMessages whose ``data.data``
          dict tree has had its tensors replaced by ``extract_tensors``
          placeholders. These are the only objects that get pickled by
          ``broadcast_pyobj`` — small dict skeletons, never tensor bytes.
        - ``tensor_lists``: per-message lists of GPU-resident tensors,
          ordered so they can be paired one-to-one with ``specs_lists``.
        - ``specs_lists``: per-message lists of :class:`_TensorSpec`
          carrying ``(path, shape, torch.dtype)``. Followers reconstruct
          their receive placeholders from these.

        The CPU-shm relay path delivers tensors on CPU; the H2D copy here
        is the same memcpy the local v1 encoder forward already does
        before its forward call (``image_encoder.py:154``), so it is not
        new work — just earlier.
        """
        meta_msgs: list[IncomingMessage] = []
        tensor_lists: list[list[torch.Tensor]] = []
        specs_lists: list[list[_TensorSpec]] = []

        for msg in messages:
            payload = msg.data
            if payload is None or not hasattr(payload, "data"):
                meta_msgs.append(msg)
                tensor_lists.append([])
                specs_lists.append([])
                continue

            stripped, tensor_dict = extract_tensors(payload.data)
            tensors: list[torch.Tensor] = []
            specs: list[_TensorSpec] = []
            lifted: dict[str, torch.Tensor] = {}
            for path, t in tensor_dict.items():
                if t.device != self.runner.device:
                    t = t.to(self.runner.device, non_blocking=True)
                tensors.append(t)
                specs.append(
                    _TensorSpec(path=path, shape=tuple(t.shape), dtype=t.dtype)
                )
                lifted[path] = t

            # Rebuild the entry rank's payload with GPU-resident tensors
            # so its downstream BatchPlan sees the same device-resident
            # tensors the non-entry ranks will reconstruct.
            payload_cls = type(payload)
            new_payload = payload_cls(
                request_id=payload.request_id,
                request=payload.request,
                data=restore_tensors(stripped, lifted),
            )
            meta_msg = dataclasses.replace(msg, data=new_payload)
            meta_msgs.append(meta_msg)
            tensor_lists.append(tensors)
            specs_lists.append(specs)

        return meta_msgs, tensor_lists, specs_lists

    def _reattach_lifted_tensors(
        self,
        meta_msgs: list[IncomingMessage],
        tensor_lists: list[list[torch.Tensor]],
        specs_lists: list[list[_TensorSpec]],
    ) -> list[IncomingMessage]:
        """Follower path: rebuild messages by stitching specs back into payloads.

        On the entry rank ``_strip_and_lift`` already reattached the lifted
        tensors, so this is a no-op (returns ``meta_msgs`` unchanged).
        Followers, however, see ``meta_msgs`` whose ``data.data`` tree
        still contains placeholder dicts; they need to map ``spec.path``
        back to the freshly received tensor.
        """
        out: list[IncomingMessage] = []
        for msg, tensors, specs in zip(meta_msgs, tensor_lists, specs_lists):
            payload = msg.data
            if payload is None or not hasattr(payload, "data") or not specs:
                out.append(msg)
                continue
            tensor_dict = {spec.path: t for spec, t in zip(specs, tensors)}
            payload_cls = type(payload)
            restored = payload_cls(
                request_id=payload.request_id,
                request=payload.request,
                data=restore_tensors(payload.data, tensor_dict),
            )
            out.append(dataclasses.replace(msg, data=restored))
        return out

    def _allocation_ready_gather(self, *, local_ok: bool) -> list[bool]:
        """Gather per-rank allocation-success flags on the TP CPU group."""
        flags: list[bool] = [False] * self.runner.tp_size
        dist.all_gather_object(
            flags,
            local_ok,
            group=self.runner.tp_group.cpu_group,
        )
        return flags

    def _recv_messages(
        self,
    ) -> tuple[list[IncomingMessage], BaseException | None]:
        """Drain inbox and broadcast inputs to TP non-entry ranks.

        Never raises — returns ``(messages, error)``. The error is non-None
        if either rank failed during this iteration. Drained messages
        are returned even on entry-rank failure so the scheduler can emit
        request-level errors against them in the unified handshake.
        """
        if self.runner.tp_size == 1:
            local, collect_err = self._collect_batch_or_error()
            if collect_err is not None or not local:
                return local, collect_err
            try:
                meta_msgs, tensor_lists, specs_lists = self._strip_and_lift(
                    local
                )
            except Exception as exc:  # noqa: BLE001
                return local, exc
            # On the entry rank meta_msgs already carry GPU-resident tensors
            # because _strip_and_lift restored them before returning.
            return meta_msgs, None

        tp = self.runner.tp_group
        src_rank = tp.ranks[0]

        if self.runner.is_entry_rank:
            local, collect_err = self._collect_batch_or_error()
            if collect_err is not None:
                broadcast_pyobj(
                    [{"kind": _RECV_ERROR_KIND, "error": repr(collect_err)}],
                    tp.rank, tp.cpu_group, src=src_rank,
                )
                return local, collect_err

            try:
                meta_msgs, tensor_lists, specs_lists = self._strip_and_lift(
                    local
                )
            except Exception as exc:  # noqa: BLE001
                broadcast_pyobj(
                    [{"kind": _RECV_ERROR_KIND, "error": repr(exc)}],
                    tp.rank, tp.cpu_group, src=src_rank,
                )
                return local, exc

            broadcast_pyobj(
                [meta_msgs, specs_lists],
                tp.rank, tp.cpu_group, src=src_rank,
            )

            ok_flags = self._allocation_ready_gather(local_ok=True)
            if not all(ok_flags):
                return local, RuntimeError(
                    "peer-rank tensor allocation failed"
                )

            for tensor_list in tensor_lists:
                for t in tensor_list:
                    dist.broadcast(t, src=src_rank, group=tp.device_group)

            return meta_msgs, None

        # Follower path
        payload = broadcast_pyobj([], tp.rank, tp.cpu_group, src=src_rank)
        if (
            payload
            and isinstance(payload[0], dict)
            and payload[0].get("kind") == _RECV_ERROR_KIND
        ):
            return [], RuntimeError(
                f"entry rank failed before metadata broadcast: "
                f"{payload[0]['error']}"
            )
        meta_msgs, specs_lists = payload

        placeholders: list[list[torch.Tensor]] = []
        alloc_err: BaseException | None = None
        try:
            for specs in specs_lists:
                placeholders.append(
                    [
                        torch.empty(
                            spec.shape,
                            dtype=spec.dtype,
                            device=self.runner.device,
                        )
                        for spec in specs
                    ]
                )
        except Exception as exc:  # noqa: BLE001
            alloc_err = exc

        ok_flags = self._allocation_ready_gather(local_ok=alloc_err is None)
        if not all(ok_flags):
            return [], (
                alloc_err
                if alloc_err is not None
                else RuntimeError("peer-rank tensor allocation failed")
            )

        for ph_list in placeholders:
            for t in ph_list:
                dist.broadcast(t, src=src_rank, group=tp.device_group)

        return self._reattach_lifted_tensors(
            meta_msgs, placeholders, specs_lists
        ), None

    # ------------------------------------------------------------------
    # Error emission
    # ------------------------------------------------------------------

    def _emit_error(
        self,
        messages: list[IncomingMessage],
        error: BaseException,
    ) -> None:
        """Emit one ``OutgoingMessage(type="error")`` per drained request.

        SimpleScheduler exposes a single-request ``_emit_error`` helper;
        reusing it here would put a list[IncomingMessage] into
        ``OutgoingMessage.request_id``, which
        ``Stage._drain_outbox_external`` would TypeError on. Iterate
        explicitly so each drained request becomes one HTTP-500.
        """
        if not messages:
            # No request was safely captured — synthesize one anonymous
            # error so the failure does not vanish silently. Stage will
            # discard it because the request_id is empty, but the log
            # captures the cause.
            logger.error(
                "EncoderScheduler iteration failed without drained requests: %s",
                error,
            )
            return
        for msg in messages:
            if msg.request_id in self._aborted_request_ids:
                continue
            self.outbox.put(
                OutgoingMessage(
                    request_id=msg.request_id,
                    type="error",
                    data=error,
                )
            )
