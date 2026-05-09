# SPDX-License-Identifier: Apache-2.0
"""End-to-end loop tests for :class:`EncoderScheduler`.

Drives the scheduler loop from its public ``start()`` entry point with
mocked TP collectives, covering the per-iteration error boundary and
the request-level error emission against drained messages.
"""
from __future__ import annotations

import queue as _queue_mod
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang_omni_v1.proto import StagePayload
from sglang_omni_v1.scheduling.encoder_scheduler import EncoderScheduler
from sglang_omni_v1.scheduling.messages import IncomingMessage


def _mk_msg(rid: str, data: dict | None = None) -> IncomingMessage:
    return IncomingMessage(
        request_id=rid,
        type="new_request",
        data=StagePayload(request_id=rid, request=None, data=data or {}),
    )


class _PassthroughAdapter:
    stage_name = "image_encoder"

    def build_batch(self, messages):
        return SimpleNamespace(
            adapter=self, image_items=[], video_items=[], audio_items=[],
            spans=[], is_empty=True,
        )

    def run_feature(self, model, plan):
        return {"image": None, "video": None, "audio": None}

    def slice_results(self, raw, plan, messages):
        return [
            StagePayload(request_id=m.request_id, request=None, data={"out": True})
            for m in messages
        ]


class _CrashingForwardAdapter(_PassthroughAdapter):
    """Adapter whose forward raises — used to test error path."""

    def run_feature(self, model, plan):
        raise RuntimeError("forward boom")


class _FakeWorker:
    def __init__(self):
        self.tp_size = 1
        self.tp_rank = 0
        self.is_entry_rank = True
        self.device = torch.device("cpu")
        self.tp_group = None

    def encode_batch(self, plan):
        return plan.adapter.run_feature(self, plan)


def _drain_outbox(sch: EncoderScheduler) -> list:
    out = []
    while True:
        try:
            out.append(sch.outbox.get_nowait())
        except _queue_mod.Empty:
            return out


def _run_until_outbox(sch: EncoderScheduler, expected: int, timeout: float = 5.0):
    """Run scheduler in a thread; stop once expected outbox events arrived."""
    t = threading.Thread(target=sch.start, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while sch.outbox.qsize() < expected:
        if time.monotonic() > deadline:
            sch.stop()
            t.join(timeout=1.0)
            raise AssertionError(
                f"timed out waiting for {expected} outbox events; got {sch.outbox.qsize()}"
            )
        time.sleep(0.01)
    sch.stop()
    t.join(timeout=2.0)


def test_loop_emits_one_result_per_request_at_tp_size_1():
    sch = EncoderScheduler(_FakeWorker(), _PassthroughAdapter())
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    _run_until_outbox(sch, expected=2)
    out = _drain_outbox(sch)
    assert {o.type for o in out} == {"result"}
    assert sorted(o.request_id for o in out) == ["r0", "r1"]


def test_loop_emits_error_on_forward_failure_at_tp_size_1():
    """Forward raises → entry rank emits one error per drained request."""
    sch = EncoderScheduler(_FakeWorker(), _CrashingForwardAdapter())
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    _run_until_outbox(sch, expected=2)
    out = _drain_outbox(sch)
    assert all(o.type == "error" for o in out)
    assert sorted(o.request_id for o in out) == ["r0", "r1"]
    for o in out:
        assert isinstance(o.data, RuntimeError)
        assert "forward boom" in str(o.data)


def test_loop_drops_aborted_request_results():
    sch = EncoderScheduler(_FakeWorker(), _PassthroughAdapter())
    sch.abort("r1")
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    sch.inbox.put(_mk_msg("r2"))
    _run_until_outbox(sch, expected=2)
    out = _drain_outbox(sch)
    rids = sorted(o.request_id for o in out)
    assert rids == ["r0", "r2"]
    assert all(o.type == "result" for o in out)


def test_loop_recovers_from_recv_error_and_continues():
    """A recv-time failure produces an error per request and the scheduler
    survives — next iteration reads the next request fine."""

    cost_calls = {"n": 0}

    def cost_fn(payload):
        cost_calls["n"] += 1
        if cost_calls["n"] == 1:
            raise RuntimeError("first cost boom")
        return 0

    sch = EncoderScheduler(
        _FakeWorker(),
        _PassthroughAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
        request_cost_fn=cost_fn,
        max_batch_cost=10**9,
    )
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    _run_until_outbox(sch, expected=2)
    out = _drain_outbox(sch)
    by_id = {o.request_id: o for o in out}
    assert by_id["r0"].type == "error"
    assert by_id["r1"].type == "result"
