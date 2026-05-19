# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import importlib.machinery
from types import SimpleNamespace
from unittest.mock import MagicMock

_av = MagicMock()
_av.__spec__ = importlib.machinery.ModuleSpec("av", loader=None)
sys.modules.setdefault("av", _av)

import torch

from sglang_omni.models.higgs_tts.model import _RequestSlot
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.sampler import HiggsSamplerState


def test_higgs_runner_samples_codebook_logits_outside_forward() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner.model = _FakeHiggsModel(num_codebooks=2)

    logits = torch.zeros((1, 2, 4), dtype=torch.float32)
    logits[0, 0, 2] = 10.0
    logits[0, 1, 3] = 10.0
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=logits),
        next_token_ids=None,
    )
    request = SimpleNamespace(
        request_id="req-1",
        data=SimpleNamespace(
            req=SimpleNamespace(is_chunked=0),
            output_codes=[],
            generation_done=False,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        ),
    )

    runner._collect_step_outputs(result, [request])

    assert result.next_token_ids.tolist() == [2]
    assert len(request.data.output_codes) == 1
    assert request.data.output_codes[0].tolist() == [2, 1024]


def test_higgs_prepare_decode_updates_graph_visible_buffers() -> None:
    model = _FakeHiggsModel(num_codebooks=2)
    runner = object.__new__(HiggsTTSModelRunner)
    runner.model = model
    forward_batch = SimpleNamespace(req_ids=None, req_pool_indices=torch.tensor([5]))
    request = SimpleNamespace(request_id="req-1")

    runner.prepare_decode(forward_batch, None, [request])

    assert forward_batch.req_ids == ["req-1"]
    assert model.updated == [(torch.tensor([5]), ["req-1"])]


class _FakeHiggsModel:
    def __init__(self, num_codebooks: int) -> None:
        self._num_codebooks = num_codebooks
        self._slots: dict[str, _RequestSlot] = {
            "req-1": _RequestSlot(
                sampler=HiggsSamplerState(num_codebooks=num_codebooks)
            )
        }
        self.updated = []

    def get_slot(self, req_id: str) -> _RequestSlot:
        return self._slots[req_id]

    def update_decode_graph_buffer(
        self, req_pool_indices: torch.Tensor, req_ids: list[str]
    ) -> None:
        self.updated.append((req_pool_indices.clone(), list(req_ids)))
