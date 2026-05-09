# RFC: Multimodal Encoder TP via SGLang Native Encoders (Plan B)

Issue: https://github.com/sgl-project/sglang-omni/issues/375
Decision: **Plan B**. SGLang-Omni imports SGLang main's native multimodal
encoder implementations and inherits SGLang TP, instead of carrying its own
encoder copies under `models/<name>/components/`.

This RFC follows from #375 and the post-refactor architecture in #188. It
is the result of a code walk over the v1 stage / scheduler / mp_runner
stack and the upstream SGLang Qwen3-Omni / Qwen3-VL / encode_server paths,
plus a review pass that fixed three earlier mistakes in the follower data
path, single-rank distributed init, and the adapter batch interface.

## Architecture

### System Overview

```text
HTTP API -> Client -> Coordinator -> StageGroup -> Stage(leader)
                                                |        \
                                                |         Stage(follower) x (tp_size - 1)
                                                v
                                  EncoderScheduler (one per rank)
                                                v
                                  SGLangEncoderWorker (one per rank)
                                                v
                              upstream SGLang encoder model
                                  thinker.get_image_feature
                                  thinker.get_video_feature
                                  thinker.get_audio_feature
```

The public pipeline topology stays the same:

```text
preprocessing -> [image_encoder, audio_encoder] -> mm_aggregate -> thinker -> ...
```

Only the implementation behind encoder stages changes:

```text
old:  Stage -> SimpleScheduler -> Qwen3OmniImageEncoder (HF copy)
new:  Stage -> EncoderScheduler -> SGLangEncoderWorker -> upstream model.thinker.get_image_feature
```

### Layer Responsibilities

| Layer | Responsibility | Change vs v1 |
| --- | --- | --- |
| `Coordinator` | Submit, completion collection, abort broadcast | None |
| `MultiProcessPipelineRunner` / `StageGroup` | Spawn one OS process per TP rank, allocate NCCL port, inject `tp_rank/tp_size/gpu_id/nccl_port` | None — already TP-capable |
| `Stage` (`single/leader/follower`) | Control plane, relay IO, input aggregation, stream routing, scheduler in/out queues, leader-only outbound traffic | None |
| `TPLeaderFanout` / `TPFollowerControlPlane` | Mirror leader-side `Shutdown/Profiler/Abort` to followers via mp.Queue | None |
| **`EncoderScheduler`** | TP-aware non-AR scheduling loop: drain inbox on entry rank, broadcast **metadata** to followers via TP CPU group, broadcast **tensor data** to followers via TP device group, run worker on every rank, emit downstream traffic only on entry rank | **New** |
| **`SGLangEncoderWorker`** | Initialize SGLang distributed state (always — even at `tp_size=1`), build encoder-only `ModelConfig`, load upstream encoder model via `get_model`, expose `encode_batch()` | **New** |
| **`Qwen3OmniEncoderAdapter`** | v1 `PipelineState.encoder_inputs` <-> upstream `MultimodalDataItem` <-> v1 `encoder_outs`, with explicit `BatchPlan` for multi-request batching | **New** |
| upstream SGLang model | Owns encoder modules, native `ColumnParallelLinear` / `RowParallelLinear`, weight sharding, NCCL collectives | **Reused** |

> Note (Cheng): the table separates *Stage* from *Scheduler*. `Stage` already
> supports the `single/leader/follower` split; we are adding a new scheduler
> **shape**, not a new stage type. This matches the same boundary
> `OmniScheduler` and `SimpleScheduler` already obey today.

### Why This Shape

Three observations from the code walk drive Plan B:

1. **The TP launch infrastructure already exists.** `_build_tp_stage_specs`
   in `pipeline/mp_runner.py:164-225` already mints one `StageProcessSpec`
   per TP rank, allocates a per-stage NCCL port, builds
   `follower_work_queues` / `follower_abort_queues`, and tags rank 0 as
   `leader`. `get_stage_process_env` (`pipeline/stage_process.py:222-276`)
   already pins each child to a single GPU through `CUDA_VISIBLE_DEVICES` +
   `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS`. Plan B requires three small
   `pipeline/` + `serve/` changes on top of this: a `single_visible_device`
   spec flag, the launcher entry-point forcing `MultiProcessPipelineRunner`
   for any `backend in {"sglang", "auto"}` stage, and a `compile_pipeline()`
   sanity reject for the same — see [GPU placement across `tp_size=1`
   and `tp_size>1` lanes](#gpu-placement-across-tp_size1-and-tp_size1-lanes)
   for the load-bearing reason and the exact change.

2. **The leader/follower control fan-out is already wired, but data-plane
   fan-out is not.** `Stage.run()` mirrors `Shutdown/Profiler/Abort` from
   leader to followers via `TPLeaderFanout.fanout_control` /
   `fanout_abort`. It explicitly does **not** mirror `SubmitMessage` /
   `DataReadyMessage` — those go through ZMQ to leader only, and
   `Stage._drain_outbox_follower` refuses to emit external traffic.
   Followers also do not have a relay endpoint and never call
   `relay_io.read_payload`. This means the `EncoderScheduler` is the only
   layer that can hand inputs to followers, and it must do so over the
   SGLang TP groups (CPU for metadata, device for tensors).

3. **`OmniScheduler` has already proven the in-scheduler TP broadcast
   pattern for control-shaped messages.** `OmniScheduler._recv_scheduler_messages`
   (`scheduling/omni_scheduler.py:393-403`) drains the inbox on the entry
   rank only and uses `broadcast_pyobj(local, rank, tp_cpu_group, src=...)`
   to fan out the work list. `EncoderScheduler` adopts this pattern for
   the metadata side, and adds an explicit tensor side. We are deliberately
   not extending `SimpleScheduler` to become TP-aware; `SimpleScheduler`
   should stay the minimal local-CPU/GPU callable runner.

The missing pieces are therefore one new scheduler shape (`EncoderScheduler`)
that owns metadata + tensor broadcast, and one minimal SGLang worker
(`SGLangEncoderWorker`) that owns the distributed init and the upstream
model. Everything else is reuse.

### Evidence From Code Walk

| Evidence | File:line |
| --- | --- |
| `StageConfig.tp_size` and `gpu: list[int]` already exist and are validated | `sglang_omni_v1/config/schema.py:65-149` |
| `MultiProcessPipelineRunner._build_tp_stage_specs` mints per-rank specs and allocates one NCCL port per stage | `sglang_omni_v1/pipeline/mp_runner.py:164-260` |
| Per-process `CUDA_VISIBLE_DEVICES` mapping with `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true` is already done before torch import | `sglang_omni_v1/pipeline/stage_process.py:222-276` |
| `Stage` already splits `single/leader/follower` and only the leader owns ZMQ IO and relay reads | `sglang_omni_v1/pipeline/stage/runtime.py:71-216, 243-287, 449-498` |
| `TPLeaderFanout` mirrors only `Shutdown/Profiler/Abort`, not `Submit/DataReady` | `sglang_omni_v1/pipeline/tp_control.py:29-56` |
| `relay_io.write_payload` extracts tensors out of the payload `data` dict tree and ships them through the relay separately from the metadata pickle | `sglang_omni_v1/pipeline/relay_io.py:40-180` |
| `relay_io.extract_tensors` / `restore_tensors` are the existing helpers for the metadata/tensor split | `sglang_omni_v1/pipeline/relay_io.py:40-90` |
| `OmniScheduler` proves the in-scheduler `broadcast_pyobj` pattern for *control-shaped* messages | `sglang_omni_v1/scheduling/omni_scheduler.py:376-412` |
| `SimpleScheduler` is single-process inbox -> fn -> outbox, with no TP path | `sglang_omni_v1/scheduling/simple_scheduler.py:23-180` |
| Current Qwen3-Omni encoder factories build local HF towers under `SimpleScheduler` | `sglang_omni_v1/models/qwen3_omni/stages.py:704-783` |
| Local v1 encoder copies that should disappear after parity | `sglang_omni_v1/models/qwen3_omni/components/{image_encoder.py,audio_encoder.py}` |
| Upstream `Qwen3OmniMoeThinkerForConditionalGeneration` exposes `get_audio_feature` and inherits `get_image_feature` / `get_video_feature` from Qwen3-VL | `sglang-workspace/sglang/python/sglang/srt/models/qwen3_omni_moe.py:438-492`, `sglang/python/sglang/srt/models/qwen3_vl.py:1193-1226` |
| Upstream Qwen3-Omni audio encoder uses `ColumnParallelLinear` + `RowParallelLinear` — TP comes for free | `sglang/python/sglang/srt/models/qwen3_omni_moe.py:49-124` |
| `MMEncoder.__init__` is the canonical encoder-only init sequence and **always** initializes distributed, regardless of `tp_size` | `sglang/python/sglang/srt/disaggregation/encode_server.py:184-244` |
| `get_tp_group()` asserts `_TP is not None` — every model that uses `ColumnParallelLinear` / `RowParallelLinear` requires `initialize_model_parallel()` to have run, including at `tp_size=1` | `sglang/python/sglang/srt/distributed/parallel_state.py:1476-1483` |
| `ServerArgs.encoder_only / language_only / mm_enable_dp_encoder` already exist | `sglang/python/sglang/srt/server_args.py:782-811` |
| `MultimodalDataItem` is the upstream input contract | `sglang/python/sglang/srt/managers/schedule_batch.py:204-358` |
| `broadcast_pyobj(data, rank, group, src)` is the upstream TP-CPU-group fan-out helper | `sglang/python/sglang/srt/utils/common.py:1264-1310` |

### Directory Layout

Target additions only — no existing module is moved:

```text
sglang_omni_v1/
|-- scheduling/
|   `-- encoder_scheduler.py        # TP-aware scheduler for encoder stages [NEW]
|-- model_runner/
|   `-- sglang_encoder_worker.py    # Minimal SGLang-native encoder worker [NEW]
`-- models/
    `-- qwen3_omni/
        |-- encoder_adapters.py     # v1 <-> SGLang adapter (with BatchPlan) [NEW]
        `-- stages.py               # Backend switch in encoder factories [MODIFIED]
```

`encoder_adapters.py` becomes the per-model convention.

### Class Diagram

```text
              +------------------+
              |       Stage      |
              |  single/leader/  |
              |     follower     |
              +--------+---------+
                       |
                       v
              +------------------+
              |  EncoderScheduler|
              | inbox/outbox     |
              | metadata bcast   |  <- TP CPU group
              | tensor bcast     |  <- TP device group
              +--------+---------+
                       |
                       v
              +------------------+         +-----------------------+
              | SGLangEncoderWor.|<------- |    EncoderAdapter     |
              | server_args      |         | build_batch(msgs) ->  |
              | model_config     |         |   BatchPlan           |
              | tp_group         |         | run_feature(model,    |
              | model            |         |   plan) -> raw        |
              | encode_batch()   |         | slice_results(raw,    |
              +--------+---------+         |   plan, msgs) -> ...  |
                       |                   +-----------------------+
                       v
            upstream SGLang Qwen3-Omni model
            thinker.get_image_feature
            thinker.get_video_feature
            thinker.get_audio_feature
```

## Inputs Across TP Ranks

This is the section the earlier draft missed. It is the load-bearing contract
of the whole RFC.

### Why followers need their own input fan-out

In v1, `Stage._on_data_ready()` is the only path that materializes a
`StagePayload`: it reads `relay_io.read_payload(...)`, which fetches the
tensor blobs the upstream stage wrote into the relay and re-attaches them
into the payload `data` dict tree (`relay_io.py:170-200`). For TP encoder
stages:

- Only the leader has a ZMQ recv endpoint and a relay reader. Followers'
  control plane is `TPFollowerControlPlane` (`pipeline/tp_control.py:59-117`),
  which only handles `Shutdown/Profiler/Abort`.
- `Stage._drain_outbox_follower` (`stage/runtime.py:490-508`) actively
  refuses any external traffic from follower ranks.

So when an `image_encoder` request arrives, only the leader's
`scheduler.inbox` ever receives an `IncomingMessage`. Followers' inboxes
stay empty unless the **scheduler itself** ships the inputs to them.

### Why `broadcast_pyobj` of the whole message is wrong

A naive `broadcast_pyobj(messages, ..., cpu_group, src=0)` would pickle the
entire `StagePayload`, including pixel-value tensors that can be hundreds of
MB or several GB for long video. That:

- pickles GPU tensors (slow, requires `.cpu()` first),
- ships the result through a CPU communicator (defeats the whole reason we
  use NCCL on the device),
- conflates control-plane traffic with data-plane traffic.

### Two-channel broadcast contract

`EncoderScheduler._recv_messages()` runs on every rank and uses **two
channels** to mirror v1's existing relay split (`relay_io.py:106` already
does the metadata/tensor extraction we need):

1. **Metadata over the TP CPU group.** On the entry rank, drain the local
   inbox and run `extract_tensors(msg.data.data)` on each
   `IncomingMessage`. That returns `(metadata_dict_no_tensors,
   tensor_dict_path_to_tensor)`. Replace each message's payload `data` with
   the tensor-free metadata dict, attach a parallel `_tensor_specs` list
   describing every extracted tensor's `path`, `shape`, `dtype`. Then
   `broadcast_pyobj(metadata_messages, rank, tp_cpu_group, src=0)`. This
   pickles only the dict skeleton, not tensor payloads.

2. **Tensors over the TP device group, on GPU.** For each tensor in the
   entry rank's `tensor_dict`, do `dist.broadcast(tensor,
   src=tp_group.ranks[0], group=tp_group.device_group)`. Followers
   pre-allocate empty tensors matching the broadcast `_tensor_specs` on
   `cuda:<local 0>` and call the matching `dist.broadcast` to receive into
   them. After all tensors are received, followers run
   `restore_tensors(metadata, tensor_dict)` to rebuild the payload `data`
   and reconstitute the `IncomingMessage` list.

### Tensor placement contract (Phase 0)

The default v1 relay backend is `shm`, and `_resolve_relay_config` in
`pipeline/mp_runner.py:228-240` deliberately does **not** inject `gpu_id`
into the relay config when the backend is shm — shm copies into host
shared memory, so the leader's reconstructed payload tensors live on
**CPU**. NCCL `dist.broadcast` over the TP device group requires CUDA
tensors on the local rank's device. Therefore the contract is:

- **Leader**, before broadcasting: for every tensor extracted from the
  payload, run `t = t.to(self.worker.device, non_blocking=True)` (where
  `self.worker.device` is `cuda:0` after the per-process
  `CUDA_VISIBLE_DEVICES` remap). Then `dist.broadcast(t, src,
  group=tp.device_group)`. Stash the device-resident tensor back into the
  `tensor_dict` so `restore_tensors` reattaches the GPU copy, not the
  original CPU one.
- **Followers**, before broadcasting: allocate the placeholder via
  `torch.empty(spec.shape, dtype=spec.dtype, device=self.worker.device)`
  and broadcast into it.

Why this direction:

1. The forward pass needs every pixel-values / feature tensor on GPU
   anyway. Doing the H2D copy here, before the broadcast, costs the same
   memcpy that `Qwen3OmniImageEncoder.forward` already does today
   (`models/qwen3_omni/components/image_encoder.py:154`); it is **not**
   extra work.
2. NCCL broadcast over a 1 GiB pixel buffer is bandwidth-bound on NVLink
   (~250 GB/s on H200) and finishes in milliseconds. The alternative —
   broadcasting CPU tensors via `gloo` over `tp.cpu_group` — pushes the
   bytes through host memory + ethernet/IPC and is at least an order of
   magnitude slower for the typical long-video workload that motivates
   this RFC.
3. Keeping the broadcast on the device group also matches what upstream
   SGLang does for `MultimodalDataItem.feature` inside `MMEncoder`: items
   are reconstructed on `cuda:gpu_id` before any TP collective runs
   (`disaggregation/encode_server.py:222-244`).

If a future model integrates a non-shm relay (`nccl`, `nixl`) that already
delivers GPU tensors, the leader-side `.to(device)` becomes a no-op.

This is structurally identical to upstream `relay_io.write_payload` /
`read_payload`, just over the SGLang TP collectives instead of the relay
backend, so that the data plane stays on the GPU bus and the metadata
stays small.

### Sketch

```python
import torch.distributed as dist
from sglang.srt.utils import broadcast_pyobj
from sglang_omni_v1.pipeline.relay_io import extract_tensors, restore_tensors


_RECV_ERROR_KIND = "encoder_recv_error"  # picklable string tag


def _recv_messages(
    self,
) -> tuple[list[IncomingMessage], BaseException | None]:
    """Drain inbox and broadcast inputs to TP followers.

    Never raises — returns (messages, error). The error is non-None if
    either rank failed during this iteration's recv. Drained messages
    are returned even on entry-rank failure so the scheduler can emit
    request-level errors against them in the unified handshake.

    Always run _strip_and_lift, even at tp_size == 1: the default shm
    relay delivers CPU tensors and upstream get_image_feature() /
    get_video_feature() only call .type(dtype), not .to(device).
    """
    if self.worker.tp_size == 1:
        # Cost-capped collect — same admission control SimpleScheduler
        # uses (max_batch_size / max_batch_wait_ms / max_batch_cost).
        # The TP path needs identical caps, otherwise long-video
        # batches that the local image encoder rejects today would
        # silently flow through and OOM in _strip_and_lift or forward.
        local = self._collect_batch_from_inbox()
        if not local:
            return local, None
        try:
            meta_msgs, tensor_lists, specs_lists = self._strip_and_lift(local)
        except Exception as exc:                          # noqa: BLE001
            return local, exc
        return self._reattach_lifted_tensors(meta_msgs, tensor_lists, specs_lists), None

    tp = self.worker.tp_group
    src_rank = tp.ranks[0]

    if self.worker.is_entry_rank:
        # Cost cap runs on the entry rank only — the broadcast below
        # ships the entry rank's already-bounded `local` list to
        # followers, so all ranks see the same admission decision.
        local = self._collect_batch_from_inbox()
        # _collect_batch_from_inbox is queue gets + cost-cap arithmetic
        # — no device work, no failure mode worth handshaking.
        # _strip_and_lift does the H2D copy + dtype coercion that can
        # OOM / TypeError; that's the failure we have to tell followers
        # about *before* the metadata broadcast, otherwise they block
        # on it forever and the runner can't see it
        # (mp_runner.py:332-342 only catches exit-code failures).
        try:
            meta_msgs, tensor_lists, specs_lists = self._strip_and_lift(local)
        except Exception as exc:                          # noqa: BLE001
            # Picklable tagged dict — survives broadcast_pyobj's
            # pickle.dumps / pickle.loads round-trip
            # (sglang utils/common.py:1286, 1309). Identity-based
            # sentinels (e.g. `object()`) would not, since pickle
            # reconstructs a fresh instance on each follower.
            broadcast_pyobj(
                [{"kind": _RECV_ERROR_KIND, "error": repr(exc)}],
                tp.rank, tp.cpu_group, src=src_rank,
            )
            return local, exc

        # specs_lists describes every tensor's (path, shape, dtype) only.
        # tensor_lists holds the GPU-resident tensors lifted from CPU shm.
        broadcast_pyobj(
            [meta_msgs, specs_lists],
            tp.rank, tp.cpu_group, src=src_rank,
        )

        # Allocation-ready handshake — see "Allocation-ready gather"
        # note below. Entry rank's tensors are already on device from
        # _strip_and_lift, so its allocation step is a no-op; it still
        # has to participate in the gather so any follower OOM unwinds
        # both ranks before the device broadcast fires.
        ok_flags = self._allocation_ready_gather(local_ok=True)
        if not all(ok_flags):
            return local, RuntimeError("peer-rank tensor allocation failed")

        for tensor_list in tensor_lists:
            for t in tensor_list:
                dist.broadcast(t, src=src_rank, group=tp.device_group)
        return (
            self._reattach_lifted_tensors(meta_msgs, tensor_lists, specs_lists),
            None,
        )

    # follower path
    payload = broadcast_pyobj([], tp.rank, tp.cpu_group, src=src_rank)
    if (
        payload
        and isinstance(payload[0], dict)
        and payload[0].get("kind") == _RECV_ERROR_KIND
    ):
        return [], RuntimeError(
            f"entry rank failed before metadata broadcast: {payload[0]['error']}"
        )
    meta_msgs, specs_lists = payload

    # Allocation-ready handshake: pre-allocate every receive tensor
    # *before* any device broadcast, then synchronize success across
    # ranks. If any rank's allocation fails (typically OOM on a long
    # video pixel buffer), every rank skips the device broadcast loop
    # and returns an error tuple. Without this gather, a follower OOM
    # mid-loop would leave the entry rank stuck waiting on the
    # corresponding `dist.broadcast` receiver.
    placeholders: list[list[torch.Tensor]] = []
    alloc_err: BaseException | None = None
    try:
        for specs in specs_lists:
            placeholders.append(
                [
                    torch.empty(spec.shape, dtype=spec.dtype,
                                device=self.worker.device)
                    for spec in specs
                ]
            )
    except Exception as exc:                              # noqa: BLE001
        alloc_err = exc

    ok_flags = self._allocation_ready_gather(local_ok=alloc_err is None)
    if not all(ok_flags):
        # Either local OOM or peer OOM — either way no rank issues
        # the device broadcast, so neither side blocks. Surface the
        # local error if we have one, otherwise a peer-failure stub.
        return [], (
            alloc_err if alloc_err is not None
            else RuntimeError("peer-rank tensor allocation failed")
        )

    rebuilt: list[IncomingMessage] = []
    for meta_msg, specs, ph_list in zip(meta_msgs, specs_lists, placeholders):
        tensor_dict = {}
        for spec, t in zip(specs, ph_list):
            dist.broadcast(t, src=src_rank, group=tp.device_group)
            tensor_dict[spec.path] = t
        meta_msg.data.data = restore_tensors(meta_msg.data.data, tensor_dict)
        rebuilt.append(meta_msg)
    return rebuilt, None


def _allocation_ready_gather(self, *, local_ok: bool) -> list[bool]:
    """Gather per-rank allocation-success flags on the TP CPU group."""
    flags = [False] * self.worker.tp_size
    dist.all_gather_object(
        flags, local_ok, group=self.worker.tp_group.cpu_group,
    )
    return flags


def _emit_error(
    self, messages: list[IncomingMessage], error: BaseException
) -> None:
    """Emit one OutgoingMessage(type="error") per drained request.

    Required because v1's existing schedulers expose a *single-request*
    helper (e.g. SimpleScheduler._emit_error(request_id: str, ...) at
    `scheduling/simple_scheduler.py:111`). Reusing that signature here
    would put a list[IncomingMessage] into OutgoingMessage.request_id,
    and Stage._drain_outbox_external (`pipeline/stage/runtime.py:466`)
    would TypeError on `out.request_id not in self._active_requests`
    (set membership requires hashable). Iterate explicitly:
    """
    for msg in messages:
        self.outbox.put(
            OutgoingMessage(
                request_id=msg.request_id,
                type="error",
                data=error,
            )
        )
```

The error sentinel is a **picklable tagged dict**, not an `object()`
identity sentinel. `broadcast_pyobj` does a full
`pickle.dumps`/`pickle.loads` round-trip
(`sglang/python/sglang/srt/utils/common.py:1286, 1309`); singleton
identity does not survive that, but `dict.get("kind") == "encoder_recv_error"`
does. The collective itself is the same `broadcast_pyobj` call
followers were already going to await, so the error rides the
existing channel.

`_recv_messages` deliberately **never raises**. Returning
`(messages, error)` lets the scheduler treat a recv-time failure
exactly like a forward-time failure: same `local_err` slot, same
`all_gather_object` handshake, same `_emit_error(messages, exc)`
emission against the drained requests. If `_recv_messages` raised
instead, the scheduler thread would die and `Stage._handle_scheduler_crash`
(`pipeline/stage/runtime.py:145`) would tear the whole stage down —
turning a single bad request into a stage-level abort.

#### Allocation-ready gather

The metadata `broadcast_pyobj` only synchronizes *what to receive*,
not *whether the receivers are ready*. The follower then has to call
`torch.empty(spec.shape, ..., device=cuda:0)` for every incoming
tensor, and on a long-video / multi-image batch that allocation can
OOM. Naïvely starting `dist.broadcast` from the entry rank
immediately after the metadata broadcast hits a deadlock in that
case: a follower OOMs mid-allocation and aborts its receive loop,
while the entry rank is already blocked inside an unmatched
`dist.broadcast` call.

Plan B inserts an `all_gather_object`-style "alloc ok?" handshake
between the metadata broadcast and the first `dist.broadcast`:

1. Followers pre-allocate **all** receive tensors up front (fail
   fast if any spec OOMs).
2. The entry rank participates in the gather as a no-op (its
   tensors already exist).
3. Both ranks gather their per-rank alloc-success boolean on
   `tp.cpu_group`.
4. If `not all(flags)`, every rank returns
   `(messages, error)` and the scheduler's outer `all_gather_object`
   handshake takes care of the rest — no device broadcast was ever
   issued, so no rank blocks.

This is a small, picklable collective added once per recv, and it
makes the device broadcast loop unconditionally safe to enter once
it starts.

`_strip_and_lift` calls `extract_tensors`, then for each extracted tensor
runs `t = t.to(self.worker.device, non_blocking=True)` and records
`(path, shape, dtype)` into a small `_TensorSpec` dataclass. The metadata
broadcast pickles only the spec list, not the tensors. `_reattach_lifted_tensors`
runs `restore_tensors` on the entry rank with the GPU-resident tensors so
that the entry rank's downstream `BatchPlan` sees the same device-resident
tensors the followers will reconstruct.

> **Why keep typed `_TensorSpec` instead of reusing the placeholder.**
> The placeholder dict `extract_tensors` produces stringifies dtype
> and device (`relay_io.py:48-49`: `"dtype": str(obj.dtype)` →
> `"torch.float16"`, not the `torch.dtype` object). Follower-side
> `torch.empty(shape, dtype=placeholder["dtype"], device=...)` would
> raise `TypeError: dtype must be a torch.dtype`. The implementation
> PR has two options: (a) carry a typed `_TensorSpec(path, shape,
> dtype: torch.dtype)` alongside `meta_msgs` as the sketch shows, or
> (b) introduce a string-to-`torch.dtype` parser and walk the
> placeholders directly. (a) is simpler and avoids one more parsing
> failure mode; the apparent "double bookkeeping" is the price of
> having `torch.dtype` objects on both sides without a parser.

### Why this is safe at the contract level

- `relay_io.extract_tensors` already walks dict trees that contain
  `pixel_values`, `image_grid_thw`, `pixel_values_videos`, `video_grid_thw`,
  `input_features`, `feature_attention_mask`, `audio_feature_lengths`. It
  is the same recursive structure preprocessing produces (`models/qwen3_omni/components/preprocessor.py`)
  and the same one `merge_for_thinker()` consumes.
- `tp_group.cpu_group` and `tp_group.device_group` are exactly the two
  groups SGLang's `OmniScheduler` already uses (`scheduling/omni_scheduler.py:336-360`),
  so we are not introducing new collectives.
- The broadcast is deterministic given identical input, so each rank can
  build the exact same `BatchPlan` independently after the broadcast lands.

## EncoderScheduler

### Public contract

```python
class EncoderScheduler:
    inbox: queue.Queue[IncomingMessage]
    outbox: queue.Queue[OutgoingMessage]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def abort(self, request_id: str) -> None: ...
```

The same shape as every other v1 scheduler — `Stage` does not need a
scheduler-type branch.

### Constructor

```python
class EncoderScheduler:
    def __init__(
        self,
        worker: "SGLangEncoderWorker",
        adapter: "EncoderAdapter",
        *,
        max_batch_size: int = 32,
        max_batch_wait_ms: int = 50,
        request_cost_fn: Callable[[Any], int] | None = None,
        max_batch_cost: int | None = None,        # activation_budget_bytes
    ):
        self.worker = worker
        self.adapter = adapter
        self.inbox = queue.Queue()
        self.outbox = queue.Queue()
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._request_cost_fn = request_cost_fn
        self._max_batch_cost = (
            max(int(max_batch_cost), 0) if max_batch_cost is not None else None
        )
        self._pending_messages: collections.deque[IncomingMessage] = collections.deque()
        self._running = False
```

The four batch-shaping knobs (`max_batch_size`, `max_batch_wait_ms`,
`request_cost_fn`, `max_batch_cost`) are intentionally identical to
`SimpleScheduler`'s — see `scheduling/simple_scheduler.py:38-49`. v1's
local image-encoder path already wires these (`models/qwen3_omni/stages.py:738-744`)
and the cost model has been tuned to keep activation peaks below OOM
on H200; the SGLang path must inherit the same admission control,
otherwise long-video / high-concurrency workloads will OOM during
forward even though the encoder is sharded.

### Loop

```python
def start(self) -> None:
    self._running = True
    while self._running:
        # ----------------------------------------------------------
        # Per-iteration error boundary covering recv, build_batch,
        # encode_batch, and slice_results. _recv_messages never
        # raises — it returns (messages, error) so a recv-time
        # failure can flow through the same handshake as a
        # forward-time failure. All four steps are deterministic
        # given the broadcast-equal `messages`; any one failing
        # without rank-sync would leave peers stuck at the next
        # collective.
        # ----------------------------------------------------------
        results = None
        local_err: BaseException | None = None

        messages, recv_err = self._recv_messages()
        if recv_err is not None:
            local_err = recv_err
        elif not messages:
            self._idle_check()
            continue
        else:
            try:
                plan = self.adapter.build_batch(messages)        # all ranks
                raw = self.worker.encode_batch(plan)             # all ranks
                if self.worker.is_entry_rank:
                    results = self.adapter.slice_results(raw, plan, messages)
            except Exception as exc:                             # noqa: BLE001
                local_err = exc

        # Synchronize error state across ranks. all_gather_object
        # marshals the picklable error flag; the exception object
        # itself stays local. tp_size==1 short-circuits.
        if self.worker.tp_size > 1:
            err_flags: list[bool] = [False] * self.worker.tp_size
            dist.all_gather_object(
                err_flags,
                local_err is not None,
                group=self.worker.tp_group.cpu_group,
            )
            any_err = any(err_flags)
        else:
            any_err = local_err is not None

        if any_err:
            if self.worker.is_entry_rank:
                # Emit one error per request that was drained on the
                # entry rank this iteration. `messages` is the entry
                # rank's drained list — populated even on recv-time
                # failure thanks to the (messages, error) return.
                self._emit_error(
                    messages,
                    local_err if local_err is not None
                    else RuntimeError("peer-rank encoder forward failed"),
                )
            # Followers do nothing here — Stage._drain_outbox_follower
            # already discards results. The next loop iteration starts
            # a fresh collective.
            continue

        if self.worker.is_entry_rank and results is not None:
            for msg, out in zip(messages, results):
                self.outbox.put(
                    OutgoingMessage(request_id=msg.request_id, type="result", data=out)
                )
```

### Batch admission control

`_collect_batch_from_inbox()` runs on the entry rank (single-rank case
included) and applies the three caps before the broadcast:

```python
def _collect_batch_from_inbox(self) -> list[IncomingMessage]:
    """Drain the inbox into a cost-bounded batch (entry rank only)."""
    first = self._next_message()                # blocks until a message or stop
    if first is None:
        return []
    if first.type != "new_request":
        # stream chunks / done signals are not batched here
        return [first]

    batch = [first]
    batch_cost = self._message_cost(first)
    deadline = time.monotonic() + self._max_batch_wait_s
    while len(batch) < self._max_batch_size:
        try:
            msg = self.inbox.get_nowait()
        except queue.Empty:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = self.inbox.get(timeout=remaining)
            except queue.Empty:
                break
        if msg.type != "new_request":
            self._pending_messages.append(msg)
            continue
        if self._max_batch_cost is not None:
            cost = self._message_cost(msg)
            if batch_cost + cost > self._max_batch_cost:
                self._pending_messages.appendleft(msg)
                break
            batch_cost += cost
        batch.append(msg)
    return batch

def _message_cost(self, msg: IncomingMessage) -> int:
    if self._request_cost_fn is None or msg.type != "new_request":
        return 0
    return max(int(self._request_cost_fn(msg.data)), 0)
```

This is the same control flow as `SimpleScheduler._collect_batch` —
intentionally so, to keep parity with the local fallback. The cost
function takes a `StagePayload` and returns a byte estimate; it is the
same `_create_image_encoder_request_cost_fn(model)` pattern v1 already
uses (`models/qwen3_omni/stages.py:144-166`).

### Where admission control runs in `_recv_messages`

`_collect_batch_from_inbox` is the **single point of truth** for the
admission decision: it runs only on the entry rank (or in the
single-rank case), and the broadcast that follows ships the
already-bounded list to followers. If both ranks ran the cost cap
independently, divergent decisions would desync the broadcast
structure — followers must never touch the inbox.

The single canonical sketch lives in
[Two-channel broadcast contract](#two-channel-broadcast-contract)
above; both lanes there call `_collect_batch_from_inbox()`. There is
no separate "admission-control variant" of `_recv_messages`.

### Responsibilities

1. Drain the stage inbox **on the entry rank only**. Entry rank is rank 0
   inside the SGLang TP group, matching `OmniScheduler.is_entry_rank`.
2. Broadcast metadata + tensors to follower ranks via the two-channel
   contract above.
3. Build a deterministic `BatchPlan` on all ranks using `adapter.build_batch`.
4. Run `worker.encode_batch(plan)` on every rank. Forward executes the
   upstream encoder; SGLang's `ColumnParallelLinear` / `RowParallelLinear`
   issue collectives internally.
5. Emit `OutgoingMessage` to the outbox **only on the entry rank**.
6. Run a **per-iteration error boundary** covering all four steps
   (`_recv_messages`, `build_batch`, `encode_batch`, `slice_results`)
   on every rank. `_recv_messages` returns `(messages, error)`
   instead of raising; the other three are wrapped in try/except.
   After the four steps, every rank exchanges its `local_err is None`
   flag through `dist.all_gather_object` on the TP CPU group. If any
   rank failed, the entry rank emits **one
   `OutgoingMessage(type="error")` per drained request** (so HTTP-500
   propagates per request, matching v1's existing request-level
   error path), and every rank `continue`s into the next loop
   iteration. Followers do **not** raise to crash the scheduler
   thread; that would trigger `Stage._handle_scheduler_crash`
   (`pipeline/stage/runtime.py:145`) and tear the whole stage down,
   which is a stage-level abort, not the request-level error this
   contract promises. `MultiProcessPipelineRunner._monitor_children`
   is the fallback only for cases where a process actually exits
   non-zero (e.g. a segfault inside the upstream model), not for
   request-level encoder errors.

### Why a new scheduler instead of extending SimpleScheduler

`SimpleScheduler` is the minimal "inbox -> fn -> outbox" runner used by
preprocessing, decode, code2wav, and any non-upstreamed local encoder. It is
deliberately single-process and TP-unaware. Two reasons not to fold TP into
it:

- **Mixed responsibility.** A scheduler that sometimes broadcasts and
  sometimes doesn't is the kind of conditional that grows accidental
  coupling. Keeping the two shapes separate keeps `SimpleScheduler` honest.
- **Different state model.** Encoder TP requires owning the SGLang TP CPU
  group, the SGLang TP device group, and the SGLang-loaded model.
  `SimpleScheduler` is an opaque callable wrapper. Fusing them would force
  every `SimpleScheduler` user to know about TP groups they never use.

`SimpleScheduler` stays as-is and remains the fallback path for
non-upstreamed encoders.

## SGLangEncoderWorker

`SGLangEncoderWorker` is a minimal SGLang-native encoder worker. It does
**not** start `MMEncoder` or any HTTP server: v1 already owns
preprocessing, control plane, relay, request lifecycle, and cache metadata.
The worker only owns SGLang's distributed state and the loaded upstream
encoder model.

### What we reuse from upstream

Patterned after `disaggregation/encode_server.py:184-244` (we copy the
calls, not the surrounding ZMQ/cache/transfer-engine machinery):

- `ServerArgs(encoder_only=True, ...)` + `set_global_server_args_for_scheduler`
- `ModelConfig.from_server_args(server_args)`
- `LoadConfig(load_format, download_dir, model_loader_extra_config,
  remote_instance_weight_loader_seed_instance_ip,
  remote_instance_weight_loader_seed_instance_service_port,
  remote_instance_weight_loader_send_weights_group_ports)` — full upstream
  argument set, see [LoadConfig fidelity](#loadconfig-fidelity).
- `init_distributed_environment(backend, world_size=tp_size, rank,
  distributed_init_method=..., local_rank=...)`
- `initialize_model_parallel(tensor_model_parallel_size=tp_size)`
- `initialize_dp_attention(server_args, model_config)`
- `get_model(model_config=..., load_config=..., device_config=...)`
- `get_tp_group()` for the TP CPU + device groups used by
  `EncoderScheduler`.

### GPU placement across `tp_size=1` and `tp_size>1` lanes

This is a load-bearing detail because v1's per-process CUDA env remap
in `pipeline/stage_process.py:222-249` is gated on `tp_size > 1`. The
contract Plan B requires is:

> The SGLangEncoderWorker process always sees exactly one CUDA device
> as `cuda:0`, regardless of `tp_size`. The configured physical GPU is
> mapped onto `cuda:0` by the launcher before `torch` is imported.
> Worker code uses `cuda_device=0`, `dist_local_rank=tp_rank` (which is
> `0` when `tp_size=1`).

This is the contract that lines up with how SGLang internally treats
`local_rank`:

- `init_distributed_environment` does **not** call
  `torch.cuda.set_device(local_rank)`
  (`distributed/parallel_state.py:1665-1745` only does
  `torch.distributed.init_process_group` and stores `local_rank` into
  the `GroupCoordinator`).
- `GroupCoordinator` uses `local_rank` two ways
  (`parallel_state.py:260-271`):
  1. **Device selection** — `device_id = 0 if SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS
     else local_rank`. The env var must be set, otherwise SGLang reads
     `local_rank` as a CUDA index.
  2. **Local-master identity** — `local_rank` becomes
     `GroupCoordinator.local_rank` and is read by callers as a
     `[0, world_size)` identity:
     - `custom_all_reduce_utils.py:280` —
       `get_world_group().local_rank == 0` decides who runs the P2P
       cache bootstrap. If every rank's `local_rank` is non-zero,
       no rank runs it.
     - `model_loader/weight_utils.py:812-820` —
       `sorted_files[local_rank::local_world_size]` shards checkpoint
       prefetch by node-local rank. A `local_rank=4, world_size=1`
       slice silently picks zero files instead of all files.

This rules out the earlier "`dist_local_rank = gpu_id` at `tp_size=1`"
shortcut: it would correctly select `cuda:gpu_id` via SGLang's device
fallback, but it would inject `gpu_id` (e.g. 4) into the
local-master / shard-index slot, which is undefined for
`world_size=1`.

#### Required launcher change

`backend="sglang"` stages — including `tp_size=1` — must reach the
worker through a process whose `CUDA_VISIBLE_DEVICES` has been remapped
to a single physical GPU and whose `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true`
is set, exactly as `_prepare_cuda_environment` already does for
`tp_size>1`.

The Phase-0 PR therefore extends the launcher path:

1. `StageProcessSpec` gains a flag, `single_visible_device: bool = False`
   (default preserves current behaviour for SimpleScheduler /
   OmniScheduler stages).
2. `MultiProcessPipelineRunner._build_stage_groups` sets the flag from
   the **resolved** factory args **before spawning the child process**.
   `_resolve_factory_args` (`config/compiler.py:138-158`) merges
   `runtime_overrides` over `stage_cfg.factory_args`, so we must read
   from the merged result, not raw `StageConfig.factory_args`:

   ```python
   base_factory_args = _resolve_factory_args(stage_cfg, config)  # already line 73
   spec.single_visible_device = (
       base_factory_args.get("backend", "local") in {"sglang", "auto"}
   )
   ```

   Reading `stage_cfg.factory_args.get("backend")` directly would miss
   the case where the user flips a stage to `backend="sglang"` via
   `PipelineConfig.runtime_overrides` (CLI / config-file path) without
   editing the StageConfig itself, leaving `single_visible_device=False`
   while the worker still expects to be the only visible CUDA device.

   The runner cannot wait for `_resolve_backend(...)` because that runs
   inside the factory in the child process, after `torch` is imported.
   `CUDA_VISIBLE_DEVICES` must be set before that, so the launcher
   takes the conservative pre-spawn decision: any stage whose resolved
   factory args request `backend="sglang"` **or** `backend="auto"` gets
   the remap, even if `_resolve_backend("auto", ...)` later falls back
   to `"local"`. That fallback case is harmless — the local HF tower
   defaults to `device="cuda"` (current device) and runs on the only
   visible GPU, which is the configured physical id by construction.
   `backend="local"` (explicit) keeps the current "see all GPUs" behaviour
   so existing single-process SimpleScheduler stages are untouched.
3. `pipeline/stage_process.py:get_stage_process_env` changes its early
   return:

   ```python
   if spec.tp_size <= 1 and not spec.single_visible_device:
       return {}
   ```

   The rest of the function — `CUDA_VISIBLE_DEVICES` remap, the two
   SGLANG env vars, and the `factory_args["gpu_id"] = 0` rewrite in
   `_prepare_cuda_environment` — runs unchanged.
4. `serve/launcher.py` forces multi-process mode whenever any stage
   asks for the SGLang backend, even if every stage is on the same
   GPU and every `tp_size == 1`. The current condition
   (`launcher.py:149-150`) is

   ```python
   needs_mp = len(gpu_ids) > 1 or any_tp
   ```

   which routes a single-stage, single-GPU `backend="sglang", tp_size=1`
   pipeline through `compile_pipeline()` instead of
   `MultiProcessPipelineRunner`. That single-process path constructs
   `Stage` and calls the factory (`config/compiler.py:65-72`) inside
   the **launcher process itself**, never crossing
   `StageProcessSpec` / `_prepare_cuda_environment`, so:

   - The single-visible-device remap from step 1 never fires —
     `gpu=4, tp_size=1` would silently load the encoder on physical
     GPU 0 because the worker fixes `cuda_device=0`.
   - Multiple `SGLangEncoderWorker` instances in the same process would
     fight over the global `init_distributed_environment` state, which
     is module-level inside SGLang and can only be initialized once
     per process.

   The fix is to extend the condition:

   ```python
   any_sglang_backend = any(
       _resolve_factory_args(s, pipeline_config).get("backend", "local")
       in {"sglang", "auto"}
       for s in pipeline_config.stages
   )
   needs_mp = len(gpu_ids) > 1 or any_tp or any_sglang_backend
   ```

   Reading from the resolved factory args matches the same source the
   `single_visible_device` flag uses (step 2), so a `runtime_overrides`
   flip is honored.
5. `config/compiler.py:compile_pipeline` rejects any stage whose
   resolved `factory_args["backend"]` is `"sglang"` or `"auto"`,
   **and** any stage with `stage_cfg.tp_size > 1`. The single-process
   compile path is by construction incompatible with the SGLang
   encoder worker (no per-rank subprocess) and equally unable to
   honor TP for **any** factory: `_resolve_factory_args` only injects
   `model_path` / `gpu_id`, never `tp_rank` / `tp_size` /
   `nccl_port` (`config/compiler.py:138-158`). A direct
   `compile_pipeline(config_with_thinker_tp=2)` call would silently
   instantiate the thinker factory with its default `tp_size=1` — TP
   completely lost, no error. The blanket `tp_size > 1` reject turns
   that silent downgrade into an early `ValueError`. This does not
   regress thinker / talker / encoder TP under `serve/launcher.py`
   because `any_tp → needs_mp` (`launcher.py:149`) routes those
   configs to `MultiProcessPipelineRunner` before `compile_pipeline`
   is even called.
6. **TP preflight reject (two-layer)** in
   `MultiProcessPipelineRunner._build_stage_groups`:

   ```python
   _TP_LAUNCH_PARAMS = {"tp_rank", "tp_size", "nccl_port"}

   for stage_cfg in pipeline_config.stages:
       if stage_cfg.tp_size <= 1:
           continue
       factory = import_string(stage_cfg.factory)
       params = inspect.signature(factory).parameters

       # Layer 1: any TP stage's factory must accept the TP launch
       # kwargs that mp_runner is about to inject.
       missing = _TP_LAUNCH_PARAMS - params.keys()
       if missing:
           raise ValueError(
               f"Stage {stage_cfg.name!r}: tp_size={stage_cfg.tp_size} > 1 "
               f"but factory {stage_cfg.factory!r} does not accept TP "
               f"launch parameters {sorted(missing)}. This factory is "
               f"not TP-capable; reduce tp_size to 1 or use a factory "
               f"that accepts tp_rank/tp_size/nccl_port."
           )

       # Layer 2: if the factory is a backend-aware encoder factory,
       # only backend="sglang" implements actual TP.
       if "backend" in params:
           resolved = _resolve_factory_args(stage_cfg, pipeline_config)
           backend = resolved.get("backend", "local")
           if backend != "sglang":
               raise ValueError(
                   f"Stage {stage_cfg.name!r}: tp_size={stage_cfg.tp_size} "
                   f"requires backend='sglang' (got {backend!r}). "
                   f"The local encoder path does not implement TP and "
                   f"would silently spawn TP-rank processes that each "
                   f"run a full local forward, with all but rank 0 "
                   f"discarded."
               )
   ```

   The two layers cover distinct failure modes:

   - **Layer 1** catches "factory has no idea about TP". Example: a
     `SimpleScheduler` callable like
     `create_aggregate_executor()` (no TP params in signature)
     mis-configured with `tp_size=2`. Without this layer, the
     launcher would still hit `any_tp → needs_mp`, spawn N
     subprocesses, then fail in each child either at factory
     argument-binding time (mp_runner injects `tp_rank/tp_size/nccl_port`
     kwargs the factory does not accept → `TypeError`) or — if the
     factory uses `**kwargs` — at follower-IO time when the stage
     can't actually fan-out batches. Layer 1 fails loud in the main
     process before any spawn.
   - **Layer 2** catches "factory has a backend knob but the user
     left it on local". Encoder factories accept `backend` *and*
     `tp_rank/tp_size/nccl_port`, so Layer 1 alone would pass them
     through; Layer 2 enforces that TP only happens when the user
     explicitly opts into the SGLang backend.

   Concrete pass / fail matrix:

   | factory                             | tp_size | backend | result |
   |---|---|---|---|
   | thinker / talker (TP params, no `backend`) | 2 | n/a | passes |
   | image_encoder (TP params + `backend`)      | 2 | "sglang" | passes |
   | image_encoder (TP params + `backend`)      | 2 | "local" / "auto" / unset | rejects (Layer 2) |
   | aggregate / preprocessing (no TP params)    | 2 | n/a | rejects (Layer 1) |
   | any factory                                  | 1 | any | passes |

#### Backend resolution contract

All four launcher-side checks above (`single_visible_device` flag in
`mp_runner`, `needs_mp` in `serve/launcher`, `compile_pipeline`
reject) read the stage's backend through one and only one source:

```python
_resolve_factory_args(stage_cfg, config).get("backend", "local")
```

`_resolve_factory_args` (`config/compiler.py:138-158`) merges
`stage_cfg.factory_args` with `config.runtime_overrides[stage.name]`
and only injects `model_path` / `gpu_id` from the factory signature.
**It deliberately does not introspect any other factory signature
defaults.** The implication is load-bearing:

- The factory function's `backend` signature default is irrelevant to
  the launcher decision.
- A StageConfig that wants the SGLang backend **must** put
  `backend="sglang"` (or `"auto"`) into `factory_args` or
  `runtime_overrides`. Relying on a future signature-default flip
  would silently desync launcher (`"local"`) from factory body
  (`"auto"`).
- Tests for any of the launcher checks should provide a StageConfig
  with explicit `factory_args["backend"]` set, never lean on signature
  defaults.

After this change, `SGLangEncoderWorker` sees `cuda:0` in both lanes
and the worker's GPU placement collapses to one rule:

```python
cuda_device = 0
dist_local_rank = tp_rank        # 0 in tp_size=1 lane, [0, tp_size) in tp>1
```

`base_gpu_id`, `set_device`, `DeviceConfig.gpu_id`, `self.device`, and
`local_rank` all use these two values directly.

Phase 1 parity testing must:

- exercise a non-zero `gpu` at `tp_size=1` (e.g. `gpu=4`) and assert
  three things at once: the child's
  `os.environ["CUDA_VISIBLE_DEVICES"] == "4"` (the launcher remap fired),
  `next(model.parameters()).device.index == 0` (the model loaded onto
  the only visible CUDA device, which appears as `cuda:0` from inside
  the child), and `get_world_group().local_rank == 0` (the
  local-master / shard-index slot is the rank 0 we asked for, not the
  physical GPU id);
- exercise `tp_size=2` and assert `get_world_group().local_rank` is
  unique per rank (0 and 1, not 0 and 0).

### Distributed init is unconditional

Upstream `MMEncoder.__init__` is unconditional — `init_distributed_environment`
runs even at `tp_size=1`. SGLang's parallel layers (`ColumnParallelLinear`
/ `RowParallelLinear`) call `get_tp_group()` during their own `__init__`,
and `get_tp_group()` asserts the group has been initialized
(`parallel_state.py:1482`). Skipping init at `tp_size=1` would crash at
model load time with `tensor model parallel group is not initialized`.

`SGLangEncoderWorker` therefore initializes distributed state **always**:

```python
class SGLangEncoderWorker:
    def __init__(
        self,
        *,
        model_path: str,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int | None,
        dtype: str | None = None,
        load_format: str | None = None,
        server_args_overrides: dict[str, Any] | None = None,
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.is_entry_rank = (tp_rank == 0)

        # tp_size=1 also gets a real init: see `Distributed init is
        # unconditional`.  Use a free-port loopback when the parent did not
        # allocate one (single-rank case).
        #
        # SGLang's ServerArgs.dist_init_addr expects `host:port`
        # (NetworkAddress.parse, network.py:474), and ServerArgs internally
        # composes the `tcp://` URL via NetworkAddress.to_tcp() when it
        # needs one (network.py:437). torch.distributed, however, expects
        # the full `tcp://host:port` URL. Keep the two forms separate.
        port = nccl_port if nccl_port is not None else _pick_free_port()
        dist_addr = f"127.0.0.1:{port}"               # for ServerArgs
        dist_init_method = f"tcp://{dist_addr}"       # for torch init

        # See "GPU placement across tp_size=1 and tp_size>1 lanes" for
        # why this collapses to (0, tp_rank) once the launcher remaps
        # CUDA_VISIBLE_DEVICES in both lanes.
        cuda_device = 0
        dist_local_rank = tp_rank
        self.device = torch.device(f"cuda:{cuda_device}")

        # Worker-managed kwargs that build_sglang_encoder_server_args is
        # about to receive as explicit positional/keyword arguments.
        # We must reject these in `server_args_overrides` BEFORE the
        # **splat below — otherwise Python raises TypeError "got multiple
        # values for keyword argument" before our helper's protected-key
        # check ever runs.
        overrides = dict(server_args_overrides or {})
        worker_managed = {
            "model_path", "tp_size", "base_gpu_id", "dist_init_addr",
            "dtype", "load_format",
        }
        clobbered = sorted(worker_managed & overrides.keys())
        if clobbered:
            raise ValueError(
                f"server_args_overrides cannot set worker-managed keys "
                f"{clobbered}. These are derived from StageConfig and "
                f"factory parameters; pass them through StageConfig "
                f"(model_path, tp_size, gpu, dtype, load_format) instead."
            )

        server_args = build_sglang_encoder_server_args(
            model_path=model_path,
            tp_size=tp_size,
            base_gpu_id=cuda_device,
            dist_init_addr=dist_addr,             # host:port for SGLang
            dtype=dtype,
            load_format=load_format,
            **overrides,                          # forwards model_loader_extra_config etc.
        )
        set_global_server_args_for_scheduler(server_args)

        self.model_config = ModelConfig.from_server_args(server_args)
        # Full upstream LoadConfig argument set
        # (matches disaggregation/encode_server.py:202-208).
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=(
                server_args.remote_instance_weight_loader_seed_instance_ip
            ),
            remote_instance_weight_loader_seed_instance_service_port=(
                server_args.remote_instance_weight_loader_seed_instance_service_port
            ),
            remote_instance_weight_loader_send_weights_group_ports=(
                server_args.remote_instance_weight_loader_send_weights_group_ports
            ),
        )

        torch.cuda.set_device(cuda_device)

        # Always run, including tp_size == 1.
        # `local_rank` here is identity (used for local-master checks),
        # NOT a CUDA index. Passing tp_rank lets followers see themselves
        # as non-zero local_rank in custom_all_reduce_utils.py:280 and
        # friends. Device selection is governed by
        # SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS in the tp>1 lane.
        init_distributed_environment(
            backend=get_default_distributed_backend("cuda"),
            world_size=tp_size,
            rank=tp_rank,
            distributed_init_method=dist_init_method,   # tcp://host:port for torch
            local_rank=dist_local_rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=tp_size)
        initialize_dp_attention(server_args, self.model_config)
        self.tp_group = get_tp_group()

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(device="cuda", gpu_id=cuda_device),
        )

    @torch.no_grad()
    def encode_batch(self, plan: "BatchPlan") -> Any:
        return plan.adapter.run_feature(self.model, plan)
```

### `build_sglang_encoder_server_args`

The existing `build_sglang_server_args` (`scheduling/sglang_backend/server_args_builder.py:10`)
is shaped for AR engines: it requires a positional `context_length` and
defaults `mem_fraction_static=0.7`, `max_running_requests=16`,
`max_prefill_tokens=16384`. None of those have a clean meaning for
encoder-only stages (no KV pool, no AR running queue, no prefill token
budget). Reusing the AR builder also fails at runtime because Phase 0's
factory does not know a meaningful `context_length` for the encoder
process.

We therefore add a sibling helper next to the AR builder:

```python
# sglang_omni_v1/scheduling/sglang_backend/server_args_builder.py

# Worker invariants that must NOT be reachable from server_args_overrides.
# Mutating these would either invalidate GPU placement, change the
# parallelism axis we promised, or flip the encoder-vs-language-only
# fork. See "GPU placement..." and Open Question 2 (encoder DP).
_ENCODER_PROTECTED_KEYS = frozenset({
    # Parallelism / placement
    "tp_size",
    "pp_size",
    "dp_size",
    "moe_dp_size",
    "ep_size",
    "attn_cp_size",
    "moe_dense_tp_size",
    "nnodes",
    "node_rank",
    "base_gpu_id",
    "dist_init_addr",
    # Encoder-only fork
    "encoder_only",
    "language_only",
    "mm_enable_dp_encoder",
    "enable_dp_attention",
    "enable_dp_attention_local_control_broadcast",
    "enable_dp_lm_head",
    "disable_cuda_graph",
    "device",
    # AR-only knobs that have no meaning for an encoder-only worker.
    # Locked so users cannot reintroduce SGLang AR memory semantics
    # through server_args_overrides — the encoder path explicitly does
    # not own a KV pool, an AR running queue, or a chunked-prefill
    # budget, and any value set here would be silently accepted by
    # ServerArgs but would diverge from what the worker actually does.
    "mem_fraction_static",
    "max_running_requests",
    "max_prefill_tokens",
    "chunked_prefill_size",
    "context_length",
})


def build_sglang_encoder_server_args(
    model_path: str,
    *,
    tp_size: int,
    base_gpu_id: int,
    dist_init_addr: str,
    dtype: str | None = None,
    load_format: str | None = None,
    **overrides: Any,
) -> ServerArgs:
    """ServerArgs configured for an encoder-only worker.

    Distinct from build_sglang_server_args because encoder stages do not
    have a meaningful context_length / mem_fraction_static / running queue.

    Raises ValueError if `overrides` tries to mutate a protected
    invariant (parallelism shape, GPU placement, encoder-only fork).
    """
    bad = sorted(_ENCODER_PROTECTED_KEYS & overrides.keys())
    if bad:
        raise ValueError(
            f"server_args_overrides cannot override protected keys: {bad}. "
            f"These are decided by the worker / pipeline runner; pass them "
            f"through StageConfig (tp_size, gpu) instead."
        )

    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": tp_size,
        "pp_size": 1,
        "base_gpu_id": base_gpu_id,
        "dist_init_addr": dist_init_addr,
        "encoder_only": True,
        "language_only": False,
        "mm_enable_dp_encoder": False,    # MVP: TP only; see Open Questions
        "disable_cuda_graph": True,       # variable shapes; no piecewise CG yet
        "random_seed": 123,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if load_format is not None:
        kwargs["load_format"] = load_format
    kwargs.update(overrides)
    return ServerArgs(**kwargs)
```

The protected-key reject covers:

- `tp_size`, `pp_size`, `dp_size`, `moe_dp_size`, `ep_size`,
  `attn_cp_size`, `moe_dense_tp_size`, `nnodes`, `node_rank`,
  `base_gpu_id`, `dist_init_addr` — pipeline runner decides these from
  `StageConfig.tp_size` / `gpu` and the per-stage NCCL port allocator.
  Phase 0 starts exactly `tp_size` local ranks; DP, EP, attention-CP, and
  multinode shapes are out of scope and cannot be activated through
  `server_args_overrides`.
- `encoder_only` / `language_only` — flipping these would route through
  a different upstream model factory and break the whole RFC.
- `mm_enable_dp_encoder` — Phase 0–2 require encoder TP only; allowing
  this through `overrides` would silently violate the rejection rule
  the factory layer also enforces.
- `enable_dp_attention`, `enable_dp_attention_local_control_broadcast`,
  `enable_dp_lm_head` — these change SGLang's data-parallel collective
  layout while the v1 launcher still only constructs the TP group
  described in this RFC.
- `disable_cuda_graph` — Phase 0 requires variable shapes; piecewise
  CUDA graph for encoder ViT lands later (PR #15320 / #16785 area).
- `device` — `SGLangEncoderWorker` hardcodes `DeviceConfig(device="cuda", ...)`
  and `get_default_distributed_backend("cuda")`. Letting `overrides`
  flip `ServerArgs.device` to `cpu/npu/xpu/mps` (the values
  `server_args.py:1218-1244` recognizes) would split the two: SGLang's
  internals would think the worker is on CPU/NPU while we still issue
  CUDA distributed calls. Cross-backend encoder support is a separate
  workstream and should not be reachable through a one-line override.

Loader / processor knobs (`model_loader_extra_config`,
`remote_instance_weight_loader_*`, `disable_fast_image_processor`, etc.)
are **not** protected and pass through unchanged via
`server_args_overrides` — those are exactly the fields users need to
forward for FP8 / NVFP4 / remote-streaming deployments. `dtype` and
`load_format` are also supported, but as top-level factory / worker
arguments rather than `server_args_overrides`, to avoid duplicate keyword
binding before the protected-key helper runs.

`SGLangEncoderWorker.__init__` calls this helper directly. AR-only
knobs (`mem_fraction_static`, `max_running_requests`,
`max_prefill_tokens`, `chunked_prefill_size`, `context_length`) are
intentionally absent from the helper signature **and** are listed in
`_ENCODER_PROTECTED_KEYS` so they are also unreachable through
`server_args_overrides`. `ServerArgs` falls back to its own defaults
for those AR fields, and the encoder-only code path never reads them
— protecting them keeps a stale-AR-knob from silently propagating
into `ServerArgs` even though the worker ignores it.

### LoadConfig fidelity

`SGLangEncoderWorker.__init__` constructs `LoadConfig` with the same six
fields upstream `MMEncoder.__init__` does
(`disaggregation/encode_server.py:202-208`):

```python
LoadConfig(
    load_format=server_args.load_format,
    download_dir=server_args.download_dir,
    model_loader_extra_config=server_args.model_loader_extra_config,
    remote_instance_weight_loader_seed_instance_ip=(
        server_args.remote_instance_weight_loader_seed_instance_ip
    ),
    remote_instance_weight_loader_seed_instance_service_port=(
        server_args.remote_instance_weight_loader_seed_instance_service_port
    ),
    remote_instance_weight_loader_send_weights_group_ports=(
        server_args.remote_instance_weight_loader_send_weights_group_ports
    ),
)
```

Why all six, not just `load_format` + `download_dir`:

- `model_loader_extra_config` carries quant-config overrides, custom
  loader plugins, and extra-file paths. Some Qwen3-Omni internal
  deployments depend on it for FP8 / NVFP4 weight files.
- The three `remote_instance_weight_loader_*` fields wire SGLang's
  remote-instance weight-streaming protocol. If a SGLang deployment uses
  remote weight streaming for the LLM, the encoder process should
  participate too; passing the fields keeps that path open.

`build_sglang_encoder_server_args` therefore accepts these values via its
`**overrides`. Phase 0 default behavior matches a local checkpoint load —
the four extra fields default to `None`/empty in `ServerArgs`, so users
who don't use these features pay nothing. Users who do can add them
through `factory_args` and they reach `LoadConfig` unchanged.

### Why not subclass / instantiate `MMEncoder`

`MMEncoder` does much more than what we need: ZMQ schedule socket, mooncake
transfer engine, multimodal cache, embedding-to-send queue, async send
timeout, image processor on GPU. Every one of those duplicates work v1
already owns. Importing the whole class would make us its consumer plus
re-implementer. Reading the lower half of `MMEncoder.__init__` and copying
just the eight calls listed above is genuinely smaller and lets v1 keep
ownership of the request lifecycle. (See
[Open Questions](#open-questions) about whether SGLang main should expose a
small `EncoderModelWorker` helper to remove the copy.)

## EncoderAdapter and BatchPlan

The earlier draft tried to flow `list[MultimodalDataItem]` through
`run_feature`, but `build_items` returns one such list **per request**, so
the batched call was implicitly `list[list[MultimodalDataItem]]` and the
per-modality filter inside `run_feature` was wrong-typed. The corrected
contract goes through an explicit `BatchPlan`.

### Skip / cache semantics (from preprocessor)

The preprocessor does **not** drop missing modalities — it stamps each
encoder stage's `encoder_inputs` slot with a sentinel:

```python
encoder_inputs["audio_encoder"] = {"_skip": True, "_result": {}}    # missing-modality
encoder_inputs["audio_encoder"] = {..., "cache_key": "..."}         # cacheable
encoder_inputs["audio_encoder"] = {..."input_features": tensor, ...} # active
```

(See `models/qwen3_omni/components/preprocessor.py:454-468` and
`models/qwen3_omni/request_builders.py:36-53`.)

The current local v1 path consumes this through `build_encoder_request()`,
which returns an `EncoderRequestData(model_inputs, cache_key,
skip_result)`. When `skip_result is not None` the local
`_run_single_encoder_payload` short-circuits and never touches tensors.
**The SGLang adapter must do the same**, otherwise the very common
"image-only" request will `KeyError` on
`inputs["input_features"]` inside the audio_encoder stage, and "audio-only"
will `KeyError` inside the image_encoder stage.

The adapter therefore enters through `build_encoder_request()`, not raw
`encoder_inputs`, and the `BatchPlan` carries per-request skip results so
that:

- Skip requests never contribute `MultimodalDataItem` to the active items.
- `run_feature` sees only active items, with the option of an empty list.
  When the entire batch is skip-only (e.g. a text-only batch arriving at
  audio_encoder during co-routing), `run_feature` returns
  `{"image": None, "video": None, "audio": None}` and never calls into
  the upstream model.
- `slice_results` returns the preserved `skip_result` for skip requests
  and slices the raw embedding for active requests.

Cache (`cache_key`) is intentionally **not** wired into the SGLang path in
Phase 0: a TP-aware cache requires the entry rank to broadcast the
hit/miss decision before forward, otherwise rank A hitting and rank B
missing would corrupt the collective. Cache becomes a Phase 2+ follow-up
once the adapter is stable; until then SGLang path passes through every
non-skip request to forward, and the existing local path keeps its cache.

### Types

```python
@dataclass(slots=True)
class _TensorSpec:
    path: str
    shape: tuple[int, ...]
    dtype: torch.dtype

@dataclass(slots=True)
class RequestSpan:
    """Slot for one request inside a batch.

    Exactly one of skip_result or active_offsets is populated.
    """
    request_id: str
    skip_result: dict | None = None        # preserved from preprocessor _skip
    image_rows: int = 0                    # number of image MM items contributed
    video_rows: int = 0
    audio_rows: int = 0
    image_token_count: int = 0             # for visual splitting
    video_token_count: int = 0
    # Audio splitting: keep both the *unpadded* per-request feature lengths
    # (passed back to merge_for_thinker as `audio_feature_lengths`) and the
    # downsampled output lengths used to slice the encoder output along the
    # token axis.
    audio_feature_lengths: torch.Tensor | None = None
    audio_output_lengths: torch.Tensor | None = None

@dataclass(slots=True)
class BatchPlan:
    adapter: "EncoderAdapter"
    image_items: list[MultimodalDataItem]   # flat across active requests only
    video_items: list[MultimodalDataItem]
    audio_items: list[MultimodalDataItem]
    spans: list[RequestSpan]                # one per request, in input order

    @property
    def is_empty(self) -> bool:
        return not (self.image_items or self.video_items or self.audio_items)
```

### Adapter protocol

```python
class EncoderAdapter(Protocol):
    stage_name: str

    def build_batch(
        self,
        messages: list[IncomingMessage],
    ) -> BatchPlan: ...

    def run_feature(
        self,
        model: Any,
        plan: BatchPlan,
    ) -> dict[str, torch.Tensor | None]: ...
        # returns {"image": ..., "video": ..., "audio": ...}
        # any modality not present in the plan is None

    def slice_results(
        self,
        raw: dict[str, torch.Tensor | None],
        plan: BatchPlan,
        messages: list[IncomingMessage],
    ) -> list[StagePayload]: ...
```

### Image / video adapter

```python
def build_batch(self, messages):
    images: list[MultimodalDataItem] = []
    videos: list[MultimodalDataItem] = []
    spans: list[RequestSpan] = []
    for msg in messages:
        state = PipelineState.from_dict(msg.data.data)
        request = build_encoder_request(state, stage_name="image_encoder")

        # Phase-0: respect preprocessor _skip; cache_key is ignored.
        if request.skip_result is not None:
            spans.append(RequestSpan(
                request_id=msg.request_id,
                skip_result=request.skip_result,
            ))
            continue

        inputs = request.model_inputs
        n_img = n_vid = 0
        img_tokens = vid_tokens = 0
        if isinstance(inputs.get("pixel_values"), torch.Tensor):
            it = MultimodalDataItem(modality=Modality.IMAGE,
                                    feature=inputs["pixel_values"])
            it.image_grid_thw = inputs["image_grid_thw"]
            images.append(it)
            n_img = int(inputs["image_grid_thw"].shape[0])
            img_tokens = int(
                (inputs["image_grid_thw"].prod(-1) // self._merge).sum().item()
            )
        if isinstance(inputs.get("pixel_values_videos"), torch.Tensor):
            v = MultimodalDataItem(modality=Modality.VIDEO,
                                   feature=inputs["pixel_values_videos"])
            v.video_grid_thw = inputs["video_grid_thw"]
            videos.append(v)
            n_vid = int(inputs["video_grid_thw"].shape[0])
            vid_tokens = int(
                (inputs["video_grid_thw"].prod(-1) // self._merge).sum().item()
            )
        spans.append(RequestSpan(
            request_id=msg.request_id,
            image_rows=n_img, video_rows=n_vid,
            image_token_count=img_tokens,
            video_token_count=vid_tokens,
        ))
    return BatchPlan(self, images, videos, [], spans)


def run_feature(self, model, plan):
    if plan.is_empty:
        return {"image": None, "video": None, "audio": None}
    image_embed = (
        model.thinker.get_image_feature(plan.image_items)
        if plan.image_items else None
    )
    video_embed = (
        model.thinker.get_video_feature(plan.video_items)
        if plan.video_items else None
    )
    return {"image": image_embed, "video": video_embed, "audio": None}


def slice_results(self, raw, plan, messages):
    out: list[StagePayload] = []
    img_row = img_tok = vid_row = vid_tok = 0
    for span, msg in zip(plan.spans, messages):
        state = PipelineState.from_dict(msg.data.data)
        if span.skip_result is not None:
            apply_encoder_result(
                state, stage_name="image_encoder", result=span.skip_result,
            )
            out.append(_payload_with_state(msg.data, state))
            continue

        result = self._slice_visual(
            raw["image"], raw["video"], plan, span,
            img_row, img_tok, vid_row, vid_tok,
        )
        img_row += span.image_rows
        img_tok += span.image_token_count
        vid_row += span.video_rows
        vid_tok += span.video_token_count
        apply_encoder_result(state, stage_name="image_encoder", result=result)
        out.append(_payload_with_state(msg.data, state))
    return out
```

`_slice_visual` reuses the existing `_split_visual_features` /
`_split_visual_multiscale` helpers from `models/qwen3_omni/stages.py`,
moved into `encoder_adapters.py` so the SGLang path does not depend on
private helpers in `stages.py`.

> Note (Cheng): we deliberately keep `get_image_feature` and
> `get_video_feature` as separate calls. Upstream
> `Qwen3VLMoeForConditionalGeneration` exposes them as two methods
> (`qwen3_vl.py:1193, 1212`), and PR #14907 (chunked vit attention) hooks
> into them at this granularity. Fusing them would silently break that
> hook.

### Audio adapter

Upstream `get_audio_feature()` (`models/qwen3_omni_moe.py:461-492`) cats
all `item.feature` along dim 0 in its first line, so different audio
clips with different time dimensions cannot be passed in raw — the cat
fails on shape mismatch. v1's local path solves this by padding every
clip to the batch's max time before cat (`models/qwen3_omni/stages.py:553-659`).
The SGLang adapter must replicate that contract; the Phase-0 plan is to
move `_normalize_audio_request_tensors`, `_pad_audio_features`, and
`_pad_audio_mask` from `stages.py` into `encoder_adapters.py` and reuse
them — with **one** mandatory fix during the move:

`_normalize_audio_request_tensors` (`stages.py:533`) currently
synthesizes a fallback mask via
`torch.arange(time_dim, dtype=torch.long).unsqueeze(0)`, which lands
on CPU. In the local v1 path that is fine because the helper runs
before any GPU lift. In the SGLang Plan B path, however,
`_recv_messages → _strip_and_lift` has already moved
`audio_feature_lengths` to `cuda:0` before the adapter sees the
request, so the subsequent `steps < lengths.unsqueeze(1)` mixes a
CPU `steps` with a GPU `lengths` and raises a device-mismatch error
on any input that has `audio_feature_lengths` but no
`feature_attention_mask`.

The moved helper must take its arange device from `lengths`:

```python
steps = torch.arange(time_dim, dtype=torch.long, device=lengths.device).unsqueeze(0)
mask = steps < lengths.unsqueeze(1)
```

This keeps the local v1 path unchanged (lengths is CPU there →
arange is CPU) while making the SGLang path correct (lengths is
GPU there → arange follows). After this fix all three helpers are
device-agnostic and safe to share between paths.

```python
def build_batch(self, messages):
    spans: list[RequestSpan] = []
    normalized: list[dict[str, torch.Tensor]] = []   # active requests only

    for msg in messages:
        state = PipelineState.from_dict(msg.data.data)
        request = build_encoder_request(state, stage_name="audio_encoder")

        if request.skip_result is not None:
            spans.append(RequestSpan(
                request_id=msg.request_id,
                skip_result=request.skip_result,
            ))
            continue

        # Reuses the v1 helper (after the device-aware fix). Returns
        # (features [B, mel, time], mask [B, time] bool, lengths [B]
        # long) on whichever device EncoderScheduler._strip_and_lift
        # already moved them to — typically `cuda:0` in the SGLang
        # Plan B path, CPU in the local v1 path. The synthesized
        # fallback mask now allocates `arange` on `lengths.device`,
        # so the helper works on either device without surgery.
        features, mask, lengths = _normalize_audio_request_tensors(request)
        out_lens = _get_feat_extract_output_lengths(lengths)
        spans.append(RequestSpan(
            request_id=msg.request_id,
            audio_rows=int(lengths.shape[0]),
            audio_feature_lengths=lengths,         # unpadded, per-request
            audio_output_lengths=out_lens,
        ))
        normalized.append({"features": features, "mask": mask, "lengths": lengths})

    if not normalized:
        return BatchPlan(self, [], [], [], spans)

    # Pad every active request to the batch-wide max time so torch.cat
    # inside upstream get_audio_feature succeeds. The padded positions
    # are masked out by `feature_attention_mask` and discarded inside
    # `get_audio_feature` via `input_features.permute(0,2,1)[mask.bool()]`.
    max_time = max(int(item["features"].shape[-1]) for item in normalized)
    audios: list[MultimodalDataItem] = []
    for item in normalized:
        feat = _pad_audio_features(item["features"], max_time)
        m = _pad_audio_mask(item["mask"], max_time)
        mm = MultimodalDataItem(modality=Modality.AUDIO, feature=feat)
        mm.feature_attention_mask = m
        audios.append(mm)
    return BatchPlan(self, [], [], audios, spans)


def run_feature(self, model, plan):
    if plan.is_empty:
        return {"image": None, "video": None, "audio": None}
    embed = model.thinker.get_audio_feature(plan.audio_items)
    return {"image": None, "video": None, "audio": embed}


def slice_results(self, raw, plan, messages):
    out: list[StagePayload] = []
    row = tok = 0
    for span, msg in zip(plan.spans, messages):
        state = PipelineState.from_dict(msg.data.data)
        if span.skip_result is not None:
            apply_encoder_result(
                state, stage_name="audio_encoder", result=span.skip_result,
            )
            out.append(_payload_with_state(msg.data, state))
            continue

        token_end = tok + int(span.audio_output_lengths.sum().item())
        result = {
            "audio_embeds": raw["audio"][tok:token_end],
            # Unpadded lengths preserved at build_batch time — we never want
            # to feed batch-wide padded lengths back to merge_for_thinker.
            "audio_feature_lengths": span.audio_feature_lengths,
            "audio_output_lengths": span.audio_output_lengths,
        }
        apply_encoder_result(state, stage_name="audio_encoder", result=result)
        out.append(_payload_with_state(msg.data, state))
        row += span.audio_rows
        tok = token_end
    return out
```

`_get_feat_extract_output_lengths` is imported directly from
`sglang/python/sglang/srt/models/qwen3_omni_moe.py`, not re-derived.

### Visual deepstack split

The upstream visual return tensor has last-dim
`vision_config.out_hidden_size * (1 + len(deepstack_visual_indexes))`
(confirmed in `disaggregation/encode_server.py:_infer_embedding_dims`):

```python
out_hs = vision_config.out_hidden_size
parts = embedding.split(out_hs, dim=-1)            # [base, ds0, ds1, ...]
base, deepstack = parts[0], list(parts[1:])

result = {
    "image_embeds": base,
    "image_grid_thw": image_grid_thw,
    "image_token_counts": image_grid_thw.prod(-1) // (spatial_merge_size ** 2),
    "deepstack_visual_embeds_image": deepstack,
}
```

This keeps `merge_for_thinker()` and the thinker prefill path stable while
the encoder implementation changes underneath.

## Pipeline Config

MVP requires no schema change. Encoder TP is expressed entirely in
`StageConfig`:

```python
StageConfig(
    name="image_encoder",
    factory="sglang_omni_v1.models.qwen3_omni.stages.create_image_encoder_executor",
    factory_args={
        "backend": "sglang",
        "max_batch_size": 32,
        "max_batch_wait_ms": 50,
        "weight_memory_fraction": 0.30,
        "activation_budget_bytes": 20 * 1024**3,
    },
    gpu=[0, 1],
    tp_size=2,
    next="mm_aggregate",
    project_payload={
        "mm_aggregate": (
            "sglang_omni_v1.models.qwen3_omni.request_builders."
            "project_encoder_to_mm_aggregate"
        )
    },
)
```

### Factory contract

```python
def create_image_encoder_executor(
    model_path: str,
    *,
    # IMPORTANT: this signature default is `"local"` and stays "local"
    # across all phases. The launcher decides single-vs-multi-process
    # before the child is spawned by reading
    # `_resolve_factory_args(stage_cfg, config).get("backend", "local")`,
    # which only sees `factory_args` + `runtime_overrides` — it does NOT
    # introspect this signature default (`compiler.py:143-158`). If we
    # ever changed the default to `"auto"`, a StageConfig that omits
    # `factory_args["backend"]` would silently disagree: launcher reads
    # "local" → goes single-process, factory body picks up "auto" →
    # tries to start an SGLang worker. To switch a deployment to the
    # SGLang backend, write `factory_args["backend"]="auto"` (or
    # `"sglang"`) into the StageConfig (or `runtime_overrides`)
    # explicitly — never rely on this default to flip.
    backend: Literal["local", "sglang", "auto"] = "local",
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_batch_size: int = 32,
    max_batch_wait_ms: int = 50,
    weight_memory_fraction: float | None = None,
    activation_budget_bytes: int | None = QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES,
    server_args_overrides: dict[str, Any] | None = None,
    device: str = "cuda",
    dtype: str | None = None,
    load_format: str | None = None,
):
    chosen = _resolve_backend(backend, model_path, stage="image_encoder")
    if chosen == "sglang":
        worker = SGLangEncoderWorker(
            model_path=model_path,
            gpu_id=gpu_id, tp_rank=tp_rank, tp_size=tp_size, nccl_port=nccl_port,
            dtype=dtype,
            load_format=load_format,
            server_args_overrides=server_args_overrides,
        )
        adapter = Qwen3OmniImageEncoderAdapter(...)
        return EncoderScheduler(
            worker=worker,
            adapter=adapter,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            # Inherit the same cost model as the local path so admission
            # control is identical across backends. The cost fn must be
            # cheap (no GPU work) since it runs on every inbox poll.
            request_cost_fn=adapter.request_cost_fn,
            max_batch_cost=activation_budget_bytes,
        )
    # Fallback: existing local-HF SimpleScheduler path, unchanged
    return _build_local_image_encoder(model_path, device=device, dtype=dtype, ...)
```

The audio encoder factory follows the same shape with
`Qwen3OmniAudioEncoderAdapter`, its own `request_cost_fn`, and
`activation_budget_bytes` defaulting to the audio-side budget (no
existing audio cost cap in v1, so the SGLang path can land with
`max_batch_cost=None` — same as v1's local audio encoder today — and a
follow-up adds it once a representative cost model is profiled).

### Adapter `request_cost_fn`

Each adapter exposes `request_cost_fn(payload: StagePayload) -> int` so
the EncoderScheduler can size batches without depending on adapter
internals. The image cost model from `models/qwen3_omni/stages.py:144-166`
is the right *shape* — same arithmetic, same calibration constants
(`QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER`,
`QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES`) — but the inputs the SGLang
adapter feeds it are different from what the local path feeds it, so it
cannot be reused unchanged.

**The deepstack double-count trap.** v1 local
`Qwen3OmniImageEncoder.__init__` (`components/image_encoder.py:131-133`)
sets `self.out_hidden_size = vision_cfg.out_hidden_size` — the *base*
hidden size — and `self.deepstack_layers = len(vision_cfg.deepstack_visual_indexes)`.
The v1 cost fn then multiplies by `output_layers = 1 + deepstack_layers`,
which is correct for the local model.

Upstream SGLang `Qwen3VLMoeVisionModel.__init__`
(`sglang/python/sglang/srt/models/qwen3_vl.py:334-336`) instead writes
`self.out_hidden_size = vision_config.out_hidden_size * (1 + len(deepstack_visual_indexes))`
— deepstack is **already folded** into the wrapper's
`out_hidden_size`. Reading `model.thinker.visual.out_hidden_size` and
also multiplying by `(1 + deepstack_layers)` would count deepstack
twice and produce a budget cap roughly `(1 + deepstack)^2 / (1 + deepstack)`
times too tight, starving long-video batches.

**Resolution.** The SGLang adapter takes its cost metadata directly
from the HF `vision_config` (which is the same source v1's local model
already uses), not from the SGLang model wrapper:

```python
class Qwen3OmniImageEncoderAdapter:
    def __init__(self, *, hf_config, dtype: torch.dtype):
        vision_cfg = hf_config.thinker_config.vision_config
        self._merge = int(vision_cfg.spatial_merge_size) ** 2
        self._base_hidden = int(vision_cfg.out_hidden_size)        # NOT the wrapper's
        self._output_layers = 1 + len(vision_cfg.deepstack_visual_indexes)
        self._dtype_bytes = torch.empty((), dtype=dtype).element_size()

    def request_cost_fn(self, payload):
        state = PipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name="image_encoder")
        if request.skip_result is not None:
            return 0
        inputs = request.model_inputs
        raw_bytes = _tensor_bytes(inputs.get("pixel_values"))
        raw_bytes += _tensor_bytes(inputs.get("pixel_values_videos"))
        visual_tokens = _grid_visual_tokens(inputs.get("image_grid_thw"), self._merge)
        visual_tokens += _grid_visual_tokens(inputs.get("video_grid_thw"), self._merge)
        output_bytes = (
            visual_tokens * self._base_hidden
            * self._dtype_bytes * self._output_layers
        )
        return (raw_bytes + output_bytes) * QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER
```

The arithmetic and scaling constants are identical to the local helper
(`stages.py:144-166`); only the data source moved. `_tensor_bytes` and
`_grid_visual_tokens` are extracted from `stages.py` into
`encoder_adapters.py` so neither path imports from the other.

The audio adapter's `request_cost_fn` returns 0 in Phase 0 (no v1 audio
cost model yet), and `max_batch_cost` defaults to None — same admission
shape as the local audio path today.

`backend="auto"` resolves to `"sglang"` if the upstream model registers a
matching encoder adapter and `tp_size == 1` baseline parity has passed; else
`"local"`. `tp_size > 1` is only accepted for `backend="sglang"` in the
MVP. `backend="local"` with `tp_size > 1` raises a config error.

## TP Launch Lifecycle

For `image_encoder` with `tp_size=2`, end to end:

1. `MultiProcessPipelineRunner._build_stage_groups` reads `tp_size=2` and
   `gpu=[0, 1]` from the config.
2. `_NcclPortAllocator.allocate()` picks an unused TCP port.
3. `_build_tp_stage_specs` mints two `StageProcessSpec` entries: one
   `role="leader"` for `tp_rank=0/gpu_id=0`, one `role="follower"` for
   `tp_rank=1/gpu_id=1`. `factory_args` is augmented with
   `tp_rank/tp_size/nccl_port/gpu_id` and the leader gets
   `follower_work_queues` / `follower_abort_queues`.
   (`pipeline/mp_runner.py:164-225`)
4. `StageGroup.spawn` launches both subprocesses via the spawn context.
5. Before torch import, `_prepare_cuda_environment` sets
   `CUDA_VISIBLE_DEVICES` to the mapped device for that rank and rewrites
   `factory_args["gpu_id"]=0`. (`pipeline/stage_process.py:252-276`)
6. The factory builds `EncoderScheduler(worker=SGLangEncoderWorker(...))`.
7. `SGLangEncoderWorker.__init__` always calls
   `init_distributed_environment(world_size=2, rank=tp_rank,
   distributed_init_method=f"tcp://127.0.0.1:{nccl_port}")`,
   `initialize_model_parallel(tensor_model_parallel_size=2)`, then
   `get_model(...)`. Two processes meet on the NCCL port and form the TP
   group. **At `tp_size=1` the same calls run with `world_size=1, rank=0,
   distributed_init_method=tcp://127.0.0.1:<free_port>`.**
8. `Stage.run()` starts. Leader binds the ZMQ recv endpoint; follower binds
   `TPFollowerControlPlane`. Both spawn a scheduler thread.
9. `EncoderScheduler` enters its loop. Leader drains the inbox and uses
   the two-channel broadcast to ship metadata + tensors to the follower;
   both run forward; leader emits results to the outbox.
10. `Stage._drain_outbox_external` (leader) routes the outbox `result`
    messages to `mm_aggregate` via the relay.
11. `Stage._drain_outbox_follower` (follower) discards outputs.
12. Aborts and shutdown go from leader to follower via the existing
    `TPLeaderFanout` queues.

## Control Plane and Data Plane

External pipeline traffic does not change:

- ZMQ carries `SubmitMessage`, `DataReadyMessage`, `AbortMessage`,
  `ProfilerStart/Stop`, `ShutdownMessage`. Only the leader binds the recv
  socket.
- Relay (SHM / NCCL / NIXL / Mooncake) carries `StagePayload` tensors
  between logical stages. Only the leader writes / reads relay blobs.

Internal TP traffic is owned by SGLang:

- `EncoderScheduler` metadata broadcast goes over the SGLang TP CPU group
  via `broadcast_pyobj`.
- `EncoderScheduler` tensor broadcast goes over the SGLang TP device group
  via `dist.broadcast(tensor, src, group=tp_group.device_group)`.
- Encoder forward collectives go over SGLang's GPU TP group via
  `ColumnParallelLinear` (output-dim shard) and `RowParallelLinear`
  (input-dim shard, all-reduce). For Qwen3-Omni audio this is layer-wise
  in `Qwen3OmniMoeAudioEncoderLayer`. For Qwen3-VL/Qwen3-Omni vision this
  is in `Qwen3_VisionMLP` / `VisionAttention(use_qkv_parallel=True)`.

## Memory Accounting

Encoder stages should not reuse AR `mem_fraction_static` semantics
directly. For AR runners that controls KV cache allocation after weights
load; for an encoder-only stage there is no KV pool, so the knob has no
clean meaning.

MVP behaviour (mirrors v1's existing image-encoder cost model):

```python
factory_args={
    "backend": "sglang",
    "weight_memory_fraction": 0.30,        # admission control vs total VRAM
    "activation_budget_bytes": 20 * 1024**3, # caps batch formation
    "max_batch_size": 32,
    "max_batch_wait_ms": 50,
}
```

- `activation_budget_bytes` is plumbed into `EncoderScheduler` as
  `max_batch_cost`, with the same `request_cost_fn` v1 already uses
  (`models/qwen3_omni/stages.py:144-166`). We keep that cost model — it
  was tuned to keep activation peaks below OOM on H200.
- `weight_memory_fraction` is informational at the StageConfig level and
  consumed by future co-location admission control. It is **not** passed
  to SGLang as `mem_fraction_static`. (We pin SGLang's encoder-only path
  to weights-only allocation.)
- If both budgets are absent, EncoderScheduler falls back to v1's current
  defaults (`max_batch_size=32`, `max_batch_wait_ms=50`).

Longer term:

- Replace `weight_memory_fraction` + `activation_budget_bytes` with a typed
  `StageMemoryConfig`. Make co-location first-class by summing declared
  budgets per GPU.
- Decide whether the canonical fraction semantics is "fraction of total
  VRAM" (vLLM-style) or "fraction of remaining VRAM" (current SGLang AR).
  Today's v1 mixes them implicitly. Pin the choice **before** TP encoders
  start sharing GPUs with the thinker.

## Fallback For Encoders Not Upstreamed

Plan B still needs a local path for encoders that do not exist in SGLang
main (custom Fish audio tokenizer, future Boson audio encoder).

Rules:

1. If an upstream SGLang encoder exists, prefer `backend="sglang"`.
2. If an encoder is model-specific and not upstreamed, use
   `backend="local"` and the current `SimpleScheduler` path. No changes
   there.
3. `backend="local"` is single-rank-only in the MVP. `tp_size > 1` plus
   `backend="local"` is rejected at config validation.
4. If local TP becomes unavoidable for a non-upstreamed encoder, upstream
   the encoder first unless a production blocker forces the inverse.

This is what the issue asks for: support both modes side by side without
the pipeline config knowing which mode a stage is in.

## Error Handling

`EncoderScheduler` owns lifecycle errors. Adapters and SGLang encoder
forwards must not catch broad `Exception`.

Rules:

- `SGLangEncoderWorker.encode_batch` and `EncoderAdapter.run_feature` only
  catch specific expected exceptions (e.g. `OutOfMemoryError` for batch
  splitting, if we ever add it). They never catch base `Exception`.
- `EncoderScheduler.start()` covers **four** steps in one
  per-iteration error boundary: `_recv_messages`, `build_batch`,
  `encode_batch`, `slice_results`. `_recv_messages` does not raise —
  it returns `(messages, error)`. The other three are wrapped in a
  try/except that funnels into the same `local_err` slot. Each step
  is deterministic given the broadcast-equal `messages`, but any one
  failing without rank-sync would leave peers stuck at the next
  collective.
- Cross-rank synchronization: after the four steps,
  `dist.all_gather_object` exchanges a per-rank "did I fail?" boolean
  on the TP CPU group. If any rank failed, every rank skips
  emit-results and starts the next iteration on a fresh collective.
  The entry rank emits one
  `OutgoingMessage(type="error", data=str(exc))` per request that
  was drained this iteration (`messages` is non-empty on recv-time
  failure thanks to the tuple return), which
  `Stage._drain_outbox_external` converts into a Coordinator failure
  → HTTP 500. If a follower failed but the entry rank did not, the
  entry rank emits `RuntimeError("peer-rank encoder forward failed")`
  against the same drained messages.
- **Pre-broadcast entry-rank failures**: `_recv_messages` does H2D
  copies inside `_strip_and_lift` *before* the metadata broadcast. A
  failure there (OOM, dtype coercion, malformed payload) on the
  entry rank would leave followers blocked on `broadcast_pyobj`
  forever — the runner cannot detect this since
  `MultiProcessPipelineRunner._monitor_children`
  (`pipeline/mp_runner.py:332-342`) only fires on
  `StageGroup.any_dead()`, which checks `not is_alive() and exitcode
  != 0` (`pipeline/stage_group.py:130-132`), and a caught exception
  keeps the process alive. The fix is internal to `_recv_messages`:
  the entry rank wraps `_strip_and_lift` in try/except and
  broadcasts a tagged dict
  `{"kind": "encoder_recv_error", "error": repr(exc)}` over the
  same CPU-group `broadcast_pyobj` slot the success path uses, then
  returns `(local, exc)`. Followers detect the dict by kind-string
  equality (not identity — pickle round-trip would break that) and
  return `([], RuntimeError(...))`. Both ranks then converge on the
  scheduler's per-iteration `all_gather_object` handshake without
  ever crashing the scheduler thread. No new control channel, no
  runner-level rescue, no `Stage._handle_scheduler_crash` (the
  stage-level abort path) involvement.
- No adapter may return a fake-success embedding (zero tensor, empty list,
  etc.). v1's broader refactor explicitly disallows that pattern.

This aligns encoder stages with the scheduler-layer error handling
direction in #188.

## Supported Pipelines

### Qwen3-Omni speech (8-stage)

```text
preprocessing -> [image_encoder, audio_encoder] -> mm_aggregate -> thinker
              -> [decode, talker_ar] -> code2wav
```

- `image_encoder` stage: `backend="sglang", tp_size=N`, `gpu=[g0..gN-1]`.
- `audio_encoder` stage: same shape, separate stage, separate process
  group, separate NCCL port. Image and audio encoders run in parallel
  because the thinker only depends on `mm_aggregate`'s fan-in.
  `mm_aggregate.wait_for = ["preprocessing", "image_encoder",
  "audio_encoder"]` already enforces the join.

### Qwen3-Omni text (6-stage)

```text
preprocessing -> thinker -> decode
```

No encoder stage runs. No change.

### Fish Audio S2-Pro (3-stage TTS)

```text
preprocessing -> tts_engine -> vocoder
```

No multimodal encoder upstream of the AR engine. Plan B does not apply.
Fish stays on `SimpleScheduler` for preprocessing and `OmniScheduler` for
`tts_engine`.

### Future MiMo / Ming-Omni

Same pattern as Qwen3-Omni. As long as the upstream SGLang model registers
encoder methods (`get_image_feature`, etc.) and ships a small
`encoder_adapters.py` in `sglang_omni_v1/models/<name>/`, the pipeline
config just sets `backend="sglang"` and `tp_size`.

## Adding A New Model

1. Add `models/<name>/encoder_adapters.py`. Implement `build_batch`,
   `run_feature`, `slice_results` for each encoder stage.
2. In `models/<name>/stages.py`, branch the encoder factory on `backend`:
   `"sglang"` builds `EncoderScheduler(SGLangEncoderWorker(...), adapter)`;
   `"local"` keeps the existing `SimpleScheduler` path.
3. In `models/<name>/config.py` (the `PipelineConfig`), set `tp_size` and
   `gpu=[...]` on the encoder stage.
4. Validate parity: `backend="local"` vs `backend="sglang", tp_size=1` on
   the same input. Then bump `tp_size`.

Everything else (`Stage`, `StageGroup`, `Coordinator`, `relay`,
`EncoderScheduler` itself) is reused without modification.
`MultiProcessPipelineRunner` and `pipeline/stage_process.py` get the
small launcher extension described in [Required launcher
change](#required-launcher-change), which is a one-time addition that
all sglang-backed encoder stages share.

## Implementation Plan

Phase 0 — landing path that doesn't break v1:

1. **Launcher extension** (`sglang_omni_v1/pipeline/` + `sglang_omni_v1/serve/` + `sglang_omni_v1/config/`).
   - Add `single_visible_device: bool = False` to `StageProcessSpec`
     (`pipeline/stage_process.py`).
   - In `MultiProcessPipelineRunner._build_stage_groups`
     (`pipeline/mp_runner.py`), set the flag pre-spawn from the resolved
     `base_factory_args.get("backend") in {"sglang", "auto"}` after
     `_resolve_factory_args` has merged `runtime_overrides`.
   - In `get_stage_process_env` (`pipeline/stage_process.py:222`),
     change the early return to
     `if spec.tp_size <= 1 and not spec.single_visible_device: return {}`.
   - In `serve/launcher.py` (`launcher.py:141-150`), extend the
     `needs_mp` predicate so any stage with resolved `backend in
     {"sglang", "auto"}` forces `MultiProcessPipelineRunner`, even on a
     single-GPU / `tp_size=1` pipeline. Without this the launcher
     short-circuits to `compile_pipeline()` and the worker never gets
     the per-process CUDA remap.
   - In `config/compiler.py:compile_pipeline`, reject any stage with
     resolved `backend in {"sglang", "auto"}` **or**
     `stage_cfg.tp_size > 1`. The `tp_size > 1` reject is structural:
     `compile_pipeline` never injects `tp_rank/tp_size/nccl_port`,
     so any direct caller bypassing `serve/launcher.py` would silently
     get a `tp_size=1` factory (thinker / talker / encoder all fail
     this way). It does not regress normal serving because
     `serve/launcher.py:149` routes any `tp_size > 1` to
     `MultiProcessPipelineRunner` first.
   - In `MultiProcessPipelineRunner._build_stage_groups`, run the
     two-layer TP preflight from rule 6 of
     [Required launcher change](#required-launcher-change) — Layer 1
     rejects any TP stage whose factory does not accept
     `tp_rank/tp_size/nccl_port`; Layer 2 rejects encoder TP stages
     whose resolved backend is not `"sglang"`.

   These six sub-steps are the load-bearing prerequisite for Phase 1's
   `tp_size=1, gpu!=0` parity lane to validate the right thing — see
   [GPU placement across `tp_size=1` and `tp_size>1` lanes](#gpu-placement-across-tp_size1-and-tp_size1-lanes)
   and [Required launcher change](#required-launcher-change).

   **Per-sub-step unit tests** (so each launcher change can be verified
   independently of the GPU lane in Phase 1):

   - `single_visible_device` flag: build a `PipelineConfig` with a
     stage whose `factory_args["backend"] = "sglang"` and assert
     `_build_stage_groups(...)` produces a `StageProcessSpec` with
     `single_visible_device=True`; flip backend through
     `runtime_overrides` and assert the flag still flips
     (covers the `_resolve_factory_args` source rule).
   - `get_stage_process_env` early return: spec with `tp_size=1,
     single_visible_device=False` returns `{}`; same spec with the
     flag flipped returns the remap env dict.
   - `needs_mp` predicate: a single-stage / single-GPU /
     `tp_size=1` config with `backend="sglang"` makes
     `await launch_pipeline(...)` take the
     `MultiProcessPipelineRunner` branch (assert by mocking the runner
     constructor). The same config with `backend="local"` keeps the
     `compile_pipeline()` branch.
   - `compile_pipeline` reject: directly calling
     `compile_pipeline(config_with_sglang_stage)` raises `ValueError`.
   - **Signature-default trap regression test**: a StageConfig whose
     `factory_args` does **not** contain a `backend` key, pointing at a
     factory whose Python signature default is `"auto"`, must still
     resolve to `"local"` at the launcher (i.e. `needs_mp` stays
     False, `single_visible_device` stays False, `compile_pipeline`
     does not raise). This locks the
     [backend resolution contract](#backend-resolution-contract) and
     prevents a future signature-default flip from silently bypassing
     the CUDA isolation.
   - **Real-factory signature lock**: `import inspect` and assert that
     `inspect.signature(create_image_encoder_executor).parameters["backend"].default == "local"`
     and the same for `create_audio_encoder_executor`. The previous
     test only proves the resolver ignores signature defaults; this
     one proves the actual production factories never have their
     default flipped to `"auto"` or `"sglang"` by accident. Cheap to
     run (no GPU, no model load, just `inspect`) and catches a
     code-review miss directly.
   - **TP preflight Layer 2 (encoder factory backend gate)**: build
     a config where the `image_encoder` stage has `tp_size=2,
     gpu=[0, 1], factory_args={}` (no backend), call
     `_build_stage_groups` and assert it raises `ValueError`
     mentioning the stage name. Then with
     `factory_args={"backend": "auto"}` — also raises (auto can fall
     back to local). Then with `factory_args={"backend": "sglang"}`
     — succeeds. Locks Layer 2 of rule 6 in
     [Required launcher change](#required-launcher-change).
   - **TP preflight does NOT regress thinker TP**: build a config
     where the `thinker` stage has `tp_size=2, gpu=[0, 1]` and no
     `backend` in `factory_args` (its factory does not accept one,
     but does accept `tp_rank/tp_size/nccl_port`).
     `_build_stage_groups` must succeed. Locks Layer 1 of rule 6 —
     proves the encoder reject is not over-broad.
   - **TP preflight Layer 1 (factory not TP-capable)**: build a
     config where some stage uses a SimpleScheduler factory like
     `create_aggregate_executor` (no `tp_rank/tp_size/nccl_port` in
     signature) and `tp_size=2`. `_build_stage_groups` must raise
     `ValueError` listing the missing parameters. This is the case
     the previous single-layer preflight silently passed through to
     subprocess spawn.
   - **`compile_pipeline` rejects `tp_size > 1` directly**: call
     `compile_pipeline(config_with_thinker_tp_size_2)` (no backend
     parameter, regular thinker factory) and assert it raises
     `ValueError`. This is the direct-call regression test — proves
     a caller bypassing `serve/launcher.py` cannot silently downgrade
     a TP config to single-rank.
   - **Allocation-ready gather (mid-recv deadlock guard)**: run
     `EncoderScheduler` with `tp_size=2` and a fake `torch.empty` on
     the follower that raises on the **second** spec (so the first
     allocation succeeds, mimicking partial OOM). Assert (a) entry
     rank does **not** issue any `dist.broadcast(t, group=device_group)`
     call (use a mock that records calls); (b) follower returns
     `(messages, error)` from `_recv_messages` instead of raising;
     (c) both ranks reach the unified `all_gather_object` handshake
     with `local_err is not None` on the follower side; (d) the
     entry rank's drained `local` is forwarded via the tuple return,
     so `_emit_error(messages, ...)` produces one error message per
     drained request. Locks the `_allocation_ready_gather` contract.
   - **Pre-broadcast error sentinel (`_recv_messages` deadlock guard)**:
     run `EncoderScheduler` with `tp_size=2` and a fake `_strip_and_lift`
     that raises on the entry rank before the metadata
     `broadcast_pyobj`. Assert (a) the follower rank does **not**
     block on `broadcast_pyobj` indefinitely (bounded-time wait with
     fail-on-timeout); (b) `_recv_messages` returns `(messages, exc)`
     on **both** ranks rather than raising — entry rank's `messages`
     equals the drained list, follower's `messages` is `[]`,
     follower's exception text quotes the entry-rank exception via
     `repr(exc)`; (c) the scheduler's `all_gather_object` handshake
     observes both ranks as failed and emits exactly one
     `OutgoingMessage(type="error")` per drained request on the
     entry rank (i.e. `len(error_emissions) == len(drained_messages)`,
     not 0 and not 2× from double-counting); (d) the scheduler thread
     stays alive and proceeds to the next loop iteration —
     `Stage._handle_scheduler_crash` must not fire. Locks the
     tagged-dict sentinel contract and the recv-error tuple return.
   - **AR-only knob protection — helper level**: directly call
     `build_sglang_encoder_server_args(model_path=..., tp_size=1,
     base_gpu_id=0, dist_init_addr="...", mem_fraction_static=0.5)`
     and assert it raises `ValueError` referencing
     `mem_fraction_static`. Repeat for `max_running_requests`,
     `chunked_prefill_size`, `context_length`, `max_prefill_tokens`.
     This is the helper signature `(..., **overrides)` so the AR knob
     goes in as a direct keyword.
   - **AR-only knob protection — factory level**: call
     `create_image_encoder_executor(...,
     server_args_overrides={"mem_fraction_static": 0.5})` and assert
     the `ValueError` propagates from helper through worker
     `**overrides` splat. This locks the dict-style entry point users
     actually configure through `StageConfig.factory_args`.
2. Add `sglang_omni_v1/model_runner/sglang_encoder_worker.py`. Behind
   `backend="sglang"` only — never the default in this PR.
3. Add `sglang_omni_v1/scheduling/encoder_scheduler.py` including the
   two-channel `_recv_messages` and the `BatchPlan` plumbing.
4. Add `sglang_omni_v1/models/qwen3_omni/encoder_adapters.py` (image +
   audio).
5. Update `create_image_encoder_executor()` and
   `create_audio_encoder_executor()` to accept `backend`, `tp_rank`,
   `tp_size`, `nccl_port`, `load_format`, `server_args_overrides`. Default
   `backend="local"` keeps current behaviour.
6. Add a Qwen3-Omni config variant (`qwen3_omni_encoder_tp.py` or a CLI
   override) that sets `backend="sglang", tp_size=2`.

Phase 1 — parity validation (gates Phase 2):

7. GPU parity test: `backend="local"` vs `backend="sglang", tp_size=1` on
   image encoder and audio encoder, in isolation, at a non-zero
   `gpu` (e.g. `gpu=4`). Asserts the launcher remap took effect:
   child env has `CUDA_VISIBLE_DEVICES=="4"`,
   `next(model.parameters()).device.index == 0` (the only visible CUDA
   device shows up as `cuda:0` inside the child), and
   `get_world_group().local_rank == 0`.
8. TP parity test: `backend="sglang", tp_size=1` vs `tp_size=2`, within
   `atol=1e-3, rtol=1e-3` on float16. Asserts
   `get_world_group().local_rank` is unique per rank in the `tp_size=2`
   case.
9. E2E speech run: long-video request that previously OOM'd on the
   thinker GPU, run with image+audio encoders sharded to separate GPUs,
   verify it completes.

Phase 2 — switch new deployments to the SGLang backend:

10. After Phase 1 passes, do **two** changes together:
    - Change `_resolve_backend("auto", ...)` to return `"sglang"` when
      the adapter exists.
    - Update the default Qwen3-Omni `PipelineConfig` template (and any
      cookbook configs / CLI helpers that build a config) so the
      `image_encoder` and `audio_encoder` stages have
      `factory_args={"backend": "auto", ...}` written **explicitly**.

    Do **not** change the factory function's signature default — it
    stays `"local"` forever. The launcher reads
    `_resolve_factory_args(...).get("backend", "local")` and only sees
    `factory_args` + `runtime_overrides`, never the signature default
    (see [Backend resolution contract](#backend-resolution-contract)).
    Bumping the signature default would create the "launcher sees
    local, factory sees auto" mismatch that silently bypasses the
    single-visible-device remap.

    The launcher already sets `single_visible_device=True` whenever
    resolved `backend in {"sglang", "auto"}` (Phase 0 step 1), so once
    the templates explicitly carry `backend="auto"` the rest works.

    **Phase 2 unit test:** load the default Qwen3-Omni
    `PipelineConfig` (the one returned by the canonical builder /
    cookbook helper) and assert that the resolved
    `_resolve_factory_args(image_encoder_stage, config).get("backend")`
    is exactly `"auto"` — i.e. the value lives in `factory_args`, not
    in the factory signature. This is the only test that proves Phase
    2's flip actually crosses the launcher boundary.

Phase 3 — clean up duplicated v1 encoder code:

11. After at least one release with Phase 2 default, delete
    `sglang_omni_v1/models/qwen3_omni/components/{image_encoder.py,audio_encoder.py}`
    and the corresponding HF tower instantiation in `stages.py`.

Phase 4 — runtime parameter plumbing (out of scope for the first PR):

12. Replace ad-hoc `weight_memory_fraction` / `activation_budget_bytes`
    factory args with a typed `StageMemoryConfig`. This is the same
    "typed stage-addressable override primitive" Cheng called out in the
    v1 refactor architecture doc.

## Validation

Minimum lanes (unit + GPU + E2E):

- **Unit**
  - Adapter `build_batch` produces correct `BatchPlan` (flat items + spans)
    for: image-only, audio-only, image+video same request, multi-request
    mixed batch.
  - `slice_results` round-trips a synthetic raw embedding back into the
    expected per-request `encoder_outs` dict shape.
  - `EncoderScheduler._recv_messages` mocks the TP groups and confirms
    that on the entry rank the metadata pickle does not contain tensor
    payload bytes, while followers reconstruct identical
    `IncomingMessage` objects after `dist.broadcast`.

- **GPU**
  - Qwen3-Omni image encoder: `backend="local"` vs `backend="sglang",
    tp_size=1`, fixed seed, equal output tensors within tolerance.
    (This is also the lane that exercises single-rank distributed init.)
  - Qwen3-Omni audio encoder: same.
  - Qwen3-Omni image+audio encoder: `tp_size=1` vs `tp_size=2`, equal
    output within tolerance.

- **E2E**
  - `examples/qwen3_omni_speech.py` long video that previously OOM'd:
    image encoder `tp_size=2` on GPU [0,1], audio encoder `tp_size=2` on
    GPU [2,3], thinker on GPU 4, talker on GPU 5, code2wav on GPU 6.
    End-to-end waveform produced, no OOM.

- **Fault injection**
  - Force encoder OOM (oversized pixel batch) and verify the request fails
    through `Coordinator` with HTTP 500 and a non-empty error body.
    Specifically reject:
    - `data=None` "success" coming back from `code2wav` (Ming-Omni shape).
    - zero-tensor "success" coming back from talker (current Qwen3-Omni
      shape in the bug discussed in #302).

- **Reuse existing CI**
  - The existing thinker / talker CI (test_qwen3_omni_v1_*.py) must pass
    unchanged in `backend="local"` mode in Phase 0 and `backend="sglang"`
    mode in Phase 2.

## Open Questions

1. **Upstream `EncoderModelWorker` helper.** Should we push the eight-call
   init sequence (`set_global_server_args_for_scheduler` → `ModelConfig` →
   `LoadConfig` → `init_distributed_environment` →
   `initialize_model_parallel` → `initialize_dp_attention` → `get_model`
   → expose `tp_group`) into SGLang main as a public helper class? That
   would delete `SGLangEncoderWorker.__init__` and remove the only place
   v1 reads into upstream `MMEncoder` internals. Recommended: file an
   upstream issue after Phase 1 lands, propose
   `sglang.srt.disaggregation.EncoderModelWorker`, driven by SGLang-Omni
   as the first consumer.

2. **DP encoder vs TP encoder.** Upstream `mm_enable_dp_encoder` works for
   Qwen2.5-VL (#13126 merged) and Qwen3-VL (#13724 merged) but Qwen3-Omni
   support is still open (#14886). PR #18721 also shows that DP encoder
   requires careful interaction with the LLM-side TP via
   `VocabParallelEmbedding`. **MVP intentionally only supports encoder
   TP.** Encoder DP becomes a follow-up after #14886 lands upstream and
   we can import the audio side. The factory rejects
   `mm_enable_dp_encoder=True` in Phase 0–2 to avoid coupling.

3. **`StageConfig.tp_size` vs future `ParallelismConfig`.** Today TP is
   the only parallelism axis we expose. Encoder DP and possibly EP would
   push us toward a single `parallelism: ParallelismConfig(tp=N, dp=M,
   ep=K)` field. Don't pre-emptively rename. Add `ParallelismConfig` only
   when the second axis (DP) actually has a real consumer in v1.

4. **Local fallback retention.** `models/qwen3_omni/components/image_encoder.py`
   and `audio_encoder.py` should be deleted in Phase 3. Some reviewers
   will want to keep them as a debug fallback for one release. Default
   plan: delete after one release with `backend="auto" -> sglang"` as the
   default.

5. **Fault-injection enforcement.** The bug Cheng called out in v1
   refactor notes (silent fake-success embeddings on OOM) can be
   detection-only with fault injection. Should we additionally land a
   lint rule that bans `except Exception` inside
   `models/<name>/components/` and `models/<name>/encoder_adapters.py`?
   That makes Rule 2 of the Scheduler-layer error handling enforceable.
   Suggested: yes, as a follow-up PR after Phase 0.

6. **TP-aware encoder cache.** `EncoderRequestData.cache_key` is honored
   by the local path through `SimpleCacheManager`, but the SGLang path in
   Phase 0 ignores cache and always forwards. Wiring cache into a
   TP-broadcasting scheduler requires the entry rank to decide hit/miss
   *before* the broadcast and ship that decision so all ranks agree
   (otherwise the broadcast shape is wrong on a rank that thinks it hit
   while another missed). Suggested: design a small `CacheBroadcastPlan`
   in Phase 2, after `tp_size>1` parity is locked.

7. **Multinode encoder TP.** Phase 0 is single-node. `nnodes` and
   `node_rank` sit in `_ENCODER_PROTECTED_KEYS` so the SGLang worker
   can never accidentally enter a multinode init path; the launcher
   only allocates one NCCL port per stage and assumes
   `127.0.0.1`-loopback dist-init addresses. Multinode encoder TP
   (sharding one encoder across machines) is a separate workstream
   that needs cross-machine NCCL bootstrap, multinode-aware
   `StageProcessSpec`, and probably a new `EncoderClusterConfig` —
   none of which Plan B intends to design now. If multinode becomes a
   real requirement, lift the lock on `nnodes`/`node_rank` and revisit
   the launcher.

## Progress Tracking

- [ ] Phase 0 land — `EncoderScheduler` + `SGLangEncoderWorker` +
      Qwen3-Omni adapter, `backend="local"` default.
- [ ] Phase 1 parity — image/audio encoder local-vs-sglang, tp1-vs-tp2.
- [ ] Phase 2 — `backend="auto"` defaults to SGLang for Qwen3-Omni.
- [ ] Phase 3 — delete duplicated local encoder files.
- [ ] Phase 4 — typed `StageMemoryConfig`.

PR #334 (v1 base) must land first; this RFC's Phase 0 PR depends on it.
