# RFC: Multimodal Encoder TP via SGLang Native Encoders (Plan B, Lean Draft)

Issue: https://github.com/sgl-project/sglang-omni/issues/375

Decision: SGLang-Omni should use SGLang main's native multimodal encoder
implementations and inherit SGLang tensor parallelism, instead of maintaining
separate encoder copies under `models/<name>/components/`.

This lean draft is the normative design contract for #375. Detailed Python
sketches, per-key server-argument protection, and assertion-level test plans
belong in Phase 0 PR code comments or non-normative implementation notes.

For the detailed implementation notes behind these contracts, see
[`encoder_tp_path_b_design.md`](encoder_tp_path_b_design.md). If the two
documents disagree, this lean RFC wins; the detailed file must be updated to
match this contract before implementation.

## Motivation

This is solving a long-sequence activation-memory OOM, not an encoder weight
residency problem. For Qwen3-Omni, the combined vision and audio encoder
weights are small relative to the thinker, roughly 2.5 GB, but encoder
activation memory scales with media length. A one-minute video can push encoder
activations past 30 GB on a single GPU. That pressure is what pushes the
colocated thinker toward OOM today and motivates workaround-style reservation
such as `--encoder-mem-reserve` in #339. The concrete pain is the long-video
path discussed around #327 / #339.

Tensor parallelism is the useful tool here because it shards encoder
activations across ranks. Giving the encoder a larger GPU only moves the cliff,
and data parallelism would replicate the same long-sequence activation footprint
on every rank. We still avoid loading full thinker/talker weights inside an
encoder stage, but the primary scaling target is activation memory.

## Goals

- Reuse upstream SGLang encoder modules and TP kernels for multimodal encoders.
- Keep the public v1 pipeline topology unchanged.
- Shard encoder activation memory for long image/video/audio inputs.
- Preserve v1 request lifecycle, relay ownership, abort handling, and
  Coordinator completion semantics.
- Keep the local HF encoder path as a fallback for encoders not yet upstreamed
  into SGLang.

## Non-Goals

- Do not instantiate the full upstream `ForConditionalGeneration` class inside
  encoder stages.
- Do not introduce encoder DP in this RFC. Upstream DP encoder support remains
  model-specific and is a follow-up path.
- Do not redesign the Stage/Coordinator control plane.
- Do not make `SimpleScheduler` TP-aware.
- Do not use `worker` / `executor` vocabulary for new encoder path classes.
  The new SGLang-backed component is a runner.

## Architecture

The external pipeline remains:

```text
preprocessing -> [image_encoder, audio_encoder] -> mm_aggregate -> thinker -> ...
```

Only the encoder-stage implementation changes:

```text
old: Stage -> SimpleScheduler -> local HF encoder copy
new: Stage -> EncoderScheduler -> SGLangEncoderRunner -> upstream encoder module
```

At runtime:

```text
HTTP API -> Client -> Coordinator -> StageGroup -> Stage(leader)
                                                |        \
                                                |         Stage(follower) x (tp_size - 1)
                                                v
                                  EncoderScheduler (one per rank)
                                                v
                                  SGLangEncoderRunner (one per rank)
                                                v
                              upstream SGLang encoder submodule(s)
```

Layer responsibilities:

| Layer | Responsibility |
| --- | --- |
| `Coordinator` | Submit requests, collect completions, broadcast aborts. |
| `StageGroup` / `MultiProcessPipelineRunner` | Spawn one process per TP rank, assign GPU ids, allocate NCCL port, monitor child exits. |
| `Stage` | Own existing ZMQ control plane, relay IO, input aggregation, stream routing, and scheduler queues. |
| `EncoderScheduler` | Drain input on the entry rank, fan out metadata/tensors to non-entry TP ranks, run the runner on all ranks, emit downstream messages only on the entry rank. |
| `SGLangEncoderRunner` | Initialize SGLang distributed state, instantiate declared encoder submodules, partial-load matching checkpoint prefixes, expose `encode_batch()`. |
| `EncoderAdapter` | Convert v1 `PipelineState` to the runner's `BatchPlan`, declare encoder module specs, and slice raw encoder outputs back into v1 payload shape. |

Terminology:

- Stage-level process roles remain the existing `single` / `leader` /
  `follower` names.
- Scheduler-level TP roles use `entry rank` and `non-entry ranks`.
  `entry rank` follows existing `OmniScheduler.is_entry_rank`; non-entry ranks
  are every TP rank other than the entry rank.
- New encoder code uses `Runner`, not `Worker` or `Executor`, to stay aligned
  with #188's vocabulary.

## Why This Shape

The existing multi-process pipeline already has the launcher pieces we need:
`StageGroup` spawns one process per TP rank, assigns rank-local GPU ids, and
sets up leader/follower control queues. The missing part is data fan-out for
encoder request inputs. Stage followers receive shutdown/profiler/abort control
messages, but they do not receive `SubmitMessage` / `DataReadyMessage`, do not
own relay endpoints, and do not call `relay_io.read_payload`.

That makes `EncoderScheduler` the right boundary. It already sits behind
`Stage`, has one instance per TP rank, and can use SGLang TP groups directly.
It mirrors the existing `OmniScheduler` pattern for entry-rank-only inbox
drain plus `broadcast_pyobj`, but adds an explicit tensor channel because
encoder payloads contain large media tensors.

`SimpleScheduler` should stay a simple local callable scheduler. Making it
conditionally TP-aware would couple every non-AR stage to TP groups it does not
need.

## Upstream Reuse Boundary

Plan B reuses upstream SGLang at the encoder-submodule level:

- Reuse: distributed init, TP groups, TP-aware layers, model loader selection,
  quantization/load-format hooks, remote weight loading, and upstream encoder
  submodule implementations.
- Declare locally: which encoder submodules a stage needs, which checkpoint
  prefixes map to those submodules, and how v1 payloads map into SGLang input
  objects.
- Do not reuse: the full upstream `ForConditionalGeneration` class. It is the
  serving entry point for full generation and may allocate language-model and
  talker modules that an encoder stage must not own.

For Qwen3-Omni:

- `image_encoder` declares the upstream vision encoder submodule only.
- `audio_encoder` declares the upstream audio encoder submodule only.
- Both stages reuse SGLang TP layers and checkpoint loading, but only for the
  declared encoder prefixes.

## Weight Loading Scope

Encoder stages must not load the full upstream `ForConditionalGeneration`
model. That class is the full serving entry point and can instantiate language
model and talker weights that belong to other pipeline stages. Loading it in
each encoder process would erase the memory win from moving long media
activations off the thinker GPU.

Phase 0 should implement partial encoder-submodule loading inside
SGLang-Omni. Each adapter declares the encoder module specs it needs, including
checkpoint prefixes and any key rewrites. `SGLangEncoderRunner` builds only
those submodules and loads only matching checkpoint keys. This keeps the
implementation unblocked without waiting for an upstream SGLang loader API.

After Phase 1 parity and fault-handling validation, we should propose an
upstream helper such as `EncoderModelRunner` / partial encoder loader so future
models can reuse the same submodule-loading contract directly from SGLang.

For Qwen3-Omni concretely, this scopes the encoder GPU to ~2 GB (vision) or
~0.5 GB (audio) instead of the full ~57 GB thinker LLM. The detailed RFC has
the per-component sizing table and the partial-load pipeline:
[`encoder_tp_path_b_design.md` → Weight Loading Scope](encoder_tp_path_b_design.md#weight-loading-scope).

## Upstream Compatibility Contract

Phase 0 may use a local `SGLangEncoderRunner` shim, but the shim has a bounded
upstream dependency surface. Omni may depend on:

- SGLang distributed setup: `init_distributed_environment`,
  `initialize_model_parallel`, `initialize_dp_attention`, and `get_tp_group`.
- SGLang config/load setup: `ServerArgs`, `ModelConfig.from_server_args`,
  `LoadConfig`, and `get_model_loader`.
- Qwen3-Omni encoder submodules and helpers:
  `Qwen3OmniMoeVisionEncoder`, `Qwen3OmniMoeAudioEncoder`, and
  `_get_feat_extract_output_lengths`.
- TP-aware layers only through those upstream encoder submodules, not by
  reimplementing or directly reaching into layer internals.

The minimum signature contract is intentionally smaller than the upstream
implementation:

- Vision: construct a vision encoder from the thinker vision config, optional
  quant config / prefix / norm epsilon, and call it with
  `pixel_values, grid_thw`.
- Audio: construct an audio encoder from the thinker audio config, call it with
  `input_features, feature_lens`, and receive an object with
  `last_hidden_state`.
- Audio length: `_get_feat_extract_output_lengths(input_lengths)` returns the
  post-encoder output lengths used to slice `audio_embeds`.

Phase 0 compatibility is pinned to the SGLang commit/version exercised in CI.
Tracking SGLang main is acceptable only if CI imports every allowed symbol,
instantiates Qwen3-Omni image/audio encoder modules at `tp_size=1`, verifies
partial loading only accepts declared prefixes, and runs adapter smoke tests
that validate output shapes. If Phase 0 needs an upstream API outside this
allowlist, it must either wrap it behind the local compatibility shim or first
upstream a helper before depending on it directly.

## TP Input Fan-Out Contract

The entry rank is the only rank whose Stage receives upstream data from ZMQ and
relay. Non-entry ranks need an in-scheduler fan-out path.

Control and data are split:

- Metadata goes over the TP CPU group with `broadcast_pyobj`. Metadata contains
  request ids, non-tensor payload fields, and tensor specs such as path, dtype,
  and shape.
- Tensor data goes over the TP device group with `dist.broadcast` on CUDA
  tensors. The entry rank lifts tensors from relay CPU memory to its local
  device before broadcast. Non-entry ranks allocate matching CUDA placeholders
  and receive into them.

This keeps large media tensors off the pickle path and preserves the existing
v1 separation between small control messages and large tensor payloads.

Before device broadcast, non-entry ranks must pre-allocate all receive tensors and
participate in an allocation-ready handshake on the TP CPU group. If any rank
cannot allocate, all ranks skip the device broadcast and the entry rank emits a
request-level error for the drained requests. This prevents unmatched NCCL
broadcasts when a non-entry rank OOMs during receive-buffer allocation.

## Scheduler Error Contract

Encoder errors split by collective boundary.

Recoverable before model forward:

- `_recv_messages` failures, including entry-rank media tensor lift and
  non-entry-rank receive-buffer allocation.
- `EncoderAdapter.build_batch` failures.

These happen before upstream model collectives start. Ranks synchronize a small
error flag through the TP CPU group. If any rank failed, no rank enters
`encode_batch`; the entry rank emits one request-level error per drained
request, and the scheduler continues.

Fatal inside model forward:

- `SGLangEncoderRunner.encode_batch` enters upstream SGLang TP collectives.
- A rank-local CUDA OOM, NCCL error, or other exception inside that region can
  leave non-entry ranks blocked in device collectives.
- Do not try to recover with a post-hoc CPU gather. The failing rank exits
  non-zero, and `StageGroup` / `MultiProcessPipelineRunner` tears down the
  whole TP group.

Recoverable after model forward:

- `EncoderAdapter.slice_results` runs only on the entry rank after all ranks
  returned from `encode_batch`.
- It can emit request-level errors and continue; no non-entry rank is waiting
  on a matching TP collective at that point.

The fatal path requires runner-level plumbing: when a child process exits
non-zero, the multi-process runner must fail all active Coordinator futures and
stream queues with a non-empty fatal error before shutdown. Otherwise HTTP
requests can hang after the TP group is killed.

## Runner Contract

`SGLangEncoderRunner` owns only SGLang runtime setup and encoder execution:

- Build encoder-only `ServerArgs` / `ModelConfig`.
- Initialize distributed state even at `tp_size=1`, matching upstream
  `MMEncoder` behavior.
- Initialize tensor model parallel groups.
- Instantiate only adapter-declared encoder submodules.
- Partial-load checkpoint weights matching declared prefixes.
- Expose `encode_batch(plan)`.

The runner always sees exactly one CUDA device as `cuda:0`. The launcher maps
the configured physical GPU into the child process with `CUDA_VISIBLE_DEVICES`
before torch is imported. This rule applies to both `tp_size=1` and
`tp_size>1`, so runner code does not branch on physical GPU ids.

`server_args_overrides` is for safe SGLang loading/runtime knobs such as
quantization, load format, attention backend, and remote weight loading. It must
not override rank topology, GPU placement, encoder-only mode, or AR-only memory
knobs such as `mem_fraction_static`, `max_running_requests`,
`max_prefill_tokens`, or `context_length`.

## Adapter Contract

Each SGLang-backed encoder stage provides an `EncoderAdapter`:

- `encoder_specs`: declares encoder submodules and checkpoint prefixes.
- `request_cost_fn(payload)` and, when additive costs are insufficient,
  `batch_cost_fn(payloads)`: estimate activation memory for admission control.
- `build_batch(messages)`: converts v1 payloads into a deterministic
  `BatchPlan` on every rank.
- `run_feature(model, plan)`: invokes the upstream encoder submodule.
- `slice_results(raw, plan, messages)`: converts raw encoder output back into
  v1 `encoder_outs` payload shape.

The adapter is model-specific shape glue. It must not catch broad `Exception`
or return fake-success outputs such as empty tensors, zero tensors, or `None`
when model execution failed.

## Admission Contract

Activation-budget admission is part of Phase 0 for both image/video and audio.
It runs only on the entry rank before TP fan-out; non-entry ranks execute the
already-admitted batch and never make independent admission decisions.

- Image/video may use the existing additive visual cost model: raw pixel bytes
  plus estimated visual output/deepstack bytes, multiplied by a calibrated
  activation factor.
- Audio must not use `request_cost_fn = 0` or `max_batch_cost = None` in the
  SGLang backend. It needs a conservative audio budget from Phase 0.
- Audio cost should be batch-aware because the input preparer right-pads
  `input_features` and `feature_attention_mask` to the maximum time dimension
  in the selected batch. A safe estimate includes padded input bytes, padded mask
  bytes, output bytes from `_get_feat_extract_output_lengths`, and an activation
  multiplier. The canonical formula and the Phase 0 multiplier default live in
  the detailed RFC — see
  [`encoder_tp_path_b_design.md` → Audio admission model](encoder_tp_path_b_design.md#audio-admission-model).
- `request_cost_fn` and `batch_cost_fn` are invoked from the scheduler hot path
  for every inbox poll. They must be **O(1) per request** and must not allocate
  GPU memory or call into model code. Adapters compute their estimates from
  CPU-side metadata: tensor shape/dtype/numel, HF config constants, and
  preprocessor-populated length fields. **Adapters must not scan tensor data
  on the hot path** — e.g. audio admission must read
  `audio_feature_lengths` from the payload (which the preprocessor is
  required to populate) instead of falling back to
  `feature_attention_mask.sum()`. A missing length is a validation error,
  not a fallback.
- Single-request guards are separate from batch formation. A request whose audio
  or visual cost exceeds the configured single-request cap fails before forward
  instead of being admitted as a batch of one and OOMing inside TP collectives.

## Config Contract

MVP keeps the public pipeline topology unchanged, but Phase 0 adds a minimal
stage memory declaration so co-located placement has a launch-time budget owner.
Encoder TP is expressed through `StageConfig` plus `StageMemoryConfig`:

```python
StageConfig(
    name="image_encoder",
    factory="sglang_omni_v1.models.qwen3_omni.stages.create_image_encoder_runner",
    factory_args={
        "backend": "sglang",
        "max_batch_size": 32,
        "max_batch_wait_ms": 50,
    },
    memory=StageMemoryConfig(
        weight_budget_bytes=3 * 1024**3,
        activation_budget_bytes=20 * 1024**3,
        relay_budget_bytes=2 * 1024**3,
    ),
    gpu=[0, 1],
    tp_size=2,
    next="mm_aggregate",
)
```

Backend rules:

- `backend="local"` keeps the current local HF `SimpleScheduler` path and is
  single-rank only.
- `backend="sglang"` uses `EncoderScheduler` + `SGLangEncoderRunner`.
- `backend="auto"` may switch to SGLang after parity is proven, but the default
  factory signature should stay `"local"`; defaults that affect launcher mode
  must live in `StageConfig.factory_args` or runtime overrides.

Launcher rules:

- Any stage with resolved `backend in {"sglang", "auto"}` must run through
  `MultiProcessPipelineRunner`, even when `tp_size=1`, so CUDA visibility and
  distributed state are process-local.
- Direct `compile_pipeline()` must reject SGLang-backed encoder stages and any
  `tp_size > 1` stage, because the single-process path cannot inject TP rank
  parameters.
- TP preflight should reject factories that do not accept TP launch parameters.
  Encoder TP also requires `backend="sglang"`.
- The parent runner allocates an `nccl_port` for every SGLang-backed stage,
  including `tp_size=1`; `SGLangEncoderRunner` rejects `nccl_port=None`.

## Memory And Co-Location Contract

Encoder and thinker ranks may share the same physical GPU in Phase 0. The budget
owner is the pipeline placement/memory planner in the parent process, not the
child runner and not SGLang AR's allocator.

Before launch, the planner groups all stage ranks by physical GPU and sums:

- AR thinker/talker weight, KV/static allocator budget.
- Encoder declared weight budget.
- Encoder activation budget used by admission control.
- Relay/staging headroom.

If the aggregate budget exceeds the target GPU budget, launch fails before any
child process starts. When an encoder rank is co-located with a thinker rank,
the planner reserves the encoder budget from the thinker `ServerArgs` according
to whether the user has pinned `mem_fraction_static` (see
[Relation to PR #430](#relation-to-pr-430-colocation-memory-work)).
`weight_memory_fraction` is not an informational field in the V0 config;
weight and activation budgets are explicit `StageMemoryConfig` inputs consumed
by the parent planner.

### Schema location: where `StageMemoryConfig` lives

`StageMemoryConfig` is a new top-level field on `StageConfig`:

```python
class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    factory: str
    # ... existing fields ...
    memory: StageMemoryConfig | None = None   # NEW
```

This is **not** nested under `runtime.resources`. v1's current
`StageConfig` (`sglang_omni_v1/config/schema.py`) has `extra="forbid"` and
no `runtime` / `resources` field; the colocation work in PR #430 added
adapter-level memory plumbing inside the SGLang backend, but it did **not**
introduce a `runtime.resources.StageResourceConfig` field on `StageConfig`.
Phase 0 therefore adds `memory` as a top-level optional field directly on
`StageConfig` rather than inventing a nested namespace that does not exist
yet.

If a future RFC (e.g. the broader typed-runtime-override primitive in Phase 4
of this design) introduces `StageConfig.runtime: StageRuntimeConfig` and
wants to relocate `memory` under it, that is a non-breaking move:
`StageConfig.memory` can become a deprecated alias for `runtime.resources.memory`
during one release. Phase 0 must not predict that nesting.

### Relation to PR #430 colocation memory work

PR #430 implemented the colocation memory plumbing in the SGLang backend
adapter (`apply_encoder_mem_reserve`, `mem_fraction_static` reservation logic
inside `create_sglang_thinker_executor_from_config`). The bridge currently
lives inside the thinker stage factory, not on `StageConfig`. Phase 0 encoder
TP adds the missing **planner-side** input — `StageConfig.memory` — that the
PR #430 bridge can read instead of probing SGLang auto-allocation behaviour.

Interaction with user-set `mem_fraction_static`:

- If the user pins `mem_fraction_static` via factory args / CLI **and** the
  planner has a co-located encoder reserve for the same GPU, the planner
  must **validate** that the pinned value already leaves room for the
  encoder reserve. If `(1 - mem_fraction_static) * total_gpu_bytes` is less
  than the encoder reserve required by `StageMemoryConfig` on the same GPU,
  launch fails with a `ValueError` naming both stages.
- If the user does not pin `mem_fraction_static`, the planner reads
  `StageMemoryConfig.weight_budget_bytes + activation_budget_bytes` for the
  co-located encoder, converts to a fraction of total GPU memory, and
  subtracts from SGLang's auto-selected `mem_fraction_static` via the
  existing `apply_encoder_mem_reserve` path.
- The planner never silently overrides a user-pinned `mem_fraction_static`.
  Conflicts surface as validation errors before any child process spawns.

## Naming

Use these names in the Phase 0 implementation:

- `SGLangEncoderRunner`, not `SGLangEncoderWorker`.
- `create_image_encoder_runner`, not `create_image_encoder_executor`.
- `create_audio_encoder_runner`, not `create_audio_encoder_executor`.
- `entry_rank` / `non_entry_rank` in new code; `entry rank` /
  `non-entry rank` in prose. Keep existing Stage `leader/follower` names only
  when discussing Stage process roles.

## Rollout

Phase 0:

- Add `SGLangEncoderRunner`.
- Add `EncoderScheduler`.
- Add Qwen3-Omni image/audio `EncoderAdapter`s.
- Add `create_image_encoder_runner` and `create_audio_encoder_runner` with
  `backend="local"` as the default.
- Add conservative image/video and audio activation admission, including
  single-request guards.
- Add `StageMemoryConfig` aggregation for co-located encoder + thinker launch.
- Add launcher validation and single-visible-device handling for SGLang-backed
  encoder stages.
- Allocate `nccl_port` in the parent for every SGLang-backed stage, including
  `tp_size=1`, and reject missing ports in the runner.
- Add fatal-path plumbing: a non-zero TP child exit tears down the stage group
  and fails all active Coordinator futures / stream queues with a non-empty
  error.
- Keep default configs on local backend.

Phase 1:

- Validate local vs SGLang parity at `tp_size=1`.
- Validate SGLang `tp_size=1` vs `tp_size=2`.
- Validate long-video Qwen3-Omni speech path without the current thinker OOM /
  encoder memory reserve workaround.

Phase 2:

- Switch selected Qwen3-Omni configs to explicit `backend="auto"` once parity
  and fault handling pass.

Phase 3:

- Delete duplicated local encoder implementations after one release with
  SGLang-backed encoders as the default for supported models.

## Validation Summary

Minimum validation should cover:

- Unit: upstream compatibility smoke for the allowlisted SGLang symbols and
  Qwen3-Omni image/audio module signatures.
- Unit: adapter batch planning, payload round-trip, skip/cache handling, and
  image/audio activation-budget admission.
- Unit: co-located encoder + thinker memory aggregation applies the encoder
  reserve to the thinker `ServerArgs` before launch.
- Unit: TP metadata/tensor fan-out does not pickle tensor payload bytes and does
  not issue device broadcasts after allocation failure.
- Unit: pre-forward failures emit request-level errors; forward-time TP faults
  take the fatal stage-group path.
- GPU: Qwen3-Omni image/audio local vs SGLang parity at `tp_size=1`.
- GPU: Qwen3-Omni image/audio SGLang `tp_size=1` vs `tp_size=2`.
- E2E: long-video speech request that previously OOMed produces valid output
  without fake-success fallbacks.

## Open Questions

1. What should the upstream partial encoder-submodule loader API look like
   after Phase 1 validates the local SGLang-Omni compatibility shim?
2. When upstream Qwen3-Omni encoder DP is complete, do we expose DP as a second
   parallelism axis or keep this RFC TP-only and add a separate DP design?

## Progress Tracking

- [ ] Phase 0: land runner, scheduler, adapters, memory aggregation, launcher
      validation.
- [ ] Phase 1: local vs SGLang and tp1 vs tp2 parity.
- [ ] Phase 2: opt selected Qwen3-Omni configs into `backend="auto"`.
- [ ] Phase 3: remove duplicated local encoder code after one release.
