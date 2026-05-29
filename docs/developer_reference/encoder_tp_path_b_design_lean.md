# RFC: Multimodal Encoder TP via SGLang Native Encoders (Plan B, Lean Draft)

Issue: https://github.com/sgl-project/sglang-omni/issues/375

Decision: SGLang-Omni should use SGLang main's native multimodal encoder
implementations and inherit SGLang tensor parallelism, instead of maintaining
separate encoder copies under `models/<name>/components/`.

This lean draft is the normative design contract for #375. Detailed Python
sketches, per-key server-argument protection, and assertion-level test plans
belong in Encoder TP PR code comments or non-normative implementation notes.

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
- Do not infer memory budgets in the Encoder TP PR. SGLang-style startup
  heuristics for deriving Omni stage-budget defaults and profile /
  benchmark-based auto tuning are follow-up work.

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

The Encoder TP PR should implement partial encoder-submodule loading inside
SGLang-Omni. Each adapter declares the encoder module specs it needs, including
checkpoint prefixes and any key rewrites. `SGLangEncoderRunner` builds only
those submodules and loads only matching checkpoint keys. This keeps the
implementation unblocked without waiting for an upstream SGLang loader API.

After parity and fault-handling validation, we should propose an
upstream helper such as `EncoderModelRunner` / partial encoder loader so future
models can reuse the same submodule-loading contract directly from SGLang.

For Qwen3-Omni concretely, this scopes the encoder GPU to ~2 GB (vision) or
~0.5 GB (audio) instead of the full ~57 GB thinker LLM. The detailed RFC has
the per-component sizing table and the partial-load pipeline:
[`encoder_tp_path_b_design.md` → Weight Loading Scope](encoder_tp_path_b_design.md#weight-loading-scope).

## Upstream Compatibility Contract

The Encoder TP PR may use a local `SGLangEncoderRunner` shim, but the shim has a bounded
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

Encoder TP compatibility is pinned to the SGLang commit/version exercised in CI.
Tracking SGLang main is acceptable only if CI imports every allowed symbol,
instantiates Qwen3-Omni image/audio encoder modules at `tp_size=1`, verifies
partial loading only accepts declared prefixes, and runs adapter smoke tests
that validate output shapes. If the Encoder TP PR needs an upstream API outside this
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

Launcher / runner CUDA view extends the existing `get_stage_process_env`
behaviour in `sglang_omni/pipeline/stage_process.py:297`:

- **`tp_size > 1`** — the launcher remaps `CUDA_VISIBLE_DEVICES` to the
  single assigned physical GPU before torch is imported. The runner sees
  that GPU as `cuda:0`, identical to thinker / talker TP today.
- **`tp_size = 1, resolved execution backend "sglang"`** — the launcher
  remaps `CUDA_VISIBLE_DEVICES` to the single assigned physical GPU
  and sets `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true` for this
  process, the same shape it uses for `tp_size>1`. The runner sees
  the GPU as `cuda:0` and uses `cuda_device=0, dist_local_rank=0`.
  This is safe because the
  [SGLang-backed stages must own their OS process](#sglang-backed-stages-must-own-their-os-process)
  rule already forbids sharing the process with any sibling, so the
  TP-only env shape no longer conflicts with shared-process
  scheduling. Using the same shape for both tp_size lanes also
  avoids the `local_rank == gpu_id` problem: upstream uses
  `local_rank` not just for CUDA index fallback
  (`parallel_state.py:267-271`) but also for **node-local-rank
  decisions** — local-master gating
  (`custom_all_reduce_utils.py:280`) and checkpoint shard slicing
  (`weight_utils.py:814-820`). At `world_size=1, local_rank=4` those
  would skip rank-0 work and slice files as if ranks 0..3 existed.
  Keeping `local_rank=0` recovers correct local-master and
  shard-slice behaviour.

This launch contract depends on treating every SGLang-backed encoder
stage as process-exclusive, even at `tp_size=1`. Local-only
`tp_size=1` stages may still use the current shared-process path and
leave `get_stage_process_env` as `{}`, but once a stage resolves to the
SGLang backend it must use the single-device remap above so SGLang's
CUDA index and node-local-rank assumptions stay aligned.

### SGLang-backed stages must own their OS process

The current process-grouping rule (`sglang_omni/config/topology.py:_build_process_groups`)
only forbids `tp_size > 1` stages from sharing an OS process. For
SGLang-backed encoders at `tp_size=1`, the topology code would
otherwise happily merge two encoder stages with the same
`stage.process` into one OS process — and that crashes at startup.

`SGLangEncoderRunner.__init__` calls `init_distributed_environment` +
`initialize_model_parallel` even at `tp_size=1`. Upstream
`initialize_model_parallel`
(`sglang/python/sglang/srt/distributed/parallel_state.py:1822-1823`)
guards against re-init with a module-level
`assert _TP is None`. The second SGLang-backed runner in the same
process would hit that assertion and abort the process.

Encoder TP rule: **any stage with resolved execution backend `"sglang"`
must own its OS process exclusively**, regardless of `tp_size`. The
condition is "any process group containing a SGLang-backed stage has
exactly one member" — sharing with another SGLang stage and sharing
with a CPU sibling (preprocessing / decode / aggregate) are both
rejected. The check fires on `len(group) > 1` after grouping, not on
"two SGLang stages" specifically, because the SGLang stage's
distributed-init globals (`_TP`, world group) conflict with whatever
torch state the sibling brings into the process even when the
sibling never calls SGLang. This rule sits next to the existing
"TP stages must own their OS process exclusively" invariant in
`stage_group.py:23-40`, and the placement planner enforces it
pre-spawn.

Implications for the launcher:

- `tp_size > 1, backend="sglang"` — already exclusive by the
  TP rule. No new behaviour.
- `tp_size = 1, backend="sglang"` — the topology builder must put
  this stage in its own `ProcessGroupPlacement`. Sharing with a
  CPU-side sibling stage (preprocessing, decode) is **also**
  rejected: SGLang's distributed-state machinery touches torch
  internals that conflict with sibling stage logic even when the
  sibling never calls SGLang.
- `tp_size = 1, backend="local"` — unchanged. May share an OS
  process with siblings per existing rules.

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

Activation-budget admission is part of the Encoder TP PR for both image/video and audio.
It runs only on the entry rank before TP fan-out; non-entry ranks execute the
already-admitted batch and never make independent admission decisions.

- Image/video uses a per-rank variant of the existing additive visual cost
  model: raw pixel bytes stay replicated on every rank, while the estimated
  visual output/deepstack activation proxy is divided by `tp_size`, then
  multiplied by the calibrated activation factor.
- Audio must not use `request_cost_fn = 0` or `max_batch_cost = None` in the
  SGLang backend. It needs a conservative audio budget from the Encoder TP PR.
- Audio cost should be batch-aware because the input preparer right-pads
  `input_features` and `feature_attention_mask` to the maximum time dimension
  in the selected batch. A safe estimate includes padded input bytes, padded mask
  bytes, output bytes from `_get_feat_extract_output_lengths`, and an activation
  multiplier. The canonical formula and the Encoder TP multiplier default live in
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
  The SGLang Qwen3-Omni encoder factories default this cap to the typed
  `encoder_activation_budget_bytes`; an explicit `max_single_request_cost`
  override may still tighten or loosen it for targeted experiments.

## Config Contract

Encoder TP reuses the existing typed `StageRuntimeConfig` /
`StageResourceConfig` shape introduced for colocation (PR #430,
`sglang_omni/config/schema.py:45-103`). The Encoder TP PR adds one
encoder temporary-admission field —
`encoder_activation_budget_bytes` — under `runtime.resources`. It does
**not** introduce a new top-level `StageMemoryConfig`:

```python
StageConfig(
    name="image_encoder",
    factory="sglang_omni.models.qwen3_omni.stages.create_image_encoder_executor",
    factory_args={
        "backend": "sglang",
        "max_batch_size": 32,
        "max_batch_wait_ms": 50,
    },
    runtime=StageRuntimeConfig(
        resources=StageResourceConfig(
            # Existing field (PR #430): per-rank/process resident/static
            # placement budget as a fraction of total physical GPU memory.
            # For AR stages it also bounds KV headroom through PR #430.
            total_gpu_memory_fraction=0.20,
            # NEW (this RFC): encoder temporary activation admission guard,
            # in bytes. Drives `EncoderScheduler.max_batch_cost` and also
            # participates in GPU-level dynamic reserve validation.
            encoder_activation_budget_bytes=20 * 1024**3,
        ),
    ),
    gpu=[0, 1],
    tp_size=2,
    next="mm_aggregate",
)
```

The factory-args resolver (`sglang_omni/config/runtime.py:resolve_stage_factory_args`)
already injects `total_gpu_memory_fraction` into the factory signature
after merging `factory_args` + `runtime_overrides`, and the Encoder TP
PR extends it with the same shape for `encoder_activation_budget_bytes`:

- If the resolved factory has `encoder_activation_budget_bytes` in its
  signature, inject
  `runtime.resources.encoder_activation_budget_bytes` as that kwarg.
- If `factory_args` or `runtime_overrides` contain
  `encoder_activation_budget_bytes` **as a key** — including the
  YAML form `encoder_activation_budget_bytes: null` that deserializes
  to `None` — fail with a `ValueError`. The check must be by key
  presence (`key in d`), not by non-None value: a `None` value would
  otherwise satisfy a `.get(...) is not None` check and disable the
  admission cap silently. Typed source is the only valid path.

See the detailed RFC for the resolver-side sketch and the rejection
rule:
[`encoder_tp_path_b_design.md` → Injection contract for `encoder_activation_budget_bytes`](encoder_tp_path_b_design.md#injection-contract-for-encoder_activation_budget_bytes).

Backend rules:

- `backend="local"` keeps the current local HF `SimpleScheduler` path and is
  single-rank only.
- `backend="sglang"` uses `EncoderScheduler` + `SGLangEncoderRunner`.
- `backend="auto"` may switch to SGLang after parity is proven, but the default
  factory signature should stay `"local"`; defaults that affect launcher mode
  must live in `StageConfig.factory_args` or runtime overrides.

Launcher rules:

- The server entry (`sglang_omni/serve/launcher.py:212`) already routes
  every pipeline through `MultiProcessPipelineRunner` unconditionally;
  there is no longer a single-process compile branch to reject from.
  `MultiProcessPipelineRunner.start()` calls
  `prepare_pipeline_runtime`
  (`sglang_omni/pipeline/runtime_config.py:78`), which builds the
  placement plan and process-topology plan; per-process spawn is
  driven by `_build_stage_groups`
  (`sglang_omni/pipeline/mp_runner.py:37`). Encoder TP
  validation hooks into those three call sites instead of a
  vanished `compile_pipeline()`.
- `prepare_pipeline_runtime` builds a stage launch-mode map before
  process topology. The map is computed from `StageConfig.factory_args`
  plus typed runtime overrides; factory defaults that affect launch mode
  are not a valid source of truth. The same map feeds topology
  validation, `_build_stage_groups`, SGLang process exclusivity,
  environment remap, TP preflight, and `nccl_port` allocation. For
  `backend="auto"`, the map records both the requested backend
  (`"auto"`) and the resolved execution backend (`"sglang"` or
  `"local"`).
- TP-launch-kwarg preflight runs inside `_build_stage_groups` and
  applies to **every stage that will receive TP launch kwargs** —
  `tp_size > 1` **or** resolved execution backend `"sglang"`
  (including single-rank SGLang encoders, which also receive a
  parent-allocated `nccl_port` and `tp_rank=0, tp_size=1`). Any such
  factory whose signature does not accept `tp_rank/tp_size/nccl_port`
  fails preflight before spawn. For `backend="auto"`, the resolved
  execution backend decides this lane: single-rank `auto -> local`
  keeps the local launch shape, while single-rank `auto -> sglang`
  receives the SGLang launch kwargs and NCCL port. A second layer
  applies only to backend-aware factories: `tp_size > 1` ⇒ resolved
  execution backend is `"sglang"`. Therefore `backend="auto",
  tp_size > 1` passes only when auto-resolution selects SGLang; if it
  resolves to local, it fails the same way as explicit
  `backend="local"`. AR factories (thinker/talker) take TP launch
  params without a `backend` parameter and pass through Layer 1
  unchanged.
- Process exclusivity for SGLang-backed stages is enforced in
  `build_process_topology_plan`
  (`sglang_omni/config/topology.py:36`): any process group that
  contains a stage with resolved execution backend `"sglang"` must have
  exactly one member.
- The parent runner allocates an `nccl_port` for every SGLang-backed
  stage (including `tp_size=1`) inside `_build_stage_groups`;
  `SGLangEncoderRunner` rejects `nccl_port=None`.

## Memory And Co-Location Contract

Encoder TP supports colocating thinker, talker, encoder, and code2wav
processes on the same physical GPU, but memory ownership is explicit.

In upstream single-engine SGLang, `mem_fraction_static = y` is the
model-weights-plus-KV budget, while `1 - y` is left for activations,
CUDA graph buffers, workspace, and other same-GPU applications. The
core sizing formula is:

```text
rest_memory =
    post_model_load_memory - pre_model_load_memory * (1 - mem_fraction_static)
```

That assumes one SGLang engine owns the GPU. Omni colocation breaks
that assumption: the physical GPU may host thinker, talker_ar,
image_encoder, audio_encoder, and code2wav processes, so
`1 - mem_fraction_static` can no longer be interpreted as the AR
stage's private dynamic reserve.

`runtime.resources.total_gpu_memory_fraction` is the per-rank/process
resident/static placement budget for every GPU stage. It accounts for
weights, KV cache, persistent buffers, GPU-resident encoder cache, and
vocoder steady state. It is **not** a peak memory cap. After TP rank
expansion, the planner sums resident budgets per physical GPU.

For AR stages, the colocated adapter mirrors
`total_gpu_memory_fraction` into SGLang `mem_fraction_static` for
compatibility, but PR #430 changes KV sizing to use a process-scoped
stage budget:

```text
stage_budget_bytes = physical_gpu_total_bytes * total_gpu_memory_fraction
available_for_kv = stage_budget_bytes - measured_ar_process_memory
```

These are not two multiplicative limits. `total_gpu_memory_fraction`
is the Omni resident placement budget; `mem_fraction_static` is the
SGLang compatibility parameter.

For non-AR encoder and code2wav stages,
`total_gpu_memory_fraction` is placement-side resident budgeting only.
It does not install a runtime allocator cap, and code2wav has no
`mem_fraction_static`.

Dynamic memory is a GPU-level shared resource:

```text
shared_dynamic_headroom_bytes =
    physical_gpu_total_bytes * (
        max_total_gpu_memory_fraction_per_gpu - resident_fraction_sum
    )
```

That headroom must cover AR activation peaks, CUDA graph buffers,
workspace, encoder temporary activations, code2wav temporary peaks,
NCCL / allocator fragmentation, and safety margin. The Encoder TP PR
therefore validates dynamic reserves before launch:

```text
sum(stage_resident_budget_bytes_on_gpu)
+ sum(resolved_dynamic_reserve_bytes_on_gpu)
<= physical_gpu_total_bytes * max_total_gpu_memory_fraction_per_gpu
```

Equivalently, `encoder_activation_budget_bytes` plus resolved AR,
code2wav, and miscellaneous dynamic reserve assumptions must fit in
the shared dynamic headroom. This prevents a configuration that assigns
all GPU memory to resident stage budgets and leaves no space for
activations.

For the Encoder TP PR, non-encoder dynamic reserves come from calibrated
model-config assumptions or advanced typed overrides. Do not expose
`thinker_activation_budget_bytes` / `talker_activation_budget_bytes` as
ordinary user-facing primary knobs; long term, a placement-aware startup
heuristic should derive them from model, topology, and runtime settings.
Here, "calibrated assumptions" means compile-time hard-coded constants
for a known `(model, hardware)` profile, not runtime inference; this is
why they do not conflict with the non-goal of avoiding budget inference
in the Encoder TP PR.

`encoder_activation_budget_bytes` is the Encoder TP admission guard for
image / video / audio temporary activation peaks. It caps encoder batch
formation through `EncoderScheduler.max_batch_cost` and participates in
dynamic reserve validation. It is a bridge field for this PR, not the
long-term user-facing memory split API.

The Encoder TP PR does not infer budgets. It consumes explicit typed
budgets, rejects ambiguous legacy/untyped paths, and logs the resolved
resident and dynamic budgets at startup. SGLang-style startup
heuristics for deriving Omni stage-budget defaults, and profile /
benchmark-based auto tuning, are follow-up work.

`encoder_mem_reserve` is a pre-#430 bridge: it subtracts an external
encoder reserve from SGLang's auto-picked AR `mem_fraction_static`.
It is not part of the Encoder TP contract. Configs using typed
`runtime.resources.total_gpu_memory_fraction` must not also set a
nonzero `encoder_mem_reserve`. The current Qwen thinker factory default
`encoder_mem_reserve=0.05` must be removed or changed to `0.0` in the
Encoder TP implementation. Legacy reserve behavior is allowed only when
nonzero `encoder_mem_reserve` is explicitly configured on the legacy
non-typed path; typed-budget configs must ignore the old default and
reject any explicit nonzero value.

See [`encoder_tp_path_b_design.md` → Memory accounting reuses PR #430](encoder_tp_path_b_design.md#memory-accounting-reuses-pr-430)
for the line-by-line walk through `SGLModelRunner._profile_available_bytes`
and the deletion of the obsolete planner-side reserve plumbing.

## Naming

Use these names in the Encoder TP implementation:

- `SGLangEncoderRunner`, not `SGLangEncoderWorker` (new class, new file).
- Existing factory names stay: `create_image_encoder_executor` and
  `create_audio_encoder_executor`
  (`sglang_omni/models/qwen3_omni/stages.py:781,823`). The Encoder TP PR extends
  them with `backend: Literal["local", "sglang", "auto"] = "local"`
  rather than renaming. Renaming would require migrating
  `sglang_omni/models/qwen3_omni/config.py:45,59` and any cookbook
  configs that already reference `_executor`; the kept name does not
  block the SGLang code path because the factory body branches on
  `backend`.
- The encoder admission kwarg the factory accepts is
  `encoder_activation_budget_bytes` — the **full**
  `runtime.resources` field name, not a shortened
  `activation_budget_bytes`. `resolve_stage_factory_args` matches the
  field name against the factory signature exactly (same shape PR #430
  uses for `total_gpu_memory_fraction`); a shorter kwarg name would
  silently fail to receive the injected value, leaving
  `max_batch_cost=None` and the admission cap disabled.
- `entry_rank` / `non_entry_rank` in new code; `entry rank` /
  `non-entry rank` in prose. Keep existing Stage `leader/follower` names only
  when discussing Stage process roles.

## Rollout

Encoder TP PR scope:

- Add `SGLangEncoderRunner`.
- Add `EncoderScheduler`.
- Add Qwen3-Omni image/audio `EncoderAdapter`s.
- Extend existing `create_image_encoder_executor` /
  `create_audio_encoder_executor` factories with
  `backend: Literal["local", "sglang", "auto"] = "local"`. Default
  stays `"local"`; the factory body routes to `SGLangEncoderRunner` +
  `EncoderScheduler` when the resolved execution backend is `"sglang"`.
- Add conservative image/video and audio activation admission, including
  single-request guards.
- Add `runtime.resources.encoder_activation_budget_bytes` field on
  `StageResourceConfig` and include it in dynamic reserve validation.
- Extend `resolve_stage_factory_args` to inject the new field and reject
  duplicates in `factory_args` / `runtime_overrides`, using the same shape
  as the existing `total_gpu_memory_fraction` injection.
- Build the stage launch-mode map before topology so topology,
  launcher preflight, process exclusivity, SGLang env remap, and
  `nccl_port` allocation all use the same backend decision.
- Add GPU-level dynamic headroom validation:
  resident budgets plus resolved dynamic reserves must fit under
  `PlacementConfig.max_total_gpu_memory_fraction_per_gpu`.
- Add launcher validation for SGLang-backed encoder stages. Widen
  `get_stage_process_env`'s existing TP remap branch so any stage with
  resolved execution backend `"sglang"` gets the single-device shape
  (`CUDA_VISIBLE_DEVICES=<one>` + `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true`)
  regardless of `tp_size`. The runner then uses `cuda_device=0,
  dist_local_rank=tp_rank` uniformly across tp lanes (which is `0` at
  `tp_size=1`). Local-only `tp_size=1` stages keep the early-return
  `{}`. The earlier direct physical-GPU pinning variant is retracted
  because it left `local_rank=gpu_id`, which
  breaks SGLang's node-local-rank checks (local-master gating and
  checkpoint shard slicing assume `local_rank ∈ [0, world_size)`).
- Allocate `nccl_port` in the parent for every SGLang-backed stage, including
  `tp_size=1`, and reject missing ports in the runner.
- Add fatal-path plumbing: a non-zero TP child exit tears down the stage group
  and fails all active Coordinator futures / stream queues with a non-empty
  error.
- Validate local vs SGLang parity at `tp_size=1`.
- Validate SGLang `tp_size=1` vs `tp_size=2`.
- Validate long-video Qwen3-Omni speech path without the current thinker OOM /
  encoder memory reserve workaround.
- Keep default configs on local backend until parity and fault handling pass;
  selected Qwen3-Omni configs may then opt into explicit `backend="auto"`.

Follow-up work:

- Derive Omni stage-budget defaults with a placement-aware startup heuristic.
- Explore profile / benchmark / OOM-feedback auto tuning.
- Delete duplicated local encoder implementations after one release with
  SGLang-backed encoders as the default for supported models.

## Validation Summary

Minimum validation should cover:

- Unit: upstream compatibility smoke for the allowlisted SGLang symbols and
  Qwen3-Omni image/audio module signatures.
- Unit: adapter batch planning, payload round-trip, skip/cache handling, and
  image/audio activation-budget admission (target the
  `runtime.resources.encoder_activation_budget_bytes` injection path
  and `reject_untyped_encoder_activation_budget_bytes`).
- Unit: launch-mode map is computed before topology and reused by
  topology validation, `_build_stage_groups`, SGLang env remap, and
  TP-launch-kwarg preflight.
- Unit: co-located encoder + thinker memory accounting under PR #430's
  process-scoped path — assert the AR runner's `gpu_mem_accounting`
  log records the `nvml_process` or `stage_load_fallback` branch and
  the available-for-KV value matches
  `total_memory * total_gpu_memory_fraction - process_used`.
- Unit: dynamic headroom validation rejects resident budgets plus
  encoder / AR / code2wav / safety dynamic reserves that exceed the
  per-GPU placement limit.
- Unit: typed `runtime.resources.total_gpu_memory_fraction` configs
  reject nonzero legacy `encoder_mem_reserve`; the Encoder TP path does
  not call planner-side `apply_encoder_mem_reserve`; the Qwen thinker
  factory default is `0.0` or otherwise ignored unless the user
  explicitly configures the legacy reserve.
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
   after the local SGLang-Omni compatibility shim is validated?
2. When upstream Qwen3-Omni encoder DP is complete, do we expose DP as a second
   parallelism axis or keep this RFC TP-only and add a separate DP design?

## Progress Tracking

- [ ] Encoder TP PR: runner, scheduler, adapters, cost admission, launch-mode
      map, memory validation, fatal-path plumbing, and parity validation.
- [ ] Follow-up: placement-aware Omni startup heuristic for stage budgets.
- [ ] Follow-up: profile / benchmark / OOM-feedback auto tuning.
- [ ] Follow-up: remove duplicated local encoder code after one release.
