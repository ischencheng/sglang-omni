# Encoder TP Plan B PR Goal

Complete issue #375 Encoder TP Plan B for all phases. The lean design is normative; the long design is supplemental.

## Inputs And Sync

Before implementation or validation:

- Confirm the current PR branch is based on the latest `sgl-project/sglang-omni` `origin/main`.
- Read issue #375, PR #423, the latest RFC/design branch discussion, `encoder_tp_path_b_design_lean.md`, and `encoder_tp_path_b_design.md`.
- Convert issue comments, PR review comments, and design requirements into a tracked checklist.
- If comments conflict with the design docs, pause and report the conflict instead of silently choosing one.
- Do not use a stale local `main` branch as upstream evidence.

## Required End State

- Implement `Stage -> EncoderScheduler -> SGLangEncoderRunner -> upstream encoder module`.
- SGLang backend must use upstream encoder modules and SGLang TP kernels.
- Encoder stages must not instantiate full `ForConditionalGeneration`.
- Keep local backend working as fallback and baseline.
- Do not make `SimpleScheduler` TP-aware.
- SGLang-backed stages run in separate OS processes even at `tp_size=1`.
- `tp_size=1` uses single visible device remap and parent-allocated `nccl_port`.
- Implement entry-rank input drain, metadata CPU broadcast, tensor device broadcast, and allocation-ready handshake.
- Large tensors must not be sent through Python pickle or CPU payload fanout.
- Implement recoverable request-level errors and fatal TP collective/forward failure semantics.
- `backend="auto"` must record requested and resolved execution backend. Topology, preflight, `nccl_port` allocation, environment remap, and launch-mode decisions must use the resolved backend.
- Implement typed memory contract:
  - `total_gpu_memory_fraction` is resident/static budget.
  - `encoder_activation_budget_bytes` is temporary activation/admission budget.
  - dynamic headroom validation is required.
  - typed configs reject explicit nonzero `encoder_mem_reserve`.
- Protected `server_args_overrides` cannot override topology, encoder/language/DP flags, AR memory knobs, cuda graph, or device.
- Do not fake success with empty tensors, zero outputs, or `None`.

## Design Checklist

Create `docs/developer_reference/encoder_tp_design_checklist.md` mapping every lean-design requirement, long-design supplement, issue comment requirement, and PR review requirement to:

- implementation location,
- test name or command,
- log or benchmark evidence,
- status: implemented, tested, or deferred.

Deferred items need an explicit reason and risk.

## Unit Tests

Cover at least:

- schema injection,
- launch-mode map,
- protected override rejection,
- dynamic headroom validation,
- legacy reserve rejection,
- TP preflight,
- `nccl_port` allocation for SGLang-backed `tp=1` and `tp>1`,
- adapter batch planning, payload roundtrip, skip/cache/admission,
- audio nonzero cost,
- padding-aware audio batch cost,
- `audio_feature_lengths` propagation,
- missing audio length validation error,
- no tensor pickle fanout,
- allocation failure no broadcast,
- recoverable error path,
- fatal forward/collective error path,
- Coordinator future/stream fail-all behavior.

## Precision

Run and report:

1. local HF backend vs SGLang backend `tp=1`.
   - This is backend baseline delta.
   - It does not need strict allclose, but must be quantified and attributed.
   - If caused by runner, adapter, partial load, dtype/device, padding/slicing, batch construction, or input mismatch, fix it.
2. SGLang `tp=1` vs SGLang `tp>1`.
   - This is the core Encoder TP precision gate.
   - Test at least `tp=2`; test `tp=4` if available.

Precision runs must use controlled experiment hygiene:

- fixed seed where applicable,
- `eval()` and no grad,
- identical input preprocessing and cached request payloads,
- identical dtype/autocast policy except for the explicit variable being tested,
- identical model revision and tokenizer/processor revision,
- identical PR commit for SGLang `tp=1` vs `tp>1`,
- stable comparison cutpoints such as adapter input, upstream encoder hidden states, projected encoder features, and final adapter output,
- predeclared numeric and semantic acceptance thresholds before judging the result.

For image, video, and audio encoders record:

- shape,
- dtype,
- device,
- input length,
- batch size,
- max abs error,
- mean abs error,
- relative error,
- cosine similarity.

If `tp=1` vs `tp>1` is not strict allclose, ablate:

- partial weight load,
- encoder submodule selection,
- adapter slicing,
- audio/video length handling,
- distributed wrapper,
- dtype/autocast,
- device placement,
- batch padding.

Only after ruling those out, attribute remaining error to TP numerical effects such as:

- RowParallel all-reduce order,
- Column/QKV/Gate-Up shard GEMM rounding,
- fused kernel differences,
- matmul precision,
- nonlinear amplification.

Do not use local-vs-SGLang baseline delta to hide TP-introduced error.

Write the precision results and attribution in `docs/developer_reference/encoder_tp_parity_findings.md`.

## E2E Tests

Run long-video and long-audio E2E tests.

Before claiming this PR fixes issue #375, build a two-layer evidence chain.

Layer 1: upstream-main reproduction.

- Use the latest `sgl-project/sglang-omni` `origin/main`, not a stale local `main` branch and not the PR branch.
- Prefer a separate git worktree or otherwise clean checkout for upstream-main reproduction so the PR branch working tree is not disturbed.
- Use the same model, long-video/long-audio request, machine class, and launch settings whenever possible.
- Show the original issue behavior on upstream main:
  - multi-modal encoder OOM,
  - missing/insufficient admission,
  - or success only after relying on the legacy `encoder_mem_reserve` workaround.
- This layer proves the issue exists on main. It does not need to use the same commit as the PR branch because main does not contain the PR implementation.
- Record the exact upstream-main SHA, `git status --short`, `git diff --stat`, and whether the local branch or worktree was synchronized before running it.

Layer 2: PR-branch controlled A/B.

- Use one PR-branch commit for both sides of the comparison.
- Change only runtime/config values needed to switch encoder TP, backend, budgets, or launch mode.
- For both long video and long audio, use the same model, same request payload, same typed budgets, same machine class, and same PR commit.
- Record `git rev-parse HEAD`, `git status --short`, `git diff --stat`, exact config files, model revision, and exact commands for every run.
- If the working tree is dirty, explain the dirty diff and why it is part of the tested PR state.
- Run baseline with no encoder TP, or with SGLang encoder `tp=1`.
- Show that the baseline either:
  - OOMs in the multi-modal encoder path,
  - is rejected by admission because encoder activation budget is insufficient,
  - or requires the legacy `encoder_mem_reserve` workaround to avoid failure.
- Run the same request with encoder `tp>1`.
- Show that TP avoids the failure without fake success and without relying on legacy `encoder_mem_reserve`.
- Record peak GPU memory per rank, admission decision, batch cost, latency, output validity, and server health for both baseline and TP runs.

The controlled evidence chain should be:

- `origin/main` + long input reproduces the issue.
- same PR commit + same input/config except encoder `tp=1` still fails or is safely rejected.
- same PR commit + same input/config except encoder `tp>1` succeeds.

If the exact original OOM cannot be reproduced because the available GPU has more memory or the model/input differs, create the smallest defensible stress case by increasing video/audio length or lowering the typed encoder activation budget until the PR-branch `tp=1` case fails or is rejected, then show `tp>1` succeeds under the same typed budget. If even that cannot be reproduced, report it as a blocker and do not claim the PR fixed the OOM; only claim that TP E2E passed under the tested conditions.

Long video must show:

- request succeeds,
- output is valid,
- no encoder OOM,
- no fake success,
- no dependency on `encoder_mem_reserve` workaround,
- server remains healthy.

Long audio must show:

- audio admission/cost is correct,
- padding-aware batch cost works,
- `audio_feature_lengths` is respected,
- long-sequence activation memory is controlled,
- `slice_results` remains aligned.

Record:

- command,
- model revision,
- branch/commit,
- GPU type/count,
- driver/CUDA/PyTorch,
- request success rate,
- output validity,
- server health,
- GPU peak memory,
- admission decision,
- batch cost,
- relevant logs.

If possible, run SeedTTS correctness/performance or a representative subset and ensure OOM/failures are not silently counted as bad WER.

## Memory Scheme Necessity

Do not only implement the new memory contract. Prove why it is needed.

Show why old knobs are insufficient:

- `mem_fraction_static` controls AR/thinker static and KV behavior, but cannot directly model temporary multi-modal encoder activation memory.
- Increasing `mem_fraction_static` may help startup KV allocation but reduces dynamic headroom.
- Decreasing `mem_fraction_static` may help runtime activation headroom but can make KV/static allocation fail.
- `encoder_mem_reserve` is a coarse legacy workaround, not typed, not request-aware, not batch-aware, and not a substitute for admission control.

Justify the new split:

- `total_gpu_memory_fraction` is resident/static per-rank/process budget.
- `encoder_activation_budget_bytes` is temporary encoder activation/admission budget.
- dynamic headroom validation prevents co-located resident and dynamic budgets from exceeding per-GPU memory policy.
- request/batch cost admission rejects or batches long video/audio before unsafe TP forward.

Required evidence:

- Try both long-video and long-audio baselines. For each, show old behavior OOMs, admission is missing, or success requires `encoder_mem_reserve`.
- If only one modality can reproduce the issue, document why the other could not be reproduced and provide admission, cost, and peak-memory evidence for the non-reproducing modality.
- Show the same or equivalent case handled by typed encoder activation budget plus admission and encoder TP.
- Include logs proving resolved resident/static budgets, resolved dynamic budgets, admission decision, batch cost, and peak memory.
- Explain why the result could not be achieved cleanly by only tuning `mem_fraction_static`.
- Explain why retaining `encoder_mem_reserve` as the primary solution would be less correct than typed budget plus admission.

The final report must clearly distinguish:

- thinker startup KV/static memory,
- thinker runtime activation memory,
- encoder resident memory,
- encoder temporary activation memory,
- unrelated external GPU memory pressure.

## Performance And Memory Report

Produce `docs/developer_reference/encoder_tp_performance_report.md` covering:

- startup/load memory,
- `pre_model_load_memory`,
- `post_model_load_memory`,
- resident/static budget,
- dynamic headroom,
- `encoder_activation_budget_bytes`,
- long-video and long-audio latency,
- throughput,
- success rate,
- peak memory,
- GPU utilization,
- local vs SGLang `tp=1` vs SGLang `tp>1`,
- activation peak per rank,
- TP memory distribution,
- broadcast/fanout overhead.

Benchmark methodology:

- Include warmup runs before measuring.
- Use repeated measured runs, not a single successful request, unless resource constraints make repetition impossible.
- Report mean, standard deviation, p50, p90, p95, failure rate, and sample count where applicable.
- Record same-machine background GPU processes and `nvidia-smi` snapshots before and after each benchmark group.
- Separate cold-start/load time from steady-state request latency.
- Do not mix historical upstream-main runs and PR-branch controlled A/B runs in the same aggregate table without labeling them.

Classify OOMs separately:

- startup KV/static OOM,
- runtime activation OOM,
- multi-modal encoder activation/admission OOM.

Explain which class this PR solves and which classes remain out of scope.

## Constraints

- Do not regress local backend.
- Do not regress v1 lifecycle or Coordinator abort/relay/failure semantics.
- Do not load full upstream generation model inside encoder stage.
- Do not introduce new worker/executor naming; use runner naming.
- Do not use fake success.
- Do not rely on legacy `encoder_mem_reserve` as the solution.
- Do not do unrelated refactors.
- Do not revert user changes.
- Do not use destructive git commands.

## Deliverables

At the end, provide or update:

- `docs/developer_reference/encoder_tp_design_checklist.md`,
- `docs/developer_reference/encoder_tp_parity_findings.md`,
- `docs/developer_reference/encoder_tp_performance_report.md`,
- a PR description/test-plan draft summarizing design coverage, precision attribution, OOM reproduction, E2E results, and residual risks.

## Iteration Strategy

- Start by generating the design checklist.
- Diff current implementation against the checklist.
- Fix the highest-risk missing contract first.
- Add or update tests with every behavior change.
- For precision, first establish local-vs-SGLang `tp=1`, then isolate SGLang `tp=1` vs `tp>1`.
- For OOM, classify the OOM before changing memory knobs.
- For performance, collect memory timeline/logs before optimizing.

## Stop Conditions

If blocked, stop and report:

- completed and incomplete checklist items,
- commands run,
- test results,
- benchmark tables,
- precision sources ruled out,
- remaining precision uncertainty,
- OOM classification and memory evidence,
- blocker reason,
- minimum input or resource needed to continue.
