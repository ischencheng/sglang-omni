# Encoder TP Plan B Design Checklist

Source of truth: `encoder_tp_plan_b_pr_goal.md` plus the normative lean RFC
`encoder_tp_path_b_design_lean.md`. Supplemental detail comes from
`encoder_tp_path_b_design.md`, issue #375 discussion, and PR #423.

Current audit base:

- Branch/worktree base: `HEAD == origin/main == 15b5e7c94ddccd0325f061e7500377ddfccd0434`
  as of 2026-05-26.
- Current PR state is a dirty worktree containing the Encoder TP changes.
- Issue #375 comments were read. PR #423 has no review submissions or inline
  review threads; its body explicitly scoped itself to Phase 0, so it is not
  sufficient evidence for the all-phases goal.
- Current GPU status: CUDA is available (`torch.cuda.device_count()==8`) on
  NVIDIA H200. GPU memory is shared with hidden processes outside this
  namespace; PR default AR `mem_fraction_static=0.7` startup OOMed on selected
  GPUs, while typed `mem_fraction_static=0.45` /
  `total_gpu_memory_fraction=0.45` lets PR-branch TP1/TP2 E2E run. Clean
  upstream-main was exercised separately; installed `qwen-vl-utils==0.0.14`
  blocks video before encoder execution, but an isolated compatible
  `qwen-vl-utils==0.0.11` run reproduces a 196-frame long-video
  missing-admission/KV-capacity failure. Long audio succeeds, and a real 300s
  audio file was also checked in both default truncating mode and
  `audio_truncation=false` mode; the latter produces a true 30000-frame audio
  encoder workload.

Current CPU/unit evidence:

- `pytest -q tests/test_encoder_tp_e2e_probe.py tests/unit_test/serve/test_openai_api.py`
  -> 17 passed.
- `timeout 90 python -m pytest -q tests/test_encoder_tp_launcher.py tests/unit_test/pipeline/test_topology.py`
  -> 45 passed, 8 warnings.
- `timeout 180 python -m pytest -q tests/test_encoder_server_args.py tests/test_encoder_scheduler_recv.py tests/test_encoder_scheduler_loop.py tests/test_encoder_module_container.py tests/test_encoder_adapters.py tests/test_encoder_tp_e2e_probe.py tests/unit_test/pipeline/test_runtime_adapter.py tests/unit_test/qwen3_omni/test_sglang_ar_budget.py`
  -> 124 passed, 2 warnings.
- `timeout 240 python -m pytest -q tests/test_encoder_tp_launcher.py tests/test_encoder_runner_fail_all.py tests/test_parity_compare.py tests/unit_test/pipeline/test_compile.py tests/unit_test/pipeline/test_runtime_schema.py tests/unit_test/pipeline/test_runtime_adapter.py tests/unit_test/pipeline/test_topology.py tests/unit_test/pipeline/test_placement.py tests/unit_test/qwen3_omni/test_config_manager.py tests/unit_test/qwen3_omni/test_pipeline.py tests/unit_test/qwen3_omni/test_sglang_ar_budget.py`
  -> 156 passed, 8 warnings.
- `timeout 240 python -m pytest -q tests/test_encoder_*.py tests/test_parity_compare.py tests/test_video_preprocessing.py -k 'not parity_gpu'`
  -> 145 passed, 7 deselected, 4 warnings.
- `python -m py_compile tests/conftest.py tests/_encoder_parity_harness.py tests/parity_compare.py tests/test_parity_compare.py`
  -> passed.
- `timeout 60 python -m pytest -q tests/test_parity_compare.py`
  -> 6 passed.
- `timeout 60 python -m pytest -q tests/test_encoder_tp_e2e_probe.py`
  -> 7 passed.
- `timeout 120 python -m pytest -q tests/test_encoder_scheduler_loop.py tests/test_encoder_scheduler_recv.py`
  -> 20 passed.
- `timeout 120 python -m pytest -q tests/test_encoder_adapters.py tests/test_encoder_tp_launcher.py tests/test_encoder_scheduler_loop.py tests/test_encoder_scheduler_recv.py tests/test_encoder_tp_e2e_probe.py`
  -> 82 passed, 2 warnings.
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. timeout 600 python -m pytest -q tests/test_encoder_tp_parity_gpu.py -m slow`
  -> 7 passed, 4 warnings.
- Current GPU artifacts: `/data/encoder_tp_evidence_20260526`.

Status meanings:

- `tested`: implementation exists and has a named unit/GPU/doc evidence lane.
- `implemented`: implementation exists, but the listed evidence still needs a
  current run or broader validation.
- `deferred`: not complete in the current worktree. Reason and risk are stated.

| ID | Requirement | Source | Implementation location | Test, command, or evidence | Status |
| --- | --- | --- | --- | --- | --- |
| A1 | Use Plan B: `Stage -> EncoderScheduler -> SGLangEncoderRunner -> upstream encoder module`. | lean RFC, issue #375 | `sglang_omni/scheduling/encoder_scheduler.py`; `sglang_omni/model_runner/sglang_encoder_runner.py`; `sglang_omni/models/qwen3_omni/stages.py` | `tests/test_encoder_scheduler_loop.py`; `tests/test_encoder_tp_launcher.py`; GPU parity harness | tested |
| A2 | Keep local HF backend as fallback/baseline. | lean RFC, issue #375 | `create_image_encoder_runner`; `create_audio_encoder_runner`; `_build_local_*_encoder` | `tests/unit_test/qwen3_omni/test_pipeline.py`; local-vs-SGLang parity doc | tested |
| A3 | Do not make `SimpleScheduler` TP-aware. | lean RFC | New TP code is in `EncoderScheduler`; `SimpleScheduler` untouched for TP | Code inspection; scheduler tests target `EncoderScheduler` | tested |
| A4 | Use upstream SGLang encoder modules and TP kernels. | lean RFC, issue #375 | `Qwen3OmniMoeVisionEncoder`, `Qwen3OmniMoeAudioEncoder` imported in `encoder_adapters.py`; TP init in runner | `tests/test_encoder_module_container.py`; `tests/test_encoder_tp_parity_gpu.py` | tested |
| A5 | Do not instantiate full upstream `ForConditionalGeneration` in encoder stages. | lean RFC, issue comment 4414315104 | `EncoderModuleContainer`; runner rejects empty specs and bypasses `loader.load_model` | `tests/test_encoder_module_container.py`; `test_runner_rejects_empty_encoder_specs`; partial-load footprint evidence | tested |
| A6 | Partial-load only adapter-declared checkpoint prefixes. | lean RFC, issue comment 4414315104 | `EncoderModuleSpec`; `EncoderModuleContainer.load_weights` | `tests/test_encoder_module_container.py` | tested |
| A7 | Avoid Worker/Executor vocabulary for new encoder path; use runner naming. | issue comment 4414355138, lean RFC | `SGLangEncoderRunner`; `create_*_encoder_runner`; executor aliases kept only for compatibility | Code inspection; launcher signature tests | tested |
| A8 | Use `entry_rank` / `non_entry_rank` vocabulary in new code. | issue comment 4414481808, lean RFC | Runner/scheduler fields and comments | Code inspection | tested |
| A9 | SGLang-backed stages run in separate OS processes even at `tp_size=1`. | lean RFC | `build_process_topology_plan`; `StageGroup._get_worker_process_env` | `tests/test_encoder_tp_launcher.py`; `tests/unit_test/pipeline/test_topology.py` | tested |
| A10 | `tp_size=1` SGLang uses single visible device remap and parent-allocated `nccl_port`. | lean RFC | `_build_single_stage_spec`; `get_stage_process_env`; `SGLangEncoderRunner` port validation | `tests/test_encoder_tp_launcher.py`; `tests/test_encoder_server_args.py` | tested |
| A11 | `backend="auto"` records requested and resolved backends, and topology/preflight/port/env launch decisions use resolved backend. | goal doc, issue comment 4487693772 | `StageLaunchMode`; `build_stage_launch_modes`; `stage_requires_single_visible_device` | `tests/test_encoder_tp_launcher.py::test_auto_backend_resolving_local_does_not_get_sglang_launch_args`; topology auto-local sharing test | tested |
| A12 | `tp_size>1` with backend-aware encoder requires resolved SGLang backend; `auto -> local` rejects. | lean RFC, issue comment 4524089582 | `_run_tp_preflight` | `tests/test_encoder_tp_launcher.py::test_tp_preflight_rejects_auto_backend_when_resolver_selects_local` | tested |
| B1 | Entry rank drains inputs; non-entry ranks receive no external pipeline inputs. | lean RFC | `EncoderScheduler._recv_messages`; Stage follower control plane unchanged | `tests/test_encoder_scheduler_recv.py` | tested |
| B2 | Metadata broadcast uses CPU TP group, tensor data uses CUDA/device broadcast. | lean RFC | `_strip_and_lift`; `_recv_messages` | `tests/test_encoder_scheduler_recv.py` mocked broadcast lanes | tested |
| B3 | Large tensors are not sent through pickle/CPU payload fanout. | lean RFC, goal doc | `extract_tensors` metadata skeleton + `_TensorSpec`; `dist.broadcast` tensors | `tests/test_encoder_scheduler_recv.py::test_strip_and_lift_returns_typed_dtype_specs` | tested |
| B4 | Allocation-ready handshake prevents unmatched device broadcasts after receive-buffer OOM. | lean RFC | `_allocation_ready_gather`; follower placeholder allocation path | `tests/test_encoder_scheduler_recv.py::test_follower_allocation_failure_returns_before_device_broadcast` | tested |
| B5 | Recoverable pre-forward errors produce request-level errors and scheduler continues. | lean RFC | `_gather_pre_forward_error`; `_emit_error` | `tests/test_encoder_scheduler_loop.py` recv/build recovery tests | tested |
| B6 | Forward/collective failures are fatal and cause parent fail-all behavior. | lean RFC | `_fatal_tp_forward_error`; `MultiProcessPipelineRunner._monitor_children`; `Coordinator.fail_all_active` | `tests/test_encoder_runner_fail_all.py`; scheduler fatal tests | tested |
| B7 | Post-forward `slice_results` errors are request-level recoverable on entry rank. | lean RFC | `EncoderScheduler.start` post-forward branch | `tests/test_encoder_scheduler_loop.py::test_loop_post_forward_slice_error_recovery` | tested |
| C1 | Add typed `runtime.resources.encoder_activation_budget_bytes`. | lean RFC, issue comment 4487693772 | `StageResourceConfig`; `resolve_stage_factory_args` | `tests/unit_test/pipeline/test_runtime_adapter.py` | tested |
| C2 | Reject untyped `encoder_activation_budget_bytes` in `factory_args` or `runtime_overrides`, including `null`. | lean RFC | `reject_untyped_encoder_activation_budget_bytes` | `tests/unit_test/pipeline/test_runtime_adapter.py` | tested |
| C3 | `total_gpu_memory_fraction` is resident/static placement budget, not encoder runtime cap. | issue comment 4487693772, lean RFC | `StagePlacementPlanner`; docs; AR memory contract logs | `tests/unit_test/pipeline/test_topology.py`; `tests/unit_test/qwen3_omni/test_sglang_ar_budget.py` | tested |
| C4 | Dynamic headroom validation includes encoder activation budget. | lean RFC | `StagePlacementPlanner._validate_dynamic_headroom` | `tests/unit_test/pipeline/test_topology.py::{test_dynamic_headroom_accepts_encoder_activation_budget,test_dynamic_headroom_rejects_encoder_activation_budget}` | tested |
| C5 | Typed configs reject explicit nonzero `encoder_mem_reserve`; old reserve remains legacy-only. | issue comment 4524089582, goal doc | `_apply_colocated_ar_memory_contract`; CLI override reject | `tests/unit_test/qwen3_omni/test_sglang_ar_budget.py`; `tests/unit_test/qwen3_omni/test_config_manager.py` | tested |
| C6 | Protected `server_args_overrides` cannot override topology, DP/encoder/language flags, AR memory knobs, cuda graph, or device. | lean RFC | `_ENCODER_PROTECTED_KEYS`; runner-managed key reject | `tests/test_encoder_server_args.py` | tested |
| C7 | SGLang runner rejects missing `nccl_port` even at `tp_size=1`. | lean RFC | `SGLangEncoderRunner.__init__` | `tests/test_encoder_server_args.py::test_runner_rejects_missing_nccl_port` | tested |
| D1 | Image/video admission uses activation-aware nonzero cost. | lean RFC | `Qwen3OmniImageEncoderAdapter.request_cost_fn` | `tests/test_encoder_adapters.py::test_image_adapter_request_cost_uses_base_hidden_not_wrapper` | tested |
| D2 | Audio admission uses nonzero, padding-aware batch cost. | lean RFC | `Qwen3OmniAudioEncoderAdapter.batch_cost_fn` | `tests/test_encoder_adapters.py::test_audio_adapter_batch_cost_accounts_for_batch_padding` | tested |
| D3 | Audio admission reads `audio_feature_lengths`; missing length is validation error. | lean RFC | `Qwen3OmniAudioEncoderAdapter.batch_cost_fn`; `_normalize_audio_request_tensors` | `tests/test_encoder_adapters.py::test_audio_adapter_cost_requires_precomputed_lengths` | tested |
| D4 | Preserve `audio_feature_lengths` and aligned `slice_results`. | goal doc | `RequestSpan.audio_feature_lengths`; audio `slice_results` | `tests/test_encoder_adapters.py::test_audio_adapter_slice_results_round_trip` | tested |
| D5 | Do not fake success with empty tensors, zero outputs, or `None` after failure. | goal doc, lean RFC | Fatal/recoverable paths emit errors; adapters only return `None` for empty/skip plans | Scheduler fatal/recoverable tests; PR E2E outputs in `encoder_tp_performance_report.md` | tested |
| E1 | Local HF vs SGLang `tp=1` precision baseline quantified and attributed. | goal doc | `docs/developer_reference/encoder_tp_parity_findings.md`; `tests/_encoder_parity_harness.py` | Current image/video/audio parity tables in parity doc | tested |
| E2 | SGLang `tp=1` vs `tp>1` precision gate with at least TP=2. | goal doc | parity harness and `tests/run_tp2_parity.sh` | Current image/video/audio TP2 comparisons in parity doc | tested |
| E3 | Precision covers image, video, and audio encoders with shape/dtype/device/length/batch/error/cosine metrics. | goal doc | `tests/_encoder_parity_harness.py --modality image|video|audio`; `tests/parity_compare.py` | Artifacts under `/data/encoder_tp_evidence_20260526`; parity doc quick-reference table | tested |
| E4 | TP precision attribution rules out partial load, submodule selection, adapter slicing, dtype/device, batch construction, and input mismatch before blaming TP numerics. | goal doc | parity doc wrapper-vs-bare, fp32 row/linear ablations; audio flatten fix; video preprocessor compatibility fix | tested for image/audio and TP plumbing; video local-vs-SGLang remains an upstream implementation gap, not TP-introduced error. |
| F1 | Upstream-main long-video/long-audio OOM or missing-admission reproduction on latest `origin/main`. | goal doc | Separate clean worktree `/data/sglang-omni-upstream-main` at `15b5e7c94ddccd0325f061e7500377ddfccd0434` | tested with caveat: upstream-main `mem_fraction_static=0.45/0.55` fails startup KV/static checks at 32768 context; installed `qwen-vl-utils==0.0.14` blocks video decode; with isolated compatible `qwen-vl-utils==0.0.11`, video128 succeeds, video196 at ctx65536 fails with KV-capacity missing-admission (`required_tokens=35704`, `kv_capacity=23257`), and audio30 succeeds. This is not an encoder activation OOM. |
| F2 | PR-branch controlled A/B: same commit, same payload/config, `tp=1` baseline fails/rejects and `tp>1` succeeds. | goal doc | `examples/qwen3_omni_encoder_tp.py`; `examples/encoder_tp_e2e_probe.py`; TP-aware adapter admission cost | tested in two-GPU colocated layout on physical GPUs 5/6: 256-frame video at 8 GiB rejects TP1 (`9542041600 > 8589934592`) and succeeds on TP2 (`7633633280 < 8589934592`, 46,608 prompt tokens, health OK). 300s no-truncation audio at 0.12 GiB rejects TP1 (`157272000 > 128849018`) and succeeds on TP2 (`117336000 < 128849018`, 3,916 prompt tokens, health OK). The earlier 196-frame/6 GiB A/B remains as backup evidence. |
| F3 | Long-video E2E success with valid output, no fake success, no `encoder_mem_reserve`, healthy server. | goal doc | Performance report | tested: TP1 and TP2 128-frame and 196-frame video succeed with valid text and health OK; the primary same-budget long-video proof is two-GPU colocated 256 frames, where TP1 rejects and TP2 succeeds. Extended TP2 long-sequence stress with a 2049-frame source records `video_max_frames=256/512/1024/2048`; the `512/1024/2048` rows safely reject before encoder forward under the 10 GiB typed activation cap. |
| F4 | Long-audio E2E success with correct admission/cost, length handling, controlled activation memory, aligned slicing. | goal doc | Performance report | tested: 30s audio succeeds on TP1 and TP2; default 300s audio clamps to `input_features=(1,128,3000)` and 406 prompt tokens; with `audio_truncation=false`, the same 300s WAV produces `input_features=(1,128,30000)` and 3,916 prompt tokens. Primary same-budget long-audio proof is two-GPU colocated: TP1 rejects at 0.12 GiB (`157272000 > 128849018`) and TP2 succeeds (`117336000 < 128849018`, health OK). |
| G1 | Prove old knobs are insufficient and typed memory split is necessary for both long-video and long-audio baselines. | goal doc | `encoder_tp_performance_report.md` | tested with caveat: upstream-main `0.45/0.55` startup failures, upstream video196 KV-capacity missing-admission, PR video196 success, PR `0.7` startup OOMs, audio30 and no-truncation audio300 cost/peak evidence, extended long-video admission rejections through 2048 frames, and same-budget TP1-vs-TP2 splits for video196/video256/audio300 show AR static/KV pressure is distinct from encoder activation budget. Exact original encoder runtime OOM is not reproduced on this host, so guarded admission is used as the reproducible OOM-path proof. |
| G2 | Report startup/load memory, pre/post model load, resident/static budget, dynamic headroom, activation budget, latency, throughput, peak memory, GPU util, TP distribution, broadcast overhead. | goal doc | `encoder_tp_performance_report.md`; `examples/encoder_tp_e2e_probe.py`; `encoder_admission` and `encoder_batch_timing` scheduler logs | tested: startup/load, admission, latency, health, TP placement, sampled global GPU peak/delta/utilization, token throughput, encoder recv/forward timing, warmed video196/audio30 TP1/TP2 throughput, process-level NVML peak attribution, primary video256/audio300 same-budget A/B, extended TP2 video `256/512/1024/2048` frame-cap outcomes, and no-truncation audio300 outcomes are reported. Broader broadcast/fanout timing remains a performance follow-up because TP2 video196 is dominated by `~11.2s` recv/coordination time. |
| G3 | Classify OOMs separately: startup KV/static, runtime activation, multimodal encoder activation/admission. | goal doc | Performance report | tested with caveat: startup KV/static, thinker KV-capacity missing-admission, and multimodal encoder admission rejection are classified, including TP1 video256/audio300 guarded failures and TP2 video `512/1024/2048` activation-budget rejections; runtime encoder activation OOM is documented as not reproduced on this host. |
| H1 | Produce PR description/test-plan draft summarizing design coverage, precision attribution, OOM reproduction, E2E results, residual risks. | goal doc | `docs/developer_reference/encoder_tp_pr_draft.md` | tested: draft updated with current GPU parity, upstream-main reproduction, primary video256/audio300 PR A/B, warmed throughput, process-local memory attribution, extended long-video stress, no-truncation audio300 evidence, and residual exact-OOM/broadcast-performance risks. |

## Residual Risks / Follow-ups

1. Upstream-main long-video missing-admission is reproduced as thinker
   KV-capacity pressure, not as encoder activation OOM.
2. Exact PR-branch runtime encoder activation OOM was not reproduced on this
   H200 host; the same-budget proof is controlled admission rejection on the
   requests that would otherwise enter the OOM path.
3. Broader broadcast/fanout optimization remains open; current TP2 video196
   correctness passes, but E2E latency is dominated by recv/coordination time.
4. Old-knob insufficiency is shown for startup/static, KV-capacity pressure,
   and guarded multimodal activation admission, but not by an unguarded runtime
   encoder activation OOM on this host.
5. Default Qwen3-Omni audio preprocessing still truncates to the 30s Whisper
   window; long-audio encoder stress now requires setting
   `audio_truncation=false`.
