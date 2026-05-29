# Encoder TP Plan B PR Draft

## Summary

Implements the Plan B encoder tensor-parallel path for Qwen3-Omni encoder
stages:

- `Stage -> EncoderScheduler -> SGLangEncoderRunner -> upstream SGLang encoder
  module`.
- Local HF encoder backend remains available as fallback and baseline.
- SGLang-backed encoder stages use isolated OS processes, single visible device
  remapping, parent-owned NCCL ports, and upstream SGLang TP initialization even
  at `tp_size=1`.
- `backend="auto"` records both requested and resolved execution backend;
  topology, preflight, NCCL port allocation, env remap, and launch decisions use
  the resolved execution backend.
- Encoder runner partial-loads only adapter-declared checkpoint prefixes and
  does not instantiate full upstream `ForConditionalGeneration` in encoder
  stages.
- EncoderScheduler implements entry-rank input drain, CPU metadata broadcast,
  device tensor broadcast, allocation-ready handshake, recoverable
  pre/post-forward request errors, fatal TP forward failure semantics, and
  fine-grained recv-path timing fields for benchmark attribution.
- Adds typed `runtime.resources.encoder_activation_budget_bytes` as the
  temporary encoder activation/admission budget, while
  `total_gpu_memory_fraction` remains resident/static placement budget.
- Adds an `audio_truncation` request override for Qwen3-Omni preprocessing so
  validation can exercise audio longer than the default Whisper 30s feature
  window.

## CPU Unit Test Evidence

- `pytest -q tests/test_encoder_tp_e2e_probe.py tests/unit_test/serve/test_openai_api.py`
  -> 17 passed.
- `timeout 180 python -m pytest -q tests/test_encoder_server_args.py tests/test_encoder_scheduler_recv.py tests/test_encoder_scheduler_loop.py tests/test_encoder_module_container.py tests/test_encoder_adapters.py tests/test_encoder_tp_e2e_probe.py tests/unit_test/pipeline/test_runtime_adapter.py tests/unit_test/qwen3_omni/test_sglang_ar_budget.py`
  -> 124 passed, 2 warnings.
- `timeout 240 python -m pytest -q tests/test_encoder_tp_launcher.py tests/test_encoder_runner_fail_all.py tests/test_parity_compare.py tests/unit_test/pipeline/test_compile.py tests/unit_test/pipeline/test_runtime_schema.py tests/unit_test/pipeline/test_runtime_adapter.py tests/unit_test/pipeline/test_topology.py tests/unit_test/pipeline/test_placement.py tests/unit_test/qwen3_omni/test_config_manager.py tests/unit_test/qwen3_omni/test_pipeline.py tests/unit_test/qwen3_omni/test_sglang_ar_budget.py`
  -> 156 passed, 8 warnings.
- `timeout 240 python -m pytest -q tests/test_encoder_*.py tests/test_parity_compare.py tests/test_video_preprocessing.py -k 'not parity_gpu'`
  -> 145 passed, 7 deselected, 4 warnings.
- `timeout 300 python -m pytest -q tests -m "not slow and not benchmark and not docs" --ignore=tests/test_model`
  -> 789 passed, 14 deselected, 4 warnings.
- `python -m py_compile tests/conftest.py tests/_encoder_parity_harness.py tests/parity_compare.py tests/test_parity_compare.py`
  -> passed.
- `timeout 60 python -m pytest -q tests/test_parity_compare.py`
  -> 6 passed.
- `timeout 60 python -m pytest -q tests/test_encoder_tp_e2e_probe.py`
  -> 7 passed.
- `timeout 120 python -m pytest -q tests/test_encoder_scheduler_loop.py tests/test_encoder_scheduler_recv.py`
  -> 20 passed.
- `pytest -q tests/test_encoder_tp_launcher.py tests/test_encoder_tp_e2e_probe.py tests/unit_test/serve/test_openai_api.py tests/unit_test/pipeline/test_topology.py`
  -> 67 passed, 2 warnings.
- `timeout 120 python -m pytest -q tests/test_encoder_adapters.py tests/test_encoder_tp_launcher.py tests/test_encoder_scheduler_loop.py tests/test_encoder_scheduler_recv.py tests/test_encoder_tp_e2e_probe.py`
  -> 82 passed, 2 warnings.

## GPU Validation Status

CUDA is accessible on this host, and current PR-branch parity/E2E evidence was
collected under `/data/encoder_tp_evidence_20260526`. The strongest current
H100 evidence is intentionally narrow:

- Memory: encoder TP lowers the per-rank activation admission ceiling. In the
  encoder-only H100 video sweep, TP2 also lowered the max-rank runtime peak
  slope, but both TP1 and TP2 completed through `2048` frame cap, so no
  TP2-only max-length win was reached. Do not claim lower total cluster memory
  or lower E2E memory.
- E2E long input: audio1000 with `audio_truncation=false` and separate-layout
  video512 are the clean same-payload/same-budget splits where TP1 admission
  fails and TP2 completes.
- Video: the 2026-05-29 separate-layout H100 run gives the primary video E2E
  success: default-pixel video512 rejects on image TP1 at the same `14.2 GiB`
  budget and completes on image TP2. The older default-pixel video256/video512
  colocated runs remain encoder-boundary evidence only because thinker KV or
  colocated memory becomes the next bottleneck.
- Latency: video128 and audio30 have warmed main / PR TP1 / PR TP2 A/Bs.
  PR TP1/TP2 fine timing shows video TP2 is slower because of measured
  metadata/admission handshake fields, not forward compute or tensor payload
  broadcast. Do not claim TP2 is faster.
- Accuracy: tensor parity scope is closed as below. CI had no benchmark
  artifact for PR #423 head `53328e698952ec14557702a3f00e676f9ded765a`, so the
  formal Video-AMME CI-50 benchmark was run locally. At concurrency 1, main
  scored `36/50` (`72%`), PR TP1 scored `34/50` (`68%`), and PR TP2 scored
  `36/50` (`72%`), all with `0` failed requests and above the repository
  `66%` CI threshold. Full task-level quality remains open.

The H100 evidence now has two layout lanes. The colocated 2-GPU lane is used
for admission/runtime-boundary, latency attribution, and the latest colocated
memory-slope check. The separate-layout video lane uses dedicated encoder and
thinker GPUs and is used for the video512 E2E success claim. Older H200
artifacts are retained below as supporting evidence, but the PR wording should
prefer the H100 tables in `encoder_tp_performance_report.md`.

Memory evidence is split into encoder-only runtime capacity/slope and admission
math. The strongest current memory artifact is
`/data/encoder_tp_evidence_20260526/h100_encoder_only_video_memory_20260529`.
It uses fresh subprocesses per point, no thinker/talker/generation, no
`video_max_pixels` override, forced `encoder_max_batch_size=1`, no activation
budget, and a measurement-only GPU guard. TP1 used one H100 (`GPU6`); TP2 used
two H100s (`GPU6,7`).

Encoder-only capacity result: TP1 and TP2 both completed image/video encoder
forward through the tested `2048` frame cap. This means this run does **not**
prove `TP2 max frame cap > TP1`; the encoder boundary was not reached.

Runtime NVML peak slope on the same encoder-only path:

| Fit target | TP1 one rank | TP2 max rank | TP2 sum, reference only |
| --- | ---: | ---: | ---: |
| `peak(frame_cap) = A + b * frame_cap` | `A=10569 MiB`, `b=8.611 MiB/frame`, `R^2=0.905` | `A=9365 MiB`, `b=6.638 MiB/frame`, `R^2=0.907` | `A=18708 MiB`, `b=13.277 MiB/frame`, `R^2=0.907` |
| `peak(k_premerge_tokens) = A + b * L` | `A=2069 MiB`, `b=47.061 MiB / 1k tokens`, `R^2=0.99996` | `A=2832 MiB`, `b=36.222 MiB / 1k tokens`, `R^2=0.99980` | `A=5641 MiB`, `b=72.444 MiB / 1k tokens`, `R^2=0.99980` |

For this encoder-only video path, TP2 max-rank peak has lower measured length
slope than TP1, with a `70.37k` pre-merge-token crossover below the smallest
measured point (`147.46k`). Do not generalize that to total cluster memory,
audio, colocated stages, or E2E memory. TP2 sum peak is higher.

Measured points:

| Frame cap | Pre-merge visual tokens | TP1 peak | TP2 max-rank peak | TP2 sum peak |
| ---: | ---: | ---: | ---: | ---: |
| 128 | `147456` | `8910 MiB` | `8164 MiB` | `16306 MiB` |
| 256 | `294912` | `15590 MiB` | `13204 MiB` | `26386 MiB` |
| 512 | `319488` | `16730 MiB` | `14084 MiB` | `28146 MiB` |
| 768 | `307200` | `16150 MiB` | `13644 MiB` | `27266 MiB` |
| 1024 | `368640` | `18992 MiB` | `15844 MiB` | `31666 MiB` |
| 1536 | `442368` | `22412 MiB` | `18486 MiB` | `36950 MiB` |
| 2048 | `589824` | `29210 MiB` | `23764 MiB` | `47506 MiB` |

TP4 encoder-only follow-up:

Artifact:
`/data/encoder_tp_evidence_20260526/h100_encoder_only_video_tp4_slope_20260529`.
Command shape: `python examples/encoder_tp_encoder_only_video_probe.py
--tp-specs '1:1;4:1,3,4,5' --frame-caps 128,256,512,768,1024`, no
`video_max_pixels` override, no thinker/talker/audio/generation, no admission
budget, fresh subprocesses per TP/frame point, `encoder_max_batch_size=1`.

| Fit target | TP1 | TP4 | PR wording |
| --- | ---: | ---: | --- |
| encoder-only max-rank peak vs pre-merge tokens | `A=2187 MiB`, `b=45.517 MiB / 1k tokens`, `R^2=0.99997` | `A=2448 MiB`, `b=32.254 MiB / 1k tokens`, `R^2=0.99999` | `b4 < b1`; TP4 max-rank runtime slope is lower on this encoder-only path. |
| encoder-only max-rank peak vs frame cap | `A=10782 MiB`, `b=8.356 MiB/frame`, `R^2=0.655` | `A=8546 MiB`, `b=5.908 MiB/frame`, `R^2=0.652` | Same direction, but token fit is preferred because preprocessing changes spatial grid. |
| encoder-only summed-rank peak vs pre-merge tokens | `A=2187 MiB`, `b=45.517 MiB / 1k tokens`, `R^2=0.99997` | `A=9823 MiB`, `b=128.822 MiB / 1k tokens`, `R^2=0.999999` | Reference only; TP4 sum peak is higher, so no total-memory claim. |

Measured TP4 points:

| Frame cap | Pre-merge visual tokens | TP1 max-rank peak | TP4 rank peaks | TP4 max-rank peak | TP4 sum peak |
| ---: | ---: | ---: | --- | ---: | ---: |
| 128 | `147456` | `8910 MiB` | `7206 / 7206 / 7206 / 7206 MiB` | `7206 MiB` | `28824 MiB` |
| 256 | `294912` | `15590 MiB` | `11966 / 11946 / 11946 / 11946 MiB` | `11966 MiB` | `47804 MiB` |
| 512 | `319488` | `16730 MiB` | `12746 / 12746 / 12746 / 12746 MiB` | `12746 MiB` | `50984 MiB` |
| 768 | `307200` | `16150 MiB` | `12346 / 12346 / 12346 / 12346 MiB` | `12346 MiB` | `49384 MiB` |
| 1024 | `368640` | `18992 MiB` | `14346 / 14326 / 14326 / 14326 MiB` | `14346 MiB` | `57324 MiB` |

This does not answer colocated TP4 or E2E TP4 memory. The current
`colocated-2gpu` launcher only supports TP1/TP2, and the existing separate E2E
recipe needs more visible GPUs for `image_tp4 + audio_tp1 + thinker + talker`
than were available in this follow-up.

Colocated-2GPU slope check:

Artifact:
`/data/encoder_tp_evidence_20260526/h100_colocated_video_memory_slope_20260529_budget10`.
Command shape: `CUDA_VISIBLE_DEVICES=1,3`,
`examples/qwen3_omni_encoder_tp.py --layout colocated-2gpu`, fresh server per
TP/frame point, `--image-encoder-activation-budget-gib 10`,
`--audio-encoder-activation-budget-gib 1`, `--thinker-mem-fraction-static 0.78`,
`--talker-mem-fraction-static 0.12`, `--thinker-max-seq-len 131072`,
`--encoder-max-batch-size 1`, `video_fps=30`, no `video_max_pixels` override,
frame caps `64,96,128,160`.

| Fit target | TP1 | TP2 | PR wording |
| --- | ---: | ---: | --- |
| colocated image max-rank peak vs frame cap | `A=2096 MiB`, `b=52.625 MiB/frame`, `R^2=0.9994` | `A=2736 MiB`, `b=54.938 MiB/frame`, `R^2=0.9976` | `b2 >= b1`; do not claim TP2 lowers colocated runtime image peak slope. |
| colocated selected whole-GPU max vs frame cap | `A=69085 MiB`, `b=52.650 MiB/frame`, `R^2=0.9994` | `A=70433 MiB`, `b=40.838 MiB/frame`, `R^2=0.9961` | Reference only; dominated by resident thinker/talker placement and encoder load splitting. |
| colocated summed selected-GPU peaks vs frame cap | `A=79228 MiB`, `b=52.650 MiB/frame`, `R^2=0.9994` | `A=85841 MiB`, `b=95.775 MiB/frame`, `R^2=0.9979` | Reference only; no lower total-memory claim. |

Measured colocated points:

| Frame cap | Prompt tokens | TP1 image peak | TP2 image rank peaks | TP2 max-rank peak | TP1 selected-GPU max | TP2 selected-GPU max |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 64 | `18448` | `5450 MiB` | `5424 / 6164 MiB` | `6164 MiB` | `72441 MiB` | `73061 MiB` |
| 96 | `27664` | `7130 MiB` | `6624 / 8084 MiB` | `8084 MiB` | `74121 MiB` | `74261 MiB` |
| 128 | `36880` | `8910 MiB` | `8164 / 9884 MiB` | `9884 MiB` | `75903 MiB` | `75803 MiB` |
| 160 | `46096` | `10470 MiB` | `9264 / 11424 MiB` | `11424 MiB` | `77463 MiB` | `76903 MiB` |

Admission remains a separate per-rank guard, not runtime peak:

```text
cost(tp) = multiplier * (replicated_bytes + sharded_bytes / tp)
```

For Qwen3-Omni image/video and audio, `multiplier=5`. Measured admission-cost
splits are image/video `60.0% replicated / 40.0% sharded` and audio `49.2% /
50.8%`. The PR should claim the lower per-rank activation admission ceiling,
plus the encoder-only video runtime-slope results above, while explicitly
noting the colocated slope rerun does not show lower TP2 image max-rank runtime
slope. It should not claim lower total cluster memory.

Runtime peak terms include:

```text
resident + replicated_runtime + sharded_runtime/tp + TP fan-out transient
  + rank0/follower staging + allocator cache/margin
  + dynamic batching + colocated stage headroom
```

The stage-to-stage payload path is CPU SHM. TP2 extra runtime cost comes after
SHM read: entry-rank H2D lift, metadata broadcast, follower allocation,
allocation handshake, tensor broadcast, rank0/follower staging, and allocator
behavior. Do not describe it as direct GPU stage-to-stage relay.

Current c4 capacity/perf classification:

| Run | Audio encoder admission batch sizes | Image encoder admission batch sizes | Interpretation |
| --- | --- | --- | --- |
| PR TP1 c1 | all `batch_size=1` | all `batch_size=1` | single-request admission path |
| PR TP1 c4 | `50x batch_size=1` | `48x batch_size=1`, `1x batch_size=2` | high concurrency mostly remains single-item batches |
| PR TP2 c1 | all `batch_size=1` | all `batch_size=1` | single-request admission path |
| PR TP2 c4 before batch cap | `25x batch_size=1` | `2x batch_size=1`, `9x batch_size=2`, `1x batch_size=3` | TP2 recv/admission/handshake latency lets requests accumulate into larger image batches |
| PR TP2 c4 after `encoder_max_batch_size=1` default | `50x batch_size=1` | `50x batch_size=1` | temporary cap prevents c4-style dynamic encoder batch accumulation |
| PR TP2 c4 after whole-GPU guard, no encoder batch cap | `50x batch_size=1` | `48x batch_size=1`, `1x batch_size=2`; `47` image candidate batches deferred by `gpu_guard` | admission guard admits larger batches only when projected whole-GPU headroom fits |

The PR TP2 c4 failure should be described as an encoder OOM /
capacity-perf boundary due to dynamic batch accumulation and colocated GPU0
headroom, not as a task-quality regression. The observed forward OOM happened on
image_encoder rank0 when GPU0 had about `222 MiB` free and rank0 requested
`248 MiB`. GPU0 was colocated with thinker (`~65.29 GiB`), image rank0
(`~10.29 GiB`), and audio rank0 (`~3.35 GiB`). The follow-up implementation
now makes encoder batch size an admission result and adds a whole-GPU guard
with cross-process in-flight reservations. The explicit
`encoder_max_batch_size` control remains available as a conservative fallback,
but TP2 `colocated-2gpu` no longer needs to default it to `1` when guard
telemetry is available.

Current H100 E2E/long-input table:

| Payload | Budget | TP1 result | TP2 result | PR wording |
| --- | ---: | --- | --- | --- |
| audio1000, `audio_truncation=false` | audio `0.43 GiB` | admission fail: `524240000 > 461708984`, health OK | audio TP2 E2E success, HTTP 200, prompt `13018`, total `13034`, latency `3.075s`, health OK | TP2 enables this longer audio request under the same per-rank budget. |
| video512, default pixels, separate layout | image `14.2 GiB` | admission fail: `16357785600 > 15247133900`, HTTP 500, health OK before/after | image TP2 E2E success, HTTP 200, prompt `79888`, total `79904`, latency `82.047s`, output `A man is drawing a guitar on a tablet with a stylus. He is`, health OK | TP2 enables this longer video request under the same per-rank budget when encoder TP is separated from thinker headroom. |
| video256, default pixels | image `12 GiB` | admission fail: `15099494400 > 12884901888`, health OK | TP2 admits and image forward completes at thinker `0.80`, then thinker KV rejects `required_tokens=73759`, `kv_capacity=65291`; raising thinker to `0.81` causes clean-GPU image rank0 OOM | TP2 pushes encoder boundary; colocated thinker KV/memory becomes next bottleneck. Not E2E success. |
| video512, default pixels, colocated layout | image `14.2 GiB` | TP1 cost measured in the separate-layout rerun above | TP2 admits and image forward completes, then thinker KV rejects `required_tokens=79903`, `kv_capacity=47892` | TP2 pushes encoder boundary; colocated thinker KV becomes next bottleneck. Not E2E success. |
| video512 + audio3000 | image `14.2 GiB`, audio `1.2 GiB` | no TP1 pair in final H100 pass | audio admits/completes; image admits then rank0 OOMs while audio rank0 holds `8.50 GiB` | Negative multimodal boundary case. |

Separate-layout video512 artifacts:
`/data/encoder_tp_evidence_20260526/h100_5gpu_separate_video_20260529`.
TP2 used:

```bash
CUDA_VISIBLE_DEVICES=1,3,5,6,7 \
SGLANG_OMNI_ENCODER_TIMING_DETAIL=1 \
SGLANG_OMNI_ENCODER_MEMORY_DETAIL=1 \
python examples/qwen3_omni_encoder_tp.py \
  --layout separate --image-tp 2 --audio-tp 1 \
  --image-encoder-activation-budget-gib 14.2 \
  --audio-encoder-activation-budget-gib 1 \
  --encoder-total-gpu-memory-fraction 0.01 \
  --thinker-mem-fraction-static 0.90 \
  --talker-mem-fraction-static 0.12 \
  --thinker-max-seq-len 131072 --port 8150
```

TP1 used:

```bash
CUDA_VISIBLE_DEVICES=1,3,6,7 \
SGLANG_OMNI_ENCODER_TIMING_DETAIL=1 \
SGLANG_OMNI_ENCODER_MEMORY_DETAIL=1 \
python examples/qwen3_omni_encoder_tp.py \
  --layout separate --image-tp 1 --audio-tp 1 \
  --image-encoder-activation-budget-gib 14.2 \
  --audio-encoder-activation-budget-gib 1 \
  --encoder-total-gpu-memory-fraction 0.01 \
  --thinker-mem-fraction-static 0.90 \
  --talker-mem-fraction-static 0.12 \
  --thinker-max-seq-len 131072 --port 8151
```

GPU5 had hidden memory by the TP1 retry, so the TP1 baseline used the clean
four-GPU subset needed by `--layout separate`. The request was
`draw_loop_2048frames.mp4`, `video_fps=30`, `video_max_frames=512`, prompt
`Briefly describe the video.`, `max_tokens=16`, and no `video_max_pixels`
override. Raw logs:
`server-tp2-separate-video.log` and
`server-tp1-separate-video-retry-reordered.log`.

Current H100 latency table:

Latency commands used H100 GPUs 6/7 from a `4 MiB` prelaunch state,
`qwen-vl-utils==0.0.11`, one warmup plus three measured repeats, and the same
checkpoint/prompt/media/decoding config. Main used
`run_qwen3_omni_speech_server.py --thinker-max-seq-len 32768
--thinker-mem-fraction-static 0.80 --talker-mem-fraction-static 0.12 --port
8140`; PR TP1 used `examples/qwen3_omni_encoder_tp.py --layout colocated-2gpu
--image-tp 1 --audio-tp 1 --image-encoder-activation-budget-gib 10
--audio-encoder-activation-budget-gib 1 --encoder-total-gpu-memory-fraction
0.01 --thinker-mem-fraction-static 0.80 --talker-mem-fraction-static 0.12
--thinker-max-seq-len 32768 --port 8141`; PR TP2 used the same PR command with
`--image-tp 2 --audio-tp 2 --port 8142`.

Profile / timing links:

- Parsed profile summary:
  [`latency_attribution_analysis.json`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/latency_attribution_analysis.json)
- Raw PR TP1 timing log:
  [`server-pr-tp1-video128-audio30.log`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/server-pr-tp1-video128-audio30.log)
- Raw PR TP2 timing log:
  [`server-pr-tp2-video128-audio30.log`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/server-pr-tp2-video128-audio30.log)
- Main server log:
  [`server-main-qv011-video128-audio30-mem080.log`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/server-main-qv011-video128-audio30-mem080.log)
  (E2E only; main does not emit the PR's fine encoder timing fields).

| Workload | main E2E mean / p95 | PR TP1 E2E mean / p95 | PR TP2 E2E mean / p95 | TP2 delta vs TP1 | Attribution |
| --- | ---: | ---: | ---: | ---: | --- |
| video128, `video_fps=16`, `video_max_frames=128`, `video_max_pixels=401408` | `7.769s / 7.988s` | `8.239s / 8.293s` | `13.704s / 13.891s` | `+5464 ms` | Image encoder critical path is `+5338 ms`; recv path is `+5400 ms`; forward is `-64 ms`. |
| audio30 | `0.303s / 0.310s` | `0.289s / 0.295s` | `0.321s / 0.328s` | `+31.8 ms` | Audio encoder critical path is `+37.6 ms`; recv path is `+28.8 ms`; forward is `+8.1 ms`. |

| Fine timing, warmup excluded | recv | inbox/admission | metadata broadcast | allocation handshake | tensor broadcast | forward | encoder total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| video128 PR TP1 image rank0 | `73.206 ms` | `73.155 ms` | `0.000 ms` | `0.000 ms` | `0.000 ms` | `234.025 ms` | `307.471 ms` |
| video128 PR TP2 image critical path | `5473.599 ms` | `99.935 ms` | `2671.053 ms` | `2701.687 ms` | `0.259 ms` | `170.406 ms` | `5645.361 ms` |
| audio30 PR TP1 audio rank0 | `92.354 ms` | `92.307 ms` | `0.000 ms` | `0.000 ms` | `0.000 ms` | `9.848 ms` | `102.417 ms` |
| audio30 PR TP2 audio critical path | `121.155 ms` | `0.000 ms` | `119.979 ms` | `0.547 ms` | `0.173 ms` | `17.919 ms` | `140.009 ms` |

For video128, the TP2 latency regression is quantitatively attributable to the
measured recv/admission path. The image encoder critical path explains nearly
all of the `+5464 ms` E2E delta. Within that encoder delta, TP2 forward is
`64 ms` lower than TP1; the increase is `+5400 ms` recv, specifically rank0
`metadata_broadcast_ms=2671 ms` and `allocation_handshake_ms=2702 ms`, plus
rank1 `metadata_broadcast_ms=5471 ms`. `tensor_broadcast_ms=0.259 ms`, so the
draft should not describe the slowdown as tensor payload transfer or forward
compute. For audio30, TP2 is close to main/TP1; the small delta is mostly
recv-path coordination with a smaller forward increase. The PR-only audio300
no-truncation A/B has the same pattern: TP2 E2E is `0.596s` vs TP1 `0.449s`,
with encoder total `281 ms` vs `142 ms`; recv is `+133 ms` while forward is
only `+5 ms`, dominated by metadata broadcast and allocation handshake.

Current quality table:

Quality/profiling links:

- Formal Video-AMME rerun:
  [`videoamme_ci50_benchmark_summary.json`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/quality_videoamme_ci50_rerun_20260529_002115/videoamme_ci50_benchmark_summary.json)
- Tensor parity attribution:
  [`encoder_tp_parity_findings.md`](/data/sglang-omni/docs/developer_reference/encoder_tp_parity_findings.md)
- c4 guard rerun:
  [`videoamme_ci50_post_gpuguard_summary.json`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/quality_videoamme_ci50_post_gpuguard_20260529_010354/videoamme_ci50_post_gpuguard_summary.json)

Accuracy attribution should stay scoped:

- Wrapper vs bare SGLang is bit-equal, so the wrapper is not the tensor-drift
  source.
- HF-vs-SGLang visual drift is an upstream implementation gap, not introduced
  by the PR.
- SGLang TP1-vs-TP2 image/video drift is fp16 TP reduction order plus nonlinear
  visual-stack amplification; audio TP1-vs-TP2 is strict-allclose.
- Task-level quality is supported by one formal Video-AMME CI-50 lane. It is
  not a claim that the full task-level suite is closed.

| Scope | main | PR TP1 | PR TP2 | Status |
| --- | ---: | ---: | ---: | --- |
| Formal Video-AMME CI-50, c1 | `36/50` (`72%`) | `34/50` (`68%`) | `36/50` (`72%`) | All pass the `66%` Video-AMME CI threshold with `0` failed requests. TP2 matches main; TP1 is `-4 pp` vs main but above threshold. Rerun artifact: `/data/encoder_tp_evidence_20260526/h100_gpu6_7/quality_videoamme_ci50_rerun_20260529_002115/videoamme_ci50_benchmark_summary.json`. |
| Video-AMME CI-50, c4 before `encoder_max_batch_size` cap | `36/50` (`72%`), `0` failed | `33/50` (`66%`), `0` failed | `14/50` (`28%`), `29` failed | Historical capacity/perf failure: TP2 c4 image encoder rank0 OOMed during visual forward after `21/50` completed. |
| Video-AMME CI-50, c4 after TP2 colocated `encoder_max_batch_size=1` default | prior `36/50` (`72%`), `0` failed | prior `33/50` (`66%`), `0` failed | `36/50` (`72%`), `0` failed | Passes threshold. TP2 c4 post-cap used `50x batch_size=1` for image/audio, mean latency `15.941s`, p95 `17.645s`, QPS `0.244`. Artifact: `/data/encoder_tp_evidence_20260526/h100_gpu6_7/quality_videoamme_ci50_post_batchcap_20260529_004439/videoamme_ci50_post_batchcap_summary.json`. |
| Video-AMME CI-50, c4 after whole-GPU guard, no encoder batch cap | prior `36/50` (`72%`), `0` failed | prior `33/50` (`66%`), `0` failed | `36/50` (`72%`), `0` failed | Passes threshold. TP2 c4 post-guard used scheduler `max_batch_size=32` without explicit `--encoder-max-batch-size`; image admitted `48x batch_size=1` and `1x batch_size=2`, with `47` image candidates deferred by `gpu_guard`; audio admitted `50x batch_size=1`. Mean latency `17.147s`, p95 `19.431s`, QPS `0.227`. Artifact: `/data/encoder_tp_evidence_20260526/h100_gpu6_7/quality_videoamme_ci50_post_gpuguard_20260529_010354/videoamme_ci50_post_gpuguard_summary.json`. |
| Smoke quality subset: `encoder_tp_smoke_quality_video128_audio30` | `6/6` (`100%`) | `6/6` (`100%`) | `6/6` (`100%`) | Supporting smoke evidence only. Artifact: `/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/smoke_quality_benchmark.json`. |
| Full task-level suite | not run | not run | not run | Open; MMMU, Video-MME, MMSU, and larger/more concurrent quality sweeps were not run. No CI artifact URL/run id was available for PR #423 head `53328e698952ec14557702a3f00e676f9ded765a`. |
| Wrapper vs bare SGLang tensors | n/a | bit-equal | bit-equal | Closed by tensor parity harness. |
| Image/video TP tensors | n/a | reference | mean cosine about `0.99998` | Drift characterized as fp16 TP reduction order plus nonlinear amplification. |
| Audio TP tensors | n/a | reference | strict-allclose | Closed for tested audio tensors. |

CI artifact lookup checked `gh pr view 423 --repo sgl-project/sglang-omni` and
found an empty `statusCheckRollup`; `gh run list` for branch
`encoder-tp-plan-b-phase0` returned no runs in both `sgl-project/sglang-omni`
and `ischencheng/sglang-omni`. The local formal benchmark used checkpoint
`/data/qwen3omni`, dataset `zhaochenyang20/Video_AMME_ci`, `max_samples=50`,
`max_tokens=256`, `temperature=0.0`, `video_fps=2`,
`video_max_frames=128`, `video_max_pixels=401408`, text output only, and the
isolated compatible `qwen-vl-utils` path `/tmp/qwen_vl_utils_0011`. Treat this
as one benchmark lane, not full task-level coverage. The 2026-05-29 c1 rerun
reproduced the same scores with mean latencies `2.371s` (main), `2.436s`
(PR TP1), and `6.282s` (PR TP2); these speed numbers are benchmark metadata,
not a TP2 speed claim. The later c4 post-guard rerun used the same PR TP2
colocated command shape without an explicit `--encoder-max-batch-size`; the
schedulers kept `max_batch_size=32`, the whole-GPU guard deferred unsafe image
candidates, and the run completed all `50/50` requests.

- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. timeout 600 python -m pytest -q tests/test_encoder_tp_parity_gpu.py -m slow`
  -> 7 passed, 4 warnings.
- Image/video/audio parity:
  - image TP1 vs TP2: max abs `0.2080`, mean abs `0.0008`, mean cosine `0.999982`;
  - video TP1 vs TP2: max abs `0.2383`, mean abs `0.0010`, mean cosine `0.999980`;
  - audio TP1 vs TP2: strict-allclose, max abs `0.0004`, mean cosine `0.999999`.
- Accuracy scope:
  - wrapper vs bare SGLang encoder is bit-equal;
  - HF-vs-SGLang drift is an upstream implementation gap, not introduced by
    the wrapper;
  - image/video TP1-vs-TP2 drift is expected from fp16 TP reduction order plus
    nonlinear amplification through the visual stack;
  - audio TP1-vs-TP2 is strict-allclose after valid-token flattening;
  - task-level Video-AMME CI-50 c1 passes for main, PR TP1, and PR TP2; TP2
    matches main, while TP1 is `-4 pp` but above the CI threshold. Full
    downstream benchmark quality remains open.
- PR E2E with typed AR `mem_fraction_static=0.45`, encoder activation budget
  `10 GiB`, no `encoder_mem_reserve`:
  - TP1 long video128: success, 18,944 prompt tokens, 13.70s, health OK;
  - TP2 long video128: success, 18,944 prompt tokens, 21.84s, health OK;
  - TP1 stress video196: success, 35,688 prompt tokens, 22.85s, health OK;
    sampled image-GPU delta `+7344 MiB`;
  - TP2 stress video196: success, 35,688 prompt tokens, 35.44s, health OK;
    sampled image-rank deltas `+6624 MiB` and `+8300 MiB`;
  - TP1 long audio30: success, 406 prompt tokens, 1.24s, health OK;
    sampled audio-GPU delta `+376 MiB`;
  - TP2 long audio30: success, 406 prompt tokens, 1.57s, health OK;
    sampled audio-rank deltas `+862 MiB` and `+862 MiB`.
- Controlled PR same-budget A/B showing TP lowers the per-rank activation
  ceiling enough to admit longer inputs:
  - 256-frame video, 8 GiB budget:
    TP1 rejects before encoder forward with `9542041600 > 8589934592`,
    health OK; TP2 succeeds with 46,608 prompt tokens in 55.61s,
    `7633633280 < 8589934592`, health OK;
  - 300s WAV with `audio_truncation=false`, 0.12 GiB budget:
    TP1 rejects before encoder forward with `157272000 > 128849018`,
    health OK; TP2 succeeds with 3,916 prompt tokens in 1.94s,
    `117336000 < 128849018`, health OK;
  - artifacts:
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_video256_colocated2gpu_budget8_ctx65536`,
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_video256_colocated2gpu_budget8_ctx65536`,
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_audio300_notrunc_colocated2gpu_budget012_ctx65536`,
    and
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio300_notrunc_colocated2gpu_budget012_ctx65536`.
- Extended TP2 long-sequence stress on a real 2049-frame, 68.35s looped video
  with `video_max_frames=256,512,1024,2048`, `video_fps=30`,
  `video_max_pixels=401408`, `thinker_max_seq_len=524288`, and the same
  10 GiB typed encoder activation budget. This is not a 512/1024/2048
  generation-success claim under that cap; the `512+` rows are safe admission
  rejections:
  - 256 frames: success, 46,608 prompt tokens, 57.74s, health OK;
    `batch_cost=7633633280 < 10737418240`; image-rank process deltas
    `+8472 MiB` and `+10660 MiB`;
  - 512 frames: HTTP 500 admission rejection before encoder forward,
    `13086228480 > 10737418240`, health OK;
  - 1024 frames: HTTP 500 admission rejection before encoder forward,
    `15099494400 > 10737418240`, health OK;
  - 2048 frames: HTTP 500 admission rejection before encoder forward,
    `24159191040 > 10737418240`, health OK;
  - artifacts:
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video256_budget10_ctx524288`,
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video512_budget10_ctx524288`,
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video1024_budget10_ctx524288`,
    and
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video2048_budget10_ctx524288`.
- Extended audio check on a real 300s looped WAV:
  - default API path succeeds but clamps to the same 30s workload as audio30:
    `input_ids=(1, 406)`, `input_features=(1, 128, 3000)`, mask sum `3000`;
  - with `audio_truncation=false`, the same 300s file produces 3916 prompt
    tokens and a true `input_features=(1, 128, 30000)` encoder workload;
  - TP2 no-truncation audio300 succeeds in 2.99s, health OK,
    `batch_cost=117336000 < 10737418240`, audio-rank process deltas
    `+3482 MiB` and `+3514 MiB`;
  - artifacts:
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio300_budget10_ctx524288`
    and
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio300_notrunc_budget10_ctx65536`.
- Clean H100 two-GPU rerun on GPUs 6/7, using `--layout colocated-2gpu`,
  image/audio TP2, no low-pixel override, `audio_truncation=false`,
  image budget `14.2 GiB`, audio budget `1.2 GiB`, thinker
  `mem_fraction_static=0.78`, and `thinker_max_seq_len=131072`:
  - GPUs 6/7 started at `4 MiB` used and no compute apps.
  - 1024-frame video: image admission passed
    (`15099494400 < 15247133900`) but image rank0 OOMed during forward
    trying to allocate another `810 MiB`; GPU peaks `78753/32552 MiB`;
    health false.
  - 512-frame video: image admission and encoder forward completed
    (`13086228480 < 15247133900`), then thinker scheduling rejected
    `input_tokens=79887`, `required_tokens=79903`, `kv_capacity=47892`;
    GPU peaks `80835/30212 MiB`; health OK.
  - 3000s audio: audio admission and generation succeeded with
    `audio_truncation=false`, `39017` prompt tokens, `9.389s`,
    `1173360000 < 1288490188`; GPU peaks `75643/20952 MiB`; health OK.
  - 512-frame video + 3000s audio: audio admitted and completed, then image
    admitted but rank0 OOMed during forward trying to allocate another
    `1.03 GiB` while audio rank0 held `8.50 GiB`; GPU peaks
    `80193/32754 MiB`; health false.
  - This rerun does not support a blanket "TP2 is faster" claim or a claim
    that high TP2 peak memory was caused by another user. On clean H100s,
    colocated thinker resident memory plus default-pixel video activations are
    enough to hit the 80GB ceiling.
  - artifacts:
    `/data/encoder_tp_evidence_20260526/h100_gpu6_7/tp2_video1024_default_pixels/h100_gpu6_7_tp2_video1024_default_pixels`,
    `/data/encoder_tp_evidence_20260526/h100_gpu6_7/tp2_video512_default_pixels/h100_gpu6_7_tp2_video512_default_pixels`,
    `/data/encoder_tp_evidence_20260526/h100_gpu6_7/tp2_audio3000_no_trunc/h100_gpu6_7_tp2_audio3000_no_trunc`,
    and
    `/data/encoder_tp_evidence_20260526/h100_gpu6_7/tp2_video512_audio3000_default_pixels_no_trunc/h100_gpu6_7_tp2_video512_audio3000_default_pixels_no_trunc`.
- Existing encoder timing logs include coarse `encoder_batch_timing`; the
  current branch now adds fine-grained recv-path fields, but the warmed A/B
  artifacts below predate that split:
  - TP1 video196 image timing: `recv_ms=152.010`, `forward_ms=2270.759`,
    `total_ms=2423.358`;
  - TP2 video196 image timing: rank0/rank1 `recv_ms≈12405`,
    `forward_ms≈2225`, `total_ms≈14633`;
  - TP1/TP2 audio30 timing: `total_ms≈691` vs `≈1150`.
- Warmed video196 success probes with one warmup and three measured requests:
  - TP1: mean latency `16.89s`, p50 `16.94s`, `2114.7` total tok/s;
    stdev `0.26s`, p90 `17.08s`, p95 `17.10s`;
    process-local image GPU7 delta `+8830 MiB`, thinker GPU5 delta
    `+1044 MiB`;
  - TP2: mean latency `27.91s`, p50 `27.87s`, `1279.4` total tok/s;
    stdev `0.28s`, p90 `28.13s`, p95 `28.17s`;
    process-local image rank deltas `+9138 MiB` and `+9976 MiB`, thinker GPU5
    delta `+1044 MiB`;
  - TP2 post-warmup encoder forward is lower (`~290 ms` per rank vs
    `~0.6-0.8s` TP1), but E2E latency is slower because video196 TP2's coarse
    pre-forward recv/fan-out bucket is `~11.2s`. Treat this older video196 row
    as coarse historical evidence; use the H100 video128 fine-timing table above
    for quantitative attribution.
  - artifacts:
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_video196_budget10_ctx65536_warmed_process`
    and
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_video196_budget10_ctx65536_warmed_process`.
- Warmed audio30 success probes with one warmup and three measured requests:
  - TP1: mean latency `0.745s`, stdev `0.010s`, p50 `0.740s`, p90 `0.754s`,
    p95 `0.755s`, `586.0` total tok/s; process-local audio GPU6 delta
    `+490 MiB`;
  - TP2: mean latency `0.737s`, stdev `0.014s`, p50 `0.732s`, p90 `0.749s`,
    p95 `0.752s`, `586.0` total tok/s; process-local audio rank deltas
    `+976 MiB` and `+976 MiB`;
  - TP1 audio admission cost is `15727200`; TP2 current per-rank audio cost is
    `11733600`.
  - artifacts:
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_audio30_budget10_ctx65536_warmed_process`
    and
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio30_budget10_ctx65536_warmed_process`.
- Controlled PR same-budget activation-admission A/B with video196:
  - TP1 at 6 GiB rejects before encoder forward:
    `encoder request cost 7305625600 exceeds max_single_request_cost=6442450944`;
    health before/after OK;
  - TP2 at 6 GiB admits and succeeds:
    `batch_cost=5844500480 max_batch_cost=6442450944`, 35,688 prompt tokens,
    33.22s, health before/after OK;
  - artifacts:
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_video196_budget6_ctx65536`
    and
    `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_video196_budget6_ctx65536`.
- PR default AR `mem_fraction_static=0.7` startup attempts OOMed in thinker KV
  allocation under hidden external GPU memory pressure on selected GPUs; 0.45
  starts cleanly.
- Clean upstream-main at `15b5e7c94ddccd0325f061e7500377ddfccd0434`:
  - `mem_fraction_static=0.45/0.55`, `thinker_max_seq_len=32768`: startup
    fails with SGLang `Not enough memory. Please try to increase
    --mem-fraction-static`;
  - `mem_fraction_static=0.7`: server starts, long audio30 succeeds in 2.81s;
  - installed `qwen-vl-utils==0.0.14` makes long video128 return HTTP 500
    before encoder execution because both `torchcodec` and `torchvision` video
    readers return too many values for clean-main `load_video_path` unpacking;
  - with isolated compatible `qwen-vl-utils==0.0.11` and no clean-worktree
    edits, long video128 succeeds in 26.00s, and video196 at
    `thinker_max_seq_len=65536` reproduces missing admission:
    `required_tokens=35704`, `kv_capacity=23257`.
- Upstream-vs-PR stress split:
  - upstream-main video196 rejects at thinker KV capacity with
    `mem_fraction_static=0.7`;
  - PR TP1 and TP2 both allocate a 66,804-token thinker KV pool at
    `mem_fraction_static=0.45` and both succeed on the same 35,704-token
    request when the encoder activation budget is 10 GiB;
  - with lowered typed encoder activation budgets, PR TP1 rejects and TP2
    succeeds for video196 and no-truncation audio300 in older H200
    low-pixel/capped runs. The latest clean-H100 default-pixel video256 result
    is an encoder-boundary result, not E2E success.

The warmed process-local release-gating evidence for the successful video196
and audio30 paths is collected. Fine-grained latency attribution is now closed
for H100 video128, audio30, and PR-only no-truncation audio300. The new
timing fields are `inbox_admission_ms`, `strip_h2d_ms`,
`metadata_broadcast_ms`, `follower_allocation_ms`,
`allocation_handshake_ms`, `tensor_broadcast_ms`, `rank_wait_skew_ms`,
`rank_arrival_skew_ms`, `build_ms`, `forward_ms`, and `slice_ms`.
`rank_wait_skew_ms` / `rank_arrival_skew_ms` require
`SGLANG_OMNI_ENCODER_TIMING_DETAIL=1`. The latest H100 video128 three-way A/B
shows TP2 E2E mean `13.704s` vs PR TP1 `8.239s` and main `7.769s`; the measured
image encoder critical-path delta vs TP1 is `+5338 ms`, dominated by
`metadata_broadcast_ms` and `allocation_handshake_ms`, while forward is
`64 ms` lower at TP2 and tensor broadcast is negligible. The H100 audio30
three-way A/B shows TP2 `0.321s` vs PR TP1 `0.289s` and main `0.303s`; the small
delta is mostly recv-path coordination plus a smaller forward increase. The
PR-only audio300 warmed A/B shows TP2 E2E mean `0.596s` vs TP1 `0.449s`; the
measured encoder-total delta is also dominated by metadata broadcast and
allocation handshake.

## Residual Risks

- Current PR E2E proves long-audio functionality under clean H100
  same-payload/same-budget conditions and older long-video functionality under
  H200 capped-pixel conditions. Default-pixel video256/video512 on clean H100
  are boundary cases, not E2E successes.
- Unit tests validate memory contracts, admission math, GPU guard defer/reject,
  and reservation release. Activation budget is still a modeled per-rank
  admission guard, but runtime admission now also uses a whole-GPU guard with
  in-flight reservations; `encoder_max_batch_size` remains an explicit
  conservative fallback rather than the primary safety mechanism.
- Performance reporting now includes main / PR TP1 / PR TP2 warmed latency for
  video128 and audio30, plus PR-only no-truncation audio300. The measured video
  TP2 slowdown is metadata/admission handshake dominated, not forward compute
  or tensor payload broadcast.
- Default Qwen3-Omni audio preprocessing still truncates to the 30s Whisper
  window; long-audio encoder stress requires `audio_truncation=false`.
