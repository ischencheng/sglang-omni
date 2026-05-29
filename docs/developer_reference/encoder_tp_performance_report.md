# Encoder TP Validation and Performance Notes

Date: 2026-05-26. Latest evidence update: 2026-05-29.

## Current Workspace Status

CUDA is accessible in the Codex workspace used for these runs:

- `torch.cuda.is_available()==True`, `torch.cuda.device_count()==8`.
- Earlier evidence was collected on NVIDIA H200; the latest convergence runs
  below were collected on NVIDIA H100 80GB GPUs. The 2026-05-28 colocated
  matrix used clean GPUs 6/7. The 2026-05-29 video E2E rerun used a separate
  layout with clean encoder/thinker GPUs to remove the colocated thinker KV
  bottleneck from the video success claim.
- The Qwen3-Omni checkpoint is available at `/data/qwen3omni`, a symlink to
  snapshot `26291f793822fb6be9555850f06dfe95f2d7e695`.
- GPU memory is shared with hidden processes outside this namespace. Earlier
  PR server attempts with the default thinker `mem_fraction_static=0.7` failed
  at startup KV allocation on selected GPUs even though `nvidia-smi` reported
  no visible local processes. Each table below records the exact AR memory
  fraction used for that run; the successful separate-layout video512 run used
  a dedicated thinker GPU with `--thinker-mem-fraction-static 0.90`.

This is enough for current PR-branch parity and E2E validation, but the
performance claim is intentionally narrow. The defensible memory claim is:
encoder TP lowers the **per-rank temporary activation ceiling** for the portion
of the encoder activation estimate that is tensor-parallel sharded. It does not
claim that total cluster memory always falls, because raw multimodal inputs,
masks, metadata, model weights, AR KV, and colocated stages can remain
replicated or move independently.

The current headline E2E long-input evidence has two same-payload/same-budget
splits. First, audio1000 with `audio_truncation=false` rejects on TP1 because
the single-rank activation estimate exceeds the `0.43 GiB` budget, while audio
TP2 admits and completes the request. Second, default-pixel video512 in the
2026-05-29 separate-layout H100 run rejects on image TP1 because
`16357785600 > 15247133900`, while image TP2 admits and completes generation
with HTTP 200. Default-pixel video256/video512 in the older 2-GPU colocated
layout remain encoder-boundary evidence only: TP2 can admit and run encoder
forward where TP1 is above budget, but colocated thinker KV or memory becomes
the next bottleneck.

The broader TP2 stress run records the requested `256, 512, 1024, 2048` video
frame caps. The `512+` rows are admission/encoder-boundary evidence, not
generation-success claims. A separate encoder-only H100 sweep now records
TP1/TP2 runtime capacity and slope from `128` through `2048` frame caps; it
must not be described as E2E evidence. A TP4 encoder-only follow-up on empty
H100s shows TP4 max-rank peak and length slope are lower than TP1 on the tested
video path, while TP4 summed rank peak is higher. A separate colocated-2GPU H100
slope run shows that the encoder-only `b2 < b1` result does not generalize to
colocated runtime peaks: colocated image max-rank `b2` is slightly higher than
TP1 in the tested `64..160` frame range. The default HF processor clamps long
audio to the 30s Whisper feature window unless `audio_truncation=false`; the PR
exposes this override and the latest H100 evidence includes true 1000s and 3000s
audio encoder workloads.

Checkpoint:
`Qwen/Qwen3-Omni-30B-A3B-Instruct`
snapshot `26291f793822fb6be9555850f06dfe95f2d7e695`.

## 2026-05-28/29 H100 Converged Evidence

This is the current PR evidence shape. It separates four claims:

- **Memory:** encoder TP lowers the per-rank activation admission ceiling. In
  the encoder-only H100 video sweep, TP2 also lowers measured max-rank runtime
  peak slope, but both TP1 and TP2 complete through the tested `2048` frame
  cap. In the TP4 encoder-only follow-up, TP4 max-rank token slope is also
  lower than TP1 (`32.254` vs `45.517 MiB / 1k pre-merge tokens`), but TP4
  summed-rank slope is higher. In the colocated-2GPU H100 slope rerun, TP2 image
  max-rank runtime slope is not lower (`54.938` vs `52.625 MiB/frame`). Do not
  claim lower total cluster memory, lower colocated runtime peak, or lower E2E
  memory.
- **E2E long input:** audio1000 and separate-layout video512 are the clean
  same-payload/same-budget E2E success splits: TP1 admission fails and TP2
  completes.
- **Latency:** video128 and audio30 now have warmed main / PR TP1 / PR TP2
  A/Bs. PR TP1/TP2 include fine-grained `encoder_batch_timing`; main is an
  E2E baseline because upstream main does not emit the new encoder timing
  fields.
- **Accuracy:** tensor parity scope is closed as described below. No CI
  benchmark artifact was available for PR #423 head
  `53328e698952ec14557702a3f00e676f9ded765a`, so a local formal
  Video-AMME CI-50 benchmark was run. At concurrency 1, main scored `36/50`
  (`72%`), PR TP1 scored `34/50` (`68%`), and PR TP2 scored `36/50` (`72%`),
  all with `0` failed requests and above the repository `66%` CI threshold.
  This is one benchmark lane, not the full task-level quality suite.

Unless otherwise noted, runs in this section used H100 80GB GPUs 6 and 7 with
`CUDA_VISIBLE_DEVICES=6,7`, `--layout colocated-2gpu`,
`SGLANG_OMNI_ENCODER_TIMING_DETAIL=1`,
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
model `/data/qwen3omni`, `--encoder-backend sglang`,
`--encoder-total-gpu-memory-fraction 0.01`,
`--talker-mem-fraction-static 0.12`, and
`--thinker-max-seq-len 131072`. The prelaunch `nvidia-smi` snapshots saved in
the artifact directories show GPUs 6/7 at `4 MiB` used and no compute app
before each fresh server. Other GPUs on the host were occupied, but these two
were memory-free for the runs below.

### Memory Experiment 1: Encoder-Only Max Video Length Capacity

This experiment removes thinker KV, generation, talker, and colocated stages.
It answers only the encoder-forward capacity question: if TP1 can use one H100
and TP2 can use two H100s, does TP2 complete a longer video encoder forward?

Artifact root:
[`/data/encoder_tp_evidence_20260526/h100_encoder_only_video_memory_20260529`](/data/encoder_tp_evidence_20260526/h100_encoder_only_video_memory_20260529).
The prelaunch snapshot recorded GPU6 and GPU7 at `4 MiB` used. The probe used
fresh subprocesses for every TP/frame point, `encoder_max_batch_size=1`,
`video_fps=30`, no `video_max_pixels` override, no activation budget, and a
measurement-only GPU guard that logs memory but does not reject candidates.
The stage-to-stage payload path is CPU SHM; TP2 then performs entry-rank H2D
lift, metadata broadcast, follower allocation, allocation handshake, tensor
broadcast, forward, and rank0 output staging inside the encoder scheduler.

Command:

```bash
python examples/encoder_tp_encoder_only_video_probe.py \
  --model-path /data/qwen3omni \
  --video /data/encoder_tp_evidence_20260526/media/draw_loop_2048frames.mp4 \
  --output-dir /data/encoder_tp_evidence_20260526/h100_encoder_only_video_memory_20260529 \
  --frame-caps 128,256,512,768,1024,1536,2048 \
  --tp1-gpu 6 \
  --tp2-gpus 6,7 \
  --video-fps 30 \
  --timeout 1200 \
  --gpu-sample-interval 0.5 \
  --stop-after-first-failure
```

| Frame cap | Actual `video_grid_thw` | Pre-merge visual tokens | Input tensor GiB | TP1 one-rank result / peak | TP2 two-rank result / max-rank peak | TP2 sum peak |
| ---: | --- | ---: | ---: | --- | --- | ---: |
| 128 | `[64,36,64]` | `147456` | `0.84` | success, `8910 MiB` | success, `8164 MiB` | `16306 MiB` |
| 256 | `[128,36,64]` | `294912` | `1.69` | success, `15590 MiB` | success, `13204 MiB` | `26386 MiB` |
| 512 | `[256,26,48]` | `319488` | `1.83` | success, `16730 MiB` | success, `14084 MiB` | `28146 MiB` |
| 768 | `[384,20,40]` | `307200` | `1.76` | success, `16150 MiB` | success, `13644 MiB` | `27266 MiB` |
| 1024 | `[512,20,36]` | `368640` | `2.11` | success, `18992 MiB` | success, `15844 MiB` | `31666 MiB` |
| 1536 | `[768,18,32]` | `442368` | `2.53` | success, `22412 MiB` | success, `18486 MiB` | `36950 MiB` |
| 2048 | `[1024,18,32]` | `589824` | `3.38` | success, `29210 MiB` | success, `23764 MiB` | `47506 MiB` |

Whole-GPU sampled peaks used `nvidia-smi` at `0.5s` intervals. They are a
coarser signal than per-process memory marks, so instantaneous allocator peaks
can be higher than the sampled whole-GPU maximum.

| Frame cap | TP1 GPU6 peak / min free | TP2 GPU6 peak / min free | TP2 GPU7 peak / min free |
| ---: | ---: | ---: | ---: |
| 128 | `8919 / 72161 MiB` | `7853 / 73227 MiB` | `7831 / 73249 MiB` |
| 256 | `15599 / 65481 MiB` | `13213 / 67867 MiB` | `13191 / 67889 MiB` |
| 512 | `16739 / 64341 MiB` | `14093 / 66987 MiB` | `14071 / 67009 MiB` |
| 768 | `16159 / 64921 MiB` | `13169 / 67910 MiB` | `13147 / 67932 MiB` |
| 1024 | `19001 / 62079 MiB` | `15453 / 65627 MiB` | `15347 / 65733 MiB` |
| 1536 | `22421 / 58658 MiB` | `18495 / 62584 MiB` | `18473 / 62606 MiB` |
| 2048 | `29219 / 51860 MiB` | `23289 / 57790 MiB` | `23267 / 57812 MiB` |

Conclusion for this implementation/path: **TP2 did not extend the maximum
successful frame cap in the tested range**, because TP1 and TP2 both completed
encoder forward through `2048`. The boundary was not reached. This table should
not be used as E2E evidence; it is encoder-only.

### Memory Experiment 2: Runtime NVML Peak Slope

The same encoder-only sweep gives a clean runtime slope because every TP/frame
point starts in a fresh process. The frame-cap fit is kept for traceability, but
the preprocessor changes spatial grid with long frame caps, so the
pre-merge-token fit is the more meaningful length proxy.

| Fit target | TP1 one rank | TP2 max rank | TP2 sum, reference only |
| --- | ---: | ---: | ---: |
| `peak(frame_cap) = A + b * frame_cap` | `A=10569 MiB`, `b=8.611 MiB/frame`, `R^2=0.905` | `A=9365 MiB`, `b=6.638 MiB/frame`, `R^2=0.907` | `A=18708 MiB`, `b=13.277 MiB/frame`, `R^2=0.907` |
| `peak(k_premerge_tokens) = A + b * L` | `A=2069 MiB`, `b=47.061 MiB / 1k tokens`, `R^2=0.99996` | `A=2832 MiB`, `b=36.222 MiB / 1k tokens`, `R^2=0.99980` | `A=5641 MiB`, `b=72.444 MiB / 1k tokens`, `R^2=0.99980` |
| Crossover | n/a | `70.37k` pre-merge tokens for TP2 max-rank vs TP1 | not used for PR claim |

Interpretation:

- On this encoder-only video path, measured TP2 **max-rank** runtime peak has
  lower length slope than TP1 (`36.222 < 47.061 MiB / 1k pre-merge tokens`).
- TP2 has higher fixed overhead in the token fit (`2832 MiB` vs `2069 MiB`),
  but the crossover is below the smallest measured point (`70.37k` vs
  `147.46k` pre-merge tokens), so TP2 max-rank peak is lower for every measured
  length in this sweep.
- TP2 **sum** peak is higher and has a higher slope. This is only a cluster
  memory reference and is not the PR memory claim.
- This result is limited to encoder-only image/video forward with
  `max_batch_size=1`. It does not imply E2E memory, colocated memory, audio
  memory, or total cluster memory is lower.

Derived fit artifact:
[`derived_runtime_fit.json`](/data/encoder_tp_evidence_20260526/h100_encoder_only_video_memory_20260529/derived_runtime_fit.json).

### Memory Experiment 3: Encoder-Only TP4 Runtime Slope

This follow-up answers whether TP4 still has higher max-rank NVML peak than
TP1, and whether `b4 > b1`. It uses the same encoder-only path as Experiments 1
and 2: no thinker, no talker, no audio encoder, no generation, no activation
budget, no `video_max_pixels` override, fresh subprocesses per TP/frame point,
and `encoder_max_batch_size=1`. This does not answer colocated or E2E TP4
memory.

Artifact root:
[`/data/encoder_tp_evidence_20260526/h100_encoder_only_video_tp4_slope_20260529`](/data/encoder_tp_evidence_20260526/h100_encoder_only_video_tp4_slope_20260529).
The prelaunch snapshot recorded GPU1/GPU3/GPU4/GPU5 at driver-only memory.

Command:

```bash
python examples/encoder_tp_encoder_only_video_probe.py \
  --model-path /data/qwen3omni \
  --video /data/encoder_tp_evidence_20260526/media/draw_loop_2048frames.mp4 \
  --output-dir /data/encoder_tp_evidence_20260526/h100_encoder_only_video_tp4_slope_20260529 \
  --frame-caps 128,256,512,768,1024 \
  --tp-specs '1:1;4:1,3,4,5' \
  --video-fps 30 \
  --timeout 1200 \
  --gpu-sample-interval 0.5 \
  --stop-after-first-failure
```

| Frame cap | Pre-merge visual tokens | TP1 max-rank peak | TP4 rank peaks | TP4 max-rank peak | TP4 sum peak |
| ---: | ---: | ---: | --- | ---: | ---: |
| 128 | `147456` | `8910 MiB` | `7206 / 7206 / 7206 / 7206 MiB` | `7206 MiB` | `28824 MiB` |
| 256 | `294912` | `15590 MiB` | `11966 / 11946 / 11946 / 11946 MiB` | `11966 MiB` | `47804 MiB` |
| 512 | `319488` | `16730 MiB` | `12746 / 12746 / 12746 / 12746 MiB` | `12746 MiB` | `50984 MiB` |
| 768 | `307200` | `16150 MiB` | `12346 / 12346 / 12346 / 12346 MiB` | `12346 MiB` | `49384 MiB` |
| 1024 | `368640` | `18992 MiB` | `14346 / 14326 / 14326 / 14326 MiB` | `14346 MiB` | `57324 MiB` |

| Fit target | TP1 | TP4 | Interpretation |
| --- | ---: | ---: | --- |
| `max_rank_peak(k_premerge_tokens)` | `A=2187 MiB`, `b=45.517 MiB / 1k tokens`, `R^2=0.99997` | `A=2448 MiB`, `b=32.254 MiB / 1k tokens`, `R^2=0.99999` | `b4 < b1`; TP4 max-rank runtime slope is lower on this encoder-only path. |
| `max_rank_peak(frame_cap)` | `A=10782 MiB`, `b=8.356 MiB/frame`, `R^2=0.655` | `A=8546 MiB`, `b=5.908 MiB/frame`, `R^2=0.652` | Same direction, but lower `R^2` because preprocessing changes spatial grid with frame cap. |
| `sum_rank_peak(k_premerge_tokens)` | `A=2187 MiB`, `b=45.517 MiB / 1k tokens`, `R^2=0.99997` | `A=9823 MiB`, `b=128.822 MiB / 1k tokens`, `R^2=0.999999` | Reference only. TP4 summed rank peak is higher and is not a lower total-memory claim. |

Interpretation:

- On this encoder-only image/video path, TP4 max-rank NVML peak is lower than
  TP1 at every measured frame cap, and `b4 < b1` when fit against actual
  pre-merge visual tokens.
- TP4 sum peak is much higher than TP1 because four ranks hold replicated and
  sharded runtime state. This reinforces that the PR should not claim lower
  total cluster memory.
- This TP4 result is not colocated TP4 evidence. The current `colocated-2gpu`
  launcher intentionally supports only TP1/TP2; the existing `separate` full
  E2E layout would require more visible GPUs for `image_tp4 + audio_tp1 +
  thinker + talker` than were available for this follow-up.

### Memory Experiment 4: Colocated-2GPU Runtime Slope

This run answers the follow-up question: if the full colocated server is present
and encoder batch size is forced to `1`, does the encoder-only `b2 < b1` runtime
slope still hold? It does not. The run used fresh servers per TP/frame point,
`CUDA_VISIBLE_DEVICES=1,3`, `--layout colocated-2gpu`,
`--thinker-mem-fraction-static 0.78`, `--talker-mem-fraction-static 0.12`,
`--thinker-max-seq-len 131072`, `--encoder-max-batch-size 1`, image budget
`10 GiB`, audio budget `1 GiB`, `video_fps=30`, no `video_max_pixels` override,
and `max_tokens=8`.

Artifact root:
[`/data/encoder_tp_evidence_20260526/h100_colocated_video_memory_slope_20260529_budget10`](/data/encoder_tp_evidence_20260526/h100_colocated_video_memory_slope_20260529_budget10).
The first prelaunch snapshot records selected GPUs at driver-only memory
(`GPU1=4 MiB`, `GPU3=10 MiB`).

Command:

```bash
python examples/encoder_tp_colocated_video_slope.py \
  --output-dir /data/encoder_tp_evidence_20260526/h100_colocated_video_memory_slope_20260529_budget10 \
  --frame-caps 64,96,128,160 \
  --cuda-visible-devices 1,3 \
  --base-port 8340 \
  --thinker-mem-fraction-static 0.78 \
  --talker-mem-fraction-static 0.12 \
  --thinker-max-seq-len 131072 \
  --image-budget-gib 10 \
  --audio-budget-gib 1 \
  --max-tokens 8 \
  --gpu-sample-interval 0.5
```

| Frame cap | Prompt tokens | TP1 image process peak | TP2 image rank peaks | TP2 image max-rank peak | TP1 selected-GPU max | TP2 selected-GPU max |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 64 | `18448` | `5450 MiB` | `5424 / 6164 MiB` | `6164 MiB` | `72441 MiB` | `73061 MiB` |
| 96 | `27664` | `7130 MiB` | `6624 / 8084 MiB` | `8084 MiB` | `74121 MiB` | `74261 MiB` |
| 128 | `36880` | `8910 MiB` | `8164 / 9884 MiB` | `9884 MiB` | `75903 MiB` | `75803 MiB` |
| 160 | `46096` | `10470 MiB` | `9264 / 11424 MiB` | `11424 MiB` | `77463 MiB` | `76903 MiB` |

| Fit target | TP1 | TP2 | Interpretation |
| --- | ---: | ---: | --- |
| `image_max_rank_peak(frame_cap) = A + b * frame_cap` | `A=2096 MiB`, `b=52.625 MiB/frame`, `R^2=0.9994` | `A=2736 MiB`, `b=54.938 MiB/frame`, `R^2=0.9976` | `b2 >= b1`; this colocated path does not support a lower runtime image max-rank slope claim. |
| `selected_whole_gpu_max(frame_cap) = A + b * frame_cap` | `A=69085 MiB`, `b=52.650 MiB/frame`, `R^2=0.9994` | `A=70433 MiB`, `b=40.838 MiB/frame`, `R^2=0.9961` | Reference only. This max is dominated by resident thinker/talker placement and split encoder load, not an encoder-only runtime peak model. |
| `sum(selected_gpu_peaks)(frame_cap) = A + b * frame_cap` | `A=79228 MiB`, `b=52.650 MiB/frame`, `R^2=0.9994` | `A=85841 MiB`, `b=95.775 MiB/frame`, `R^2=0.9979` | Reference only. Summed selected-GPU peak is higher for TP2 and is not a lower total-memory claim. |

Interpretation:

- Colocated image max-rank peak has higher fixed overhead and slightly higher
  slope for TP2 in this `64..160` frame range.
- The TP2 whole-GPU max can look lower at larger frame caps because rank1 work
  is moved to GPU3 while GPU1 is dominated by thinker resident memory. That is
  not evidence that encoder runtime peak is lower.
- The PR should keep the memory claim split: TP2 lowers the modeled per-rank
  activation admission ceiling; encoder-only video forward also showed lower
  max-rank runtime slope; colocated runtime peak remains dependent on resident
  stages, TP fan-out/staging, allocator behavior, and GPU headroom.

### Admission Memory (Separate From Runtime Peak)

The admission model remains:

```text
cost(tp) = multiplier * (replicated_bytes + sharded_bytes / tp)
```

For Qwen3-Omni image/video and audio encoders, `multiplier=5`. Image/video
costs solve to a `60.0% replicated / 40.0% sharded` pre-multiplier split.
Audio costs solve to `49.2% replicated / 50.8% sharded`. These proportions are
admission-cost proportions; they are not NVML whole-GPU memory proportions.

| Payload | Encoder stage | Replicated / sharded | TP1 admission cost | TP2 admission cost | Budget used | Admission result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| video256, default pixels, `video_fps=30`, `video_max_frames=256` | image | `60.0% / 40.0%` | `15099494400` | `12079595520` | `12884901888` (`12 GiB`) | TP1 rejects; TP2 admits |
| video512, default pixels, `video_fps=30`, `video_max_frames=512` | image | `60.0% / 40.0%` | `16357785600` measured | `13086228480` measured | `15247133900` (`14.2 GiB`) | TP1 rejects; TP2 admits |
| audio300, `audio_truncation=false` | audio | `49.2% / 50.8%` | `157272000` | `117336000` | `461708984` (`0.43 GiB`) | both admit, used for latency |
| audio1000, `audio_truncation=false` | audio | `49.2% / 50.8%` | `524240000` | `391120000` | `461708984` (`0.43 GiB`) | TP1 rejects; TP2 admits |
| audio3000, `audio_truncation=false` | audio | `49.2% / 50.8%` | `1572720000` derived | `1173360000` measured | `1288490188` (`1.2 GiB`) | TP2 admits |

The admission table supports the per-rank activation-ceiling claim. It should
not be converted into a runtime NVML peak claim.

### Runtime Caveats and Source of TP2 Extra Peak

Runtime peak is:

```text
resident_rank
  + replicated_runtime
  + sharded_runtime / tp
  + TP fan-out transient
  + rank0/follower staging
  + allocator cache/fragmentation
  + dynamic batching
  + colocated stage headroom
```

In the TP2 scheduler, payload tensors arrive through CPU SHM and are then lifted
to the entry rank GPU. The TP-specific runtime terms are metadata broadcast,
follower allocation, allocation handshake, tensor broadcast, rank0/follower
staging, and allocator behavior. Do not describe stage-to-stage payloads as
large tensors directly relayed through GPU memory.

Existing H100 memory-mark reruns show TP2 colocated/E2E peaks at or after tensor
fan-out and forward:

| H100 memory-mark run | Peak source | Peak phase | Result |
| --- | --- | --- | --- |
| video128, `video_max_pixels=401408` | image rank0 `5484 MiB`, image rank1 `6364 MiB` | `after_forward` / `after_cleanup_synchronize` | E2E success |
| video256, default pixels, cold first request | image rank0 `10744 MiB`, image rank1 `13384 MiB` | `after_forward` / `after_cleanup_synchronize` | encoder success; thinker KV reject |
| video512, default pixels, separate layout | image rank0 `14104 MiB`, image rank1 `17844 MiB` | `after_forward` / `after_cleanup_synchronize`; TP2 timing was dominated by measured metadata/admission handshake fields | E2E success |
| audio30 | audio rank0/rank1 `3370 MiB` | tensor broadcast raises NVML; `after_forward` is max | E2E success |
| audio1000, `audio_truncation=false` | audio rank0 `8190 MiB`, audio rank1 `8290 MiB` | `after_forward` / `after_cleanup_synchronize` | E2E success |

The clean-H100 video256 colocated OOM is not explained by another user on the
GPU. It is explained by colocated resident memory plus default-pixel video
runtime pressure. For c4, classify TP2 OOM as a capacity/perf boundary caused
by dynamic encoder batch accumulation plus colocated GPU0 headroom, not a
quality regression.

#### Admission-Driven Batching / Whole-GPU Guard

The activation budget is an admission guard for modeled encoder activation
cost, not a complete whole-GPU memory model. Before the follow-up guard, it did
not force the encoder batch size to be the largest **safe** admitted batch under
whole-GPU headroom. Under c4, TP2's recv/admission/handshake latency allowed
image requests to accumulate into larger dynamic batches before forward:

| Run | Audio encoder admission batch sizes | Image encoder admission batch sizes | Interpretation |
| --- | --- | --- | --- |
| PR TP1 c1 | all `batch_size=1` | all `batch_size=1` | single-request admission path |
| PR TP1 c4 | `50x batch_size=1` | `48x batch_size=1`, `1x batch_size=2` | high concurrency mostly remains single-item batches |
| PR TP2 c1 | all `batch_size=1` | all `batch_size=1` | single-request admission path |
| PR TP2 c4 before batch cap | `25x batch_size=1` | `2x batch_size=1`, `9x batch_size=2`, `1x batch_size=3` | TP2 request accumulation creates larger image batches |
| PR TP2 c4 after `encoder_max_batch_size=1` default | `50x batch_size=1` | `50x batch_size=1` | temporary cap prevents c4-style dynamic encoder batch accumulation |
| PR TP2 c4 after whole-GPU guard, no encoder batch cap | `50x batch_size=1` | `48x batch_size=1`, `1x batch_size=2`; `47` image candidate batches deferred by `gpu_guard` | admission guard allows a larger batch only when projected whole-GPU headroom fits, otherwise shrinks/defer without OOM |

The PR TP2 c4 failure is therefore classified as an encoder OOM /
capacity-perf boundary due to dynamic batch accumulation and colocated GPU0
headroom, not a task-quality regression. The image encoder rank0 failed in
forward when GPU0 had only about `222 MiB` free and rank0 tried to allocate
`248 MiB`. GPU0 was simultaneously holding the thinker process
(`~65.29 GiB`), image rank0 (`~10.29 GiB`), and audio rank0 (`~3.35 GiB`).

The follow-up implementation now makes encoder batch size an admission result
and adds a per-GPU projected-memory guard with cross-process in-flight
reservations. The guard samples whole-GPU free memory, adds already-admitted
encoder reservations on the same physical GPU, then checks the candidate
per-rank activation cost plus transient and allocator margins before
admission. The temporary `encoder_max_batch_size` control remains available as
an explicit fallback, but the TP2 `colocated-2gpu` launcher no longer needs to
default it to `1` when guard telemetry is available.

### E2E Long-Input Results

| Payload | TP1 result | TP2 result | Classification |
| --- | --- | --- | --- |
| audio1000, `audio_truncation=false`, same `0.43 GiB` audio budget | admission reject: `524240000 > 461708984`; health OK | E2E success, HTTP 200, prompt `13018`, total `13034`, latency `3.075s`; output begins `["How many cars are there in the picture?", "How many` | Main same-payload/same-budget E2E enablement evidence |
| video512 default pixels, separate layout, same `14.2 GiB` image budget | admission reject: `16357785600 > 15247133900`; HTTP 500; health OK before/after | E2E success, HTTP 200, prompt `79888`, total `79904`, latency `82.047s`; output: `A man is drawing a guitar on a tablet with a stylus. He is` | Main video same-payload/same-budget E2E enablement evidence |
| video256 default pixels, same `12 GiB` image budget | admission reject: `15099494400 > 12884901888`; health OK | TP2 admits and image forward completes at thinker 0.80, then thinker KV rejects; at thinker 0.81 image rank0 OOMs | TP2 pushes encoder boundary; colocated thinker KV / memory becomes next bottleneck |
| video512 default pixels, colocated layout, `14.2 GiB` image budget | TP1 not rerun in this H100 main-matrix pass; TP1 cost is measured in the separate-layout rerun above | TP2 admits and image forward completes, then thinker KV rejects | TP2 pushes encoder boundary; colocated thinker KV becomes next bottleneck |
| video512 + audio3000 default pixels/no trunc | no TP1 pair in this pass | audio admits and completes; image admits then rank0 OOMs while audio rank0 holds `8.50 GiB` | Negative multimodal boundary case, not E2E success |

#### 2026-05-29 Separate-Layout Video512 E2E A/B

Artifacts are under
[`/data/encoder_tp_evidence_20260526/h100_5gpu_separate_video_20260529`](/data/encoder_tp_evidence_20260526/h100_5gpu_separate_video_20260529).
The request used `/data/encoder_tp_evidence_20260526/media/draw_loop_2048frames.mp4`,
`video_fps=30`, `video_max_frames=512`, prompt
`Briefly describe the video.`, `max_tokens=16`, and no
`video_max_pixels` override. The checkpoint, prompt/media set, and decoding
config were identical for TP1 and TP2.

| Run | Server command / env | GPU initial state | Admission / E2E result | Artifact |
| --- | --- | --- | --- | --- |
| TP2 | `CUDA_VISIBLE_DEVICES=1,3,5,6,7 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_OMNI_ENCODER_TIMING_DETAIL=1 SGLANG_OMNI_ENCODER_MEMORY_DETAIL=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python examples/qwen3_omni_encoder_tp.py --model /data/qwen3omni --encoder-backend sglang --layout separate --image-tp 2 --audio-tp 1 --image-encoder-activation-budget-gib 14.2 --audio-encoder-activation-budget-gib 1 --encoder-total-gpu-memory-fraction 0.01 --thinker-mem-fraction-static 0.90 --talker-mem-fraction-static 0.12 --thinker-max-seq-len 131072 --port 8150` | Prelaunch `nvidia_smi_before_tp2_server.txt`: physical GPUs 1/3/5/6/7 were driver-only (`4/10/4/4/4 MiB`); GPU4 was busy and not selected. | `encoder_admission` admitted `batch_cost=13086228480 < 15247133900`; HTTP 200; prompt `79888`, total `79904`; output `A man is drawing a guitar on a tablet with a stylus. He is`; health OK before/after. | [`server-tp2-separate-video.log`](/data/encoder_tp_evidence_20260526/h100_5gpu_separate_video_20260529/server-tp2-separate-video.log), [`tp2_separate_video512_no_pixel_override`](/data/encoder_tp_evidence_20260526/h100_5gpu_separate_video_20260529/tp2_separate_video512_no_pixel_override) |
| TP1 | `CUDA_VISIBLE_DEVICES=1,3,6,7 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_OMNI_ENCODER_TIMING_DETAIL=1 SGLANG_OMNI_ENCODER_MEMORY_DETAIL=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python examples/qwen3_omni_encoder_tp.py --model /data/qwen3omni --encoder-backend sglang --layout separate --image-tp 1 --audio-tp 1 --image-encoder-activation-budget-gib 14.2 --audio-encoder-activation-budget-gib 1 --encoder-total-gpu-memory-fraction 0.01 --thinker-mem-fraction-static 0.90 --talker-mem-fraction-static 0.12 --thinker-max-seq-len 131072 --port 8151` | Prelaunch `nvidia_smi_before_tp1_server_retry_reordered.txt`: physical GPUs 1/3/6/7 were driver-only (`4/10/4/4 MiB`). GPU5 had hidden memory by the retry, so TP1 used the clean four-GPU subset required by `--layout separate`. | Admission rejected before encoder forward: `encoder request cost 16357785600 exceeds max_single_request_cost=15247133900`; HTTP 500 for the request; health OK before/after. | [`server-tp1-separate-video-retry-reordered.log`](/data/encoder_tp_evidence_20260526/h100_5gpu_separate_video_20260529/server-tp1-separate-video-retry-reordered.log), [`tp1_separate_video512_no_pixel_override_budget142_retry_reordered`](/data/encoder_tp_evidence_20260526/h100_5gpu_separate_video_20260529/tp1_separate_video512_no_pixel_override_budget142_retry_reordered) |

TP2 encoder timing for this single E2E success is recorded for diagnosis only;
it is not a warmed latency claim:

| Rank | recv | inbox/admission | strip+H2D | metadata broadcast | follower allocation | allocation handshake | tensor broadcast | rank wait/skew | build | forward | slice | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| image rank0 | `26873.101 ms` | `157.878 ms` | `0.099 ms` | `11089.138 ms` | `0.000 ms` | `15421.312 ms` | `191.574 ms` | `0.917 ms` | `1.397 ms` | `2341.081 ms` | `0.318 ms` | `29242.679 ms` |
| image rank1 | `26873.146 ms` | `0.000 ms` | `0.000 ms` | `26653.331 ms` | `13.761 ms` | `2.092 ms` | `191.657 ms` | `0.961 ms` | `1.510 ms` | `2341.069 ms` | `0.000 ms` | `29238.672 ms` |

The initial same-order TP1 launch attempted to place the thinker on physical
GPU5 after that GPU had accumulated hidden memory and failed during thinker
startup. That startup failure is recorded in
[`server-tp1-separate-video.log`](/data/encoder_tp_evidence_20260526/h100_5gpu_separate_video_20260529/server-tp1-separate-video.log)
but is not used as PR evidence. The PR evidence is the retry above, where the
server starts on clean GPUs and the request fails specifically at image encoder
admission under the same payload and budget.

### Latency Breakdown

The three-way latency rerun uses `qwen-vl-utils==0.0.11`, H100 GPUs 6/7 from a
`4 MiB` prelaunch state, one warmup plus three measured repeats, and the same
checkpoint/prompt/media/decoding config across main, PR TP1, and PR TP2.
Workloads are intentionally both-success latency attribution workloads:

- video128: `tests/data/draw.mp4`, `video_fps=16`,
  `video_max_frames=128`, `video_max_pixels=401408`, `max_tokens=16`.
- audio30: `/data/encoder_tp_evidence_20260526/long_query_to_cars_30s.wav`,
  default truncation, `max_tokens=32`.

Server args:

| Config | Server command shape | Timing scope | Artifact |
| --- | --- | --- | --- |
| upstream main | `/data/sglang-omni-upstream-main/examples/run_qwen3_omni_speech_server.py --gpu-thinker 0 --gpu-talker 1 --gpu-code2wav 1 --gpu-image-encoder 0 --gpu-audio-encoder 0 --thinker-max-seq-len 32768 --thinker-mem-fraction-static 0.80 --talker-mem-fraction-static 0.12 --port 8140` | E2E only; no fine encoder timing fields in main | `/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528` |
| PR TP1 | `examples/qwen3_omni_encoder_tp.py --layout colocated-2gpu --image-tp 1 --audio-tp 1 --image-encoder-activation-budget-gib 10 --audio-encoder-activation-budget-gib 1 --encoder-total-gpu-memory-fraction 0.01 --thinker-mem-fraction-static 0.80 --talker-mem-fraction-static 0.12 --thinker-max-seq-len 32768 --port 8141` | E2E plus fine encoder timing | same |
| PR TP2 | same as PR TP1 but `--image-tp 2 --audio-tp 2 --port 8142` | E2E plus fine encoder timing | same |

Profiling / timing artifacts:

| Artifact | What it proves |
| --- | --- |
| [`latency_attribution_analysis.json`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/latency_attribution_analysis.json) | Parsed profile summary for main/PR TP1/PR TP2 E2E latency and PR fine-grained `encoder_batch_timing` fields. This is the source for the latency attribution tables below. |
| [`server-pr-tp1-video128-audio30.log`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/server-pr-tp1-video128-audio30.log) | Raw PR TP1 `encoder_batch_timing` logs. |
| [`server-pr-tp2-video128-audio30.log`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/server-pr-tp2-video128-audio30.log) | Raw PR TP2 `encoder_batch_timing` logs. |
| [`server-main-qv011-video128-audio30-mem080.log`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/server-main-qv011-video128-audio30-mem080.log) | Main E2E server log; main does not emit the new encoder timing fields. |

E2E results:

| Workload | main E2E mean / p95 | PR TP1 E2E mean / p95 | PR TP2 E2E mean / p95 | Tokens | E2E interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| video128 | `7.769s / 7.988s` | `8.239s / 8.293s` | `13.704s / 13.891s` | `18944 -> 18960` | TP2 is `+5.464s` vs PR TP1 and `+5.934s` vs main. |
| audio30 | `0.303s / 0.310s` | `0.289s / 0.295s` | `0.321s / 0.328s` | `406 -> 438` | TP2 is `+31.8ms` vs PR TP1 and `+18.2ms` vs main. |

Fine-grained PR encoder critical-path timing, measured means with warmup
excluded:

| Workload / config | recv | inbox/admission | metadata broadcast | allocation handshake | tensor broadcast | build | forward | slice | encoder total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| video128 PR TP1 image rank0 | `73.206 ms` | `73.155 ms` | `0.000 ms` | `0.000 ms` | `0.000 ms` | `0.123 ms` | `234.025 ms` | `0.086 ms` | `307.471 ms` |
| video128 PR TP2 image critical path | `5473.599 ms` | `99.935 ms` | `2671.053 ms` | `2701.687 ms` | `0.259 ms` | `0.477 ms` | `170.406 ms` | `0.129 ms` | `5645.361 ms` |
| audio30 PR TP1 audio rank0 | `92.354 ms` | `92.307 ms` | `0.000 ms` | `0.000 ms` | `0.000 ms` | `0.146 ms` | `9.848 ms` | `0.042 ms` | `102.417 ms` |
| audio30 PR TP2 audio critical path | `121.155 ms` | `0.000 ms` | `119.979 ms` | `0.547 ms` | `0.173 ms` | `0.267 ms` | `17.919 ms` | `0.000 ms` | `140.009 ms` |

Video attribution: PR TP2 is `+5464 ms` E2E over PR TP1. The image encoder
critical-path delta is `+5338 ms`, which accounts for nearly all of the E2E
delta. Within that encoder delta, forward is not the cause: TP2 image forward
is `63.6 ms` lower than TP1. The measured increase is the recv/admission path:
`recv_ms` is `+5400 ms`, with rank0 spending `2671 ms` in
`metadata_broadcast_ms` and `2702 ms` in `allocation_handshake_ms`; rank1
spends `5471 ms` in `metadata_broadcast_ms`. `tensor_broadcast_ms` is only
`0.26 ms`. Therefore the defensible wording is **metadata/admission handshake
dominates the current video TP2 latency delta**. Do not describe it as tensor
payload broadcast or forward compute.

Audio attribution: PR TP2 is `+31.8 ms` E2E over PR TP1. The audio encoder
critical path is `+37.6 ms`; the measured split is `+28.8 ms` recv path and
`+8.1 ms` forward. For short audio30, TP2 is close to main/TP1, but the
measurable slowdown is still mostly encoder coordination with a smaller forward
increase. This is a measured latency attribution only, not a TP2 speed claim.

The previous long-audio PR-only warmed A/B uses audio300 with
`audio_truncation=false`, same payload, same `0.43 GiB` audio budget, one
warmup and three measured repeats. TP1 used
`--image-tp 1 --audio-tp 1 --port 8136`; TP2 used
`--image-tp 1 --audio-tp 2 --port 8135`. Both had health OK before/after. This
does not include upstream main because main does not expose the
`audio_truncation=false` override used to force true 300s encoder input.

| Metric | TP1 rank0 mean | TP2 rank0 mean | TP2 rank1 mean |
| --- | ---: | ---: | ---: |
| E2E latency | `0.449s` | `0.596s` | `0.596s` |
| recv total | `111.896 ms` | `244.966 ms` | `244.970 ms` |
| inbox/admission | `111.847 ms` | `94.368 ms` | `0.000 ms` |
| strip + H2D lift | `0.044 ms` | `0.044 ms` | `0.000 ms` |
| metadata broadcast | `0.000 ms` | `75.652 ms` | `243.411 ms` |
| follower allocation | `0.000 ms` | `0.000 ms` | `0.028 ms` |
| allocation handshake | `0.000 ms` | `74.334 ms` | `0.954 ms` |
| tensor broadcast | `0.000 ms` | `0.192 ms` | `0.191 ms` |
| rank wait/skew | `0.000 ms` | `0.366 ms` | `0.347 ms` |
| rank arrival skew | `0.000 ms` | `0.022 ms` | `0.022 ms` |
| build | `0.179 ms` | `0.239 ms` | `0.234 ms` |
| forward | `29.976 ms` | `35.295 ms` | `35.295 ms` |
| slice | `0.044 ms` | `0.050 ms` | `0.000 ms` |
| encoder total | `142.123 ms` | `281.166 ms` | `281.100 ms` |

Attribution for this audio300 run: TP2 E2E mean is `+0.147s` over TP1. The
measured encoder total delta is about `+0.139s`; the recv delta is
`+133.1 ms` while forward is only `+5.3 ms`. Rank0 spends `75.7 ms` in
`metadata_broadcast_ms` and `74.3 ms` in `allocation_handshake_ms`; rank1
spends `243.4 ms` in `metadata_broadcast_ms`. This supports the same
recv/admission-path pattern for longer audio, but remains PR-only rather than
main/PR parity.

### Accuracy / Quality

Accuracy evidence is split deliberately into tensor-level attribution and
task-level benchmark quality:

- Tensor-level attribution is recorded in
  [`encoder_tp_parity_findings.md`](/data/sglang-omni/docs/developer_reference/encoder_tp_parity_findings.md).
  The wrapper-vs-bare SGLang lane is bit-equal, so the wrapper itself is not a
  source of tensor drift. HF-vs-SGLang visual drift is an upstream
  implementation gap. SGLang TP1-vs-TP2 image/video drift is the expected fp16
  TP reduction-order tail plus nonlinear visual-stack amplification. Audio
  TP1-vs-TP2 is strict-allclose after valid-token flattening.
- Task-level quality is measured by the local Video-AMME CI-50 lane. The
  rerun artifact is
  [`videoamme_ci50_benchmark_summary.json`](/data/encoder_tp_evidence_20260526/h100_gpu6_7/quality_videoamme_ci50_rerun_20260529_002115/videoamme_ci50_benchmark_summary.json).
  This supports **no regression on this CI-50 benchmark lane**: main and PR TP2
  are both `36/50` (`72%`), PR TP1 is `34/50` (`68%`), all have `0` failed
  requests, and all pass the `66%` CI threshold.
- The c4 Video-AMME rows below are capacity/perf safety evidence, not the
  primary task-quality lane. The pre-guard TP2 c4 failure is classified as
  encoder OOM/capacity, not quality regression. The post-guard c4 run shows the
  guard avoids that capacity failure.

| Benchmark / scope | main score | PR TP1 score | PR TP2 score | Delta vs main | Pass threshold | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Formal Video-AMME CI-50, `max_concurrency=1` | `36/50` (`72.0%`) | `34/50` (`68.0%`) | `36/50` (`72.0%`) | PR TP1 `-4.0 pp`; PR TP2 `0.0 pp` | Video-AMME CI threshold `>=66%` accuracy and `0` failed requests | Passes threshold for all three. TP2 matches main on this benchmark lane; TP1 passes threshold but is lower than main in this run. Rerun artifact: `/data/encoder_tp_evidence_20260526/h100_gpu6_7/quality_videoamme_ci50_rerun_20260529_002115/videoamme_ci50_benchmark_summary.json`. |
| Video-AMME CI-50, `max_concurrency=4` before `encoder_max_batch_size` cap | `36/50` (`72.0%`), `0` failed | `33/50` (`66.0%`), `0` failed | `14/50` (`28.0%`), `29` failed | PR TP1 `-6.0 pp`; PR TP2 not quality-comparable | same benchmark threshold | Historical capacity/perf failure: TP2 c4 image encoder rank0 OOMed during visual forward after only `21/50` completed. |
| Video-AMME CI-50, `max_concurrency=4` after TP2 colocated `encoder_max_batch_size=1` default | prior `36/50` (`72.0%`), `0` failed | prior `33/50` (`66.0%`), `0` failed | `36/50` (`72.0%`), `0` failed | PR TP2 `0.0 pp` vs prior main | same benchmark threshold | Passes threshold. TP2 c4 post-cap used `50x batch_size=1` for image and audio, mean latency `15.941s`, p95 `17.645s`, QPS `0.244`. Artifact: `/data/encoder_tp_evidence_20260526/h100_gpu6_7/quality_videoamme_ci50_post_batchcap_20260529_004439/videoamme_ci50_post_batchcap_summary.json`. |
| Video-AMME CI-50, `max_concurrency=4` after whole-GPU guard, no encoder batch cap | prior `36/50` (`72.0%`), `0` failed | prior `33/50` (`66.0%`), `0` failed | `36/50` (`72.0%`), `0` failed | PR TP2 `0.0 pp` vs prior main | same benchmark threshold | Passes threshold. TP2 c4 post-guard used scheduler `max_batch_size=32` with no explicit `--encoder-max-batch-size`; image admitted `48x batch_size=1` and `1x batch_size=2`, with `47` image candidates deferred by `gpu_guard`; audio admitted `50x batch_size=1`. Mean latency `17.147s`, p95 `19.431s`, QPS `0.227`. Artifact: `/data/encoder_tp_evidence_20260526/h100_gpu6_7/quality_videoamme_ci50_post_gpuguard_20260529_010354/videoamme_ci50_post_gpuguard_summary.json`. |
| Smoke quality subset: `encoder_tp_smoke_quality_video128_audio30` | `6/6` (`100%`) | `6/6` (`100%`) | `6/6` (`100%`) | `0.0 pp` for PR TP1 and PR TP2 | no per-task pass-rate drop vs main on fixed video128/audio30 subset | Supporting smoke evidence only. Artifact: `/data/encoder_tp_evidence_20260526/h100_gpu6_7/latency_main_pr_tp_20260528/smoke_quality_benchmark.json`. |
| Full task-level quality suite | not run | not run | not run | n/a | MMMU/Video-MME/Video-AMME/MMSU-style suite with fixed thresholds | Still open. The local Video-AMME CI-50 lane is stronger than smoke evidence but does not prove full task-level no regression. |
| Wrapper vs bare SGLang encoder tensors | n/a | bit-equal | bit-equal | n/a | exact equality for same SGLang module path | Closed by tensor parity harness. |
| HF local vs SGLang encoder tensors | n/a | characterized drift | characterized drift | n/a | no PR regression claim; upstream implementation gap | Characterized as upstream HF-vs-SGLang gap. |
| Image/video TP1 vs TP2 tensors | n/a | reference | mean cosine about `0.99998` | n/a | tensor drift attributed to fp16 TP reduction order plus nonlinear amplification | Characterized, not a task-quality benchmark. |
| Audio TP1 vs TP2 tensors | n/a | reference | strict-allclose | n/a | strict allclose after valid-token flattening | Closed for tested audio tensors. |

CI artifact lookup used `gh pr view 423 --repo sgl-project/sglang-omni` and
found an empty `statusCheckRollup`; `gh run list` for branch
`encoder-tp-plan-b-phase0` returned no runs in both `sgl-project/sglang-omni`
and `ischencheng/sglang-omni`. The local Video-AMME benchmark used checkpoint
`/data/qwen3omni`, dataset `zhaochenyang20/Video_AMME_ci`, `max_samples=50`,
`max_tokens=256`, `temperature=0.0`, `video_fps=2`,
`video_max_frames=128`, `video_max_pixels=401408`, text output only, and the
isolated compatible `qwen-vl-utils` path `/tmp/qwen_vl_utils_0011`. Main used
the upstream speech server with local encoders and
`thinker_mem_fraction_static=0.80`; PR TP1/TP2 used SGLang encoders under
`--layout colocated-2gpu`,
`image/audio` budgets `8/1 GiB`, and the same decoding/media config. The c1
run remains the primary task-quality comparison. The 2026-05-29 c1 rerun
reproduced the same scores with mean latencies `2.371s` (main), `2.436s` (PR
TP1), and `6.282s` (PR TP2); those speed numbers are reported as benchmark
metadata, not as a TP2 speed claim. The 2026-05-29 c4 post-cap rerun is kept
as historical evidence for the temporary cap. The later c4 post-guard rerun
used the same PR TP2 colocated command shape without an explicit
`--encoder-max-batch-size`; each encoder scheduler kept `max_batch_size=32`,
the whole-GPU guard deferred unsafe image candidates, and the run completed all
`50/50` requests.

## Memory Model

The current memory evidence is the H100 encoder-only sweep in
[`h100_encoder_only_video_memory_20260529`](/data/encoder_tp_evidence_20260526/h100_encoder_only_video_memory_20260529)
plus the TP4 encoder-only follow-up in
[`h100_encoder_only_video_tp4_slope_20260529`](/data/encoder_tp_evidence_20260526/h100_encoder_only_video_tp4_slope_20260529)
and the colocated-2GPU slope rerun in
[`h100_colocated_video_memory_slope_20260529_budget10`](/data/encoder_tp_evidence_20260526/h100_colocated_video_memory_slope_20260529_budget10).
It should be read in this order:

1. **Encoder-only max video capacity:** TP1 on one H100 and TP2 on two H100s
   both completed image/video encoder forward through the tested `2048` frame
   cap. This did not show a TP2-only maximum-length win; the boundary was not
   reached.
2. **Runtime NVML peak slope:** on the same encoder-only path,
   TP2 max-rank peak has lower length slope than TP1 when fit against actual
   pre-merge visual tokens (`36.222` vs `47.061 MiB / 1k tokens`). TP2 sum
   peak is higher and is only a total-cluster reference.
3. **TP4 encoder-only follow-up:** on the same encoder-only image/video path,
   TP4 max-rank peak is lower than TP1 at every measured point and token slope
   is lower (`32.254` vs `45.517 MiB / 1k tokens`). TP4 summed rank peak is
   higher (`128.822` vs `45.517 MiB / 1k tokens`) and is only a
   total-cluster reference.
4. **Colocated runtime slope:** with the full colocated-2GPU server,
   `encoder_max_batch_size=1`, and `64..160` frame caps, TP2 image max-rank
   slope is not lower (`54.938` vs `52.625 MiB/frame`). Whole-GPU max is
   dominated by resident thinker/talker placement and should not be treated as
   an encoder runtime peak model.
5. **Admission math:** `cost(tp) = multiplier * (replicated_bytes +
   sharded_bytes / tp)` remains the admission guard and is not a runtime peak
   model.
6. **Caveats:** colocated/E2E peak can still be dominated by resident thinker
   memory, CPU-SHM-to-entry-rank H2D lift, metadata/tensor fan-out, follower
   allocation, allocation handshake, rank0/follower staging, allocator cache,
   and dynamic batching.

The PR wording should therefore be precise: TP2 lowers the modeled per-rank
activation admission ceiling. In encoder-only video forward, it also lowered
the measured max-rank runtime slope in the 128-2048 frame-cap sweep. It does
the same for the TP4 follow-up. It does not lower the measured colocated image
max-rank runtime slope in the 64-160 frame-cap rerun. It does not claim lower
total cluster memory, and it does not claim E2E memory is lower.

## Accuracy

- Audio no-truncation API/probe focused checks passed:
  `python -m py_compile ...` for the touched request/probe/preprocessor files,
  then `pytest -q tests/test_encoder_tp_e2e_probe.py
  tests/unit_test/serve/test_openai_api.py` -> 17 passed.
- Two-GPU colocated launcher/startup checks passed:
  `pytest -q tests/test_encoder_tp_launcher.py tests/test_encoder_tp_e2e_probe.py
  tests/unit_test/serve/test_openai_api.py tests/unit_test/pipeline/test_topology.py`
  -> 67 passed, 2 warnings.
- `tests/test_encoder_tp_parity_gpu.py -m slow` passed:
  7 passed in 11.60s.
- TP=2 parity harness passed on GPUs 2 and 3:
  `rank0_rc=0 rank1_rc=0`.
- Non-model unit suite passed:
  `pytest -q tests -m "not slow and not benchmark and not docs" --ignore=tests/test_model`
  → 789 passed, 14 deselected, 4 warnings.
- Current focused non-GPU suite:
  `timeout 240 python -m pytest -q tests/test_encoder_*.py tests/test_parity_compare.py tests/test_video_preprocessing.py -k 'not parity_gpu'`
  → 145 passed, 7 deselected, 4 warnings.
- TP=1 vs TP=2 comparison for `image_embeds`:
  mean per-token cosine similarity `0.999982`, minimum `0.986150`.
  Strict `atol=1e-3, rtol=1e-3` still fails because TP changes fp16
  reduction order. No token has max absolute difference above `1.0`.
- Video TP=1 vs TP=2 comparison for `video_embeds`: max abs `0.2383`, mean abs
  `0.0010`, mean cosine `0.999980`, no element above `1.0`.
- Audio local-vs-SGLang and TP=1-vs-TP=2 comparisons are strict-allclose after
  flattening the upstream audio adapter output to valid audio tokens:
  TP max abs `0.0004`, mean cosine `0.999999`.

Parity / quality scope:

| Scope | Conclusion | Evidence status |
| --- | --- | --- |
| Wrapper vs bare SGLang encoder | Bit-equal. The wrapper does not introduce tensor drift when it calls the same upstream SGLang encoder module. | Closed by tensor parity harness. |
| HF local encoder vs SGLang encoder | Drift is an upstream implementation gap, not introduced by the wrapper. | Characterized, not fixed by this PR. |
| Image/video TP1 vs TP2 | Small drift is expected from fp16 TP reduction order and nonlinear amplification through the visual stack; mean cosine remains about `0.99998`. | Tensor-level parity characterized. |
| Audio TP1 vs TP2 | Strict-allclose after valid-token flattening. | Closed for tested audio tensors. |
| Task-level Video-AMME CI-50 quality | At concurrency 1, main `72%`, PR TP1 `68%`, PR TP2 `72%`, all `0` failed and above the `66%` CI threshold. | One formal benchmark lane. TP2 matches main here; TP1 passes threshold but is `-4 pp` vs main. |
| Full task-level quality | Not fully closed. MMMU, Video-MME, MMSU, and larger/more concurrent quality sweeps were not run in this evidence pass. | Do not claim full task-quality parity from these artifacts alone. |

## PR-Branch E2E Evidence

Primary A/B evidence keeps the input payload, commit, two-GPU colocated
placement, AR memory fraction, context setting, and typed encoder activation
budget fixed; only encoder TP is changed. The TP1 failures below are the
guarded form of the OOM scenario: without TP, the single-rank encoder
activation estimate is above the configured budget, while TP2 moves the same
request under budget and completes generation. The two-GPU layout maps
`image_encoder` and `audio_encoder` TP ranks to visible GPUs `[0, 1]`, places
`thinker` on visible GPU `0`, and places `talker_ar` / `code2wav` on visible
GPU `1`; in these runs visible GPUs `0/1` are physical GPUs `5/6`.

| Workload | Encoder TP | Budget | Cost vs cap | Result | Artifact |
| --- | ---: | ---: | --- | --- | --- |
| 256-frame video from the 2049-frame looped MP4 (`video_fps=30`, `video_max_pixels=401408`) | image/audio TP=1 | 8 GiB | `9542041600 > 8589934592` | HTTP 500 admission rejection, health before/after OK; no encoder forward | `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_video256_colocated2gpu_budget8_ctx65536` |
| same 256-frame video | image/audio TP=2 | 8 GiB | `7633633280 < 8589934592` | success, 46,608 prompt tokens, 55.61s, health before/after OK | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_video256_colocated2gpu_budget8_ctx65536` |
| 300s WAV with `audio_truncation=false` | image/audio TP=1 | 0.12 GiB | `157272000 > 128849018` | HTTP 500 admission rejection, health before/after OK; no encoder forward | `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_audio300_notrunc_colocated2gpu_budget012_ctx65536` |
| same 300s WAV with `audio_truncation=false` | image/audio TP=2 | 0.12 GiB | `117336000 < 128849018` | success, 3,916 prompt tokens, 1.94s, health before/after OK | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio300_notrunc_colocated2gpu_budget012_ctx65536` |

Server-log evidence:

- Video TP1:
  `/data/encoder_tp_evidence_20260526/pr-tp1-server-colocated2gpu-budget8-ctx65536-video256.log`
  records `RuntimeError: encoder request cost 9542041600 exceeds
  max_single_request_cost=8589934592`.
- Video TP2:
  `/data/encoder_tp_evidence_20260526/pr-tp2-server-colocated2gpu-budget8-ctx65536-video256.log`
  records `encoder_admission stage=image_encoder decision=admit batch_size=1
  batch_cost=7633633280 max_batch_cost=8589934592`; rank0/rank1 image forward
  is about `2212-2217 ms`.
- Audio TP1:
  `/data/encoder_tp_evidence_20260526/pr-tp1-server-colocated2gpu-budget012-ctx65536-audio300-notrunc.log`
  records `RuntimeError: encoder request cost 157272000 exceeds
  max_single_request_cost=128849018`.
- Audio TP2:
  `/data/encoder_tp_evidence_20260526/pr-tp2-server-colocated2gpu-budget012-ctx65536-audio300-notrunc.log`
  records `encoder_admission stage=audio_encoder decision=admit batch_size=1
  batch_cost=117336000 max_batch_cost=128849018`; rank0/rank1 audio forward
  is about `364 ms`.

Unless otherwise noted, runs below use branch `encoder-tp-plan-b-all`, commit
`15b5e7c94ddccd0325f061e7500377ddfccd0434`, dirty worktree containing the PR
implementation, model `/data/qwen3omni`, encoder activation budget `10 GiB`,
AR `mem_fraction_static=0.45`, and no `encoder_mem_reserve`.

| Case | Encoder TP | Media | Prompt tokens | Result | Latency | Artifact |
| --- | ---: | --- | ---: | --- | ---: | --- |
| short video+audio | image/audio TP=1 | `tests/data/draw.mp4` + `tests/data/query_to_cars.wav` | 3536 | success, health OK | 8.37s | `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_short_video_audio_mem045` |
| short video+audio | image/audio TP=2 | same | 3536 | success, health OK | 9.04s | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_short_video_audio_mem045` |
| long video, 128-frame cap | image/audio TP=1 | `tests/data/draw.mp4`, `video_fps=16`, `video_max_frames=128`, `video_max_pixels=401408` | 18944 | success, health OK | 13.70s | `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_long_video128_mem045_ctx32768` |
| long video, 128-frame cap | image/audio TP=2 | same | 18944 | success, health OK | 21.84s | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_long_video128_mem045_ctx32768b` |
| stress video, 196-frame cap | image/audio TP=1 | `tests/data/draw.mp4`, `video_fps=30`, `video_max_frames=196`, `video_max_pixels=401408` | 35688 | success, health OK | 22.85s | `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_video196_mem045_ctx65536_sampled` |
| stress video, 196-frame cap | image/audio TP=2 | same | 35688 | success, health OK | 35.44s | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_video196_mem045_ctx65536_sampled` |
| long audio, 30s | image/audio TP=1 | `/data/encoder_tp_evidence_20260526/long_query_to_cars_30s.wav` | 406 | success, health OK | 1.24s | `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_long_audio30_mem045_ctx65536_sampled` |
| long audio, 30s | image/audio TP=2 | same | 406 | success, health OK | 1.57s | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_long_audio30_mem045_ctx65536_sampled` |

Extended TP2 long-sequence stress used
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`,
`--thinker-max-seq-len 524288`, `image/audio TP=2`,
`mem_fraction_static=0.45`, encoder activation budget `10 GiB`, and media
`/data/encoder_tp_evidence_20260526/media/draw_loop_2048frames.mp4`
(`3840x2160`, about 30 fps, `68.35s`, `2049` frames). The thinker AR process
accepted the requested `context_length=524288`, but its actual KV allocation was
`#tokens: 64415`; therefore the `512+` rows below are encoder admission-safety
evidence, not end-to-end generation successes.

| Video cap | Prompt tokens | Encoder cost vs cap | Result | Latency | Process-local peak/delta memory | Artifact |
| ---: | ---: | --- | --- | ---: | --- | --- |
| 256 frames | 46608 | `7633633280 < 10737418240` | success, health OK | 57.74s | image GPU1 `2128 -> 10600 MiB` (`+8472 MiB`); image GPU2 `2128 -> 12788 MiB` (`+10660 MiB`); thinker GPU4 `65998 -> 67042 MiB` (`+1044 MiB`) | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video256_budget10_ctx524288` |
| 512 frames | n/a | `13086228480 > 10737418240` | HTTP 500 admission rejection, health OK; no encoder forward | 43.91s | image GPU1 `10600 -> 12474 MiB` (`+1874 MiB`) during preprocessing/request staging | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video512_budget10_ctx524288` |
| 1024 frames | n/a | `15099494400 > 10737418240` | HTTP 500 admission rejection, health OK; no encoder forward | 61.58s | image GPU1 `12474 -> 14636 MiB` (`+2162 MiB`) during preprocessing/request staging | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video1024_budget10_ctx524288` |
| 2048 frames | n/a | `24159191040 > 10737418240` | HTTP 500 admission rejection, health OK; no encoder forward | 112.94s | image GPU1 `14636 -> 18094 MiB` (`+3458 MiB`) during preprocessing/request staging | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video2048_budget10_ctx524288` |

Long-audio follow-up used
`/data/encoder_tp_evidence_20260526/media/query_to_cars_300s.wav`; `soundfile`
confirmed `14,400,000` samples at `48 kHz` (`300.0s`). The default API request
succeeded but did not create a longer audio encoder workload; the new
`audio_truncation=false` request did:

| Audio file | Processor output | Encoder cost vs cap | Result | Artifact |
| --- | --- | --- | --- | --- |
| 30s looped WAV | `input_ids=(1, 406)`, `input_features=(1, 128, 3000)`, mask sum `3000` | `11733600 < 10737418240` on TP2 warmed audio | success | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio30_budget10_ctx65536_warmed_process` |
| 300s looped WAV, default | `input_ids=(1, 406)`, `input_features=(1, 128, 3000)`, mask sum `3000` | `11733600 < 10737418240` | success, 406 prompt tokens, 2.15s, health OK | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio300_budget10_ctx524288` |
| 300s looped WAV, `audio_truncation=false` | `input_ids=(1, 3916)`, `input_features=(1, 128, 30000)`, mask sum `30000` | `117336000 < 10737418240` | success, 3916 prompt tokens, 2.99s, health OK | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio300_notrunc_budget10_ctx65536` |

The processor evidence is from a local `AutoProcessor.from_pretrained` run:
the loaded Whisper feature extractor reports `chunk_length=30`,
`n_samples=480000`, and `nb_max_frames=3000`; passing
`audio_kwargs={"truncation": False}` produces `30000` feature frames for the
300s file. The public API/probe path now exposes this as
`audio_truncation=false` / `--audio-no-truncation`.

Additional H100 rerun, 2026-05-28:

- Hardware: clean H100 80GB GPUs 6 and 7. Each was at `4 MiB` used, `0%`
  util, and had no compute app before the fresh server runs.
- Server layout: `CUDA_VISIBLE_DEVICES=6,7`, `--layout colocated-2gpu`,
  image/audio encoder TP2, image activation budget `14.2 GiB`, audio
  activation budget `1.2 GiB`, encoder resident fraction `0.01`, thinker
  `mem_fraction_static=0.78`, talker `mem_fraction_static=0.12`, and
  `thinker_max_seq_len=131072`.
- Placement validated GPU6 as `resident=63.72GiB`, `dynamic=15.40GiB`,
  `limit=79.65GiB`; thinker KV capacity was still only `47893` tokens because
  the colocated thinker process used `62.69 GiB`.
- Video requests intentionally did **not** pass `video_max_pixels`; only
  `video_fps=30` and the frame cap were set. Audio requests used
  `audio_truncation=false`.

| H100 clean-GPU setting | Admission / encoder status | End result | GPU6 peak | GPU7 peak | Artifact |
| --- | --- | --- | ---: | ---: | --- |
| video1024 | image admitted: `15099494400 < 15247133900`; image rank0 OOM during forward, tried another `810 MiB` with only `466.81 MiB` free in the OOM report | HTTP 500, health false | `78753 MiB` | `32552 MiB` | `/data/encoder_tp_evidence_20260526/h100_gpu6_7/tp2_video1024_default_pixels/h100_gpu6_7_tp2_video1024_default_pixels` |
| video512 | image admitted: `13086228480 < 15247133900`; image encoder forward completed | rejected before thinker scheduling: `input_tokens=79887`, `required_tokens=79903`, `kv_capacity=47892`; health OK | `80835 MiB` | `30212 MiB` | `/data/encoder_tp_evidence_20260526/h100_gpu6_7/tp2_video512_default_pixels/h100_gpu6_7_tp2_video512_default_pixels` |
| audio3000 | audio admitted: `1173360000 < 1288490188`; audio encoder forward completed | success, `39017` prompt tokens, `9.389s`, health OK | `75643 MiB` | `20952 MiB` | `/data/encoder_tp_evidence_20260526/h100_gpu6_7/tp2_audio3000_no_trunc/h100_gpu6_7_tp2_audio3000_no_trunc` |
| video512 + audio3000 | audio admitted and completed; image admitted, then image rank0 OOM during forward, tried another `1.03 GiB` while audio rank0 held `8.50 GiB` | HTTP 500, health false | `80193 MiB` | `32754 MiB` | `/data/encoder_tp_evidence_20260526/h100_gpu6_7/tp2_video512_audio3000_default_pixels_no_trunc/h100_gpu6_7_tp2_video512_audio3000_default_pixels_no_trunc` |

This H100 rerun answers the "was TP2 peak high because another user shared the
GPU?" concern for these payloads: the runs began on clean GPUs, but TP2 still
approaches or hits the 80GB ceiling when no video pixel cap is supplied and the
encoder ranks are colocated with the thinker. The cause is not external
contention in this rerun; it is colocated resident memory plus the default-pixel
video activation/cache footprint. The only fully successful long-input case in
this clean H100 matrix is `audio3000` with `audio_truncation=false`.

Earlier backup same-payload/same-commit activation-budget A/B after adding
TP-aware per-rank admission cost and defaulting the single-request guard to the
typed encoder activation budget:

| Case | Encoder TP | Budget | Cost vs cap | Result | Artifact |
| --- | ---: | ---: | --- | --- | --- |
| stress video, 196-frame cap | image/audio TP=1 | 6 GiB | `7305625600 > 6442450944` | HTTP 500 admission rejection, health before/after OK; no encoder forward | `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_video196_budget6_ctx65536` |
| stress video, 196-frame cap | image/audio TP=2 | 6 GiB | `5844500480 < 6442450944` | success, 35,688 prompt tokens, 33.22s, health before/after OK | `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_video196_budget6_ctx65536` |

Server logs:

- TP1: `/data/encoder_tp_evidence_20260526/pr-tp1-server-budget6-ctx65536.log`
  records `RuntimeError: encoder request cost 7305625600 exceeds
  max_single_request_cost=6442450944`.
- TP2: `/data/encoder_tp_evidence_20260526/pr-tp2-server-budget6-ctx65536.log`
  records `encoder_admission stage=image_encoder decision=admit batch_size=1
  batch_cost=5844500480 max_batch_cost=6442450944`. The requested port 8027 was
  still in use, so the launcher selected port 54165 for this run.

The 128-frame video needs `--thinker-max-seq-len 32768`; with the default
8192-token thinker/preprocessor limit it is rejected before encoder execution:
`The input (18944 tokens) is longer than the model's context length (8192
tokens)`, and the server remains healthy.

Relevant server-log lines:

- TP=1 long video: `encoder_admission stage=image_encoder decision=admit
  batch_cost=3876454400 max_batch_cost=10737418240`; prefill chunks
  `8192 + 8192 + 2560`.
- TP=2 budget-6 stress video: TP-aware per-rank admission cost
  `5844500480` with cap `6442450944`; prefill chunks
  `8192 + 8192 + 8192 + 8192 + 2920`.
- TP=1 video256 under the 8 GiB controlled budget: the same 256-frame request
  that TP2 completes is rejected with `encoder request cost 9542041600 exceeds
  max_single_request_cost=8589934592`.
- TP=2 video256 under the 8 GiB controlled budget: `encoder_admission
  stage=image_encoder decision=admit batch_size=1 batch_cost=7633633280
  max_batch_cost=8589934592`; output has `46608` prompt tokens.
- TP=1 and TP=2 stress video196: `encoder_admission stage=image_encoder
  decision=admit` at the 10 GiB budget; the current TP1 cost remains
  `7305625600`, while TP2's per-rank activation cost is `5844500480`.
- TP=2 long-sequence video256: `encoder_admission stage=image_encoder
  decision=admit batch_size=1 batch_cost=7633633280
  max_batch_cost=10737418240`; rank0/rank1 forward is about `2250 ms`, total
  about `16781 ms`.
- TP=2 long-sequence video512/video1024/video2048: admission rejects before
  encoder forward at costs `13086228480`, `15099494400`, and `24159191040`
  against the same `10737418240` cap; health remains OK after each rejection.
- PR stress video196 thinker KV capacity: both TP1 and TP2 allocate `66804`
  tokens at `mem_fraction_static=0.45`, enough for the 35,704-token request
  that upstream-main rejects at `kv_capacity=23257`.
- TP=1 long audio: `encoder_admission stage=audio_encoder decision=admit
  batch_cost=15727200 max_batch_cost=10737418240`.
- TP=2 warmed long audio: `encoder_admission stage=audio_encoder
  decision=admit batch_cost=11733600 max_batch_cost=10737418240`; the padded
  input/mask bytes remain replicated and the output activation proxy is
  per-rank.
- TP=2 audio300: same `batch_cost=11733600` as audio30 because the public
  processor path clamps both files to `3000` audio feature frames.
- TP=2 audio300 with `audio_truncation=false`: `encoder_admission
  stage=audio_encoder decision=admit batch_size=1 batch_cost=117336000
  max_batch_cost=10737418240`; rank0/rank1 forward is about `863-865 ms`,
  total about `1751 ms`; thinker prefill receives `3916` prompt tokens.
- TP=1 audio300 with `audio_truncation=false` under the 0.12 GiB controlled
  budget: the same request that TP2 completes is rejected with `encoder request
  cost 157272000 exceeds max_single_request_cost=128849018`.
- TP=2 audio300 with `audio_truncation=false` under the 0.12 GiB controlled
  budget: `encoder_admission stage=audio_encoder decision=admit batch_size=1
  batch_cost=117336000 max_batch_cost=128849018`; output has `3916` prompt
  tokens.

### Historical Latency Attribution Status

The older warmed artifacts in this section were collected before the recv path was split
into sub-timers. Their `recv_ms` is a coarse bucket that includes inbox drain /
admission, strip-and-H2D lift, metadata fan-out, follower allocation,
allocation handshake, tensor fan-out, and rank skew. They should not be used
for fine-grained attribution. The current fine-grained video attribution comes
from the H100 video128 main / PR TP1 / PR TP2 table near the top of this report.

New server logs emit:

```text
inbox_admission_ms, strip_h2d_ms, metadata_broadcast_ms,
follower_allocation_ms, allocation_handshake_ms, tensor_broadcast_ms,
rank_wait_skew_ms, rank_arrival_skew_ms, build_ms, forward_ms, slice_ms
```

`rank_wait_skew_ms` / `rank_arrival_skew_ms` require
`SGLANG_OMNI_ENCODER_TIMING_DETAIL=1`, because they add one CPU-group gather
for benchmark attribution.

Sampled one-run performance/memory evidence was collected with
`--gpu-sample-interval 0.25`. The sampler records global `nvidia-smi` values
and, for newer artifacts, `nvidia-smi --query-compute-apps` process rows. This
H200 section is retained as historical/coarse evidence only; the pre-run
snapshots show large pre-existing GPU memory pressure, so the H100 tables above
should be cited for the PR's main measured-memory and latency claims.

| Case | TP | Samples | Throughput | Relevant peak/delta memory | Encoder timing |
| --- | ---: | ---: | --- | --- | --- |
| video196 | 1 | 61 | `1562.3` total tok/s, `0.70` output tok/s | image GPU4 `73187 MiB` peak, `+7344 MiB`; thinker GPU5 `+1042 MiB` | image `recv_ms=152.010`, `forward_ms=2270.759`, `total_ms=2423.358` |
| video196 | 2 | 85 | `1074.6` total tok/s, `0.48` output tok/s | image GPU1 `133363 MiB`, `+6624 MiB`; image GPU2 `135035 MiB`, `+8300 MiB`; thinker GPU5 `+1042 MiB` | image rank0/rank1 `recv_ms≈11262`, `forward_ms≈2057`, `total_ms≈13321`; `recv_ms` is a coarse pre-forward bucket, not yet attributed |
| audio30 | 1 | 4 | `352.3` total tok/s, `25.74` output tok/s | audio GPU7 `61171 MiB` peak, `+376 MiB` | audio `recv_ms=150.905`, `forward_ms=537.338`, `total_ms=691.004` |
| audio30 | 2 | 4 | `278.9` total tok/s, `20.38` output tok/s | audio GPU4 `67381 MiB`, `+862 MiB`; audio GPU6 `69633 MiB`, `+862 MiB` | audio rank0/rank1 `recv_ms≈554`, `forward_ms≈545`, `total_ms≈1150` |

Warmed success probes were collected with one warmup request plus three
measured requests and the updated process-level sampler:

| Case | TP | Samples | Measured latency / throughput | Process-local peak/delta memory | Encoder timing |
| --- | ---: | ---: | --- | --- | --- |
| video196 warmed | 1 | 149 | mean `16.89s`, stdev `0.26s`, p50 `16.94s`, p90 `17.08s`, p95 `17.10s`, `2114.7` total tok/s | image GPU7 process `1912 -> 10742 MiB` (`+8830 MiB`); thinker GPU5 process `66012 -> 67056 MiB` (`+1044 MiB`) | measured image forwards `837.685`, `827.317`, `612.267 ms`; totals `959.669`, `977.660`, `749.699 ms` |
| video196 warmed | 2 | 238 | mean `27.91s`, stdev `0.28s`, p50 `27.87s`, p90 `28.13s`, p95 `28.17s`, `1279.4` total tok/s | image GPU1 process `2128 -> 11266 MiB` (`+9138 MiB`); image GPU2 process `2128 -> 12104 MiB` (`+9976 MiB`); thinker GPU5 process `66014 -> 67058 MiB` (`+1044 MiB`) | measured rank0/rank1 forward `≈292`, `≈289`, `≈290 ms`; total `≈11513`, `≈11588`, `≈11548 ms`; coarse recv bucket `≈11217-11295 ms` |
| audio30 warmed | 1 | 11 | mean `0.745s`, stdev `0.010s`, p50 `0.740s`, p90 `0.754s`, p95 `0.755s`, `586.0` total tok/s | audio GPU6 process `1984 -> 2474 MiB` (`+490 MiB`); thinker GPU5 process `66012 -> 66130 MiB` (`+118 MiB`) | measured audio forwards `124.621`, `95.365`, `132.778 ms`; totals `210.769`, `213.002`, `192.655 ms` |
| audio30 warmed | 2 | 10 | mean `0.737s`, stdev `0.014s`, p50 `0.732s`, p90 `0.749s`, p95 `0.752s`, `586.0` total tok/s | audio GPU6 process `2588 -> 3564 MiB` (`+976 MiB`); audio GPU7 process `2588 -> 3564 MiB` (`+976 MiB`); thinker GPU5 process `66014 -> 66132 MiB` (`+118 MiB`) | measured rank0/rank1 forward `≈175`, `≈132`, `≈198-200 ms`; total `≈330`, `≈271`, `≈296 ms` |

Artifacts:

- TP1 server: `/data/encoder_tp_evidence_20260526/pr-tp1-server-budget10-ctx65536-warmed.log`
- TP1 probe: `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_video196_budget10_ctx65536_warmed_process`
- TP2 server: `/data/encoder_tp_evidence_20260526/pr-tp2-server-budget10-ctx65536-warmed.log`
- TP2 probe: `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_video196_budget10_ctx65536_warmed_process`
- TP1 audio server: `/data/encoder_tp_evidence_20260526/pr-tp1-server-budget10-ctx65536-audio-warmed.log`
- TP1 audio probe: `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_audio30_budget10_ctx65536_warmed_process`
- TP2 audio server: `/data/encoder_tp_evidence_20260526/pr-tp2-server-budget10-ctx65536-audio-warmed.log`
- TP2 audio probe: `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio30_budget10_ctx65536_warmed_process`
- TP2 long-sequence server:
  `/data/encoder_tp_evidence_20260526/pr-tp2-server-budget10-ctx524288-longseq-allow.log`
- TP2 long-sequence probes:
  `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video256_budget10_ctx524288`,
  `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video512_budget10_ctx524288`,
  `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video1024_budget10_ctx524288`,
  `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_longseq_video2048_budget10_ctx524288`,
  and `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio300_budget10_ctx524288`.
- TP2 no-truncation audio300 server:
  `/data/encoder_tp_evidence_20260526/pr-tp2-server-budget10-ctx65536-audio300-notrunc.log`
- TP2 no-truncation audio300 probe:
  `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio300_notrunc_budget10_ctx65536`
- Controlled video256 A/B:
  `/data/encoder_tp_evidence_20260526/pr-tp1-server-colocated2gpu-budget8-ctx65536-video256.log`,
  `/data/encoder_tp_evidence_20260526/pr-tp2-server-colocated2gpu-budget8-ctx65536-video256.log`,
  `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_video256_colocated2gpu_budget8_ctx65536`,
  and `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_video256_colocated2gpu_budget8_ctx65536`.
- Controlled no-truncation audio300 A/B:
  `/data/encoder_tp_evidence_20260526/pr-tp1-server-colocated2gpu-budget012-ctx65536-audio300-notrunc.log`,
  `/data/encoder_tp_evidence_20260526/pr-tp2-server-colocated2gpu-budget012-ctx65536-audio300-notrunc.log`,
  `/data/encoder_tp_evidence_20260526/e2e/pr_tp1_audio300_notrunc_colocated2gpu_budget012_ctx65536`,
  and `/data/encoder_tp_evidence_20260526/e2e/pr_tp2_audio300_notrunc_colocated2gpu_budget012_ctx65536`.

Interpretation for these older H200 capped-pixel/control runs: TP2 materially
lowers the per-rank admission cost enough to turn selected longer requests from
guarded failures into successful generations under the same activation budget.
In that older matrix, video256 moves from `9.54 GiB > 8 GiB` at TP1 to
`7.63 GiB < 8 GiB` at TP2, and no-truncation audio300 moves from
`157272000 > 128849018` at TP1 to `117336000 < 128849018` at TP2. The latest
clean-H100 default-pixel video256/video512 runs above should be cited instead
as encoder-boundary evidence, not E2E video success. The warmed video196 path
shows lower post-warmup encoder forward time at TP2 (`~290 ms` per rank vs
`~0.6-0.8 s` TP1), but end-to-end TP2 video latency is slower on that host.
Because those older video196 logs have only coarse `recv_ms`, cite the new H100
video128 three-way table for fine-grained video latency attribution.

Fine-grained latency status:

| Workload | TP1 status | TP2 status | Required evidence before attribution |
| --- | --- | --- | --- |
| video128 warmed | main E2E plus PR TP1 fine timing complete | PR TP2 fine timing complete | Closed for this both-success latency shape; TP2 video delta is metadata/admission handshake dominated. |
| video196 warmed | Existing coarse A/B only | Existing coarse A/B only | Keep as coarse historical evidence unless rerun with `SGLANG_OMNI_ENCODER_TIMING_DETAIL=1`. |
| video256 warmed | TP1 same-budget run currently rejects; no warmed success A/B | TP2 one-run success only | Rerun if a success-vs-success budget is selected, or report as long-input admission A/B only. |
| audio30 warmed | main E2E plus PR TP1 fine timing complete | PR TP2 fine timing complete | Closed for main-comparable short-audio latency. |
| audio300 `audio_truncation=false` warmed | PR TP1 fine timing complete | PR TP2 fine timing complete | Closed for PR-only long-audio latency; main is not comparable because main lacks the no-truncation override. |

The H100 clean-GPU long-input boundary rows below add TP2-only fine recv
breakdowns, but they are not TP1/TP2 latency A/Bs. Use them only as
single-config boundary diagnostics:

| H100 setting | Stage/rank | recv | inbox/admission | metadata broadcast | allocation handshake | tensor broadcast | forward | total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| video512 | image rank0 | `22111.875 ms` | `102.964 ms` | `10822.226 ms` | `11000.964 ms` | `184.735 ms` | `2009.574 ms` | `24126.634 ms` |
| video512 | image rank1 | `22111.851 ms` | `0.000 ms` | `21917.761 ms` | `1.201 ms` | `184.913 ms` | `2009.581 ms` | `24126.409 ms` |
| audio3000 | audio rank0 | `2105.744 ms` | `118.869 ms` | `908.705 ms` | `818.922 ms` | `258.158 ms` | `511.880 ms` | `2652.083 ms` |
| audio3000 | audio rank1 | `2105.814 ms` | `0.000 ms` | `1844.961 ms` | `0.974 ms` | `258.337 ms` | `512.345 ms` | `2652.404 ms` |

The apparent split between metadata broadcast and allocation handshake is
rank-role dependent in this instrumentation, so the report treats it as recv
path breakdown evidence, not proof of a single communication bottleneck.

## Upstream-Main Reproduction Evidence

A clean upstream-main worktree was created at
`15b5e7c94ddccd0325f061e7500377ddfccd0434` in
`/data/sglang-omni-upstream-main` and remained clean during probing.

Startup attempts at `--thinker-max-seq-len 32768`:

- `mem_fraction_static=0.45`:
  `/data/encoder_tp_evidence_20260526/upstream-main-server-mem045-ctx32768.log`
  fails after colocated image/audio/thinker load with
  `RuntimeError: Not enough memory. Please try to increase --mem-fraction-static`.
- `mem_fraction_static=0.55`:
  `/data/encoder_tp_evidence_20260526/upstream-main-server-mem055-ctx32768.log`
  fails with the same SGLang KV/static-memory check.
- `mem_fraction_static=0.7`:
  `/data/encoder_tp_evidence_20260526/upstream-main-server-mem070-ctx32768.log`
  binds on `127.0.0.1:8019`; thinker startup logs
  `pre_load_avail_mem=84.55987548828125`, model load `mem usage=57.06 GB`,
  KV cache `#tokens: 23280`, and `post_load_avail_mem=24.56378173828125`.

Probe results against the `0.7` upstream-main server with the installed
`qwen-vl-utils==0.0.14`:

| Case | Result | Artifact |
| --- | --- | --- |
| long video, 128-frame cap | HTTP 500 before encoder execution: `Failed to decode video path=tests/data/draw.mp4; torchcodec failed with ValueError: too many values to unpack (expected 2); torchvision failed with ValueError: too many values to unpack (expected 2)`; health before/after OK | `/data/encoder_tp_evidence_20260526/e2e/upstream_main_long_video128_mem070_ctx32768` |
| long audio, 30s | success, 406 prompt tokens, 2.81s, health before/after OK | `/data/encoder_tp_evidence_20260526/e2e/upstream_main_long_audio30_mem070_ctx32768` |

To remove the installed-package compatibility blocker without editing the clean
upstream worktree, `qwen-vl-utils==0.0.11` was installed under
`/tmp/qwen_vl_utils_0011` and prepended to `PYTHONPATH` for the upstream-main
server. That version returns the two-value video-reader API expected by clean
main.

Probe results with the isolated `qwen-vl-utils==0.0.11` override:

| Case | Result | Artifact |
| --- | --- | --- |
| 128-frame video, `thinker_max_seq_len=32768` | success, 18,944 prompt tokens, 26.00s, health OK; prefill chunks `8192 + 8192 + 2560` | `/data/encoder_tp_evidence_20260526/e2e/upstream_main_qv011_long_video128_mem070_ctx32768` |
| 196-frame video, `thinker_max_seq_len=32768` | HTTP 400 preprocessor rejection: `The input (35688 tokens) is longer than the model's context length (32768 tokens)`; health OK | `/data/encoder_tp_evidence_20260526/e2e/upstream_main_qv011_video196_mem070_ctx32768` |
| 196-frame video, `thinker_max_seq_len=65536` | HTTP 500 scheduler rejection before generation: `Request requires more tokens than the thinker KV cache can hold (input_tokens=35688, max_new_tokens=16, required_tokens=35704, kv_capacity=23257). Current mem_fraction_static is 0.700; try setting --thinker-mem-fraction-static higher.`; health OK | `/data/encoder_tp_evidence_20260526/e2e/upstream_main_qv011_video196_mem070_ctx65536` |

Interpretation: upstream-main evidence now proves colocated upstream startup is
sensitive to the single AR static/KV knob, and the 196-frame run reproduces a
long-video missing-admission/KV-capacity failure on clean main with a compatible
video utility. The PR branch succeeds on the same 196-frame request because the
thinker runs in a separate process/GPU and receives a larger effective KV pool.
This still does not prove an encoder activation OOM. The controlled PR stress
case separately proves the requested `tp=1` rejection vs `tp>1` success split
under the same typed encoder activation budget, but it is a budget-admission
reproduction rather than the exact original encoder runtime OOM.

## Error Attribution

The TP1-vs-TP2 error is not all from one source. The following current-run
ablations isolate what can and cannot be excluded:

| Probe | Result | Conclusion |
| --- | --- | --- |
| Bare upstream SGLang full model vs `SGLangEncoderRunner` at TP=1 | bare `(6042, 8192)` split into four `(6042, 2048)` tensors is `torch.equal=True` against wrapper `image_embeds` + 3 deepstack tensors | Excludes wrapper, partial-load, fused-shard dispatch, adapter slicing, and TP=1 distributed init as error sources. |
| Default TP1 vs TP2 | `image_embeds` max abs `0.2080`, mean abs `0.0008`, mean cosine `0.999982`; no token > `1.0` max diff | Production TP arithmetic is semantically aligned but not strict-allclose at `1e-3`. |
| `fp32_row_parallel` TP1 vs TP2 | mean abs drops to `0.0004`, low-cosine tail collapses, but max abs remains `0.2646` with 2 tokens > `0.1` | RowParallel fp16 local output rounding / all-reduce order is a major source, but not the only source. |
| `fp32_linear` TP1 vs TP2 | max abs drops to `0.1377`, mean abs `0.0004`, 1 token > `0.1`, mean cosine `0.999997` | Column/QKV/Gate-Up shard-local GEMMs are a secondary source. Remaining tail is from different TP kernel shapes plus downstream LayerNorm/MLP/merger amplification. |

Ruled out:

- input preprocessing and tensor placement: both lanes use the same saved
  preprocessor output and CPU `grid_thw` contract;
- checkpoint loading / wrapper logic: bare upstream and partial-loaded runner
  match bit-for-bit at TP=1;
- shape/layout/gather bugs for final tensors: token counts, dtype, and output
  distributions match, and all TP comparisons keep `tokens_with_max_abs_diff >
  1.0 == 0`.

Not fully removable without changing the production arithmetic graph:

- fp16 row-parallel reduction order and partial-output rounding;
- shard-local Column/QKV/Gate-Up GEMM rounding;
- nonlinear amplification in later LayerNorm, attention, MLP, and merger blocks.

## Performance

The current local validation covers encoder path startup/load footprint,
single-run sampled latency/peak-memory/timing evidence, warmed multi-repeat
video196 throughput with process-level NVML attribution, a fine-grained H100
main / PR TP1 / PR TP2 latency attribution for video128 and audio30, a
fine-grained PR-only audio300 latency A/B, TP2 long-video frame caps through
`2048` with safe admission/boundary behavior, and true long-audio encoder
requests using `audio_truncation=false`. It is still not a full online workload sweep. The
real SGLang encoder path loads only declared encoder submodules through
`EncoderModuleContainer`; it does not instantiate the full Qwen3-Omni thinker
or talker in encoder processes.

The launcher now validates explicit encoder activation budgets against
per-GPU dynamic headroom when total GPU memory is available. Default Qwen3-Omni
configs use conservative 2 GiB image/audio admission budgets so existing
colocated resident fractions keep startup-valid headroom. The encoder-TP
reference launcher raises those budgets to 10 GiB on dedicated encoder TP
GPU pairs. Startup validation is separate from the runtime whole-GPU guard:
runtime encoder admission now samples whole-GPU free memory and uses
cross-process in-flight reservations to avoid admitting dynamic batches whose
projected per-rank activation cost plus margins would exceed residual
headroom. The guard is still a conservative projection, not a proof that total
cluster memory drops or that every workload shape is OOM-safe.

Remaining performance follow-ups:

- optimize the measured TP2 video recv path, specifically metadata broadcast and
  allocation handshake in the H100 video128 attribution run;
- tune the new whole-GPU guard margins and fallback policy on additional
  colocated workloads; keep `encoder_max_batch_size` as an explicit
  conservative fallback when guard telemetry is unavailable or insufficient;
- optionally rerun video196/default-pixel video under the same fine timing if a
  larger latency-shape confirmation is needed;
- run a broader online workload sweep; current evidence is single-request
  warmed latency, not production concurrency.

## GPU Validation Runbook

Use `examples/encoder_tp_e2e_probe.py` to collect request/health/git/GPU
artifacts from any already-running server. The probe records:

- `request.json`;
- `git_rev_parse_head.json`, `git_status_short.json`, and `git_diff_stat.json`
  for the repo passed with `--repo`;
- `nvidia_smi_before.txt` and `nvidia_smi_after.txt`;
- `gpu_samples.jsonl` and `gpu_peak_summary.json` when GPU sampling is enabled;
- `gpu_process_peak_summary.json` from NVML compute-app rows when available;
- `health_before.json` and `health_after.json`;
- `runs.jsonl` and `summary.json` with latency and output-validity checks.

Server logs should be saved next to the probe artifacts. Encoder admission
decisions appear as log lines like:

```text
encoder_admission stage=image_encoder decision=admit batch_size=1 batch_cost=...
encoder_admission stage=audio_encoder decision=defer batch_size=... candidate_cost=...
```

Recommended artifact layout:

```bash
export MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
export ARTIFACT_ROOT=/tmp/encoder_tp_375_evidence
export LONG_VIDEO=/absolute/path/to/long_video.mp4
export LONG_AUDIO=/absolute/path/to/long_audio.wav
```

With `qwen-vl-utils==0.0.14`, clean upstream main currently fails video decode
before encoder execution because its video readers return three values while
clean main expects two. To reproduce the upstream long-video KV-capacity
failure without editing the clean worktree, install a compatible package target
and prepend it to `PYTHONPATH` for the upstream server:

```bash
python -m pip install --target /tmp/qwen_vl_utils_0011 --no-deps qwen-vl-utils==0.0.11
```

Upstream-main reproduction, using a clean worktree at the current remote main
SHA:

```bash
MAIN_SHA=$(git ls-remote https://github.com/sgl-project/sglang-omni.git refs/heads/main | awk '{print $1}')
git worktree add /tmp/sglang-omni-main "$MAIN_SHA"
cd /tmp/sglang-omni-main
PYTHONPATH=/tmp/qwen_vl_utils_0011:$PWD python examples/run_qwen3_omni_server.py \
  --model-path "$MODEL" \
  --model-name qwen3-omni \
  --mem-fraction-static 0.7 \
  --thinker-max-seq-len 65536 \
  --port 8000 \
  > "$ARTIFACT_ROOT/upstream-main-server.log" 2>&1
```

From the PR worktree, probe the upstream-main server while pointing `--repo` at
the upstream-main worktree:

```bash
python examples/encoder_tp_e2e_probe.py \
  --base-url http://127.0.0.1:8000 \
  --repo /tmp/sglang-omni-main \
  --case-id upstream-main-long-video \
  --output-dir "$ARTIFACT_ROOT" \
  --video "$LONG_VIDEO" \
  --video-fps 30 \
  --video-max-frames 196 \
  --video-max-pixels 401408 \
  --prompt "Describe the video in detail." \
  --max-tokens 16 \
  --timeout 900

python examples/encoder_tp_e2e_probe.py \
  --base-url http://127.0.0.1:8000 \
  --repo /tmp/sglang-omni-main \
  --case-id upstream-main-long-audio \
  --output-dir "$ARTIFACT_ROOT" \
  --audio "$LONG_AUDIO" \
  --prompt "Transcribe or summarize this audio." \
  --max-tokens 128 \
  --timeout 900
```

PR-branch `tp=1` baseline:

```bash
python examples/qwen3_omni_encoder_tp.py \
  --model "$MODEL" \
  --encoder-backend sglang \
  --image-tp 1 \
  --audio-tp 1 \
  --encoder-activation-budget-gib 10 \
  --thinker-mem-fraction-static 0.45 \
  --talker-mem-fraction-static 0.45 \
  --thinker-max-seq-len 32768 \
  --port 8001 \
  > "$ARTIFACT_ROOT/pr-tp1-server.log" 2>&1

python examples/encoder_tp_e2e_probe.py \
  --base-url http://127.0.0.1:8001 \
  --repo . \
  --case-id pr-tp1-long-video \
  --output-dir "$ARTIFACT_ROOT" \
  --video "$LONG_VIDEO" \
  --video-fps 16 \
  --video-max-frames 128 \
  --video-max-pixels 401408 \
  --prompt "Describe the video in detail." \
  --max-tokens 128 \
  --timeout 900

python examples/encoder_tp_e2e_probe.py \
  --base-url http://127.0.0.1:8001 \
  --repo . \
  --case-id pr-tp1-long-audio \
  --output-dir "$ARTIFACT_ROOT" \
  --audio "$LONG_AUDIO" \
  --prompt "Transcribe or summarize this audio." \
  --max-tokens 128 \
  --timeout 900
```

PR-branch `tp>1` comparison, same PR commit and same request payloads:

```bash
python examples/qwen3_omni_encoder_tp.py \
  --model "$MODEL" \
  --encoder-backend sglang \
  --image-tp 2 \
  --audio-tp 2 \
  --encoder-activation-budget-gib 10 \
  --thinker-mem-fraction-static 0.45 \
  --talker-mem-fraction-static 0.45 \
  --thinker-max-seq-len 32768 \
  --port 8002 \
  > "$ARTIFACT_ROOT/pr-tp2-server.log" 2>&1

python examples/encoder_tp_e2e_probe.py \
  --base-url http://127.0.0.1:8002 \
  --repo . \
  --case-id pr-tp2-long-video \
  --output-dir "$ARTIFACT_ROOT" \
  --video "$LONG_VIDEO" \
  --video-fps 16 \
  --video-max-frames 128 \
  --video-max-pixels 401408 \
  --prompt "Describe the video in detail." \
  --max-tokens 128 \
  --timeout 900

python examples/encoder_tp_e2e_probe.py \
  --base-url http://127.0.0.1:8002 \
  --repo . \
  --case-id pr-tp2-long-audio \
  --output-dir "$ARTIFACT_ROOT" \
  --audio "$LONG_AUDIO" \
  --audio-no-truncation \
  --prompt "Transcribe or summarize this audio." \
  --max-tokens 128 \
  --timeout 900
```

Each server run must be stopped cleanly before starting the next one. The final
report should copy the generated `summary.json` values, relevant server-log
memory/admission lines, and the before/after `nvidia-smi` snapshots into the
tables below.
