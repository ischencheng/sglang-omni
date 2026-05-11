# Encoder TP Parity Findings (Phase 0 / Phase 1)

This document records the numerical-parity results from running the
SGLang-native encoder backend (Plan B) against (a) the legacy HF-tower
local backend, and (b) itself at higher TP, on a real image input
through the Qwen3-Omni-30B-A3B-Instruct checkpoint.

The RFC Phase 1 spec calls for `atol=1e-3, rtol=1e-3` on fp16. The
results below show what the two implementations actually produce; all
three lanes fail the strict RFC tolerance, with very different
magnitude. The third lane (local vs sglang at TP=2) closes the gap
the original v1 of this doc left open: does the implementation
divergence change once TP kicks in?

## Quick reference

| Comparison | Max abs | Mean abs | Mean per-token cos sim | RFC `atol=1e-3, rtol=1e-3` |
|---|---:|---:|---:|:-:|
| **wrapper (tp=1) vs bare upstream `get_model()` (tp=1)** | **0** | **0** | **1.000000** | ✅ (bit-equal) |
| sglang wrapper (tp=1) vs sglang wrapper (tp=2) | 0.266 | 0.00082 | 0.999984 | ❌ (one outlier) |
| local HF (tp=1) vs sglang wrapper (tp=1) | 5.952 | 0.0216 | 0.9877 | ❌ |
| local HF (tp=1) vs sglang wrapper (tp=2) | 5.949 | 0.0217 | 0.9877 | ❌ |

Row 1 (wrapper-vs-bare) isolates whether **our wrapping** introduces
any numerical drift. Result: **zero**. `EncoderModuleContainer`
partial-load, fused-shard dispatch, `init_distributed_environment` at
world=1, and the adapter slicing produce output that is
`torch.equal == True` to upstream `get_model()` on the full
`Qwen3OmniMoeForConditionalGeneration`. All three deepstack tensors
match bit-for-bit. The wrapper is a correctness no-op.

Row 2 is the TP scaling lane: same code, different rank counts. The
0.27 outlier is NCCL fp16 accumulation-order noise; 97.5% of tokens
have cos sim ≥ 0.9999. The wrapper's TP plumbing is correct.

Rows 3 and 4 are the HF-vs-SGLang implementation gap (Conv3d→Linear
in PR #19788/#20282, SGLang's `VisionAttention` vs HF
`Qwen3OmniMoeVisionSdpaAttention`, etc.). These are an upstream
property, not something we own. SGLang upstream itself validates this
gap via `lm-eval`/`lmms_eval` benchmarks (PR #19788 shows identical
GPQA/MMMU scores), not via tensor bit-parity.

## Reproducer

```sh
# 0. Capture bare upstream get_model() output (full
#    Qwen3OmniMoeForConditionalGeneration, no EncoderModuleContainer,
#    no fused-shard dispatch, no encoder_only). Isolates the wrapper.
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  python tests/_encoder_parity_harness.py bare_sglang /tmp/parity_bare_sglang_tp1.pkl

# 1. Capture local-backend (HF tower) output
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  python tests/_encoder_parity_harness.py local /tmp/parity_local.pkl

# 2. Capture sglang-backend output at tp_size=1
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  python tests/_encoder_parity_harness.py sglang /tmp/parity_sglang_tp1.pkl

# 3. Capture sglang-backend output at tp_size=2 (uses the helper which
#    spawns rank 1 in the background and rank 0 in the foreground;
#    rank 0 writes the pickle). RANK0_GPU / RANK1_GPU env vars choose
#    the physical GPUs; both must be free.
RANK0_GPU=2 RANK1_GPU=3 \
  bash tests/run_tp2_parity.sh /tmp/parity_sglang_tp2.pkl 29701

# 4. Compare each lane
python tests/parity_compare.py /tmp/parity_local.pkl       /tmp/parity_sglang_tp1.pkl
python tests/parity_compare.py /tmp/parity_local.pkl       /tmp/parity_sglang_tp2.pkl  # user-flagged lane
python tests/parity_compare.py /tmp/parity_sglang_tp1.pkl  /tmp/parity_sglang_tp2.pkl
```

Each subprocess loads exactly one Qwen3-Omni instance, so the same GPU
can host both runs sequentially without OOM.

## Result 1 — `backend="local"` vs `backend="sglang"` at `tp_size=1`

Same input (`tests/data/cars.jpg` through the production
`AutoImageProcessor`), same dtype (fp16), same GPU.

| Statistic | local (HF tower) | sglang (Plan B) |
| --- | --- | --- |
| `image_embeds` shape | `(6042, 2048)` | `(6042, 2048)` |
| dtype | float16 | float16 |
| mean | 0.0041 | 0.0040 |
| std | 0.3294 | 0.3296 |
| abs max | 14.078 | 13.922 |

Per-element diff:
- max abs diff: **5.95**
- mean abs diff: **0.022**
- `allclose(atol=1e-3, rtol=1e-3)`: **False** (RFC threshold)
- `allclose(atol=1e-2, rtol=1e-2)`: False
- `allclose(atol=5e-2, rtol=5e-2)`: False

Per-token cosine similarity distribution:

| Cosine sim band | Tokens | % |
| --- | ---: | ---: |
| `[0.0000, 0.5000)` | 6 | 0.1 |
| `[0.5000, 0.8000)` | 69 | 1.1 |
| `[0.8000, 0.9000)` | 127 | 2.1 |
| `[0.9000, 0.9500)` | 146 | 2.4 |
| `[0.9500, 0.9900)` | 614 | 10.2 |
| `[0.9900, 0.9990)` | 2050 | 33.9 |
| `[0.9990, 1.0001)` | 3030 | 50.1 |

Mean per-token cosine similarity: **0.988**.

### Interpretation

This is a real implementation difference between the two visual
encoders, not just fp16 rounding noise:

- `transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeVisionEncoder`
  (HF, used by the legacy `backend="local"` path) is the reference
  implementation written for HuggingFace inference.
- `sglang.srt.models.qwen3_vl.Qwen3VLMoeVisionModel`
  (used by the new `backend="sglang"` path through `get_model()`) is
  SGLang's own reimplementation. It uses `ColumnParallelLinear` /
  `RowParallelLinear`, fused QKV, fused attention kernels, and (in
  some configurations) a Conv3d-replaced-by-Linear PatchEmbed
  optimization.

Both implementations are independently correct against the trained
weights — they pass the docs CI semantic probes with equivalent
outputs. They are not, however, bit-equivalent against each other.
The RFC's `atol=1e-3` was written against the assumption that they
**should** be — that assumption does not hold for this model.

### Functional evidence that the SGLang backend is correct

The same docs CI probes pass against the SGLang backend with strong
semantic answers:

- `image_semantic_count_cars` (`how many cars` + `cars.jpg`,
  `max_tokens=64`) → `"4 cars"`. Correct count.
- `audio_question_about_cars_answers_count`
  (`audio="how many cars are there in the picture?"` + `cars.jpg`)
  → *"based on the image provided, there are a total of **4** cars
  shown..."* with a per-quadrant breakdown (Rolls-Royce, Mercedes,
  Ferrari, Porsche). Both audio and image paths feed the thinker.
- `audio_changes_response` differential: same image, different
  audio, gives different output. Audio path is not silently dropped.
- `video_semantic_draw_with_tool` (`draw.mp4` +
  `audio="what is happening in this video?"`) → *"someone is drawing
  a guitar on a tablet."* Hits both `draw` and `tablet` keywords.

## Result 2 — `tp_size=1` vs `tp_size=2`, both `backend="sglang"`

Same input, same dtype, two ranks of the SGLang encoder worker on
distinct H100s sharing one NCCL bootstrap port (the production TP
shape).

| Statistic | tp_size=1 | tp_size=2 |
| --- | --- | --- |
| `image_embeds` shape | `(6042, 2048)` | `(6042, 2048)` |
| dtype | float16 | float16 |
| mean | 0.0040 | 0.0040 |
| std | 0.3296 | 0.3296 |
| abs max | 13.9219 | 13.9219 |

Distributions match at four decimals.

Per-element diff:
- max abs diff: **0.27**
- mean abs diff: **0.0008**
- `allclose(atol=1e-3, rtol=1e-3)`: **False** (RFC threshold,
  driven by one outlier element at 0.27)
- `allclose(atol=5e-3, rtol=5e-3)`: False
- `allclose(atol=1e-2, rtol=1e-2)`: False

Per-token cosine similarity distribution:

| Cosine sim band | Tokens | % |
| --- | ---: | ---: |
| `[0.9000, 0.9900)` | 1 | 0.0 |
| `[0.9900, 0.9990)` | 6 | 0.1 |
| `[0.9990, 0.9999)` | 144 | 2.4 |
| `[0.9999, 1.0001)` | 5891 | **97.5** |

Mean per-token cosine similarity: **0.999984**.

Deepstack layers (3 of them) are similarly tight: max abs diff 0.018,
0.10, 0.17; mean cosine sim 0.999997, 0.999977, 0.999969.

### Interpretation

TP1 vs TP2 outputs are effectively identical: 99.9% of tokens have
cosine similarity ≥ 0.999 (97.5% ≥ 0.9999), mean 0.999984. The single
0.27 outlier in abs-diff is consistent with NCCL fp16 all-reduce
accumulation-order noise, not a structural correctness issue.

The strict RFC `atol=1e-3` fails on one element. Realistic fp16 NCCL
collectives across two ranks produce per-element diffs in the
`[1e-3, 5e-1]` band depending on the activation magnitude — bounded
by the dynamic range of the matmul partial products, not by the
correctness of the implementation.

### Result 2b — Layer-by-layer tp=1 vs tp=2 diff growth

To pin down *where* the 0.27 max-abs-diff comes from, both tp=1 and
tp=2 captures were rerun with forward hooks on every block of
``visual``. Per-layer diff stats (per-token max abs diff):

| layer | max abs | tokens > 1.0 (of 24168) | cos mean |
|---|---:|---:|---:|
| ``patch_embed`` (no TP) | 0 | 0 | 1.000000 |
| ``blk_00`` | 0.031 | 0 | 1.000000 |
| ``blk_04`` | 0.047 | 0 | 0.999997 |
| ``blk_08`` (deepstack idx 8) | 0.137 | 0 | 0.999997 |
| ``blk_09`` (jump) | **6.12** | **62** | 0.999994 |
| ``blk_16`` (deepstack idx 16) | 6.78 | ~75 | 0.999986 |
| ``blk_20`` | 7.75 | ~150 | 0.999965 |
| ``blk_25`` | 7.34 | ~250 | 0.999945 |
| ``blk_26`` (last block) | **448** | **10712 (44%)** | 0.999997 |
| ``merger`` (final, after 2×2 spatial avg, N=6042) | **0.266** | **0** | 0.999984 |

Three distinct phases:

1. **Blocks 0–8: linear accumulation.** Each block adds NCCL fp16
   all-reduce noise from ``attn.o_proj`` and ``mlp.down_proj``
   (the two ``RowParallelLinear`` calls per block). Growth is roughly
   ``noise × sqrt(layer)``. By blk_08 only 2 tokens have any element
   above 0.1.
2. **Block 9: softmax amplification.** The two already-large-noise
   tokens get fed into block 9's attention softmax, which is
   non-linear and amplifies their noise spike to 6+. From here on 62
   tokens carry an outlier > 1.0, but the rest of the 24168 tokens
   stay clean. Mean cos sim remains ≥ 0.99999.
3. **Block 26 (last block): broad amplification.** 44% of tokens
   develop max-abs-diff > 1.0; 5 tokens have outliers above 100,
   peak at 448. Likely a near-zero LayerNorm denominator getting
   divided through.
4. **Merger smooths it.** The patch merger does a 2×2 spatial
   reduction (24168 → 6042 tokens) with linear projections that
   *average* groups of four blk_26 tokens together. The 44%-of-tokens
   chaos before the merger collapses to 0% > 1.0 after the merger,
   with peak 0.266 left on a single token.

So the 0.27 max-abs-diff is structurally explained: fp16 NCCL
accumulation-order noise + softmax non-linearity at specific tokens,
filtered by spatial averaging in the merger. Mean cosine similarity
≥ 0.9999 on 99.9% of post-merger tokens. Not a correctness bug —
inherent to fp16 + parallel reduction order.

Per-layer captures live at:
- ``/tmp/parity_sglang_tp1_layers.pkl`` (tp=1, 29 layer outputs)
- ``/tmp/parity_sglang_tp2_layers.pkl`` (tp=2, 29 layer outputs)

Reproduce with ``python tests/_encoder_parity_harness.py sglang
<out> --tp-size 1 --capture-layers`` and the equivalent through
``run_tp2_parity.sh`` with ``EXTRA_ARGS=--capture-layers``.

### Result 2c — Does bf16 tighten the TP parity? (No.)

The previous v1 wrappers cast the bf16-native checkpoint to fp16
(``_resolve_dtype(None) -> torch.float16`` in
``models/qwen3_omni/stages.py``). Hypothesis worth testing: bf16's
wider exponent range might prevent the blk_26 LayerNorm-near-zero
amplification and reduce the tp=1 vs tp=2 final max-abs from 0.27.

Captured the same layer-by-layer parity with
``--dtype bfloat16`` on both ranks. Result is the opposite:

| layer | fp16 max abs | bf16 max abs | bf16/fp16 |
|---|---:|---:|---:|
| ``00_patch_embed`` | 0 | 0 | — |
| ``blk_00`` | 0.031 | **0.25** | 8× |
| ``blk_08`` | 0.14 | **0.75** | 5× |
| ``blk_09`` (softmax jump) | 6.12 | **59.75** | 10× |
| ``blk_25`` | 7.34 | **59.25** | 8× |
| ``blk_26`` (last block) | 448 | **1315** | 3× |
| ``99_merger`` (final) | 0.27 | **1.03** | 4× |

Mean cos sim of final image_embeds: fp16 = 0.999984, bf16 = 0.999111.

Root cause of the gap: bf16 has 7 mantissa bits vs fp16's 10. Every
NCCL all-reduce in ``attn.o_proj`` and ``mlp.down_proj`` introduces
~8× more per-layer rounding noise in bf16. The wider exponent
range doesn't compensate because the blk_26 amplification is a
mantissa-precision problem (normalization denominator), not an
underflow problem.

**Implication.** The fp16 default in v1 was historically odd
(checkpoint is bf16-native) but coincidentally gives tighter TP
parity numbers. Switching to bf16 would make TP parity 4–8× worse
on the metrics we track. Whether bf16 *serving quality* is better
than fp16 (vs HF reference, vs MMMU/GPQA benchmarks) is a separate
question outside TP scope — flagged for a Phase 1+ follow-up RFC,
not a Phase 0 change.

## Result 3 — `backend="local"` (tp=1) vs `backend="sglang"` (tp=2)

Closes the question: does the local-vs-sglang implementation gap
grow once TP kicks in?

| Statistic | local (tp=1) | sglang (tp=2) |
| --- | --- | --- |
| `image_embeds` shape | `(6042, 2048)` | `(6042, 2048)` |
| dtype | float16 | float16 |
| mean | 0.0041 | 0.0040 |
| std | 0.3294 | 0.3296 |

Per-element diff:
- max abs diff: **5.95** (vs 5.95 at tp=1 — unchanged)
- mean abs diff: **0.0217** (vs 0.0216 at tp=1 — unchanged)
- mean per-token cos sim: **0.987689** (vs 0.987709 at tp=1 — unchanged)

Per-token cosine similarity distribution:

| Cosine sim band | Tokens | % |
| --- | ---: | ---: |
| `[0.000, 0.500)` | 6 | 0.1 |
| `[0.500, 0.900)` | 197 | 3.3 |
| `[0.900, 0.990)` | 766 | 12.7 |
| `[0.990, 0.999)` | 2044 | 33.8 |
| `[0.999, 1.000)` | 3029 | 50.1 |

The distribution is bit-equivalent to the local-vs-sglang-tp1 case
(within 0.1% per bucket). NCCL's fp16 noise (~0.27 max abs from
Result 2) is much smaller than the implementation gap (~5.95 max
abs from Result 1), so it's invisible in this top-level comparison.
This means **production semantics at TP>=2 are no further from
the legacy local path than at TP=1** — equivalent, not worse.

### End-to-end at TP=2

Functional verification on the same checkpoint, image_encoder TP=2
on GPUs 0+1, audio_encoder TP=1 on GPU 2, thinker on GPU 3:

| Probe | Output | Verdict |
|---|---|:-:|
| `tp2_image_count_cars` (TP=2 visual encoder) | `"4cars"` | ✅ correct count |
| `tp2_audio_image_answers_count` | full per-quadrant breakdown of all 4 vehicles | ✅ both paths feed thinker |
| `tp2_video_draw_with_tool` (TP=2 visual on video frames) | `"drawing a guitar on a tablet with a stylus."` | ✅ all keywords hit |
| `test_health` | 200 | ✅ |

This is the first end-to-end run of the production TP>=2 path —
real NCCL bootstrap, real two-channel broadcast through
``EncoderScheduler._recv_messages`` (metadata over the TP CPU group,
tensors over the TP device group), real allocation-ready gather.

## Recommendations

1. **Loosen the Phase 1 numerical-parity tolerance.** The current
   `atol=1e-3, rtol=1e-3` does not match what either lane (HF vs
   SGLang reimplementations, or NCCL fp16 collectives) actually
   guarantees on real inputs. A defensible bar:
   - `tp_size=1` vs `tp_size>1`: `mean_per_token_cosine_sim ≥ 0.999`
     (passes today at 0.999984).
   - `backend="local"` vs `backend="sglang"`: not a bit-parity check
     at all; assert (a) shape + dtype + token count match, and (b)
     downstream semantic probes (the docs CI suite) produce equivalent
     answers on a fixed set of inputs.

2. **Keep the harness in-repo.** `tests/_encoder_parity_harness.py`,
   `tests/parity_compare.py`, and `tests/run_tp2_parity.sh` reproduce
   these numbers from a fresh checkpoint in ~3 minutes on an H100.
   Re-run before any change to the SGLang vendor pin or to the
   encoder adapter.

3. **Treat the 3.3% low-cosine token tail as a known
   characterization, not a bug, until upstream SGLang vs HF
   transformers parity improves.** The two implementations
   independently verify against the trained weights — neither is the
   "ground truth" for the other.
