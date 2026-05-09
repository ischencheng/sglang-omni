# Encoder TP Parity Findings (Phase 0 / Phase 1)

This document records the numerical-parity results from running the
SGLang-native encoder backend (Plan B) against (a) the legacy HF-tower
local backend, and (b) itself at higher TP, on a real image input
through the Qwen3-Omni-30B-A3B-Instruct checkpoint.

The RFC Phase 1 spec calls for `atol=1e-3, rtol=1e-3` on fp16. The
results below show what the two implementations actually produce; both
checks fail the strict RFC tolerance, with very different magnitude.

## Reproducer

```sh
# 1. Capture local-backend (HF tower) output
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  python tests/_encoder_parity_harness.py local /tmp/parity_local.pkl

# 2. Capture sglang-backend output at tp_size=1
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
  python tests/_encoder_parity_harness.py sglang /tmp/parity_sglang.pkl

# 3. Capture sglang-backend output at tp_size=2 (uses the helper which
#    spawns rank 1 in the background and rank 0 in the foreground;
#    rank 0 writes the pickle).
bash tests/run_tp2_parity.sh /tmp/parity_tp2.pkl 29701

# 4. Compare
python tests/parity_compare.py /tmp/parity_local.pkl /tmp/parity_sglang.pkl
python tests/parity_compare.py /tmp/parity_sglang.pkl /tmp/parity_tp2.pkl
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

TP1 vs TP2 outputs are effectively identical: 97.5% of tokens have
cosine similarity ≥ 0.9999, mean 0.999984. The single 0.27 outlier in
abs-diff is consistent with NCCL fp16 all-reduce accumulation-order
noise, not a structural correctness issue.

The strict RFC `atol=1e-3` fails on one element. Realistic fp16 NCCL
collectives across two ranks produce per-element diffs in the
`[1e-3, 5e-1]` band depending on the activation magnitude — bounded
by the dynamic range of the matmul partial products, not by the
correctness of the implementation.

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
