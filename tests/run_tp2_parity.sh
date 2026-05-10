#!/bin/bash
# Drive 2 SGLangEncoderRunner ranks on GPUs 3 and 4, share one NCCL bootstrap port.
set -e

OUT_PATH="${1:?out_path}"
PORT="${2:-29577}"

# Allocate the port (just trust caller). Each rank's process sees a single
# physical GPU as cuda:0 (CUDA_VISIBLE_DEVICES remap), exactly like the
# launcher does for production multi-process TP.

CUDA_VISIBLE_DEVICES=4 SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true \
    PYTHONPATH=/data/sglang-omni \
    python /data/sglang-omni/tests/_encoder_parity_harness.py \
        sglang /tmp/parity_tp2_rank1_dummy.pkl \
        --tp-size 2 --tp-rank 1 --nccl-port "$PORT" \
        > /tmp/parity_tp2_rank1.log 2>&1 &
RANK1_PID=$!

CUDA_VISIBLE_DEVICES=3 SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true \
    PYTHONPATH=/data/sglang-omni \
    python /data/sglang-omni/tests/_encoder_parity_harness.py \
        sglang "$OUT_PATH" \
        --tp-size 2 --tp-rank 0 --nccl-port "$PORT" \
        > /tmp/parity_tp2_rank0.log 2>&1
RANK0_RC=$?

# Wait for follower to finish (rank 0 emits the pickle; rank 1 just exits).
wait $RANK1_PID
RANK1_RC=$?

echo "rank0_rc=$RANK0_RC rank1_rc=$RANK1_RC"
exit $RANK0_RC
