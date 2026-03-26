#!/bin/bash
# Chunked training with process restart to reset ANE firmware state.
#
# M3 Ultra ANE throughput degrades per-process after ~28K eval calls.
# This wrapper runs 40-step chunks, checkpoints, and restarts the process.
# Each fresh process gets a clean ANE connection = full 1.0s/step throughput.
#
# Usage: bash scripts/train-chunked.sh [total_steps]
set -euo pipefail

TOTAL_STEPS=${1:-10000}
CHUNK_STEPS=40
BIN="./target/release/train"
DATA_DIR="/Users/admin/rustane/data"
CKPT_DIR="/Users/admin/rustane/checkpoints/chunked"
LOG_DIR="/Users/admin/rustane/logs"
TIMESTAMP=$(date +%Y%m%d-%H%M)
LOG="${LOG_DIR}/chunked-${TIMESTAMP}.log"

MODEL="custom:1536,4096,20,512"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

echo "=== Chunked Training ===" | tee "$LOG"
echo "total_steps: $TOTAL_STEPS, chunk_size: $CHUNK_STEPS" | tee -a "$LOG"
echo "log: $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

COMPLETED=0
CHUNK_NUM=0
RESUME_ARG=""

while [ "$COMPLETED" -lt "$TOTAL_STEPS" ]; do
    REMAINING=$((TOTAL_STEPS - COMPLETED))
    THIS_CHUNK=$((REMAINING < CHUNK_STEPS ? REMAINING : CHUNK_STEPS))
    CHUNK_NUM=$((CHUNK_NUM + 1))

    echo "--- Chunk $CHUNK_NUM: steps $COMPLETED-$((COMPLETED + THIS_CHUNK)) ---" | tee -a "$LOG"

    # Find latest checkpoint for resume
    if [ "$COMPLETED" -gt 0 ]; then
        LATEST_CKPT=$(ls -t "$CKPT_DIR"/ckpt_*.bin 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            RESUME_ARG="--resume $LATEST_CKPT"
        fi
    fi

    $BIN --model "$MODEL" \
        --data "${DATA_DIR}/train.bin" \
        --val "${DATA_DIR}/val.bin" \
        --token-bytes "${DATA_DIR}/token_bytes.bin" \
        --steps "$THIS_CHUNK" \
        --lr 3e-4 --accum 1 --warmup 0 \
        --embed-lr 1.0 --beta2 0.99 \
        --loss-scale 1 --grad-clip 1 \
        --val-interval 999999 \
        --ckpt-interval "$THIS_CHUNK" \
        --ckpt-dir "$CKPT_DIR" \
        $RESUME_ARG \
        2>&1 | tee -a "$LOG" | tail -3

    COMPLETED=$((COMPLETED + THIS_CHUNK))
    echo "  [completed $COMPLETED / $TOTAL_STEPS]" | tee -a "$LOG"
done

# Final validation
echo "" | tee -a "$LOG"
echo "=== Final validation ===" | tee -a "$LOG"
LATEST_CKPT=$(ls -t "$CKPT_DIR"/ckpt_*.bin 2>/dev/null | head -1)
$BIN --model "$MODEL" \
    --data "${DATA_DIR}/train.bin" \
    --val "${DATA_DIR}/val.bin" \
    --token-bytes "${DATA_DIR}/token_bytes.bin" \
    --steps 0 \
    --val-interval 1 --val-steps 50 \
    --resume "$LATEST_CKPT" \
    2>&1 | tee -a "$LOG" | grep -E 'val_|complete'

echo "" | tee -a "$LOG"
echo "=== All $TOTAL_STEPS steps complete ===" | tee -a "$LOG"
echo "Log: $LOG"
