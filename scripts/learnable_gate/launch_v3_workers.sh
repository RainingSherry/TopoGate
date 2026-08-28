#!/usr/bin/env bash
# Spawn 3 workers for v3 smoke test, each on a different GPU.
# Usage: bash scripts/learnable_gate/launch_v3_workers.sh [gpus...]
set -u
cd /home/luolie/ToPoGate

GPUS="${1:-4 5 6}"
mkdir -p result/learnable_gate_smoke/v3_smoke
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=/tmp/topogate_v3_worker_${TS}
mkdir -p "$LOGDIR"
echo "$TS" > "$LOGDIR/ts"

# Read GPU ids into array
IFS=' ' read -ra GPU_ARR <<< "$GPUS"
GPU0=${GPU_ARR[0]:-4}
GPU1=${GPU_ARR[1]:-5}
GPU2=${GPU_ARR[2]:-6}

# Worker 0: baseline + v3_lgm (all seeds, all datasets)
CUDA_VISIBLE_DEVICES=$GPU0 nohup python3 -u scripts/learnable_gate/run_v3_smoke.py \
  --variants baseline v3_lgm \
  --seeds 42 123 7 \
  --gpu 0 > "$LOGDIR/w0.log" 2>&1 &
PID0=$!
echo "Worker 0 (gpu=$GPU0, baseline + v3_lgm): PID=$PID0 log=$LOGDIR/w0.log"

# Worker 1: v3_lr
CUDA_VISIBLE_DEVICES=$GPU1 nohup python3 -u scripts/learnable_gate/run_v3_smoke.py \
  --variants v3_lr \
  --seeds 42 123 7 \
  --gpu 0 > "$LOGDIR/w1.log" 2>&1 &
PID1=$!
echo "Worker 1 (gpu=$GPU1, v3_lr): PID=$PID1 log=$LOGDIR/w1.log"

# Worker 2: v3_full
CUDA_VISIBLE_DEVICES=$GPU2 nohup python3 -u scripts/learnable_gate/run_v3_smoke.py \
  --variants v3_full \
  --seeds 42 123 7 \
  --gpu 0 > "$LOGDIR/w2.log" 2>&1 &
PID2=$!
echo "Worker 2 (gpu=$GPU2, v3_full): PID=$PID2 log=$LOGDIR/w2.log"

echo "All workers spawned. Use 'bash scripts/learnable_gate/check_v3_workers.sh $TS' to monitor."
echo "TS=$TS"
