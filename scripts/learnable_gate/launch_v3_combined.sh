#!/usr/bin/env bash
# 3 workers on 3 GPUs for v3_combined
# - Worker 0: lr_mul=3.0 (no lgm) — most conservative
# - Worker 1: lr_mul=5.0 (no lgm) — slightly bolder
# - Worker 2: lr_mul=10.0 (no lgm) — same as v3_lr but with all other v3 features
set -u
cd /home/luolie/ToPoGate

LOGDIR=/tmp/topogate_v3_combined_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

# Worker 0: lr_mul=3.0 (no lgm)
CUDA_VISIBLE_DEVICES=4 nohup python3 -u scripts/learnable_gate/run_v3_combined.py \
  --lr_mul 3.0 --gpu 0 > "$LOGDIR/lr3.log" 2>&1 &
PID0=$!
echo "v3_combined lr3 PID=$PID0 log=$LOGDIR/lr3.log"

# Worker 1: lr_mul=5.0 (no lgm)
CUDA_VISIBLE_DEVICES=5 nohup python3 -u scripts/learnable_gate/run_v3_combined.py \
  --lr_mul 5.0 --gpu 0 > "$LOGDIR/lr5.log" 2>&1 &
PID1=$!
echo "v3_combined lr5 PID=$PID1 log=$LOGDIR/lr5.log"

# Worker 2: lr_mul=10.0 (no lgm)
CUDA_VISIBLE_DEVICES=6 nohup python3 -u scripts/learnable_gate/run_v3_combined.py \
  --lr_mul 10.0 --gpu 0 > "$LOGDIR/lr10.log" 2>&1 &
PID2=$!
echo "v3_combined lr10 PID=$PID2 log=$LOGDIR/lr10.log"

echo "All spawned. TS=$LOGDIR"