#!/usr/bin/env bash
# 3 workers on 3 GPUs for v3_best
set -u
cd /home/luolie/ToPoGate

LOGDIR=/tmp/topogate_v3_best_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

# Worker 0: seed 42 only (smaller, faster)
CUDA_VISIBLE_DEVICES=1 nohup python3 -u scripts/learnable_gate/run_v3_best.py \
  --seeds 42 --gpu 0 > "$LOGDIR/seed42.log" 2>&1 &
PID0=$!
echo "v3_best seed42 PID=$PID0 log=$LOGDIR/seed42.log"

# Worker 1: seed 123
CUDA_VISIBLE_DEVICES=4 nohup python3 -u scripts/learnable_gate/run_v3_best.py \
  --seeds 123 --gpu 0 > "$LOGDIR/seed123.log" 2>&1 &
PID1=$!
echo "v3_best seed123 PID=$PID1 log=$LOGDIR/seed123.log"

# Worker 2: seed 7
CUDA_VISIBLE_DEVICES=5 nohup python3 -u scripts/learnable_gate/run_v3_best.py \
  --seeds 7 --gpu 0 > "$LOGDIR/seed7.log" 2>&1 &
PID2=$!
echo "v3_best seed7 PID=$PID2 log=$LOGDIR/seed7.log"

echo "All spawned. TS=$LOGDIR"