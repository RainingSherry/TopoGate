#!/usr/bin/env bash
# 3 workers on 3 GPUs for v3_tune (each worker = 1 variant × 5 ds × 3 seeds)
set -u
cd /home/luolie/ToPoGate

LOGDIR=/tmp/topogate_v3_tune_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

# Worker 0: v3_conservative (lr 5x + lgm)
CUDA_VISIBLE_DEVICES=4 nohup python3 -u scripts/learnable_gate/run_v3_tune_one.py \
  --variant v3_conservative \
  --gpu 0 > "$LOGDIR/conservative.log" 2>&1 &
PID0=$!
echo "v3_conservative PID=$PID0 log=$LOGDIR/conservative.log"

# Worker 1: v3_lr3 (lr 3x + lgm)
CUDA_VISIBLE_DEVICES=5 nohup python3 -u scripts/learnable_gate/run_v3_tune_one.py \
  --variant v3_lr3 \
  --gpu 0 > "$LOGDIR/lr3.log" 2>&1 &
PID1=$!
echo "v3_lr3 PID=$PID1 log=$LOGDIR/lr3.log"

# Worker 2: v3_lr10_no_lgm (lr 10x, no lgm)
CUDA_VISIBLE_DEVICES=6 nohup python3 -u scripts/learnable_gate/run_v3_tune_one.py \
  --variant v3_lr10_no_lgm \
  --gpu 0 > "$LOGDIR/lr10_no_lgm.log" 2>&1 &
PID2=$!
echo "v3_lr10_no_lgm PID=$PID2 log=$LOGDIR/lr10_no_lgm.log"

echo "All spawned. TS=$LOGDIR"