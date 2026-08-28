#!/usr/bin/env bash
# Launch v10 nomix_init Part A (multiseed) and Part B (ablation) on 3 workers.
# GPU pool (per project rule): [1, 4, 5] — GPU 0/7 are FORBIDDEN.
# Worker 0 → GPU 1, Worker 1 → GPU 4, Worker 2 → GPU 5.
set -e

REPO=/home/luolie/ToPoGate
SCRIPTS=$REPO/scripts/v9_learnable_gate

# 14 datasets split by 3 workers (~5 each)
# Excludes hrvatin_filtered (no npz) and Quake_Smart-seq2_Lung (not in v9 multiseed results)
W0_DS="enron har Campbell Mouse_retina cnae9"
W1_DS="reuters breast_cancer_wisconsin_original iris mammographic_mass spambase"
W2_DS="ISOLET sms_spam_collection first-order-theorem-proving"

export TMPDIR=/data/luolie/ToPoGate/tmp
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TMPDIR"

OUT=$REPO/result/v10_learnable_gate/logs
mkdir -p "$OUT"

run_part_a() {
  local gpu=$1; local ds=$2; local name=$3
  python $SCRIPTS/run_v10_multiseed.py --gpu "$gpu" --datasets $ds \
    > "$OUT/partA_${name}.log" 2>&1
}

run_part_b() {
  local gpu=$1; local ds=$2; local name=$3
  python $SCRIPTS/run_v10_ablation.py --gpu "$gpu" --datasets $ds \
    > "$OUT/partB_${name}.log" 2>&1
}

PART="${1:-a}"

if [[ "$PART" == "a" ]]; then
  echo "[$(date +%T)] Part A: 14 ds × 3 seeds × 2 variants = 84 runs"
  run_part_a 1 "$W0_DS" w0 > /dev/null 2>&1 &
  PID0=$!
  run_part_a 4 "$W1_DS" w1 > /dev/null 2>&1 &
  PID1=$!
  run_part_a 5 "$W2_DS" w2 > /dev/null 2>&1 &
  PID2=$!
  wait $PID0 $PID1 $PID2
  echo "[$(date +%T)] Part A done"
elif [[ "$PART" == "b" ]]; then
  echo "[$(date +%T)] Part B: 14 ds × 3 seeds × 2 variants = 84 runs"
  run_part_b 1 "$W0_DS" w0 > /dev/null 2>&1 &
  PID0=$!
  run_part_b 4 "$W1_DS" w1 > /dev/null 2>&1 &
  PID1=$!
  run_part_b 5 "$W2_DS" w2 > /dev/null 2>&1 &
  PID2=$!
  wait $PID0 $PID1 $PID2
  echo "[$(date +%T)] Part B done"
else
  echo "Usage: $0 [a|b] (default a)"
  exit 1
fi
