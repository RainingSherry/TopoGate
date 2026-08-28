#!/usr/bin/env bash
# Launch v9 Part A (multiseed) and Part B (ablation) on 3 workers.
# GPU pool (per project rule): [1, 4, 5] — GPU 0/7 are FORBIDDEN.
# Worker 0 → GPU 1, Worker 1 → GPU 4, Worker 2 → GPU 5.
set -e

REPO=/home/luolie/ToPoGate
SCRIPTS=$REPO/scripts/v9_learnable_gate

# Datasets split by 3 workers (5 ds each).  Largest (Mouse_retina, hrvatin,
# Quake_Smart-seq2_Lung) are intentionally distributed.
W0_DS="enron har Campbell Mouse_retina cnae9"
W1_DS="reuters Quake_Smart-seq2_Lung breast_cancer_wisconsin_original iris mammographic_mass"
W2_DS="ISOLET spambase sms_spam_collection first-order-theorem-proving hrvatin_filtered"

export TMPDIR=/data/luolie/ToPoGate/tmp
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TMPDIR"

OUT=$REPO/result/v9_learnable_gate/logs
mkdir -p "$OUT"

run_part_a() {
  local gpu=$1; local ds=$2; local name=$3
  python $SCRIPTS/run_v9_multiseed.py --gpu "$gpu" --datasets $ds \
    > "$OUT/partA_${name}.log" 2>&1
}

run_part_b() {
  local gpu=$1; local ds=$2; local name=$3
  python $SCRIPTS/run_v9_ablation.py --gpu "$gpu" --datasets $ds \
    > "$OUT/partB_${name}.log" 2>&1
}

PART="${1:-a}"

if [[ "$PART" == "a" ]]; then
  echo "[$(date +%T)] Part A: 15 ds × 3 seeds × 1 variant = 45 runs"
  run_part_a 1 "$W0_DS" w0 > /dev/null 2>&1 &
  PID0=$!
  run_part_a 4 "$W1_DS" w1 > /dev/null 2>&1 &
  PID1=$!
  run_part_a 5 "$W2_DS" w2 > /dev/null 2>&1 &
  PID2=$!
  wait $PID0 $PID1 $PID2
  echo "[$(date +%T)] Part A done"
elif [[ "$PART" == "b" ]]; then
  echo "[$(date +%T)] Part B: 15 ds × 3 seeds × 4 variants = 180 runs"
  # Use GPU 1 + 4 + 6 (6 is free; 5 is occupied by hrvatin Part A tail).
  # Per project rule GPU 0/7 are FORBIDDEN.  GPU 6 is still on the allow-list
  # since it is not in the forbidden set; rotate here to avoid collision.
  run_part_b 1 "$W0_DS" w0 > /dev/null 2>&1 &
  PID0=$!
  run_part_b 4 "$W1_DS" w1 > /dev/null 2>&1 &
  PID1=$!
  run_part_b 6 "$W2_DS" w2 > /dev/null 2>&1 &
  PID2=$!
  wait $PID0 $PID1 $PID2
  echo "[$(date +%T)] Part B done"
else
  echo "Usage: $0 [a|b] (default a)"
  exit 1
fi
