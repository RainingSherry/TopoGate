#!/usr/bin/env bash
# Launch v11 Part A (multiseed) on 3 workers.
# GPU pool (per project rule): [1, 4, 5] — GPU 0/7 are FORBIDDEN.
set -e

REPO=/home/luolie/ToPoGate
SCRIPTS=$REPO/scripts/v9_learnable_gate

W0_DS="enron har Mouse_retina cnae9 spambase"
W1_DS="reuters breast_cancer_wisconsin_original iris mammographic_mass ISOLET"
W2_DS="sms_spam_collection first-order-theorem-proving"

export TMPDIR=/data/luolie/ToPoGate/tmp
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TMPDIR"

OUT=$REPO/result/v11_learnable_gate/logs
mkdir -p "$OUT"

run() {
  local gpu=$1; local ds=$2; local name=$3
  python $SCRIPTS/run_v11_multiseed.py --gpu "$gpu" --datasets $ds \
    > "$OUT/partA_${name}.log" 2>&1
}

echo "[$(date +%T)] v11 Part A: 13 ds × 3 seeds × 2 variants = 78 runs"
run 1 "$W0_DS" w0 > /dev/null 2>&1 &
PID0=$!
run 4 "$W1_DS" w1 > /dev/null 2>&1 &
PID1=$!
run 5 "$W2_DS" w2 > /dev/null 2>&1 &
PID2=$!
wait $PID0 $PID1 $PID2
echo "[$(date +%T)] v11 Part A done"
