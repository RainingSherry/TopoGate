#!/usr/bin/env bash
# Multi-seed v5_1g_ste benchmark across 7 datasets.
# Total: 7 datasets × 3 seeds = 21 runs. Each ~30s on har, ~60s on big datasets.
set -u
cd /home/luolie/ToPoGate

GPU=4
EPOCHS=30
DATASETS=(
  "har:datasets/har.npz"
  "iris:datasets/iris.npz"
  "spambase:datasets/spambase.npz"
  "breast_cancer:datasets/breast_cancer_wisconsin_original.npz"
  "mammographic:datasets/mammographic_mass.npz"
  "enron:datasets/enron.npz"
  "Mouse_retina:datasets/Mouse_retina.npz"
)
SEEDS=(1 2 3)

OUT_DIR=result/learnable_gate_smoke/v5_multiseed
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/run_all.log"
echo "=== Multi-seed v5_1g_ste benchmark ===" > "$LOG"
date >> "$LOG"

for seed in "${SEEDS[@]}"; do
  for entry in "${DATASETS[@]}"; do
    name="${entry%%:*}"
    path="${entry##*:}"
    if [ ! -f "$path" ]; then
      echo "[$name seed=$seed] SKIP (missing file $path)" | tee -a "$LOG"
      continue
    fi
    echo "[$name seed=$seed] starting..." | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/learnable_gate/run_v5_separate.py \
      --data_path "$path" \
      --save_dir "$OUT_DIR" \
      --dataset_name "$name" \
      --variant_name "v5_1g_ste" \
      --epochs "$EPOCHS" \
      --v5_gamma_mode one_param_scalar \
      --mask_ratio_learnable \
      --mask_ratio_init 0.0 \
      --seed "$seed" 2>&1 | tail -5 | tee -a "$LOG"
  done
done

echo "=== Done ===" >> "$LOG"
date >> "$LOG"
echo "All 21 runs complete. See $LOG for full output."
