#!/usr/bin/env bash
# Phase 2.2 v5 ablation — 7 datasets × 3 variants × 1 seed
set -u

GPU_ARR=(4 5)
NGPU=${#GPU_ARR[@]}

# Small datasets (skip Campbell = 1GB, can run separately)
DATASETS=(
  "Mouse_retina"
  "breast_cancer_wisconsin_original"
  "enron"
  "har"
  "iris"
  "mammographic_mass"
  "spambase"
)
SEEDS=(1)

VARIANTS=(
  "v5_1g_ste"
  "v5_1g_fixed"
  "v5_4f_fixed"
)

EPOCHS=30
SUB_DIR=v5_main

job_idx=0
for variant in "${VARIANTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      gpu=${GPU_ARR[$((job_idx % NGPU))]}
      job_idx=$((job_idx + 1))
      log_file=/tmp/v5_${variant}_${dataset}_seed${seed}.log
      echo "[v5] variant=$variant dataset=$dataset seed=$seed gpu=$gpu"
      python3 -u scripts/learnable_gate/run_v5_separate.py \
        --data_path datasets/${dataset}.npz \
        --save_dir result/learnable_gate_smoke/v5/${SUB_DIR}/${dataset}__${variant}__seed${seed} \
        --dataset_name ${dataset} \
        --variant_name ${variant} \
        --epochs ${EPOCHS} \
        --seed ${seed} \
        --gpu ${gpu} \
        --v5_gamma_mode $( [ "$variant" = "v5_4f_fixed" ] && echo "all_params_4f" || echo "one_param_scalar" ) \
        $( [ "$variant" = "v5_1g_ste" ] && echo "--mask_ratio_learnable" ) \
        > ${log_file} 2>&1 &
    done
  done
done

echo "v5: all jobs launched. Waiting..."
wait
echo "v5: all done."
