#!/bin/bash
# Launch HVF + Adaptive PCA experiment with 3 workers on GPU 1/4/5
# 7 datasets × 5 configs × 3 seeds = 105 runs total
# v2_baseline and hvf2000_adaptive / full_adaptive / full_adaptive_nomix are mostly done
# (seed=7 done for small datasets, seed=42/123 for hrvatin/enron still running)
# This run: fill remaining seeds + add hvf2000_adaptive_nomix

set -e
REPO="/home/luolie/ToPoGate"
SCRIPT="$REPO/scripts/hvf_adaptive/run_hvf_adaptive_smoke.py"
OUT="$REPO/result/hvf_adaptive_pca"

mkdir -p "$OUT"

# Worker 0 → GPU 1: hrvatin (slow, still has pending seeds)
# Worker 1 → GPU 4: enron (slow, still has pending seeds) + hvf2000_adaptive_nomix on all ds
# Worker 2 → GPU 5: hvf2000_adaptive_nomix on all ds

echo "=== Worker 0 (GPU 1): hrvatin_filtered (v2_baseline seeds 123/7) ==="
CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
    --gpu 1 \
    --datasets hrvatin_filtered \
    --variants v2_baseline \
    --seeds 123 7 \
    &

echo "=== Worker 1 (GPU 4): enron (v2_baseline remaining seeds) + hvf2000_adaptive_nomix (all ds, all seeds) ==="
CUDA_VISIBLE_DEVICES=4 python "$SCRIPT" \
    --gpu 4 \
    --datasets enron \
    --variants v2_baseline \
    --seeds 123 7 \
    &

echo "=== Worker 2 (GPU 5): hvf2000_adaptive_nomix all datasets, all seeds ==="
CUDA_VISIBLE_DEVICES=5 python "$SCRIPT" \
    --gpu 5 \
    --datasets enron Mouse_retina hrvatin_filtered sms_spam_collection ISOLET Quake_Smart-seq2_Lung iris \
    --variants hvf2000_adaptive_nomix \
    --seeds 42 123 7 \
    &

echo "=== All workers launched. PIDs: $(jobs -p | tr '\n' ' ') ==="
echo "Monitor: tail -f $OUT/comparison.csv (written at end)"
