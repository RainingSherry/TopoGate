#!/usr/bin/env python3
"""Run V9 full and topology/mixing ablations on AHDPC candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "baseline" / "CLUBench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "2")

import numpy as np

from methods.TopoGate.learnable_gate.run_npz import run_topogate

DATA_ROOT = ROOT / "datasets" / "AHDPC" / "processed"
# Keep experiment products in the project result tree.  The repository root is
# reserved for source, configuration, and documentation.
OUT_ROOT = ROOT / "result" / "v9_results_2026-08-02_advantage_ablation"
SEEDS = (42, 123, 7)
DATASETS = ("balance_scale", "landsat", "spect_heart", "glass", "vehicle", "vertebral_column", "image_segment")

BASE = {
    "variant": "learnable_gate_v9_adaptive",
    "epochs": 80,
    "mask_ratio": 0.3,
    "neighbor_k": 5,
    "mix_neighbors": 4,
    "warmup_epochs": 20,
    "ramp_epochs": 10,
    "n_top_features": 0,
    "knn_pca_mode": "adaptive",
    "knn_pca_dim": 2000,
    "config_dir": str(ROOT / "methods" / "TopoGate" / "learnable_gate" / "configs"),
    "scale_input": False,
}

VARIANTS = {
    "v9_full": {"gate_mode": "learned", "mix_mode": "reliability", "pseudo_weight": 0.3},
    "v9_nomix": {"gate_mode": "learned", "mix_mode": "none", "pseudo_weight": 0.0},
    "v9_static": {"gate_mode": "topology", "mix_mode": "reliability", "pseudo_weight": 0.3},
    "v9_random": {"gate_mode": "learned", "mix_mode": "random", "pseudo_weight": 0.3},
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_npz(path: Path):
    data = np.load(path)
    x_key = "x" if "x" in data.files else "X"
    return np.asarray(data[x_key], dtype=np.float64), np.asarray(data["y"], dtype=np.int64).ravel()

def run_one(dataset: str, variant: str, seed: int, output_root: Path) -> dict:
    source = DATA_ROOT / f"{dataset}.npz"
    output = output_root / f"{dataset}__{variant}__seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    x, y = load_npz(source)
    options = dict(BASE)
    options.update(VARIANTS[variant])
    try:
        predictions, elapsed, metrics = run_topogate(
            x,
            n_clusters=int(np.unique(y).size),
            y=y,
            gpu=4,
            seed=seed,
            return_metrics=True,
            save_dir=str(output),
            **options,
        )
        np.save(output / "predictions.npy", np.asarray(predictions, dtype=np.int64))
        np.save(output / "labels_true.npy", y)
        row = {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "status": "completed",
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "n_clusters": int(np.unique(y).size),
            "acc": metrics.get("acc"),
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "fmi": metrics.get("fmi"),
            "elapsed": float(elapsed),
            "source_sha256": sha256(source),
            "error": None,
        }
    except Exception as exc:
        row = {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "status": "failed",
            "error": f"{exc}\n{traceback.format_exc()}",
        }
    (output / "run_record.json").write_text(json.dumps(row, indent=2, default=float), encoding="utf-8")
    return row

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(args.datasets) * len(args.variants) * len(args.seeds)
    index = 0
    for dataset in args.datasets:
        for variant in args.variants:
            for seed in args.seeds:
                index += 1
                print(f"[{index}/{total}] {dataset} {variant} seed={seed}", flush=True)
                row = run_one(dataset, variant, seed, args.output_dir)
                rows.append(row)
                print(f"  {row['status']} ARI={row.get('ari')}", flush=True)
    columns = sorted({key for row in rows for key in row})
    with (args.output_dir / "ablation_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_dir / 'ablation_runs.csv'}; rows={len(rows)}")

if __name__ == "__main__":
    main()
