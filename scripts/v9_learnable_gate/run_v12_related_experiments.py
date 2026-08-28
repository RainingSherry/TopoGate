#!/usr/bin/env python3
"""Run auditable V9/V12 full-vs-nomix experiments on related datasets.

This runner only invokes TopoGate's existing learnable-gate entry point.  It
does not alter external baselines and derives K from the supplied labels solely
for benchmark evaluation.
"""
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

DATA_ROOT = ROOT / "datasets" / "AHDPC_related_advantage"
# Keep experiment products in the project result tree.  The repository root is
# reserved for source, configuration, and documentation.
OUT_ROOT = ROOT / "result" / "v12_results_2026-08-03_advantage"
SEEDS = (42, 123, 7)
DATASETS = (
    "satellite_image.npz", "statlog_image_segmentation.npz",
    "spectf_heart.npz", "ionosphere.npz", "seeds.npz",
    "banknote_authentication.npz", "rice_dataset_cammeo_and_osmancik.npz",
    "wine.npz", "vehicle.npz", "glass_identification.npz",
    "image_segmentation.npz", "vertebral_column.npz",
)

BASE = {
    "epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5,
    "mix_neighbors": 4, "warmup_epochs": 20, "ramp_epochs": 10,
    "n_top_features": 0, "knn_pca_mode": "adaptive",
    "knn_pca_dim": 2000, "config_dir": str(ROOT / "methods" / "TopoGate" / "learnable_gate" / "configs"),
    "scale_input": False, "hidden_size": 128,
}
VARIANTS = {
    "v9_full": {"variant": "learnable_gate_v9_adaptive", "gate_mode": "learned", "mix_mode": "reliability", "pseudo_weight": 0.3, "risk_adaptive_mix": False},
    "v9_nomix": {"variant": "learnable_gate_v9_adaptive", "gate_mode": "learned", "mix_mode": "none", "pseudo_weight": 0.0, "risk_adaptive_mix": False},
    "v12_full": {"variant": "learnable_gate_v12_risk_adaptive", "gate_mode": "learned", "mix_mode": "reliability", "pseudo_weight": 0.3, "risk_adaptive_mix": True, "risk_adaptive_temperature": 1.0},
    "v12_nomix": {"variant": "learnable_gate_v12_risk_adaptive", "gate_mode": "learned", "mix_mode": "none", "pseudo_weight": 0.0, "risk_adaptive_mix": True, "risk_adaptive_temperature": 1.0},
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_npz(path: Path):
    with np.load(path) as data:
        keys = data.files
        xkey = "x" if "x" in keys else "X" if "X" in keys else "data"
        ykey = "y" if "y" in keys else "Y" if "Y" in keys else "labels"
        return np.asarray(data[xkey], dtype=np.float64), np.asarray(data[ykey]).ravel()

def run_one(dataset_file: str, variant_name: str, seed: int, output_root: Path) -> dict:
    source = DATA_ROOT / dataset_file
    dataset = source.stem
    output = output_root / f"{dataset}__{variant_name}__seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    row = {"dataset": dataset, "dataset_file": dataset_file, "variant": variant_name, "seed": seed, "source": str(source), "source_sha256": None}
    try:
        x, y = load_npz(source)
        row.update({"n_samples": int(x.shape[0]), "n_features": int(x.shape[1]), "n_clusters": int(np.unique(y).size), "source_sha256": sha256(source)})
        options = dict(BASE); options.update(VARIANTS[variant_name])
        pred, elapsed, metrics = run_topogate(x, n_clusters=int(np.unique(y).size), y=y, gpu=3, seed=seed, return_metrics=True, save_dir=str(output), **options)
        np.save(output / "predictions.npy", np.asarray(pred, dtype=np.int64)); np.save(output / "labels_true.npy", y)
        row.update({"status": "completed", "elapsed": float(elapsed), **{k: metrics.get(k) for k in ("acc", "nmi", "ari", "fmi")}, "error": None})
    except Exception as exc:
        row.update({"status": "failed", "error": f"{exc}\n{traceback.format_exc()}"})
    (output / "run_record.json").write_text(json.dumps(row, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return row

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []; total = len(args.datasets) * len(args.variants) * len(args.seeds); i = 0
    for ds in args.datasets:
        for var in args.variants:
            for seed in args.seeds:
                i += 1; print(f"[{i}/{total}] {ds} {var} seed={seed}", flush=True)
                row = run_one(ds, var, seed, args.output_dir); rows.append(row); print(f"  {row.get('status')} ARI={row.get('ari')}", flush=True)
    fields = sorted({k for r in rows for k in r})
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Wrote {args.output_dir / 'runs.csv'} rows={len(rows)}")

if __name__ == "__main__":
    main()
