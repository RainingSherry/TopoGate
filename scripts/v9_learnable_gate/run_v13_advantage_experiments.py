#!/usr/bin/env python3
"""Run V13 independent assignment-residual topology experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / "datasets" / "AHDPC_related_advantage"
CONFIG = ROOT / "methods/TopoGate/V11/configs/topogate_v13_advantage_residual.yaml"
# Keep experiment products in the project result tree.  The repository root is
# reserved for source, configuration, and documentation.
OUT_ROOT = ROOT / "result" / "v13_results_2026-08-03_advantage"
DATASETS = (
    "spectf_heart.npz", "vehicle.npz", "vertebral_column.npz",
    "satellite_image.npz", "statlog_image_segmentation.npz",
    "ionosphere.npz", "seeds.npz", "banknote_authentication.npz",
    "rice_dataset_cammeo_and_osmancik.npz", "wine.npz",
    "glass_identification.npz", "image_segmentation.npz",
)
VARIANTS = {
    "v13_full": {"use_topology": True, "use_dynamic_graph": True, "use_graph_prior": True, "use_edge_consistency": True},
    "v13_nomix": {"use_topology": False, "use_dynamic_graph": False, "use_graph_prior": False, "use_edge_consistency": False},
}
SEEDS = (42, 123, 7)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_npz(path: Path):
    with np.load(path) as d:
        xkey = "X" if "X" in d.files else "x" if "x" in d.files else "data"
        ykey = "y" if "y" in d.files else "labels"
        return np.asarray(d[xkey]), np.asarray(d[ykey]).ravel()

def run_one(data_file: str, variant: str, seed: int, out_root: Path) -> dict:
    source = DATA_ROOT / data_file
    name = source.stem
    output = out_root / f"{name}__{variant}__seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    row = {"dataset": name, "dataset_file": data_file, "variant": variant, "seed": seed, "source_path": str(source), "source_sha256": sha256(source)}
    try:
        X, y = load_npz(source)
        row.update({"n_samples": int(X.shape[0]), "n_features": int(X.shape[1]), "n_clusters": int(np.unique(y).size)})
        from methods.TopoGate.V11.run import run_v11
        _, elapsed, metrics = run_v11(
            X, int(np.unique(y).size), y, config_path=CONFIG, save_dir=output,
            dataset_name=name, gpu=3, seed=seed, source_path=source,
            k_protocol="benchmark_oracle_from_y", **VARIANTS[variant]
        )
        row.update({"status": "completed", "elapsed": float(elapsed), **{k: metrics.get(k) for k in ("acc","nmi","ari","fmi")}, "error": None})
    except Exception as exc:
        row.update({"status": "failed", "error": f"{exc}\n{traceback.format_exc()}"})
    (output / "run_record.json").write_text(json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return row

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; total=len(args.datasets)*len(args.variants)*len(args.seeds); i=0
    for ds in args.datasets:
        for var in args.variants:
            for seed in args.seeds:
                i+=1; print(f"[{i}/{total}] {ds} {var} seed={seed}",flush=True)
                r=run_one(ds,var,seed,args.output_dir); rows.append(r); print(f"  {r.get('status')} ARI={r.get('ari')}",flush=True)
    fields=sorted({k for r in rows for k in r})
    with (args.output_dir/"runs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Wrote {args.output_dir/'runs.csv'} rows={len(rows)}")

if __name__ == "__main__":
    main()
