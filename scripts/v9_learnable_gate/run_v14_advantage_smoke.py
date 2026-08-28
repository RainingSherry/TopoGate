#!/usr/bin/env python3
"""Small, auditable V14 mechanism smoke on representative advantage datasets.

This runner intentionally defaults to short single-seed engineering smoke. It
does not select a variant using labels and never passes labels into V11Trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / "datasets" / "AHDPC_related_advantage"
PAPER_ROOT = ROOT / "datasets" / "AHDPC" / "processed"
CONFIG = ROOT / "methods/TopoGate/V11/configs/topogate_v14_advantage_minimum.yaml"
# Generated outputs belong under the project result tree.  Callers may still
# override this with --output-dir for a writable local/archive target.
OUT_ROOT = ROOT / "result" / "V14" / "smoke"
DATASETS = ("balance_scale.npz", "spectf_heart.npz", "landsat.npz", "vehicle.npz", "vertebral_column.npz")
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

def run_one(data_file: str, variant: str, seed: int, out_root: Path, epochs: int) -> dict:
    source = DATA_ROOT / data_file
    if not source.exists():
        # The paper protocol names differ from the related-dataset aliases.
        # Resolve to the verified AHDPC processed files; do not synthesize or
        # copy data into the related-dataset directory.
        aliases = {
            "landsat.npz": PAPER_ROOT / "landsat.npz",
            "balance_scale.npz": PAPER_ROOT / "balance_scale.npz",
            "spectf_heart.npz": ROOT / "datasets" / "spectf_heart.npz",
            "vehicle.npz": ROOT / "datasets" / "vehicle.npz",
            "vertebral_column.npz": PAPER_ROOT / "vertebral_column.npz",
        }
        source = aliases.get(data_file, source)
    name = source.stem
    out = out_root / f"{name}__{variant}__seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    row = {"dataset": name, "dataset_file": data_file, "variant": variant, "seed": seed, "source_path": str(source), "source_sha256": sha256(source) if source.exists() else None}
    try:
        X, y = load_npz(source)
        from methods.TopoGate.V11.run import run_v11
        overrides = {
            "epochs": int(epochs),
            "warmup_epochs": max(1, min(epochs // 3, 3)),
            "ramp_epochs": max(1, min(epochs // 4, 2)),
            "batch_size": min(256, int(X.shape[0])),
            "use_topology": variant == "v14_full",
            "use_dynamic_graph": variant == "v14_full",
            "use_graph_prior": variant == "v14_full",
            "use_edge_consistency": variant == "v14_full",
        }
        _, elapsed, metrics = run_v11(
            X, int(np.unique(y).size), y, config_path=CONFIG, save_dir=out,
            dataset_name=name, gpu=3, seed=seed, source_path=source,
            k_protocol="benchmark_oracle_from_y", **overrides
        )
        row.update({"status": "completed", "elapsed": float(elapsed), "ari": metrics.get("ari"), "head_ari": metrics.get("head", {}).get("ari"), "kmeans_ari": metrics.get("kmeans", {}).get("ari"), "error": None})
        summary = json.loads((out / "summary.json").read_text())
        history = summary.get("history", [])
        active = [h for h in history if float(h.get("ramp", 0.0)) > 0]
        for key in ("gate", "target_gate", "gate_evidence", "graph", "edge_consistency", "topology_cls"):
            vals = [float(h[key]) for h in active if key in h]
            row[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
            row[f"{key}_last"] = float(vals[-1]) if vals else 0.0
    except Exception as exc:
        row.update({"status": "failed", "error": f"{exc}\n{traceback.format_exc()}"})
    (out / "run_record.json").write_text(json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return row

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--variants", nargs="*", default=["v14_full", "v14_nomix"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        for var in args.variants:
            for seed in args.seeds:
                print(f"{ds} {var} seed={seed}", flush=True)
                rows.append(run_one(ds, var, seed, args.output_dir, args.epochs))
    fields = sorted({k for r in rows for k in r})
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Wrote {args.output_dir / 'runs.csv'} rows={len(rows)}")

if __name__ == "__main__":
    main()
