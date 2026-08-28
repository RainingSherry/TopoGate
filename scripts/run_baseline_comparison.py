"""Benchmark runner: G-CEALS, IDC, TableDC, ZEUS, TopoGate.

Runs all 5 clusterers on 15 specified datasets, saves per-model CSV files
and a combined summary.csv.  This script is part of the integration effort
that ties G-CEALS / IDC / TableDC / ZEUS into CLUBench without modifying
upstream source files.

Run from the ToPoGate repo root:
    python scripts/run_baseline_comparison.py [--seeds 1] [--datasets iris ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

# Repo setup
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "baseline" / "CLUBench"))

from CLUBench import (  # noqa: E402
    GCEALS, IDC, TableDC, ZEUS, TopoGate, clustering_evaluation,
)

DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
RESULT_DIR = Path("/data/luolie/ToPoGate/result/baseline_comparison")

DATASETS = [
    "Mouse_retina",
    "sms_spam_collection",
    "enron",
    "ISOLET",
    "har",
    "spambase",
    "breast_cancer_wisconsin_original",
    "reuters",
    "cnae9",
    "Campbell",
    "mammographic_mass",
    "first-order-theorem-proving",
    "hrvatin_filtered",
    "Quake_Smart-seq2_Lung",
    "iris",
]

# Per-dataset model kwargs (e.g. reduce ZEUS n_init for large data)
MODEL_KWARGS_OVERRIDE = {
    "hrvatin_filtered": {
        "ZEUS": {"n_init": 5},  # default 100 too slow on 48k samples
        "TopoGate": {"subsample_size": 10000},  # kNN graph OOM on 48k samples
    },
}


def _get_model_kwargs(dataset_name, model_name):
    return MODEL_KWARGS_OVERRIDE.get(dataset_name, {}).get(model_name, {})


def _get_timeout(dataset_name, default):
    return {
        "hrvatin_filtered": 7200,  # 2 hours for 48k samples
    }.get(dataset_name, default)

MODELS = {
    "GCEALS": lambda n_clusters: GCEALS(n_clusters=n_clusters),
    "IDC": lambda n_clusters: IDC(n_clusters=n_clusters),
    "TableDC": lambda n_clusters: TableDC(n_clusters=n_clusters),
    "ZEUS": lambda n_clusters: ZEUS(n_clusters=n_clusters),
    "TopoGate": lambda n_clusters: TopoGate(n_clusters=n_clusters),
}

# HPC overrides — use the JSON configs we shipped for each model
def _load_hpc(name):
    cfg_path = _REPO_ROOT / "baseline/CLUBench/CLUBench/hpc" / f"{name.lower()}.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def _model_with_hpc(model_name, n_clusters):
    base_kwargs = {"n_clusters": n_clusters}
    if model_name == "TopoGate":
        return TopoGate(n_clusters=n_clusters)
    cfg = _load_hpc(model_name)
    base_kwargs.update(cfg)
    base_kwargs.pop("device", None)  # let the wrapper decide based on cuda availability
    return MODELS[model_name](n_clusters)


def _worker(model_name, dataset_name, X_path, y_path, output_csv, timeout, pca_dim=None):
    """Worker entry for multiprocessing."""
    try:
        data = np.load(X_path)
        X = data["x"]
        y = np.load(y_path)["y"] if y_path else None
        n_clusters = int(np.unique(y).size) if y is not None else int(np.load(X_path)["y"].size)
        # n_clusters derived from y if available
        # Optional PCA preprocessing (for memory-constrained large datasets).
        # Applied BEFORE handing to the model so the wrapper sees the reduced
        # feature space. The reported `n_features` becomes the post-PCA dim.
        pre_features = int(X.shape[1])
        if pca_dim is not None and X.shape[1] > pca_dim:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=pca_dim, random_state=42)
            X = pca.fit_transform(X)
        model_kwargs = _get_model_kwargs(dataset_name, model_name)
        if model_name == "TopoGate":
            model = TopoGate(n_clusters=n_clusters, **model_kwargs)
        else:
            from CLUBench import GCEALS, IDC, TableDC, ZEUS
            wrappers = {
                "GCEALS": GCEALS, "IDC": IDC, "TableDC": TableDC, "ZEUS": ZEUS,
            }
            model = wrappers[model_name](n_clusters=n_clusters, **model_kwargs)
        t0 = time.time()
        labels = model.fit_predict(X)
        dt = time.time() - t0
        if y is not None:
            metrics = clustering_evaluation(np.asarray(y).astype(int), np.asarray(labels).astype(int))
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "n_clusters": n_clusters,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "ACC": float(metrics["acc"]),
                "NMI": float(metrics["nmi"]),
                "ARI": float(metrics["ari"]),
                "time_sec": float(dt),
            }
        else:
            row = {
                "dataset": dataset_name, "model": model_name,
                "n_clusters": n_clusters, "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "ACC": None, "NMI": None, "ARI": None,
                "time_sec": float(dt),
            }
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
    except Exception as exc:
        with open(str(output_csv) + ".error", "a") as f:
            f.write(f"[{dataset_name} / {model_name}] {exc}\n{traceback.format_exc()}\n")


# Datasets that require PCA preprocessing (otherwise GPU OOM or memory blowup)
PCA_DIM_OVERRIDE = {
    "hrvatin_filtered": 500,  # 48266x25187 → 48266x500
}


def _get_pca_dim(dataset_name, default=None):
    return PCA_DIM_OVERRIDE.get(dataset_name, default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of datasets to run; default = all in DATASETS list")
    parser.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                        help="Subset of models to run; default = all")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-dataset per-model timeout in seconds")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds per (model, dataset) pair")
    parser.add_argument("--pca-dim", type=int, default=None,
                        help="Apply PCA to reduce input features (default: only "
                             "for hrvatin_filtered via PCA_DIM_OVERRIDE)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just list datasets/models and exit")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    selected_datasets = args.datasets or DATASETS
    selected_models = args.models

    # Sanity: ensure each dataset exists
    available = []
    for name in selected_datasets:
        npz = DATA_DIR / f"{name}.npz"
        if npz.exists():
            available.append(name)
        else:
            print(f"[skip] {name}.npz not found at {npz}")
    selected_datasets = available

    if args.dry_run:
        print("Datasets:", selected_datasets)
        print("Models:", selected_models)
        return

    # Run sequentially (avoid GPU contention). Each model writes a single
    # CSV with one row per dataset. Skip if CSV already has a row for this
    # dataset — supports resumable runs.
    for model_name in selected_models:
        out_csv = RESULT_DIR / f"{model_name}.csv"
        existing_rows = set()
        if out_csv.exists():
            with open(out_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows.add(row.get("dataset", ""))
            write_header = False
        else:
            write_header = True
        csv_file = open(out_csv, "a", newline="")
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow([
                "dataset", "model", "n_clusters", "n_samples", "n_features",
                "ACC", "NMI", "ARI", "time_sec",
            ])

        for dataset_name in selected_datasets:
            if dataset_name in existing_rows:
                print(f"[skip] {model_name} on {dataset_name} (already done)")
                continue
            npz = DATA_DIR / f"{dataset_name}.npz"
            pca_dim = _get_pca_dim(dataset_name, args.pca_dim)
            print(f"[run] {model_name} on {dataset_name}"
                  + (f" (PCA={pca_dim})" if pca_dim else ""))
            try:
                _worker(model_name, dataset_name, str(npz), str(npz),
                        str(out_csv), args.timeout, pca_dim=pca_dim)
            except Exception as exc:
                print(f"  FAILED: {exc}")
        csv_file.close()

    # Summary CSV
    summary_path = RESULT_DIR / "summary.csv"
    summary_rows = []
    for model_name in selected_models:
        per_model = RESULT_DIR / f"{model_name}.csv"
        if per_model.exists():
            with open(per_model) as f:
                for row in csv.DictReader(f):
                    summary_rows.append(row)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()