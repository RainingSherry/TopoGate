#!/usr/bin/env python3
"""Multi-seed runner for the complete TopoGate V10 variant."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from methods.TopoGate.v10_reliable_graph.run import run_v10


GPU_POOL = [1, 4, 5]  # worker id -> physical GPU; GPU 0 and GPU 7 are forbidden
DEFAULT_SEEDS = [42, 123, 7]
DEFAULT_DATASETS = [
    "enron",
    "har",
    "Campbell",
    "Mouse_retina",
    "cnae9",
    "reuters",
    "Quake_Smart-seq2_Lung",
    "breast_cancer_wisconsin_original",
    "iris",
    "mammographic_mass",
    "ISOLET",
    "spambase",
    "sms_spam_collection",
    "first-order-theorem-proving",
    "hrvatin_filtered",
]
DEFAULT_DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result" / "v10_reliable_graph" / "multiseed"
DEFAULT_CONFIG = REPO_ROOT / "methods" / "TopoGate" / "v10_reliable_graph" / "configs" / "topogate_v10_reliable_graph.yaml"
VARIANT_CONFIGS = {
    "topogate_v10_reliable_graph": DEFAULT_CONFIG,
    "topogate_v10_feature_only": (
        REPO_ROOT
        / "methods"
        / "TopoGate"
        / "v10_reliable_graph"
        / "configs"
        / "topogate_v10_feature_only.yaml"
    ),
    "topogate_v10_fixed_graph": (
        REPO_ROOT
        / "methods"
        / "TopoGate"
        / "v10_reliable_graph"
        / "configs"
        / "topogate_v10_fixed_graph.yaml"
    ),
}


def _run_one(
    dataset: str,
    variant: str,
    seed: int,
    gpu: int,
    data_dir: Path,
    output_dir: Path,
    config: Path,
    resume: bool,
    no_cuda: bool | None,
) -> dict:
    data_path = data_dir / f"{dataset}.npz"
    record = {"dataset": dataset, "variant": variant, "seed": seed, "gpu": gpu}
    if not data_path.exists():
        return {**record, "error": f"dataset not found: {data_path}"}
    payload = np.load(data_path, allow_pickle=False)
    X = payload.get("X", payload.get("x", payload.get("data")))
    y = payload.get("y", payload.get("labels", payload.get("label", None)))
    if X is None:
        return {**record, "error": f"X/x/data missing from {data_path}"}
    y = None if y is None else np.asarray(y).ravel()
    n_clusters = None if y is None else int(np.unique(y).size)
    run_dir = output_dir / f"{dataset}__{variant}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and (run_dir / "summary.json").exists() and (run_dir / "metrics.json").exists():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        return {
            **record,
            "n_clusters": summary.get("n_clusters", n_clusters),
            "elapsed": summary.get("train_seconds"),
            "acc": metrics.get("acc"),
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "resumed": True,
            "error": None,
        }
    try:
        overrides = {} if no_cuda is None else {"no_cuda": no_cuda}
        _, elapsed, metrics = run_v10(
            X,
            y=y,
            n_clusters=n_clusters,
            gpu=gpu,
            seed=seed,
            save_dir=run_dir,
            return_metrics=True,
            config_path=config,
            **overrides,
        )
        return {
            **record,
            "n_clusters": n_clusters,
            "elapsed": elapsed,
            "acc": metrics.get("acc"),
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "resumed": False,
            "error": None,
        }
    except Exception as error:  # preserve each failure without hiding later jobs
        return {**record, "n_clusters": n_clusters, "error": f"{error}\n{traceback.format_exc()}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--variants", nargs="*", choices=VARIANT_CONFIGS, default=list(VARIANT_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--worker_id", type=int, default=0, choices=range(len(GPU_POOL)))
    parser.add_argument("--gpu", type=int, default=None, choices=GPU_POOL)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_cuda", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    gpu = args.gpu if args.gpu is not None else GPU_POOL[args.worker_id]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(dataset, variant, seed) for dataset in args.datasets for variant in args.variants for seed in args.seeds]
    rows: list[dict] = []
    for position, (dataset, variant, seed) in enumerate(jobs, start=1):
        started = time.time()
        print(f"[{position}/{len(jobs)}] {dataset} {variant} seed={seed} gpu={gpu}", flush=True)
        result = _run_one(
            dataset,
            variant,
            seed,
            gpu,
            args.data_dir,
            args.output_dir,
            VARIANT_CONFIGS[variant],
            args.resume,
            args.no_cuda,
        )
        rows.append(result)
        with (args.output_dir / f"{dataset}__{variant}__seed{seed}.json").open("w") as handle:
            json.dump(result, handle, indent=2)
        if result.get("error"):
            print(f"  ERROR: {str(result['error']).splitlines()[0]}", flush=True)
        else:
            print(
                f"  ARI={result['ari']:.4f} NMI={result['nmi']:.4f} ACC={result['acc']:.4f} "
                f"wall={time.time() - started:.1f}s",
                flush=True,
            )

    columns = ["dataset", "variant", "seed", "gpu", "n_clusters", "elapsed", "acc", "nmi", "ari", "resumed", "error"]
    with (args.output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in rows])

    successful = [row for row in rows if not row.get("error")]
    aggregate_rows: list[dict] = []
    for dataset in sorted({row["dataset"] for row in successful}):
        for variant in sorted({row["variant"] for row in successful if row["dataset"] == dataset}):
            group = [row for row in successful if row["dataset"] == dataset and row["variant"] == variant]
            aggregate = {"dataset": dataset, "variant": variant, "n_seeds": len(group)}
            for metric in ("acc", "nmi", "ari"):
                values = np.asarray([row[metric] for row in group], dtype=np.float64)
                aggregate[f"{metric}_mean"] = float(values.mean())
                aggregate[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            aggregate_rows.append(aggregate)
    aggregate_columns = [
        "dataset", "variant", "n_seeds",
        "acc_mean", "acc_std", "nmi_mean", "nmi_std", "ari_mean", "ari_std",
    ]
    with (args.output_dir / "mean_std.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    indexed = {(row["dataset"], row["variant"], row["seed"]): row for row in successful}
    paired_rows: list[dict] = []
    full_variant = "topogate_v10_reliable_graph"
    for dataset in sorted({row["dataset"] for row in successful}):
        for comparator in ("topogate_v10_fixed_graph", "topogate_v10_feature_only"):
            deltas = []
            for seed in args.seeds:
                full = indexed.get((dataset, full_variant, seed))
                control = indexed.get((dataset, comparator, seed))
                if full is not None and control is not None:
                    deltas.append(float(full["ari"]) - float(control["ari"]))
            if deltas:
                values = np.asarray(deltas, dtype=np.float64)
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{full_variant}-minus-{comparator}",
                        "n_pairs": int(values.size),
                        "ari_delta_mean": float(values.mean()),
                        "ari_delta_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    }
                )
    with (args.output_dir / "paired_deltas.csv").open("w", newline="") as handle:
        columns = ["dataset", "comparison", "n_pairs", "ari_delta_mean", "ari_delta_std"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(paired_rows)
    print(f"Completed {len(rows)} runs; errors={sum(bool(row.get('error')) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
