#!/usr/bin/env python3
"""Pre-registered multi-seed runner for TopoGate V11 and its ablations."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

GPU_POOL = [1, 2, 3, 4, 5, 6]  # worker id -> physical GPU; GPU 0 and 7 are forbidden

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from methods.TopoGate.V11.run import run_v11


DATASETS = [
    "enron", "har", "Campbell", "Mouse_retina", "cnae9",
    "reuters", "Quake_Smart-seq2_Lung", "breast_cancer_wisconsin_original",
    "iris", "mammographic_mass", "ISOLET", "spambase",
    "sms_spam_collection", "first-order-theorem-proving", "hrvatin_filtered",
]
DATA_DIR = REPO_ROOT / "datasets"
# The AHDPC files are intentionally mapped explicitly instead of copied into
# the root dataset directory. This keeps the source manifest and output
# provenance visible in each run while preserving the legacy default suite.
DATASET_PATHS = {
    **{name: DATA_DIR / f"{name}.npz" for name in DATASETS},
    "balance_scale": REPO_ROOT / "datasets" / "AHDPC" / "processed" / "balance_scale.npz",
    "spect_heart": REPO_ROOT / "datasets" / "AHDPC" / "processed" / "spect_heart.npz",
    "banknote": REPO_ROOT / "datasets" / "AHDPC" / "processed" / "banknote.npz",
    "flame": REPO_ROOT / "datasets" / "AHDPC" / "processed" / "flame.npz",
    "vehicle": REPO_ROOT / "datasets" / "AHDPC" / "processed" / "vehicle.npz",
}
SEEDS = [42, 123, 7]
CONFIG = REPO_ROOT / "methods" / "TopoGate" / "V11" / "configs" / "topogate_v11.yaml"
OUTPUT_DIR = REPO_ROOT / "result" / "V11" / "multiseed"

VARIANTS = {
    "V11_full": {},
    "V11_nomix": {"use_topology": False},
    "V11_static_graph": {"use_dynamic_graph": False},
    "V11_uniform_edges": {"use_edge_reliability": False},
    "V11_no_teacher": {"use_teacher": False},
    "V11_no_cluster_head": {"use_cluster_head": False},
    "V11_no_mixed_reconstruction": {"use_mixed_reconstruction": False},
    "V11_no_mixed_cluster_consistency": {"mixed_cluster_weight": 0.0},
    "V11_no_graph_prior": {"use_graph_prior": False},
    "V11_edge_consistency": {"use_edge_consistency": True},
    "V11_legacy_risk": {"risk_target_mode": "legacy_student_train"},
    "V11_topology_gated_residual": {
        "topology_path": "assignment_residual",
        "gate_target_source": "temporal_agreement",
        "use_mixed_reconstruction": False,
        "mixed_cluster_weight": 0.0,
    },
    "V11_tda_h0_mst": {
        "tda_prior_mode": "h0_mst",
        "tda_prior_weight": 1.0,
    },
    "V11_tda_fixed_filtration": {
        "tda_prior_mode": "fixed_filtration",
        "tda_prior_weight": 1.0,
    },
    "V11_tda_random": {
        "tda_prior_mode": "random",
        "tda_prior_weight": 1.0,
    },
    "V11_tda_h0_early_mst": {
        "tda_prior_mode": "h0_early_mst",
        "tda_prior_weight": 1.0,
    },
}


def run_one(
    dataset: str,
    variant: str,
    overrides: dict,
    seed: int,
    gpu: int,
    no_cuda: bool,
    config_path: Path,
    output_dir: Path,
) -> dict:
    source = DATASET_PATHS.get(dataset, DATA_DIR / f"{dataset}.npz")
    if not source.exists():
        return {"dataset": dataset, "variant": variant, "seed": seed, "error": f"missing {source}"}
    data = np.load(source)
    X = data["X"] if "X" in data.files else data["x"]
    y = data["y"] if "y" in data.files else None
    if y is None:
        return {"dataset": dataset, "variant": variant, "seed": seed, "error": "labels absent; benchmark K unavailable"}
    n_clusters = int(np.unique(y).size)
    save_dir = output_dir / f"{dataset}__{variant}__seed{seed}"
    try:
        _, elapsed, metrics = run_v11(
            X,
            n_clusters,
            y,
            config_path=config_path,
            save_dir=save_dir,
            dataset_name=dataset,
            gpu=gpu,
            no_cuda=no_cuda,
            seed=seed,
            source_path=source,
            k_protocol="benchmark_oracle_from_y",
            **overrides,
        )
        return {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "n_clusters": n_clusters,
            "acc": metrics.get("acc"),
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "f1_macro": metrics.get("f1_macro"),
            "silhouette": metrics.get("silhouette"),
            "elapsed": elapsed,
            "error": None,
        }
    except Exception as exc:
        return {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0, choices=range(len(GPU_POOL)))
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=["V11_full"])
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    # An explicit suite may use an audited source outside the legacy default
    # list, such as datasets/AHDPC/processed/*.npz. Preserve the requested
    # order so the CSV is reproducible and do not silently run an empty suite.
    selected_datasets = DATASETS if args.datasets is None else list(dict.fromkeys(args.datasets))
    gpu = GPU_POOL[args.worker_id]
    args.config = args.config.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(selected_datasets) * len(args.variants) * len(args.seeds)
    counter = 0
    for dataset in selected_datasets:
        for variant in args.variants:
            for seed in args.seeds:
                counter += 1
                print(f"[{counter}/{total}] {dataset} {variant} seed={seed} gpu={gpu}", flush=True)
                row = run_one(
                    dataset,
                    variant,
                    VARIANTS[variant],
                    seed,
                    gpu,
                    args.no_cuda,
                    args.config,
                    args.output_dir,
                )
                rows.append(row)
                with open(args.output_dir / f"{dataset}__{variant}__seed{seed}.json", "w", encoding="utf-8") as handle:
                    json.dump(row, handle, indent=2, default=str)
                if row.get("error"):
                    print(f"  ERROR: {str(row['error']).splitlines()[-1]}", flush=True)
                else:
                    print(
                        f"  ACC={row['acc']:.4f} NMI={row['nmi']:.4f} ARI={row['ari']:.4f} "
                        f"K={row['n_clusters']} time={row['elapsed']:.1f}s",
                        flush=True,
                    )

    csv_path = args.output_dir / "comparison.csv"
    columns = [
        "dataset", "variant", "seed", "n_clusters", "acc", "nmi", "ari",
        "f1_macro", "silhouette", "elapsed", "error",
    ]
    with open(args.output_dir / "comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    print(
        f"Wrote {args.output_dir / 'comparison.csv'}; "
        f"errors={sum(bool(row.get('error')) for row in rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
