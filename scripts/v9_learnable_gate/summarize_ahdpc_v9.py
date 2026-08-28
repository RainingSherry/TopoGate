#!/usr/bin/env python3
"""Compare V9 runs with the persisted AHDPC/HDPC benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    rand_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "AHDPC" / "processed"
DEFAULT_V9_DIR = REPO_ROOT / "result" / "v9_results_2026-08-02"
DEFAULT_AHDPC_DIR = REPO_ROOT / "result" / "AHDPC" / "full_table_2026-07-31"
DEFAULT_SUPPLEMENTAL_DIR = DEFAULT_V9_DIR / "olivetti_hdpc_reference"


def aligned_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    matrix = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    true_index = {value: idx for idx, value in enumerate(true_values)}
    pred_index = {value: idx for idx, value in enumerate(pred_values)}
    for truth, prediction in zip(y_true, y_pred):
        matrix[true_index[truth], pred_index[prediction]] += 1
    rows, cols = linear_sum_assignment(-matrix)
    matched = int(matrix[rows, cols].sum())
    return matched / max(1, len(y_true))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(aligned_accuracy(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "fmi": float(fowlkes_mallows_score(y_true, y_pred)),
        "ri": float(rand_score(y_true, y_pred)),
    }


def load_truth(data_dir: Path, dataset: str) -> np.ndarray:
    data = np.load(data_dir / f"{dataset}.npz")
    key = "y" if "y" in data.files else "labels"
    return np.asarray(data[key]).ravel()


def reference_path(
    ahdpc_dir: Path,
    supplemental_dir: Path,
    dataset: str,
    method: str,
) -> Path | None:
    if dataset == "olivetti_faces":
        if method == "ahdpc":
            path = ahdpc_dir / "olivetti" / "labels.npy"
            return path if path.exists() else None
        if method == "hdpc":
            path = supplemental_dir / "predictions.npy"
            return path if path.exists() else None
        return None
    for category in ("synthetic", "uci"):
        path = ahdpc_dir / category / f"{dataset}__{method}" / "labels.npy"
        if path.exists():
            return path
    return None


def reference_note(
    ahdpc_dir: Path, supplemental_dir: Path, dataset: str, method: str
) -> str:
    if dataset == "olivetti_faces" and method == "ahdpc":
        return "t-SNE(2D, perplexity=30, max_iter=1000, seed=42)+AHDPC"
    if dataset == "olivetti_faces" and method == "hdpc":
        return "t-SNE(2D, perplexity=30, max_iter=1000, seed=42)+HDPC"
    if dataset == "olivetti_faces":
        return "not available in persisted full_table"
    return "paper-preprocessed X; table_reproduction"


def load_v9_rows(v9_dir: Path, data_dir: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(v9_dir.glob("*__v9_adaptive__seed*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("run_status") != "completed":
            rows.append(
                {
                    "method": "V9",
                    "dataset": summary.get("dataset", summary_path.parent.name),
                    "seed": summary.get("seed"),
                    "status": "failed",
                    "error": summary.get("error"),
                }
            )
            continue
        dataset = str(summary["dataset"])
        y = load_truth(data_dir, dataset)
        pred_path = summary_path.parent / summary.get("prediction_path", "predictions.npy")
        prediction = np.load(pred_path)
        row = {
            "method": "V9",
            "dataset": dataset,
            "seed": int(summary["seed"]),
            "status": "completed",
            "protocol": (
                "raw processed X, StandardScaler; V9 adaptive PCA kNN "
                "(upper bound 2000), reliability mix"
            ),
            "source_sha256": summary.get("source_sha256"),
        }
        row.update(metrics(y, prediction))
        rows.append(row)
    return rows


def load_reference_rows(
    ahdpc_dir: Path,
    supplemental_dir: Path,
    data_dir: Path,
    datasets: list[str],
) -> list[dict]:
    rows = []
    for dataset in datasets:
        y = load_truth(data_dir, dataset)
        for method in ("AHDPC", "HDPC"):
            key = method.lower()
            labels_path = reference_path(ahdpc_dir, supplemental_dir, dataset, key)
            if labels_path is None:
                rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "seed": 42 if dataset == "olivetti_faces" else None,
                        "status": "missing",
                        "protocol": reference_note(
                            ahdpc_dir, supplemental_dir, dataset, key
                        ),
                    }
                )
                continue
            prediction = np.load(labels_path)
            row = {
                "method": method,
                "dataset": dataset,
                "seed": 42 if dataset == "olivetti_faces" else None,
                "status": "completed",
                "protocol": reference_note(
                    ahdpc_dir, supplemental_dir, dataset, key
                ),
            }
            row.update(metrics(y, prediction))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "method", "dataset", "seed", "status", "protocol",
        "acc", "ari", "ami", "nmi", "fmi", "ri",
        "source_sha256", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    values_np = np.asarray(values, dtype=np.float64)
    return float(values_np.mean()), float(values_np.std(ddof=0))


def aggregate(rows: list[dict], datasets: list[str]) -> tuple[list[dict], list[dict]]:
    by_method_dataset: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            by_method_dataset[(row["method"], row["dataset"])].append(row)

    summary = []
    for dataset in datasets:
        v9 = by_method_dataset.get(("V9", dataset), [])
        ahdpc = by_method_dataset.get(("AHDPC", dataset), [])
        hdpc = by_method_dataset.get(("HDPC", dataset), [])
        row: dict[str, object] = {
            "dataset": dataset,
            "v9_n": len(v9),
            "ahdpc_n": len(ahdpc),
            "hdpc_n": len(hdpc),
        }
        for metric in ("acc", "ari", "ami", "nmi", "fmi", "ri"):
            v9_mean, v9_std = mean_std([float(r[metric]) for r in v9])
            ahdpc_val = float(ahdpc[0][metric]) if ahdpc else None
            hdpc_val = float(hdpc[0][metric]) if hdpc else None
            row[f"v9_{metric}_mean"] = v9_mean
            row[f"v9_{metric}_std"] = v9_std
            row[f"ahdpc_{metric}"] = ahdpc_val
            row[f"hdpc_{metric}"] = hdpc_val
            row[f"v9_minus_ahdpc_{metric}"] = (
                v9_mean - ahdpc_val if v9_mean is not None and ahdpc_val is not None else None
            )
            row[f"v9_minus_hdpc_{metric}"] = (
                v9_mean - hdpc_val if v9_mean is not None and hdpc_val is not None else None
            )
        summary.append(row)

    overall = []
    for method in ("V9", "AHDPC", "HDPC"):
        for metric in ("acc", "ari", "ami", "nmi", "fmi", "ri"):
            values = [
                float(row[metric])
                for row in rows
                if row.get("method") == method
                and row.get("status") == "completed"
                and metric in row
            ]
            mean, std = mean_std(values)
            overall.append(
                {
                    "method": method,
                    "metric": metric,
                    "dataset_or_seed_count": len(values),
                    "macro_mean": mean,
                    "macro_std": std,
                }
            )
    return summary, overall


def write_markdown(path: Path, summary: list[dict]) -> None:
    lines = [
        "# V9 vs AHDPC vs HDPC",
        "",
        "V9 uses three seeds (42, 123, 7). AHDPC/HDPC values are the persisted",
        "single benchmark outputs. Values below are recomputed from prediction",
        "arrays using common ARI/NMI/FMI/RI/ACC definitions.",
        "",
        "| Dataset | V9 ARI mean±std | AHDPC ARI | HDPC ARI | V9−AHDPC | V9−HDPC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt(value):
            return "NA" if value is None else f"{value:.4f}"

        v9 = (
            "NA"
            if row["v9_ari_mean"] is None
            else f"{row['v9_ari_mean']:.4f}±{row['v9_ari_std']:.4f}"
        )
        lines.append(
            f"| {row['dataset']} | {v9} | {fmt(row['ahdpc_ari'])} | "
            f"{fmt(row['hdpc_ari'])} | {fmt(row['v9_minus_ahdpc_ari'])} | "
            f"{fmt(row['v9_minus_hdpc_ari'])} |"
        )
    lines.extend(
        [
            "",
            "Olivetti AHDPC and HDPC use the same t-SNE reference protocol (seed 42).",
            "V9 uses the original 4096-dimensional processed input, so Olivetti is",
            "not a same-input comparison with those density-peak references.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-dir", type=Path, default=DEFAULT_V9_DIR)
    parser.add_argument("--ahdpc-dir", type=Path, default=DEFAULT_AHDPC_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--supplemental-dir", type=Path, default=DEFAULT_SUPPLEMENTAL_DIR
    )
    args = parser.parse_args()
    args.v9_dir = args.v9_dir.resolve()
    args.ahdpc_dir = args.ahdpc_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.supplemental_dir = args.supplemental_dir.resolve()

    datasets = sorted(path.stem for path in args.data_dir.glob("*.npz"))
    # Exclude the two schema-safe variants from the paper's 24-row manifest.
    manifest = json.loads((args.data_dir.parent / "MANIFEST.json").read_text(encoding="utf-8"))
    datasets = sorted(
        name for name, row in manifest["datasets"].items()
        if row.get("status") == "prepared"
    )

    v9_rows = load_v9_rows(args.v9_dir, args.data_dir)
    reference_rows = load_reference_rows(
        args.ahdpc_dir, args.supplemental_dir, args.data_dir, datasets
    )
    all_rows = v9_rows + reference_rows
    summary, overall = aggregate(all_rows, datasets)

    write_csv(args.v9_dir / "v9_runs.csv", v9_rows)
    write_csv(args.v9_dir / "comparison_per_run.csv", all_rows)
    summary_columns = sorted({key for row in summary for key in row})
    with (args.v9_dir / "comparison_by_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary)
    with (args.v9_dir / "comparison_overall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "dataset_or_seed_count", "macro_mean", "macro_std"])
        writer.writeheader()
        writer.writerows(overall)
    write_markdown(args.v9_dir / "V9_vs_AHDPC_HDPC.md", summary)

    print(f"V9 rows: {len(v9_rows)}")
    print(f"Reference rows: {len(reference_rows)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Failed/missing: {sum(1 for row in all_rows if row.get('status') != 'completed')}")
    print(f"Wrote: {args.v9_dir / 'comparison_by_dataset.csv'}")
    print(f"Wrote: {args.v9_dir / 'V9_vs_AHDPC_HDPC.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
