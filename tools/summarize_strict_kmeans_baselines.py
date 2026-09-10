"""Summarize strict split KMeans baseline JSON into archive CSV files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    seed_rows = []
    selection_rows = []
    validation_rows = []
    for record in records:
        for method, runs in record["final"].items():
            for run in runs:
                seed_rows.append({
                    "panel": "CLUBench", "protocol": record["protocol"],
                    "dataset": record["dataset"], "model": method,
                    "seed": run["seed"], "ARI": run["metrics"]["ari"],
                    "NMI": run["metrics"]["nmi"], "ACC": run["metrics"]["acc"],
                    "split_hash": record["split_hash"], "data_hash": record["data_hash"],
                    "direct_comparable": "true", "status": "complete",
                })
        for selection in record.get("selection", []):
            winner = selection["winner"]
            selection_rows.append({
                "panel": "CLUBench", "dataset": record["dataset"],
                "model": selection["method"],
                "candidate_rule": "PCA dimensions 20,50,100 clipped and deduplicated",
                "candidate_count": len(selection["all_candidates"]),
                "selected_pca_dim": winner["pca_dim_actual"],
                "selection_metric": "validation ARI mean seeds 42,123,7",
                "validation_ari_mean": winner["validation_ari_mean"],
                "status": "complete",
            })
            for candidate_index, candidate in enumerate(selection["all_candidates"], start=1):
                validation_rows.append({
                    "panel": "CLUBench", "dataset": record["dataset"],
                    "model": selection["method"], "candidate_index": candidate_index,
                    "pca_dim_actual": candidate["pca_dim_actual"],
                    "seed": 42,
                    "validation_ARI": candidate["seed42"]["validation"]["ari"],
                    "validation_NMI": candidate["seed42"]["validation"]["nmi"],
                    "validation_ACC": candidate["seed42"]["validation"]["acc"],
                    "selection_status": "winner" if candidate is winner else "candidate",
                    "protocol_status": "strict_split_direct",
                })
                for seed in (123, 7):
                    if "refinement" in candidate and str(seed) in candidate["refinement"]:
                        validation_rows.append({
                            "panel": "CLUBench", "dataset": record["dataset"],
                            "model": selection["method"], "candidate_index": candidate_index,
                            "pca_dim_actual": candidate["pca_dim_actual"], "seed": seed,
                            "validation_ARI": candidate["refinement"][str(seed)]["validation"]["ari"],
                            "validation_NMI": candidate["refinement"][str(seed)]["validation"]["nmi"],
                            "validation_ACC": candidate["refinement"][str(seed)]["validation"]["acc"],
                            "selection_status": "winner" if candidate is winner else "refined_candidate",
                            "protocol_status": "strict_split_direct",
                        })
    seed_fields = list(seed_rows[0])
    write_csv(args.archive / "test_metrics_per_seed.csv", seed_fields, seed_rows)
    write_csv(args.archive / "summaries" / "test_metrics_per_seed.csv", seed_fields, seed_rows)
    selected_fields = list(selection_rows[0])
    write_csv(args.archive / "selected_hyperparameters.csv", selected_fields, selection_rows)
    validation_fields = list(validation_rows[0])
    write_csv(args.archive / "validation_metrics_long.csv", validation_fields, validation_rows)
    write_csv(args.archive / "summaries" / "validation_metrics_long.csv", validation_fields, validation_rows)

    summary_rows = []
    for dataset in sorted({r["dataset"] for r in seed_rows}):
        for model in ("KMeans", "PCA+KMeans"):
            group = [r for r in seed_rows if r["dataset"] == dataset and r["model"] == model]
            row = {"panel": "CLUBench", "protocol": group[0]["protocol"], "dataset": dataset, "model": model, "n_seeds": len(group)}
            for metric in ("ARI", "NMI", "ACC"):
                values = np.asarray([float(r[metric]) for r in group])
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            row["direct_comparable"] = "true"
            summary_rows.append(row)
    summary_fields = list(summary_rows[0])
    for name in ("matched_budget_summary.csv", "per_dataset_comparison.csv"):
        write_csv(args.archive / name, summary_fields, summary_rows)
        write_csv(args.archive / "summaries" / name, summary_fields, summary_rows)

    panel_rows = []
    for model in ("KMeans", "PCA+KMeans"):
        group = [r for r in summary_rows if r["model"] == model]
        panel_rows.append({
            "panel": "CLUBench", "protocol": group[0]["protocol"], "model": model,
            "dataset_count": len(group), "coverage": len(group) / 131,
            "ARI_macro_mean": float(np.mean([r["ARI_mean"] for r in group])),
            "NMI_macro_mean": float(np.mean([r["NMI_mean"] for r in group])),
            "ACC_macro_mean": float(np.mean([r["ACC_mean"] for r in group])),
            "status": "complete",
        })
    panel_fields = list(panel_rows[0])
    write_csv(args.archive / "panel_summary.csv", panel_fields, panel_rows)
    write_csv(args.archive / "summaries" / "panel_summary.csv", panel_fields, panel_rows)

    paired = []
    for dataset in sorted({r["dataset"] for r in summary_rows}):
        km = next(r for r in summary_rows if r["dataset"] == dataset and r["model"] == "KMeans")
        pca = next(r for r in summary_rows if r["dataset"] == dataset and r["model"] == "PCA+KMeans")
        paired.append({"panel": "CLUBench", "dataset": dataset, "model_a": "PCA+KMeans", "model_b": "KMeans", "ARI_delta": pca["ARI_mean"] - km["ARI_mean"], "NMI_delta": pca["NMI_mean"] - km["NMI_mean"], "ACC_delta": pca["ACC_mean"] - km["ACC_mean"]})
    paired_fields = list(paired[0])
    write_csv(args.archive / "paired_comparisons.csv", paired_fields, paired)
    write_csv(args.archive / "summaries" / "paired_comparisons.csv", paired_fields, paired)
    print(json.dumps({"datasets": len(records), "seed_rows": len(seed_rows), "summary_rows": len(summary_rows), "selection_rows": len(selection_rows)}))


if __name__ == "__main__":
    main()
