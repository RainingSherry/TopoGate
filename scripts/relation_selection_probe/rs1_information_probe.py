"""Run the RS1 grouped diagnostic probes for relation information.

Feature extraction is label-free.  Labels are loaded only after the frozen
feature table exists to form the two diagnostic targets and are never passed to
feature construction or selector code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .relation_features import (
    DATASETS,
    FEATURE_FAMILIES,
    MATERIALITY_DELTA,
    PRIMARY_DATASETS,
    RS1_DELTA_AP,
    RS1_LIFT,
    S1_ROOT,
    S0_ROOT,
    EdgeTable,
    extract_edge_features,
    jsonable,
    load_h0_and_pool,
    save_edge_table,
    sha256_array,
    write_json,
)


DEFAULT_OUTPUT = Path("result/relation_selection_probe/RS1_information")


def _load_labels(dataset: str) -> np.ndarray:
    path = S1_ROOT / dataset / "seed42" / "R" / "labels_true.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing audited labels-after-fit artifact: {path}")
    labels = np.asarray(np.load(path), dtype=np.int64)
    return labels


def _load_reference_target(dataset: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    path = S1_ROOT / dataset / "seed42" / "O_pool" / "directed_graph.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing audited O_pool directed graph: {path}")
    graph = sp.load_npz(path).tocsr()
    return np.asarray(graph[rows, cols]).reshape(-1).astype(np.float32) > 0.0


def _ndcg_at_budget(target: np.ndarray, score: np.ndarray, rows: np.ndarray, budget: np.ndarray) -> float:
    discounts_cache: dict[int, np.ndarray] = {}
    values: list[float] = []
    for row in range(int(budget.size)):
        start = int(np.searchsorted(rows, row, side="left"))
        end = int(np.searchsorted(rows, row, side="right"))
        edge_ids = np.arange(start, end, dtype=np.int64)
        b_i = int(budget[row])
        if b_i <= 0 or edge_ids.size == 0:
            continue
        b_i = min(b_i, edge_ids.size)
        order = edge_ids[np.argsort(-score[edge_ids], kind="mergesort")[:b_i]]
        if b_i not in discounts_cache:
            discounts_cache[b_i] = 1.0 / np.log2(np.arange(2, b_i + 2))
        discounts = discounts_cache[b_i]
        dcg = float(np.sum(target[order].astype(np.float64) * discounts))
        ideal = np.sort(target[edge_ids].astype(np.float64))[::-1][:b_i]
        idcg = float(np.sum(ideal * discounts))
        values.append(dcg / idcg if idcg > 0.0 else 0.0)
    return float(np.mean(values)) if values else 0.0


def _top_budget_metrics(
    target: np.ndarray,
    score: np.ndarray,
    rows: np.ndarray,
    budget: np.ndarray,
) -> dict[str, float]:
    selected: list[int] = []
    for row in range(int(budget.size)):
        start = int(np.searchsorted(rows, row, side="left"))
        end = int(np.searchsorted(rows, row, side="right"))
        edge_ids = np.arange(start, end, dtype=np.int64)
        b_i = min(int(budget[row]), int(edge_ids.size))
        if b_i > 0:
            selected.extend(edge_ids[np.argsort(-score[edge_ids], kind="mergesort")[:b_i]].tolist())
    selected_ids = np.asarray(selected, dtype=np.int64)
    prevalence = float(np.mean(target)) if target.size else 0.0
    selected_positive = float(np.sum(target[selected_ids])) if selected_ids.size else 0.0
    precision = selected_positive / selected_ids.size if selected_ids.size else 0.0
    total_positive = float(np.sum(target))
    recall = selected_positive / total_positive if total_positive else 0.0
    return {
        "precision_at_b": precision,
        "recall_at_b": recall,
        "lift_at_b": precision / prevalence if prevalence > 0.0 else 0.0,
        "ndcg_at_b": _ndcg_at_budget(target, score, rows, budget),
    }


def grouped_probe(
    table: EdgeTable,
    target: np.ndarray,
    family: str,
    *,
    n_splits: int = 5,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return grouped out-of-fold diagnostics and scores for one target/family."""
    target = np.asarray(target, dtype=np.int64)
    if target.shape != (table.rows.size,):
        raise ValueError("diagnostic target does not align with edge table")
    if np.unique(target).size < 2:
        return (
            {
                "status": "not_estimable",
                "reason": "constant_target",
                "family": family,
                "edge_count": int(target.size),
            },
            np.full(target.size, float(np.mean(target)), dtype=np.float64),
        )
    x = table.family(family)
    scores = np.full(target.size, np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train, test) in enumerate(splitter.split(x, target, groups=table.rows)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=200,
                random_state=0,
                solver="lbfgs",
            ),
        )
        model.fit(x[train], target[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("grouped probe did not produce complete out-of-fold scores")
    prevalence = float(np.mean(target))
    ap = float(average_precision_score(target, scores))
    try:
        auroc = float(roc_auc_score(target, scores))
    except ValueError:
        auroc = float("nan")
    top = _top_budget_metrics(target, scores, table.rows, table.budget)
    result = {
        "status": "completed_valid",
        "family": family,
        "edge_count": int(target.size),
        "group_count": int(np.unique(table.rows).size),
        "folds": n_splits,
        "positive_count": int(np.sum(target)),
        "positive_prevalence": prevalence,
        "average_precision": ap,
        "delta_ap": ap - prevalence,
        "auroc": auroc,
        **top,
        "labels_used_in_feature_extraction": False,
        "labels_used_in_diagnostic_target": True,
        "group_split": "anchor_sample",
    }
    return result, scores


def _build_dataset_table(dataset: str, output_dir: Path) -> tuple[EdgeTable, np.ndarray, np.ndarray]:
    h0, pool = load_h0_and_pool(dataset)
    table = extract_edge_features(h0, pool)
    table_dir = output_dir / "features" / dataset
    table_dir.mkdir(parents=True, exist_ok=True)
    save_edge_table(table_dir / "edge_features.npz", table)
    write_json(table_dir / "feature_metadata.json", table.metadata)
    labels = _load_labels(dataset)
    if labels.size != table.n_samples:
        raise ValueError(f"labels/H0 mismatch for {dataset}")
    class_target = (labels[table.cols] == labels[table.rows]).astype(np.int64)
    reference_target = _load_reference_target(dataset, table.rows, table.cols).astype(np.int64)
    np.savez_compressed(
        table_dir / "diagnostic_targets.npz",
        same_class=class_target,
        pool_reference_membership=reference_target,
    )
    write_json(
        table_dir / "diagnostic_audit.json",
        {
            "dataset": dataset,
            "h0_sha256": sha256_array(h0),
            "edge_count": int(table.rows.size),
            "labels_source": "S1_oracle_v2/seed42/R/labels_true.npy",
            "labels_used_in_features": False,
            "labels_used_in_targets": True,
            "reference_source": "S1_oracle_v2/seed42/O_pool/directed_graph.npz",
        },
    )
    return table, class_target, reference_target


def run(output_dir: Path = DEFAULT_OUTPUT, datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    dataset_status: dict[str, Any] = {}
    for dataset in datasets:
        table, class_target, reference_target = _build_dataset_table(dataset, output_dir)
        dataset_status[dataset] = {
            "edge_count": int(table.rows.size),
            "feature_count": int(table.features.shape[1]),
            "same_class_prevalence": float(np.mean(class_target)),
            "pool_reference_prevalence": float(np.mean(reference_target)),
            "labels_used_in_features": False,
        }
        for target_name, target in (
            ("same_class", class_target),
            ("pool_reference_membership", reference_target),
        ):
            for family in FEATURE_FAMILIES:
                result, scores = grouped_probe(table, target, family)
                result.update({"dataset": dataset, "target": target_name})
                rows.append(result)
                np.save(
                    output_dir / "features" / dataset / f"oof_{target_name}_{family.replace('+', '_')}.npy",
                    scores.astype(np.float32),
                )
    # The information gate is evaluated only on the pre-registered primary set.
    # The two diagnostic targets remain separate: class-target information is a
    # semantic diagnostic, while reference-target information is the direct
    # solvability diagnostic for a future selector.
    family_gate: dict[str, dict[str, Any]] = {}
    for target_name in ("same_class", "pool_reference_membership"):
        family_gate[target_name] = {}
        for family in FEATURE_FAMILIES:
            passing_datasets: list[str] = []
            for dataset in PRIMARY_DATASETS:
                matches = [
                    row
                    for row in rows
                    if row.get("dataset") == dataset
                    and row.get("target") == target_name
                    and row.get("family") == family
                    and row.get("status") == "completed_valid"
                    and float(row.get("delta_ap", -np.inf)) >= RS1_DELTA_AP
                    and float(row.get("lift_at_b", -np.inf)) >= RS1_LIFT
                ]
                if matches:
                    passing_datasets.append(dataset)
            family_gate[target_name][family] = {
                "passing_primary_datasets": passing_datasets,
                "passes_information_gate": len(passing_datasets) >= 2,
            }
    information_passes = any(
        value["passes_information_gate"]
        for target_values in family_gate.values()
        for value in target_values.values()
    )
    summary = {
        "project_id": "relation_selection_probe",
        "stage": "RS1_information",
        "protocol_id": "relation_selection_probe_rs1_v1",
        "status": "completed_valid",
        "datasets": list(datasets),
        "primary_datasets": list(PRIMARY_DATASETS),
        "dataset_status": dataset_status,
        "rows": rows,
        "family_gate": family_gate,
        "information_passes": information_passes,
        "decision": (
            "relation_information_present"
            if information_passes
            else "current_relation_evidence_not_sufficient"
        ),
        "labels_used_in_feature_extraction": False,
        "labels_used_in_diagnostic_targets": True,
        "materiality_delta": MATERIALITY_DELTA,
        "rs1_delta_ap_threshold": RS1_DELTA_AP,
        "rs1_lift_threshold": RS1_LIFT,
    }
    write_json(output_dir / "rs1_summary.json", summary)
    write_json(output_dir / "rs1_manifest.json", {
        "project_id": "relation_selection_probe",
        "stage": "RS1_information",
        "protocol_id": "relation_selection_probe_rs1_v1",
        "datasets": list(datasets),
        "feature_families": {key: list(value) for key, value in FEATURE_FAMILIES.items()},
        "target_names": ["same_class", "pool_reference_membership"],
        "group_split": "GroupKFold_by_anchor_sample_5",
        "labels_used_in_features": False,
        "labels_used_in_diagnostic_targets": True,
        "status": "completed_valid",
    })
    (output_dir / "rs1_metrics.json").write_text(json.dumps(jsonable(rows), indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    args = parser.parse_args()
    selected = tuple(args.dataset) if args.dataset else DATASETS
    result = run(args.output_dir, selected)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
