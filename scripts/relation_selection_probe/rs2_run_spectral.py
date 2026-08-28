"""Run the five fixed relation selectors with the frozen Spectral consumer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scripts.representation_consumer_probe.protocol import spectral_predict_with_audit

from .relation_features import (
    DATASETS,
    MATERIALITY_DELTA,
    PILOT_SEEDS,
    PRIMARY_DATASETS,
    S1_ROOT,
    EdgeTable,
    FEATURE_FAMILIES,
    load_edge_table,
    load_h0_and_pool,
    extract_edge_features,
    save_edge_table,
    sha256_array,
    write_json,
)
from .selectors import SELECTORS, selected_graph, selector_contract


DEFAULT_OUTPUT = Path("result/relation_selection_probe/RS2_simple_selectors")
FEATURE_ROOT = Path("result/relation_selection_probe/RS1_information/features")


def _optimal_mapping_acc(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    true_values = np.unique(labels)
    pred_values = np.unique(predictions)
    matrix = np.zeros((true_values.size, pred_values.size), dtype=np.int64)
    true_index = {value: index for index, value in enumerate(true_values)}
    pred_index = {value: index for index, value in enumerate(pred_values)}
    for true, pred in zip(labels, predictions, strict=True):
        matrix[true_index[int(true)], pred_index[int(pred)]] += 1
    rows, cols = linear_sum_assignment(-matrix)
    matched = int(matrix[rows, cols].sum())
    return float(matched / labels.size) if labels.size else 0.0


def _load_labels(dataset: str) -> np.ndarray:
    path = S1_ROOT / dataset / "seed42" / "R" / "labels_true.npy"
    return np.asarray(np.load(path), dtype=np.int64)


def _reference_metrics(dataset: str, seed: int, arm: str) -> dict[str, Any]:
    path = S1_ROOT / dataset / f"seed{seed}" / arm / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ARI": float(value["metrics"]["ARI"]),
        "NMI": float(value["metrics"]["NMI"]),
        "ACC": float(value["metrics"]["ACC"]),
        "graph_hash": value.get("graph_hash"),
        "protocol_id": value.get("protocol_id"),
    }


def _load_or_extract_table(dataset: str) -> EdgeTable:
    path = FEATURE_ROOT / dataset / "edge_features.npz"
    if path.exists():
        return load_edge_table(path)
    # RS2 may be rerun from a clean result directory only after the same frozen
    # feature extractor has been used; no labels are involved in this fallback.
    h0, pool = load_h0_and_pool(dataset)
    table = extract_edge_features(h0, pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_edge_table(path, table)
    write_json(path.with_name("feature_metadata.json"), table.metadata)
    return table


def _run_one(
    dataset: str,
    selector: str,
    table: EdgeTable,
    labels: np.ndarray,
    graph: sp.csr_matrix,
    directed_graph: sp.csr_matrix,
    output_dir: Path,
) -> list[dict[str, Any]]:
    k = int(np.unique(labels).size)
    selector_dir = output_dir / dataset / selector
    selector_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in PILOT_SEEDS:
        run_dir = selector_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions, embedding, consumer_metadata = spectral_predict_with_audit(graph, k, seed=seed)
        metrics = {
            "ARI": float(adjusted_rand_score(labels, predictions)),
            "NMI": float(normalized_mutual_info_score(labels, predictions)),
            "ACC": _optimal_mapping_acc(labels, predictions),
        }
        reference_r = _reference_metrics(dataset, seed, "R")
        reference_pool = _reference_metrics(dataset, seed, "O_pool")
        h_pool = float(reference_pool["ARI"] - reference_r["ARI"])
        delta = float(metrics["ARI"] - reference_r["ARI"])
        capture = float(delta / h_pool) if h_pool >= MATERIALITY_DELTA else None
        np.save(run_dir / "embedding.npy", embedding.astype(np.float32, copy=False))
        np.save(run_dir / "predictions.npy", predictions.astype(np.int64, copy=False))
        write_json(
            run_dir / "resolved_config.json",
            {
                "project_id": "relation_selection_probe",
                "protocol_id": "relation_selection_probe_rs2_v1",
                "dataset": dataset,
                "selector": selector,
                "seed": int(seed),
                "consumer": "Spectral",
                "K": k,
                "K_source": "benchmark_oracle_from_y",
                "labels_used_during_fit": False,
                "labels_used_for_outer_metrics": True,
                "feature_families": {key: list(value) for key, value in FEATURE_FAMILIES.items()},
                "candidate_pool_reused": True,
                "R_O_pool_reused": True,
            },
        )
        summary = {
            "project_id": "relation_selection_probe",
            "protocol_id": "relation_selection_probe_rs2_v1",
            "dataset": dataset,
            "selector": selector,
            "seed": int(seed),
            "status": "completed_valid",
            "K": k,
            "K_source": "benchmark_oracle_from_y",
            "metrics": metrics,
            "reference_R": reference_r,
            "reference_O_pool": reference_pool,
            "H_pool": h_pool,
            "Delta_S": delta,
            "Capture_S": capture,
            "consumer_metadata": consumer_metadata,
            "labels_used_during_fit": False,
            "labels_used_for_outer_metrics": True,
            "selector_labels_used": False,
            "embedding_finite": bool(np.isfinite(embedding).all()),
            "prediction_unique": int(np.unique(predictions).size),
            "graph_hash": sha256_array(np.asarray(graph.data)),
        }
        write_json(run_dir / "summary.json", summary)
        write_json(
            run_dir / "audit.json",
            {
                "audit_ok": bool(summary["status"] == "completed_valid" and summary["embedding_finite"]),
                "dataset": dataset,
                "selector": selector,
                "seed": int(seed),
                "labels_used_during_fit": False,
                "labels_used_for_outer_metrics": True,
                "selector_labels_used": False,
                "row_budget_contract": True,
                "reference_artifacts_reused": True,
            },
        )
        rows.append(summary)
    return rows


def run(output_dir: Path = DEFAULT_OUTPUT, datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    selector_contracts: list[dict[str, Any]] = []
    for dataset in datasets:
        table = _load_or_extract_table(dataset)
        labels = _load_labels(dataset)
        if labels.size != table.n_samples:
            raise ValueError(f"labels/H0 mismatch for {dataset}")
        for selector in SELECTORS:
            graph, mask = selected_graph(table, selector)
            directed_graph = sp.csr_matrix(
                (
                    table.cosine[mask].astype(np.float32, copy=False),
                    (table.rows[mask], table.cols[mask]),
                ),
                shape=(table.n_samples, table.n_samples),
            )
            graph_dir = output_dir / "graphs" / dataset
            graph_dir.mkdir(parents=True, exist_ok=True)
            sp.save_npz(graph_dir / f"{selector}.npz", graph)
            sp.save_npz(graph_dir / f"{selector}.directed.npz", directed_graph)
            contract = selector_contract(selector, table, mask)
            contract.update({"dataset": dataset, "graph_nnz": int(graph.nnz)})
            selector_contracts.append(contract)
            all_rows.extend(_run_one(dataset, selector, table, labels, graph, directed_graph, output_dir / "runs"))
    aggregate: dict[str, Any] = {}
    for selector in SELECTORS:
        aggregate[selector] = {}
        for dataset in datasets:
            subset = [row for row in all_rows if row["selector"] == selector and row["dataset"] == dataset]
            if not subset:
                continue
            deltas = np.asarray([row["Delta_S"] for row in subset], dtype=np.float64)
            captures = np.asarray(
                [row["Capture_S"] for row in subset if row["Capture_S"] is not None],
                dtype=np.float64,
            )
            h_pool = float(np.mean([row["H_pool"] for row in subset]))
            aggregate[selector][dataset] = {
                "H_pool": h_pool,
                "Delta_S_mean": float(np.mean(deltas)),
                "Delta_S_median": float(np.median(deltas)),
                "Delta_S_all_seeds": [float(value) for value in deltas],
                "Capture_S_median": float(np.median(captures)) if captures.size else None,
                "Capture_S_all_material_seeds": [float(value) for value in captures],
                "material_opportunity": bool(h_pool >= MATERIALITY_DELTA),
            }
    primary_gate: dict[str, Any] = {}
    for selector in SELECTORS:
        qualifying = []
        for dataset in PRIMARY_DATASETS:
            row = aggregate.get(selector, {}).get(dataset, {})
            if (
                row.get("Delta_S_mean", -np.inf) >= MATERIALITY_DELTA
                and row.get("Capture_S_median") is not None
                and row["Capture_S_median"] >= 0.0
            ):
                qualifying.append(dataset)
        captures = [aggregate[selector][dataset]["Capture_S_median"] for dataset in qualifying]
        primary_gate[selector] = {
            "qualifying_primary_datasets": qualifying,
            "qualifying_count": len(qualifying),
            "median_capture_over_qualifying": float(np.median(captures)) if captures else None,
            "simple_rule_sufficient": bool(
                len(qualifying) >= 2
                and np.median(captures) >= 0.25
                if captures
                else False
            ),
        }
    any_simple_sufficient = any(value["simple_rule_sufficient"] for value in primary_gate.values())
    summary = {
        "project_id": "relation_selection_probe",
        "stage": "RS2_simple_selectors",
        "protocol_id": "relation_selection_probe_rs2_v1",
        "status": "completed_valid",
        "datasets": list(datasets),
        "selectors": list(SELECTORS),
        "seeds": list(PILOT_SEEDS),
        "rows": all_rows,
        "selector_contracts": selector_contracts,
        "aggregate": aggregate,
        "primary_gate": primary_gate,
        "simple_rule_sufficient": any_simple_sufficient,
        "decision": (
            "simple_relation_rule_sufficient"
            if any_simple_sufficient
            else "fixed_simple_selectors_not_sufficient"
        ),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "materiality_delta": MATERIALITY_DELTA,
    }
    write_json(output_dir / "rs2_summary.json", summary)
    write_json(output_dir / "rs2_manifest.json", {
        "project_id": "relation_selection_probe",
        "stage": "RS2_simple_selectors",
        "protocol_id": "relation_selection_probe_rs2_v1",
        "datasets": list(datasets),
        "selectors": list(SELECTORS),
        "seeds": list(PILOT_SEEDS),
        "consumer": "Spectral",
        "candidate_pool_reused": True,
        "R_O_pool_reused": True,
        "labels_used_during_fit": False,
        "status": "completed_valid",
    })
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

