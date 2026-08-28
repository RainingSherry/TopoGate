"""Audited no-fit controls, reuse, and stage-level aggregation.

This module deliberately keeps labels outside every transformation and fit.
Labels are loaded only to obtain the benchmark-known K and to compute metrics
after a representation or no-fit feature matrix has been produced.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import corruption, protocol, runner


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_source_hashes(dataset: str) -> dict[str, str]:
    h0_path = protocol.INPUT_ROOT / dataset / "H0.npy"
    budget_path = protocol.INPUT_ROOT / dataset / "budget_manifest.json"
    label_path = protocol.LABEL_ROOT / dataset / "seed42" / "R" / "labels_true.npy"
    if not all(path.exists() for path in (h0_path, budget_path, label_path)):
        raise FileNotFoundError(f"missing current source artifact for {dataset}")
    return {
        "H0_sha256": sha256(h0_path),
        "budget_manifest_sha256": sha256(budget_path),
        "labels_sha256": sha256(label_path),
    }


def _metrics(labels: np.ndarray, features: np.ndarray, seed: int) -> dict[str, float]:
    k = int(np.unique(labels).size)
    predictions = KMeans(n_clusters=k, n_init=20, random_state=int(seed)).fit_predict(features)
    # ACC is intentionally omitted here: E1/E2 primary gates are ARI-based,
    # and no-fit diagnostics must not add another label-derived decision.
    return {
        "ARI": float(adjusted_rand_score(labels, predictions)),
        "NMI": float(normalized_mutual_info_score(labels, predictions)),
    }


def _audit_corruption(clean: np.ndarray, corrupted: np.ndarray, audit: dict[str, Any]) -> dict[str, Any]:
    changed = np.asarray(audit.get("changed_mask", np.abs(corrupted - clean) > 1e-7), dtype=bool)
    return {
        "effective_changed_coordinate_rate": float(np.mean(changed)),
        "support_change_rate": float(audit.get("support_change_rate", 0.0)),
        "value_change_rate": float(audit.get("value_change_rate", 0.0)),
        "total_absolute_change": float(audit.get("total_absolute_change", np.sum(np.abs(corrupted - clean)))),
        "exact_budget": bool(audit.get("exact_budget", True)),
        "labels_used": False,
    }


def nofit_run(dataset: str, arm: str, seed: int, output_dir: Path) -> dict[str, Any]:
    """Run the frozen H0 -> KMeans diagnostic without an autoencoder fit."""

    protocol.validate_contract()
    if dataset not in protocol.DEVELOPMENT_PANEL or arm not in protocol.E1_ARMS:
        raise ValueError("nofit dataset/arm outside frozen contract")
    if seed not in protocol.PRIMARY_SEEDS:
        raise ValueError("nofit seed outside frozen contract")
    h0, source = runner._load_h0(dataset)
    clean_scaled, _, _ = runner._standardize(h0)
    rng = np.random.default_rng(int(seed))
    corrupted_raw, corruption_audit = corruption.make_corruption(h0, arm, rng)
    corrupted_scaled = ((corrupted_raw - np.mean(h0, axis=0, dtype=np.float64)) /
                        np.where(np.std(h0, axis=0, dtype=np.float64) < 1e-6,
                                 1.0, np.std(h0, axis=0, dtype=np.float64))).astype(np.float32)
    # The feature transformation and KMeans input are fully materialized before
    # the benchmark labels are loaded.  Labels are then used only for K and
    # post-hoc metrics, matching the fit firewall used by the GPU runner.
    labels, label_source = runner._load_labels(dataset)
    metrics = _metrics(labels, corrupted_scaled, seed)
    current = current_source_hashes(dataset)
    summary = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "E1b_nofit",
        "dataset": dataset,
        "role": protocol.ROLE_BY_DATASET[dataset],
        "arm": arm,
        "seed": int(seed),
        "status": "completed_valid" if bool(corruption_audit.get("exact_budget", True)) else "protocol_mismatch",
        "metrics": metrics,
        "K": int(np.unique(labels).size),
        "K_source": "benchmark_oracle_from_y_outer_readout_only",
        "labels_used_during_feature_transform": False,
        "labels_used_for_outer_metrics": True,
        "source": {**source, **label_source, **current, "mean_std_fit_on_clean_H0_only": True},
        "corruption_audit": _audit_corruption(h0, corrupted_raw, corruption_audit),
        "support_semantics": "threshold_defined_dense_H0_only; raw_X_support_not_used",
        "raw_arrays_persisted": False,
    }
    audit = {
        "audit_ok": summary["status"] == "completed_valid",
        "labels_used_during_feature_transform": False,
        "labels_used_for_outer_metrics": True,
        "source_hashes_match_current": all(summary["source"].get(key) == value for key, value in current.items()),
        "raw_arrays_persisted": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "audit.json", audit)
    write_json(output_dir / "resolved_config.json", {
        **protocol.resolved_config(),
        "stage": "E1b_nofit",
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
    })
    return summary


def _source_matches(summary: dict[str, Any], dataset: str) -> bool:
    try:
        expected = current_source_hashes(dataset)
    except FileNotFoundError:
        return False
    source = summary.get("source", {})
    return all(source.get(key) == value for key, value in expected.items())


def existing_valid_run(
    run_dir: Path,
    *,
    dataset: str,
    arm: str,
    objective: str,
    seed: int,
    stage: str,
) -> bool:
    """Validate a resumable run against current sources and exact config."""

    summary_path = run_dir / "summary.json"
    audit_path = run_dir / "audit.json"
    if not summary_path.exists() or not audit_path.exists():
        return False
    try:
        summary = read_json(summary_path)
        audit = read_json(audit_path)
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("status") not in {"completed_valid", "reused"} or audit.get("audit_ok") is not True:
        return False
    if {
        summary.get("dataset"), summary.get("arm"), summary.get("objective"),
        summary.get("seed"), summary.get("stage"),
    } != {dataset, arm, objective, int(seed), stage}:
        return False
    if summary.get("labels_used_during_fit") is not False and stage != "E1b_nofit":
        return False
    return _source_matches(summary, dataset)


def reuse_c2_run(dataset: str, arm: str, seed: int, output_dir: Path, *, stage: str, objective: str) -> dict[str, Any]:
    """Create a compact, hash-checked wrapper around a closed C2 run."""

    if arm not in {"P0_Random", "P2_SupportTarget"} or objective != "O0_GlobalMSE":
        raise ValueError("only C2 P0/P2 O0 controls may be reused")
    source_dir = protocol.C2_ROOT / dataset / arm / f"seed{seed}"
    source_summary_path = source_dir / "summary.json"
    source_audit_path = source_dir / "audit.json"
    if not source_summary_path.exists() or not source_audit_path.exists():
        raise FileNotFoundError(f"missing closed C2 source for {dataset}/{arm}/seed{seed}")
    source_summary = read_json(source_summary_path)
    source_audit = read_json(source_audit_path)
    if source_summary.get("status") != "completed_valid" or source_audit.get("audit_ok") is not True:
        raise ValueError(f"closed C2 source is not valid: {source_summary_path}")
    if not _source_matches(source_summary, dataset):
        raise ValueError(f"closed C2 source hash mismatch: {source_summary_path}")
    summary = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": stage,
        "dataset": dataset,
        "role": protocol.ROLE_BY_DATASET[dataset],
        "arm": arm,
        "objective": objective,
        "seed": int(seed),
        "status": "reused",
        "reused_from": str(source_summary_path),
        "reused_from_protocol_id": source_summary.get("protocol_id"),
        "metrics": dict(source_summary.get("metrics", {})),
        "checkpoint_metrics": source_summary.get("checkpoint_metrics", []),
        "training_metrics": source_summary.get("training_metrics", []),
        "K": source_summary.get("K"),
        "K_source": source_summary.get("K_source"),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "source": dict(source_summary.get("source", {})),
        "support_semantics": "threshold_defined_dense_H0_only; raw_X_support_not_used",
        "raw_arrays_persisted": False,
    }
    audit = {
        "audit_ok": True,
        "reused": True,
        "reused_from": str(source_summary_path),
        "source_hashes_match_current": True,
        "labels_used_during_fit": False,
        "raw_arrays_persisted": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "audit.json", audit)
    write_json(output_dir / "resolved_config.json", {
        **protocol.resolved_config(),
        "stage": stage,
        "dataset": dataset,
        "arm": arm,
        "objective": objective,
        "seed": int(seed),
        "reused_from": str(source_summary_path),
    })
    return summary


def _summary(path: Path) -> dict[str, Any] | None:
    try:
        summary = read_json(path / "summary.json")
        audit = read_json(path / "audit.json")
    except (OSError, json.JSONDecodeError):
        return None
    if audit.get("audit_ok") is not True:
        return None
    return summary


def _metric(summary: dict[str, Any], key: str = "ARI") -> float:
    return float(summary.get("metrics", {}).get(key, float("nan")))


def _complete_seed_cell(rows: Iterable[dict[str, Any] | None]) -> bool:
    values = list(rows)
    return len(values) == len(protocol.PRIMARY_SEEDS) and all(
        row is not None and row.get("status") in {"completed_valid", "reused"} and np.isfinite(_metric(row))
        for row in values
    )


def aggregate_e1(root: Path, nofit_root: Path) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    complete = True
    for dataset in protocol.DEVELOPMENT_PANEL:
        arm_rows: dict[str, list[dict[str, Any] | None]] = {
            arm: [_summary(root / dataset / arm / f"seed{seed}") for seed in protocol.PRIMARY_SEEDS]
            for arm in protocol.E1_ARMS
        }
        nofit_rows: dict[str, list[dict[str, Any] | None]] = {
            arm: [_summary(nofit_root / dataset / arm / f"seed{seed}") for seed in protocol.PRIMARY_SEEDS]
            for arm in protocol.E1_ARMS
        }
        cell_ok = all(_complete_seed_cell(arm_rows[arm]) for arm in protocol.E1_ARMS)
        nofit_ok = all(_complete_seed_cell(nofit_rows[arm]) for arm in protocol.E1_ARMS)
        complete = complete and cell_ok and nofit_ok
        row: dict[str, Any] = {
            "dataset": dataset,
            "role": protocol.ROLE_BY_DATASET[dataset],
            "status": "completed_valid" if cell_ok and nofit_ok else "incomplete_compute",
            "seed_count": len(protocol.PRIMARY_SEEDS),
            "reused_entries": int(sum(s.get("status") == "reused" for s in arm_rows["P0_Random"] + arm_rows["P2_SupportTarget"] if s)),
        }
        if cell_ok and nofit_ok:
            model = {arm: np.asarray([_metric(s) for s in arm_rows[arm]], dtype=float) for arm in protocol.E1_ARMS}
            raw = {arm: np.asarray([_metric(s) for s in nofit_rows[arm]], dtype=float) for arm in protocol.E1_ARMS}
            delta_random = model["P2_SupportTarget"] - model["P0_Random"]
            delta_clean = model["P2_SupportTarget"] - model["Clean"]
            raw_delta_random = raw["P2_SupportTarget"] - raw["P0_Random"]
            row.update({
                "ARI_means": {arm: float(values.mean()) for arm, values in model.items()},
                "nofit_ARI_means": {arm: float(values.mean()) for arm, values in raw.items()},
                "delta_random_values": delta_random.tolist(),
                "delta_clean_values": delta_clean.tolist(),
                "delta_random_mean": float(delta_random.mean()),
                "delta_clean_mean": float(delta_clean.mean()),
                "delta_random_positive_seed_count": int(np.sum(delta_random > 0.0)),
                "delta_clean_positive_seed_count": int(np.sum(delta_clean > 0.0)),
                "nofit_delta_random_mean": float(raw_delta_random.mean()),
                "training_amplification": float(delta_random.mean() - raw_delta_random.mean()),
                "raw_delta_random_values": raw_delta_random.tolist(),
            })
        dataset_rows.append(row)
    nonbio = [row for row in dataset_rows if row["dataset"] in protocol.NONBIOLOGICAL_DATASETS and row["status"] == "completed_valid"]
    g1_wins = [row for row in nonbio if row["delta_random_mean"] >= protocol.MATERIAL_DELTA_ARI and row["delta_clean_mean"] >= protocol.MATERIAL_DELTA_ARI and row["delta_random_positive_seed_count"] >= protocol.E1_MIN_SEED_POSITIVE_COUNT and row["delta_clean_positive_seed_count"] >= protocol.E1_MIN_SEED_POSITIVE_COUNT]
    g2_wins = [row for row in nonbio if row["training_amplification"] >= protocol.MATERIAL_DELTA_ARI]
    gate = {
        "complete_matrix": complete,
        "g1_cross_domain_opportunity": bool(complete and len(g1_wins) >= protocol.E1_MIN_DATASET_COUNT),
        "g1_winner_count": len(g1_wins),
        "g1_winner_datasets": [row["dataset"] for row in g1_wins],
        "g2_training_amplification": bool(complete and len(g2_wins) >= protocol.E1_MIN_DATASET_COUNT),
        "g2_winner_count": len(g2_wins),
        "g2_winner_datasets": [row["dataset"] for row in g2_wins],
        "material_delta_ari": protocol.MATERIAL_DELTA_ARI,
    }
    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "E1_opportunity",
        "status": "completed_valid" if complete else "incomplete_compute",
        "expected_logical_runs": len(protocol.DEVELOPMENT_PANEL) * len(protocol.E1_ARMS) * len(protocol.PRIMARY_SEEDS),
        "dataset_rows": dataset_rows,
        "gate": gate,
        "labels_used_during_fit": False,
        "support_semantics": "threshold_defined_dense_H0_only; raw_X_support_not_used",
    }
    write_json(root / "e1_aggregate.json", result)
    return result


def aggregate_e2(root: Path) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    complete = True
    for dataset in protocol.DEVELOPMENT_PANEL:
        objective_rows: dict[str, dict[str, list[dict[str, Any] | None]]] = {
            objective: {
                arm: [_summary(root / dataset / arm / objective / f"seed{seed}") for seed in protocol.PRIMARY_SEEDS]
                for arm in protocol.E2_CORRUPTIONS
            }
            for objective in protocol.E2_OBJECTIVES
        }
        cell_ok = all(_complete_seed_cell(objective_rows[objective][arm]) for objective in protocol.E2_OBJECTIVES for arm in protocol.E2_CORRUPTIONS)
        complete = complete and cell_ok
        row: dict[str, Any] = {"dataset": dataset, "role": protocol.ROLE_BY_DATASET[dataset], "status": "completed_valid" if cell_ok else "incomplete_compute"}
        if cell_ok:
            deltas: dict[str, float] = {}
            p0_gains: dict[str, float] = {}
            p2_gains: dict[str, float] = {}
            for objective in protocol.E2_OBJECTIVES:
                p0 = np.asarray([_metric(s) for s in objective_rows[objective]["P0_Random"]], dtype=float)
                p2 = np.asarray([_metric(s) for s in objective_rows[objective]["P2_SupportTarget"]], dtype=float)
                deltas[objective] = float(np.mean(p2 - p0))
                if objective != "O0_GlobalMSE":
                    p0_gains[objective] = float(np.mean(p0) - np.mean([_metric(s) for s in objective_rows["O0_GlobalMSE"]["P0_Random"]]))
                    p2_gains[objective] = float(np.mean(p2) - np.mean([_metric(s) for s in objective_rows["O0_GlobalMSE"]["P2_SupportTarget"]]))
            row["delta_by_objective"] = deltas
            row["interaction_by_objective"] = {objective: float(deltas[objective] - deltas["O0_GlobalMSE"]) for objective in protocol.E2_OBJECTIVES[1:]}
            row["p0_gain_by_objective"] = p0_gains
            row["p2_gain_by_objective"] = p2_gains
        dataset_rows.append(row)
    candidates: dict[str, dict[str, Any]] = {}
    for objective in protocol.E2_OBJECTIVES[1:]:
        wins = [row for row in dataset_rows if row["status"] == "completed_valid" and row["interaction_by_objective"][objective] >= protocol.MATERIAL_DELTA_ARI]
        opposing = [row for row in dataset_rows if row["status"] == "completed_valid" and row["interaction_by_objective"][objective] <= -protocol.MATERIAL_DELTA_ARI]
        candidates[objective] = {
            "material_positive_count": len(wins),
            "material_positive_datasets": [row["dataset"] for row in wins],
            "opposing_material_count": len(opposing),
            "opposing_material_datasets": [row["dataset"] for row in opposing],
            "biological_positive_count": sum(row["dataset"] in protocol.BIOLOGICAL_DATASETS for row in wins),
            "nonbiological_positive_count": sum(row["dataset"] in protocol.NONBIOLOGICAL_DATASETS for row in wins),
            "strong_candidate": bool(len(wins) >= protocol.E2_MIN_DATASET_COUNT and len(wins) >= 4 and sum(row["dataset"] in protocol.BIOLOGICAL_DATASETS for row in wins) >= 1 and sum(row["dataset"] in protocol.NONBIOLOGICAL_DATASETS for row in wins) >= 1 and len(opposing) <= protocol.E2_OPPOSING_SIGN_MAX_COUNT),
        }
    generic_o2 = [row for row in dataset_rows if row["status"] == "completed_valid" and row["p0_gain_by_objective"].get("O2_BalancedMSE", -999.0) >= protocol.MATERIAL_DELTA_ARI and row["p2_gain_by_objective"].get("O2_BalancedMSE", -999.0) >= protocol.MATERIAL_DELTA_ARI and abs(row["interaction_by_objective"].get("O2_BalancedMSE", 999.0)) < protocol.MATERIAL_DELTA_ARI]
    strong = [objective for objective, info in candidates.items() if info["strong_candidate"]]
    decision = "CORRUPTION_AWARE_OBJECTIVE_OPPORTUNITY" if strong else "STATIC_CORRUPTION_REPLICATION"
    if not complete:
        decision = "STOP_GENERAL_CORRUPTION"
    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "E2_objective",
        "status": "completed_valid" if complete else "incomplete_compute",
        "dataset_rows": dataset_rows,
        "candidates": candidates,
        "generic_balanced_improvement_datasets": [row["dataset"] for row in generic_o2],
        "decision": decision,
        "labels_used_during_fit": False,
    }
    write_json(root / "e2_aggregate.json", result)
    return result
