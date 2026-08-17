"""Run the frozen S1 opportunity-only Spectral matrix.

S1 deliberately contains no trainable model and no T arm.  Labels are loaded
only for the diagnostic O_pool/O_full graph builders and for post-fit metrics;
F/U/R/Spectral never receive a label vector.  The runner writes one auditable
artifact directory per ``(dataset, arm, seed)`` and a dataset-level summary of
H_pool, H_full, and the matched-budget candidate gap C.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.representation_consumer_probe.protocol import (  # noqa: E402
    CONFIG,
    STRESS_DATASETS,
    CandidatePool,
    IncompleteComputeError,
    build_oracle_full_graph,
    build_oracle_pool_graph,
    build_random_graph,
    build_ungated_graph,
    budget_profile,
    graph_budget_audit,
    jsonable,
    row_l2_normalize,
    sha256_array,
    sha256_file,
    spectral_predict_with_audit,
    symmetrize_graph,
)


S0_ROOT = ROOT / "result/representation_consumer_probe/S0_freeze"
DEFAULT_OUTPUT = ROOT / "result/representation_consumer_probe/S1_oracle_v2"
S1_PROTOCOL_ID = "representation_consumer_probe_s1_opportunity_spectral_v2"
ARMS: tuple[str, ...] = ("F", "U", "R", "O_pool", "O_full")
SEEDS: tuple[int, ...] = (42, 123, 7)
MATERIALITY_DELTA = 0.03


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_bytes(*arrays: np.ndarray, shape: tuple[int, ...] | None = None) -> str:
    digest = hashlib.sha256()
    if shape is not None:
        digest.update(json.dumps(list(shape), separators=(",", ":")).encode())
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def graph_hash(graph: sp.spmatrix) -> str:
    value = sp.csr_matrix(graph, dtype=np.float32)
    return _hash_bytes(value.data, value.indices, value.indptr, shape=value.shape)


def _artifact_hash_manifest(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    root_manifest = root / "artifact_hashes.json"
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != root_manifest):
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "manifest_id": "representation_consumer_probe_s1_artifacts_v1",
        "root": str(root.resolve()),
        "file_count": len(records),
        "files": records,
    }


def _verify_artifact_hashes(root: Path) -> bool:
    path = root / "artifact_hashes.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        listed_paths = {str(record["path"]) for record in manifest.get("files", [])}
        actual_paths = {
            str(target.relative_to(root))
            for target in root.rglob("*")
            if target.is_file() and target != path
        }
        # The manifest is an exact-tree contract.  Unlisted files must not be
        # silently accepted as outside the provenance boundary.
        if listed_paths != actual_paths:
            return False
        for record in manifest.get("files", []):
            target = root / record["path"]
            if not target.is_file() or target.stat().st_size != int(record["size_bytes"]):
                return False
            if sha256_file(target) != record["sha256"]:
                return False
        return int(manifest.get("file_count", -1)) == len(manifest.get("files", []))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _audit_contract_valid(run_dir: Path, dataset: str, arm: str, seed: int) -> bool:
    """Require the run-level semantic audit before reuse or aggregation."""
    audit_path = run_dir / "audit.json"
    config_path = run_dir / "resolved_config.json"
    if not audit_path.exists() or not config_path.exists():
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    try:
        audit_seed = int(audit.get("seed"))
        config_seed = int(config.get("seed"))
    except (TypeError, ValueError):
        return False
    return bool(
        audit.get("audit_ok") is True
        and audit.get("protocol_id") == S1_PROTOCOL_ID
        and audit.get("dataset") == dataset
        and audit.get("arm") == arm
        and audit_seed == int(seed)
        and audit.get("labels_used_during_fit") is False
        and config.get("protocol_id") == S1_PROTOCOL_ID
        and config.get("dataset") == dataset
        and config.get("arm") == arm
        and config_seed == int(seed)
        and config.get("labels_used_during_fit") is False
    )


def _load_candidate_pool(directory: Path) -> CandidatePool:
    with np.load(directory / "candidate_pool.npz", allow_pickle=False) as archive:
        indices = np.asarray(archive["indices"], dtype=np.int64)
        cosine = np.asarray(archive["cosine"], dtype=np.float32)
        positive_counts = np.asarray(archive["positive_counts"], dtype=np.int64)
        effective_budget = np.asarray(archive["effective_budget"], dtype=np.int64)
    if not np.array_equal(effective_budget, np.minimum(CONFIG.budget_cap, positive_counts)):
        raise ValueError(f"effective budget mismatch in {directory}")
    profile_path = directory / "budget_manifest.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    return CandidatePool(indices, cosine, positive_counts, profile)


def _load_labels(path: Path) -> np.ndarray:
    """Read only y from a dataset archive; never materialize its x member."""
    with np.load(path, allow_pickle=True) as archive:
        if "y" not in archive.files:
            raise ValueError(f"missing y in benchmark archive {path}")
        labels = np.asarray(archive["y"]).reshape(-1)
    return labels


def _accuracy_by_optimal_mapping(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_values, true_codes = np.unique(labels, return_inverse=True)
    del true_values
    n_pred = int(np.max(predictions)) + 1 if predictions.size else 0
    contingency = np.zeros((int(np.max(true_codes)) + 1, max(n_pred, 1)), dtype=np.int64)
    np.add.at(contingency, (true_codes, predictions.astype(np.int64)), 1)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = int(contingency[row_ind, col_ind].sum())
    return float(matched / labels.size) if labels.size else 0.0


def _label_ncut(graph: sp.spmatrix, labels: np.ndarray) -> float:
    w = symmetrize_graph(graph).astype(np.float64)
    degrees = np.asarray(w.sum(axis=1)).ravel()
    total = 0.0
    for group in np.unique(labels):
        idx = np.flatnonzero(labels == group)
        outside = np.ones(labels.size, dtype=bool)
        outside[idx] = False
        volume = float(degrees[idx].sum())
        if volume > 0.0:
            total += float(w[idx][:, outside].sum()) / volume
    return float(total)


def graph_diagnostics(
    graph: sp.spmatrix,
    labels: np.ndarray,
    *,
    pool: CandidatePool | None = None,
    oracle_full_directed: sp.spmatrix | None = None,
    oracle_pool_directed: sp.spmatrix | None = None,
) -> dict[str, Any]:
    """Compute post-hoc graph diagnostics; labels never enter the consumer."""
    w = symmetrize_graph(graph)
    degrees = np.asarray(w.sum(axis=1)).ravel().astype(np.float64)
    n_components, component_labels = connected_components(w, directed=False, return_labels=True)
    component_sizes = np.bincount(component_labels) if component_labels.size else np.array([], dtype=np.int64)
    coo = w.tocoo()
    purity = float(np.mean(labels[coo.row] == labels[coo.col])) if coo.nnz else 0.0
    degree_mean = float(np.mean(degrees)) if degrees.size else 0.0
    degree_std = float(np.std(degrees)) if degrees.size else 0.0
    degree_cv = degree_std / degree_mean if degree_mean > 0.0 else 0.0
    result: dict[str, Any] = {
        "edge_count": int(w.nnz),
        "edge_purity": purity,
        "connected_components": int(n_components),
        "isolated_nodes": int(np.sum(degrees <= 0.0)),
        "giant_component_ratio": float(component_sizes.max() / labels.size)
        if component_sizes.size and labels.size
        else 0.0,
        "degree_mean": degree_mean,
        "degree_std": degree_std,
        "degree_cv": degree_cv,
        "ground_truth_ncut": _label_ncut(w, labels),
        "labels_used_for_diagnostic": True,
    }
    if pool is None or oracle_full_directed is None:
        return result

    ideal_graph = sp.csr_matrix(oracle_full_directed)
    pool_graph = sp.csr_matrix(oracle_pool_directed) if oracle_pool_directed is not None else None
    recalls: list[float] = []
    pool_recalls: list[float] = []
    for row, budget in enumerate(pool.effective_budget):
        if int(budget) == 0:
            continue
        ideal_neighbors = set(
            int(col)
            for col in ideal_graph.getrow(row).indices
            if labels[int(col)] == labels[row]
        )
        if not ideal_neighbors:
            continue
        candidate_neighbors = set(
            int(col)
            for col, weight in zip(pool.indices[row], pool.cosine[row], strict=True)
            if int(col) >= 0 and float(weight) > 0.0
        )
        recalls.append(float(len(candidate_neighbors & ideal_neighbors) / len(ideal_neighbors)))
        if pool_graph is not None:
            selected = set(
                int(col)
                for col in pool_graph.getrow(row).indices
                if labels[int(col)] == labels[row]
            )
            pool_recalls.append(float(len(selected & ideal_neighbors) / len(ideal_neighbors)))
    result.update(
        {
            "candidate_recall_at_b_mean": float(np.mean(recalls)) if recalls else 0.0,
            "candidate_recall_at_b_rows": len(recalls),
            "pool_oracle_same_class_recall_at_b_mean": float(np.mean(pool_recalls))
            if pool_recalls
            else 0.0,
            "pool_oracle_same_class_recall_at_b_rows": len(pool_recalls),
        }
    )
    return result


def _metric_summary(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "ARI": float(adjusted_rand_score(labels, predictions)),
        "NMI": float(normalized_mutual_info_score(labels, predictions)),
        "ACC": _accuracy_by_optimal_mapping(labels, predictions),
    }


def feature_only_embedding(h0: np.ndarray) -> np.ndarray:
    """Frozen F arm carrier: raw common H0, with no graph-side normalization."""
    return np.asarray(h0, dtype=np.float32)


def _load_s0_rows() -> dict[str, dict[str, Any]]:
    decision_path = S0_ROOT / "s0_decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(f"missing formal S0 decision: {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "adapter_not_estimable":
        raise RuntimeError(f"S1 requires adapter_not_estimable S0, got {decision.get('status')}")
    if not decision.get("s1_opportunity_only_allowed"):
        raise RuntimeError("formal S0 did not authorize opportunity-only S1")
    manifest = json.loads((S0_ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))
    rows = {str(row["dataset"]): row for row in manifest}
    missing = [dataset for dataset in STRESS_DATASETS if dataset not in rows]
    if missing:
        raise RuntimeError(f"S0 manifest missing datasets: {missing}")
    return rows


def _prepare_dataset(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    source = Path(row["source_path"])
    if not source.exists() or sha256_file(source) != row["source_sha256"]:
        raise RuntimeError(f"source preflight mismatch for {dataset}")
    h0_path = Path(row["H0_path"])
    pool_dir = Path(row["candidate_pool_path"]).parent
    h0 = np.asarray(np.load(h0_path, allow_pickle=False), dtype=np.float32)
    pool = _load_candidate_pool(pool_dir)
    labels = _load_labels(source)
    if h0.shape[0] != labels.size or pool.indices.shape[0] != labels.size:
        raise RuntimeError(f"H0/pool/labels shape mismatch for {dataset}")
    if int(np.unique(labels).size) < 2:
        raise RuntimeError(f"dataset has fewer than two labels: {dataset}")
    oracle_pool = build_oracle_pool_graph(pool, labels)
    oracle_full = build_oracle_full_graph(h0, labels, pool=pool)
    graph_budget_audit(oracle_pool, pool)
    graph_budget_audit(oracle_full, pool)
    return {
        "dataset": dataset,
        "source": source,
        "source_sha256": row["source_sha256"],
        "h0": h0,
        "h0_sha256": sha256_array(h0),
        "pool": pool,
        "labels": labels,
        "K": int(np.unique(labels).size),
        "oracle_pool": oracle_pool,
        "oracle_full": oracle_full,
        "ungated": build_ungated_graph(pool),
    }


def _write_run_hashes(run_dir: Path) -> None:
    _write_json(run_dir / "artifact_hashes.json", _artifact_hash_manifest(run_dir))


def _existing_run_valid(run_dir: Path, dataset: str, arm: str, seed: int) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists() or not _verify_artifact_hashes(run_dir):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    feature_contract_ok = arm != "F" or summary.get("feature_only_input") == "H0_raw"
    return bool(
        summary.get("status") == "completed_valid"
        and summary.get("protocol_id") == S1_PROTOCOL_ID
        and summary.get("dataset") == dataset
        and summary.get("arm") == arm
        and int(summary.get("seed")) == int(seed)
        and feature_contract_ok
        and _audit_contract_valid(run_dir, dataset, arm, seed)
    )


def _run_one(
    data: dict[str, Any],
    arm: str,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    dataset = data["dataset"]
    labels = data["labels"]
    h0 = data["h0"]
    pool: CandidatePool = data["pool"]
    K = data["K"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if _existing_run_valid(run_dir, dataset, arm, seed):
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary["execution_status"] = "reused"
        return summary

    summary: dict[str, Any] = {
        "project_id": CONFIG.project_id,
        "protocol_id": S1_PROTOCOL_ID,
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "consumer": "Spectral",
        "status": "incomplete_compute",
        "execution_status": "queued",
        "source_path": str(data["source"]),
        "source_sha256": data["source_sha256"],
        "H0_sha256": data["h0_sha256"],
        "K": int(K),
        "K_source": CONFIG.k_source,
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "labels_vector_used_in_fit": False,
        "budget_profile": budget_profile(pool),
        "feature_only_input": "H0_raw" if arm == "F" else None,
    }
    try:
        if arm == "F":
            embedding = feature_only_embedding(h0)
            model = KMeans(n_clusters=K, n_init=20, random_state=int(seed))
            predictions = model.fit_predict(embedding).astype(np.int64, copy=False)
            consumer_meta = {
                "consumer": "FeatureOnlyKMeans",
                "K_used_in_representation": False,
                "K_used_in_readout": True,
                "labels_vector_used_in_fit": False,
                "status": "completed",
                "kmeans_n_init": 20,
                "kmeans_random_state": int(seed),
            }
            graph_metrics: dict[str, Any] = {"graph_condition": "none"}
            graph_hash_value = None
            directed = None
            selected = None
        else:
            if arm == "U":
                directed = data["ungated"]
            elif arm == "R":
                directed = build_random_graph(pool, seed=int(seed))
            elif arm == "O_pool":
                directed = data["oracle_pool"]
            elif arm == "O_full":
                directed = data["oracle_full"]
            else:
                raise ValueError(f"unknown S1 arm: {arm}")
            selected = symmetrize_graph(directed)
            predictions, embedding, consumer_meta = spectral_predict_with_audit(selected, K, seed=int(seed))
            graph_hash_value = graph_hash(selected)
            graph_metrics = graph_diagnostics(
                selected,
                labels,
                pool=pool,
                oracle_full_directed=data["oracle_full"],
                oracle_pool_directed=data["oracle_pool"],
            )
            graph_metrics["graph_condition"] = arm
            graph_metrics["directed_edge_count"] = int(sp.csr_matrix(directed).nnz)
            if arm == "U":
                graph_metrics["budget_contract"] = "not_applicable_ungated"
                graph_metrics["directed_row_count_min"] = int(np.diff(sp.csr_matrix(directed).indptr).min())
                graph_metrics["directed_row_count_max"] = int(np.diff(sp.csr_matrix(directed).indptr).max())
            else:
                graph_metrics["budget_audit"] = graph_budget_audit(directed, pool)
            sp.save_npz(run_dir / "directed_graph.npz", sp.csr_matrix(directed), compressed=True)
            sp.save_npz(run_dir / "selected_graph.npz", selected, compressed=True)

        np.save(run_dir / "embedding.npy", embedding)
        np.save(run_dir / "predictions.npy", predictions)
        np.save(run_dir / "labels_true.npy", labels)
        metrics = _metric_summary(labels, predictions)
        summary.update(
            {
                "status": "completed_valid",
                "execution_status": "completed",
                "metrics": metrics,
                "graph_hash": graph_hash_value,
                "graph_diagnostics": graph_metrics,
                "consumer_metadata": consumer_meta,
                "artifacts": {
                    "embedding": str((run_dir / "embedding.npy").resolve()),
                    "predictions": str((run_dir / "predictions.npy").resolve()),
                    "labels_true": str((run_dir / "labels_true.npy").resolve()),
                },
            }
        )
        if arm.startswith("O_"):
            _write_json(
                run_dir / "oracle_manifest.json",
                {
                    "labels_used": True,
                    "purpose": "diagnostic_only",
                    "method_claim": False,
                    "arm": arm,
                    "source_graph": "H0_positive_cosine",
                    "weights_changed_by_oracle": False,
                    "budget_hash": pool.budget_hash,
                },
            )
    except IncompleteComputeError as exc:
        summary.update({"status": "incomplete_compute", "execution_status": "incomplete_compute", "error": str(exc)})
    except Exception as exc:  # preserve the failed key and continue the matrix
        summary.update(
            {
                "status": "incomplete_compute",
                "execution_status": "incomplete_compute",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    _write_json(run_dir / "audit.json", {
        "audit_ok": summary.get("status") == "completed_valid",
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "labels_used_in_oracle_graph": bool(arm.startswith("O_")),
        "arm": arm,
        "dataset": dataset,
        "seed": int(seed),
        "protocol_id": S1_PROTOCOL_ID,
    })
    _write_json(run_dir / "resolved_config.json", {
        "project_id": CONFIG.project_id,
        "protocol_id": S1_PROTOCOL_ID,
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "K": int(K),
        "K_source": CONFIG.k_source,
        "consumer": "Spectral",
        "feature_only_input": "H0_raw" if arm == "F" else None,
        "labels_used_during_fit": False,
        "legal_gpu_pool": [1, 2, 3, 4, 5, 6],
        "forbidden_gpu_ids": [0, 7],
        "execution_device": "cpu_sparse",
    })
    _write_json(run_dir / "summary.json", summary)
    _write_run_hashes(run_dir)
    return summary


def _effect_summary(values: Iterable[float], delta: float = MATERIALITY_DELTA) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(np.mean(array)) if array.size else 0.0
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    positive = int(np.sum(array > 0.0))
    negative = int(np.sum(array < 0.0))
    material_positive = bool(array.size >= 3 and mean >= delta and positive >= 2)
    material_negative = bool(array.size >= 3 and mean <= -delta and negative >= 2)
    return {
        "values": array.tolist(),
        "mean": mean,
        "std": std,
        "positive_seed_count": positive,
        "negative_seed_count": negative,
        "materiality_delta": float(delta),
        "material_positive": material_positive,
        "material_negative": material_negative,
        "classification": "material_positive"
        if material_positive
        else "material_negative"
        if material_negative
        else "observed_small",
    }


def _aggregate_dataset(dataset: str, summaries: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    by_arm = {
        arm: {
            "ARI": _effect_summary(summaries[arm][seed]["metrics"]["ARI"] for seed in SEEDS),
            "NMI": _effect_summary(summaries[arm][seed]["metrics"]["NMI"] for seed in SEEDS),
            "ACC": _effect_summary(summaries[arm][seed]["metrics"]["ACC"] for seed in SEEDS),
        }
        for arm in ARMS
    }
    h_pool = _effect_summary(
        summaries["O_pool"][seed]["metrics"]["ARI"] - summaries["R"][seed]["metrics"]["ARI"]
        for seed in SEEDS
    )
    h_full = _effect_summary(
        summaries["O_full"][seed]["metrics"]["ARI"] - summaries["R"][seed]["metrics"]["ARI"]
        for seed in SEEDS
    )
    candidate_gap = _effect_summary(
        (
            summaries["O_full"][seed]["metrics"]["ARI"]
            - summaries["O_pool"][seed]["metrics"]["ARI"]
        )
        for seed in SEEDS
    )
    support = summaries["R"][SEEDS[0]]["budget_profile"]
    within_present = bool(h_pool["material_positive"])
    gap_present = bool(candidate_gap["material_positive"])
    full_present = bool(h_full["material_positive"])
    if gap_present:
        label = "candidate_family_requires_review"
    elif within_present:
        label = "opportunity_present_within_frozen_candidate_pool"
    elif full_present:
        label = "candidate_family_requires_review"
    else:
        label = "spectral_opportunity_not_observed_s2_conditional"
    return {
        "dataset": dataset,
        "consumer": "Spectral",
        "seed_count": len(SEEDS),
        "arms": by_arm,
        "H_pool": h_pool,
        "H_full": h_full,
        "C_matched_budget_candidate_gap": candidate_gap,
        "within_pool_opportunity": "present" if within_present else "absent",
        "candidate_gap": "present" if gap_present else "absent",
        "full_opportunity": "present" if full_present else "absent",
        "terminal_label_recommendation": label,
        "s2_required_before_negative_conclusion": not (within_present or full_present),
        "support_deficiency": {
            "effective_budget_mean": support["effective_budget_mean"],
            "effective_budget_min": support["effective_budget_min"],
            "fraction_budget_below_cap": support["fraction_budget_below_cap"],
            "fraction_budget_zero": support["fraction_budget_zero"],
            "zero_budget_nodes": support["zero_budget_nodes"],
            "budget_hash": support["effective_budget_hash"],
        },
    }


def run(output_dir: Path = DEFAULT_OUTPUT, *, datasets: tuple[str, ...] = STRESS_DATASETS) -> dict[str, Any]:
    """Execute or resume the full frozen S1 matrix."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_s0_rows()
    _write_json(output_dir / "resolved_config.json", {
        "project_id": CONFIG.project_id,
        "protocol_id": S1_PROTOCOL_ID,
        "s0_decision": str((S0_ROOT / "s0_decision.json").resolve()),
        "arms": list(ARMS),
        "consumer": "Spectral",
        "seeds": list(SEEDS),
        "materiality_delta": MATERIALITY_DELTA,
        "execution_device": "cpu_sparse",
        "labels_used_during_fit": False,
        "oracle_non_tuning": True,
    })
    manifest_rows: list[dict[str, Any]] = []
    dataset_aggregates: dict[str, Any] = {}
    for dataset in datasets:
        data = _prepare_dataset(dataset, rows[dataset])
        dataset_root = output_dir / dataset.replace("/", "_")
        dataset_root.mkdir(parents=True, exist_ok=True)
        _write_json(dataset_root / "budget_profile.json", budget_profile(data["pool"]))
        _write_json(
            dataset_root / "candidate_diagnostics.json",
            {
                "dataset": dataset,
                "labels_used": True,
                "purpose": "diagnostic_only",
                "candidate_pool_profile": data["pool"].profile,
                "budget_profile": budget_profile(data["pool"]),
                "candidate_pool_graph": graph_diagnostics(
                    symmetrize_graph(data["ungated"]),
                    data["labels"],
                    pool=data["pool"],
                    oracle_full_directed=data["oracle_full"],
                    oracle_pool_directed=data["oracle_pool"],
                ),
                "oracle_pool_directed_graph_hash": graph_hash(data["oracle_pool"]),
                "oracle_full_directed_graph_hash": graph_hash(data["oracle_full"]),
            },
        )
        summaries: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
        for seed in SEEDS:
            for arm in ARMS:
                run_dir = dataset_root / f"seed{seed}" / arm
                summary = _run_one(data, arm, seed, run_dir)
                if summary.get("status") == "completed_valid" and not _audit_contract_valid(
                    run_dir, dataset, arm, seed
                ):
                    summary = dict(summary)
                    summary.update(
                        {
                            "status": "incomplete_compute",
                            "execution_status": "incomplete_compute",
                            "error": "semantic_audit_failed",
                        }
                    )
                    _write_json(run_dir / "summary.json", summary)
                    _write_run_hashes(run_dir)
                manifest_rows.append(
                    {
                        "run_key": f"{S1_PROTOCOL_ID}::{dataset}::{arm}::{seed}",
                        "dataset": dataset,
                        "arm": arm,
                        "seed": int(seed),
                        "output_dir": str(run_dir.resolve()),
                        "status": summary.get("status"),
                        "execution_status": summary.get("execution_status"),
                        "labels_used_during_fit": False,
                    }
                )
                if summary.get("status") == "completed_valid":
                    summaries[arm][seed] = summary
        if all(len(summaries[arm]) == len(SEEDS) for arm in ARMS):
            dataset_aggregates[dataset] = _aggregate_dataset(dataset, summaries)
        else:
            dataset_aggregates[dataset] = {
                "dataset": dataset,
                "status": "incomplete_compute",
                "completed_rows": {arm: sorted(summaries[arm]) for arm in ARMS},
            }
        _write_json(output_dir / "s1_manifest.json", manifest_rows)
        _write_json(output_dir / "s1_dataset_aggregates.json", dataset_aggregates)

    completed = sum(row.get("status") == "completed_valid" for row in manifest_rows)
    expected = len(datasets) * len(ARMS) * len(SEEDS)
    valid_datasets = [row for row in dataset_aggregates.values() if row.get("status", "completed_valid") == "completed_valid"]
    summary = {
        "project_id": CONFIG.project_id,
        "protocol_id": S1_PROTOCOL_ID,
        "status": "completed_valid" if completed == expected else "incomplete_compute",
        "execution_device": "cpu_sparse",
        "datasets": list(datasets),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "expected_run_count": expected,
        "completed_valid_run_count": completed,
        "incomplete_run_count": expected - completed,
        "dataset_aggregates": dataset_aggregates,
        "decision_vocabulary": {
            "within_pool_opportunity": "present/absent",
            "candidate_gap": "present/absent",
            "materiality_delta": MATERIALITY_DELTA,
            "S_graph_estimable": False,
        },
        "labels_used_during_fit": False,
        "oracle_non_tuning": True,
        "s2_policy": "conditional_if_spectral_opportunity_not_observed",
        "note": "H_pool/H_full/C are label-derived diagnostic opportunity quantities, not deployable method performance.",
    }
    _write_json(output_dir / "s1_manifest.json", manifest_rows)
    _write_json(output_dir / "s1_dataset_aggregates.json", dataset_aggregates)
    _write_json(output_dir / "s1_summary.json", summary)
    _write_json(output_dir / "artifact_hashes.json", _artifact_hash_manifest(output_dir))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", action="append", choices=STRESS_DATASETS)
    args = parser.parse_args()
    selected = tuple(args.dataset) if args.dataset else STRESS_DATASETS
    result = run(args.output_dir, datasets=selected)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed_valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
