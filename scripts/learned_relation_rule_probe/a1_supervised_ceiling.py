"""Run the frozen A1 diagnostic supervised actionable-ceiling probe.

The scorer is deliberately diagnostic: its target is the inherited
``O_pool`` reference membership and therefore is never a deployable,
label-free rule.  Labels are read only by the diagnostic target builder and
by the post-fit benchmark evaluator.  All graph membership and row budgets
come from the audited RS1/S1 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.relation_selection_probe.relation_features import (
    DATASETS,
    GEOMETRY_FEATURES,
    load_edge_table,
    sha256_array,
)
from scripts.representation_consumer_probe.protocol import spectral_predict_with_audit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RS1_ROOT = PROJECT_ROOT / "result/relation_selection_probe/RS1_information/features"
S1_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S1_oracle_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "result/learned_relation_rule_probe/A1_supervised_ceiling"
PRIMARY_DATASETS = ("cnae9", "Campbell", "sms_spam_collection")
SEEDS = (42, 123, 7)
N_FOLDS = 5
MATERIAL_DELTA_ARI = 0.03
CAPTURE_THRESHOLD = 0.25
FULL_VIEW = "Full"
NO_GEOMETRY_VIEW = "No-geometry"
NO_RANK_VIEW = "No-rank"
VIEWS = (FULL_VIEW, NO_GEOMETRY_VIEW, NO_RANK_VIEW)
SCORERS = ("Logistic", "TinyMLP")
NO_RANK_FEATURES = {"cosine_rank", "cosine_percentile"}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def _visible_cuda_is_legal() -> bool:
    """Return true only when CUDA_VISIBLE_DEVICES explicitly excludes 0/7."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return False
    try:
        ids = {int(item.strip()) for item in visible.split(",") if item.strip()}
    except ValueError:
        return False
    return bool(ids) and ids.isdisjoint({0, 7})


def _fit_tiny_mlp(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    """Fit the exact p->32->1 diagnostic scorer, using a legal visible GPU when requested."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("TinyMLP requires torch") from exc
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and _visible_cuda_is_legal() else "cpu")
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train_x = torch.as_tensor((x_train - mean) / scale, dtype=torch.float32, device=device)
    test_x = torch.as_tensor((x_test - mean) / scale, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y_train.astype(np.float32), dtype=torch.float32, device=device).reshape(-1, 1)
    model = nn.Sequential(nn.Linear(train_x.shape[1], 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [max(float((y_train == 0).sum()), 1.0) / max(float((y_train == 1).sum()), 1.0)],
        dtype=torch.float32,
        device=device,
    ))
    model.train()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(test_x)).reshape(-1).detach().cpu().numpy()
    del model, optimizer, train_x, test_x, train_y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(score, dtype=np.float64)


def _fit_logistic(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=300,
            random_state=0,
            solver="lbfgs",
        ),
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64)


def _view_columns(feature_names: tuple[str, ...], view: str) -> np.ndarray:
    if view == FULL_VIEW:
        keep = np.ones(len(feature_names), dtype=bool)
    elif view == NO_GEOMETRY_VIEW:
        keep = np.asarray([name not in set(GEOMETRY_FEATURES) for name in feature_names], dtype=bool)
    elif view == NO_RANK_VIEW:
        keep = np.asarray([name not in NO_RANK_FEATURES for name in feature_names], dtype=bool)
    else:
        raise ValueError(f"unknown A1 feature view: {view}")
    if not np.any(keep):
        raise ValueError(f"empty feature view: {view}")
    return np.flatnonzero(keep)


def _grouped_oof_scores(
    x: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    scorer: str,
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if x.ndim != 2 or target.ndim != 1 or x.shape[0] != target.size or groups.shape != target.shape:
        raise ValueError("A1 scorer inputs are not aligned")
    splitter = GroupKFold(n_splits=N_FOLDS)
    scores = np.full(target.size, np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    test_groups: list[set[int]] = []
    for fold, (train, test) in enumerate(splitter.split(x, target, groups=groups)):
        train_groups = set(int(v) for v in np.unique(groups[train]))
        test_group_set = set(int(v) for v in np.unique(groups[test]))
        if train_groups & test_group_set:
            raise AssertionError("anchor group leaked across GroupKFold")
        test_groups.append(test_group_set)
        if scorer == "Logistic":
            fold_score = _fit_logistic(x[train], target[train], x[test])
        elif scorer == "TinyMLP":
            fold_score = _fit_tiny_mlp(x[train], target[train], x[test], seed + fold)
        else:
            raise ValueError(scorer)
        scores[test] = fold_score
        try:
            fold_ap = float(average_precision_score(target[test], fold_score))
            fold_auc = float(roc_auc_score(target[test], fold_score))
        except ValueError:
            fold_ap, fold_auc = float("nan"), float("nan")
        fold_rows.append({
            "fold": fold,
            "train_edge_count": int(train.size),
            "test_edge_count": int(test.size),
            "train_anchor_count": len(train_groups),
            "test_anchor_count": len(test_group_set),
            "test_coverage": int(test.size),
            "average_precision": fold_ap,
            "auroc": fold_auc,
        })
    if not np.isfinite(scores).all():
        raise AssertionError("OOF prediction coverage is not 100%")
    unique_test = set().union(*test_groups) if test_groups else set()
    audit = {
        "fold_count": N_FOLDS,
        "oof_edge_count": int(scores.size),
        "oof_coverage": float(np.mean(np.isfinite(scores))),
        "oof_coverage_100pct": bool(np.isfinite(scores).all()),
        "anchor_group_count": int(np.unique(groups).size),
        "test_anchor_union_count": len(unique_test),
        "anchor_disjoint": True,
        "labels_used_in_scorer": True,
        "target_name": "pool_reference_membership",
    }
    return scores, fold_rows, audit


def _select_graph_from_scores(rows: np.ndarray, cols: np.ndarray, cosine: np.ndarray, budget: np.ndarray, scores: np.ndarray, n_samples: int) -> tuple[sp.csr_matrix, np.ndarray]:
    selected = np.zeros(rows.size, dtype=bool)
    for row in range(n_samples):
        edge_ids = np.flatnonzero(rows == row)
        b_i = int(budget[row])
        if b_i and edge_ids.size:
            order = np.lexsort((cols[edge_ids], -cosine[edge_ids].astype(np.float64), -scores[edge_ids]))
            selected[edge_ids[order[: min(b_i, edge_ids.size)]]] = True
    counts = np.bincount(rows[selected], minlength=n_samples)
    if not np.array_equal(counts.astype(np.int64), budget.astype(np.int64)):
        raise AssertionError("A1 scorer changed the frozen row budget")
    graph = sp.csr_matrix((cosine[selected].astype(np.float32), (rows[selected], cols[selected])), shape=(n_samples, n_samples))
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph = ((graph + graph.T) * 0.5).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph, selected


def _load_dataset(dataset: str) -> dict[str, Any]:
    feature_root = RS1_ROOT / dataset
    table = load_edge_table(feature_root / "edge_features.npz")
    with np.load(feature_root / "diagnostic_targets.npz", allow_pickle=False) as archive:
        target = np.asarray(archive["pool_reference_membership"], dtype=np.int64)
    meta = json.loads((feature_root / "feature_metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((feature_root / "diagnostic_audit.json").read_text(encoding="utf-8"))
    labels_path = S1_ROOT.parent / "S1_oracle_v2" / dataset / "seed42" / "R" / "labels_true.npy"
    labels = np.asarray(np.load(labels_path), dtype=np.int64)
    if labels.size != table.n_samples or target.size != table.rows.size:
        raise ValueError(f"A1 source shape mismatch for {dataset}")
    if meta.get("labels_used") is not False or audit.get("labels_used_in_features") is not False:
        raise ValueError(f"A1 feature source is not label-free: {dataset}")
    k = int(np.unique(labels).size)
    reference: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        run = S1_ROOT.parent / "S1_oracle_v2" / dataset / f"seed{seed}" / "R"
        oracle = S1_ROOT.parent / "S1_oracle_v2" / dataset / f"seed{seed}" / "O_pool"
        if not (run / "predictions.npy").exists() or not (oracle / "predictions.npy").exists():
            raise FileNotFoundError(f"missing audited S1 R/O_pool artifact for {dataset}, seed {seed}")
        r_pred = np.asarray(np.load(run / "predictions.npy"), dtype=np.int64)
        o_pred = np.asarray(np.load(oracle / "predictions.npy"), dtype=np.int64)
        reference[seed] = {
            "r_ari": float(adjusted_rand_score(labels, r_pred)),
            "r_nmi": float(normalized_mutual_info_score(labels, r_pred)),
            "o_ari": float(adjusted_rand_score(labels, o_pred)),
            "o_nmi": float(normalized_mutual_info_score(labels, o_pred)),
            "r_graph_hash": sha256_file(run / "selected_graph.npz"),
            "o_graph_hash": sha256_file(oracle / "selected_graph.npz"),
        }
    return {
        "dataset": dataset,
        "table": table,
        "target": target,
        "labels": labels,
        "K": k,
        "reference": reference,
        "source_files": {
            "edge_features": sha256_file(feature_root / "edge_features.npz"),
            "diagnostic_targets": sha256_file(feature_root / "diagnostic_targets.npz"),
            "feature_metadata": sha256_file(feature_root / "feature_metadata.json"),
            "labels": sha256_file(labels_path),
        },
    }


def engineering_preflight(datasets: tuple[str, ...] = PRIMARY_DATASETS) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        data = _load_dataset(dataset)
        table = data["table"]
        target = data["target"]
        fold = next(GroupKFold(n_splits=N_FOLDS).split(table.features, target, groups=table.rows))
        train, test = fold
        rows.append({
            "dataset": dataset,
            "edge_count": int(table.rows.size),
            "anchor_count": int(np.unique(table.rows).size),
            "budget_capacity_sufficient": bool(np.all(np.bincount(table.rows, minlength=table.n_samples) >= table.budget)),
            "target_binary": bool(np.unique(target).size == 2),
            "groupkfold_anchor_disjoint": bool(set(table.rows[train]) .isdisjoint(set(table.rows[test]))),
            "oof_contract": "5-fold GroupKFold by anchor; full coverage asserted during run",
            "labels_used_in_feature_extraction": False,
            "labels_used_in_diagnostic_target": True,
            "r_reference_graphs_present": all(data["reference"][seed]["r_graph_hash"] for seed in SEEDS),
            "o_pool_reference_graphs_present": all(data["reference"][seed]["o_graph_hash"] for seed in SEEDS),
        })
    passed = all(
        bool(row["budget_capacity_sufficient"])
        and bool(row["target_binary"])
        and bool(row["groupkfold_anchor_disjoint"])
        and row["labels_used_in_feature_extraction"] is False
        and bool(row["r_reference_graphs_present"])
        and bool(row["o_pool_reference_graphs_present"])
        for row in rows
    )
    return {"status": "completed_valid" if passed else "protocol_mismatch", "datasets": rows, "oof_coverage_required": 1.0}


def run(output_dir: Path = DEFAULT_OUTPUT, datasets: tuple[str, ...] = PRIMARY_DATASETS) -> dict[str, Any]:
    preflight = engineering_preflight(datasets)
    if preflight["status"] != "completed_valid":
        raise RuntimeError(f"A1 engineering preflight failed: {preflight}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    shortcut_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {"datasets": {}, "base_commit": "c80877cf904e41950315d37b95374825c33a7362"}
    for dataset in datasets:
        data = _load_dataset(dataset)
        table = data["table"]
        target = data["target"]
        labels = data["labels"]
        source_manifest["datasets"][dataset] = data["source_files"]
        h_pool = float(np.mean([data["reference"][seed]["o_ari"] - data["reference"][seed]["r_ari"] for seed in SEEDS]))
        configs: list[dict[str, Any]] = []
        for scorer in SCORERS:
            for view in VIEWS:
                columns = _view_columns(table.feature_names, view)
                scores, fold_metrics, fold_audit = _grouped_oof_scores(
                    table.features[:, columns], target, table.rows, scorer, seed=42
                )
                for row in fold_metrics:
                    fold_rows.append({"dataset": dataset, "scorer": scorer, "view": view, **row})
                graph, selected = _select_graph_from_scores(
                    table.rows, table.cols, table.cosine, table.budget, scores, table.n_samples
                )
                seed_metrics: list[dict[str, Any]] = []
                for seed in SEEDS:
                    pred, embedding, consumer_meta = spectral_predict_with_audit(graph, data["K"], seed=seed)
                    ari = float(adjusted_rand_score(labels, pred))
                    nmi = float(normalized_mutual_info_score(labels, pred))
                    seed_metrics.append({"seed": seed, "ari": ari, "nmi": nmi, "embedding_finite": bool(np.isfinite(embedding).all()), "active_nodes": int(consumer_meta["active_nodes"])})
                delta_values = [row["ari"] - data["reference"][row["seed"]]["r_ari"] for row in seed_metrics]
                delta_mean = float(np.mean(delta_values))
                capture = delta_mean / h_pool if h_pool >= MATERIAL_DELTA_ARI else None
                ap = float(average_precision_score(target, scores))
                auc = float(roc_auc_score(target, scores))
                shortcut_rows.append({
                    "dataset": dataset,
                    "scorer": scorer,
                    "view": view,
                    "feature_count": int(columns.size),
                    "average_precision": ap,
                    "ap_lift": ap - float(np.mean(target)),
                    "auroc": auc,
                    "delta_sup_mean": delta_mean,
                    "delta_sup_min": float(np.min(delta_values)),
                    "delta_sup_max": float(np.max(delta_values)),
                    "capture_sup": "" if capture is None else capture,
                    "oof_coverage": fold_audit["oof_coverage"],
                    "anchor_disjoint": fold_audit["anchor_disjoint"],
                    "labels_used_in_scorer": fold_audit["labels_used_in_scorer"],
                })
                configs.append({"scorer": scorer, "view": view, "delta_sup_mean": delta_mean, "capture_sup": capture, "seed_metrics": seed_metrics})
        best = sorted(configs, key=lambda row: (-row["delta_sup_mean"], row["scorer"], row["view"]))[0]
        dataset_rows.append({
            "dataset": dataset,
            "H_pool_mean": h_pool,
            "best_scorer": best["scorer"],
            "best_view": best["view"],
            "Delta_sup_mean": best["delta_sup_mean"],
            "Capture_sup": "" if best["capture_sup"] is None else best["capture_sup"],
            "material_delta_pass": bool(best["delta_sup_mean"] >= MATERIAL_DELTA_ARI),
            "capture_gate_pass": bool(best["capture_sup"] is not None and best["capture_sup"] >= CAPTURE_THRESHOLD),
            "seed_spread": float(np.ptp([row["ari"] - data["reference"][row["seed"]]["r_ari"] for row in best["seed_metrics"]])),
        })
    material_rows = [row for row in dataset_rows if row["material_delta_pass"]]
    captures = [float(row["Capture_sup"]) for row in material_rows if row["Capture_sup"] != ""]
    gate_pass = len(material_rows) >= 2 and bool(captures) and float(np.median(captures)) >= CAPTURE_THRESHOLD
    decision = {
        "stage": "A1",
        "status": "completed_valid" if gate_pass else "predictable_reference_not_actionable_for_selection",
        "primary_gate_pass": bool(gate_pass),
        "material_datasets": [row["dataset"] for row in material_rows],
        "material_dataset_count": len(material_rows),
        "median_capture_sup_material": float(np.median(captures)) if captures else None,
        "next_stage_authorized": bool(gate_pass),
        "authorized_next_stage": "A2" if gate_pass else None,
        "terminal_reason": None if gate_pass else "A1 supervised actionable ceiling did not meet frozen 2-of-3 and median-capture gate",
        "diagnostic_supervision": True,
        "deployable_label_free_rule": False,
    }
    audit = {
        "project_id": "learned_relation_rule_probe",
        "stage": "A1",
        "status": "completed_valid",
        "preflight": preflight,
        "fold_count": N_FOLDS,
        "datasets": list(datasets),
        "scorers": list(SCORERS),
        "views": list(VIEWS),
        "oof_coverage_100pct": all(float(row["oof_coverage"]) == 1.0 for row in shortcut_rows),
        "anchor_disjoint_all": all(bool(row["anchor_disjoint"]) for row in shortcut_rows),
        "labels_used_in_feature_extraction": False,
        "labels_used_in_diagnostic_target_builder": True,
        "labels_used_in_scorer": True,
        "labels_used_for_outer_metrics": True,
        "candidate_pool_frozen": True,
        "row_budget_frozen": True,
        "reference_arm": "R_inherited_matched_random",
        "oracle_arm": "O_pool_inherited_diagnostic_only",
        "gpu_firewall": {"forbidden": [0, 7], "visible_cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "tiny_mlp_gpu_enabled": bool(_visible_cuda_is_legal())},
        "issues": [],
    }
    _write_csv(output_dir / "a1_fold_metrics.csv", fold_rows)
    _write_csv(output_dir / "a1_shortcut_audit.csv", shortcut_rows)
    _write_csv(output_dir / "a1_dataset_summary.csv", dataset_rows)
    write_json(output_dir / "decision.json", decision)
    write_json(output_dir / "audit.json", audit)
    write_json(output_dir / "preflight.json", preflight)
    write_json(output_dir / "source_manifest.json", source_manifest)
    write_json(output_dir / "resolved_config.json", {
        "project_id": "learned_relation_rule_probe",
        "protocol_id": "learned_relation_rule_probe_a0_v1",
        "stage": "A1",
        "datasets": list(datasets),
        "scorers": list(SCORERS),
        "views": list(VIEWS),
        "seeds": list(SEEDS),
        "folds": N_FOLDS,
        "material_delta_ari": MATERIAL_DELTA_ARI,
        "capture_threshold": CAPTURE_THRESHOLD,
        "labels_used_during_fit": False,
        "diagnostic_target": "pool_reference_membership_from_inherited_O_pool",
        "reference": "R_inherited_matched_random_read_only",
    })
    write_json(output_dir / "run_manifest.json", {
        "project_id": "learned_relation_rule_probe",
        "stage": "A1",
        "status": "completed_valid",
        "dataset_count": len(datasets),
        "scorer_count": len(SCORERS),
        "view_count": len(VIEWS),
        "fold_count": N_FOLDS,
        "job_count": len(datasets) * len(SCORERS) * len(VIEWS) * N_FOLDS,
        "raw_artifacts_published": False,
    })
    write_json(output_dir / "artifact_hashes.json", {
        "stage": "A1",
        "files": {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"},
        "raw_artifacts_included": False,
    })
    report = [
        "# A1 supervised actionable ceiling",
        "",
        "This is diagnostic supervision against inherited O_pool reference membership; it is not a label-free method.",
        "",
        f"Gate: `{decision['status']}`; material datasets={decision['material_dataset_count']}/3; median capture={decision['median_capture_sup_material']}",
        "",
        "| dataset | H_pool | best scorer | best view | Delta_sup | Capture_sup | material |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in dataset_rows:
        report.append(f"| {row['dataset']} | {row['H_pool_mean']:.6f} | {row['best_scorer']} | {row['best_view']} | {row['Delta_sup_mean']:.6f} | {row['Capture_sup']} | {row['material_delta_pass']} |")
    report += [
        "",
        "The full/no-geometry/no-rank rows and five grouped OOF folds are in the compact local CSV audit. Raw scores, graphs, embeddings and predictions are not publication artifacts.",
    ]
    (PROJECT_ROOT / "reports/learned_relation_rule_probe/A1_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"decision": decision, "audit": audit, "dataset_summary": dataset_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", action="append", choices=PRIMARY_DATASETS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    datasets = tuple(args.dataset) if args.dataset else PRIMARY_DATASETS
    if args.preflight_only:
        print(json.dumps(engineering_preflight(datasets), indent=2, sort_keys=True, default=_json_default))
    else:
        print(json.dumps(run(args.output_dir, datasets), indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
