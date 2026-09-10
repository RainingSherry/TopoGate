"""Inductive KMeans and PCA+KMeans baselines on frozen TopoGate splits.

This script deliberately keeps labels out of fitting. They are read only for
validation configuration selection and final test metrics. It writes each
dataset atomically enough to resume after an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


SEEDS = (42, 123, 7)
PCA_DIMENSIONS = (20, 50, 100)


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    matrix = np.zeros((labels_true.size, labels_pred.size), dtype=np.int64)
    for i, value in enumerate(labels_true):
        mask = y_true == value
        for j, cluster in enumerate(labels_pred):
            matrix[i, j] = int(np.count_nonzero(y_pred[mask] == cluster))
    rows, cols = linear_sum_assignment(matrix.max() - matrix)
    return float(matrix[rows, cols].sum() / y_true.size)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "acc": acc(y, pred),
    }


def load_npz(path: Path) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if "x" not in data or "y" not in data:
        raise ValueError("expected NPZ keys x and y")
    return data["x"], np.asarray(data["y"]).reshape(-1)


def fit_projection(x_train, dim: int):
    max_dim = min(dim, x_train.shape[0] - 1, x_train.shape[1])
    if max_dim < 1:
        raise ValueError("training data has no valid projection dimension")
    if sparse.issparse(x_train):
        return TruncatedSVD(n_components=max_dim, random_state=42)
    return PCA(n_components=max_dim, random_state=42)


def evaluate_dataset(dataset_dir: Path, output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_file = output_dir / "kmeans_baselines_final.json"
    if final_file.exists() and not force:
        return
    experiment = json.loads((dataset_dir / "experiment.json").read_text(encoding="utf-8"))
    split = json.loads((dataset_dir / "split.json").read_text(encoding="utf-8"))
    x, y = load_npz(Path(experiment["dataset"]["path"]))
    train = np.asarray(split["train"], dtype=np.int64)
    val = np.asarray(split["validation"], dtype=np.int64)
    test = np.asarray(split["test"], dtype=np.int64)
    k = int(experiment["dataset"]["n_clusters"])
    started = time.time()
    x_train, x_val, x_test = x[train], x[val], x[test]
    y_val, y_test = y[val], y[test]
    selections: list[dict] = []
    candidates = []
    for dim in PCA_DIMENSIONS:
        if min(x_train.shape[0] - 1, x_train.shape[1]) < 1:
            continue
        projection = fit_projection(x_train, dim)
        z_train, z_val = projection.fit_transform(x_train), projection.transform(x_val)
        candidate = {"pca_dim_requested": dim, "pca_dim_actual": int(z_train.shape[1]), "seed42": {}}
        for seed in (42,):
            model = KMeans(n_clusters=k, n_init=20, random_state=seed)
            pred = model.fit_predict(z_train)
            candidate["seed42"] = {"validation": metrics(y_val, model.predict(z_val))}
        candidates.append(candidate)
    candidates.sort(key=lambda c: c["seed42"]["validation"]["ari"], reverse=True)
    for candidate in candidates[:2]:
        dim = candidate["pca_dim_actual"]
        projection = fit_projection(x_train, dim)
        z_train, z_val = projection.fit_transform(x_train), projection.transform(x_val)
        candidate["refinement"] = {}
        for seed in (123, 7):
            model = KMeans(n_clusters=k, n_init=20, random_state=seed)
            model.fit(z_train)
            candidate["refinement"][str(seed)] = {"validation": metrics(y_val, model.predict(z_val))}
        aris = [candidate["seed42"]["validation"]["ari"]] + [candidate["refinement"][str(s)]["validation"]["ari"] for s in (123, 7)]
        candidate["validation_ari_mean"] = float(np.mean(aris))
    winner = max(candidates[:2], key=lambda c: c.get("validation_ari_mean", -np.inf))
    selections.append({"method": "PCA+KMeans", "winner": winner, "all_candidates": candidates})
    methods = {"KMeans": None, "PCA+KMeans": int(winner["pca_dim_actual"])}
    results = {"dataset": experiment["dataset"]["dataset_id"], "protocol": "strict_split_kmeans_baselines_v1", "split_hash": experiment["split_hash"], "data_hash": experiment["data_hash"], "k": k, "labels_fit": False, "labels_validation_selection": True, "labels_test_selection": False, "selection": selections, "final": {}}
    x_fit = x[np.concatenate([train, val])]
    for method, dim in methods.items():
        if dim is None:
            transformer = StandardScaler(with_mean=not sparse.issparse(x_fit))
        else:
            transformer = fit_projection(x_fit, dim)
        z_fit, z_test = transformer.fit_transform(x_fit), transformer.transform(x_test)
        records = []
        for seed in SEEDS:
            model = KMeans(n_clusters=k, n_init=20, random_state=seed)
            model.fit(z_fit)
            prediction = model.predict(z_test)
            pred_file = output_dir / f"{method.replace('+', '_')}_test_seed{seed}.npy"
            tmp = pred_file.with_suffix(".tmp.npy")
            np.save(tmp, prediction)
            os.replace(tmp, pred_file)
            records.append({"seed": seed, "metrics": metrics(y_test, prediction), "prediction": pred_file.name})
        results["final"][method] = records
    results["elapsed_seconds"] = time.time() - started
    atomic_json(final_file, results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in sorted(args.datasets_root.iterdir()):
        if path.is_dir() and (path / "experiment.json").exists() and (path / "split.json").exists():
            try:
                evaluate_dataset(path, args.output_root / path.name, args.force)
                print(json.dumps({"dataset": path.name, "status": "complete"}), flush=True)
            except Exception as exc:
                atomic_json(args.output_root / path.name / "kmeans_baselines_failure.json", {"dataset": path.name, "error": repr(exc)})
                print(json.dumps({"dataset": path.name, "status": "failed", "error": repr(exc)}), flush=True)


if __name__ == "__main__":
    main()
