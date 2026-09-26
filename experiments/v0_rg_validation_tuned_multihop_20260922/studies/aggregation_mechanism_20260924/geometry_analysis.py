"""Posthoc, readout-independent class geometry of locked H1/H8 embeddings.

No model is fitted here. Original full-X labels are loaded only after the
historical embeddings and predictions have been frozen. All distances are
normalized separately within each embedding; raw coordinates are never paired.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

KNN_K = 15
MAX_ROWS = 4000
SMOKE_ROWS = 256
SAMPLE_SEED = 20260924
VALID_DATASETS = ("Tosches", "Baron", "fabert")
VALID_SEEDS = (42, 123, 7)
VERSION = "geometry_audit_v1"


def _prior_module(previous_root: Path):
    path = previous_root / "step1_outcomes.py"
    if not path.is_file():
        raise FileNotFoundError(f"previous study loader missing: {path}")
    spec = importlib.util.spec_from_file_location("prior_step1_outcomes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_array(path: Path, key: str, ndim: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"{path}: missing {key}")
        array = np.asarray(payload[key])
    if array.ndim != ndim:
        raise ValueError(f"{path}: expected {ndim} dimensions; got {array.shape}")
    return array


def sample_indices(n: int, max_rows: int = MAX_ROWS) -> np.ndarray:
    """Uniform, label-free sample, identical for every arm and run of a dataset."""
    if n < 2 or max_rows < 2:
        raise ValueError("at least two rows required")
    if n <= max_rows:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(SAMPLE_SEED).choice(n, max_rows, replace=False))


def _knn_indices(embedding: np.ndarray, k: int) -> np.ndarray:
    n = len(embedding)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        raise ValueError("at least two sampled rows required")
    # Asking for all non-self neighbors handles duplicate coordinates and ties.
    raw = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", algorithm="auto").fit(embedding).kneighbors(embedding, return_distance=False)
    out = np.empty((n, k_eff), dtype=np.int64)
    for i, neighbors in enumerate(raw):
        filtered = neighbors[neighbors != i]
        if len(filtered) < k_eff:
            raise AssertionError("nearest-neighbor search returned too few non-self rows")
        out[i] = filtered[:k_eff]
    return out


def geometry_metrics(embedding: np.ndarray, labels: np.ndarray, class_names: list[str],
                     k: int = KNN_K) -> tuple[dict, list[dict], list[dict]]:
    """Return normalized global, all-class, and all-pair geometry summaries."""
    z = np.asarray(embedding, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if z.ndim != 2 or len(z) != len(y) or len(z) < 2 or z.shape[1] < 1:
        raise ValueError("invalid embedding/label shape")
    if not np.isfinite(z).all():
        raise ValueError("embedding has non-finite coordinates")
    if np.any((y < 0) | (y >= len(class_names))):
        raise ValueError("class ids outside class_names")
    centered = z - z.mean(axis=0)
    global_spread = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(global_spread) or global_spread <= 0:
        raise ValueError("zero or non-finite global embedding spread")
    neighbors = _knn_indices(z, k)
    k_eff = neighbors.shape[1]
    n_classes = len(class_names)
    centroids = np.full((n_classes, z.shape[1]), np.nan)
    radius = np.full(n_classes, np.nan)
    class_rows: list[dict] = []
    class_masks = [np.flatnonzero(y == c) for c in range(n_classes)]
    for c, members in enumerate(class_masks):
        count = len(members)
        if count == 0:
            class_rows.append({"class_id": c, "class_name": class_names[c], "sample_count": 0,
                               "compactness_over_global_spread": None, "knn_cross_class_fraction": None,
                               "same_class_components": None, "largest_component_fraction": None,
                               "fragmented_fraction": None})
            continue
        points = z[members]
        centroid = points.mean(axis=0)
        centroids[c] = centroid
        radius[c] = float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
        mixing = float(np.mean(y[neighbors[members]] != c))
        # Induced graph of a class within the GLOBAL symmetrized kNN graph.
        local = np.full(len(z), -1, dtype=np.int64)
        local[members] = np.arange(count)
        src = np.repeat(members, k_eff)
        dst = neighbors[members].reshape(-1)
        keep = y[dst] == c
        adjacency = csr_matrix((np.ones(int(keep.sum()), dtype=np.int8),
                                (local[src[keep]], local[dst[keep]])), shape=(count, count))
        component_count, component_id = connected_components(adjacency, directed=False)
        largest = int(np.bincount(component_id).max())
        largest_fraction = largest / count
        class_rows.append({"class_id": c, "class_name": class_names[c], "sample_count": count,
                           "compactness_over_global_spread": float(radius[c] / global_spread),
                           "knn_cross_class_fraction": mixing,
                           "same_class_components": int(component_count),
                           "largest_component_fraction": largest_fraction,
                           "fragmented_fraction": 1.0 - largest_fraction})
    pair_rows: list[dict] = []
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            distance = (float(np.linalg.norm(centroids[a] - centroids[b]))
                        if len(class_masks[a]) and len(class_masks[b]) else None)
            pooled = (float(np.sqrt((radius[a] ** 2 + radius[b] ** 2) / 2))
                      if distance is not None else None)
            pair_rows.append({"class_a_id": a, "class_a_name": class_names[a],
                              "class_b_id": b, "class_b_name": class_names[b],
                              "centroid_distance_over_global_spread": distance / global_spread if distance is not None else None,
                              "pooled_scatter_over_global_spread": pooled / global_spread if pooled is not None else None,
                              "centroid_distance_over_pooled_scatter": distance / max(pooled, global_spread * 1e-12)
                              if distance is not None else None})
    global_row = {"n_sampled": len(z), "embedding_dimensions": z.shape[1], "knn_k_effective": k_eff,
                  "global_spread_rms": global_spread,
                  "mean_knn_cross_class_fraction": float(np.mean(y[neighbors] != y[:, None]))}
    return global_row, class_rows, pair_rows


def pair_merge_errors(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> list[dict]:
    """Full-X predicted co-clustering counts for every true cross-class pair."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    pred = np.asarray(predictions).reshape(-1)
    if len(y) != len(pred) or np.any((y < 0) | (y >= len(class_names))):
        raise ValueError("prediction/label mismatch")
    _, pi = np.unique(pred, return_inverse=True)
    tab = np.zeros((len(class_names), int(pi.max()) + 1), dtype=np.int64)
    np.add.at(tab, (y, pi), 1)
    counts = tab.sum(axis=1)
    rows = []
    for a in range(len(class_names)):
        for b in range(a + 1, len(class_names)):
            merged = int(np.dot(tab[a], tab[b]))
            possible = int(counts[a] * counts[b])
            rows.append({"class_a_id": a, "class_b_id": b, "full_x_class_a_count": int(counts[a]),
                         "full_x_class_b_count": int(counts[b]), "full_x_merge_pairs": merged,
                         "full_x_possible_cross_class_pairs": possible,
                         "full_x_merge_rate": merged / possible if possible else None})
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _figure(out: Path, dataset: str, seed: int, class_rows: list[dict], pair_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"h1": "#697487", "h8": "#C34E63"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    arms = ("h1", "h8")
    for arm in arms:
        cr = [r for r in class_rows if r["arm"] == arm and r["sample_count"]]
        pr = [r for r in pair_rows if r["arm"] == arm and r["centroid_distance_over_pooled_scatter"] is not None]
        x = np.asarray([r["class_id"] for r in cr])
        axes[0].scatter(x, [r["compactness_over_global_spread"] for r in cr], s=19, alpha=.8,
                        color=colors[arm], label=arm.upper())
        axes[1].scatter(x, [r["knn_cross_class_fraction"] for r in cr], s=19, alpha=.8, color=colors[arm])
        axes[2].scatter(x, [r["fragmented_fraction"] for r in cr], s=19, alpha=.8, color=colors[arm])
        axes[3].scatter([r["centroid_distance_over_pooled_scatter"] for r in pr],
                        [r["full_x_merge_rate"] for r in pr], s=19, alpha=.55, color=colors[arm])
    axes[0].set(xlabel="Class ID", ylabel="Class RMS radius / global RMS spread")
    axes[1].set(xlabel="Class ID", ylabel="Cross-class share of 15 nearest neighbors", ylim=(-.04, 1.04))
    axes[2].set(xlabel="Class ID", ylabel="Fraction outside largest same-class component", ylim=(-.04, 1.04))
    axes[3].set(xlabel="Centroid distance / pooled within-class RMS", ylabel="Full-X saved-cluster merge rate")
    axes[0].legend(frameon=False)
    if dataset == "fabert":
        selected = [r for r in pair_rows if r["arm"] == "h1" and r["illustrative_top2_posthoc"]]
        for row in selected:
            if row["centroid_distance_over_pooled_scatter"] is None:
                continue
            axes[3].annotate(f"{row['class_a_name']}/{row['class_b_name']}",
                             (row["centroid_distance_over_pooled_scatter"], row["full_x_merge_rate"]),
                             xytext=(3, 4), textcoords="offset points", fontsize=8)
    fig.suptitle(f"{dataset}, seed {seed}: within-embedding normalized geometry (posthoc full-X labels)")
    for suffix in (".png", ".pdf"):
        fig.savefig(out / f"geometry{suffix}", dpi=210 if suffix == ".png" else None)
    plt.close(fig)


def run(previous_root: Path, out: Path, dataset: str, seed: int, smoke: bool = False) -> dict:
    if dataset not in VALID_DATASETS or seed not in VALID_SEEDS:
        raise ValueError("dataset or seed outside the fixed three-case, three-seed scope")
    prior = _prior_module(previous_root)
    lock_path = previous_root / "inputs" / "selection_lock.json"
    background_path = previous_root / "inputs" / "validation_best_multihop_vs_h1_full_x_60.json"
    lock, background = prior.read_json(lock_path), prior.read_json(background_path)
    if lock.get("status") != "LOCKED_BEFORE_TEST" or lock.get("selection_scope") != "validation_only":
        raise ValueError("selection lock is not validation-only and frozen")
    if background.get("candidate_selection_scope") != "validation_only" or background.get("evaluation_scope") != "all_rows_of_original_X":
        raise ValueError("background provenance is incompatible")
    row = next(r for r in lock["rows"] if r["dataset"] == dataset)
    bg = next(r for r in background["rows"] if r["dataset"] == dataset)
    artifact = prior.fixed_artifacts(row, seed, bg)
    source = Path(bg["source_path"])
    y, names = prior.load_labels(source)
    if len(y) != int(row["n_samples"]):
        raise ValueError("source label length differs from locked sample count")
    idx = sample_indices(len(y), SMOKE_ROWS if smoke else MAX_ROWS)
    out.mkdir(parents=True, exist_ok=True)
    class_rows: list[dict] = []
    pair_rows: list[dict] = []
    arm_summary: dict = {}
    arm_provenance: dict = {}
    for arm in ("h1", "h8"):
        paths = artifact[arm]["paths"]
        pred = _read_array(paths["prediction"], "predictions", 1)
        embed = _read_array(paths["embedding"], "embedding", 2)
        if len(pred) != len(y) or len(embed) != len(y):
            raise ValueError(f"{arm}: artifact length differs from source labels")
        g, classes, pairs = geometry_metrics(embed[idx], y[idx], names)
        merge = pair_merge_errors(y, pred, names)
        for c in classes:
            class_rows.append({"dataset": dataset, "seed": seed, "arm": arm,
                               "full_x_count": int(np.sum(y == c["class_id"])), **c})
        for pair, error in zip(pairs, merge):
            if pair["class_a_id"] != error["class_a_id"] or pair["class_b_id"] != error["class_b_id"]:
                raise AssertionError("class pair ordering changed")
            pair_rows.append({"dataset": dataset, "seed": seed, "arm": arm, **pair,
                              **{k: v for k, v in error.items() if k not in ("class_a_id", "class_b_id")},
                              "illustrative_top2_posthoc": False})
        arm_summary[arm] = g
        arm_provenance[arm] = {k: {"path": str(v["path"]), "sha256": v["sha256"]}
                               for k, v in artifact[arm]["provenance"].items()}
    # Descriptive class-pair selection is explicit and never affects training or inference.
    h1_pairs = sorted((r for r in pair_rows if r["arm"] == "h1"),
                      key=lambda r: (-r["full_x_merge_pairs"], r["class_a_id"], r["class_b_id"]))
    highlighted = {(r["class_a_id"], r["class_b_id"]) for r in h1_pairs[:2]}
    for pair in pair_rows:
        pair["illustrative_top2_posthoc"] = (pair["class_a_id"], pair["class_b_id"]) in highlighted
    _write_csv(out / "per_class.csv", class_rows)
    _write_csv(out / "per_class_pair.csv", pair_rows)
    _write_csv(out / "sample_indices.csv", [{"row_index": int(i)} for i in idx])
    _figure(out, dataset, seed, class_rows, pair_rows)
    summary = {
        "status": "completed", "script_version": VERSION, "dataset": dataset, "seed": seed,
        "smoke": smoke, "arms": ["h1", "h8"], "n_full_x": len(y), "n_sampled": len(idx),
        "sample_rule": "uniform_without_replacement_label_free", "sample_seed": SAMPLE_SEED,
        "sample_cap": SMOKE_ROWS if smoke else MAX_ROWS, "knn_k": KNN_K,
        "normalization": "each_embedding_own_global_RMS_spread; pair_distance_over_own_pooled_RMS_scatter",
        "cross_embedding_raw_coordinates_compared": False,
        "selection_scope": "locked_validation_only", "evaluation_scope": "all_rows_of_original_X_descriptive_not_heldout",
        "posthoc_full_x_labels_loaded": True, "labels_used_for_model_fit": False,
        "illustrative_top2_selection": "two class pairs with largest H1 full-X saved-cluster merge counts; posthoc descriptive only",
        "illustrative_top2_class_pairs": sorted([list(pair) for pair in highlighted]),
        "source_path": str(source), "source_labels_sha256_int64": prior.labels_digest(y),
        "lock_sha256": prior.sha256(lock_path), "background_sha256": prior.sha256(background_path),
        "artifact_provenance": arm_provenance, "geometry": arm_summary,
        "outputs": ["per_class.csv", "per_class_pair.csv", "sample_indices.csv", "geometry.png", "geometry.pdf"],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=VALID_DATASETS)
    parser.add_argument("--seed", required=True, type=int, choices=VALID_SEEDS)
    parser.add_argument("--smoke", action="store_true", help="run the same real-artifact audit with a 256-row geometry sample")
    args = parser.parse_args()
    print(json.dumps({"status": run(args.previous_root, args.out, args.dataset, args.seed, args.smoke)["status"],
                      "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
