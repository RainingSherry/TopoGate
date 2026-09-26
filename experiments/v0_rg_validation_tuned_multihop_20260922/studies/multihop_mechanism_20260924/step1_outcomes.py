"""Posthoc outcome localization for fixed, validation-selected H1/H8 runs.

This script never fits the TopoGate model or selects a configuration.  It reads
the nine locked pairs of saved predictions and embeddings and then loads full-X
labels solely for descriptive analysis.  Full-X scores are not heldout scores.
"""
from __future__ import annotations

import os

# Set before importing NumPy / BLAS / sklearn.  KMeans repeats are CPU-only.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
import traceback
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


SEEDS = (42, 123, 7)
DEFAULT_DATASETS = ("Tosches", "fabert", "Baron")
READOUT_STATES = tuple(range(10))
N_INIT = 20
PCA_MAX_ROWS = 2500
PCA_MAX_FEATURES = 2048
SCRIPT_VERSION = "step1_outcomes_v1"
RED = "#B5465D"
GREY = "#797A8E"
WHITE = "#F4F0EA"


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_provenance(path: str | Path, with_hash: bool = True) -> dict:
    p = Path(path)
    stat = p.stat()
    result = {"path": str(p), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if with_hash:
        result["sha256"] = sha256(p)
    return result


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def choose2(values: np.ndarray) -> int:
    v = np.asarray(values, dtype=np.int64)
    return int(np.sum(v * (v - 1) // 2, dtype=np.int64))


def group_index(*columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.column_stack(columns)
    _, inverse, counts = np.unique(encoded, axis=0, return_inverse=True, return_counts=True)
    return inverse, counts.astype(np.int64)


def pair_decomposition(y: np.ndarray, h1: np.ndarray, h8: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    """All unordered-pair counts use contingencies; no n-by-n array is formed."""
    y = np.asarray(y).reshape(-1)
    h1 = np.asarray(h1).reshape(-1)
    h8 = np.asarray(h8).reshape(-1)
    if not (len(y) == len(h1) == len(h8)):
        raise ValueError("pair_decomposition arrays have different lengths")
    n = len(y)
    yi, yc = group_index(y)
    ai, ac = group_index(h1)
    bi, bc = group_index(h8)
    yai, yac = group_index(y, h1)
    ybi, ybc = group_index(y, h8)
    abi, abc = group_index(h1, h8)
    yabi, yabc = group_index(y, h1, h8)
    true_same = choose2(yc)
    tp1 = choose2(yac)
    tp8 = choose2(ybc)
    tp_both = choose2(yabc)
    merged_both = choose2(abc) - tp_both
    split1, split8 = true_same - tp1, true_same - tp8
    merge1, merge8 = choose2(ac) - tp1, choose2(bc) - tp8
    corrected_split = tp8 - tp_both
    worsened_split = tp1 - tp_both
    corrected_merge = merge1 - merged_both
    worsened_merge = merge8 - merged_both
    stable_split = true_same - tp1 - tp8 + tp_both
    total_pairs = n * (n - 1) // 2
    true_different = total_pairs - true_same
    stable_correct_different = true_different - merge1 - merge8 + merged_both
    report = {
        "n_samples": n, "total_pairs": total_pairs,
        "h1": {"split": split1, "merge": merge1, "total_error": split1 + merge1},
        "h8": {"split": split8, "merge": merge8, "total_error": split8 + merge8},
        "transition": {
            "corrected_split": corrected_split, "worsened_split": worsened_split,
            "stable_split": stable_split, "corrected_merge": corrected_merge,
            "worsened_merge": worsened_merge, "stable_merge": merged_both,
            "stable_correct_same_class": tp_both,
            "stable_correct_different_class": stable_correct_different,
            "net_error_reduction": corrected_split + corrected_merge - worsened_split - worsened_merge,
        },
    }
    per_sample = {
        "h1_split": yc[yi] - yac[yai],
        "h1_merge": ac[ai] - yac[yai],
        "h8_split": yc[yi] - ybc[ybi],
        "h8_merge": bc[bi] - ybc[ybi],
        "corrected_split": ybc[ybi] - yabc[yabi],
        "worsened_split": yac[yai] - yabc[yabi],
        "stable_split": yc[yi] - yac[yai] - ybc[ybi] + yabc[yabi],
        "corrected_merge": ac[ai] - yac[yai] - abc[abi] + yabc[yabi],
        "worsened_merge": bc[bi] - ybc[ybi] - abc[abi] + yabc[yabi],
        "stable_merge": abc[abi] - yabc[yabi],
    }
    per_sample["net_error_reduction"] = (
        per_sample["corrected_split"] + per_sample["corrected_merge"]
        - per_sample["worsened_split"] - per_sample["worsened_merge"]
    )
    if (min(int(np.min(v)) for k, v in per_sample.items() if k != "net_error_reduction") < 0
            or report["h1"]["total_error"] - report["h8"]["total_error"] != report["transition"]["net_error_reduction"]):
        raise AssertionError("pair contingency identity failed")
    return report, per_sample


def _string_labels(values: np.ndarray) -> np.ndarray:
    def string(value: object) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    return np.asarray([string(value) for value in np.asarray(values).reshape(-1)], dtype=str)


def load_labels(source: Path) -> tuple[np.ndarray, list[str]]:
    """Read original labels, without loading or changing model input features."""
    suffix = source.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(source, "r") as handle:
            if "Y" not in handle:
                raise ValueError(f"original H5 has no Y: {source}")
            raw = np.asarray(handle["Y"])
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            key = next((key for key in ("y", "labels", "label") if key in payload.files), None)
            if key is None:
                raise ValueError(f"original NPZ has no labels: {source}")
            raw = np.asarray(payload[key])
    else:
        raise ValueError(f"unsupported original label source: {source}")
    names, encoded = np.unique(_string_labels(raw), return_inverse=True)
    return encoded.astype(np.int64), names.tolist()


def labels_digest(y: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(y, dtype="<i8").tobytes()).hexdigest()


def evaluate(y: np.ndarray, pred: np.ndarray, class_names: list[str]) -> tuple[dict, list[dict], np.ndarray]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if len(y) != len(pred):
        raise ValueError("prediction/label length mismatch")
    clusters, pi = np.unique(pred, return_inverse=True)
    table = np.zeros((len(class_names), len(clusters)), dtype=np.int64)
    np.add.at(table, (y, pi), 1)
    rr, cc = linear_sum_assignment(-table)
    mapping = {int(c): int(r) for r, c in zip(rr, cc)}
    assigned = np.full(len(clusters), -1, dtype=np.int64)
    for cluster_idx, class_idx in mapping.items():
        assigned[cluster_idx] = class_idx
    mapped = assigned[pi]
    class_rows = []
    for class_idx, name in enumerate(class_names):
        selected_cluster = next((cluster_idx for cluster_idx, mapped_class in mapping.items() if mapped_class == class_idx), None)
        support = int(table[class_idx].sum())
        predicted = int(table[:, selected_cluster].sum()) if selected_cluster is not None else 0
        tp = int(table[class_idx, selected_cluster]) if selected_cluster is not None else 0
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        class_rows.append({
            "class_id": class_idx, "class_name": name,
            "assigned_cluster": str(clusters[selected_cluster]) if selected_cluster is not None else None,
            "support": support, "predicted_count": predicted, "tp": tp,
            "recall": recall, "precision": precision,
            "f1": 2 * recall * precision / (recall + precision) if recall + precision else 0.0,
        })
    metrics = {
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "acc": float(sum(int(table[r, c]) for r, c in zip(rr, cc)) / len(y)),
        "n_pred_clusters": len(clusters),
        "unmatched_cluster_count": len(clusters) - len(mapping),
    }
    return metrics, class_rows, mapped


def _npz_array(path: Path, key: str, ndim: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"{path}: missing {key}, found {payload.files}")
        array = np.asarray(payload[key])
    if array.ndim != ndim:
        raise ValueError(f"{path}: expected ndim={ndim}, found shape={array.shape}")
    return array


def fixed_artifacts(row: dict, seed: int, background_row: dict) -> dict:
    result = {}
    for arm, paths, expected_id in (
        ("h1", row["h1_records"], row["h1_config_id"]),
        ("h8", row["selected_records"], row["selected_config_id"]),
    ):
        if paths is None:
            raise ValueError(f"{row['dataset']} has no locked {arm} records")
        record = Path(paths[str(seed)])
        if record.name != f"record_seed_{seed}.json":
            raise ValueError(f"unexpected record filename: {record}")
        rec = read_json(record)
        if (rec.get("status") != "completed" or rec.get("seed") != seed
                or rec.get("selection_label_scope") != "validation_only"
                or rec.get("test_labels_loaded") is not False
                or rec.get("feature_sha256") != row["feature_sha256"]
                or rec.get("validation_labels_sha256") != row["validation_labels_sha256"]
                or rec.get("labels_used_during_fit") is not False
                or rec.get("K_provenance") != "benchmark_oracle_from_y"):
            raise ValueError(f"locked record provenance mismatch: {record}")
        identity_parts = expected_id.split("_")
        expected_variant = "_".join(part for part in identity_parts if not part.startswith(("q", "a")))
        if str(rec["config"]["hop_variant"]) != expected_variant:
            raise ValueError(f"locked config mismatch: {record}")
        for part, field in ((p, "high_hop_probability_multiplier") if p.startswith("q")
                            else (p, "mix_amplitude_multiplier") for p in identity_parts
                            if p.startswith(("q", "a"))):
            expected_value = float(part[1:].replace("p", "."))
            if not np.isclose(float(rec["config"].get(field, 1.0)), expected_value, rtol=0, atol=1e-12):
                raise ValueError(f"locked config multiplier mismatch: {record}: {field}")
        pred_path = record.with_name(f"predictions_seed_{seed}.npz")
        embed_path = record.with_name(f"embedding_seed_{seed}.npz")
        result[arm] = {
            "record": rec,
            "paths": {"record": record, "prediction": pred_path, "embedding": embed_path},
            "provenance": {key: file_provenance(path) for key, path in
                           (("record", record), ("prediction", pred_path), ("embedding", embed_path))},
        }
    if background_row["multihop_config_id"] != row["selected_config_id"] or background_row["h1_config_id"] != row["h1_config_id"]:
        raise ValueError(f"background config mismatch for {row['dataset']}")
    return result


def repeat_kmeans(y: np.ndarray, embedding: np.ndarray, class_names: list[str], k: int) -> list[dict]:
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains non-finite values")
    rows = []
    for state in READOUT_STATES:
        pred = KMeans(n_clusters=k, random_state=state, n_init=N_INIT, algorithm="lloyd").fit_predict(embedding)
        metrics, _, _ = evaluate(y, pred, class_names)
        rows.append({"random_state": state, "n_init": N_INIT, **metrics})
    return rows


def summarize_readout(rows: list[dict]) -> dict:
    return {metric: {
        "mean": mean(float(row[metric]) for row in rows),
        "std_population": pstdev(float(row[metric]) for row in rows),
        "min": min(float(row[metric]) for row in rows),
        "max": max(float(row[metric]) for row in rows),
    } for metric in ("ari", "nmi", "acc")}


def analyze_item(dataset: str, seed: int, row: dict, background_row: dict, source: Path,
                 out: Path, lock_provenance: dict, background_provenance: dict,
                 script_hash: str) -> dict:
    item_dir = out / "items" / dataset
    item_path = item_dir / f"seed_{seed}.json"
    artifact = fixed_artifacts(row, seed, background_row)
    config = {
        "script_version": SCRIPT_VERSION, "script_sha256": script_hash,
        "dataset": dataset, "seed": seed, "readout_states": READOUT_STATES, "readout_n_init": N_INIT,
        "lock_sha256": lock_provenance["sha256"],
        "background_sha256": background_provenance["sha256"],
        "source_size_bytes": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns,
        "artifact_sha256": {arm: {key: value["sha256"] for key, value in arm_data["provenance"].items()}
                            for arm, arm_data in artifact.items()},
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=list).encode("utf-8")).hexdigest()
    sample_path = item_dir / f"sample_pair_changes_seed_{seed}.csv"
    if item_path.exists():
        previous = read_json(item_path)
        if (previous.get("status") == "completed" and previous.get("config_hash") == config_hash
                and sample_path.exists()):
            return previous
    y, names = load_labels(source)
    if len(y) != row["n_samples"] or len(y) != background_row["n_samples"]:
        raise ValueError(f"original labels row count mismatch for {dataset}")
    scored = {}
    predictions = {}
    for arm in ("h1", "h8"):
        paths = artifact[arm]["paths"]
        pred = _npz_array(paths["prediction"], "predictions", 1)
        embed = _npz_array(paths["embedding"], "embedding", 2)
        if len(pred) != len(y) or embed.shape[0] != len(y):
            raise ValueError(f"saved output length mismatch for {dataset}/{seed}/{arm}")
        record_k = int(artifact[arm]["record"]["n_clusters"])
        if record_k != len(names):
            raise ValueError(f"record K does not equal benchmark label count for {dataset}/{seed}/{arm}")
        metrics, class_rows, mapped = evaluate(y, pred, names)
        expected = background_row["per_seed"][str(seed)]["multihop" if arm == "h8" else "h1"]
        for metric in ("ari", "nmi", "acc"):
            if not np.isclose(metrics[metric], expected[metric], rtol=0, atol=1e-10):
                raise ValueError(f"background score mismatch {dataset}/{seed}/{arm}/{metric}: {metrics[metric]} vs {expected[metric]}")
        readouts = repeat_kmeans(y, embed, names, record_k)
        scored[arm] = {"saved_prediction_metrics": metrics, "class_metrics": class_rows,
                       "readout_repeats": readouts, "readout_distribution": summarize_readout(readouts),
                       "saved_embedding_shape": list(embed.shape)}
        predictions[arm] = pred
    pairs, per_sample = pair_decomposition(y, predictions["h1"], predictions["h8"])
    readout_delta = []
    for left, right in zip(scored["h1"]["readout_repeats"], scored["h8"]["readout_repeats"]):
        if left["random_state"] != right["random_state"]:
            raise AssertionError("readout random states are not paired")
        readout_delta.append({"random_state": left["random_state"],
                              **{f"delta_{metric}": right[metric] - left[metric]
                                 for metric in ("ari", "nmi", "acc")}})
    class_incident = []
    for class_id, class_name in enumerate(names):
        mask = y == class_id
        class_incident.append({
            "class_id": class_id, "class_name": class_name, "support": int(mask.sum()),
            **{key + "_incident": int(np.sum(values[mask], dtype=np.int64)) for key, values in per_sample.items()},
        })
    sample_rows = [{
        "row_index": i, "class_id": int(y[i]), "class_name": names[int(y[i])],
        "h1_cluster": str(predictions["h1"][i]), "h8_cluster": str(predictions["h8"][i]),
        **{key: int(values[i]) for key, values in per_sample.items()},
    } for i in range(len(y))]
    atomic_csv(sample_path, sample_rows, list(sample_rows[0]))
    result = {
        "status": "completed", "script_version": SCRIPT_VERSION, "config_hash": config_hash,
        "config": config, "dataset": dataset, "seed": seed,
        "model_selection_scope": "validation_only", "posthoc_full_x_labels_loaded": True,
        "case_selection_scope": "posthoc_purposive_requested_cases",
        "evaluation_scope": "all_rows_of_original_X_descriptive_not_heldout",
        "full_x_or_test_scores_used_for_configuration_selection": False,
        "K_provenance": "benchmark_oracle_from_y", "n_samples": len(y),
        "class_names": names, "original_label_sha256_int64": labels_digest(y),
        "source_provenance": file_provenance(source, with_hash=False),
        "lock_provenance": lock_provenance,
        "background_provenance": background_provenance,
        "artifacts": {arm: value["provenance"] for arm, value in artifact.items()},
        "h1_config_id": row["h1_config_id"], "h8_config_id": row["selected_config_id"],
        "h1": scored["h1"], "h8": scored["h8"], "readout_paired_delta": readout_delta,
        "pair_errors": pairs, "class_pair_incident": class_incident,
        "sample_pair_changes_csv": str(sample_path),
        "delta_saved_ari": scored["h8"]["saved_prediction_metrics"]["ari"] - scored["h1"]["saved_prediction_metrics"]["ari"],
    }
    atomic_json(item_path, result)
    return result


def shared_raw_pca(source: Path, n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """One raw feature layout shared by both prediction arms; uniform row sample."""
    rng = np.random.default_rng(20260924)
    indices = np.sort(rng.choice(n, size=min(PCA_MAX_ROWS, n), replace=False))
    if source.suffix.lower() in (".h5", ".hdf5"):
        import h5py
        with h5py.File(source, "r") as handle:
            matrix = np.asarray(handle["X"][indices, :], dtype=np.float32)
    elif source.suffix.lower() == ".npz":
        import scipy.sparse as sp
        with np.load(source, allow_pickle=False) as payload:
            if {"data", "indices", "indptr", "shape"}.issubset(payload.files):
                shape = tuple(int(v) for v in payload["shape"])
                full = sp.csr_matrix((payload["data"], payload["indices"], payload["indptr"]), shape=shape)
                matrix = full[indices].toarray().astype(np.float32)
            else:
                key = next((k for k in ("X", "x", "features", "data") if k in payload.files), None)
                if key is None:
                    raise ValueError("NPZ has no raw feature matrix for shared layout")
                matrix = np.asarray(payload[key][indices, :], dtype=np.float32)
    else:
        raise ValueError("shared raw PCA requires H5 or NPZ source")
    feature_indices = np.arange(min(PCA_MAX_FEATURES, matrix.shape[1]), dtype=np.int64)
    matrix = np.nan_to_num(matrix[:, feature_indices], nan=0.0, posinf=0.0, neginf=0.0)
    if min(matrix.shape) < 3:
        raise ValueError("too few rows/features for shared raw PCA")
    coordinates = PCA(n_components=2, svd_solver="randomized", random_state=20260924).fit_transform(matrix)
    meta = {"layout": "one_shared_raw_X_PCA", "source": str(source),
            "sample_rule": "uniform_without_replacement_seed_20260924",
            "n_sampled": len(indices), "feature_rule": "first_up_to_2048_original_columns",
            "n_features": len(feature_indices), "labels_used_for_layout": False,
            "note": "Descriptive raw-X layout; neither model embedding is reprojected into a separate coordinate system."}
    return indices, coordinates, meta


def make_figures(dataset: str, items: list[dict], source: Path, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plot_dir = out / "figures" / dataset
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": WHITE,
                         "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    seeds = [item["seed"] for item in items]
    positions = np.arange(len(items))
    for offset, arm, color in ((-0.17, "h1", GREY), (0.17, "h8", RED)):
        axes[0].bar(positions + offset, [item[arm]["saved_prediction_metrics"]["ari"] for item in items],
                    width=0.32, color=color, label=arm.upper())
    axes[0].set_xticks(positions, seeds)
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Full-X ARI (descriptive)")
    axes[0].legend(frameon=False)
    labels = ["split", "merge"]
    width = 0.32
    for offset, arm, color in ((-width / 2, "h1", GREY), (width / 2, "h8", RED)):
        axes[1].bar(np.arange(2) + offset,
                    [mean(item["pair_errors"][arm][key] / item["pair_errors"]["total_pairs"] for item in items) for key in labels],
                    width=width, color=color)
    axes[1].set_xticks(np.arange(2), labels)
    axes[1].set_ylabel("Mean error pairs / all pairs")
    for i, arm in enumerate(("h1", "h8")):
        all_values = [item[arm]["readout_distribution"]["ari"]["mean"] for item in items]
        axes[2].scatter(np.full(len(all_values), i) + np.linspace(-0.08, 0.08, len(all_values)),
                        all_values, color=GREY if arm == "h1" else RED, s=35)
        axes[2].plot([i - 0.16, i + 0.16], [mean(all_values)] * 2, color="black", lw=1)
    axes[2].set_xticks((0, 1), ("H1", "H8"))
    axes[2].set_ylabel("10-repeat KMeans ARI mean")
    fig.suptitle(f"{dataset}: fixed runs, all seeds; full-X descriptive analysis", fontsize=12)
    stem = plot_dir / "outcome_overview"
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    figure_info = {"overview_png": str(stem.with_suffix(".png")), "overview_pdf": str(stem.with_suffix(".pdf"))}
    try:
        indices, coordinates, meta = shared_raw_pca(source, items[0]["n_samples"])
        atomic_json(plot_dir / "shared_raw_pca_meta.json", meta)
        for item in items:
            sample_path = Path(item["sample_pair_changes_csv"])
            with sample_path.open("r", encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            net = np.asarray([int(records[i]["net_error_reduction"]) for i in indices], dtype=float)
            scale = max(float(np.percentile(np.abs(net), 95)), 1.0)
            fig, ax = plt.subplots(figsize=(5.8, 4.7), constrained_layout=True)
            scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=np.clip(net, -scale, scale),
                                 cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0, vmin=-scale, vmax=scale),
                                 s=4, alpha=0.8, linewidths=0)
            fig.colorbar(scatter, ax=ax, label="Net pair errors corrected per sample (+ improves)")
            ax.set(xlabel="Raw-X PC1", ylabel="Raw-X PC2",
                   title=f"{dataset}, seed {item['seed']}: shared raw-X PCA")
            stem = plot_dir / f"shared_raw_pca_seed_{item['seed']}"
            fig.savefig(stem.with_suffix(".png"), dpi=220)
            fig.savefig(stem.with_suffix(".pdf"))
            plt.close(fig)
        figure_info["shared_raw_pca"] = {"status": "completed", **meta,
                                          "seed_figures": [str(plot_dir / f"shared_raw_pca_seed_{item['seed']}.png") for item in items]}
    except Exception as exc:
        figure_info["shared_raw_pca"] = {
            "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}",
            "embedding_comparison": "omitted; independent embedding projections would not share coordinates",
        }
    return figure_info


def aggregate(items: list[dict], failures: list[dict], figures: dict, background: dict,
              lock_provenance: dict, background_provenance: dict, out: Path) -> dict:
    per_seed, per_class, readout, pair_rows = [], [], [], []
    for item in items:
        base = {"dataset": item["dataset"], "seed": item["seed"],
                "h1_config_id": item["h1_config_id"], "h8_config_id": item["h8_config_id"]}
        per_seed.append({**base, "n_samples": item["n_samples"],
                         **{f"{arm}_{metric}": item[arm]["saved_prediction_metrics"][metric]
                            for arm in ("h1", "h8") for metric in ("ari", "nmi", "acc")},
                         "delta_ari": item["delta_saved_ari"],
                         "pair_error_reduction": item["pair_errors"]["transition"]["net_error_reduction"]})
        for arm in ("h1", "h8"):
            per_class.extend({**base, "arm": arm, **row} for row in item[arm]["class_metrics"])
            readout.extend({**base, "arm": arm, **row} for row in item[arm]["readout_repeats"])
        pair_rows.append({**base, "total_pairs": item["pair_errors"]["total_pairs"],
                          **{f"{arm}_{key}": value for arm in ("h1", "h8") for key, value in item["pair_errors"][arm].items()},
                          **item["pair_errors"]["transition"]})
    for name, rows in (("per_seed.csv", per_seed), ("per_class.csv", per_class),
                       ("readout_repeats.csv", readout), ("pair_errors.csv", pair_rows)):
        if rows:
            atomic_csv(out / name, rows, list(rows[0]))
    positive = [r for r in background["rows"] if r.get("delta_ari_mean") is not None and r["delta_ari_mean"] > 1e-12]
    paired = [r for r in background["rows"] if r.get("delta_ari_mean") is not None]
    result = {
        "status": "completed" if not failures and len(items) > 0 else "partial",
        "script_version": SCRIPT_VERSION, "model_selection_scope": "validation_only",
        "posthoc_full_x_labels_loaded": True,
        "case_selection_scope": "posthoc_purposive_requested_cases",
        "evaluation_scope": "all_rows_of_original_X_descriptive_not_heldout",
        "background_60": {"dataset_count": len(background["rows"]), "paired_count": len(paired),
                          "positive_mean_ari_count": len(positive),
                          "positive_fraction_of_paired": len(positive) / len(paired)},
        "requested_item_count": len(items) + len(failures), "completed_item_count": len(items),
        "failed_item_count": len(failures), "failures": failures,
        "lock_provenance": lock_provenance, "background_provenance": background_provenance,
        "figures": figures,
        "rows": [{"dataset": item["dataset"], "seed": item["seed"],
                  "item_json": str(out / "items" / item["dataset"] / f"seed_{item['seed']}.json"),
                  "delta_saved_ari": item["delta_saved_ari"],
                  "pair_error_reduction": item["pair_errors"]["transition"]["net_error_reduction"]} for item in items],
    }
    atomic_json(out / "step1_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sibling = Path(__file__).resolve().parent.parent / "多跳Top5扩充_无测试集调参_20260923"
    parser.add_argument("--lock", type=Path, default=sibling / "selection_lock.json")
    parser.add_argument("--background", type=Path, default=sibling / "validation_best_multihop_vs_h1_full_x_60.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", nargs="+", default=list(DEFAULT_DATASETS))
    args = parser.parse_args()
    if len(set(args.dataset)) != len(args.dataset):
        parser.error("duplicate --dataset values")
    lock, background = read_json(args.lock), read_json(args.background)
    if (lock.get("selection_scope") != "validation_only" or lock.get("status") != "LOCKED_BEFORE_TEST"
            or background.get("candidate_selection_scope") != "validation_only"
            or background.get("evaluation_scope") != "all_rows_of_original_X"
            or background.get("dataset_count") != 60 or len(background["rows"]) != 60):
        raise ValueError("selection lock or full-X background protocol mismatch")
    lock_rows = {row["dataset"]: row for row in lock["rows"]}
    bg_rows = {row["dataset"]: row for row in background["rows"]}
    if len(lock_rows) != 60 or len(bg_rows) != 60:
        raise ValueError("duplicate/missing dataset in 60-dataset lock or background")
    unknown = set(args.dataset) - set(lock_rows)
    if unknown:
        raise ValueError(f"unknown locked datasets: {sorted(unknown)}")
    lock_provenance = file_provenance(args.lock)
    background_provenance = file_provenance(args.background)
    script_hash = sha256(__file__)
    args.out.mkdir(parents=True, exist_ok=True)
    items, failures, figures = [], [], {}
    for dataset in args.dataset:
        row, bg = lock_rows[dataset], bg_rows[dataset]
        source = Path(bg["source_path"])
        dataset_items = []
        for seed in SEEDS:
            try:
                item = analyze_item(dataset, seed, row, bg, source, args.out,
                                    lock_provenance, background_provenance, script_hash)
                dataset_items.append(item)
                items.append(item)
                print(f"completed {dataset} seed={seed}", flush=True)
            except Exception as exc:
                error = {"dataset": dataset, "seed": seed, "error_type": type(exc).__name__,
                         "error": str(exc), "traceback": traceback.format_exc()}
                failures.append(error)
                atomic_json(args.out / "items" / dataset / f"seed_{seed}.error.json", error)
                print(f"failed {dataset} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        if dataset_items:
            try:
                figures[dataset] = make_figures(dataset, dataset_items, source, args.out)
            except Exception as exc:
                figures[dataset] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                failures.append({"dataset": dataset, "stage": "figures", "error_type": type(exc).__name__, "error": str(exc)})
    result = aggregate(items, failures, figures, background, lock_provenance, background_provenance, args.out)
    print(json.dumps({"status": result["status"], "completed": len(items), "failures": len(failures)}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
