"""Frozen-embedding readout audit for the three multihop mechanism cases.

The only fitted object here is KMeans on an already saved embedding. Labels are
opened after predictions have been saved in memory and are used for evaluation.
The full-X scores are descriptive, not independent heldout estimates.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


DATASETS = ("Tosches", "Baron", "fabert")
SEEDS = (42, 123, 7)
STATES = tuple(range(20))
N_INIT = 20
VERSION = "readout_analysis_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def npz_array(path: Path, key: str, ndim: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as packed:
        if key not in packed:
            raise ValueError(f"{path}: missing {key}")
        value = np.asarray(packed[key])
    if value.ndim != ndim:
        raise ValueError(f"{path}: expected {ndim} dimensions, got {value.shape}")
    return value


def _label_string(value: object) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def load_labels(source: Path) -> tuple[np.ndarray, list[str]]:
    if source.suffix.lower() in (".h5", ".hdf5"):
        import h5py
        with h5py.File(source, "r") as handle:
            raw = np.asarray(handle["Y"])
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as packed:
            key = next((name for name in ("y", "labels", "label") if name in packed), None)
            if key is None:
                raise ValueError(f"{source}: no label key")
            raw = np.asarray(packed[key])
    else:
        raise ValueError(f"unsupported label source: {source}")
    names, encoded = np.unique(np.asarray([_label_string(x) for x in raw.reshape(-1)]),
                               return_inverse=True)
    return encoded.astype(np.int64), names.tolist()


def contingency(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """True-class by predicted-cluster counts; cluster IDs need not be contiguous."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if len(y) != len(pred) or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("invalid labels or prediction length")
    _, cluster_index = np.unique(pred, return_inverse=True)
    table = np.zeros((n_classes, int(cluster_index.max()) + 1), dtype=np.int64)
    np.add.at(table, (y, cluster_index), 1)
    return table


def pair_rates(y: np.ndarray, pred: np.ndarray, class_names: list[str]) -> tuple[list[dict], list[dict]]:
    """Split and merge rates from one contingency, without an n-by-n matrix.

    Split(c) = true same-class pairs in different predicted clusters / C(n_c,2).
    Merge(a,b) = true cross-class pairs in one predicted cluster / (n_a*n_b).
    Rates with zero denominator are null and excluded from selection.
    """
    counts = contingency(y, pred, len(class_names))
    supports = counts.sum(axis=1)
    class_rows: list[dict] = []
    pair_rows: list[dict] = []
    for c, name in enumerate(class_names):
        denom = int(supports[c] * (supports[c] - 1) // 2)
        together = int(np.sum(counts[c] * (counts[c] - 1) // 2, dtype=np.int64))
        split = denom - together
        class_rows.append({"class_id": c, "class_name": name, "support": int(supports[c]),
                           "true_pair_count": denom, "split_count": split,
                           "split_rate": split / denom if denom else None})
    for a, b in combinations(range(len(class_names)), 2):
        denom = int(supports[a] * supports[b])
        merged = int(np.dot(counts[a], counts[b]))
        pair_rows.append({"class_a_id": a, "class_a_name": class_names[a],
                          "class_b_id": b, "class_b_name": class_names[b],
                          "support_a": int(supports[a]), "support_b": int(supports[b]),
                          "true_pair_count": denom, "merge_count": merged,
                          "merge_rate": merged / denom if denom else None})
    return class_rows, pair_rows


def _locked_inputs(previous_root: Path, dataset: str, seed: int) -> dict:
    lock_path = previous_root / "inputs" / "selection_lock.json"
    background_path = previous_root / "inputs" / "validation_best_multihop_vs_h1_full_x_60.json"
    lock, background = read_json(lock_path), read_json(background_path)
    if lock.get("status") != "LOCKED_BEFORE_TEST" or lock.get("selection_scope") != "validation_only":
        raise ValueError("historical selection lock is invalid")
    if background.get("candidate_selection_scope") != "validation_only":
        raise ValueError("background selection scope is invalid")
    row = next((r for r in lock["rows"] if r["dataset"] == dataset), None)
    bg = next((r for r in background["rows"] if r["dataset"] == dataset), None)
    if row is None or bg is None or row["selected_config_id"] != bg["multihop_config_id"]:
        raise ValueError("dataset/config absent from frozen lock or background")
    if row["h1_config_id"] != bg["h1_config_id"]:
        raise ValueError("H1 configuration differs from frozen background")
    artifacts = {}
    for arm, record_map, config_id in (("h1", row["h1_records"], row["h1_config_id"]),
                                       ("h8", row["selected_records"], row["selected_config_id"])):
        record_path = Path(record_map[str(seed)])
        if record_path.name != f"record_seed_{seed}.json":
            raise ValueError(f"unexpected historical record filename: {record_path}")
        record = read_json(record_path)
        if (record.get("status") != "completed" or record.get("seed") != seed
                or record.get("selection_label_scope") != "validation_only"
                or record.get("test_labels_loaded") is not False
                or record.get("labels_used_during_fit") is not False
                or record.get("feature_sha256") != row["feature_sha256"]
                or record.get("validation_labels_sha256") != row["validation_labels_sha256"]
                or record.get("K_provenance") != "benchmark_oracle_from_y"):
            raise ValueError(f"historical record failed provenance: {record_path}")
        identity = config_id.split("_")
        variant = "_".join(part for part in identity if not part.startswith(("q", "a")))
        if record["config"].get("hop_variant") != variant:
            raise ValueError(f"historical variant differs from lock: {record_path}")
        for part in identity:
            if part.startswith(("q", "a")):
                field = ("high_hop_probability_multiplier" if part.startswith("q")
                         else "mix_amplitude_multiplier")
                expected = float(part[1:].replace("p", "."))
                if not np.isclose(float(record["config"].get(field, 1.0)), expected,
                                  rtol=0, atol=1e-12):
                    raise ValueError(f"historical {field} differs from lock: {record_path}")
        artifacts[arm] = {"record_path": record_path, "record": record,
                          "embedding_path": record_path.with_name(f"embedding_seed_{seed}.npz")}
    if int(artifacts["h1"]["record"]["n_clusters"]) != int(artifacts["h8"]["record"]["n_clusters"]):
        raise ValueError("historical arms disagree on K")
    return {"lock_path": lock_path, "background_path": background_path,
            "row": row, "background": bg, "artifacts": artifacts,
            "source_path": Path(bg["source_path"]),
            "k": int(artifacts["h1"]["record"]["n_clusters"])}


def _plot_readouts(rows: list[dict], out: Path, dataset: str, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for arm, color in (("h1", "#797A8E"), ("h8", "#B5465D")):
        ordered = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["random_state"])
        ax.plot([r["random_state"] for r in ordered], [r["ari"] for r in ordered],
                marker="o", markersize=3, linewidth=1.1, color=color, label=arm.upper())
    ax.set(xlabel="KMeans random_state", ylabel="Full-X ARI (descriptive)",
           title=f"{dataset}, seed {seed}: fixed-embedding readouts")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "readout_ari.png", dpi=180)
    fig.savefig(out / "readout_ari.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.3, 4.0))
    for arm, color in (("h1", "#797A8E"), ("h8", "#B5465D")):
        selected = [r for r in rows if r["arm"] == arm]
        ax.scatter([r["normalized_inertia"] for r in selected],
                   [r["ari"] for r in selected], s=23, color=color, label=arm.upper())
    ax.set(xlabel="KMeans inertia / total scatter", ylabel="Full-X ARI (descriptive)",
           title=f"{dataset}, seed {seed}: objective vs ARI")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "objective_vs_ari.png", dpi=180)
    fig.savefig(out / "objective_vs_ari.pdf")
    plt.close(fig)


def _objective_ari_association(rows: list[dict]) -> dict:
    from scipy.stats import pearsonr, spearmanr
    result = {}
    for arm in ("h1", "h8"):
        selected = [r for r in rows if r["arm"] == arm]
        x = np.asarray([r["normalized_inertia"] for r in selected], dtype=float)
        y = np.asarray([r["ari"] for r in selected], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 3 and np.ptp(x[valid]) > 0 and np.ptp(y[valid]) > 0:
            pearson = float(pearsonr(x[valid], y[valid]).statistic)
            spearman = float(spearmanr(x[valid], y[valid]).statistic)
        else:
            pearson = spearman = None
        result[arm] = {"n": int(valid.sum()), "pearson_r": pearson, "spearman_r": spearman,
                       "scope": "within_arm_fixed_embedding_20_initializations"}
    return result


def run_case(previous_root: Path, out: Path, dataset: str, seed: int, smoke: bool = False) -> dict:
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("dataset or seed outside the frozen three-case protocol")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output is nonempty; use a unique task directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    inputs = _locked_inputs(previous_root, dataset, seed)
    k = inputs["k"]  # Frozen record K; labels are not loaded to determine it.
    states = STATES[:2] if smoke else STATES
    predictions: dict[str, np.ndarray] = {}
    objectives: dict[str, list[dict]] = {}
    provenance = {}
    sample_index = None
    for arm, artifact in inputs["artifacts"].items():
        embedding = npz_array(artifact["embedding_path"], "embedding", 2)
        if len(embedding) != int(inputs["row"]["n_samples"]) or not np.isfinite(embedding).all():
            raise ValueError(f"invalid saved embedding for {dataset}/{seed}/{arm}")
        if sample_index is None:
            sample_index = (np.linspace(0, len(embedding) - 1, min(512, len(embedding)),
                                        dtype=np.int64) if smoke else np.arange(len(embedding)))
        embedding = embedding[sample_index]
        centered = np.asarray(embedding, dtype=np.float64) - np.asarray(embedding, dtype=np.float64).mean(axis=0)
        total_scatter = float(np.sum(centered * centered, dtype=np.float64))
        if total_scatter <= 0:
            raise ValueError(f"zero total scatter in saved embedding: {dataset}/{seed}/{arm}")
        arm_predictions = []
        arm_objectives = []
        for state in states:
            model = KMeans(n_clusters=k, random_state=state, n_init=N_INIT, algorithm="lloyd")
            arm_predictions.append(model.fit_predict(embedding).astype(np.int32))
            inertia = float(model.inertia_)
            arm_objectives.append({"inertia": inertia, "total_scatter": total_scatter,
                                   "normalized_inertia": inertia / total_scatter})
        predictions[arm] = np.stack(arm_predictions)
        objectives[arm] = arm_objectives
        provenance[arm] = {"record_path": str(artifact["record_path"]),
                           "record_sha256": sha256(artifact["record_path"]),
                           "embedding_path": str(artifact["embedding_path"]),
                           "embedding_sha256": sha256(artifact["embedding_path"]),
                           "embedding_shape": list(embedding.shape)}
    # KMeans has completed before any posthoc label access.
    labels, class_names = load_labels(inputs["source_path"])
    if len(labels) != int(inputs["row"]["n_samples"]) or len(class_names) != k:
        raise ValueError("source labels disagree with frozen sample count or K")
    labels = labels[sample_index]
    repeats, class_rows, pair_rows = [], [], []
    for arm in ("h1", "h8"):
        for ix, state in enumerate(states):
            pred = predictions[arm][ix]
            repeats.append({"dataset": dataset, "seed": seed, "arm": arm,
                            "random_state": state, "n_init": N_INIT,
                            **objectives[arm][ix],
                            "ari": float(adjusted_rand_score(labels, pred))})
            classes, pairs = pair_rates(labels, pred, class_names)
            class_rows.extend({"dataset": dataset, "seed": seed, "arm": arm,
                               "random_state": state, **row} for row in classes)
            pair_rows.extend({"dataset": dataset, "seed": seed, "arm": arm,
                              "random_state": state, **row} for row in pairs)
    atomic_csv(out / "readout_repeats.csv", repeats, list(repeats[0]))
    atomic_csv(out / "class_split_rates.csv", class_rows, list(class_rows[0]))
    atomic_csv(out / "pair_merge_rates.csv", pair_rows, list(pair_rows[0]))
    with (out / "readout_predictions.npz").open("wb") as stream:
        np.savez_compressed(stream, h1=predictions["h1"], h8=predictions["h8"])
    with (out / "labels_true.npy").open("wb") as stream:
        np.save(stream, labels.astype(np.int64))
    _plot_readouts(repeats, out, dataset, seed)
    summary = {"status": "completed", "protocol_version": VERSION, "dataset": dataset,
               "seed": seed, "smoke": smoke, "previous_root": str(previous_root),
               "posthoc_full_x_labels_loaded": True, "labels_used_for_model_fit": False,
               "labels_used_for_model_selection": False,
               "labels_used_for_exploratory_example_selection": False,
               "evaluation_scope": "all_rows_of_original_X_descriptive_not_heldout" if not smoke else "deterministic_subset_smoke",
               "selection_scope": "historical_validation_only", "K_provenance": "benchmark_oracle_from_y",
               "k": k, "n_samples_evaluated": len(labels), "n_samples_total": inputs["row"]["n_samples"],
               "class_names": class_names,
               "labels_sha256_int64": hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest(),
               "readout_random_states": list(states), "n_init": N_INIT,
               "ari_mean": {arm: float(np.mean([r["ari"] for r in repeats if r["arm"] == arm]))
                            for arm in ("h1", "h8")},
               "objective_ari_association": _objective_ari_association(repeats),
               "delta_ari_mean_h8_minus_h1": float(np.mean([r["ari"] for r in repeats if r["arm"] == "h8"])
                                                   - np.mean([r["ari"] for r in repeats if r["arm"] == "h1"])),
               "inputs": {"lock_path": str(inputs["lock_path"]), "lock_sha256": sha256(inputs["lock_path"]),
                          "background_path": str(inputs["background_path"]),
                          "background_sha256": sha256(inputs["background_path"]),
                          "source_path": str(inputs["source_path"]),
                          "source_size_bytes": inputs["source_path"].stat().st_size,
                          "source_mtime_ns": inputs["source_path"].stat().st_mtime_ns,
                          "artifacts": provenance},
               "outputs": ["readout_predictions.npz", "labels_true.npy",
                           "readout_repeats.csv", "class_split_rates.csv", "pair_merge_rates.csv",
                           "readout_ari.png", "readout_ari.pdf",
                           "objective_vs_ari.png", "objective_vs_ari.pdf"]}
    atomic_json(out / "summary.json", summary)
    return summary


def select_exploratory(dataset: str, case_dirs: list[Path], out: Path) -> dict:
    """Rank complete per-class/per-pair tables using all 3 seeds and all 20 states."""
    if dataset not in DATASETS or len(case_dirs) != 3:
        raise ValueError("selection requires one of the three cases and exactly three seed outputs")
    summaries = [read_json(path / "summary.json") for path in case_dirs]
    if ({s.get("seed") for s in summaries} != set(SEEDS)
            or any(s.get("dataset") != dataset or s.get("status") != "completed" or s.get("smoke")
                   or s.get("readout_random_states") != list(STATES) for s in summaries)):
        raise ValueError("selection requires complete non-smoke outputs for seeds 42, 123, 7")
    if len({s["inputs"]["lock_sha256"] for s in summaries}) != 1:
        raise ValueError("seed outputs use different selection locks")
    kind = "class" if dataset in ("Tosches", "Baron") else "pair"
    file_name = "class_split_rates.csv" if kind == "class" else "pair_merge_rates.csv"
    rate_field = "split_rate" if kind == "class" else "merge_rate"
    identity_fields = ("class_id", "class_name", "support", "true_pair_count") if kind == "class" else (
        "class_a_id", "class_a_name", "class_b_id", "class_b_name",
        "support_a", "support_b", "true_pair_count")
    buckets: dict[tuple, dict] = {}
    for case_dir in case_dirs:
        summary = read_json(case_dir / "summary.json")
        one = {}
        for row in read_csv(case_dir / file_name):
            key = (int(row["random_state"]),
                   int(row["class_id"]) if kind == "class" else (int(row["class_a_id"]), int(row["class_b_id"])))
            one.setdefault(key, {})[row["arm"]] = row
        for key, arms in one.items():
            if set(arms) != {"h1", "h8"}:
                raise ValueError(f"missing paired readout: {case_dir}, {key}")
            left, right = arms["h1"], arms["h8"]
            if any(left[field] != right[field] for field in identity_fields):
                raise ValueError("paired arm supports or class identities differ")
            value_key = key[1]
            bucket = buckets.setdefault(value_key, {"identity": {f: left[f] for f in identity_fields}, "values": []})
            if left[rate_field] and right[rate_field]:
                bucket["values"].append({"seed": summary["seed"], "state": key[0],
                                         "h1": float(left[rate_field]), "h8": float(right[rate_field])})
    complete = []
    for identity, data in buckets.items():
        values = data["values"]
        seed_means = {seed: (float(np.mean([v["h1"] - v["h8"] for v in values
                                              if v["seed"] == seed]))
                             if sum(v["seed"] == seed for v in values) == len(STATES) else None)
                      for seed in SEEDS}
        if len(values) != 3 * len(STATES) or {(v["seed"], v["state"]) for v in values} != {
                (seed, state) for seed in SEEDS for state in STATES}:
            # A singleton class or absent pair has no defined rate; preserve it with null metrics.
            improvement = h1_mean = h8_mean = None
        else:
            h1_mean = float(np.mean([v["h1"] for v in values]))
            h8_mean = float(np.mean([v["h8"] for v in values]))
            improvement = h1_mean - h8_mean
        complete.append({**data["identity"], "h1_mean_rate": h1_mean,
                         "h8_mean_rate": h8_mean, "mean_improvement_h1_minus_h8": improvement,
                         **{f"seed_{seed}_mean_improvement": seed_means[seed] for seed in SEEDS},
                         "n_positive_readout_pairs": sum(v["h1"] > v["h8"] for v in values),
                         "n_positive_seed_means": sum(value is not None and value > 0
                                                       for value in seed_means.values()),
                         "n_seed_state_pairs": len(values), "selected_top2": False})
    id_key = (lambda r: int(r["class_id"])) if kind == "class" else (
        lambda r: (int(r["class_a_id"]), int(r["class_b_id"])))
    eligible = sorted((r for r in complete if r["mean_improvement_h1_minus_h8"] is not None
                       and r["mean_improvement_h1_minus_h8"] > 0),
                      key=lambda r: (-r["mean_improvement_h1_minus_h8"], id_key(r)))
    selected_ids = {id_key(r) for r in eligible[:2]}
    for row in complete:
        row["selected_top2"] = id_key(row) in selected_ids
    complete.sort(key=lambda r: (r["mean_improvement_h1_minus_h8"] is None,
                                 -(r["mean_improvement_h1_minus_h8"] or 0), id_key(r)))
    out.mkdir(parents=True, exist_ok=True)
    atomic_csv(out / f"{kind}_selection_complete.csv", complete, list(complete[0]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = [r for r in complete if r["selected_top2"]]
    names = ([r["class_name"] for r in chosen] if kind == "class" else
             [f"{r['class_a_name']} / {r['class_b_name']}" for r in chosen])
    fig, ax = plt.subplots(figsize=(max(5.2, 2.8 * len(chosen)), 3.6))
    x = np.arange(len(chosen))
    ax.bar(x - 0.18, [r["h1_mean_rate"] for r in chosen], 0.34, label="H1", color="#797A8E")
    ax.bar(x + 0.18, [r["h8_mean_rate"] for r in chosen], 0.34, label="H8", color="#B5465D")
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set(ylabel=f"{rate_field.replace('_', ' ')} (3 seeds × 20 states)",
           title=f"{dataset}: exploratory top-two {kind} examples")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "selection_top2.png", dpi=180)
    fig.savefig(out / "selection_top2.pdf")
    plt.close(fig)
    report = {"status": "completed", "dataset": dataset, "selection_kind": kind,
              "metric": rate_field, "improvement_direction": "h1_rate_minus_h8_rate",
              "ranking_rule": "positive_mean_across_all_3_seeds_and_all_20_fixed_KMeans_states_then_id_tiebreak",
              "positive_pair_note": "paired_readout_frequency_is_descriptive_not_independent_training_replication",
              "exploratory_posthoc_label_based_examples": True,
              "labels_used_for_model_fit": False, "labels_used_for_model_selection": False,
              "seeds": list(SEEDS), "states": list(STATES),
              "case_summaries": [{"path": str(path / "summary.json"),
                                  "sha256": sha256(path / "summary.json")} for path in case_dirs],
              "selected_top2": [r for r in complete if r["selected_top2"]],
              "complete_table": f"{kind}_selection_complete.csv",
              "figures": ["selection_top2.png", "selection_top2.pdf"]}
    atomic_json(out / "selection_provenance.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Exact unique per-case output directory")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--smoke", action="store_true", help="Two states on deterministic 512-row subset")
    args = parser.parse_args()
    result = run_case(args.previous_root, args.out, args.dataset, args.seed, args.smoke)
    print(json.dumps({"status": result["status"], "dataset": result["dataset"],
                      "seed": result["seed"], "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
