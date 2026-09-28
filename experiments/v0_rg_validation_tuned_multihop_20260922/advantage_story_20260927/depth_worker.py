#!/usr/bin/env python3
"""Posthoc E2/E6 readouts and candidate coverage for one frozen case.

No encoder training or parameter selection occurs here. New-fit coverage uses
the saved plan and graph; historical reuse coverage reconstructs its plan from
the frozen feature/config/seed contract and records that provenance.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

BASE = Path("/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922")
HERE = BASE / "advantage_story_20260927"
for _path in (HERE, BASE, BASE / "source_coverage_20260925",
              BASE / "multihop_mechanism_20260924", Path("/home/luolie/ToPoGate")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from cohort_worker import load_aligned_labels, compute_split_merge_rates, sha256, atomic_json
from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
from v0rg_tunable_multihop.tuning import SplitPreprocessor
from v0rg_tunable_multihop.config import V0_RGConfig
from v0rg_tunable_multihop.hop_sampling import build_exact_hop_plan
from factorial_plan import build_frozen_graph, h1_plan_arrays, plan_arrays

SEEDS = (7, 42, 123)
STRATEGIES = ("H1", "H2", "H4", "H8", "Kwide")
OUTPUTS = ("readouts.csv", "per_seed.csv", "summary.csv", "coverage_by_anchor.csv",
           "coverage_summary.csv", "missing_cells.json", "readout_predictions.npz",
           "depth_and_kwide.png", "depth_and_kwide.pdf", "depth_and_kwide.svg")


def atomic_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def require_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return actual


def materialize(template: dict, variant: str | None = None) -> V0_RGConfig:
    config = dict(template)
    config.pop("protocol_id", None)
    if variant is not None:
        config["hop_variant"] = variant
    return V0_RGConfig(**config)


def same_config(record_config: dict, expected: V0_RGConfig) -> bool:
    left = asdict(V0_RGConfig(**record_config))
    right = asdict(expected)
    left.pop("protocol_id", None)
    right.pop("protocol_id", None)
    return left == right


def expected_variant(strategy: str, selected: str) -> str:
    return "H1" if strategy == "H1" else (
        selected if strategy == "Kwide" else f"{strategy}_{selected.split('_')[1]}")


def load_cell(dataset: str, strategy: str, seed: int, manifest: dict,
              input_hashes: dict) -> tuple[np.ndarray, dict, Path, bool]:
    cell = next((c for c in manifest["depth_cells"]
                 if c["dataset"] == dataset and c["depth"] == strategy and int(c["seed"]) == seed), None)
    if strategy != "Kwide" and cell is None:
        raise ValueError("manifest depth cell missing")
    if cell is not None and cell["status"] == "reuse":
        record_path = Path(cell["record_path"])
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            raise ValueError("historical record incomplete")
        embedding_path = Path(cell["artifacts"][f"embedding_seed_{seed}.npz"])
        # Older records can lack artifact_sha256. Validate any listed hashes,
        # then pin every current input digest in this posthoc record.
        for name, digest in record.get("artifact_sha256", {}).items():
            require_hash(record_path.parent / name, digest)
        for name, raw_path in cell["artifacts"].items():
            path = Path(raw_path)
            input_hashes[f"{strategy}_seed{seed}_{name}"] = sha256(path)
        input_hashes[f"{strategy}_seed{seed}_record"] = sha256(record_path)
        with np.load(embedding_path, allow_pickle=False) as packed:
            embedding = np.asarray(packed["embedding"], dtype=np.float32)
        return embedding, record, record_path.parent, True

    task_id = f"{dataset}__{strategy}__seed{seed}"
    root = HERE / "outputs/train" / task_id
    reasons = []
    for attempt in sorted(root.glob("attempt_*/record.json")):
        try:
            record = json.loads(attempt.read_text(encoding="utf-8"))
            if record.get("status") != "completed" or record.get("task_seed") != seed:
                raise ValueError("attempt status/seed mismatch")
            if record.get("task_dataset") != dataset or record.get("task_strategy") != (
                    "kwide" if strategy == "Kwide" else "path"):
                raise ValueError("attempt dataset/strategy mismatch")
            listed = record.get("artifact_sha256")
            if not isinstance(listed, dict) or not listed:
                raise ValueError("attempt lacks artifact hashes")
            for name, digest in listed.items():
                require_hash(attempt.parent / name, digest)
            task_path = HERE / "tasks" / f"{task_id}.json"
            require_hash(task_path, record["task_sha256"])
            embedding = np.asarray(np.load(attempt.parent / "embedding.npy", allow_pickle=False), dtype=np.float32)
            input_hashes[f"{strategy}_seed{seed}_record"] = sha256(attempt)
            input_hashes[f"{strategy}_seed{seed}_task"] = sha256(task_path)
            for name, digest in listed.items():
                input_hashes[f"{strategy}_seed{seed}_{name}"] = digest
            return embedding, record, attempt.parent, False
        except Exception as exc:
            reasons.append(f"{attempt.parent.name}: {exc}")
    raise FileNotFoundError(f"no valid {task_id} attempt: {'; '.join(reasons) or 'no record'}")


def load_saved_plan(path: Path, n: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as packed:
        arrays = {key: np.asarray(packed[key]) for key in
                  ("offsets", "nodes", "probs", "hops", "draw_count")}
    offsets, nodes, probs, hops = (arrays[k] for k in ("offsets", "nodes", "probs", "hops"))
    if (offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(nodes)
            or np.any(np.diff(offsets) <= 0) or len(probs) != len(nodes)
            or len(hops) != len(nodes) or np.any(nodes < 0) or np.any(nodes >= n)
            or not np.isfinite(probs).all() or np.any(probs < 0)
            or arrays["draw_count"].shape != (1,)):
        raise ValueError(f"invalid saved plan arrays: {path}")
    return arrays


def verify_saved_graph(folder: Path, frozen: dict) -> None:
    with np.load(folder / "graph.npz", allow_pickle=False) as packed:
        if not np.array_equal(packed["indices"], frozen["graph"].indices):
            raise ValueError("saved training graph indices differ from reconstructed graph")
        for name in ("edge_weights", "node_gate"):
            if not np.allclose(packed[name], frozen[name], rtol=1e-8, atol=1e-10):
                raise ValueError(f"saved training {name} differs from reconstructed graph")


def coverage(arrays: dict, near: np.ndarray, communities: np.ndarray,
             displacement: np.ndarray | None = None) -> list[dict]:
    offsets, nodes, probs = arrays["offsets"], arrays["nodes"], arrays["probs"]
    rows = []
    for anchor in range(len(communities)):
        start, end = int(offsets[anchor]), int(offsets[anchor + 1])
        candidates = nodes[start:end]
        probabilities = probs[start:end]
        row_mass = float(probabilities.sum())
        if not np.isfinite(row_mass) or abs(row_mass - 1.0) > 1e-5:
            raise ValueError(f"candidate probabilities do not sum to one at anchor {anchor}")
        # H1 edge weights are float32; a saved row can sum to 1 ± 2.4e-7.
        probabilities = probabilities / row_mass
        absent = ~near[anchor, communities[candidates]]
        rows.append({"anchor": anchor, "candidate_count": int(end - start),
                     "fraction_candidates_community_absent_from_h1": float(absent.mean()),
                     "candidate_probability_mass_community_absent_from_h1": float(probabilities[absent].sum()),
                     "actual_training_displacement_mean": None if displacement is None else float(displacement[anchor])})
    return rows


def readouts(embedding: np.ndarray, labels: np.ndarray, K: int) -> tuple[list[dict], np.ndarray]:
    if embedding.ndim != 2 or len(embedding) != len(labels) or not np.isfinite(embedding).all():
        raise ValueError("embedding shape/finite check failed")
    rows, predictions = [], []
    for state in range(10):
        prediction = KMeans(n_clusters=K, n_init=20, random_state=state).fit_predict(embedding)
        split, merge, _, _ = compute_split_merge_rates(labels, prediction)
        rows.append({"readout_state": state, "ARI": float(adjusted_rand_score(labels, prediction)),
                     "split": float(split), "merge": float(merge)})
        predictions.append(prediction.astype(np.int32))
    return rows, np.stack(predictions)


def plot_results(per_seed: list[dict], summary: list[dict], selected: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    colors = {"H1": "#797A8E", "H2": "#BA7B8F", "H4": "#B5465D",
              "H8": "#8F284F", "Kwide": "#4F5260"}
    by_strategy = {r["strategy"]: r for r in summary}
    for row in per_seed:
        if row["strategy"] != "Kwide":
            axes[0].scatter(int(row["strategy"][1:]), row["ARI_mean"],
                            color=colors[row["strategy"]], alpha=0.6, s=27)
    xs = [int(s[1:]) for s in STRATEGIES[:4] if s in by_strategy]
    ys = [by_strategy[f"H{x}"]["ARI_mean"] for x in xs]
    if xs:
        axes[0].plot(xs, ys, color="#B5465D", marker="o", linewidth=1.5)
    axes[0].set(xlabel="Maximum source depth", ylabel="ARI (10 readouts per seed)",
                title="Fixed-config depth response")
    axes[0].set_xticks((1, 2, 4, 8), ("H1", "H2", "H4", "H8"))
    groups = ("H1", selected.split("_")[0], "Kwide")
    for x, strategy in enumerate(groups):
        vals = [r["ARI_mean"] for r in per_seed if r["strategy"] == strategy]
        if not vals:
            continue
        axes[1].bar(x, np.mean(vals), color=colors[strategy], width=0.62,
                    edgecolor="#484650", linewidth=0.6)
        axes[1].scatter(np.full(len(vals), x) + np.linspace(-0.09, 0.09, len(vals)),
                        vals, color="#25242A", s=24, zorder=3)
    axes[1].set_xticks(range(3), ("H1", selected, "Kwide"))
    axes[1].set(ylabel="ARI (seed means)", title="Path versus count-matched direct kNN")
    for ax in axes:
        ax.grid(axis="y", color="#E4DFE1", zorder=0)
        ax.set_axisbelow(True)
    fig.patch.set_facecolor("#F4F0EA")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        path = out / f"depth_and_kwide.{suffix}"
        tmp = out / f".depth_and_kwide.tmp.{suffix}"
        fig.savefig(tmp, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
        tmp.replace(path)
    plt.close(fig)


def run(dataset: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "record.json").exists():
        raise FileExistsError("completed depth output already exists")
    manifest_path = HERE / "REUSE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][dataset]
    selected = case["variant"]
    input_hashes = {"REUSE_MANIFEST.json": sha256(manifest_path)}
    for rel, digest in manifest["core_source_sha256"].items():
        require_hash(BASE / rel, digest)
    feature_path = Path(case["feature_path"])
    input_hashes["feature_only"] = require_hash(feature_path, case["feature_sha256"])
    loaded = load_npz_matrix_only(feature_path)
    if loaded.labels is not None or loaded.profile.get("labels_accessed") is not False:
        raise ValueError("feature-only adapter accessed labels")
    X_raw = loaded.X
    prep = SplitPreprocessor(case["input_kind"], case["feature_limit"]).fit(X_raw)
    if prep.fingerprint != case["preprocessing_hash"]:
        raise ValueError("preprocessing fingerprint changed")
    X = prep.transform(X_raw)
    lock_path = BASE / "source_coverage_20260925/outputs/p0/selection_lock_v2.json"
    expected_lock = (HERE / "COHORT_LOCK_SHA256.txt").read_text(encoding="utf-8").strip()
    input_hashes["selection_lock_v2.json"] = require_hash(lock_path, expected_lock)
    lock_info = json.loads(lock_path.read_text(encoding="utf-8"))["datasets"][dataset]
    labels, names, alignment = load_aligned_labels(lock_info, X_raw)
    K = int(case["n_clusters"])
    if len(labels) != len(X) or len(np.unique(labels)) != K:
        raise ValueError("posthoc label alignment/oracle K mismatch")
    community_path = BASE / f"coverage_screen_20260925/outputs/formal/screen/{dataset}/community_ids.npy"
    input_hashes["community_ids.npy"] = sha256(community_path)
    communities = np.load(community_path, allow_pickle=False)
    if communities.shape != labels.shape or np.any(communities < 0):
        raise ValueError("community alignment mismatch")
    readout_rows, coverage_rows, missing, predictions = [], [], [], {}
    seed_frozen = {}
    for seed in SEEDS:
        frozen = build_frozen_graph(X, materialize(case["expected_config_template"], "H1"), seed)
        seed_frozen[seed] = frozen
        graph_indices = frozen["graph"].indices
        near = np.zeros((len(labels), int(communities.max()) + 1), dtype=bool)
        near[np.arange(len(labels))[:, None], communities[graph_indices]] = True
        for strategy in STRATEGIES:
            key = f"{strategy}_seed{seed}"
            try:
                embedding, rec, folder, reused = load_cell(dataset, strategy, seed, manifest, input_hashes)
                expected = materialize(case["expected_config_template"], expected_variant(strategy, selected))
                if not same_config(rec["config"], expected):
                    raise ValueError("non-depth configuration mismatch")
                if (rec.get("preprocessing_hash") != case["preprocessing_hash"]
                        or rec.get("feature_sha256") != case["feature_sha256"]
                        or rec.get("n_clusters") != K
                        or rec.get("K_provenance") != "benchmark_oracle_from_y"
                        or rec.get("seed", rec.get("task_seed")) != seed
                        or rec.get("labels_used_during_fit", rec.get("labels_used_for_model_fit")) is not False
                        or rec.get("test_labels_loaded") is not False):
                    raise ValueError("record input/K/label provenance mismatch")
                if reused:
                    if strategy == "H1":
                        arrays = h1_plan_arrays(graph_indices, frozen["edge_weights"], expected.mix_neighbors)
                    else:
                        plan = build_exact_hop_plan(frozen["graph"], frozen["edge_weights"],
                                                    expected.hop_variant, expected.mix_neighbors,
                                                    expected.high_hop_probability_multiplier)
                        arrays = plan_arrays(plan)
                    displacement = None
                    coverage_source = "historical_reconstructed_frozen_graph_plan"
                else:
                    verify_saved_graph(folder, frozen)
                    arrays = load_saved_plan(folder / "plan.npz", len(labels))
                    if strategy == "Kwide":
                        ref = load_saved_plan(folder / "reference_plan.npz", len(labels))
                        if not np.array_equal(np.diff(arrays["offsets"]), np.diff(ref["offsets"])):
                            raise ValueError("Kwide per-anchor candidate count differs from selected path")
                        if not same_config(rec["reference_config"], materialize(case["expected_config_template"], selected)):
                            raise ValueError("Kwide reference config differs from selected path")
                    with np.load(folder / "displacement_by_anchor.npz", allow_pickle=False) as packed:
                        displacement = np.asarray(packed["mean"], dtype=np.float64)
                    if displacement.shape != (len(labels),) or not np.isfinite(displacement).all():
                        raise ValueError("saved displacement shape/finite mismatch")
                    coverage_source = "actual_saved_training_graph_and_plan"
                rows, preds = readouts(embedding, labels, K)
                if len(rows) != 10 or preds.shape != (10, len(labels)):
                    raise ValueError("incomplete readout states")
                cell_coverage = coverage(arrays, near, communities, displacement)
                for row in rows:
                    readout_rows.append({"strategy": strategy, "seed": seed, **row})
                for row in cell_coverage:
                    coverage_rows.append({"strategy": strategy, "seed": seed,
                                          "coverage_source": coverage_source, **row})
                predictions[key] = preds
            except Exception as exc:
                missing.append({"strategy": strategy, "seed": seed,
                                "reason": f"{type(exc).__name__}: {exc}"})
    per_seed = []
    coverage_summary = []
    for strategy in STRATEGIES:
        for seed in SEEDS:
            rows = [r for r in readout_rows if r["strategy"] == strategy and r["seed"] == seed]
            cov = [r for r in coverage_rows if r["strategy"] == strategy and r["seed"] == seed]
            if len(rows) != 10 or len(cov) != len(labels):
                continue
            per_seed.append({"strategy": strategy, "seed": seed,
                             **{f"{name}_mean": float(np.mean([r[name] for r in rows]))
                                for name in ("ARI", "split", "merge")},
                             "n_readouts": 10})
            displacement_values = [r["actual_training_displacement_mean"] for r in cov
                                   if r["actual_training_displacement_mean"] is not None]
            coverage_summary.append({"strategy": strategy, "seed": seed,
                                     "candidate_count_mean": float(np.mean([r["candidate_count"] for r in cov])),
                                     "candidate_count_min": min(r["candidate_count"] for r in cov),
                                     "candidate_count_max": max(r["candidate_count"] for r in cov),
                                     "fraction_candidates_community_absent_from_h1_mean": float(np.mean([
                                         r["fraction_candidates_community_absent_from_h1"] for r in cov])),
                                     "candidate_probability_mass_community_absent_from_h1_mean": float(np.mean([
                                         r["candidate_probability_mass_community_absent_from_h1"] for r in cov])),
                                     "actual_training_displacement_mean": (
                                         float(np.mean(displacement_values)) if displacement_values else None),
                                     "coverage_source": cov[0]["coverage_source"]})
    summary = []
    for strategy in STRATEGIES:
        rows = [r for r in per_seed if r["strategy"] == strategy]
        summary.append({"strategy": strategy, "n_seeds": len(rows),
                        **{f"{name}_{stat}": (float(np.mean([r[f'{name}_mean'] for r in rows]))
                             if stat == "mean" and rows else
                             float(np.std([r[f'{name}_mean'] for r in rows], ddof=1))
                             if stat == "sd" and len(rows) > 1 else None)
                           for name in ("ARI", "split", "merge") for stat in ("mean", "sd")}})
    atomic_csv(out / "readouts.csv", readout_rows)
    atomic_csv(out / "per_seed.csv", per_seed)
    atomic_csv(out / "summary.csv", summary)
    atomic_csv(out / "coverage_by_anchor.csv", coverage_rows)
    atomic_csv(out / "coverage_summary.csv", coverage_summary)
    atomic_json(out / "missing_cells.json", missing)
    atomic_npz(out / "readout_predictions.npz", **predictions)
    plot_results(per_seed, summary, selected, out)
    for rel, digest in manifest["core_source_sha256"].items():
        require_hash(BASE / rel, digest)
    if sha256(manifest_path) != input_hashes["REUSE_MANIFEST.json"]:
        raise ValueError("reuse manifest changed during posthoc analysis")
    if sha256(feature_path) != input_hashes["feature_only"]:
        raise ValueError("feature file changed during posthoc analysis")
    record = {"status": "completed" if not missing and len(per_seed) == 15 and len(readout_rows) == 150 else "partial",
              "dataset": dataset, "selected_variant": selected, "expected_cells": 15,
              "completed_cells": len(per_seed), "expected_readouts": 150,
              "completed_readouts": len(readout_rows), "missing_cells": missing,
              "K_provenance": "benchmark_oracle_from_y", "n_clusters": K,
              "labels_used_for_model_fit": False, "posthoc_full_x_labels_loaded": True,
              "new_encoder_training": False, "model_config_scope": "fixed_except_depth_or_Kwide",
              "readout_states": list(range(10)), "n_init": 20,
              "candidate_coverage_definition": "communities absent from each anchor's seed-specific H1 graph neighbors",
              "candidate_probability_not_realized_source_weight": True,
              "displacement_matching_claimed": False,
              "alignment": alignment, "input_sha256": input_hashes,
              "core_source_sha256": manifest["core_source_sha256"],
              "artifact_sha256": {name: sha256(out / name) for name in OUTPUTS}}
    atomic_json(out / "record.json", record)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("Baron", "Quake_Smart-seq2_Lung", "student_grade"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        record = run(args.dataset, Path(args.out))
    print(f"{args.dataset}: {record['status']}, {record['completed_cells']}/15 cells, "
          f"{record['completed_readouts']}/150 readouts")


if __name__ == "__main__":
    main()
