#!/usr/bin/env python3
"""Posthoc source-role accounting for the five saved Baron flat_anchor traces.

Each draw is weighted by its realized training mixture weight. Its role uses
aligned labels and the fixed first-seed community partition; novelty uses that
seed's saved graph. Averages are over anchors within seed, then over seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922")
SOURCE = BASE / "source_coverage_20260925"
SEEDS = (1019, 2027, 3037, 4051, 5059)
TRACE_DTYPE = np.dtype([
    ("epoch", "<u2"), ("anchor", "<i4"), ("source", "<i4"),
    ("hop", "u1"), ("weight", "<f4"), ("slot_probability", "<f4"),
])
ROLE_NAMES = ("same_class_same_community", "same_class_cross_community", "different_class")
NOVELTY_NAMES = ("familiar", "new")
CATEGORIES = tuple(f"{role}_{novelty}" for role in ROLE_NAMES for novelty in NOVELTY_NAMES)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def near_communities(indices: np.ndarray, community_ids: np.ndarray) -> np.ndarray:
    n = len(community_ids)
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[0] != n or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("saved graph indices shape/type mismatch")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("saved graph contains invalid source index")
    if np.any(community_ids < 0):
        raise ValueError("negative community id")
    near = np.zeros((n, int(community_ids.max()) + 1), dtype=bool)
    near[np.arange(n)[:, None], community_ids[indices]] = True
    return near


def accumulate_block(block: np.ndarray, labels: np.ndarray, community_ids: np.ndarray,
                     near: np.ndarray, mass: np.ndarray, total: np.ndarray) -> None:
    """Classify one bounded trace block; malformed draws are hard errors."""
    n = len(labels)
    a, s, hop = block["anchor"], block["source"], block["hop"]
    w = block["weight"].astype(np.float64)
    if (np.any(a < 0) or np.any(a >= n) or np.any(s < 0) or np.any(s >= n)
            or np.any(hop < 1) or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isfinite(block["slot_probability"]).all()):
        raise ValueError("invalid index, hop, weight, or slot probability in trace")
    same_class = labels[a] == labels[s]
    same_comm = community_ids[a] == community_ids[s]
    role = np.where(~same_class, 2, np.where(same_comm, 0, 1)).astype(np.intp)
    novel = (hop >= 2) & ~near[a, community_ids[s]]
    category = role * 2 + novel.astype(np.intp)
    np.add.at(mass, (a, category), w)
    np.add.at(total, a, w)


def process_seed(seed: int, labels: np.ndarray, communities: np.ndarray,
                 train_dir: Path, chunk_size: int = 500_000) -> dict:
    folder = train_dir / f"seed_{seed}"
    record_path, graph_path, trace_path = (folder / name for name in
        ("record.json", "graph.npz", "source_trace.bin"))
    for path in (record_path, graph_path, trace_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record.get("status") != "completed" or record.get("dataset") != "Baron"
            or record.get("arm") != "flat_anchor" or int(record.get("seed", -1)) != seed
            or record.get("labels_used_during_fit") is not False
            or record.get("test_labels_loaded") is not False
            or record.get("K_provenance") != "benchmark_oracle_from_y"):
        raise ValueError(f"historical record provenance mismatch: seed {seed}")
    expected_hashes = record.get("artifact_sha256", {})
    graph_hash = sha256(graph_path)
    if expected_hashes.get("graph.npz") != graph_hash:
        raise ValueError(f"historical graph hash mismatch: seed {seed}")
    with np.load(graph_path, allow_pickle=False) as packed:
        indices = packed["indices"]
    near = near_communities(indices, communities)
    if len(labels) != int(record.get("n_samples", -1)):
        raise ValueError(f"historical sample count mismatch: seed {seed}")
    meta = record.get("training_trace", {})
    if meta.get("trace_complete") is not True or meta.get("trace_dtype") != [list(x) for x in TRACE_DTYPE.descr]:
        raise ValueError(f"historical trace metadata mismatch: seed {seed}")
    nbytes = trace_path.stat().st_size
    if nbytes % TRACE_DTYPE.itemsize:
        raise ValueError(f"truncated trace record: seed {seed}")
    nrows = nbytes // TRACE_DTYPE.itemsize
    if nrows <= 0 or nrows != meta.get("draws"):
        raise ValueError(f"historical trace draw count mismatch: seed {seed}")
    mass = np.zeros((len(labels), 6), dtype=np.float64)
    total = np.zeros(len(labels), dtype=np.float64)
    hasher = hashlib.sha256()
    seen = 0
    with trace_path.open("rb") as stream:
        while True:
            block = np.fromfile(stream, dtype=TRACE_DTYPE, count=chunk_size)
            if len(block) == 0:
                break
            hasher.update(block.tobytes())
            accumulate_block(block, labels, communities, near, mass, total)
            seen += len(block)
    if seen != nrows or hasher.hexdigest() != expected_hashes.get("source_trace.bin"):
        raise ValueError(f"historical trace hash/count mismatch: seed {seed}")
    if not np.isfinite(total).all() or np.any(total <= 0):
        raise ValueError(f"anchor with zero/nonfinite realized source weight: seed {seed}")
    normalized = mass / total[:, None]
    if not np.allclose(normalized.sum(axis=1), 1.0, atol=2e-12, rtol=0):
        raise ValueError(f"six role/novelty masses do not sum to one: seed {seed}")
    return {"seed": seed, "normalized": normalized, "total": total,
            "nrows": nrows, "graph_hash": graph_hash, "trace_hash": hasher.hexdigest(),
            "record_hash": sha256(record_path)}


def aggregate(results: list[dict], labels: np.ndarray):
    seeds = [r["seed"] for r in results]
    per_anchor = np.stack([r["normalized"] for r in results])
    totals = np.stack([r["total"] for r in results])
    per_seed = per_anchor.mean(axis=1)
    seed_mean = per_seed.mean(axis=0)
    class_ids = sorted(int(x) for x in np.unique(labels))
    class_seed = np.stack([[r["normalized"][labels == klass].mean(axis=0)
                            for klass in class_ids] for r in results])
    macro_seed = class_seed.mean(axis=1)
    if not (np.allclose(per_seed.sum(axis=1), 1.0, atol=2e-12, rtol=0)
            and np.allclose(class_seed.sum(axis=2), 1.0, atol=2e-12, rtol=0)):
        raise ValueError("aggregate role masses do not sum to one")
    seed_rows = []
    class_rows = []
    class_mean_rows = []
    for i, seed in enumerate(seeds):
        for j, category in enumerate(CATEGORIES):
            seed_rows.append({"seed": seed, "category": category,
                              "anchor_mean_mass": float(per_seed[i, j]),
                              "class_macro_mass": float(macro_seed[i, j]),
                              "n_anchors": int(per_anchor.shape[1]),
                              "trace_rows": int(results[i]["nrows"])})
        for k, klass in enumerate(class_ids):
            for j, category in enumerate(CATEGORIES):
                class_rows.append({"seed": seed, "class_id": klass,
                                   "class_n": int(np.sum(labels == klass)),
                                   "category": category,
                                   "within_class_anchor_mean_mass": float(class_seed[i, k, j])})
    for k, klass in enumerate(class_ids):
        for j, category in enumerate(CATEGORIES):
            class_mean_rows.append({"class_id": klass, "class_n": int(np.sum(labels == klass)),
                                    "category": category,
                                    "five_seed_mean_mass": float(class_seed[:, k, j].mean()),
                                    "five_seed_sd": float(class_seed[:, k, j].std(ddof=1))})
    aggregate_rows = [{"category": name, "five_seed_mean_mass": float(seed_mean[j]),
                       "five_seed_sd": float(per_seed[:, j].std(ddof=1)),
                       "class_macro_five_seed_mean_mass": float(macro_seed[:, j].mean()),
                       "class_macro_five_seed_sd": float(macro_seed[:, j].std(ddof=1))}
                      for j, name in enumerate(CATEGORIES)]
    return {"per_anchor": per_anchor, "totals": totals, "per_seed": per_seed,
            "seed_mean": seed_mean, "macro_seed": macro_seed,
            "seed_rows": seed_rows, "class_rows": class_rows,
            "class_mean_rows": class_mean_rows, "aggregate_rows": aggregate_rows,
            "class_ids": class_ids}


def plot_distribution(seed_mean: np.ndarray, per_seed: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(3)
    width = 0.31
    colors = ("#c3c1c5", "#c44078")
    for novelty in range(2):
        positions = x + (novelty - 0.5) * width
        bars = ax.bar(positions, 100 * seed_mean[novelty::2], width,
                      color=colors[novelty], edgecolor="#4b4750", linewidth=0.65,
                      label=NOVELTY_NAMES[novelty], zorder=2)
        for i in range(per_seed.shape[0]):
            ax.scatter(positions + (i - 2) * 0.017, 100 * per_seed[i, novelty::2],
                       s=19, color="#34313a", alpha=0.75, zorder=3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.65,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ("Same class\nsame community", "Same class\ncross community",
                      "Different class"))
    ax.set_ylabel("Realized source weight per anchor (%)")
    ax.set_title("Baron: actual training-source role and one-hop novelty")
    ax.set_ylim(0, max(100 * seed_mean) * 1.18)
    ax.grid(axis="y", color="#dedce0", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(0.01, -0.23, "Bars: five-seed mean; dots: individual seeds. Fixed communities; seed-specific saved graphs.",
            transform=ax.transAxes, fontsize=8, color="#625e68")
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        path = out_dir / f"source_role_distribution.{suffix}"
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.{suffix}")
        fig.savefig(tmp, dpi=240, bbox_inches="tight")
        tmp.replace(path)
    plt.close(fig)


def run(out_dir: Path, base: Path = BASE, seeds=SEEDS, chunk_size: int = 500_000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "record.json").exists():
        raise FileExistsError("completed trace output already exists")
    factorial = base / "source_coverage_20260925/outputs/factorial/Baron"
    train_dir = base / "source_coverage_20260925/outputs/train/Baron/flat_anchor"
    labels_path = factorial / "labels_true.npy"
    communities_path = factorial / "community_ids.npy"
    labels = np.load(labels_path, allow_pickle=False)
    communities = np.load(communities_path, allow_pickle=False)
    if labels.ndim != 1 or communities.shape != labels.shape or len(labels) == 0:
        raise ValueError("aligned labels/common communities shape mismatch")
    results = [process_seed(int(seed), labels, communities, train_dir, chunk_size) for seed in seeds]
    agg = aggregate(results, labels)
    if not np.isclose(agg["seed_mean"].sum(), 1.0, atol=2e-12, rtol=0):
        raise ValueError("six five-seed masses do not sum to one")
    atomic_npz(out_dir / "per_anchor_mass.npz", seeds=np.asarray(seeds, dtype=np.int32),
               labels_true=labels, community_ids=communities,
               category_names=np.asarray(CATEGORIES), normalized_mass=agg["per_anchor"],
               total_realized_weight=agg["totals"])
    atomic_csv(out_dir / "per_seed.csv", agg["seed_rows"])
    atomic_csv(out_dir / "per_class.csv", agg["class_rows"])
    atomic_csv(out_dir / "per_class_five_seed.csv", agg["class_mean_rows"])
    atomic_csv(out_dir / "aggregate.csv", agg["aggregate_rows"])
    plot_distribution(agg["seed_mean"], agg["per_seed"], out_dir)
    artifacts = ("per_anchor_mass.npz", "per_seed.csv", "per_class.csv",
                 "per_class_five_seed.csv", "aggregate.csv",
                 "source_role_distribution.pdf", "source_role_distribution.svg",
                 "source_role_distribution.png")
    new_mass = float(agg["seed_mean"][1::2].sum())
    same_new = float(agg["seed_mean"][1] + agg["seed_mean"][3])
    record = {"status": "completed", "mode": "trace", "dataset": "Baron",
              "protocol": "actual_full_training_trace_anchor_then_seed_mean_v1",
              "seeds": [int(s) for s in seeds], "n_anchors": int(len(labels)),
              "n_communities": int(communities.max()) + 1,
              "n_classes": len(agg["class_ids"]),
              "posthoc_full_x_labels_loaded": True,
              "labels_used_for_model_fit": False,
              "new_encoder_training": False,
              "source_weights": "actual_training_mix_before_gate",
              "community_policy": "fixed_first_flat_anchor_seed_partition",
              "near_policy": "each_seed_saved_graph_indices",
              "normalization": "per_anchor_then_seed_mean; class_macro_equal_class_weight",
              "six_category_mass_sum": float(agg["seed_mean"].sum()),
              "total_new_community_mass": new_mass,
              "same_class_new_community_mass": same_new,
              "five_seed_reference_total_new": 0.38928652,
              "five_seed_reference_same_class_new": 0.21449490,
              "input_sha256": {"labels_true.npy": sha256(labels_path),
                               "community_ids.npy": sha256(communities_path),
                               **{f"seed_{r['seed']}_record.json": r["record_hash"] for r in results},
                               **{f"seed_{r['seed']}_graph.npz": r["graph_hash"] for r in results},
                               **{f"seed_{r['seed']}_source_trace.bin": r["trace_hash"] for r in results}},
              "trace_rows_by_seed": {str(r["seed"]): int(r["nrows"]) for r in results},
              "artifact_sha256": {name: sha256(out_dir / name) for name in artifacts}}
    atomic_json(out_dir / "record.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["trace"], default="trace")
    parser.add_argument("--dataset", choices=["Baron"], default="Baron")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    record = run(Path(args.out))
    print(f"Completed five-seed Baron actual trace: new={record['total_new_community_mass']:.8f}, "
          f"same-class new={record['same_class_new_community_mass']:.8f}")


if __name__ == "__main__":
    main()
