#!/usr/bin/env python3
"""Audit the cross-version TopoGate advantage landscape.

The script only reads completed CSV/JSON artifacts.  It deliberately keeps
Full-vs-NoMix comparisons within one result batch, records source hashes, and
does not use labels to select a method or a hyperparameter.  Outputs are
written below the repository's ``result`` symlink.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
OUT = RESULT / "analysis"
DATE = "2026-08-03"
EPSILON = 1e-3

PAIR_SPECS = [
    ("V9", "v9_advantage_ablation", "v9_full", "v9_nomix", "ari"),
    ("V11", "v11_minimum_5x3", "V11_full", "V11_nomix", "ari"),
    ("V12", "v12_advantage", "v12_full", "v12_nomix", "ari"),
    ("V13", "v13_advantage", "v13_full", "v13_nomix", "ari"),
    ("V14", "v14_advantage_5ds", "v14_full", "v14_nomix", "head_ari"),
    ("StaticGate", "static_gate_legacy_table", "static_gate_full", "static_gate_nomix", "ari"),
]

RAW_SPECS = {
    "V9": {
        "path": RESULT / "v9_results_2026-08-02_advantage_ablation" / "ablation_runs.csv",
        "source_columns": ["source_sha256"],
    },
    "V12": {
        "path": RESULT / "v12_results_2026-08-03_advantage" / "runs.csv",
        "source_columns": ["source", "source_sha256"],
    },
    "V13": {
        "path": RESULT / "v13_results_2026-08-03_advantage" / "runs.csv",
        "source_columns": ["source_path", "source_sha256"],
    },
    "V14": {
        "path": RESULT / "v14_results_2026-08-03_advantage_5ds" / "runs.csv",
        "source_columns": ["source_path", "source_sha256"],
    },
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def number(value: Any) -> float | None:
    if value is None or value == "" or value == "None":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def unique_join(values: list[Any]) -> str:
    clean = sorted({str(value) for value in values if value not in (None, "", "nan")})
    return "|".join(clean)


def direction(value: float | None, epsilon: float = EPSILON) -> str:
    if value is None or not math.isfinite(value):
        return "missing"
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "near_neutral"


def seed_pattern(values: list[float]) -> str:
    if not values:
        return "missing"
    signs = [direction(value) for value in values]
    prefix = "stable" if len(values) >= 3 else "single_seed"
    if all(sign == "positive" for sign in signs):
        return f"{prefix}_positive"
    if all(sign == "negative" for sign in signs):
        return f"{prefix}_negative"
    if all(sign == "near_neutral" for sign in signs):
        return "near_neutral"
    return "mixed_seed"


def format_delta(value: Any) -> str:
    parsed = number(value)
    return "NA" if parsed is None else f"{parsed:+.4f}"


def ensure_result_target() -> None:
    expected = Path("/data/luolie/ToPoGate/result").resolve()
    actual = RESULT.resolve()
    if actual != expected:
        raise RuntimeError(f"result target mismatch: {actual} != {expected}")
    OUT.mkdir(parents=True, exist_ok=True)


def load_cross_version() -> pd.DataFrame:
    path = OUT / f"cross_version_evidence_{DATE}.csv"
    frame = read_csv(path)
    required = {"batch", "version", "variant", "dataset", "ari_mean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    return frame


def load_raw_pair_rows() -> pd.DataFrame:
    """Return per-seed Full/NoMix metrics and provenance fields."""
    records: list[dict[str, Any]] = []

    v9 = read_csv(RAW_SPECS["V9"]["path"])
    v9 = v9[v9["variant"].isin(["v9_full", "v9_nomix"])]
    v9 = v9[v9["status"].eq("completed")]
    for row in v9.to_dict("records"):
        records.append(
            {
                "version": "V9",
                "batch": "v9_advantage_ablation",
                "dataset": row.get("dataset"),
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "metric": number(row.get("ari")),
                "source_sha256": row.get("source_sha256"),
                "source_path": None,
            }
        )

    for version in ["V12", "V13", "V14"]:
        spec = RAW_SPECS[version]
        frame = read_csv(spec["path"])
        full = f"{version.lower()}_full"
        nomix = f"{version.lower()}_nomix"
        frame = frame[frame["variant"].isin([full, nomix])]
        frame = frame[frame["status"].eq("completed")]
        for row in frame.to_dict("records"):
            metric_column = "head_ari" if version == "V14" else "ari"
            summary_path = (
                spec["path"].parent
                / f"{row['dataset']}__{row['variant']}__seed{int(row['seed'])}"
                / "summary.json"
            )
            summary = read_json(summary_path) if summary_path.is_file() else {}
            records.append(
                {
                    "version": version,
                    "batch": {
                        "V12": "v12_advantage",
                        "V13": "v13_advantage",
                        "V14": "v14_advantage_5ds",
                    }[version],
                    "dataset": row.get("dataset"),
                    "variant": row.get("variant"),
                    "seed": row.get("seed"),
                    "metric": number(row.get(metric_column)),
                    "source_sha256": row.get("source_sha256"),
                    "source_path": row.get("source") or row.get("source_path"),
                    "n_samples": row.get("n_samples") or summary.get("n_samples"),
                    "n_features": row.get("n_features") or summary.get("n_features"),
                    "n_clusters": row.get("n_clusters") or summary.get("n_clusters"),
                }
            )

    v11_root = RESULT / "V11" / "topogate_v11_minimum_5x3"
    for path in sorted(v11_root.glob("**/comparison.csv")):
        frame = read_csv(path)
        frame = frame[frame["variant"].isin(["V11_full", "V11_nomix"])]
        frame = frame[frame["error"].isna() | frame["error"].eq("")]
        for row in frame.to_dict("records"):
            summary_path = next(
                (
                    candidate
                    for candidate in v11_root.glob(
                        f"**/{row['dataset']}__{row['variant']}__seed{int(row['seed'])}/summary.json"
                    )
                ),
                None,
            )
            summary = read_json(summary_path) if summary_path else {}
            records.append(
                {
                    "version": "V11",
                    "batch": "v11_minimum_5x3",
                    "dataset": row.get("dataset"),
                    "variant": row.get("variant"),
                    "seed": row.get("seed"),
                    "metric": number(row.get("ari")),
                    "source_sha256": summary.get("source_sha256"),
                    "source_path": summary.get("source_path"),
                    "n_samples": summary.get("n_samples"),
                    "n_features": summary.get("n_features"),
                    "n_clusters": summary.get("n_clusters"),
                }
            )

    static = read_csv(RESULT / "ablation" / "merged_summary.csv")
    static = static[static["variant"].isin(["static_gate_full", "static_gate_nomix"])]
    for row in static.to_dict("records"):
        records.append(
            {
                "version": "StaticGate",
                "batch": "static_gate_legacy_table",
                "dataset": row.get("dataset"),
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "metric": number(row.get("ari")),
                "source_sha256": None,
                "source_path": None,
                "n_samples": row.get("n_samples"),
                "n_features": row.get("n_features"),
                "n_clusters": row.get("n_clusters"),
            }
        )

    return pd.DataFrame.from_records(records)


def build_pair_rows(aggregate: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for version, batch, full_variant, nomix_variant, metric_name in PAIR_SPECS:
        subset = aggregate[(aggregate["version"] == version) & (aggregate["batch"] == batch)]
        full = subset[subset["variant"] == full_variant].set_index("dataset")
        nomix = subset[subset["variant"] == nomix_variant].set_index("dataset")
        raw_subset = raw[(raw["version"] == version) & (raw["batch"] == batch)]
        for dataset in sorted(set(full.index).intersection(nomix.index)):
            full_row = full.loc[dataset]
            nomix_row = nomix.loc[dataset]
            seed_rows = raw_subset[raw_subset["dataset"] == dataset]
            seed_pivot = seed_rows.pivot_table(index="seed", columns="variant", values="metric", aggfunc="first")
            deltas = []
            if full_variant in seed_pivot and nomix_variant in seed_pivot:
                deltas = (seed_pivot[full_variant] - seed_pivot[nomix_variant]).dropna().tolist()
            hashes = seed_rows["source_sha256"].dropna().astype(str).unique().tolist()
            source_paths = seed_rows["source_path"].dropna().astype(str).unique().tolist()
            aggregate_hash_count = number(full_row.get("source_sha256_count"))
            hash_count = len(hashes) if hashes else int(aggregate_hash_count or 0)
            mean_delta = number(full_row.get("ari_mean")) - number(nomix_row.get("ari_mean"))
            records.append(
                {
                    "version": version,
                    "batch": batch,
                    "dataset": dataset,
                    "full_variant": full_variant,
                    "nomix_variant": nomix_variant,
                    "metric": metric_name,
                    "n_runs_full": number(full_row.get("n_runs")),
                    "n_runs_nomix": number(nomix_row.get("n_runs")),
                    "seeds": unique_join(seed_rows["seed"].tolist()),
                    "full_ari_mean": number(full_row.get("ari_mean")),
                    "nomix_ari_mean": number(nomix_row.get("ari_mean")),
                    "full_ari_std": number(full_row.get("ari_std")),
                    "nomix_ari_std": number(nomix_row.get("ari_std")),
                    "delta_ari": mean_delta,
                    "seed_delta_mean": sum(deltas) / len(deltas) if deltas else None,
                    "seed_delta_min": min(deltas) if deltas else None,
                    "seed_delta_max": max(deltas) if deltas else None,
                    "seed_positive_count": sum(d > EPSILON for d in deltas),
                    "seed_negative_count": sum(d < -EPSILON for d in deltas),
                    "seed_neutral_count": sum(abs(d) <= EPSILON for d in deltas),
                    "seed_pattern": seed_pattern(deltas),
                    "direction": direction(mean_delta),
                    "n_samples": number(full_row.get("n_samples")) or number(seed_rows["n_samples"].dropna().iloc[0])
                    if "n_samples" in seed_rows and not seed_rows["n_samples"].dropna().empty
                    else number(full_row.get("n_samples")),
                    "n_features": number(full_row.get("n_features")) or number(seed_rows["n_features"].dropna().iloc[0])
                    if "n_features" in seed_rows and not seed_rows["n_features"].dropna().empty
                    else number(full_row.get("n_features")),
                    "n_clusters": number(full_row.get("n_clusters")) or number(seed_rows["n_clusters"].dropna().iloc[0])
                    if "n_clusters" in seed_rows and not seed_rows["n_clusters"].dropna().empty
                    else number(full_row.get("n_clusters")),
                    "source_sha256": unique_join(hashes),
                    "source_sha256_count": hash_count,
                    "source_path": unique_join(source_paths),
                    "k_source": full_row.get("k_source"),
                    "labels_used_during_fit": full_row.get("labels_used_during_fit"),
                    "provenance_status": full_row.get("provenance_status"),
                    "evidence_tier": full_row.get("evidence_tier"),
                }
            )
    return pd.DataFrame.from_records(records)


def build_version_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for version, subset in pairs.groupby("version", sort=False):
        deltas = subset["delta_ari"].dropna()
        records.append(
            {
                "version": version,
                "batch": unique_join(subset["batch"].tolist()),
                "datasets": len(subset),
                "mean_delta_ari": deltas.mean() if len(deltas) else None,
                "median_delta_ari": deltas.median() if len(deltas) else None,
                "std_delta_ari": deltas.std(ddof=1) if len(deltas) > 1 else 0.0,
                "positive_dataset_count": int((deltas > EPSILON).sum()),
                "negative_dataset_count": int((deltas < -EPSILON).sum()),
                "near_neutral_dataset_count": int((deltas.abs() <= EPSILON).sum()),
                "stable_positive_dataset_count": int((subset["seed_pattern"] == "stable_positive").sum()),
                "stable_negative_dataset_count": int((subset["seed_pattern"] == "stable_negative").sum()),
                "mixed_seed_dataset_count": int((subset["seed_pattern"] == "mixed_seed").sum()),
                "single_seed_or_table_count": int((subset["n_runs_full"] <= 1).sum()),
                "hash_missing_count": int(subset["source_sha256"].eq("").sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def build_trajectory(pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset, subset in pairs.groupby("dataset", sort=True):
        missing_hash = subset["source_sha256"].eq("").any()
        hashes = [value for value in subset["source_sha256"].tolist() if value]
        hash_set = set(hashes)
        if len(subset) < 2:
            identity = "single_version"
        elif missing_hash:
            identity = "hash_missing_or_partial"
        elif not hashes:
            identity = "hash_missing"
        elif len(hash_set) == 1:
            identity = "same_sha256"
        else:
            identity = "multiple_sha256_do_not_merge"
        deltas = {row["version"]: row["delta_ari"] for row in subset.to_dict("records")}
        directions = {version: direction(value) for version, value in deltas.items()}
        records.append(
            {
                "dataset": dataset,
                "versions": "|".join(sorted(directions)),
                "source_identity": identity,
                "source_sha256": unique_join(hashes),
                "trajectory_signature": "|".join(
                    f"{version}:{directions[version]}" for version in sorted(directions)
                ),
                "v9_delta_ari": deltas.get("V9"),
                "v11_delta_ari": deltas.get("V11"),
                "v12_delta_ari": deltas.get("V12"),
                "v13_delta_ari": deltas.get("V13"),
                "v14_delta_ari": deltas.get("V14"),
                "staticgate_delta_ari": deltas.get("StaticGate"),
            }
        )
    return pd.DataFrame.from_records(records)


def load_tda_rows() -> pd.DataFrame:
    root = RESULT / "V11" / "tda_h0_pilot_2026-08-03"
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        summary = read_json(path)
        metrics = summary.get("metrics", {})
        head = metrics.get("head", metrics)
        kmeans = metrics.get("kmeans", {})
        config = summary.get("config", {})
        run_parts = path.parent.name.split("__")
        variant = run_parts[1] if len(run_parts) >= 3 else summary.get("variant")
        records.append(
            {
                "dataset": summary.get("dataset"),
                "variant": variant,
                "seed": summary.get("seed"),
                "head_ari": number(head.get("ari")),
                "head_nmi": number(head.get("nmi")),
                "kmeans_ari": number(kmeans.get("ari")),
                "kmeans_nmi": number(kmeans.get("nmi")),
                "n_samples": number(summary.get("n_samples")),
                "n_features": number(summary.get("n_features")),
                "n_clusters": number(summary.get("n_clusters")),
                "source_sha256": summary.get("source_sha256"),
                "source_path": summary.get("source_path"),
                "k_protocol": summary.get("k_protocol"),
                "labels_used_during_fit": summary.get("labels_used_during_fit"),
                "tda_prior_mode": config.get("tda_prior_mode", "none"),
                "tda_prior_weight": number(config.get("tda_prior_weight")),
            }
        )
    return pd.DataFrame.from_records(records)


def build_tda_effects(tda: pd.DataFrame) -> pd.DataFrame:
    baseline = tda[tda["variant"] == "V11_full"].set_index(["dataset", "seed"])
    records: list[dict[str, Any]] = []
    for variant in ["V11_nomix", "V11_tda_h0_mst", "V11_tda_fixed_filtration", "V11_tda_random"]:
        subset = tda[tda["variant"] == variant]
        for dataset, group in subset.groupby("dataset", sort=True):
            seed_deltas_head: list[float] = []
            seed_deltas_kmeans: list[float] = []
            for row in group.to_dict("records"):
                key = (dataset, row["seed"])
                if key not in baseline.index:
                    continue
                base = baseline.loc[key]
                head_delta = number(row["head_ari"]) - number(base["head_ari"])
                seed_deltas_head.append(head_delta)
                if number(row["kmeans_ari"]) is not None and number(base["kmeans_ari"]) is not None:
                    seed_deltas_kmeans.append(number(row["kmeans_ari"]) - number(base["kmeans_ari"]))
            if not seed_deltas_head:
                continue
            mean_head = sum(seed_deltas_head) / len(seed_deltas_head)
            mean_kmeans = (
                sum(seed_deltas_kmeans) / len(seed_deltas_kmeans) if seed_deltas_kmeans else None
            )
            if mean_kmeans is None:
                effect = direction(mean_head)
            elif direction(mean_head) == direction(mean_kmeans):
                effect = f"both_{direction(mean_head)}"
            elif direction(mean_head) == "near_neutral" and direction(mean_kmeans) == "near_neutral":
                effect = "near_neutral"
            else:
                effect = "readout_split"
            first = group.iloc[0]
            records.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "n_runs": len(seed_deltas_head),
                    "seeds": unique_join(group["seed"].tolist()),
                    "head_delta_vs_v11_full": mean_head,
                    "kmeans_delta_vs_v11_full": mean_kmeans,
                    "head_seed_pattern": seed_pattern(seed_deltas_head),
                    "kmeans_seed_pattern": seed_pattern(seed_deltas_kmeans),
                    "effect_label": effect,
                    "source_sha256": unique_join(group["source_sha256"].tolist()),
                    "source_path": unique_join(group["source_path"].tolist()),
                    "n_samples": first["n_samples"],
                    "n_features": first["n_features"],
                    "n_clusters": first["n_clusters"],
                    "tda_prior_mode": first["tda_prior_mode"],
                    "tda_prior_weight": first["tda_prior_weight"],
                    "k_protocol": first["k_protocol"],
                    "labels_used_during_fit": first["labels_used_during_fit"],
                }
            )
    return pd.DataFrame.from_records(records)


def build_feature_correlations(pairs: pd.DataFrame, tda_effects: pd.DataFrame) -> pd.DataFrame:
    feature_path = OUT / f"topogate_dataset_features_{DATE}.csv"
    features = read_csv(feature_path)
    feature_columns = [
        "n",
        "d",
        "metadata_k",
        "log_nd",
        "mean_mutual_ratio",
        "mean_snn",
        "effective_neighbor_proxy",
        "sparse_graph_components",
        "sparse_graph_largest_component_fraction",
        "sparse_graph_cycle_rank",
        "tda_h0_q90_death_norm",
        "tda_h0_tail10_share",
        "tda_h0_total_persistence_norm",
    ]
    records: list[dict[str, Any]] = []
    for version, group in pairs.groupby("version", sort=False):
        merged = group[["dataset", "delta_ari"]].merge(
            features[["dataset"] + [col for col in feature_columns if col in features.columns]],
            on="dataset",
            how="inner",
        )
        for feature in feature_columns:
            if feature not in merged:
                continue
            valid = merged[[feature, "delta_ari"]].dropna()
            if len(valid) < 3 or valid[feature].nunique() < 2 or valid["delta_ari"].nunique() < 2:
                continue
            records.append(
                {
                    "analysis": "Full_minus_NoMix",
                    "outcome": f"{version}_delta_ari",
                    "feature": feature,
                    "n": len(valid),
                    "spearman_rho": valid[feature].corr(valid["delta_ari"], method="spearman"),
                    "scope_note": "exploratory post-hoc descriptor; no feature was used for selection",
                }
            )

    # TDA pilot effects use a separate baseline and are kept separate from
    # Full-vs-NoMix trajectories.
    if "variant" in tda_effects.columns:
        tda_effects = tda_effects[tda_effects["variant"] == "V11_tda_h0_mst"]
    if not tda_effects.empty:
        merged = tda_effects[["dataset", "head_delta_vs_v11_full"]].merge(
            features[["dataset"] + [col for col in feature_columns if col in features.columns]],
            on="dataset",
            how="inner",
        )
        for feature in feature_columns:
            if feature not in merged:
                continue
            valid = merged[[feature, "head_delta_vs_v11_full"]].dropna()
            if (
                len(valid) < 3
                or valid[feature].nunique() < 2
                or valid["head_delta_vs_v11_full"].nunique() < 2
            ):
                continue
            records.append(
                {
                    "analysis": "TDA_H0_minus_V11_full",
                    "outcome": "V11_tda_h0_mst_head_delta",
                    "feature": feature,
                    "n": len(valid),
                    "spearman_rho": valid[feature].corr(
                        valid["head_delta_vs_v11_full"], method="spearman"
                    ),
                    "scope_note": "five-dataset exploratory descriptor; no feature was used for selection",
                }
            )
    return pd.DataFrame.from_records(records)


def write_markdown(
    version_summary: pd.DataFrame,
    pairs: pd.DataFrame,
    trajectory: pd.DataFrame,
    tda_effects: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    lines = [
        "# TopoGate 跨版本优势/劣势景观审计",
        "",
        f"生成时间：{DATE}。本报告是对已完成结果的只读、事后描述性分析；不重新训练、不用标签选择配置、不修改既有模型或外部 baseline。",
        "",
        "## 结果与范围",
        "",
        "- 输出根目录：`/home/luolie/ToPoGate/result`，实际目标：`/data/luolie/ToPoGate/result`。",
        "- Full/NoMix 只在同一 batch、同一数据集、同一 seed 配对；`delta_ari = Full - NoMix`。",
        f"- seed 方向阈值为 `|delta| > {EPSILON}`；小于等于该阈值只标为 `near_neutral`。",
        "- StaticGate 的 `merged_summary.csv` 是单 seed/单行表格，不能被解释为多 seed 稳定性证据。",
        "- 同名数据集仅当 source SHA-256 一致时才允许纵向解释；多个 hash 或缺 hash 的行保留为审计警告。",
        "",
        "## Full-NoMix 总览",
        "",
        "| Version | Batch | Datasets | Mean ΔARI | Median ΔARI | Positive | Negative | Near-neutral | Stable + | Stable - | Mixed seed | Single/table |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in version_summary.to_dict("records"):
        lines.append(
            f"| {row['version']} | {row['batch']} | {int(row['datasets'])} | "
            f"{format_delta(row['mean_delta_ari'])} | {format_delta(row['median_delta_ari'])} | "
            f"{int(row['positive_dataset_count'])} | {int(row['negative_dataset_count'])} | "
            f"{int(row['near_neutral_dataset_count'])} | {int(row['stable_positive_dataset_count'])} | "
            f"{int(row['stable_negative_dataset_count'])} | {int(row['mixed_seed_dataset_count'])} | "
            f"{int(row['single_seed_or_table_count'])} |"
        )

    lines.extend(["", "## 稳定正向、稳定负向与混合数据集", ""])
    for version, group in pairs.groupby("version", sort=False):
        positive = group[group["seed_pattern"] == "stable_positive"]["dataset"].tolist()
        negative = group[group["seed_pattern"] == "stable_negative"]["dataset"].tolist()
        mixed = group[group["seed_pattern"] == "mixed_seed"]["dataset"].tolist()
        near = group[group["seed_pattern"] == "near_neutral"]["dataset"].tolist()
        lines.extend(
            [
                f"### {version}",
                "",
                f"- stable positive: `{', '.join(positive) or 'none'}`",
                f"- stable negative: `{', '.join(negative) or 'none'}`",
                f"- mixed across seeds: `{', '.join(mixed) or 'none'}`",
                f"- near neutral: `{', '.join(near) or 'none'}`",
                "",
            ]
        )

    lines.extend([
        "## 逐数据集 Full-NoMix 表",
        "",
        "| Version | Dataset | n/d/K | ΔARI | Seed pattern | Seed range | Source identity |",
        "|---|---|---|---:|---|---|---|",
    ])
    for row in pairs.sort_values(["version", "dataset"]).to_dict("records"):
        dimensions = f"{int(row['n_samples']) if pd.notna(row['n_samples']) else 'NA'}/" \
            f"{int(row['n_features']) if pd.notna(row['n_features']) else 'NA'}/" \
            f"{int(row['n_clusters']) if pd.notna(row['n_clusters']) else 'NA'}"
        source = row["source_sha256"][:12] if row["source_sha256"] else "missing"
        seed_range = f"{format_delta(row['seed_delta_min'])}..{format_delta(row['seed_delta_max'])}"
        lines.append(
            f"| {row['version']} | {row['dataset']} | {dimensions} | {format_delta(row['delta_ari'])} | "
            f"{row['seed_pattern']} | {seed_range} | {source} |"
        )

    lines.extend(["", "## 同名数据集的 source hash 审计", ""])
    for row in trajectory[
        trajectory["source_identity"].isin(["hash_missing_or_partial", "multiple_sha256_do_not_merge"])
    ].to_dict("records"):
        lines.append(
            f"- `{row['dataset']}`：`{row['source_identity']}`；版本轨迹 `{row['trajectory_signature']}`；"
            f"hash `{row['source_sha256'] or 'missing'}`。不能把这些版本强行合并为一个纵向结论。"
        )
    same = trajectory[trajectory["source_identity"] == "same_sha256"]
    if not same.empty:
        lines.append("")
        lines.append("同一 SHA-256 的可比纵向条目：")
        for row in same.to_dict("records"):
            lines.append(f"- `{row['dataset']}`：`{row['trajectory_signature']}`。")

    lines.extend([
        "",
        "## TDA H0 pilot：相对 V11_full",
        "",
        "TDA 结果单独与同一 pilot batch 的 `V11_full` 配对，不与 V11 minimum 5x3 混合。",
        "",
        "| Variant | Dataset | Head ΔARI | KMeans ΔARI | Head seed pattern | KMeans seed pattern | Effect |",
        "|---|---|---:|---:|---|---|---|",
    ])
    for row in tda_effects.sort_values(["variant", "dataset"]).to_dict("records"):
        lines.append(
            f"| {row['variant']} | {row['dataset']} | {format_delta(row['head_delta_vs_v11_full'])} | "
            f"{format_delta(row['kmeans_delta_vs_v11_full'])} | {row['head_seed_pattern']} | "
            f"{row['kmeans_seed_pattern']} | {row['effect_label']} |"
        )
    for variant, group in tda_effects.groupby("variant", sort=False):
        head = group["head_delta_vs_v11_full"]
        km = group["kmeans_delta_vs_v11_full"]
        lines.extend([
            "",
            f"- `{variant}`：head mean ΔARI `{head.mean():+.6f}`，KMeans mean ΔARI "
            f"`{km.mean():+.6f}`，head positive/negative/neutral "
            f"`{int((head > EPSILON).sum())}/{int((head < -EPSILON).sum())}/{int((head.abs() <= EPSILON).sum())}`。",
        ])

    lines.extend([
        "",
        "## 无标签特征的描述性关系",
        "",
        "特征来自已有 `topogate_dataset_features` 审计，结果差值来自事后运行。下表仅报告 Spearman 描述，不是因果分析、显著性证明或配置选择依据；样本量很小且跨版本协议不同。",
        "",
        "| Analysis | Outcome | Feature | n | Spearman rho |",
        "|---|---|---|---:|---:|",
    ])
    for row in correlations.sort_values(["analysis", "outcome", "feature"]).to_dict("records"):
        lines.append(
            f"| {row['analysis']} | {row['outcome']} | {row['feature']} | {int(row['n'])} | "
            f"{row['spearman_rho']:+.3f} |"
        )

    lines.extend([
        "",
        "## 解释边界与下一步",
        "",
        "1. V9 的拓扑相关收益最清晰地出现在 `balance_scale`，但同一批次也有 `spect_heart`、`vehicle`、`vertebral_column` 的负向差值；这支持数据集依赖，而不是普遍优势。",
        "2. V11--V14 的 Full-NoMix 平均差值接近零，且正负数据集并存；不能把较新的版本写成拓扑增益已经稳定解决。",
        "3. H0 pilot 的正向增强项没有在 head 与 KMeans 两个 readout 上形成一致、稳定的跨数据集收益；当前应保留为诊断 no-go。源码中的正向 persistence 分数会强调晚合并边，可能包含跨组件 bridge，下一候选只能作为默认关闭、可回退的 detached prior 假设，需先做 toy graph 单元测试再决定是否训练。",
        "4. 当前 `kNN`、mutual/SNN、动态图和 edge reliability 是有限度量图结构；它们不能被扩写为完整 persistent homology。拓扑学、数学分析、概率混合与深度聚类的本地教材审计见 `TopoGate_whole_project_math_TDA_audit_2026-08-03.md`。",
        "5. 本报告不改变 V9、V10、V11/V12/V13/V14 或外部 baseline 的实现，也不把单 seed StaticGate 表格升级为正式多种子证据。",
        "",
        "## 可复核文件",
        "",
        f"- `scripts/analysis/analyze_topogate_cross_version_landscape.py`",
        f"- `result/analysis/cross_version_evidence_{DATE}.csv`",
        f"- `result/analysis/paired_version_deltas_{DATE}.csv`",
        f"- `result/analysis/topogate_dataset_features_{DATE}.csv`",
        f"- `result/V11/tda_h0_pilot_{DATE}/**/summary.json`（75 runs）",
    ])
    (OUT / f"topogate_cross_version_landscape_{DATE}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    ensure_result_target()
    aggregate = load_cross_version()
    raw = load_raw_pair_rows()
    pairs = build_pair_rows(aggregate, raw)
    version_summary = build_version_summary(pairs)
    trajectory = build_trajectory(pairs)
    tda = load_tda_rows()
    if len(tda) != 75:
        raise RuntimeError(f"expected 75 TDA pilot summaries, found {len(tda)}")
    tda_effects = build_tda_effects(tda)
    correlations = build_feature_correlations(pairs, tda_effects)

    prefix = OUT / f"topogate_cross_version_landscape_{DATE}"
    pairs.to_csv(prefix.with_name(prefix.name + "_per_dataset.csv"), index=False)
    version_summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    trajectory.to_csv(prefix.with_name(prefix.name + "_trajectory.csv"), index=False)
    tda_effects.to_csv(prefix.with_name(prefix.name + "_tda.csv"), index=False)
    correlations.to_csv(prefix.with_name(prefix.name + "_correlations.csv"), index=False)
    write_markdown(version_summary, pairs, trajectory, tda_effects, correlations)
    print(f"wrote {len(pairs)} Full-NoMix rows, {len(trajectory)} trajectory rows, {len(tda_effects)} TDA rows")


if __name__ == "__main__":
    main()
