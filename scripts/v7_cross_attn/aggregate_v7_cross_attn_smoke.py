#!/usr/bin/env python
"""Aggregate v7 smoke results vs v3_full (LearnableGate) and v1 ablation 8 variants.

读取：
  - result/v7_cross_attn/smoke/<dataset>__v7_cross_attn__seed42/metrics.json
  - result/learnable_gate_smoke/multiseed/<dataset>__learnable_gate_sched__seed42/metrics.json
    (v3_full baseline；seed=42 single-seed)
  - result/ablation/merged_summary.csv (8 v1 ablation variants, 单 seed=42)

输出：
  - result/v7_cross_attn/smoke/v7_vs_ablations.csv (9 行：8 个 v1 ablation + v7)
  - 打印 per-dataset delta (v7 vs v3_full, v7 vs 最佳 v1 ablation)

数据集范围：6 个 v1 消融负效应数据集
"""
from __future__ import annotations

import json
import csv
from pathlib import Path

ROOT = Path("/home/luolie/ToPoGate")
V7_DIR = ROOT / "result" / "v7_cross_attn" / "smoke"
LG_DIR = ROOT / "result" / "learnable_gate_smoke" / "multiseed"
ABLATION_CSV = ROOT / "result" / "ablation" / "merged_summary.csv"

DATASETS = [
    "enron",
    "sms_spam_collection",
    "ISOLET",
    "cnae9",
    "Quake_Smart-seq2_Lung",
    "iris",
]

V7_VARIANT = "v7_cross_attn"
LG_VARIANT = "learnable_gate_sched"


def load_metric(d: Path) -> float | None:
    mf = d / "metrics.json"
    if mf.exists():
        with open(mf) as f:
            return json.load(f).get("ari")
    flat = d.with_suffix(".json")
    if flat.exists():
        with open(flat) as f:
            return json.load(f).get("ari")
    return None


def load_ablation_table() -> dict:
    """返回 {dataset: {variant: ari}}"""
    table: dict = {}
    if not ABLATION_CSV.exists():
        return table
    with open(ABLATION_CSV) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            ds = row.get("dataset", "").strip()
            var = row.get("variant", "").strip()
            try:
                ari = float(row.get("ari", "").strip())
            except (ValueError, KeyError):
                continue
            table.setdefault(ds, {})[var] = ari
    return table


def main():
    print("v7 cross-attn vs baselines (single-seed=42 smoke):")
    print("=" * 90)

    abl = load_ablation_table()

    # v7 vs v3_full per-dataset
    print("\n[Section 1] v7 vs v3_full (LearnableGate@sched, seed=42 single-seed)")
    print("-" * 90)
    print(f"{'dataset':40s}  {'v7 ARI':>8}  {'v3 ARI':>8}  {'delta':>8}  {'best_v1_abl':>10}  {'best_var':>26}")
    print("-" * 90)

    deltas_to_v3 = []  # 用于"v7 vs v3_full"整体判断
    deltas_to_best_abl = []  # 用于"v7 vs 最佳 v1 ablation"整体判断
    rows = []

    for ds in DATASETS:
        v7_path = V7_DIR / f"{ds}__{V7_VARIANT}__seed42"
        v3_path = LG_DIR / f"{ds}__{LG_VARIANT}__seed42"
        v7_ari = load_metric(v7_path)
        v3_ari = load_metric(v3_path)

        # 找 8 个 v1 ablation 中最佳的 ARI
        abl_row = abl.get(ds, {})
        best_abl_var, best_abl_ari = None, None
        for var, ari in abl_row.items():
            if best_abl_ari is None or ari > best_abl_ari:
                best_abl_var = var
                best_abl_ari = ari

        if v7_ari is None:
            print(f"  {ds:40s}  {'N/A':>8}  {'??':>8}  {'??':>8}  {'??':>10}  {'??':>26}")
            continue

        delta_v3 = v7_ari - v3_ari if v3_ari is not None else None
        delta_abl = v7_ari - best_abl_ari if best_abl_ari is not None else None

        if delta_v3 is not None:
            deltas_to_v3.append(delta_v3)
        if delta_abl is not None:
            deltas_to_best_abl.append(delta_abl)

        d_v3_s = f"{delta_v3:+.4f}" if delta_v3 is not None else "N/A"
        d_abl_s = f"{best_abl_ari:.4f}" if best_abl_ari is not None else "N/A"
        var_s = best_abl_var or "N/A"
        v3_s = f"{v3_ari:.4f}" if v3_ari is not None else "N/A"
        print(f"  {ds:40s}  {v7_ari:.4f}   {v3_s:>8}  {d_v3_s:>8}  {d_abl_s:>10}  {var_s:>26}")

        rows.append({
            "dataset": ds,
            "v7_ari": v7_ari,
            "v3_full_ari": v3_ari,
            "delta_v7_v3": delta_v3,
            "best_v1_ablation_ari": best_abl_ari,
            "best_v1_ablation_variant": best_abl_var,
            "delta_v7_best_abl": delta_abl,
        })

    print()
    print(f"Sum: {len(deltas_to_v3)} datasets with v3_full baseline")

    if deltas_to_v3:
        avg_v3 = sum(deltas_to_v3) / len(deltas_to_v3)
        wins_v3 = sum(1 for d in deltas_to_v3 if d > 0)
        losses_v3 = sum(1 for d in deltas_to_v3 if d < 0)
        ties_v3 = len(deltas_to_v3) - wins_v3 - losses_v3
        print()
        print(f"[v7 vs v3_full] avg Δ ARI = {avg_v3:+.4f}")
        print(f"  Wins/Losses/Ties: {wins_v3} / {losses_v3} / {ties_v3}")
        print(f"  ≤-0.03 退化: {sum(1 for d in deltas_to_v3 if d <= -0.03)}/6")
        print(f"  ≤-0.05 严重退化: {sum(1 for d in deltas_to_v3 if d <= -0.05)}/6")

    if deltas_to_best_abl:
        avg_abl = sum(deltas_to_best_abl) / len(deltas_to_best_abl)
        wins_abl = sum(1 for d in deltas_to_best_abl if d > 0)
        losses_abl = sum(1 for d in deltas_to_best_abl if d < 0)
        ties_abl = len(deltas_to_best_abl) - wins_abl - losses_abl
        print()
        print(f"[v7 vs best v1 ablation] avg Δ ARI = {avg_abl:+.4f}")
        print(f"  Wins/Losses/Ties: {wins_abl} / {losses_abl} / {ties_abl}")
        print(f"  v7 至少 ≥ 最佳 ablation (假设验证): {wins_abl}/6")

    # GO / NO-GO 判断
    print()
    print("=" * 90)
    print("GO/NO-GO judgment (single-seed smoke only; multi-seed confirmation required)")
    print("=" * 90)

    not_regressed = sum(1 for d in deltas_to_v3 if d >= -0.03)
    severely_regressed = sum(1 for d in deltas_to_v3 if d <= -0.05)
    beats_best_abl = sum(1 for d in deltas_to_best_abl if d > 0)

    print(f"Condition 1: v7 ARI ≥ v3_full - 0.03 on ≥ 4/6 datasets: {not_regressed}/6")
    print(f"Condition 2 (severe stop): v7 ARI ≤ v3_full - 0.05 on ≥ 2/6 datasets: {severely_regressed}/6")
    print(f"Condition 3 (hypothesis): v7 ARI ≥ best v1 ablation on ≥ 1/6 datasets: {beats_best_abl}/6")

    if severely_regressed >= 2:
        print()
        print(">>> NO-GO: v7 ≥ 2/6 数据集严重退化（Δ ≤ -0.05）。停止扩展，写 CHANGELOG_errors.md")
    elif not_regressed >= 4 and beats_best_abl >= 1:
        print()
        print(">>> GO: v7 机制验证通过，建议进入 multi-seed 验证。")
    elif not_regressed >= 4:
        print()
        print(">>> CONDITIONAL GO: v7 未引入新退化，但尚未在 ≥1 数据集压过最佳 ablation，可延长实验。")
    else:
        print()
        print(">>> NO-GO: v7 < 4/6 数据集 ≥ v3_full - 0.03。停止扩展。")

    # 写入 v7_vs_ablations.csv
    csv_path = V7_DIR / "v7_vs_ablations.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "variant", "seed", "ARI",
            "v3_full_ari", "delta_v7_v3",
            "best_v1_ablation_variant", "best_v1_ablation_ari", "delta_v7_best_abl",
        ])
        # 写 v7 行
        for r in rows:
            w.writerow([
                r["dataset"], V7_VARIANT, 42, r["v7_ari"],
                r["v3_full_ari"], r["delta_v7_v3"],
                r["best_v1_ablation_variant"], r["best_v1_ablation_ari"], r["delta_v7_best_abl"],
            ])
        # 写 8 个 ablation 行（从 merged_summary.csv 中提取）
        if ABLATION_CSV.exists():
            with open(ABLATION_CSV) as f2:
                rdr = csv.DictReader(f2)
                for row in rdr:
                    ds = row.get("dataset", "").strip()
                    if ds in DATASETS:
                        try:
                            ari = float(row.get("ari", "").strip())
                        except (ValueError, KeyError):
                            continue
                        w.writerow([
                            ds, row.get("variant", "").strip(), 42, ari,
                            "", "", "", "", "",
                        ])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
