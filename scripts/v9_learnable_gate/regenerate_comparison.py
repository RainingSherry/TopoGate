#!/usr/bin/env python3
"""Regenerate v9_learnable_gate/{multiseed,ablation}/comparison.csv from existing JSON files.

Useful when a launcher process was killed (SIGTERM/143) before writing the final csv.
"""
import json
import csv
from pathlib import Path

ROOT = Path("/home/luolie/ToPoGate/result/v9_learnable_gate")


def regen(subdir: str) -> None:
    sub = ROOT / subdir
    rows = []
    for jp in sorted(sub.glob("*.json")):
        try:
            data = json.loads(jp.read_text())
        except Exception as e:
            print(f"  bad json: {jp.name} ({e})")
            continue
        beta = data.get("beta") or {}
        rows.append({
            "dataset": data.get("dataset"),
            "variant": data.get("variant"),
            "seed": data.get("seed"),
            "n_clusters": data.get("n_clusters"),
            "acc": data.get("acc"),
            "nmi": data.get("nmi"),
            "ari": data.get("ari"),
            "elapsed": data.get("elapsed"),
            "beta_mutual": beta.get("beta_mutual"),
            "beta_snn": beta.get("beta_snn"),
            "beta_perturb": beta.get("beta_perturb"),
            "beta_uncertainty": beta.get("beta_uncertainty"),
            "error": data.get("error"),
        })
    csv_path = sub / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "seed", "n_clusters", "acc", "nmi", "ari",
                    "elapsed", "beta_mutual", "beta_snn", "beta_perturb", "beta_uncertainty", "error"])
        for r in rows:
            w.writerow([r[k] for k in ["dataset", "variant", "seed", "n_clusters",
                                       "acc", "nmi", "ari", "elapsed",
                                       "beta_mutual", "beta_snn", "beta_perturb",
                                       "beta_uncertainty", "error"]])
    errs = sum(1 for r in rows if r.get("error"))
    print(f"{subdir}: {len(rows)} rows, {errs} errors  -> {csv_path}")


if __name__ == "__main__":
    regen("multiseed")
    regen("ablation")
