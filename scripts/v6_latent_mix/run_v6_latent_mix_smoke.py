#!/usr/bin/env python3
"""Smoke test for v6 latent-space mix variant.

Mirrors `scripts/learnable_gate/run_learnable_gate_smoke.py` layout but:
  - variant: v6_latent_mix
  - output dir: result/v6_latent_mix/smoke/<dataset>__seed<seed>/
  - results.json / comparison.csv are written the same way
  - dataset list is the same 5 core datasets used by LearnableGate smoke

K is always auto-detected from labels (per project rules — never hardcode).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "baseline" / "CLUBench")):
    if p not in sys.path:
        sys.path.insert(0, p)

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

# Re-use the same import path as run_npz
from methods.TopoGate.v6_latent_mix.v6_runner import run_v6  # noqa: E402

V6_ROOT = REPO_ROOT / "methods" / "TopoGate" / "v6_latent_mix"

DATASETS = [
    "Mouse_retina",
    "enron",
    "har",
    "Campbell",
    "breast_cancer_wisconsin_original",
]

VARIANT_NAME = "v6_latent_mix_smoke"

DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
OUTPUT_DIR = REPO_ROOT / "result" / "v6_latent_mix" / "smoke"


def _load_yaml_overrides(yaml_path: Path) -> dict:
    """Tiny YAML reader to avoid importing pyyaml just for these keys."""
    import yaml as _yaml
    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f)
    return cfg or {}


def run_one(dataset: str, seed: int, gpu: int, overrides: dict) -> dict:
    npz = DATA_DIR / f"{dataset}.npz"
    if not npz.exists():
        return {"dataset": dataset, "seed": seed, "error": f"{npz} not found"}
    data = __import__("numpy").load(npz)
    X = data["X"] if "X" in data.files else data["x"]
    y = data["y"] if "y" in data.files else None
    import numpy as np
    n_clusters = int(np.unique(y).size) if y is not None else None

    out_dir = OUTPUT_DIR / f"{dataset}__{VARIANT_NAME}__seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        labels, elapsed, metrics = run_v6(
            X, n_clusters=n_clusters, y=y, gpu=gpu,
            seed=seed, return_metrics=True,
            save_dir=str(out_dir),
            **overrides,
        )
        summary_path = out_dir / "summary.json"
        beta = None
        if summary_path.exists():
            with open(summary_path) as f:
                beta = json.load(f).get("learned_gate_final_beta")
        return {
            "dataset": dataset,
            "variant": VARIANT_NAME,
            "seed": seed,
            "n_clusters": n_clusters,
            "acc": metrics.get("acc"),
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "elapsed": float(elapsed),
            "beta": beta,
            "error": None,
        }
    except Exception as exc:
        return {
            "dataset": dataset, "variant": VARIANT_NAME, "seed": seed,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of dataset names; default = all 5")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42],
                        help="Seeds to run; default = [42]")
    parser.add_argument("--config", type=str,
                        default=str(V6_ROOT / "configs" / "v6_latent_mix_smoke.yaml"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_ds = [d for d in DATASETS if (not args.datasets or d in set(args.datasets))]

    yaml_cfg = _load_yaml_overrides(Path(args.config))

    rows = []
    total = len(selected_ds) * len(args.seeds)
    print(f"v6 smoke will run {total} jobs ({len(selected_ds)} ds × {len(args.seeds)} seeds)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Config: {args.config}")
    print(f"Overrides from YAML: {yaml_cfg}")
    print()

    i = 0
    for ds_name in selected_ds:
        for seed in args.seeds:
            i += 1
            print(f"[{i}/{total}] {ds_name}  seed={seed}", flush=True)
            res = run_one(ds_name, seed, args.gpu, yaml_cfg)
            if res.get("error"):
                print(f"  !! ERROR: {str(res['error']).splitlines()[-1]}", flush=True)
            else:
                print(
                    f"  ACC={res['acc']:.4f}  NMI={res['nmi']:.4f}  ARI={res['ari']:.4f}  "
                    f"K={res['n_clusters']}  time={res['elapsed']:.1f}s",
                    flush=True,
                )
            rows.append(res)
            with open(OUTPUT_DIR / f"{ds_name}__{VARIANT_NAME}__seed{seed}.json", "w") as f:
                json.dump(res, f, indent=2, default=str)

    csv_path = OUTPUT_DIR / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "seed", "n_clusters", "acc", "nmi", "ari",
                     "elapsed", "beta_mutual", "beta_snn", "beta_perturb", "beta_uncertainty", "error"])
        for r in rows:
            beta = r.get("beta") or {}
            w.writerow([
                r["dataset"], r["variant"], r.get("seed"),
                r.get("n_clusters"), r.get("acc"), r.get("nmi"), r.get("ari"),
                r.get("elapsed"),
                beta.get("beta_mutual"), beta.get("beta_snn"),
                beta.get("beta_perturb"), beta.get("beta_uncertainty"),
                r.get("error"),
            ])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()