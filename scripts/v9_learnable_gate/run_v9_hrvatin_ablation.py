#!/usr/bin/env python3
"""Run Part B ablation on hrvatin_filtered with HVF2000 + Adaptive PCA cap=500
(same configuration as the v9_multiseed hrvatin run that completed successfully).

4 variants × 3 seeds = 12 runs.
"""
from __future__ import annotations

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

import numpy as np
from methods.TopoGate.learnable_gate.run_npz import run_topogate

LEARNABLE_GATE_ROOT = REPO_ROOT / "methods" / "TopoGate" / "learnable_gate"
DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
OUTPUT_DIR = Path("/home/luolie/ToPoGate/result/v9_learnable_gate/ablation")

VARIANTS = {
    "v9_static_gate": {
        "gate_mode": "topology", "mix_mode": "reliability", "pseudo_weight": 0.3,
    },
    "v9_random_neighbors": {
        "gate_mode": "learned", "mix_mode": "random", "pseudo_weight": 0.3,
    },
    "v9_static_random": {
        "gate_mode": "topology", "mix_mode": "random", "pseudo_weight": 0.3,
    },
    "v9_nomix": {
        "gate_mode": "learned", "mix_mode": "none", "pseudo_weight": 0.0,
    },
}


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seeds = [42, 123, 7]

    npz = DATA_DIR / "hrvatin_filtered.npz"
    data = np.load(npz)
    X = data["X"] if "X" in data.files else data["x"]
    y = data["y"] if "y" in data.files else None
    n_clusters = int(np.unique(y).size)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for variant_name, v_over in VARIANTS.items():
        for seed in seeds:
            out_dir = OUTPUT_DIR / f"hrvatin_filtered__{variant_name}__seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = OUTPUT_DIR / f"hrvatin_filtered__{variant_name}__seed{seed}.json"
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        d = json.load(f)
                    if d.get("ari") is not None and d.get("error") is None:
                        print(f"hrvatin  {variant_name}  seed={seed}: cached ARI={d['ari']:.4f}, skip")
                        continue
                except Exception:
                    pass
            overrides = {
                "variant": "learnable_gate_v9_adaptive",
                "epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5,
                "warmup_epochs": 20, "ramp_epochs": 10,
                "n_top_features": 2000,
                "knn_pca_mode": "adaptive",
                "knn_pca_dim": 500,
                "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
                **v_over,
            }
            print(f"hrvatin  {variant_name}  seed={seed}", flush=True)
            t0 = time.time()
            try:
                labels, elapsed, metrics = run_topogate(
                    X, n_clusters=n_clusters, y=y, gpu=gpu,
                    seed=seed, return_metrics=True,
                    save_dir=str(out_dir),
                    **overrides,
                )
                res = {
                    "dataset": "hrvatin_filtered", "variant": variant_name, "seed": seed,
                    "n_clusters": n_clusters,
                    "acc": metrics.get("acc"), "nmi": metrics.get("nmi"),
                    "ari": metrics.get("ari"), "elapsed": float(elapsed),
                    "error": None,
                }
            except Exception as exc:
                res = {"dataset": "hrvatin_filtered", "variant": variant_name, "seed": seed,
                       "error": f"{exc}\n{traceback.format_exc()}"}
            with open(json_path, "w") as f:
                json.dump(res, f, indent=2, default=str)
            if res.get("error"):
                print(f"  !! ERROR: {str(res['error']).splitlines()[-1]}")
            else:
                print(f"  ACC={res['acc']:.4f}  NMI={res['nmi']:.4f}  ARI={res['ari']:.4f}  "
                      f"K={res['n_clusters']}  time={res['elapsed']:.1f}s  ({(time.time()-t0):.1f}s wall)")


if __name__ == "__main__":
    main()