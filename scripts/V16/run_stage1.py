from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.TopoGate.V16_predictive_graph_gate.config import load_config
from scripts.V16.run_paired import DEFAULT_VARIANTS, run_one


def main() -> None:
    parser = argparse.ArgumentParser(description="V16 fixed Stage-1 paired runs")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--save_root", default="result/V16/stage1")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 7])
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--condition", choices=("clean", "compound"), default="clean")
    args = parser.parse_args()
    if set(args.variants) != set(DEFAULT_VARIANTS):
        raise ValueError("V16 Stage 1 uses the fixed five-way paired readout; use run_paired.py directly")
    for dataset in args.datasets:
        for seed in args.seeds:
            path = Path(args.data_root) / dataset
            command = {
                "dataset": dataset,
                "condition": args.condition,
                "variants": list(DEFAULT_VARIANTS),
                "seed": seed,
                "save_root": str(Path(args.save_root)),
            }
            print(command)
            if args.dry_run:
                continue
            with np.load(path, allow_pickle=False) as data:
                y = np.asarray(data["y"]) if "y" in data.files else None
            if y is None:
                raise ValueError(f"{dataset} has no labels; provide an explicit K runner before fitting")
            overrides = {"variant": "V16_predictive_gate", "seed": int(seed), "no_cuda": bool(args.no_cuda)}
            if args.epochs is not None:
                overrides["epochs"] = int(args.epochs)
            config = load_config(args.config, overrides)
            run_one(path, Path(args.save_root), int(seed), config, condition=args.condition)


if __name__ == "__main__":
    main()
