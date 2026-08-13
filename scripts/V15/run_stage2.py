#!/usr/bin/env python3
"""Single-seed Stage-2 exploratory launcher over eligible current NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V15_counterfactual_gate.config import load_config
from methods.TopoGate.V15_counterfactual_gate.run import fit_v15, load_npz
from scripts.V15.run_corruption_diagnostics import corrupt


MODES = ("clean", "feature_mask", "heavy_tail_noise", "random_graph_replacement", "row_contamination", "compound")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-root", type=Path, default=ROOT / "result" / "V15" / "stage2")
    parser.add_argument("--config", type=Path, default=ROOT / "methods" / "TopoGate" / "V15_counterfactual_gate" / "configs" / "topogate_v15.yaml")
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-fraction", type=float, default=0.2)
    parser.add_argument("--row-fraction", type=float, default=0.1)
    parser.add_argument("--noise-scale", type=float, default=0.2)
    parser.add_argument("--graph-replacement-fraction", type=float, default=1.0)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    wanted = set(args.dataset) if args.dataset else None
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    overrides: dict[str, Any] = {}
    for value in args.overrides:
        if "=" not in value:
            raise ValueError(f"override must be key=value: {value}")
        key, raw = value.split("=", 1)
        overrides[key] = yaml.safe_load(raw)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    all_paths = sorted(args.dataset_root.rglob("*.npz"))
    unique_paths: list[Path] = []
    for stem in sorted({path.stem for path in all_paths}):
        matches = [path for path in all_paths if path.stem == stem]
        direct = args.dataset_root / f"{stem}.npz"
        unique_paths.append(direct if direct.exists() else matches[0])
    for path in unique_paths:
        if wanted is not None and path.stem not in wanted:
            continue
        try:
            X, y = load_npz(path)
            if X.shape[0] < 300 or X.shape[1] < 500:
                continue
            if y is None:
                failures.append({"dataset": path.stem, "error": "no labels for benchmark K"})
                continue
            K = int(np.unique(y).size)
            for mode in modes:
                corrupted, corruption_meta, _ = corrupt(
                    X,
                    mode,
                    args.seed,
                    feature_fraction=args.feature_fraction,
                    row_fraction=args.row_fraction,
                    noise_scale=args.noise_scale,
                )
                mode_overrides = {**overrides, "seed": args.seed}
                if mode == "random_graph_replacement":
                    mode_overrides["graph_replacement_fraction"] = args.graph_replacement_fraction
                save_dir = args.output_root / f"{path.stem}__{mode}__seed{args.seed}"
                config = load_config(args.config, mode_overrides)
                fit_v15(
                    corrupted,
                    K,
                    y,
                    config=config,
                    save_dir=save_dir,
                    dataset_name=path.stem,
                    source_path=path,
                    k_protocol="benchmark_oracle_from_y",
                    run_metadata={
                        "stage": "Stage2",
                        "condition": mode,
                        "corruption": corruption_meta,
                        "labels_used_during_fit": False,
                    },
                )
                records.append({"dataset": path.stem, "mode": mode, "seed": args.seed, "output": str(save_dir)})
        except Exception as exc:
            failures.append({"dataset": path.stem, "error": f"{type(exc).__name__}: {exc}"})
    manifest = {
        "stage": "Stage2",
        "seed": args.seed,
        "modes": modes,
        "records": records,
        "failures": failures,
        "eligibility": "n >= 300 and d >= 500",
        "labels_used_during_fit": False,
        "note": "Single-seed mechanism/failed-case exploration; not a paper-level claim.",
    }
    (args.output_root / "stage2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "failures": len(failures), "manifest": str(args.output_root / 'stage2_manifest.json')}))


if __name__ == "__main__":
    main()
