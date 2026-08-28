#!/usr/bin/env python3
"""Run the registered V15 Stage-1 mechanism panel.

The default panel is intentionally small and representative. It is a gate
audit, not a benchmark launcher; use the Stage-1 results to decide whether a
large multi-dataset run is scientifically justified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

# Support both ``python scripts/V15/run_stage1.py`` and module execution.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V15_counterfactual_gate.config import load_config
from methods.TopoGate.V15_counterfactual_gate.run import fit_v15, load_npz


DEFAULT_DATASETS = ("sms_spam_collection", "cnae9", "enron", "Mouse_retina", "cifar10", "olivetti_faces")


def controlled_diagnostic(seed: int = 42) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """2-D clusters plus noisy dimensions with known diagnostic masks."""
    rng = np.random.default_rng(seed)
    centres = np.asarray([[-3.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for cluster, centre in enumerate(centres):
        blocks.append(rng.normal(centre, 0.45, size=(100, 2)).astype(np.float32))
        labels.append(np.full(100, cluster, dtype=np.int64))
    boundary = rng.normal([0.0, 0.0], [0.25, 0.12], size=(24, 2)).astype(np.float32)
    blocks.append(boundary)
    labels.append(np.where(np.arange(boundary.shape[0]) % 2 == 0, 0, 1).astype(np.int64))
    low_density = rng.normal(centres[2], 1.2, size=(24, 2)).astype(np.float32)
    blocks.append(low_density)
    labels.append(np.full(low_density.shape[0], 2, dtype=np.int64))
    outliers = rng.uniform([-10.0, -10.0], [10.0, 10.0], size=(12, 2)).astype(np.float32)
    nearest = np.argmin(((outliers[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1).astype(np.int64)
    blocks.append(outliers)
    labels.append(nearest)
    base = np.concatenate(blocks, axis=0)
    y = np.concatenate(labels, axis=0)
    noisy = rng.normal(0.0, 3.0, size=(base.shape[0], 32)).astype(np.float32)
    X = np.concatenate([base, noisy], axis=1)
    boundary_mask = np.zeros(X.shape[0], dtype=np.uint8)
    boundary_mask[300:324] = 1
    low_density_mask = np.zeros(X.shape[0], dtype=np.uint8)
    low_density_mask[324:348] = 1
    outlier_mask = np.zeros(X.shape[0], dtype=np.uint8)
    outlier_mask[348:] = 1
    masks = {
        "boundary": boundary_mask,
        "low_density": low_density_mask,
        "outlier": outlier_mask,
    }
    return X, y, masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-root", type=Path, default=ROOT / "result" / "V15" / "stage1")
    parser.add_argument("--config", type=Path, default=ROOT / "methods" / "TopoGate" / "V15_counterfactual_gate" / "configs" / "topogate_v15.yaml")
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-controlled", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    names = tuple(args.dataset) if args.dataset else DEFAULT_DATASETS
    overrides: dict[str, Any] = {}
    for value in args.overrides:
        if "=" not in value:
            raise ValueError(f"override must be key=value: {value}")
        key, raw = value.split("=", 1)
        overrides[key] = yaml.safe_load(raw)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    completed: list[str] = []
    for name in names:
        path = args.dataset_root / f"{name}.npz"
        if not path.exists():
            failures.append({"dataset": name, "error": f"missing dataset: {path}"})
            continue
        try:
            X, y = load_npz(path)
            if y is None:
                failures.append({"dataset": name, "error": "Stage 1 representative panel requires labels for benchmark K"})
                continue
            K = int(np.unique(y).size)
            config = load_config(args.config, {**overrides, "seed": args.seed})
            save_dir = output_root / f"{name}__direct_counterfactual__seed{args.seed}"
            fit_v15(
                X,
                K,
                y,
                config=config,
                save_dir=save_dir,
                dataset_name=name,
                source_path=path,
                k_protocol="benchmark_oracle_from_y",
                run_metadata={"stage": "Stage1", "representative": True, "labels_used_during_fit": False},
            )
            completed.append(name)
        except Exception as exc:
            failures.append({"dataset": name, "error": f"{type(exc).__name__}: {exc}"})
    if args.include_controlled:
        try:
            X, y, masks = controlled_diagnostic(args.seed)
            config = load_config(args.config, {**overrides, "seed": args.seed})
            save_dir = output_root / f"controlled_2d_noisy__direct_counterfactual__seed{args.seed}"
            fit_v15(
                X,
                3,
                y,
                config=config,
                save_dir=save_dir,
                dataset_name="controlled_2d_noisy",
                k_protocol="explicit",
                run_metadata={"stage": "Stage1", "representative": False, "mechanism_diagnostic": True, "labels_used_during_fit": False},
            )
            np.savez_compressed(save_dir / "mechanism_masks.npz", **masks)
            completed.append("controlled_2d_noisy")
        except Exception as exc:
            failures.append({"dataset": "controlled_2d_noisy", "error": f"{type(exc).__name__}: {exc}"})
    manifest = {
        "stage": "Stage1",
        "seed": args.seed,
        "datasets_requested": list(names),
        "completed": completed,
        "failures": failures,
        "labels_used_during_fit": False,
        "note": "Engineering/mechanism panel; not a paper-level multi-seed performance claim.",
    }
    (output_root / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"completed": len(completed), "failures": len(failures), "manifest": str(output_root / 'stage1_manifest.json')}))


if __name__ == "__main__":
    main()
