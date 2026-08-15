#!/usr/bin/env python3
"""Summarize V25 E1 N/R/T artifacts using the preregistered four states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


THRESHOLD = 0.03


def classify_effect(values: Iterable[float], threshold: float = THRESHOLD) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"state": "Inconclusive", "mean": None, "n_seeds": 0, "same_sign_count": 0}
    mean = sum(values) / len(values)
    if mean > threshold:
        same_sign = sum(value > 0 for value in values)
        state = "Positive" if same_sign >= 2 else "Inconclusive"
    elif mean < -threshold:
        same_sign = sum(value < 0 for value in values)
        state = "Negative" if same_sign >= 2 else "Inconclusive"
    elif abs(mean) <= threshold and all(abs(value) < threshold for value in values):
        same_sign = sum(value > 0 for value in values)
        state = "Observed-Small"
    else:
        same_sign = max(sum(value > 0 for value in values), sum(value < 0 for value in values))
        state = "Inconclusive"
    return {"state": state, "mean": mean, "n_seeds": len(values), "same_sign_count": same_sign, "seed_values": values}


def _read_metric(run: Path, arm: str) -> float:
    path = run / arm / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "ari" not in payload:
        raise ValueError(f"E1 run has no primary ARI: {path}")
    return float(payload["ari"])


def summarize_dataset(run_dirs: dict[int, Path], threshold: float = THRESHOLD) -> dict[str, Any]:
    i_values: list[float] = []
    s_values: list[float] = []
    for seed, run in sorted(run_dirs.items()):
        n = _read_metric(run, "N")
        r = _read_metric(run, "R")
        t = _read_metric(run, "T")
        i_values.append(r - n)
        s_values.append(t - r)
    return {
        "I_d": classify_effect(i_values, threshold),
        "S_d": classify_effect(s_values, threshold),
        "seeds": sorted(run_dirs),
        "statistical_unit": "dataset; seeds are repeated measurements",
    }


def phase_gate(
    dataset_summaries: dict[str, dict[str, Any]],
    expected_datasets: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Apply the frozen continuation rule to any audited E1 phase.

    The rule was originally named ``pilot_gate`` even when serialized for
    confirmation and holdout audits.  Keep the computation phase-agnostic;
    callers should use the returned ``phase_gate`` field to avoid implying
    that an incomplete confirmation/holdout is a pilot decision.
    """
    expected = sorted(set(expected_datasets)) if expected_datasets is not None else sorted(dataset_summaries)
    material: list[str] = []
    for dataset in expected:
        summary = dataset_summaries.get(dataset)
        if summary is None:
            continue
        states = {summary["I_d"]["state"], summary["S_d"]["state"]}
        if states & {"Positive", "Negative"}:
            material.append(dataset)
    required = (2 * len(expected) + 2) // 3
    return {
        "passes": len(material) >= required if expected else False,
        "material_dataset_count": len(material),
        "required_dataset_count": required,
        "expected_dataset_count": len(expected),
        "expected_datasets": expected,
        "material_datasets": material,
        "same_sign_across_datasets_not_required": True,
        "rule": "at least 2/3 datasets have a seed-stable material I or S effect; signs may differ",
    }


# Source compatibility for focused tests and older analysis notebooks.  New
# result schemas use ``phase_gate`` and never serialize this legacy name.
pilot_gate = phase_gate


def summarize_root(root: Path, dataset_dirs: dict[str, dict[int, Path]]) -> dict[str, Any]:
    datasets = {dataset: summarize_dataset(runs) for dataset, runs in sorted(dataset_dirs.items())}
    return {"protocol_id": "v25_e1_summary_v1", "datasets": datasets, "phase_gate": phase_gate(datasets)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory containing dataset/seed run directories")
    parser.add_argument("--dataset", action="append", default=[], help="dataset_id=seed42_dir,seed123_dir,seed7_dir")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dirs: dict[str, dict[int, Path]] = {}
    for spec in args.dataset:
        dataset, raw_dirs = spec.split("=", 1)
        paths = [Path(value) for value in raw_dirs.split(",") if value]
        if len(paths) != 3:
            raise ValueError("each --dataset requires exactly three seed directories")
        dataset_dirs[dataset] = {seed: path for seed, path in zip((42, 123, 7), paths)}
    payload = summarize_root(args.root, dataset_dirs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
