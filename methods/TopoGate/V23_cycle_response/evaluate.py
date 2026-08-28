from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import file_sha256, load_labels_outer
from .evaluation import evaluate_fingerprints


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V23 fingerprints with outer-only labels/K")
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None, help="outer-only labels.npy")
    parser.add_argument("--external-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> dict[str, object]:
    args = _parse_args()
    with np.load(args.fingerprints, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    labels = None if args.labels is None else load_labels_outer(args.labels)
    result = evaluate_fingerprints(arrays, labels=labels, external_k=args.external_k, seed=args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name, predictions in result.predictions.items():
        np.save(output / f"predictions__{name}.npy", predictions)
    if labels is not None:
        _, encoded = np.unique(labels, return_inverse=True)
        np.save(output / "labels_true.npy", encoded.astype(np.int64))
    _write_json(output / "metrics.json", result.metrics)
    _write_json(output / "benchmark_validity_profile.json", result.benchmark_validity)
    summary = {
        "status": "completed",
        "stage": "evaluate",
        "fingerprints": str(args.fingerprints.resolve()),
        "fingerprints_sha256": file_sha256(args.fingerprints),
        "labels_path": None if args.labels is None else str(args.labels.resolve()),
        "labels_sha256": None if args.labels is None else file_sha256(args.labels),
        "seed": int(args.seed),
        "labels_available_outer_only": labels is not None,
        "K_source": result.metrics["K_source"],
        "primary_scientific_object": "cycle_repair_standardized",
        "secondary_mechanistic_object": "recovery_gain_standardized",
        "primary_distance": "cosine",
        "benchmark_validity_interpretation": result.benchmark_validity.get("external_metric_interpretation"),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
