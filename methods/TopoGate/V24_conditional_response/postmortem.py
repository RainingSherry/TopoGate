from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import analyze_response
from .config import V24Q1Config


V23_WORLDS = (
    "cluster_specific_dependency",
    "mean_only_shared_dependency",
    "conditional_dependency_destroyed",
    "global_structure_destroyed_sanity",
)


def run_postmortem(v23_root: Path, output_root: Path, *, bootstrap_replicates: int = 0) -> dict[str, object]:
    """Read V23 artifacts only; never invokes training or profiling."""

    config = V24Q1Config()
    records: list[dict[str, object]] = []
    for world in V23_WORLDS:
        for seed in (42, 123, 7):
            run_root = v23_root / "runs" / world / f"seed{seed}"
            matrix_path = v23_root / "generated_data" / world / f"seed{seed}" / "matrix_only.npz"
            labels_path = v23_root / "generated_data" / world / f"seed{seed}" / "labels_true.npy"
            fingerprint_path = run_root / "profile" / "fingerprints.npz"
            if not all(path.is_file() for path in (matrix_path, labels_path, fingerprint_path)):
                records.append({"world": world, "seed": seed, "status": "incomplete_compute"})
                continue
            with np.load(matrix_path, allow_pickle=False) as loaded:
                matrix = np.asarray(loaded["X"], dtype=np.float32)
            with np.load(fingerprint_path, allow_pickle=False) as loaded:
                fingerprints = {name: np.asarray(loaded[name]) for name in loaded.files}
            labels = np.asarray(np.load(labels_path, allow_pickle=False), dtype=np.int64)
            summary, arrays = analyze_response(
                matrix,
                fingerprints,
                labels=labels,
                config=config,
                seed=seed,
                bootstrap_replicates=bootstrap_replicates,
            )
            destination = output_root / world / f"seed{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / "conditional_response.npz", **arrays)
            (destination / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            records.append(
                {
                    "world": world,
                    "seed": seed,
                    "status": "completed",
                    "delta_auc": summary["conditional_pair_utility"]["delta_auc"],
                    "mean_residual_r2": summary["residualizer"]["mean_cross_fitted_r2"],
                }
            )
    complete = bool(records) and all(record.get("status") == "completed" for record in records)
    result = {
        "status": "completed" if complete else "incomplete_compute",
        "protocol_id": config.protocol_id,
        "stage": "P0_v23_read_only_postmortem",
        "retrained": False,
        "records": records,
        "labels_accessible_during_fit": False,
        "labels_accessible_during_profile": False,
        "labels_accessible_during_analysis": True,
        "cannot_change_p1_protocol": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run read-only V24 P0 analysis over V23 artifacts")
    parser.add_argument("--v23-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    args = parser.parse_args()
    result = run_postmortem(args.v23_root, args.output_root, bootstrap_replicates=args.bootstrap_replicates)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
