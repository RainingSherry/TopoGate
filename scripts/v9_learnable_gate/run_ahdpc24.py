#!/usr/bin/env python3
"""Run the frozen V9 LearnableGate protocol on the prepared AHDPC datasets.

This runner deliberately uses the repository's existing ``run_topogate``
implementation and V9 adaptive-PCA configuration. It adds only experiment
bookkeeping needed for the AHDPC comparison: source hashes, explicit K
provenance, separated prediction/ground-truth arrays, resumable per-run
summaries, and a compact CSV index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for p in (REPO_ROOT, REPO_ROOT / "baseline" / "CLUBench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from methods.TopoGate.learnable_gate.run_npz import run_topogate


LEARNABLE_GATE_ROOT = REPO_ROOT / "methods" / "TopoGate" / "learnable_gate"
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "AHDPC" / "processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result" / "AHDPC" / "v9_full_table_2026-08-02"
DEFAULT_SEEDS = [42, 123, 7]
DEFAULT_VARIANT = "learnable_gate_v9_adaptive"
PAPER_CONFIG_PATH = REPO_ROOT / "baseline" / "AHDPC" / "configs" / "paper_datasets.json"

V9_OVERRIDES = {
    "variant": DEFAULT_VARIANT,
    "epochs": 80,
    "mask_ratio": 0.3,
    "neighbor_k": 5,
    "mix_neighbors": 4,
    "warmup_epochs": 20,
    "ramp_epochs": 10,
    "n_top_features": 0,
    "knn_pca_mode": "adaptive",
    "knn_pca_dim": 2000,
    "mix_mode": "reliability",
    "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    x_key = "X" if "X" in data.files else "x"
    y_key = "y" if "y" in data.files else "labels"
    if y_key not in data.files:
        raise ValueError(f"{path} does not contain ground-truth labels")
    return np.asarray(data[x_key], dtype=np.float64), np.asarray(data[y_key]).ravel()


def prepared_dataset_names(data_dir: Path) -> list[str]:
    manifest_path = data_dir.parent / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["datasets"]
        names = [
            name for name, row in manifest.items()
            if row.get("status") == "prepared"
            and row.get("processed", {}).get("path")
            and (data_dir / Path(row["processed"]["path"]).name).exists()
        ]
        if names:
            return sorted(names)
    return sorted(path.stem for path in data_dir.glob("*.npz"))


def run_one(
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    seed: int,
    gpu: int,
    no_cuda: bool,
    epochs: int | None,
    scale_input: bool,
    input_preprocessing: str,
    output_variant: str,
    force: bool,
) -> dict:
    data_path = (data_dir / f"{dataset_name}.npz").resolve()
    run_dir = output_dir / f"{dataset_name}__{output_variant}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    if summary_path.exists() and not force:
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("run_status") == "completed":
                return {
                    "dataset": dataset_name,
                    "variant": "v9_adaptive",
                    "seed": seed,
                    "status": "skipped_existing",
                    "error": None,
                    **existing.get("metrics", {}),
                }
        except Exception:
            pass

    x, y = load_npz(data_path)
    n_clusters = int(np.unique(y).size)
    source_sha256 = sha256_file(data_path)
    overrides = dict(V9_OVERRIDES)
    overrides["scale_input"] = bool(scale_input)
    if epochs is not None:
        overrides["epochs"] = int(epochs)

    started = time.time()
    try:
        predictions, elapsed, metrics = run_topogate(
            x,
            n_clusters=n_clusters,
            y=y,
            gpu=gpu,
            seed=seed,
            return_metrics=True,
            save_dir=str(run_dir),
            **overrides,
        )
        predictions = np.asarray(predictions, dtype=np.int64)
        np.save(run_dir / "predictions.npy", predictions)
        np.save(run_dir / "labels_true.npy", y.astype(np.int64))
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_summary.update(
            {
                "run_status": "completed",
                "dataset": dataset_name,
                "variant": output_variant,
                "source_path": str(data_path),
                "source_sha256": source_sha256,
                "n_samples": int(x.shape[0]),
                "n_features_raw": int(x.shape[1]),
                "n_clusters": n_clusters,
                "k_source": "labels_unique",
                "labels_used_during_fit": False,
                "input_preprocessing": input_preprocessing,
                "prediction_path": "predictions.npy",
                "labels_true_path": "labels_true.npy",
                "wall_seconds": float(time.time() - started),
                "protocol": {
                    "method": "TopoGate V9 LearnableGate",
                    "config": DEFAULT_VARIANT,
                    "adaptive_pca": True,
                    "adaptive_pca_upper_bound": 2000,
                    "hvf_features": 0,
                    "epochs": int(overrides["epochs"]),
                    "seed": int(seed),
                    "scale_input": bool(scale_input),
                },
            }
        )
        summary_path.write_text(json.dumps(run_summary, indent=2, default=float), encoding="utf-8")
        return {
            "dataset": dataset_name,
            "variant": output_variant,
            "seed": seed,
            "status": "completed",
            "error": None,
            "n_clusters": n_clusters,
            "acc": metrics.get("acc"),
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "f1_macro": metrics.get("f1_macro"),
            "fmi": metrics.get("fmi"),
            "v_measure": metrics.get("v_measure"),
            "elapsed": float(elapsed),
            "source_sha256": source_sha256,
        }
    except Exception as exc:
        error = f"{exc}\n{traceback.format_exc()}"
        failed = {
            "run_status": "failed",
            "dataset": dataset_name,
            "variant": output_variant,
            "seed": int(seed),
            "source_path": str(data_path),
            "source_sha256": source_sha256,
            "n_samples": int(x.shape[0]),
            "n_features_raw": int(x.shape[1]),
            "n_clusters": n_clusters,
            "k_source": "labels_unique",
            "labels_used_during_fit": False,
            "error": error,
            "wall_seconds": float(time.time() - started),
        }
        summary_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        return {
            "dataset": dataset_name,
            "variant": "v9_adaptive",
            "seed": seed,
            "status": "failed",
            "error": error,
            "n_clusters": n_clusters,
        }


def write_index(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "v9_runs.csv"
    columns = [
        "dataset", "variant", "seed", "status", "n_clusters",
        "acc", "nmi", "ari", "f1_macro", "fmi", "v_measure",
        "elapsed", "source_sha256", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--preprocess-mode",
        choices=["standardized", "paper"],
        default="standardized",
        help="standardized preserves historical V9; paper matches AHDPC raw/zscore rows.",
    )
    parser.add_argument(
        "--output-variant",
        default="v9_adaptive",
        help="Directory/summary variant label; use a new label for a new protocol.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # ``run_topogate`` serializes ordinary overrides as ``--key value`` and
    # therefore cannot receive ``run_npz.py``'s action-style ``--no_cuda``.
    # Hide CUDA before the first runner call for an explicit CPU protocol.
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = prepared_dataset_names(args.data_dir)
    selected = available if args.datasets is None else [
        name for name in args.datasets if name in available
    ]
    unknown = sorted(set(args.datasets or []) - set(available))
    if unknown:
        raise SystemExit(f"Unknown/unprepared datasets: {unknown}")
    if not selected:
        raise SystemExit("No datasets selected")

    paper_config = json.loads(PAPER_CONFIG_PATH.read_text(encoding="utf-8"))
    if args.preprocess_mode == "paper":
        input_preprocessing = {
            name: str(paper_config.get(name, {}).get("input_preprocessing") or "raw")
            for name in selected
        }
    else:
        input_preprocessing = {name: "standardized" for name in selected}

    total = len(selected) * len(args.seeds)
    print(
        f"Running V9 adaptive on {len(selected)} datasets × {len(args.seeds)} seeds "
        f"= {total} jobs; output={args.output_dir}",
        flush=True,
    )
    rows: list[dict] = []
    for index, (dataset_name, seed) in enumerate(
        ((name, seed) for name in selected for seed in args.seeds), start=1
    ):
        print(f"[{index}/{total}] {dataset_name} seed={seed}", flush=True)
        row = run_one(
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=int(seed),
            gpu=int(args.gpu),
            no_cuda=bool(args.no_cuda),
            epochs=args.epochs,
            scale_input=(
                args.preprocess_mode == "standardized"
                or input_preprocessing[dataset_name] == "zscore"
            ),
            input_preprocessing=input_preprocessing[dataset_name],
            output_variant=args.output_variant,
            force=bool(args.force),
        )
        rows.append(row)
        if row.get("error"):
            print(f"  FAILED: {str(row['error']).splitlines()[-1]}", flush=True)
        else:
            print(
                f"  status={row.get('status')} ARI={row.get('ari')} "
                f"NMI={row.get('nmi')} time={row.get('elapsed')}",
                flush=True,
            )

    index_path = write_index(args.output_dir, rows)
    errors = sum(1 for row in rows if row.get("error"))
    print(f"Wrote {index_path}; rows={len(rows)} errors={errors}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
