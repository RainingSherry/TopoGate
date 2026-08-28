"""Conditional representation-space augmentation localization probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import aggregate, metrics, model, protocol, raw_adapter, run_main


def eligible(main_root: Path) -> bool:
    evaluation = aggregate.evaluate(aggregate.collect(main_root))
    return bool((evaluation["g1"]["passed"] or evaluation["g2"]["passed"]) and not evaluation["necessity"]["passed"])


def run(dataset: str, arm: str, seed: int, *, output_root: Path, use_cpu: bool = False) -> dict[str, Any]:
    device, physical = run_main._device(use_cpu)
    data = raw_adapter.load_dataset(dataset)
    z, svd_meta = metrics.svd_embedding(data.x0, int(seed), protocol.SVD_COMPONENTS)
    support = __import__("numpy").ones(z.shape, dtype=bool)
    fit = model.fit_autoencoder(z, support, arm=arm, seed=int(seed), device=device, epochs=protocol.BACKBONE["epochs"], batch_size=min(512, z.shape[0]), loss_mode="all")
    labels, label_meta = raw_adapter.load_labels_after_fit(dataset)
    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "REPR_LOCALIZATION",
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "metrics": metrics.evaluate_after_fit(fit.embedding, labels, int(seed)),
        "svd_meta": svd_meta,
        "labels_loaded_during_fit": False,
        "labels_loaded_after_fit": True,
        "labels_sha256": label_meta["labels_sha256"],
        "gpu_physical_id": physical,
        "model_init_hash": fit.model_init_hash,
        "batch_schedule_hash": fit.batch_schedule_hash,
        "mask_schedule_hash": fit.mask_schedule_hash,
        "training_history": fit.history,
        "status": "completed_valid",
    }
    run_main.write_json_atomic(output_root / dataset / arm / f"seed{int(seed)}" / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, default=protocol.MAIN_ROOT)
    parser.add_argument("--output-root", type=Path, default=protocol.REPR_ROOT)
    parser.add_argument("--dataset", choices=protocol.DATASETS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not eligible(args.main_root):
        print(json.dumps({"status": "locked", "reason": "G1_or_G2_not_passed_or_G3_passed"}))
        return 0
    datasets = [args.dataset] if args.dataset else list(protocol.DATASETS)
    rows = [run(dataset, arm, seed, output_root=args.output_root, use_cpu=args.cpu) for dataset in datasets for arm in ("Z_FIXED", "Z_VARIABLE") for seed in protocol.SEEDS]
    run_main.write_json_atomic(args.output_root / "summary.json", {"status": "completed_valid", "rows": rows})
    print(json.dumps({"status": "completed_valid", "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
