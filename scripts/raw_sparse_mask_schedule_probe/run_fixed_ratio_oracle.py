"""Conditional fixed-ratio diagnostic unlocked only when G2 passes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import aggregate, metrics, model, protocol, raw_adapter, run_main


RATIOS = (0.05, 0.15, 0.25, 0.35, 0.45)


def _gate_passes(main_root: Path) -> bool:
    return bool(aggregate.evaluate(aggregate.collect(main_root))["g2"]["passed"])


def run(dataset: str, ratio: float, *, output_root: Path, use_cpu: bool = False) -> dict[str, Any]:
    device, physical = run_main._device(use_cpu)
    data = raw_adapter.load_dataset(dataset)
    batch_size, _ = run_main._batch_preflight(data, device, 42, protocol.MAIN_ROOT)
    out = output_root / dataset / f"ratio_{ratio:.2f}" / "seed42"
    out.mkdir(parents=True, exist_ok=True)
    fit = model.fit_autoencoder(data.x0, data.active, arm="ACTIVE_FIXED", seed=42, device=device, epochs=protocol.BACKBONE["epochs"], batch_size=batch_size, fixed_ratio=ratio)
    labels, label_meta = raw_adapter.load_labels_after_fit(dataset)
    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "FIXED_RATIO_ORACLE",
        "dataset": dataset,
        "ratio": ratio,
        "seed": 42,
        "metrics": metrics.evaluate_after_fit(fit.embedding, labels, 42),
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
    run_main.write_json_atomic(out / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, default=protocol.MAIN_ROOT)
    parser.add_argument("--output-root", type=Path, default=protocol.FIXED_ROOT)
    parser.add_argument("--dataset", choices=protocol.DATASETS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not _gate_passes(args.main_root):
        print(json.dumps({"status": "locked", "reason": "G2_not_passed"}))
        return 0
    datasets = [args.dataset] if args.dataset else list(protocol.DATASETS)
    rows = []
    for dataset in datasets:
        for ratio in RATIOS:
            if ratio == protocol.FIXED_MASK_RATIO and (args.output_root / dataset / f"ratio_{ratio:.2f}" / "seed42" / "summary.json").exists():
                continue
            rows.append(run(dataset, ratio, output_root=args.output_root, use_cpu=args.cpu))
    run_main.write_json_atomic(args.output_root / "summary.json", {"status": "completed_valid", "rows": rows, "conditional": True})
    print(json.dumps({"status": "completed_valid", "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
