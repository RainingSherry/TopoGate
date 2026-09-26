"""Restartable validation-only queue for the frozen four-dataset pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import traceback

from .config import V0_RGConfig, load_config
from .validation_protocol import run_validation_only


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _append_event(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _valid_completed_record(path: Path, task: dict) -> bool:
    """Only reuse an atomic, complete record from this exact validation task."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        split = json.loads(Path(task["tuning_manifest_path"]).read_text(encoding="utf-8"))
        expected_config = asdict(V0_RGConfig(**task["config"]))
        recorded_config = asdict(V0_RGConfig(**record["config"]))
        return (
            record.get("status") == "completed"
            and record.get("seed") == int(task["seed"])
            and recorded_config == expected_config
            and record.get("test_labels_loaded") is False
            and record.get("selection_label_scope") == "validation_only"
            and record.get("feature_sha256") == split["feature_sha256"]
            and record.get("validation_labels_sha256") == split["validation_labels_sha256"]
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def run_queue(manifest_path: str | Path, output_root: str | Path, *, device: str,
              precompleted_root: str | Path | None = None,
              datasets: set[str] | None = None,
              worker_index: int = 0,
              worker_count: int = 1,
              max_attempts: int = 1,
              continue_on_error: bool = False) -> None:
    if worker_count < 1 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    events = output_root / "queue_events.jsonl"
    reuse = None if precompleted_root is None else Path(precompleted_root)
    for task in manifest["tasks"]:
        dataset, hop, seed = task["dataset"], task["hop_variant"], int(task["seed"])
        config_id = str(task.get("config_id", hop))
        if not config_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in config_id):
            raise ValueError(f"unsafe config_id: {config_id!r}")
        if datasets is not None and dataset not in datasets:
            continue
        task_key = f"{dataset}|{config_id}|{seed}"
        bucket = int(hashlib.sha256(task_key.encode("utf-8")).hexdigest()[:16], 16) % worker_count
        if bucket != worker_index:
            continue
        seed_root = output_root / "runs" / dataset / config_id / f"seed_{seed}"
        existing_records = sorted(seed_root.glob(f"attempt_*/record_seed_{seed}.json"))
        if any(_valid_completed_record(path, task) for path in existing_records):
            _append_event(events, {"status": "skipped_completed", "dataset": dataset, "config_id": config_id,
                                   "hop_variant": hop, "seed": seed})
            continue
        reuse_record = None if reuse is None or config_id != hop else reuse / dataset / hop / f"seed_{seed}" / f"record_seed_{seed}.json"
        if reuse_record is not None and reuse_record.exists():
            source = json.loads(reuse_record.read_text(encoding="utf-8"))
            if source.get("test_labels_loaded") is not False:
                raise ValueError("precompleted run accessed final-test labels")
            _append_event(events, {"status": "reused_completed", "dataset": dataset, "config_id": config_id, "hop_variant": hop,
                                   "seed": seed, "record_path": str(reuse_record.resolve())})
            continue
        existing_attempts = sorted(seed_root.glob("attempt_*"))
        next_attempt = len(existing_attempts) + 1
        while next_attempt <= int(max_attempts):
            run_dir = seed_root / f"attempt_{next_attempt:03d}"
            run_dir.mkdir(parents=True, exist_ok=False)
            config_path = run_dir / "config.json"
            _write_json(config_path, task["config"])
            try:
                record = run_validation_only(task["tuning_manifest_path"], run_dir, config=load_config(config_path),
                                             seed=seed, input_kind=task["input_kind"],
                                             feature_limit=int(task["feature_limit"]), device=device)
                _append_event(events, {"status": "completed", "dataset": dataset, "config_id": config_id,
                                       "hop_variant": hop, "seed": seed,
                                       "validation_ari": record["validation_metrics"]["ari"]})
                break
            except Exception as exc:
                _write_json(run_dir / "failed.json", {"status": "failed", "error": repr(exc),
                                                        "traceback": traceback.format_exc()})
                _append_event(events, {"status": "failed", "dataset": dataset, "config_id": config_id,
                                       "hop_variant": hop, "seed": seed, "attempt": next_attempt,
                                       "error": repr(exc)})
                next_attempt += 1
                if next_attempt > int(max_attempts) and not continue_on_error:
                    raise


def main() -> None:
    parser = argparse.ArgumentParser(description="run frozen V0_RG validation-only pilot queue")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precompleted-root")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    run_queue(args.manifest, args.output_root, device=args.device, precompleted_root=args.precompleted_root,
              datasets=None if args.datasets is None else set(args.datasets),
              worker_index=args.worker_index, worker_count=args.worker_count,
              max_attempts=args.max_attempts, continue_on_error=args.continue_on_error)


if __name__ == "__main__":
    main()
