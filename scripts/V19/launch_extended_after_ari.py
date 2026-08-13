#!/usr/bin/env python
"""Launch the fixed sparse extension after the ARI selection is available.

This helper is intentionally a sequencing wrapper.  It does not select a
dataset, candidate, seed, or metric after observing extension outcomes.  It
waits for the preregistered V19 ARI refine selection, verifies that no V19
worker is still active, materializes the selected configuration, then starts
one extension worker per explicitly allowed GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
DEFAULT_ARI_ROOT = ROOT / "result" / "V19" / "v19_rg_ari_dev_tuning_v1"
DEFAULT_MANIFEST = ROOT / "result" / "V19" / "v19_rg_extended_sparse_manifest_20260811.json"
DEFAULT_OUTPUT = ROOT / "result" / "V19" / "v19_rg_extended_sparse_ari_v1"
DEFAULT_BASE_CONFIG = ROOT / "methods" / "TopoGate" / "V19_rg_adapter" / "configs" / "v19_rg.yaml"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_free_mib() -> dict[int, int]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for the GPU extension launcher")
    completed = subprocess.run(
        [executable, "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    result: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            continue
        index, used, total = (int(value) for value in fields)
        result[index] = total - used
    return result


def _active_v19_processes() -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        check=False,
        capture_output=True,
        text=True,
    )
    needles = (
        "continue_ari_pipeline.py",
        "launch_ari_dev.py",
        "tune_ari_dev.py",
        "run_ari_final.py",
    )
    own_pid = str(os.getpid())
    rows: list[str] = []
    for line in completed.stdout.splitlines():
        if own_pid in line:
            continue
        if any(needle in line for needle in needles):
            rows.append(line.strip())
    return rows


def _wait_for_selection(ari_root: Path, wait_seconds: int) -> dict[str, Any]:
    selected_path = ari_root / "refine" / "selected_config.json"
    selection_summary = ari_root / "refine" / "ari_selection.json"
    while True:
        if selected_path.is_file() and selection_summary.is_file():
            selected = _read(selected_path)
            summary = _read(selection_summary)
            if (
                selected.get("status") == "completed"
                and selected.get("protocol_id") == "v19_rg_ari_dev_tuning_v1"
                and int(summary.get("completed_runs", 0)) == int(summary.get("expected_runs", 288)) == 288
                and selected.get("labels_used_during_fit") is False
                and selected.get("labels_used_for_selection") is True
            ):
                return selected
        time.sleep(max(5, int(wait_seconds)))


def _materialize_config(base_config: Path, selected: dict[str, Any], output_root: Path) -> Path:
    loaded = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"base V19 config is not a mapping: {base_config}")
    overrides = selected.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("ARI selected config overrides must be a mapping")
    loaded.update({str(key): value for key, value in overrides.items()})
    loaded["protocol_id"] = "v19_rg_extended_sparse_ari_transfer_v1"
    loaded["variant"] = "rg_full"
    config_path = output_root / "resolved_ari_transfer_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
    return config_path


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"extension launcher lock already exists: {path}") from exc


def _launch_workers(
    *,
    manifest: Path,
    output_root: Path,
    config: Path,
    gpus: list[int],
) -> int:
    logs = output_root / "launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, int, subprocess.Popen[str], Any]] = []
    for worker_id, gpu in enumerate(gpus):
        environment = dict(os.environ)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_handle = (logs / f"worker{worker_id}_gpu{gpu}.log").open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "V19" / "run_extended_matrix.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_root),
            "--config",
            str(config),
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(len(gpus)),
            "--gpu",
            str(gpu),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_id, gpu, process, log_handle))
    _write(
        output_root / "extension_launcher_status.json",
        {
            "status": "running",
            "manifest": str(manifest.resolve()),
            "config": str(config.resolve()),
            "gpus": gpus,
            "workers": [{"worker_id": wid, "gpu": gpu, "pid": proc.pid} for wid, gpu, proc, _ in processes],
            "started_at": time.time(),
        },
    )
    exit_codes: dict[str, int] = {}
    for worker_id, gpu, process, log_handle in processes:
        exit_codes[str(worker_id)] = int(process.wait())
        log_handle.close()
    completed = all(code == 0 for code in exit_codes.values())
    summary_code = None
    if completed:
        summary_command = [
            sys.executable,
            str(ROOT / "scripts" / "V19" / "summarize_extended_matrix.py"),
            "--manifest",
            str(manifest),
            "--result-dir",
            str(output_root),
            "--output-dir",
            str(output_root / "summary"),
        ]
        summary_log = (logs / "summarize.log").open("a", encoding="utf-8")
        try:
            summary_code = int(
                subprocess.run(
                    summary_command,
                    cwd=ROOT,
                    check=False,
                    stdout=summary_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).returncode
            )
        finally:
            summary_log.close()
        completed = summary_code == 0
    _write(
        output_root / "extension_launcher_status.json",
        {
            "status": "completed" if completed else "incomplete_compute",
            "manifest": str(manifest.resolve()),
            "config": str(config.resolve()),
            "gpus": gpus,
            "exit_codes": exit_codes,
            "summary_exit_code": summary_code,
            "finished_at": time.time(),
        },
    )
    return 0 if completed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ari-root", type=Path, default=DEFAULT_ARI_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--min-free-mib", type=int, default=30000)
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(gpu) for gpu in args.gpus]
    if not gpus or any(gpu not in ALLOWED_GPUS for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU pool must be distinct physical GPUs in {sorted(ALLOWED_GPUS)}")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    selected = _wait_for_selection(args.ari_root, int(args.wait_seconds))
    active = _active_v19_processes()
    while active:
        time.sleep(max(5, int(args.wait_seconds)))
        active = _active_v19_processes()
    free = _gpu_free_mib()
    eligible = [gpu for gpu in gpus if int(free.get(gpu, 0)) >= int(args.min_free_mib)]
    if not eligible:
        raise RuntimeError(f"no requested GPU has {args.min_free_mib} MiB free: {free}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_config(args.base_config, selected, args.output_dir)
    _write(args.output_dir / "ari_selection_snapshot.json", selected)
    _write(
        args.output_dir / "extension_launch_spec.json",
        {
            "protocol_id": "v19_rg_extended_sparse_ari_transfer_v1",
            "manifest": str(args.manifest.resolve()),
            "config": str(config_path.resolve()),
            "selected_candidate_id": selected.get("candidate_id"),
            "selected_overrides": selected.get("overrides", {}),
            "selection_source": str((args.ari_root / "refine" / "selected_config.json").resolve()),
            "selection_uses_labels": True,
            "labels_used_during_fit": False,
            "gpus_requested": gpus,
            "gpus_selected": eligible,
            "free_mib_at_launch": {str(gpu): int(free.get(gpu, 0)) for gpu in eligible},
            "started_at": time.time(),
        },
    )
    lock_fd = _acquire_lock(args.output_dir / "extension_launcher.lock")
    try:
        if args.dry_run:
            print(json.dumps({"selected_candidate_id": selected.get("candidate_id"), "config": str(config_path), "gpus": eligible}, ensure_ascii=True))
            return 0
        return _launch_workers(manifest=args.manifest, output_root=args.output_dir, config=config_path, gpus=eligible)
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
