"""Shared, dependency-light helpers for the source_coverage_20260925 experiment.

Everything here is label-free.  Paths default to the deployed layout
``R/source_coverage_20260925`` and can be overridden with ``V0RG_ROOT``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EXP = Path(__file__).resolve().parent
R = Path(os.environ.get("V0RG_ROOT",
                        "/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922"))
CORE_PACKAGE = "v0rg_tunable_multihop"
AUTHORIZED_GPUS = (2, 3, 4, 6)
HISTORICAL_TRAIN_SEEDS = (42, 123, 7)
ARMS = ("H1_anchor", "flat_anchor", "geometric_anchor", "flat_mixed", "geometric_mixed")
ARM_SPECS = {
    "H1_anchor": {"candidates": "H1", "prior": "one_hop", "target": "anchor"},
    "flat_anchor": {"candidates": "multihop", "prior": "flat_q", "target": "anchor"},
    "geometric_anchor": {"candidates": "multihop", "prior": "geometric", "target": "anchor"},
    "flat_mixed": {"candidates": "multihop", "prior": "flat_q", "target": "mixed"},
    "geometric_mixed": {"candidates": "multihop", "prior": "geometric", "target": "mixed"},
}
# Code files whose hashes the parent approval must bind.  Extension scripts
# referenced by queue commands are added dynamically by the controller.
CODE_FILES = ("common.py", "audit_select.py", "prepare_factorial.py", "factorial_plan.py",
              "run_factorial.py", "analyze_cases.py", "controller.py", "launch.sh",
              "tests_factorial.py")
TRAIN_REQUIRED_ARTIFACTS = (
    "model.pt", "embedding.npy", "predictions.npy", "cluster_centers.npy",
    "preprocessor.npz", "preprocessor.json", "plan.npz", "graph.npz",
    "source_trace.bin", "displacement_by_anchor.npz", "training_history.json",
    "hop_sampling.json", "plan_audit.json",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _tmp(path)
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path: str | Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_npy(path: str | Path, value: np.ndarray) -> None:
    path = Path(path)
    temporary = _tmp(path)
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def atomic_npz(path: str | Path, **values: np.ndarray) -> None:
    path = Path(path)
    temporary = _tmp(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def atomic_torch(path: str | Path, value: Any) -> None:
    import torch

    path = Path(path)
    temporary = _tmp(path)
    torch.save(value, temporary)
    temporary.replace(path)


def write_once_json(path: str | Path, value: Any) -> str:
    """Write an immutable JSON file; identical re-writes are no-ops, changes refuse."""
    path = Path(path)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite immutable file {path}")
        return sha256(path)
    atomic_bytes(path, encoded)
    atomic_text(path.with_suffix(path.suffix + ".sha256"), sha256(path) + "\n")
    return sha256(path)


def verify_sidecar_hash(path: str | Path) -> str:
    path = Path(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text().strip()
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    return actual


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def now() -> float:
    return time.time()


def core_source_hashes(root: Path | None = None) -> dict[str, str]:
    base = (root or R) / CORE_PACKAGE
    return {path.name: sha256(path) for path in sorted(base.glob("*.py"))}


def imported_core_hashes() -> dict[str, str]:
    """Hash the package that is actually imported in this process."""
    from v0rg_tunable_multihop import trainer

    base = Path(trainer.__file__).resolve().parent
    return {path.name: sha256(path) for path in sorted(base.glob("*.py"))}


def code_hashes(names: Iterable[str], root: Path = EXP) -> dict[str, str]:
    return {name: sha256(root / name) for name in names}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def resolve_path_variant(hop_variant: str) -> tuple[int, int] | None:
    """(max_hop, width) for exact-shell path variants; None for H1/Kwide."""
    legacy = {"H12": (2, 1), "H123": (3, 1)}
    if hop_variant in legacy:
        return legacy[hop_variant]
    matched = re.fullmatch(r"H([2-8])_B(1|2|4)", str(hop_variant))
    return None if matched is None else (int(matched.group(1)), int(matched.group(2)))


def hop_family(hop_variant: str) -> str:
    if hop_variant == "H1":
        return "H1"
    if resolve_path_variant(hop_variant) is not None:
        return "multihop"
    return "other"


# ---------------------------------------------------------------- validation

class ValidationError(RuntimeError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_train_output(task: dict, task_path: Path, approved_code: dict | None = None) -> dict:
    """Verify every required training artifact and the record audit; returns record."""
    out = Path(task["out"])
    record_path = out / "record.json"
    _check(record_path.is_file(), "record.json missing")
    record = read_json(record_path)
    _check(record.get("status") == "completed", "record status is not completed")
    _check(record.get("smoke") is False, "smoke record cannot count as formal output")
    _check(record.get("epochs_override") is None, "epochs override used")
    _check(record.get("labels_used_during_fit") is False, "labels_used_during_fit is not false")
    _check(record.get("test_labels_loaded") is False, "test_labels_loaded is not false")
    _check(record.get("full_labels_loaded") is False, "full_labels_loaded is not false")
    _check(record.get("task_id") == task["id"], "task id mismatch")
    _check(record.get("task_sha256") == sha256(task_path), "task file hash mismatch")
    task_payload = read_json(task_path)
    _check(record.get("config_sha256") == canonical_sha256(task_payload["config"]),
           "config hash mismatch")
    _check(record.get("arm") == task_payload["arm"], "arm mismatch")
    _check(int(record.get("seed", -1)) == int(task_payload["seed"]), "seed mismatch")
    _check(record.get("source_hashes_before") == record.get("source_hashes_after"),
           "core source changed during run")
    if approved_code is not None:
        _check(record.get("runner_sha256") == approved_code.get("run_factorial.py"),
               "runner hash differs from approved code")
        _check(record.get("plan_module_sha256") == approved_code.get("factorial_plan.py"),
               "plan module hash differs from approved code")
    artifacts = record.get("artifact_sha256", {})
    for name in TRAIN_REQUIRED_ARTIFACTS:
        _check(name in artifacts, f"artifact not recorded: {name}")
    for name, expected in artifacts.items():
        path = out / name
        _check(path.is_file() and path.stat().st_size > 0, f"artifact missing/empty: {name}")
        _check(sha256(path) == expected, f"artifact hash mismatch: {name}")
    trace = record.get("training_trace", {})
    _check(int(trace.get("draws", 0)) > 0, "no source draws traced")
    _check(int(trace.get("anchor_training_occurrences", 0)) ==
           int(record["n_samples"]) * int(record["config"]["epochs"]),
           "trace does not cover every anchor occurrence")
    if ARM_SPECS[task_payload["arm"]]["target"] == "mixed":
        calls = record.get("target_calls", {})
        _check(int(calls.get("mixed", 0)) > 0 and int(calls.get("mixed", 0)) == int(calls.get("real", -1)),
               "mixed target was not applied to every pseudo loss")
    return record


def validate_summary_output(task: dict) -> dict:
    out = Path(task["out"])
    path = out / task.get("summary", "summary.json")
    _check(path.is_file(), f"summary missing: {path.name}")
    summary = read_json(path)
    allowed = ({"completed"} | ({"blocked"} if task.get("allow_blocked") else set())
               | ({"partial"} if task.get("allow_partial") else set()))
    _check(summary.get("status") in allowed, f"summary status {summary.get('status')!r}")
    for name, expected in (summary.get("output_sha256") or {}).items():
        target = out / name
        _check(target.is_file(), f"summary output missing: {name}")
        _check(sha256(target) == expected, f"summary output hash mismatch: {name}")
    if task["kind"] == "refcheck":
        _check(summary.get("all_checks_passed") is True, "reference checks did not pass")
    return summary


def validate_task_output(task: dict, root: Path = EXP, approved_code: dict | None = None) -> dict:
    task = dict(task)
    if not Path(task["out"]).is_absolute():
        task["out"] = str(root / task["out"])
    if task.get("validator") == "schntl":
        from run_schntl import validate_output
        result = validate_output(root / task["task_file"], task["out"])
        if approved_code is not None:
            _check(result.get("runner_sha256") == approved_code.get("run_schntl.py"), "baseline runner drift")
        return result
    if task["kind"] == "gpu":
        return validate_train_output(task, root / task["task_file"], approved_code)
    return validate_summary_output(task)


def directory_manifest(out: Path, exclude: Iterable[str] = ("TASK_COMPLETED.json",)) -> dict:
    skip = set(exclude)
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in skip and not path.name.endswith(".tmp"):
            files[str(path.relative_to(out))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return files
