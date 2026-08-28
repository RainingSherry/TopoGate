#!/usr/bin/env python3
"""Freeze source/config/data hashes for a V21 formal matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "methods/TopoGate/V21_assignment_adversarial_gate/config.py",
    "methods/TopoGate/V21_assignment_adversarial_gate/graph.py",
    "methods/TopoGate/V21_assignment_adversarial_gate/input_adapter.py",
    "methods/TopoGate/V21_assignment_adversarial_gate/model.py",
    "methods/TopoGate/V21_assignment_adversarial_gate/run.py",
    "methods/TopoGate/V21_assignment_adversarial_gate/trainer.py",
    "scripts/V21/run_formal_matrix.py",
    "scripts/V21/summarize_formal_matrix.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, relative: str) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "relative_path": relative,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze V21 matrix provenance")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/V21/v21_formal6_full_20260811_graphfix"),
    )
    args = parser.parse_args()
    root = args.output_dir
    spec = json.loads((root / "stage_spec.json").read_text(encoding="utf-8"))
    jobs = json.loads((root / "launcher_state.json").read_text(encoding="utf-8"))["jobs"]
    data_paths = {str(Path(job["data"]).resolve()) for job in jobs}
    config_paths = {str(Path(job["config"]).resolve()) for job in jobs}
    payload = {
        "matrix_protocol_id": spec["protocol_id"],
        "model_protocol_id": "v21_assignment_adversarial_v2_graphfix_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [_file_record(ROOT / relative, relative) for relative in SOURCE_FILES],
        "config_files": [_file_record(Path(path), str(Path(path).relative_to(ROOT))) for path in sorted(config_paths)],
        "data_files": [_file_record(Path(path), str(Path(path))) for path in sorted(data_paths)],
        "label_isolation": spec["label_isolation"],
    }
    target = root / "provenance.json"
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
