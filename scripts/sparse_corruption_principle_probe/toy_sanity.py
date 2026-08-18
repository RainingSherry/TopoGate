"""Run the C2 toy apparatus sensitivity audit and write compact artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import protocol
from .corruption_library import compact_audit, corrupt_matrix, geometry_importance, residual_proxy
from .toy_fixtures import audit_world_definitions, make_world


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def run(output_dir: Path) -> dict[str, Any]:
    protocol.validate_contract()
    fixture_audit = audit_world_definitions()
    rows: list[dict[str, Any]] = []
    for world_name in ("S", "V", "M"):
        world = make_world(world_name)
        geometry = geometry_importance(world.x, k=min(protocol.GEOMETRY_K, world.x.shape[0] - 1))
        residual = residual_proxy(world.x)
        for principle in protocol.PRINCIPLES:
            corrupted, audit = corrupt_matrix(
                world.x,
                principle,
                np.random.default_rng(42),
                residual_scores=residual,
                geometry_scores=geometry,
            )
            rows.append(
                {
                    "world": world_name,
                    "principle": principle,
                    "status": "completed_valid" if audit["exact_budget"] and np.isfinite(corrupted).all() else "protocol_insensitive",
                    "labels_used_during_corruption": False,
                    **compact_audit(audit),
                }
            )
    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "C2_toy_sanity",
        "status": "completed_valid" if fixture_audit["status"] == "completed_valid" and all(row["status"] == "completed_valid" for row in rows) else "protocol_insensitive",
        "fixture_audit": fixture_audit,
        "rows": rows,
        "labels_used_during_corruption": False,
        "labels_used_for_fixture_definition_audit_only": True,
        "performance_claim": False,
    }
    _write_json(output_dir / "toy_sanity.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=protocol.RESULT_ROOT / "C2_toy_sanity")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
