"""Produce the preregistered V26 effect table and case decision without refitting."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V26_support_oracle import protocol


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def _pairwise(rows: dict[str, dict[int, dict[str, Any]]], left: str, right: str) -> dict[str, Any]:
    shared = sorted(set(rows.get(left, {})) & set(rows.get(right, {})))
    values = [float(rows[left][seed]["metrics"]["ARI"]) - float(rows[right][seed]["metrics"]["ARI"]) for seed in shared]
    return {"left": left, "right": right, "metric": "ARI", "paired_seeds": shared, "per_seed": values, "mean": _mean(values), "median": _median(values)}


def _case(diagnostic: dict[str, Any] | None, oracle_gap: float | None) -> dict[str, Any]:
    if diagnostic is None or oracle_gap is None:
        return {"case": "INCOMPLETE", "reason": "missing corrected diagnostics or paired formal cells"}
    support_ari = float(diagnostic["support_only"]["ARI"])
    value_ari = float(diagnostic["value_only"]["ARI"])
    support_advantage = support_ari - value_ari
    support_strong = support_ari >= protocol.SUPPORT_DIAGNOSTIC_MIN_ARI and support_advantage >= protocol.MATERIAL_DELTA_ARI
    material = protocol.MATERIAL_DELTA_ARI
    if support_strong and abs(oracle_gap) < material:
        label, action = "A", "support already appears utilized; continue with adaptive selection"
    elif support_strong and oracle_gap >= material:
        label, action = "B", "algorithmic space remains; study label-free discovery of important support regions"
    elif not support_strong and oracle_gap < material:
        label, action = "C", "freeze the support route and prioritize value representation or a sparse encoder"
    else:
        label, action = "INCONCLUSIVE", "weak support diagnostic with material oracle gain; audit data/protocol before forcing A/B/C"
    return {
        "case": label,
        "action": action,
        "support_only_ARI": support_ari,
        "value_only_ARI": value_ari,
        "support_advantage_ARI": support_advantage,
        "support_strong": support_strong,
        "oracle_gap_ARI_O_LABEL_ORACLE_minus_P2": oracle_gap,
        "material_delta_ARI": material,
    }


def build_summary(output_root: Path) -> dict[str, Any]:
    expected_digest = protocol.implementation_sha256()
    freeze_path = output_root / "FREEZE" / "manifest.json"
    freeze = _read(freeze_path) if freeze_path.exists() else None
    frozen_sources = {item["dataset"]: item for item in (freeze or {}).get("datasets", [])}
    expected_ids = [spec.identifier for spec in protocol.DATASETS]
    manifest_ids = [item.get("dataset") for item in (freeze or {}).get("datasets", [])]
    reconciliation: list[dict[str, Any]] = []
    all_rows: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    coverage: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for spec in protocol.DATASETS:
        rows: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in protocol.ARMS}
        for arm in protocol.ARMS:
            for seed in protocol.SEEDS:
                path = output_root / "runs" / spec.identifier / arm / f"seed{seed}" / "summary.json"
                if not path.exists():
                    continue
                record = _read(path)
                if record.get("status") == "completed_valid" and record.get("implementation", {}).get("source_sha256") == expected_digest:
                    rows[arm][seed] = record
        all_rows[spec.identifier] = rows
        cell_records = [record for arm in protocol.ARMS for record in rows[arm].values()]
        frozen = frozen_sources.get(spec.identifier)
        metadata_fields = ("source_path", "source_size_bytes", "source_mtime_ns")
        metadata_matches = bool(frozen) and all(
            all(record.get("source", {}).get(field) == frozen.get(field) for field in metadata_fields)
            for record in cell_records
        )
        reconciliation.append({
            "dataset": spec.identifier,
            "manifest_record_present": frozen is not None,
            "completed_cell_records": len(cell_records),
            "expected_cell_records": len(protocol.ARMS) * len(protocol.SEEDS),
            "cell_to_manifest_metadata_match": metadata_matches,
            "cell_source_sha256_policy": "joined_from_freeze_manifest; never recomputed per cell",
            "frozen_source_sha256": None if frozen is None else frozen.get("source_sha256"),
        })
        diagnostic_path = output_root / "diagnostics" / spec.identifier / "summary.json"
        diagnostic = _read(diagnostic_path) if diagnostic_path.exists() else None
        if diagnostic is not None and diagnostic.get("implementation", {}).get("source_sha256") != expected_digest:
            diagnostic = None
        complete = all(len(rows[arm]) == len(protocol.SEEDS) for arm in protocol.ARMS)
        arm_metrics = {
            arm: {
                metric: _mean([float(record["metrics"][metric]) for record in rows[arm].values()])
                for metric in ("ARI", "NMI", "ACC")
            }
            for arm in protocol.ARMS
        }
        p2_vs_p1 = _pairwise(rows, "P2_SUPPORT_TARGET", "P1_SUPPORT_PRESERVE")
        p1_vs_p0 = _pairwise(rows, "P1_SUPPORT_PRESERVE", "P0_RANDOM")
        oracle_vs_p2 = _pairwise(rows, "O_LABEL_ORACLE", "P2_SUPPORT_TARGET")
        decision = _case(diagnostic, oracle_vs_p2["mean"])
        coverage.append({"dataset": spec.identifier, "completed_cells": int(sum(len(rows[arm]) for arm in protocol.ARMS)), "expected_cells": len(protocol.ARMS) * len(protocol.SEEDS), "complete": complete})
        results.append({"dataset": spec.identifier, "display_name": spec.display_name, "domain": spec.domain, "frozen_source": frozen_sources.get(spec.identifier), "corrected_diagnostics": diagnostic, "formal_arm_means": arm_metrics, "effects": {"P2_minus_P1": p2_vs_p1, "P1_minus_P0": p1_vs_p0, "oracle_gap": oracle_vs_p2}, "decision": decision})
    domain_summary = {}
    for domain in sorted({spec.domain for spec in protocol.DATASETS}):
        domain_rows = [item for item in results if item["domain"] == domain]
        domain_summary[domain] = {
            "datasets": [item["dataset"] for item in domain_rows],
            "median_support_advantage_ARI": _median([item["decision"]["support_advantage_ARI"] for item in domain_rows if "support_advantage_ARI" in item["decision"]]),
            "median_oracle_gap_ARI": _median([item["decision"]["oracle_gap_ARI_O_LABEL_ORACLE_minus_P2"] for item in domain_rows if "oracle_gap_ARI_O_LABEL_ORACLE_minus_P2" in item["decision"]]),
            "case_counts": {case: sum(item["decision"]["case"] == case for item in domain_rows) for case in ("A", "B", "C", "INCONCLUSIVE", "INCOMPLETE")},
        }
    manifest_sha256 = hashlib.sha256(freeze_path.read_bytes()).hexdigest() if freeze_path.exists() else None
    join_is_bijective = (
        len(manifest_ids) == len(set(manifest_ids)) == len(expected_ids)
        and set(manifest_ids) == set(expected_ids) == set(frozen_sources)
    )
    metadata_reconciled = all(item["cell_to_manifest_metadata_match"] for item in reconciliation)
    return {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "implementation_revision": protocol.IMPLEMENTATION_REVISION,
        "formal_result_source_sha256": expected_digest,
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_implementation_sha256": (freeze or {}).get("implementation", {}).get("source_sha256"),
        "freeze_manifest_sha256": manifest_sha256,
        "source_reconciliation": {
            "dataset_id_join_is_unique_and_complete": join_is_bijective,
            "cell_metadata_matches_frozen_manifest": metadata_reconciled,
            "records": reconciliation,
        },
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision_rule": protocol.resolved_config()["decision_rule"],
        "coverage": coverage,
        "per_dataset": results,
        "domain_summary": domain_summary,
        "status": "completed_valid" if all(row["complete"] for row in coverage) and join_is_bijective and metadata_reconciled else "incomplete_compute",
    }


def main() -> int:
    summary = build_summary(protocol.RESULT_ROOT)
    _write(protocol.RESULT_ROOT / "ANALYSIS" / "support_oracle_decision.json", summary)
    print(json.dumps({"status": summary["status"], "output": str(protocol.RESULT_ROOT / "ANALYSIS" / "support_oracle_decision.json")}, sort_keys=True))
    return 0 if summary["status"] == "completed_valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
