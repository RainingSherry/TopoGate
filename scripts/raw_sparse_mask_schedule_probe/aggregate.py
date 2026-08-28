"""Aggregate MAIN/SVD compact summaries and apply frozen gates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from . import protocol, provenance


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summaries(root: Path, arm_set: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(arm_set)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in root.glob("**/summary.json"):
        try:
            row = _read(path)
        except Exception:
            continue
        if row.get("arm") in wanted:
            row["_path"] = str(path)
            rows.append(row)
    return rows


def collect(main_root: Path = protocol.MAIN_ROOT) -> dict[str, Any]:
    rows = _summaries(main_root, protocol.ARMS)
    svd = _summaries(main_root / "SVD32", ("SVD32",))
    by_key = {(r.get("dataset"), r.get("arm"), int(r.get("seed", -1))): r for r in rows}
    svd_by_key = {(r.get("dataset"), int(r.get("seed", -1))): r for r in svd}
    expected = {(d, a, s) for d in protocol.DATASETS for a in protocol.ARMS for s in protocol.SEEDS}
    observed = set(by_key)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    return {"rows": rows, "svd_rows": svd, "by_key": by_key, "svd_by_key": svd_by_key, "missing": missing, "extra": extra}


def _metric(row: dict[str, Any], name: str = "ARI") -> float | None:
    value = row.get("metrics", {}).get(name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _paired_values(bundle: dict[str, Any], arm_a: str, arm_b: str, dataset: str) -> list[float]:
    values: list[float] = []
    for seed in protocol.SEEDS:
        left = bundle["by_key"].get((dataset, arm_a, seed))
        right = bundle["by_key"].get((dataset, arm_b, seed))
        if left is None or right is None:
            return []
        left_value, right_value = _metric(left), _metric(right)
        if left_value is None or right_value is None:
            return []
        values.append(float(left_value - right_value))
    return values


def comparison(bundle: dict[str, Any], dataset: str, arm_a: str, arm_b: str) -> dict[str, Any]:
    values = _paired_values(bundle, arm_a, arm_b, dataset)
    mean = sum(values) / len(values) if values else None
    return {"dataset": dataset, "left": arm_a, "right": arm_b, "seed_deltas": values, "mean_delta": mean, "positive_seed_count": sum(v > 0 for v in values), "negative_seed_count": sum(v < 0 for v in values), "material_positive": bool(mean is not None and mean >= protocol.MATERIAL_DELTA_ARI and sum(v > 0 for v in values) >= 2), "material_negative": bool(mean is not None and mean <= -protocol.MATERIAL_DELTA_ARI and sum(v < 0 for v in values) >= 2)}


def _g0(bundle: dict[str, Any]) -> dict[str, Any]:
    rows = bundle["rows"]
    failures: list[str] = []
    current_code_hash = provenance.code_sha256()
    freeze_code_hash = None
    freeze_path = protocol.FREEZE_ROOT / "freeze_manifest.json"
    if freeze_path.exists():
        try:
            freeze_code_hash = _read(freeze_path).get("code_sha256")
        except Exception:
            freeze_code_hash = None
    if freeze_code_hash != current_code_hash:
        failures.append("freeze_code_hash_drift")
    if bundle["missing"] or bundle["extra"] or len(rows) != len(protocol.DATASETS) * len(protocol.ARMS) * len(protocol.SEEDS):
        failures.append("main_matrix_coverage")
    for row in rows:
        if row.get("status") != "completed_valid":
            failures.append(f"status:{row.get('dataset')}:{row.get('arm')}:{row.get('seed')}")
        if row.get("audit_ok") is not True or row.get("labels_used_during_fit") is not False or row.get("labels_loaded_during_fit") is not False:
            failures.append(f"label_or_audit:{row.get('dataset')}:{row.get('arm')}:{row.get('seed')}")
        gpu = row.get("gpu_physical_id")
        if gpu is not None and int(gpu) not in protocol.LEGAL_GPU_POOL:
            failures.append(f"gpu:{gpu}")
        if not row.get("source_sha256") or not row.get("adapter_hash") or not row.get("scale_hash"):
            failures.append(f"hash:{row.get('dataset')}:{row.get('arm')}:{row.get('seed')}")
        if not row.get("code_sha256") or row.get("code_sha256") != current_code_hash:
            failures.append(f"code_hash:{row.get('dataset')}:{row.get('arm')}:{row.get('seed')}")
        if row.get("arm") != "CLEAN_AE" and not row.get("mask_schedule_hash"):
            failures.append(f"mask_hash:{row.get('dataset')}:{row.get('arm')}:{row.get('seed')}")
        if row.get("adapter_manifest", {}).get("zero_pattern_preserved") is not True:
            failures.append(f"zero_pattern:{row.get('dataset')}:{row.get('arm')}:{row.get('seed')}")
        history = row.get("training_history")
        if not isinstance(history, list) or not history or any(h.get("mask_count_exact") is not True for h in history):
            failures.append(f"mask_budget_audit:{row.get('dataset')}:{row.get('arm')}:{row.get('seed')}")
        if any(not isinstance(h.get("loss"), (int, float)) or h.get("loss") != h.get("loss") for h in history or []):
            failures.append(f"nonfinite_history:{row.get('dataset')}:{row.get('arm')}:{row.get('seed')}")
    for dataset in protocol.DATASETS:
        for seed in protocol.SEEDS:
            cells = [bundle["by_key"].get((dataset, arm, seed)) for arm in protocol.ARMS]
            cells = [cell for cell in cells if cell is not None]
            init_hashes = {cell.get("model_init_hash") for cell in cells}
            batch_hashes = {cell.get("batch_schedule_hash") for cell in cells}
            if len(init_hashes) != 1:
                failures.append(f"paired_init_hash:{dataset}:{seed}")
            if len(batch_hashes) != 1:
                failures.append(f"paired_batch_hash:{dataset}:{seed}")
    expected_svd = {(d, s) for d in protocol.DATASETS for s in protocol.SEEDS}
    observed_svd = set(bundle["svd_by_key"])
    if observed_svd != expected_svd:
        failures.append("svd_matrix_coverage")
    for key, row in bundle["svd_by_key"].items():
        if row.get("status") != "completed_valid" or row.get("labels_loaded_during_fit") is not False or row.get("code_sha256") != current_code_hash or _metric(row) is None:
            failures.append(f"svd_invalid:{key[0]}:{key[1]}")
    return {"passed": not failures, "failures": failures, "expected_cells": len(protocol.DATASETS) * len(protocol.ARMS) * len(protocol.SEEDS), "completed_rows": len(rows)}


def evaluate(bundle: dict[str, Any]) -> dict[str, Any]:
    g0 = _g0(bundle)
    comparisons = {
        "Delta_support": [comparison(bundle, d, "ACTIVE_FIXED", "ALL_FIXED") for d in protocol.DATASETS],
        "Delta_scale_active": [comparison(bundle, d, "ACTIVE_VARIABLE", "ACTIVE_FIXED") for d in protocol.DATASETS],
        "Delta_scale_all": [comparison(bundle, d, "ALL_VARIABLE", "ALL_FIXED") for d in protocol.DATASETS],
    }
    for d in protocol.DATASETS:
        active = comparisons["Delta_scale_active"][protocol.DATASETS.index(d)]
        all_scale = comparisons["Delta_scale_all"][protocol.DATASETS.index(d)]
        active["I_mask"] = (active["mean_delta"] - all_scale["mean_delta"]) if active["mean_delta"] is not None and all_scale["mean_delta"] is not None else None

    def role_pass(comparison_rows: list[dict[str, Any]], role: str) -> bool:
        selected = [r for r in comparison_rows if protocol.ROLE_BY_DATASET[r["dataset"]] == role]
        return sum(bool(r["material_positive"]) for r in selected) >= 2

    g1 = role_pass(comparisons["Delta_support"], "biological") and role_pass(comparisons["Delta_support"], "nonbiological")
    scale_rows = comparisons["Delta_scale_active"]
    g2 = role_pass(scale_rows, "biological") and role_pass(scale_rows, "nonbiological") and sum(bool(r["material_negative"]) for r in scale_rows) <= 1
    candidate = "ACTIVE_VARIABLE" if g2 else ("ACTIVE_FIXED" if g1 else None)
    necessity: dict[str, Any] = {"candidate": candidate, "clean": [], "svd": [], "passed": False}
    if candidate:
        necessity["clean"] = [comparison(bundle, d, candidate, "CLEAN_AE") for d in protocol.DATASETS]
        necessity["svd"] = [
            {"dataset": d, "left": candidate, "right": "SVD32", "seed_deltas": [(_metric(bundle["by_key"][(d, candidate, s)]) - _metric(bundle["svd_by_key"][(d, s)])) if (d, candidate, s) in bundle["by_key"] and (d, s) in bundle["svd_by_key"] and _metric(bundle["by_key"][(d, candidate, s)]) is not None and _metric(bundle["svd_by_key"][(d, s)]) is not None else None for s in protocol.SEEDS]}
            for d in protocol.DATASETS
        ]
        for row in necessity["svd"]:
            values = list(row["seed_deltas"])
            if len(values) != len(protocol.SEEDS) or any(v is None for v in values):
                row["incomplete"] = True
                values = []
            row["mean_delta"] = sum(values) / len(values) if values else None
            row["positive_seed_count"] = sum(v > 0 for v in values)
            row["material_positive"] = bool(row["mean_delta"] is not None and row["mean_delta"] >= protocol.MATERIAL_DELTA_ARI and row["positive_seed_count"] >= 2)
        necessity["passed"] = role_pass(necessity["clean"], "biological") and role_pass(necessity["clean"], "nonbiological") and role_pass(necessity["svd"], "biological") and role_pass(necessity["svd"], "nonbiological")
    if not g0["passed"]:
        decision = "INCOMPLETE_COMPUTE"
    elif not g1 and not g2:
        decision = "STOP_RAW_SUPPORT_MASKING"
    elif not g2:
        decision = "STOP_MULTISCALE_MASKING"
    elif not necessity["passed"]:
        decision = "MECHANISM_SENSITIVITY_NO_METHOD_NECESSITY"
    else:
        decision = "RAW_SPARSE_MASK_PRINCIPLE_CANDIDATE"
    return {"g0": g0, "g1": {"passed": g1, "comparison": comparisons["Delta_support"]}, "g2": {"passed": g2, "comparison": comparisons["Delta_scale_active"]}, "comparisons": comparisons, "necessity": necessity, "decision": decision, "status": "completed_valid" if g0["passed"] else "incomplete_compute"}


def _md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    out = ["| " + " | ".join(header for header, _ in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join(out)


def write_outputs(bundle: dict[str, Any], evaluation: dict[str, Any], report_root: Path = protocol.REPORT_ROOT, final_root: Path = protocol.FINAL_ROOT) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    for d in protocol.DATASETS:
        row: dict[str, Any] = {"Dataset": d, "Domain": protocol.ROLE_BY_DATASET[d]}
        for arm in ("SVD32",) + protocol.ARMS:
            source = bundle["svd_by_key"].get((d, 42)) if arm == "SVD32" else bundle["by_key"].get((d, arm, 42))
            row[arm] = "" if source is None else f"{_metric(source):.6f}" if _metric(source) is not None else ""
        result_rows.append(row)
    comparison_rows = []
    for d, support, active, all_scale in zip(protocol.DATASETS, evaluation["comparisons"]["Delta_support"], evaluation["comparisons"]["Delta_scale_active"], evaluation["comparisons"]["Delta_scale_all"]):
        clean = next((r for r in evaluation["necessity"].get("clean", []) if r["dataset"] == d), {})
        svd = next((r for r in evaluation["necessity"].get("svd", []) if r["dataset"] == d), {})
        comparison_rows.append({"Dataset": d, "Delta_support": support.get("mean_delta"), "Delta_scale_active": active.get("mean_delta"), "Delta_scale_all": all_scale.get("mean_delta"), "I_mask": active.get("I_mask"), "Delta_clean(candidate)": clean.get("mean_delta"), "Delta_svd(candidate)": svd.get("mean_delta")})
    main_md = "# MAIN results\n\nThis is a frozen six-sentinel mechanism panel; it is not a holdout/generalization benchmark.\n\n" + _md_table(result_rows, [(k, k) for k in result_rows[0]]) + "\n\n## Paired estimands (seed-level paired deltas)\n\n" + _md_table(comparison_rows, [(k, k) for k in comparison_rows[0]]) + "\n"
    (report_root / "MAIN_RESULTS.md").write_text(main_md, encoding="utf-8")
    audit = {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "g0": evaluation["g0"], "external_review_status": "pending_or_recorded_in_review_stage", "gpu_legal_pool": list(protocol.LEGAL_GPU_POOL), "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS), "labels_after_fit_only": True, "main_rows": len(bundle["rows"]), "svd_rows": len(bundle["svd_rows"]), "decision": evaluation["decision"]}
    (report_root / "INTEGRITY_AUDIT.md").write_text("# Integrity audit\n\n```json\n" + json.dumps(audit, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    decision = {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "primary_terminal_decision": evaluation["decision"], "gates": evaluation, "claim_boundary": "six predeclared sentinel datasets under the frozen raw zero-preserving small-AE protocol; no holdout or universal claim", "forbidden_output": "BUILD_NEW_MODEL"}
    (report_root / "DECISION.md").write_text("# Decision\n\nPrimary terminal decision: `" + evaluation["decision"] + "`.\n\n```json\n" + json.dumps(decision, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    (final_root / "DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (final_root / "MAIN_SUMMARY.json").write_text(json.dumps({"result_rows": result_rows, "comparison_rows": comparison_rows, "decision": evaluation["decision"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, default=protocol.MAIN_ROOT)
    parser.add_argument("--report-root", type=Path, default=protocol.REPORT_ROOT)
    parser.add_argument("--final-root", type=Path, default=protocol.FINAL_ROOT)
    args = parser.parse_args()
    bundle = collect(args.main_root)
    evaluation = evaluate(bundle)
    write_outputs(bundle, evaluation, args.report_root, args.final_root)
    print(json.dumps({"status": evaluation["status"], "decision": evaluation["decision"], "g0": evaluation["g0"]["passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
