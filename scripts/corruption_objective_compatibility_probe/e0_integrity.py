"""E0 closure of the support matcher, without mutating the closed D1 project."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import corrected_matching as matching
from . import protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _toy_checks() -> dict[str, Any]:
    # The last four entries are genuinely below the frozen 5% H0 threshold;
    # using merely small-but-active values would make the toy unable to test a
    # support crossing at all.
    values = np.asarray([1.0, 0.5, -1.0, -0.5, 0.001, -0.001], dtype=np.float32)
    ranges_a = matching.row_constructive_ranges(values, row=0, seed=42, pair_count=1)
    ranges_b = matching.row_constructive_ranges(values, row=0, seed=42, pair_count=1)
    same = matching.build_common_dose_row(values, row=0, seed=42, pair_count=1, ranges=ranges_a)
    tie_row = np.asarray([1.0, 1.0, 1.0, 0.001, 0.001, 0.001], dtype=np.float32)
    tie_pairs = {
        tuple(matching.greedy_matching(
            tie_row[:3], tie_row[3:], 2, mode="min", seed=seed, row=0
        )[0].reshape(-1).tolist())
        for seed in (1, 2, 3, 4)
    }
    disjoint_pairs, _ = matching.greedy_matching(
        np.asarray([1.0, 1.0, 1.0]), np.asarray([0.1, 0.1, 0.1]), 2,
        mode="min", seed=42, row=0
    )
    endpoints = np.asarray(disjoint_pairs, dtype=np.int64)
    return {
        "reproducibility": bool(ranges_a == ranges_b),
        "tie_sensitivity": bool(len(tie_pairs) >= 2),
        "tie_matching_count": len(tie_pairs),
        "disjointness": bool(
            len(np.unique(endpoints[:, 0])) == endpoints.shape[0]
            and len(np.unique(endpoints[:, 1])) == endpoints.shape[0]
        ),
        "exact_support_crossing": bool(
            same["cross_audit"]["support_change_exact"]
            and same["cross_audit"]["support_change_count"] == 2
            and same["preserve_audit"]["support_change_exact"]
            and same["preserve_audit"]["support_change_count"] == 0
        ),
        "toy_common_match_ok": bool(same["match_ok"]),
    }


def _run_dataset_seed(dataset: str, seed: int) -> dict[str, Any]:
    h0_path = protocol.INPUT_ROOT / dataset / "H0.npy"
    clean = np.asarray(np.load(h0_path, mmap_mode="r"), dtype=np.float32)
    support = matching.support_mask(clean, reference=clean)
    _, pair_counts = matching.dataset_budget(clean)
    positive = common = matched = 0
    mismatches: list[float] = []
    cross_dose = preserve_dose = 0.0
    exact_fail = support_fail = preserve_fail = multiset_fail = range_fail = 0
    for row_idx, pair_count_raw in enumerate(pair_counts):
        pair_count = int(pair_count_raw)
        if pair_count <= 0:
            continue
        positive += 1
        ranges = matching.row_constructive_ranges(clean[row_idx], row=row_idx, seed=seed, pair_count=pair_count)
        if not ranges.get("common_exists", False):
            range_fail += 1
            continue
        common += 1
        result = matching.build_common_dose_row(
            clean[row_idx], row=row_idx, seed=seed, pair_count=pair_count, ranges=ranges
        )
        if not result.get("match_ok", False):
            continue
        matched += 1
        mismatches.append(float(result["row_relative_mismatch"]))
        cross_dose += float(result["cross_dose"])
        preserve_dose += float(result["preserve_dose"])
        ca = result["cross_audit"]
        pa = result["preserve_audit"]
        exact_fail += int(not (ca["exact_changed_count"] and pa["exact_changed_count"]))
        support_fail += int(not ca["support_change_exact"])
        preserve_fail += int(not pa["support_change_exact"])
        multiset_fail += int(not (ca["row_value_multiset_ok"] and pa["row_value_multiset_ok"]))
    total_mismatch = abs(cross_dose - preserve_dose) / max(cross_dose, protocol.DOSE_EPS)
    median_mismatch = float(np.median(mismatches)) if mismatches else float("inf")
    common_fraction = common / max(positive, 1)
    gate = {
        "positive_budget_rows": positive,
        "common_interval_rows": common,
        "common_interval_fraction": common_fraction,
        "all_common_rows_constructed": matched == common,
        "dataset_total_relative_mismatch_ok": total_mismatch <= 0.05,
        "median_row_relative_mismatch_ok": median_mismatch <= 0.10,
        "exact_changed_count_ok": exact_fail == 0,
        "cross_support_change_exact_ok": support_fail == 0,
        "preserve_support_change_exact_zero_ok": preserve_fail == 0,
        "row_value_multiset_ok": multiset_fail == 0,
        "range_constructive_failure_count": range_fail,
    }
    gate["pass"] = bool(
        common_fraction >= 0.95
        and all(gate[key] for key in (
            "all_common_rows_constructed",
            "dataset_total_relative_mismatch_ok",
            "median_row_relative_mismatch_ok",
            "exact_changed_count_ok",
            "cross_support_change_exact_ok",
            "preserve_support_change_exact_zero_ok",
            "row_value_multiset_ok",
        ))
    )
    return {
        "dataset": dataset,
        "seed": int(seed),
        "H0_sha256": _sha256(h0_path),
        "labels_not_loaded": True,
        "gpu_runs_started": 0,
        "cross_total_dose": cross_dose,
        "preserve_total_dose": preserve_dose,
        "dataset_total_relative_mismatch": total_mismatch,
        "median_row_relative_mismatch": median_mismatch,
        "range_kind": "constructive_min_max_witnesses_corrected_edge_hash",
        "failure_counts": {
            "no_common_interval": positive - common,
            "range_constructive_failure": range_fail,
            "target_match_failure": common - matched,
            "exact_budget": exact_fail,
            "cross_support": support_fail,
            "preserve_support": preserve_fail,
            "row_value_multiset": multiset_fail,
        },
        "gate": gate,
    }


def run(output_dir: Path = protocol.RESULT_ROOT / "E0_integrity") -> dict[str, Any]:
    protocol.validate_contract()
    output_dir.mkdir(parents=True, exist_ok=True)
    old_audit = protocol.SUPPORT_D1_ROOT / "audit.json"
    old_decision = protocol.SUPPORT_D1_ROOT / "decision.json"
    if not old_audit.exists() or not old_decision.exists():
        raise FileNotFoundError("closed support D1 compact artifacts are required")
    old_audit_data = _json(old_audit)
    old_decision_data = _json(old_decision)
    baseline_checks = {
        "closed_d1_audit_ok": old_audit_data.get("audit_ok") is True,
        "closed_d1_gate_failed": old_audit_data.get("d1_gate_pass") is False,
        "closed_d2_not_authorized": old_decision_data.get("d2_authorized") is False,
        "closed_d2_runs_zero": old_decision_data.get("d2_gpu_runs_started") == 0,
    }
    toy = _toy_checks()
    summaries = [_run_dataset_seed(dataset, seed) for dataset in protocol.BIOLOGICAL_DATASETS for seed in protocol.PRIMARY_SEEDS]
    all_pass = all(row["gate"]["pass"] for row in summaries)
    audit = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "E0_integrity",
        "audit_ok": bool(all(baseline_checks.values()) and all(toy.values()) and len(summaries) == 9),
        "baseline_checks": baseline_checks,
        "baseline_artifacts": {
            "audit_path": "support_crossing_common_dose_probe/D1_feasibility/audit.json",
            "audit_sha256": _sha256(old_audit),
            "decision_path": "support_crossing_common_dose_probe/D1_feasibility/decision.json",
            "decision_sha256": _sha256(old_decision),
            "mutated": False,
        },
        "toy_checks": toy,
        "dataset_seed_count": len(summaries),
        "corrected_d1_gate_pass": all_pass,
        "support_specific_attribution": "frozen_no_d2_authorization",
        "d2_authorized": False,
        "gpu_runs_started": 0,
        "labels_not_loaded": True,
        "summaries": summaries,
        "claim_boundary": "Corrected constructive matching closure does not establish a causal support effect or universal feasibility.",
    }
    decision = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "support_specific_attribution_frozen",
        "audit_ok": audit["audit_ok"],
        "corrected_d1_gate_pass": all_pass,
        "support_specific_attribution": "frozen_no_d2_authorization",
        "d2_authorized": False,
        "gpu_runs_started": 0,
        "interpretation": "The closed support line remains read-only; corrected matcher code is confined to this project.",
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(protocol.resolved_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"audit": audit, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=protocol.RESULT_ROOT / "E0_integrity")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["decision"], sort_keys=True))
    return 0 if result["audit"]["audit_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
