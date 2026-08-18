"""Independent compact audit for the completed C2 static matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import c2_matrix, protocol


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=c2_matrix._json_default) + "\n", encoding="utf-8")


def _compact_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_hashes.json" or "_attempts" in path.parts:
            continue
        if path.suffix in {".npy", ".npz", ".log"} or "score_artifacts" in path.parts:
            continue
        files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def run(root: Path) -> dict[str, Any]:
    protocol.validate_c2_authorization()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    expected_jobs = len(protocol.DEVELOPMENT_PANEL) * len(protocol.PRINCIPLES) * len(protocol.PRIMARY_SEEDS)
    job_rows: list[dict[str, Any]] = []
    forbidden_files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or "score_artifacts" in path.parts or "_attempts" in path.parts:
            continue
        if path.name in {"embedding.npy", "predictions.npy", "labels_true.npy"} or path.suffix in {".pt", ".pth", ".ckpt", ".npz"}:
            forbidden_files.append(str(path.relative_to(root)))
    for dataset in protocol.DEVELOPMENT_PANEL:
        _, source = c2_matrix._load_h0(dataset)
        _, label_source = c2_matrix._load_labels(dataset)
        for principle in protocol.PRINCIPLES:
            for seed in protocol.PRIMARY_SEEDS:
                run_dir = root / dataset / principle / f"seed{seed}"
                summary_path = run_dir / "summary.json"
                audit_path = run_dir / "audit.json"
                config_path = run_dir / "resolved_config.json"
                metrics_path = run_dir / "training_metrics.csv"
                row: dict[str, Any] = {"dataset": dataset, "principle": principle, "seed": int(seed), "status": "missing"}
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    with metrics_path.open(newline="", encoding="utf-8") as handle:
                        history = list(csv.DictReader(handle))
                except (OSError, json.JSONDecodeError):
                    job_rows.append(row)
                    continue
                metrics = summary.get("metrics", {})
                corruption = summary.get("corruption_audit", {})
                recorded_source = summary.get("source", {})
                gpu = int(summary.get("physical_gpu", -1))
                row.update({
                    "status": summary.get("status"),
                    "audit_ok": audit.get("audit_ok"),
                    "history_rows": len(history),
                    "gpu": gpu,
                    "finite_metrics": bool(metrics) and bool(np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all()),
                    "source_hash_match": recorded_source.get("H0_sha256") == source.get("H0_sha256") and recorded_source.get("budget_manifest_sha256") == source.get("budget_manifest_sha256") and recorded_source.get("labels_sha256") == label_source.get("labels_sha256"),
                    "identity_match": summary.get("dataset") == dataset and summary.get("principle") == principle and int(summary.get("seed", -1)) == int(seed) and config.get("dataset") == dataset and config.get("principle") == principle and int(config.get("seed", -1)) == int(seed),
                    "label_firewall": summary.get("labels_used_during_fit") is False and audit.get("labels_used_during_fit") is False and summary.get("labels_used_for_outer_metrics") is True,
                    "exact_budget": corruption.get("exact_budget_all_epochs") is True and audit.get("exact_budget_all_epochs") is True,
                    "history_finite": len(history) == int(protocol.BACKBONE_CONTRACT["epochs"]),
                    "raw_arrays_persisted": summary.get("raw_arrays_persisted") is False and audit.get("raw_artifacts_persisted") is False,
                    "gpu_legal": gpu in protocol.LEGAL_GPU_POOL and gpu not in protocol.FORBIDDEN_GPU_IDS,
                })
                job_rows.append(row)

    checks["all_expected_jobs_present"] = len(job_rows) == expected_jobs
    checks["all_jobs_completed_valid"] = all(row.get("status") == "completed_valid" for row in job_rows)
    checks["all_run_audits_ok"] = all(row.get("audit_ok") is True for row in job_rows)
    checks["all_identity_matches"] = all(row.get("identity_match") is True for row in job_rows)
    checks["all_source_hashes_match"] = all(row.get("source_hash_match") is True for row in job_rows)
    checks["label_firewall_all_jobs"] = all(row.get("label_firewall") is True for row in job_rows)
    checks["exact_budget_all_jobs"] = all(row.get("exact_budget") is True for row in job_rows)
    checks["history_contract_all_jobs"] = all(row.get("history_rows") == int(protocol.BACKBONE_CONTRACT["epochs"]) and row.get("history_finite") is True for row in job_rows)
    checks["gpu_allowlist_all_jobs"] = all(row.get("gpu_legal") is True for row in job_rows)
    checks["no_raw_performance_arrays"] = not forbidden_files
    checks["root_audit_ok"] = json.loads((root / "audit.json").read_text(encoding="utf-8")).get("audit_ok") is True
    checks["positive_control_passed"] = json.loads((root / "positive_control.json").read_text(encoding="utf-8")).get("status") == "completed_valid"

    with (root / "c2_dataset_summary.csv").open(newline="", encoding="utf-8") as handle:
        dataset_rows = list(csv.DictReader(handle))
    checks["dataset_summary_full_paired"] = len(dataset_rows) == len(protocol.DEVELOPMENT_PANEL) * len(protocol.PRINCIPLES) and all(row.get("status") == "completed_valid" and row.get("seed_count") == "3" and row.get("paired_seed_count") == "3" for row in dataset_rows)
    decision = json.loads((root / "decision.json").read_text(encoding="utf-8"))
    checks["decision_matches_complete_matrix"] = decision.get("completed_valid_runs") == expected_jobs and decision.get("status") != "incomplete_compute"
    checks["adaptive_and_c3_locked"] = decision.get("adaptive_policy_unlocked") is False and decision.get("c3_holdout_runs_unlocked") is False
    checks["support_firewall_present"] = protocol.resolved_config()["support_interpretation_firewall"] in (root / "C2_RESULTS.md").read_text(encoding="utf-8")
    checks["score_caveat_present"] = "P4 uses standardized clean H0 residuals" in (root / "C2_RESULTS.md").read_text(encoding="utf-8")

    audit = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.C2_PROTOCOL_ID,
        "stage": "C2_static_matrix_independent_audit",
        "audit_ok": all(checks.values()),
        "checks": checks,
        "details": {
            "expected_jobs": expected_jobs,
            "job_rows": job_rows,
            "forbidden_files": forbidden_files,
            "gpu_ids_observed": sorted({row.get("gpu") for row in job_rows}),
            "support_interpretation_firewall": protocol.resolved_config()["support_interpretation_firewall"],
        },
        "scientific_performance_claim": False,
        "raw_arrays_published": False,
    }
    _write_json(root / "C2_INTEGRITY_AUDIT.json", audit)
    lines = [
        "# C2 Independent Integrity Audit",
        "",
        f"Audit: `{'PASS' if audit['audit_ok'] else 'FAIL'}`; checks: `{sum(checks.values())}/{len(checks)}`.",
        "",
        "This is an execution-integrity audit, not an independent performance claim.",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend([
        "",
        "> Support in C2 denotes the frozen threshold-defined support of dense H0, not raw-X zero/nonzero support; raw sparse-support claims require a separate validation.",
        "",
        "Raw score arrays, H0, labels, embeddings, predictions, checkpoints and logs remain local.",
        "",
    ])
    (root / "C2_INTEGRITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(root / "artifact_hashes.json", {
        "stage": "C2_static_matrix",
        "files": _compact_files(root),
        "raw_local_exclusions": ["score_artifacts/**/*.npy", "_attempts/**"],
        "exact_tree_policy": "compact_non_array_files_only",
    })
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=protocol.RESULT_ROOT / "C2_static_matrix")
    args = parser.parse_args()
    result = run(args.result_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=c2_matrix._json_default))
    raise SystemExit(0 if result["audit_ok"] else 1)


if __name__ == "__main__":
    main()
