#!/usr/bin/env python3
"""Summarize completed ACCG panels with datasets as the population unit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap(values: np.ndarray, seed: int = 20260816, replicates: int = 5000) -> tuple[float, float]:
    if not values.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = np.mean(values[rng.integers(0, values.size, size=values.size)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _audit_job(job: dict[str, Any], output: Path) -> list[str]:
    reasons: list[str] = []
    has_outer_labels = job.get("record", {}).get("labels_present", True) is not False
    arms = ("N", "R", "T_s", "T_c") if job["role"] == "main" else ("T_c",)
    required = [
        output / "summary.json",
        output / "runner_profile.json",
        output / "resolved_config.json",
    ]
    for arm in arms:
        required.extend(
            (
                output / arm / "metrics.json",
                output / arm / "predictions.npy",
                output / arm / "structural_audit.json",
            )
        )
    if job["role"] == "main":
        required.extend((output / "audit.json", output / "branchpoint.pt"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return [f"missing_artifacts:{len(missing)}"]
    try:
        summary = _read(output / "summary.json")
        runner = _read(output / "runner_profile.json")
        resolved = _read(output / "resolved_config.json")
        protocol_audit = _read(output / "audit.json") if job["role"] == "main" else None
    except (OSError, json.JSONDecodeError):
        return ["invalid_json"]
    if summary.get("status") != "completed":
        reasons.append("summary_not_completed")
    if int(summary.get("seed", -1)) != int(job["seed"]):
        reasons.append("seed_mismatch")
    if runner.get("dataset") != job["record"]["name"]:
        reasons.append("dataset_mismatch")
    if runner.get("dataset_sha256") != job["record"]["source_sha256"]:
        reasons.append("source_hash_mismatch")
    if runner.get("config_sha256") != job.get("config_sha256"):
        reasons.append("config_hash_mismatch")
    if runner.get("labels_used_during_fit") is not False:
        reasons.append("label_isolation_failed")
    if bool(runner.get("branchpoint_reused")) != (job["role"] == "ablation"):
        reasons.append("branchpoint_reuse_mismatch")
    if job["role"] == "main":
        if resolved.get("variant") != "accg_joint" or summary.get("variant") != "accg_joint":
            reasons.append("main_variant_mismatch")
        assert protocol_audit is not None
        matching = protocol_audit.get("matched_schedule", {})
        if not matching or any(not all(bool(value) for value in row.values()) for row in matching.values()):
            reasons.append("matched_schedule_failed")
    elif summary.get("reused_from") != str(Path(job["reused_from"]).resolve()):
        reasons.append("canonical_control_source_mismatch")
    labels_path = output / "labels_true.npy"
    if not has_outer_labels:
        return reasons
    if not labels_path.is_file():
        reasons.append("missing_outer_labels_for_confirmatory_metrics")
        return reasons
    try:
        labels = np.load(labels_path, allow_pickle=False)
    except (OSError, ValueError):
        reasons.append("invalid_outer_labels")
        return reasons
    for arm in arms:
        try:
            predictions = np.load(output / arm / "predictions.npy", allow_pickle=False)
            metrics = _read(output / arm / "metrics.json")
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append(f"{arm}:invalid_metric_artifact")
            continue
        if predictions.shape[0] != labels.shape[0]:
            reasons.append(f"{arm}:prediction_length_mismatch")
            continue
        recomputed = {
            "ari": float(adjusted_rand_score(labels, predictions)),
            "nmi": float(normalized_mutual_info_score(labels, predictions)),
        }
        for name, value in recomputed.items():
            stored = metrics.get(name)
            if stored is None or not np.isclose(float(stored), value, atol=1e-12, rtol=0.0):
                reasons.append(f"{arm}:{name}_mismatch")
    return reasons


def summarize(manifest_path: Path) -> dict[str, Any]:
    manifest = _read(manifest_path)
    rows = []
    incomplete = []
    for job in manifest["jobs"]:
        output = Path(job["output_dir"])
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            incomplete.append({"run_key": job["run_key"], "status": "missing"})
            continue
        summary = _read(summary_path)
        if summary.get("status") != "completed":
            incomplete.append({"run_key": job["run_key"], "status": summary.get("status")})
            continue
        audit_reasons = _audit_job(job, output)
        if audit_reasons:
            incomplete.append({"run_key": job["run_key"], "status": "audit_failed", "reasons": audit_reasons})
            continue
        if job["role"] == "main":
            metrics = {arm: _read(output / arm / "metrics.json") for arm in ("N", "R", "T_s", "T_c")}
            structural = {arm: _read(output / arm / "structural_audit.json") for arm in ("R", "T_s", "T_c")}
            row = {
                "dataset_id": job["dataset_id"],
                "domain": job["record"]["domain"],
                "source_family": job["record"]["source_family"],
                "seed": int(job["seed"]),
                "role": "main" if job["record"].get("labels_present", True) is not False else "operational",
                "k_source": job["record"].get("K_source"),
                "ari_N": metrics["N"].get("ari"),
                "ari_R": metrics["R"].get("ari"),
                "ari_Ts": metrics["T_s"].get("ari"),
                "ari_Tc": metrics["T_c"].get("ari"),
                "nmi_Tc": metrics["T_c"].get("nmi"),
                "joint_delta_Ts": structural["T_s"]["joint_delta_mean"],
                "joint_delta_Tc": structural["T_c"]["joint_delta_mean"],
                "constraint_violation_Tc": structural["T_c"]["constraint_violation_rate"],
                "constraint_infeasible_Tc": structural["T_c"]["constraint_infeasible_rate"],
                "budget_fill_Tc": structural["T_c"]["budget_fill"],
            }
            rows.append(row)
        else:
            metrics = _read(output / "T_c/metrics.json")
            structural = _read(output / "T_c/structural_audit.json")
            rows.append(
                {
                    "dataset_id": job["dataset_id"],
                    "domain": job["record"]["domain"],
                    "source_family": job["record"]["source_family"],
                    "seed": int(job["seed"]),
                    "role": "ablation",
                    "ablation": job["ablation"],
                    "ari_Tc": metrics.get("ari"),
                    "nmi_Tc": metrics.get("nmi"),
                    "joint_delta_Tc": structural["joint_delta_mean"],
                    "constraint_violation_Tc": structural["constraint_violation_rate"],
                    "constraint_infeasible_Tc": structural["constraint_infeasible_rate"],
                    "budget_fill_Tc": structural["budget_fill"],
                    "reused_from": job["reused_from"],
                }
            )
    main = [row for row in rows if row["role"] == "main" and row.get("ari_Tc") is not None]
    operational = [row for row in rows if row["role"] == "operational"]
    dataset_rows = []
    for dataset_id in sorted({row["dataset_id"] for row in main}):
        group = [row for row in main if row["dataset_id"] == dataset_id]
        deltas = np.asarray([row["ari_Tc"] - row["ari_Ts"] for row in group], dtype=np.float64)
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "domain": group[0]["domain"],
                "source_family": group[0]["source_family"],
                "seeds": len(group),
                "S_c_minus_s": float(np.mean(deltas)),
                "S_c_minus_s_std": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
                "all_seed_positive": bool(np.all(deltas > 0.0)),
                "all_seed_negative": bool(np.all(deltas < 0.0)),
                "ari_Tc_mean": float(np.mean([row["ari_Tc"] for row in group])),
                "constraint_violation_mean": float(np.mean([row["constraint_violation_Tc"] for row in group])),
                "constraint_infeasible_mean": float(np.mean([row["constraint_infeasible_Tc"] for row in group])),
                "budget_fill_mean": float(np.mean([row["budget_fill_Tc"] for row in group])),
            }
        )
    effects = np.asarray([row["S_c_minus_s"] for row in dataset_rows], dtype=np.float64)
    ci_low, ci_high = _bootstrap(effects)
    return {
        "manifest_id": manifest.get("manifest_id"),
        "status": "complete" if not incomplete else "incomplete_compute",
        "statistical_unit": "dataset",
        "seed_role": "repeated_measurement",
        "main_dataset_count": len(dataset_rows),
        "operational_dataset_count": len({row["dataset_id"] for row in operational}),
        "operational_run_rows": operational,
        "main_effect_mean": float(np.mean(effects)) if effects.size else None,
        "main_effect_median": float(np.median(effects)) if effects.size else None,
        "main_effect_bootstrap_ci": [ci_low, ci_high],
        "dataset_rows": dataset_rows,
        "run_rows": rows,
        "incomplete": incomplete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "datasets": payload["main_dataset_count"]}, indent=2))
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
