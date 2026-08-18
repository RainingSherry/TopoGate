"""M1 magnitude-matched support-preserving control matrix.

Only the nine new ``P2_MM_SupportPreserve`` jobs are trained.  C2 P2 metrics
are read-only evidence and are never rerun.  Action construction is label-free
and deterministic; all structural checks are required before a run is counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import protocol
from .frozen_adapter import (
    SmallMAE,
    adapter_manifest,
    clustering_acc,
    device_or_fail,
    embedding_diagnostics,
    load_h0,
    load_labels,
    seed_everything,
    standardize,
)
from .m0_freeze import _c2_summary, _json, _write_json, sha256_file
from .replay import build_magnitude_matched_epoch, compact_epoch_audit, replay_p2_epoch


def _cuda_visible_is_legal() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        ids = {int(item.strip()) for item in visible.split(",") if item.strip()}
    except ValueError:
        return False
    return len(ids) == 1 and ids.issubset(set(protocol.LEGAL_GPU_POOL)) and ids.isdisjoint(set(protocol.FORBIDDEN_GPU_IDS))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in fields} for row in materialized)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    seed_everything(int(seed))


def _existing_valid(run_dir: Path, dataset: str, seed: int, freeze_manifest: dict[str, Any]) -> bool:
    summary_path = run_dir / "summary.json"
    audit_path = run_dir / "audit.json"
    if not summary_path.exists() or not audit_path.exists():
        return False
    try:
        summary = _json(summary_path)
        audit = _json(audit_path)
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("status") != "completed_valid" or audit.get("audit_ok") is not True:
        return False
    if summary.get("protocol_id") != protocol.M1_PROTOCOL_ID or summary.get("dataset") != dataset or int(summary.get("seed", -1)) != int(seed):
        return False
    expected = next((row for row in freeze_manifest["c2_p2_records"] if row["dataset"] == dataset and int(row["seed"]) == int(seed)), None)
    source = summary.get("source", {})
    if expected is None or source.get("H0_sha256") != expected["H0_sha256"] or source.get("budget_manifest_sha256") != expected["budget_manifest_sha256"]:
        return False
    return True


def _metrics(embedding: np.ndarray, reconstruction: np.ndarray, target: np.ndarray, labels: np.ndarray, seed: int) -> tuple[dict[str, float], np.ndarray]:
    k = int(np.unique(labels).size)
    predictions = KMeans(n_clusters=k, n_init=20, random_state=int(seed)).fit_predict(embedding)
    metrics = {
        "ARI": float(adjusted_rand_score(labels, predictions)),
        "NMI": float(normalized_mutual_info_score(labels, predictions)),
        "ACC": clustering_acc(labels, predictions),
        "L_rec": float(np.mean((reconstruction - target) ** 2)),
        **embedding_diagnostics(embedding),
    }
    return metrics, predictions


def run_job(dataset: str, seed: int, output_dir: Path) -> dict[str, Any]:
    protocol.validate_contract()
    _require_preflight_authorization()
    if dataset not in protocol.DEVELOPMENT_PANEL or seed not in protocol.PRIMARY_SEEDS:
        raise ValueError("dataset/seed outside frozen M1 matrix")
    if not _cuda_visible_is_legal():
        raise RuntimeError("M1 formal jobs require exactly one legal physical GPU in CUDA_VISIBLE_DEVICES")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    device, physical_gpu = device_or_fail()
    h0_raw, source = load_h0(dataset)
    h0_scaled, mean, std = standardize(h0_raw)
    p2_path, p2_summary = _c2_summary(dataset, seed)
    if p2_summary.get("status") != "completed_valid":
        raise ValueError(f"C2 P2 evidence is not valid: {p2_path}")

    model = SmallMAE(device, h0_raw.shape[1])
    action_rng = np.random.default_rng(seed)
    epoch_rows: list[dict[str, Any]] = []
    total_p2_l1 = 0.0
    total_mm_l1 = 0.0
    row_relative_mismatch: list[float] = []
    exact_budget = True
    exact_support_preserve = True
    source_set_match = True
    multiset_mismatch = 0
    match_failures = 0
    for epoch in range(protocol.EPOCHS):
        p2_raw, p2_audit = replay_p2_epoch(h0_raw, action_rng)
        mm_raw, mm_audit = build_magnitude_matched_epoch(h0_raw, p2_raw, p2_audit)
        exact_budget = exact_budget and bool(mm_audit["exact_budget"])
        exact_support_preserve = exact_support_preserve and bool(mm_audit["support_change_rate"] == 0.0)
        source_set_match = source_set_match and bool(np.array_equal(p2_audit["source_mask"], mm_audit["source_mask"]))
        multiset_mismatch += int(mm_audit["row_value_multiset_mismatch_count"])
        match_failures += int(mm_audit["match_failure_count"])
        total_p2_l1 += float(mm_audit["p2_total_absolute_change"])
        total_mm_l1 += float(mm_audit["total_absolute_change"])
        row_relative_mismatch.extend(float(record["relative_mismatch"]) for record in mm_audit["row_records"])
        mm_scaled = ((mm_raw - mean) / std).astype(np.float32)
        train_loss = model.fit_epoch(mm_scaled, h0_scaled, action_rng)
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss_before_step": float(train_loss),
                "exact_budget": bool(mm_audit["exact_budget"]),
                "support_change_rate": float(mm_audit["support_change_rate"]),
                "p2_total_absolute_change": float(mm_audit["p2_total_absolute_change"]),
                "mm_total_absolute_change": float(mm_audit["total_absolute_change"]),
                "dataset_total_relative_mismatch": float(mm_audit["dataset_total_relative_mismatch"]),
                "median_row_relative_mismatch": float(mm_audit["median_row_relative_mismatch"]),
                "row_relative_mismatch_max": float(mm_audit["row_relative_mismatch_max"]),
                "row_value_multiset_mismatch_count": int(mm_audit["row_value_multiset_mismatch_count"]),
                "labels_used": False,
            }
        )

    embedding, reconstruction = model.predict(h0_scaled)
    # Labels are introduced only after all fit steps and structural audits.
    labels, label_source = load_labels(dataset)
    if labels.size != h0_raw.shape[0]:
        raise ValueError(f"label/H0 mismatch for {dataset}: {labels.size} != {h0_raw.shape[0]}")
    metrics, _predictions = _metrics(embedding, reconstruction, h0_scaled, labels, seed)
    dataset_total_relative = abs(total_mm_l1 - total_p2_l1) / max(abs(total_p2_l1), protocol.ROW_REL_EPS)
    median_row_relative = float(np.median(np.asarray(row_relative_mismatch, dtype=np.float64))) if row_relative_mismatch else 0.0
    match_estimable = bool(
        dataset_total_relative <= protocol.TOTAL_L1_REL_TOLERANCE
        and median_row_relative <= protocol.MEDIAN_ROW_REL_TOLERANCE
        and match_failures == 0
    )
    structural_ok = bool(exact_budget and exact_support_preserve and source_set_match and multiset_mismatch == 0 and np.isfinite(embedding).all())
    if not structural_ok:
        status = "protocol_mismatch"
    elif not match_estimable:
        status = "magnitude_match_not_estimable"
    else:
        status = "completed_valid"
    audit = {
        "audit_ok": status == "completed_valid",
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.M1_PROTOCOL_ID,
        "stage": "M1_h0_support_crossing_isolation",
        "dataset": dataset,
        "seed": int(seed),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "cuda_visible_is_legal": _cuda_visible_is_legal(),
        "physical_gpu": physical_gpu,
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "exact_changed_coordinate_budget": exact_budget,
        "support_change_rate_exact_zero": exact_support_preserve,
        "same_p2_source_set": source_set_match,
        "row_value_multiset_mismatch_count": multiset_mismatch,
        "match_failure_count": match_failures,
        "dataset_total_relative_mismatch": dataset_total_relative,
        "median_row_relative_mismatch": median_row_relative,
        "magnitude_match_estimable": match_estimable,
        "embedding_finite": bool(np.isfinite(embedding).all()),
        "raw_arrays_persisted": False,
    }
    summary = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.M1_PROTOCOL_ID,
        "stage": "M1_h0_support_crossing_isolation",
        "dataset": dataset,
        "seed": int(seed),
        "control": protocol.M1_CONTROL,
        "status": status,
        "device": str(device),
        "physical_gpu": physical_gpu,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "K": int(np.unique(labels).size),
        "K_source": "benchmark_oracle_from_y_outer_readout_only",
        "metrics": metrics,
        "matching": {
            "dataset_total_relative_mismatch": dataset_total_relative,
            "median_row_relative_mismatch": median_row_relative,
            "total_l1_relative_tolerance": protocol.TOTAL_L1_REL_TOLERANCE,
            "median_row_relative_tolerance": protocol.MEDIAN_ROW_REL_TOLERANCE,
            "estimable": match_estimable,
        },
        "structural_audit": {
            "exact_changed_coordinate_budget": exact_budget,
            "support_change_rate": 0.0 if exact_support_preserve else float(np.mean([row["support_change_rate"] for row in epoch_rows])),
            "same_p2_source_set": source_set_match,
            "row_value_multiset_mismatch_count": multiset_mismatch,
            "epochs_audited": len(epoch_rows),
        },
        "source": {
            **source,
            **label_source,
            "p2_summary_path": str(p2_path.resolve()),
            "p2_summary_sha256": sha256_file(p2_path),
            "mean_std_fit_on_clean_H0_only": True,
        },
        "p2_reused_from": str(p2_path.resolve()),
        "p2_reused_ARI": float(p2_summary["metrics"]["ARI"]),
        "backbone": {
            "source": "frozen_C2_small_matched_reconstruction_probe",
            "epochs": protocol.EPOCHS,
            "control_only_new_training": True,
        },
        "support_interpretation_firewall": protocol.resolved_config()["support_interpretation_firewall"],
        "raw_arrays_persisted": False,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "audit.json", audit)
    _write_json(
        output_dir / "resolved_config.json",
        {
            **protocol.resolved_config(),
            "stage": "M1_h0_support_crossing_isolation",
            "dataset": dataset,
            "seed": int(seed),
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "frozen_c2_adapter": adapter_manifest(),
            "p2_reused_from": str(p2_path.resolve()),
        },
    )
    _write_csv(output_dir / "training_metrics.csv", epoch_rows)
    return summary


def _run_one_subprocess(dataset: str, seed: int, root: Path, gpu_id: int) -> dict[str, Any]:
    run_dir = root / dataset / protocol.M1_CONTROL / f"seed{seed}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "4"
    command = [
        sys.executable,
        "-m",
        "scripts.support_target_validation_probe.m1_matrix",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
    ]
    completed = subprocess.run(command, cwd=str(protocol.PROJECT_ROOT), env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"M1 job failed ({dataset}, seed={seed}, gpu={gpu_id}): {completed.stderr[-2000:]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"M1 job emitted invalid JSON ({dataset}, seed={seed}): {completed.stdout[-1000:]}") from exc


def _read_freeze(root: Path) -> dict[str, Any]:
    path = root / "M0_freeze" / "freeze_manifest.json"
    audit_path = root / "M0_freeze" / "audit.json"
    if not path.exists() or not audit_path.exists():
        raise FileNotFoundError("run M0 freeze before M1")
    manifest = _json(path)
    audit = _json(audit_path)
    if audit.get("audit_ok") is not True or manifest.get("m1_authorized") is not True:
        raise RuntimeError("M0 did not authorize M1")
    return manifest


def _require_preflight_authorization() -> dict[str, Any]:
    path = protocol.RESULT_ROOT / "M1_preflight" / "decision.json"
    if not path.exists():
        raise RuntimeError("run the full no-training M1 preflight before any GPU job")
    decision = _json(path)
    if decision.get("formal_m1_gpu_runs_authorized") is not True or decision.get("status") != "magnitude_match_estimable":
        raise RuntimeError("M1 GPU matrix is blocked by magnitude_match_not_estimable preflight")
    return decision


def aggregate(root: Path, freeze_manifest: dict[str, Any]) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    for dataset in protocol.DEVELOPMENT_PANEL:
        for seed in protocol.PRIMARY_SEEDS:
            path = root / dataset / protocol.M1_CONTROL / f"seed{seed}" / "summary.json"
            if path.exists():
                try:
                    summary = _json(path)
                except json.JSONDecodeError:
                    summary = {}
            else:
                summary = {}
            if summary.get("status") == "completed_valid":
                run_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "status": "completed_valid",
                        "MM_ARI": float(summary["metrics"]["ARI"]),
                        "MM_NMI": float(summary["metrics"]["NMI"]),
                        "MM_ACC": float(summary["metrics"]["ACC"]),
                        "dataset_total_relative_mismatch": float(summary["matching"]["dataset_total_relative_mismatch"]),
                        "median_row_relative_mismatch": float(summary["matching"]["median_row_relative_mismatch"]),
                    }
                )
            elif summary.get("status") == "magnitude_match_not_estimable":
                run_rows.append({"dataset": dataset, "seed": int(seed), "status": "magnitude_match_not_estimable"})
            else:
                run_rows.append({"dataset": dataset, "seed": int(seed), "status": "incomplete_compute"})

    dataset_rows: list[dict[str, Any]] = []
    for dataset in protocol.DEVELOPMENT_PANEL:
        selected = [row for row in run_rows if row["dataset"] == dataset and row["status"] == "completed_valid"]
        p2_rows = [
            _json(_c2_summary(dataset, seed)[0])
            for seed in protocol.PRIMARY_SEEDS
        ]
        nonvalid = [row for row in run_rows if row["dataset"] == dataset and row["status"] != "completed_valid"]
        if len(selected) != len(protocol.PRIMARY_SEEDS):
            dataset_rows.append({"dataset": dataset, "status": "magnitude_match_not_estimable" if any(row["status"] == "magnitude_match_not_estimable" for row in nonvalid) else "incomplete_compute", "seed_count": len(selected)})
            continue
        selected_by_seed = {int(row["seed"]): row for row in selected}
        deltas = [float(p2_rows[index]["metrics"]["ARI"]) - float(selected_by_seed[seed]["MM_ARI"]) for index, seed in enumerate(protocol.PRIMARY_SEEDS)]
        delta_mean = float(np.mean(deltas))
        dataset_rows.append(
            {
                "dataset": dataset,
                "status": "completed_valid",
                "seed_count": len(selected),
                "P2_ARI_reused_mean": float(np.mean([float(row["metrics"]["ARI"]) for row in p2_rows])),
                "MM_ARI_mean": float(np.mean([selected_by_seed[seed]["MM_ARI"] for seed in protocol.PRIMARY_SEEDS])),
                "Delta_cross_mean": delta_mean,
                "Delta_cross_seed_values": deltas,
                "positive_seed_count": int(sum(delta > 0.0 for delta in deltas)),
                "material": bool(delta_mean >= protocol.MATERIAL_DELTA_ARI),
                "strong_negative": bool(delta_mean <= -protocol.MATERIAL_DELTA_ARI),
                "dataset_total_relative_mismatch_mean": float(np.mean([selected_by_seed[seed]["dataset_total_relative_mismatch"] for seed in protocol.PRIMARY_SEEDS])),
                "median_row_relative_mismatch_mean": float(np.mean([selected_by_seed[seed]["median_row_relative_mismatch"] for seed in protocol.PRIMARY_SEEDS])),
            }
        )

    complete = len(run_rows) == 9 and all(row["status"] == "completed_valid" for row in run_rows)
    match_not_estimable = any(row["status"] == "magnitude_match_not_estimable" for row in run_rows)
    passing = [row for row in dataset_rows if row.get("status") == "completed_valid" and row.get("material") is True and int(row.get("positive_seed_count", 0)) >= 2]
    strong_negative = [row for row in dataset_rows if row.get("status") == "completed_valid" and row.get("strong_negative") is True]
    if match_not_estimable:
        status = "magnitude_match_not_estimable"
    elif not complete:
        status = "incomplete_compute"
    elif strong_negative:
        status = "support_crossing_not_isolated"
    elif len(passing) >= 2:
        status = "threshold_support_crossing_effect_supported"
    else:
        status = "support_crossing_not_isolated"
    decision = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.M1_PROTOCOL_ID,
        "stage": "M1_h0_support_crossing_isolation",
        "status": status,
        "expected_new_runs": 9,
        "completed_valid_new_runs": sum(row["status"] == "completed_valid" for row in run_rows),
        "dataset_rows": dataset_rows,
        "passing_dataset_count": len(passing),
        "strong_negative_dataset_count": len(strong_negative),
        "p2_recomputed": False,
        "p2_reused_only": True,
        "m2_authorized": False,
        "m3_authorized": False,
        "m4_authorized": False,
        "adaptive_locked": True,
        "gan_locked": True,
        "support_interpretation_firewall": protocol.resolved_config()["support_interpretation_firewall"],
        "interpretation_scope": "descriptive_matched_support_role_contrast; not strict causal isolation",
    }
    _write_csv(root / "m1_run_summary.csv", run_rows)
    _write_csv(root / "m1_dataset_summary.csv", dataset_rows)
    _write_json(root / "decision.json", decision)
    _write_json(root / "run_manifest.json", {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.M1_PROTOCOL_ID,
        "stage": "M1_h0_support_crossing_isolation",
        "expected_new_runs": 9,
        "completed_valid_new_runs": decision["completed_valid_new_runs"],
        "publication_scope": "compact summaries and audits only",
        "p2_reused_only": True,
    })
    audit = {
        "audit_ok": bool(status not in {"incomplete_compute", "magnitude_match_not_estimable"} and complete),
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.M1_PROTOCOL_ID,
        "stage": "M1_h0_support_crossing_isolation",
        "expected_new_run_count": 9,
        "completed_valid_new_run_count": decision["completed_valid_new_runs"],
        "all_jobs_completed_valid": complete,
        "p2_recomputed": False,
        "p2_reused_only": True,
        "m2_m3_m4_locked": True,
        "adaptive_locked": True,
        "gan_locked": True,
        "labels_used_during_fit": False,
        "gpu_pool": list(protocol.LEGAL_GPU_POOL),
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "raw_arrays_persisted": False,
        "support_interpretation_firewall": protocol.resolved_config()["support_interpretation_firewall"],
        "interpretation_scope": "descriptive_matched_support_role_contrast; not strict causal isolation",
    }
    _write_json(root / "audit.json", audit)
    lines = [
        "# M1 H0 Support-Crossing Isolation",
        "",
        f"Status: `{status}`; valid new runs: `{decision['completed_valid_new_runs']}/9`.",
        "",
        "Primary endpoint: `Delta_cross = ARI(P2_reused) - ARI(P2_MM_SupportPreserve)`.",
        "C2 P2 is reused read-only; it is not retrained or retuned.",
        "",
        "| Dataset | P2 ARI (reused) | MM ARI | Delta_cross | Positive seeds | Material |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_rows:
        if row.get("status") == "completed_valid":
            lines.append(f"| {row['dataset']} | {row['P2_ARI_reused_mean']:.6f} | {row['MM_ARI_mean']:.6f} | {row['Delta_cross_mean']:.6f} | {row['positive_seed_count']}/3 | {row['material']} |")
    lines.extend([
        "",
        "M1 is a descriptive matched support-role contrast, not a strict causal isolation: an active-active MM control may be easier for reconstruction, making Delta_cross conservative/downward-biased. M2 raw-X bridge, M3 holdout, M4 full-backbone transfer, adaptive policy and GAN remain locked until the frozen M1 gate is satisfied.",
        "",
        f"> {protocol.resolved_config()['support_interpretation_firewall']}",
    ])
    (root / "M1_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"decision": decision, "audit": audit, "dataset_rows": dataset_rows}


def run_matrix(root: Path = protocol.RESULT_ROOT / "M1_h0_isolation", *, gpu_pool: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> dict[str, Any]:
    protocol.validate_contract()
    if not gpu_pool or any(gpu not in protocol.LEGAL_GPU_POOL for gpu in gpu_pool) or set(gpu_pool) & set(protocol.FORBIDDEN_GPU_IDS):
        raise ValueError(f"illegal M1 GPU pool: {gpu_pool}")
    freeze_manifest = _read_freeze(protocol.RESULT_ROOT)
    _require_preflight_authorization()
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(dataset, seed) for dataset in protocol.DEVELOPMENT_PANEL for seed in protocol.PRIMARY_SEEDS]
    ledger: list[dict[str, Any]] = []
    to_run: list[tuple[str, int]] = []
    for dataset, seed in jobs:
        run_dir = root / dataset / protocol.M1_CONTROL / f"seed{seed}"
        if _existing_valid(run_dir, dataset, seed, freeze_manifest):
            ledger.append({"dataset": dataset, "seed": int(seed), "status": "reused"})
        else:
            ledger.append({"dataset": dataset, "seed": int(seed), "status": "queued"})
            to_run.append((dataset, seed))
    _write_json(root / "launch_manifest.json", {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.M1_PROTOCOL_ID,
        "stage": "M1_h0_support_crossing_isolation",
        "expected_new_runs": len(jobs),
        "new_jobs": len(to_run),
        "reused_jobs": len(jobs) - len(to_run),
        "gpu_pool": list(gpu_pool),
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "p2_recomputed": False,
        "jobs": ledger,
    })
    with ThreadPoolExecutor(max_workers=len(gpu_pool)) as executor:
        futures = {
            executor.submit(_run_one_subprocess, dataset, seed, root, gpu_pool[index % len(gpu_pool)]): (dataset, seed)
            for index, (dataset, seed) in enumerate(to_run)
        }
        for future in as_completed(futures):
            dataset, seed = futures[future]
            try:
                result = future.result()
                status = "completed" if result.get("status") == "completed_valid" else "incomplete"
                result_status = result.get("status")
            except Exception as exc:
                status = "incomplete"
                result_status = "incomplete_compute"
                result = {"dataset": dataset, "seed": seed, "status": result_status, "error": str(exc)}
            for row in ledger:
                if row["dataset"] == dataset and int(row["seed"]) == int(seed):
                    row.update({"status": status, "result_status": result_status})
                    break
            _write_json(root / "launch_manifest.json", {
                "project_id": protocol.PROJECT_ID,
                "protocol_id": protocol.M1_PROTOCOL_ID,
                "stage": "M1_h0_support_crossing_isolation",
                "expected_new_runs": len(jobs),
                "new_jobs": len(to_run),
                "reused_jobs": len(jobs) - len(to_run),
                "gpu_pool": list(gpu_pool),
                "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
                "p2_recomputed": False,
                "jobs": ledger,
            })
    return aggregate(root, freeze_manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=protocol.DEVELOPMENT_PANEL)
    parser.add_argument("--seed", type=int, choices=protocol.PRIMARY_SEEDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-matrix", action="store_true")
    parser.add_argument("--gpu-pool", default="1,2,3,4,5,6")
    args = parser.parse_args()
    if args.run_matrix:
        pool = tuple(int(item) for item in args.gpu_pool.split(",") if item.strip())
        print(json.dumps(run_matrix(args.output_dir or protocol.RESULT_ROOT / "M1_h0_isolation", gpu_pool=pool), indent=2, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value))
    elif args.dataset and args.seed is not None:
        if args.output_dir is None:
            parser.error("direct M1 jobs require --output-dir")
        print(json.dumps(run_job(args.dataset, args.seed, args.output_dir), indent=2, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value))
    else:
        parser.error("choose --run-matrix or --dataset/--seed/--output-dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
