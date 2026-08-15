#!/usr/bin/env python3
"""Build auditable, dataset-level summaries for one frozen E1 phase."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.V25.summarize_e1 import classify_effect, phase_gate


PROTOCOL_ID = "v25_e1_v21_matched_nrt_v1"
SEEDS = (42, 123, 7)
ARMS = ("N", "R", "T")
PAIRS = ("I_full_ARI", "S_full_ARI", "I_1step_ARI", "S_1step_ARI")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_manifest_payload(manifest: dict[str, Any], phase: str) -> dict[str, Any]:
    """Return the frozen arm-job payload for one auditable phase."""
    if manifest.get("manifest_id") == "v25_holdout_e1_manifest_v1":
        if phase != "holdout":
            raise ValueError("the holdout manifest can only audit phase=holdout")
        payload = manifest
    else:
        payload = manifest.get("phases", {}).get(phase)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest has no frozen phase {phase!r}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"manifest phase {phase!r} has no jobs")
    return payload


def manifest_expectations(manifest: dict[str, Any], phase: str) -> dict[str, Any]:
    """Validate and extract the expected panel/dataset/seed coverage.

    The phase denominator is derived from the frozen arm manifest, never from
    directories that happen to remain in a result tree.  A panel is identified
    by its manifest ``panel_run_key`` and must contain exactly one N/R/T arm
    triplet with a single dataset and seed.
    """
    payload = _phase_manifest_payload(manifest, phase)
    errors: list[str] = []
    panel_map: dict[str, dict[str, Any]] = {}
    for job in payload["jobs"]:
        if not isinstance(job, dict):
            errors.append("non-object job row")
            continue
        panel_key = str(job.get("panel_run_key", ""))
        arm = str(job.get("arm", ""))
        if not panel_key:
            errors.append("job missing panel_run_key")
            continue
        if arm not in ARMS:
            errors.append(f"panel {panel_key} has invalid arm {arm!r}")
        entry = panel_map.setdefault(
            panel_key,
            {"dataset": job.get("dataset"), "seed": job.get("seed"), "arms": []},
        )
        if entry["dataset"] != job.get("dataset") or entry["seed"] != job.get("seed"):
            errors.append(f"panel {panel_key} differs across dataset/seed")
        entry["arms"].append(arm)
    for panel_key, entry in panel_map.items():
        if sorted(entry["arms"]) != sorted(ARMS) or len(entry["arms"]) != len(ARMS):
            errors.append(f"panel {panel_key} does not contain exactly one N/R/T triplet")
    expected_panel_count = int(payload.get("expected_panel_jobs", len(panel_map)))
    if expected_panel_count != len(panel_map):
        errors.append(
            f"manifest expected_panel_jobs={expected_panel_count} but has {len(panel_map)} panel keys"
        )
    expected_arm_count = int(payload.get("expected_arm_jobs", len(payload["jobs"])))
    if expected_arm_count != len(payload["jobs"]):
        errors.append(
            f"manifest expected_arm_jobs={expected_arm_count} but has {len(payload['jobs'])} rows"
        )
    return {
        "manifest_id": manifest.get("manifest_id"),
        "phase": phase,
        "valid": not errors,
        "errors": errors,
        "expected_panel_count": expected_panel_count,
        "expected_arm_count": expected_arm_count,
        "panel_map": panel_map,
        "panel_keys": sorted(panel_map),
        "datasets": sorted({str(entry["dataset"]) for entry in panel_map.values()}),
        "seeds": sorted({int(entry["seed"]) for entry in panel_map.values()}),
        "dataset_seed_map": {
            dataset: sorted({int(entry["seed"]) for entry in panel_map.values() if str(entry["dataset"]) == dataset})
            for dataset in sorted({str(entry["dataset"]) for entry in panel_map.values()})
        },
    }


def _infer_phase(root: Path, manifest: dict[str, Any], explicit_phase: str | None) -> str:
    if explicit_phase:
        return explicit_phase
    if manifest.get("manifest_id") == "v25_holdout_e1_manifest_v1":
        return "holdout"
    phases = manifest.get("phases", {})
    if root.name in phases:
        return root.name
    if len(phases) == 1:
        return next(iter(phases))
    raise ValueError("phase is ambiguous; pass --phase explicitly")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _panel_audit(panel: Path) -> tuple[dict[str, Any], dict[str, float]]:
    summary = _read(panel / "summary.json")
    audit = _read(panel / "audit.json")
    manifest = _read(panel / "manifest_record.json")
    profile = _read(panel / "runner_profile.json")
    arm_metrics = {
        arm: _read(panel / arm / "metrics.json")
        for arm in ARMS
        if (panel / arm / "metrics.json").is_file()
    }
    recomputed_ari: dict[str, float] = {}
    labels_recomputed = False
    data_path = str(profile.get("data_path", ""))
    source_path = Path(data_path) if data_path else None
    if source_path is not None and source_path.is_file():
        try:
            from methods.TopoGate.V21_assignment_adversarial_gate.input_adapter import load_npz

            loaded = load_npz(source_path)
            if loaded.labels is not None:
                labels = np.asarray(loaded.labels).astype(str)
                for arm in ARMS:
                    prediction_path = panel / arm / "predictions.npy"
                    if prediction_path.is_file():
                        predictions = np.asarray(np.load(prediction_path, allow_pickle=False))
                        if predictions.shape == (labels.shape[0],):
                            recomputed_ari[arm] = float(adjusted_rand_score(labels, predictions))
                labels_recomputed = len(recomputed_ari) == len(ARMS)
        except (OSError, ValueError, TypeError):
            labels_recomputed = False
    recomputed_pairs: dict[str, float] = {}
    if labels_recomputed:
        recomputed_pairs = {
            "I_full_ARI": recomputed_ari["R"] - recomputed_ari["N"],
            "S_full_ARI": recomputed_ari["T"] - recomputed_ari["R"],
        }
    one_step_payload = _read(panel / "one_step.json") if (panel / "one_step.json").is_file() else {}
    one_step_metrics = {
        arm: one_step_payload.get(arm, {}).get("metrics", {})
        for arm in ARMS
    }
    one_step_pairs: dict[str, float] = {}
    if all(isinstance(one_step_metrics[arm].get("ari"), (int, float)) and math.isfinite(float(one_step_metrics[arm]["ari"])) for arm in ARMS):
        one_step_pairs = {
            "I_1step_ARI": float(one_step_metrics["R"]["ari"]) - float(one_step_metrics["N"]["ari"]),
            "S_1step_ARI": float(one_step_metrics["T"]["ari"]) - float(one_step_metrics["R"]["ari"]),
        }
    stored_pairs = summary.get("pairs", {})
    stored_primary_pairs_match = bool(recomputed_pairs) and all(
        key in stored_pairs
        and math.isfinite(float(stored_pairs[key]))
        and math.isclose(float(stored_pairs[key]), value, rel_tol=1e-10, abs_tol=1e-10)
        for key, value in recomputed_pairs.items()
    )
    stored_one_step_pairs_match = bool(one_step_pairs) and all(
        key in stored_pairs
        and math.isfinite(float(stored_pairs[key]))
        and math.isclose(float(stored_pairs[key]), value, rel_tol=1e-10, abs_tol=1e-10)
        for key, value in one_step_pairs.items()
    )
    checks: dict[str, bool] = {
        "summary_completed": summary.get("status") == "completed",
        "protocol_match": summary.get("protocol_id") == PROTOCOL_ID and audit.get("protocol_id") == PROTOCOL_ID,
        "seed_match": int(summary.get("seed", -1)) == int(manifest.get("seed", -2)) == int(profile.get("seed", -3)),
        "seed_declared": int(summary.get("seed", -1)) in SEEDS,
        "source_exists": source_path is not None and source_path.is_file(),
        "source_hash_match": False,
        "labels_used_during_fit_false": audit.get("labels_used_during_fit") is False,
        "tr_shared_schedule": all(audit.get("TR_shared_schedule_hashes", {}).get(key) is True for key in ("donor", "eligible", "budget", "selection_noise")),
        "none_no_assignment_or_js": audit.get("none_contract", {}).get("assignment_forward_calls") == 0 and audit.get("none_contract", {}).get("js_forward_calls") == 0,
        "pairs_complete": set(summary.get("pairs", {})) == set(PAIRS) and all(
            isinstance(summary["pairs"].get(key), (int, float)) and math.isfinite(float(summary["pairs"][key]))
            for key in PAIRS
        ),
        "arms_complete": all((panel / arm / "metrics.json").is_file() for arm in ARMS),
        "arm_metrics_finite": len(arm_metrics) == len(ARMS) and all(
            isinstance(metrics.get("ari"), (int, float)) and math.isfinite(float(metrics["ari"]))
            for metrics in arm_metrics.values()
        ),
        "primary_ari_recomputed_from_saved_predictions": labels_recomputed,
        "stored_primary_pairs_match_recomputed": stored_primary_pairs_match,
        "stored_one_step_pairs_match_recomputed": stored_one_step_pairs_match,
        "labels_after_fit_only": len(arm_metrics) == len(ARMS) and all(metrics.get("labels_used_after_fit_only") is True for metrics in arm_metrics.values()),
        "branchpoint_contract": audit.get("branchpoint", {}).get("warmup_branchpoint_before_first_assignment") is True and audit.get("branchpoint", {}).get("head_initialised") is True,
        "one_step_present": (panel / "one_step.json").is_file() and bool(one_step_pairs),
    }
    if checks["source_exists"] and manifest.get("source_sha256"):
        checks["source_hash_match"] = _sha256(Path(profile["data_path"])) == manifest["source_sha256"]
    row: dict[str, Any] = {
        "dataset": manifest.get("dataset", profile.get("dataset")),
        "seed": int(summary.get("seed")),
        "panel_run_key": manifest.get("panel_run_key"),
        "panel_path": str(panel.resolve()),
        "audit_ok": all(checks.values()),
        **checks,
    }
    pairs = {}
    if row["audit_ok"]:
        pairs.update(recomputed_pairs)
        pairs.update(
            {
                key: float(summary["pairs"][key])
                for key in ("I_1step_ARI", "S_1step_ARI")
                if summary.get("pairs", {}).get(key) is not None
            }
        )
    return row, pairs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_phase(root: Path, manifest_path: Path | None = None, phase: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    if manifest_path is None:
        manifest_path = root / "manifest_snapshot.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"frozen E1 manifest is required for phase audit: {manifest_path}; "
            "pass --manifest or preserve manifest_snapshot.json"
        )
    manifest = _read(manifest_path)
    phase = _infer_phase(root, manifest, phase)
    expected = manifest_expectations(manifest, phase)
    panel_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    one_step_rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[int, list[float]]] = {}
    panel_pairs: dict[str, dict[str, float]] = {}
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in {"logs", "mplconfig", "E2"}:
            continue
        for seed_dir in sorted(dataset_dir.glob("seed*")):
            if not (seed_dir / "summary.json").is_file():
                continue
            try:
                panel, pairs = _panel_audit(seed_dir)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                panel = {
                    "dataset": dataset_dir.name,
                    "seed": int(seed_dir.name.removeprefix("seed")) if seed_dir.name.removeprefix("seed").isdigit() else -1,
                    "panel_run_key": None,
                    "panel_path": str(seed_dir.resolve()),
                    "audit_ok": False,
                    "audit_error": f"{type(exc).__name__}: {exc}",
                }
                pairs = {}
            panel_rows.append(panel)
            if panel.get("panel_path"):
                panel_pairs[str(Path(str(panel["panel_path"])).resolve())] = pairs
    observed_by_key: dict[str, list[dict[str, Any]]] = {}
    for panel in panel_rows:
        key = str(panel.get("panel_run_key") or "")
        observed_by_key.setdefault(key, []).append(panel)
    expected_keys = set(expected["panel_keys"])
    observed_keys = {key for key in observed_by_key if key}
    duplicate_keys = sorted(key for key, rows in observed_by_key.items() if key and len(rows) > 1)
    unexpected_keys = sorted(key for key in observed_keys - expected_keys)
    missing_keys = sorted(expected_keys - observed_keys)
    for panel in panel_rows:
        key = str(panel.get("panel_run_key") or "")
        coverage_ok = bool(key and key in expected_keys and len(observed_by_key.get(key, [])) == 1)
        if coverage_ok:
            spec = expected["panel_map"][key]
            coverage_ok = panel.get("dataset") == spec["dataset"] and int(panel.get("seed", -1)) == int(spec["seed"])
        panel["manifest_coverage_ok"] = coverage_ok
        panel["artifact_audit_ok"] = bool(panel.get("audit_ok"))
        panel["coverage_error"] = None if coverage_ok else (
            "unexpected_or_missing_panel_key" if key not in expected_keys else "duplicate_or_dataset_seed_mismatch"
        )
        panel["audit_ok"] = bool(panel["artifact_audit_ok"] and coverage_ok)
    for key in missing_keys:
        spec = expected["panel_map"][key]
        panel_rows.append(
            {
                "dataset": spec["dataset"],
                "seed": int(spec["seed"]),
                "panel_run_key": key,
                "panel_path": None,
                "audit_ok": False,
                "artifact_audit_ok": False,
                "manifest_coverage_ok": False,
                "coverage_error": "missing_expected_panel",
                "missing_expected_panel": True,
            }
        )
    # Invalid or undeclared panels remain visible in the audit table, but
    # cannot contribute a dataset-level effect.
    for panel in panel_rows:
        if not panel.get("audit_ok"):
            continue
        panel_path = Path(str(panel["panel_path"]))
        pairs = panel_pairs.get(str(panel_path.resolve()), {})
        try:
            if not set(("I_full_ARI", "S_full_ARI")) <= set(pairs):
                raise ValueError("valid panel has incomplete primary pair values")
            pair_rows.append({"dataset": panel["dataset"], "seed": panel["seed"], **pairs})
            grouped.setdefault(panel["dataset"], {}).setdefault(int(panel["seed"]), [])
            grouped[panel["dataset"]][int(panel["seed"])] = [pairs[key] for key in ("I_full_ARI", "S_full_ARI")]
            t_probe = _read(panel_path / "T" / "gradient_probe.json")
            for timepoint, values in sorted(t_probe.items()):
                gradient_rows.append({"dataset": panel["dataset"], "seed": panel["seed"], "timepoint": timepoint, **values})
            one_step = _read(panel_path / "one_step.json")
            for arm in ARMS:
                one_step_rows.append({"dataset": panel["dataset"], "seed": panel["seed"], "arm": arm, "ari": one_step[arm]["metrics"].get("ari"), "nmi": one_step[arm]["metrics"].get("nmi"), "loss": one_step[arm].get("loss")})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            panel["audit_ok"] = False
            panel["artifact_audit_ok"] = False
    datasets: dict[str, Any] = {}
    expected_datasets = set(expected["datasets"])
    for dataset in sorted(expected_datasets):
        seed_map = grouped.get(dataset, {})
        available_seeds = sorted(seed_map)
        required_seeds = expected["dataset_seed_map"].get(dataset, list(SEEDS))
        if set(available_seeds) != set(required_seeds):
            datasets[dataset] = {
                "I_d": classify_effect([]),
                "S_d": classify_effect([]),
                "seeds": available_seeds,
                "required_seeds": required_seeds,
                "inference_status": "inconclusive_invalid_or_incomplete_panel_set",
                "statistical_unit": "dataset; seeds are repeated measurements",
            }
            continue
        i_values = [values[0] for _, values in sorted(seed_map.items())]
        s_values = [values[1] for _, values in sorted(seed_map.items())]
        datasets[dataset] = {
            "I_d": classify_effect(i_values),
            "S_d": classify_effect(s_values),
            "seeds": available_seeds,
                "required_seeds": required_seeds,
            "inference_status": "complete_valid_seed_set",
            "statistical_unit": "dataset; seeds are repeated measurements",
        }
    for dataset, seed_map in sorted(grouped.items()):
        if dataset in expected_datasets:
            continue
        datasets[dataset] = {
            "I_d": classify_effect([]),
            "S_d": classify_effect([]),
            "seeds": sorted(seed_map),
            "required_seeds": [],
            "inference_status": "inconclusive_unexpected_dataset",
            "statistical_unit": "dataset; seeds are repeated measurements",
        }
    coverage_complete = (
        expected["valid"]
        and not missing_keys
        and not unexpected_keys
        and not duplicate_keys
        and observed_keys == expected_keys
        and set(panel.get("panel_run_key") for panel in panel_rows if panel.get("panel_run_key")) == expected_keys
    )
    return {
        "protocol_id": "v25_e1_phase_audit_v1",
        "e1_protocol_id": PROTOCOL_ID,
        "phase_root": str(root.resolve()),
        "manifest_id": expected["manifest_id"],
        "manifest_path": str(manifest_path.resolve()),
        "phase": phase,
        "manifest_valid": expected["valid"],
        "manifest_errors": expected["errors"],
        "expected_panel_count": expected["expected_panel_count"],
        "expected_arm_count": expected["expected_arm_count"],
        "expected_panel_keys": expected["panel_keys"],
        "expected_datasets": expected["datasets"],
        "expected_seeds": expected["seeds"],
        "observed_panel_count": len(panel_rows) - len(missing_keys),
        "observed_panel_keys": sorted(observed_keys),
        "missing_expected_panel_keys": missing_keys,
        "unexpected_panel_keys": unexpected_keys,
        "duplicate_panel_keys": duplicate_keys,
        "coverage_complete": coverage_complete,
        "panel_count": len(panel_rows),
        "audit_ok_count": sum(bool(row["audit_ok"]) for row in panel_rows),
        "datasets": datasets,
        "phase_gate": phase_gate(datasets, expected_datasets),
        "invalid_or_incomplete_panel_count": sum(not bool(row["audit_ok"]) for row in panel_rows),
        "statistical_unit": "dataset; seeds are repeated measurements",
        "equivalence_claim": False,
        "effect_state_rule": "Positive/Negative/Observed-Small/Inconclusive at delta=0.03; no bootstrap equivalence inference",
        "panel_rows": panel_rows,
        "pair_rows": pair_rows,
        "gradient_rows": gradient_rows,
        "one_step_rows": one_step_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--phase", choices=("pilot", "confirmation", "holdout"), default=None)
    args = parser.parse_args()
    payload = audit_phase(args.root, args.manifest, args.phase)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in payload.items() if key not in {"panel_rows", "pair_rows", "gradient_rows", "one_step_rows"}}
    (args.out_dir / "phase_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    _write_csv(args.out_dir / "panel_audit.csv", payload["panel_rows"])
    _write_csv(args.out_dir / "pair_effects.csv", payload["pair_rows"])
    _write_csv(args.out_dir / "gradient_probe.csv", payload["gradient_rows"])
    _write_csv(args.out_dir / "one_step_metrics.csv", payload["one_step_rows"])
    print(json.dumps({key: summary[key] for key in ("panel_count", "audit_ok_count", "phase_gate")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
