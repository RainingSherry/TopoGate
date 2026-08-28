#!/usr/bin/env python3
"""Audit and summarize the V22 full single-seed queue without fitting anything."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "result" / "V22" / "v22_full_single_seed_20260812"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _audit_job(job: dict[str, Any]) -> dict[str, Any]:
    output = Path(job["output_dir"])
    stratum = str(job["record"].get("stratum", job["record"].get("family", "unclassified")))
    summary = _read_json(output / "summary.json")
    config = _read_json(output / "resolved_config.json")
    required = [
        "summary.json",
        "resolved_config.json",
        "metrics.json",
        "training_history.json",
        "embedding_final.npy",
        "predictions.npy",
        "checkpoint.pt",
        "manifest_record.json",
        "launch_record.json",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if summary is None or config is None:
        if job.get("status") == "incomplete_compute":
            _write_json(
                output / "incomplete_compute.json",
                {
                    "status": "incomplete_compute",
                    "run_key": job["run_key"],
                    "dataset_id": job["dataset_id"],
                    "dataset": job["record"].get("name"),
                    "returncode": job.get("returncode"),
                    "error": job.get("error"),
                    "launcher_status": "interrupted",
                    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "claim_boundary": "no model result; retained as incomplete compute",
                },
            )
        return {
            "run_key": job["run_key"],
            "dataset_id": job["dataset_id"],
            "stratum": stratum,
            "status": job["status"],
            "artifact_ok": False,
            "missing": missing,
            "labels_used_during_fit": None,
            "ari": None,
            "nmi": None,
        }
    diagnostics = summary.get("diagnostics", {})
    history = diagnostics.get("history", [])
    finite_history = bool(history) and all(
        isinstance(row, dict)
        and all(isinstance(value, (int, float)) and value == value for value in row.values())
        for row in history
    )
    artifact_ok = (
        not missing
        and summary.get("status") == "completed"
        and summary.get("protocol_id") == job["protocol_id"]
        and summary.get("variant") == job["variant"]
        and int(summary.get("seed", -1)) == 42
        and summary.get("dataset") == job["record"]["name"]
        and summary.get("source_sha256") == job["record"]["source_sha256"]
        and summary.get("labels_used_during_fit") is False
        and summary.get("K_used_during_fit") is False
        and config.get("source_sha256") == job["record"]["source_sha256"]
        and finite_history
    )
    metrics = summary.get("metrics", {})
    return {
        "run_key": job["run_key"],
        "dataset_id": job["dataset_id"],
        "dataset": summary.get("dataset"),
        "stratum": stratum,
        "status": summary.get("status"),
        "artifact_ok": bool(artifact_ok),
        "missing": missing,
        "labels_used_during_fit": summary.get("labels_used_during_fit"),
        "K_used_during_fit": summary.get("K_used_during_fit"),
        "K_source": summary.get("K_source"),
        "n_samples": summary.get("n_samples"),
        "n_features": summary.get("n_features"),
        "ari": metrics.get("ari"),
        "nmi": metrics.get("nmi"),
        "acc": metrics.get("acc"),
        "discriminator_steps": diagnostics.get("discriminator_steps"),
        "gate_updates": diagnostics.get("gate_updates"),
        "gate_nonzero_update_rate": diagnostics.get("gate_nonzero_update_rate"),
        "effective_mask_rate_last": history[-1].get("adversarial_effective_rate") if history else None,
        "d_real_accuracy_last": history[-1].get("discriminator_real_accuracy") if history else None,
        "d_gate_fake_accuracy_last": history[-1].get("discriminator_gate_fake_accuracy") if history else None,
        "d_scmae_fake_accuracy_last": history[-1].get("discriminator_scmae_fake_accuracy") if history else None,
    }


def summarize(root: Path) -> dict[str, Any]:
    state = _read_json(root / "queue_state.json")
    manifest = _read_json(root / "manifest.json")
    if state is None or manifest is None:
        raise FileNotFoundError(f"queue_state.json and manifest.json are required under {root}")
    rows = [_audit_job(job) for job in state.get("jobs", [])]
    by_stratum: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("ari"), (float, int)):
            by_stratum[str(row["stratum"])].append(float(row["ari"]))
    strata = {
        key: {"completed_with_ari": len(values), "mean_ari": sum(values) / len(values) if values else None}
        for key, values in sorted(by_stratum.items())
    }
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[str(row.get("status"))] += 1
    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_id": manifest.get("manifest_id"),
        "protocol_id": manifest.get("protocol_id"),
        "queue_status": state.get("status"),
        "job_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "audit_ok_count": sum(bool(row.get("artifact_ok")) for row in rows),
        "strata": strata,
        "claim_boundary": "single-seed full-component evidence; no cross-dataset efficacy claim",
        "rows": rows,
    }
    (root / "aggregate_summary.json").write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    lines = [
        "# V22 Full Single-Seed Audit",
        "",
        f"- Manifest: `{result['manifest_id']}`",
        f"- Queue status: `{result['queue_status']}`",
        f"- Jobs: `{result['job_count']}`",
        f"- Status counts: `{result['status_counts']}`",
        f"- Artifact audits passed: `{result['audit_ok_count']}/{result['job_count']}`",
        "- Boundary: single-seed full-component evidence only; no efficacy claim or configuration selection.",
        "",
        "| Dataset | Stratum | Status | Artifact | ARI | NMI | D steps | Gate updates | Gate nonzero |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset', row['dataset_id'])} | {row['stratum']} | {row['status']} | "
            f"{row.get('artifact_ok')} | {row.get('ari')} | {row.get('nmi')} | "
            f"{row.get('discriminator_steps')} | {row.get('gate_updates')} | {row.get('gate_nonzero_update_rate')} |"
        )
    lines.extend(["", "## Strata", ""])
    for key, value in strata.items():
        lines.append(f"- `{key}`: {value}")
    (root / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps({key: result[key] for key in ("queue_status", "job_count", "status_counts", "audit_ok_count", "strata")}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
