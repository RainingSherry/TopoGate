#!/usr/bin/env python
"""Audit and summarize the fixed sparse/high-dimensional V19 extension matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 7)
EXTENSION_PROTOCOLS = frozenset(
    {
        "v19_rg_extended_sparse_v1",
        "v19_rg_extended_sparse_batch2_v1",
    }
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read(path)
    if manifest.get("protocol_id") not in EXTENSION_PROTOCOLS:
        raise ValueError(f"unexpected extension protocol: {manifest.get('protocol_id')}")
    return manifest


def summarize(manifest_path: Path, output_dir: Path, result_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []
    for record in manifest["datasets"]:
        if record.get("status") != "eligible":
            continue
        dataset_id = str(record["dataset_id"])
        for variant in ("rg_full", "scmae_only"):
            for seed in SEEDS:
                path = result_dir / dataset_id / variant / f"seed{seed}"
                summary_path = path / "summary.json"
                status_path = path / "status.json"
                key = f"{dataset_id}::{variant}::seed{seed}"
                if not summary_path.exists() or not status_path.exists():
                    missing.append(key)
                    continue
                try:
                    summary = _read(summary_path)
                    status = _read(status_path)
                except Exception:
                    invalid.append(key)
                    continue
                if summary.get("status") != "completed" or status.get("status") != "completed":
                    invalid.append(key)
                    continue
                metrics = summary.get("metrics", {})
                if not all(metric in metrics for metric in ("ari", "nmi", "acc")):
                    invalid.append(key)
                    continue
                if summary.get("labels_used_during_fit") is not False:
                    invalid.append(f"{key}:labels_used_during_fit")
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset": record["name"],
                        "family": record.get("family"),
                        "input_protocol": record["input_protocol"],
                        "comparison_scope": record.get("comparison_scope"),
                        "variant": variant,
                        "seed": int(seed),
                        "ari": float(metrics["ari"]),
                        "nmi": float(metrics["nmi"]),
                        "acc": float(metrics["acc"]),
                        "labels_used_during_fit": summary.get("labels_used_during_fit"),
                        "K_source": summary.get("K_source"),
                    }
                )
    index = {(row["dataset_id"], row["variant"], row["seed"]): row for row in rows}
    detail: list[dict[str, Any]] = []
    for record in manifest["datasets"]:
        if record.get("status") != "eligible":
            continue
        dataset_id = str(record["dataset_id"])
        rg = [index[(dataset_id, "rg_full", seed)] for seed in SEEDS if (dataset_id, "rg_full", seed) in index]
        scmae = [index[(dataset_id, "scmae_only", seed)] for seed in SEEDS if (dataset_id, "scmae_only", seed) in index]
        if len(rg) != len(SEEDS) or len(scmae) != len(SEEDS):
            continue
        delta_ari = [rg_row["ari"] - sc_row["ari"] for rg_row, sc_row in zip(rg, scmae, strict=True)]
        delta_nmi = [rg_row["nmi"] - sc_row["nmi"] for rg_row, sc_row in zip(rg, scmae, strict=True)]
        delta_acc = [rg_row["acc"] - sc_row["acc"] for rg_row, sc_row in zip(rg, scmae, strict=True)]
        detail.append(
            {
                "dataset_id": dataset_id,
                "dataset": record["name"],
                "family": record.get("family"),
                "input_protocol": record["input_protocol"],
                "comparison_scope": record.get("comparison_scope"),
                "rg_ari_mean": statistics.mean(row["ari"] for row in rg),
                "scmae_ari_mean": statistics.mean(row["ari"] for row in scmae),
                "delta_ari_mean": statistics.mean(delta_ari),
                "delta_ari_std": statistics.stdev(delta_ari),
                "delta_ari_positive_seed_count": sum(value > 0.0 for value in delta_ari),
                "rg_nmi_mean": statistics.mean(row["nmi"] for row in rg),
                "scmae_nmi_mean": statistics.mean(row["nmi"] for row in scmae),
                "delta_nmi_mean": statistics.mean(delta_nmi),
                "rg_acc_mean": statistics.mean(row["acc"] for row in rg),
                "scmae_acc_mean": statistics.mean(row["acc"] for row in scmae),
                "delta_acc_mean": statistics.mean(delta_acc),
                "promotion_rg_win_by_mean_ari": statistics.mean(delta_ari) > 0.0,
                "promotion_rg_confirmed_all_seeds": all(value > 0.0 for value in delta_ari),
            }
        )
    detail.sort(key=lambda row: (-float(row["delta_ari_mean"]), str(row["dataset_id"])))
    completed = len(rows)
    expected = len([row for row in manifest["datasets"] if row.get("status") == "eligible"]) * 2 * len(SEEDS)
    audit_ok = not missing and not invalid and completed == expected and len(detail) == expected // (2 * len(SEEDS))
    result = {
        "status": "completed" if audit_ok else "incomplete_compute",
        "protocol_id": manifest["protocol_id"],
        "manifest_id": manifest["manifest_id"],
        "expected_runs": expected,
        "completed_runs": completed,
        "missing_runs": missing,
        "invalid_runs": invalid,
        "audit_ok": audit_ok,
        "labels_used_during_fit": False,
        "selection_uses_labels_or_outcomes": False,
        "selection_evidence_type": "fixed pre-registered extension transfer",
        "n_datasets": len(detail),
        "rg_above_scmae_by_mean_ari": sum(bool(row["promotion_rg_win_by_mean_ari"]) for row in detail),
        "rg_confirmed_all_seeds": sum(bool(row["promotion_rg_confirmed_all_seeds"]) for row in detail),
        "datasets": detail,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "extension_summary.json", result)
    with (output_dir / "extension_dataset_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(detail[0].keys()) if detail else ["dataset_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail)
    with (output_dir / "extension_run_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["dataset_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# V19 sparse/high-dimensional extension",
        "",
        f"- Matrix audit: `{'audit_ok' if audit_ok else 'incomplete_compute'}` ({completed}/{expected} runs).",
        "- Dataset selection was fixed before extension outcomes; labels were not used during fitting or input selection.",
        "- `rg_win_by_mean_ari` is an evaluation summary. It does not authorize changing the candidate panel.",
        "",
        "| Dataset | RG ARI | scMAE ARI | Delta ARI | Positive seeds | All seeds positive |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in detail:
        report.append(
            f"| {row['dataset_id']} | {row['rg_ari_mean']:.4f} | {row['scmae_ari_mean']:.4f} | "
            f"{row['delta_ari_mean']:+.4f} | {row['delta_ari_positive_seed_count']}/3 | "
            f"{'yes' if row['promotion_rg_confirmed_all_seeds'] else 'no'} |"
        )
    (output_dir / "extension_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.manifest, args.output_dir, args.result_dir)
    print(json.dumps({key: result[key] for key in ("status", "expected_runs", "completed_runs", "audit_ok", "rg_above_scmae_by_mean_ari", "rg_confirmed_all_seeds")}, ensure_ascii=True))
    return 0 if result["audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
