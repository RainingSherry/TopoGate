#!/usr/bin/env python
"""Summarize post-freeze V19 results and archived SOTA controls.

The external CSV is treated as an archived reference, not as a fresh matched
run.  This script never selects a variant, seed, or hyperparameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FINAL_PROTOCOL_ID = "v19_rg_final_postfreeze_v1"
COMPARABLE_SCOPE = "archived_sota_bridge_eligible"
METRICS = ("ari", "nmi", "acc")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _underlying(dataset_id: str) -> str:
    return str(dataset_id).split("__", 1)[0]


def _display_names(summary: dict[str, Any]) -> set[str]:
    dataset_id = str(summary.get("dataset_id", ""))
    dataset = str(summary.get("dataset", ""))
    values = {dataset, _underlying(dataset_id)}
    normalized = {value.casefold().replace(" ", "_") for value in values if value}
    if "baron_human" in normalized:
        values.update({"Baron Human", "baron_human"})
    if "mouse_retina" in normalized:
        values.update({"Mouse_retina", "mouse_retina"})
    if "campbell" in normalized:
        values.update({"Campbell", "campbell"})
    return {value for value in values if value}


def _float_metric(summary: dict[str, Any], metric: str) -> float | None:
    metrics = summary.get("metrics", {})
    key = {"ari": "ari", "nmi": "nmi", "acc": "acc"}[metric]
    value = metrics.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _load_final(root: Path, *, allow_diagnostic: bool = False) -> tuple[list[dict[str, Any]], bool]:
    status_path = root / "launcher_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"missing final launcher status: {status_path}")
    status = _read_json(status_path)
    if status.get("protocol_id") != FINAL_PROTOCOL_ID or status.get("status") != "completed":
        raise ValueError("final root is not a completed V19 post-freeze matrix")
    if status.get("audit_ok") is not True:
        raise ValueError("final root does not have audit_ok=true")
    diagnostic = bool(status.get("no_go", False))
    if diagnostic and not allow_diagnostic:
        raise ValueError(
            "final root was run from a no_go selection; pass --allow-diagnostic to label it explicitly"
        )
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        payload = _read_json(path)
        if payload.get("status") != "completed":
            continue
        if payload.get("final_protocol_id") != FINAL_PROTOCOL_ID:
            raise ValueError(f"unexpected final protocol in {path}")
        if payload.get("labels_used_during_fit") is not False:
            raise ValueError(f"label-fit audit failed in {path}")
        if not payload.get("evaluation_variant"):
            raise ValueError(f"missing evaluation_variant in {path}")
        rows.append(payload)
    if not rows:
        raise ValueError("no completed final summaries found")
    return rows, diagnostic


def _load_baseline(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate(values: list[float]) -> tuple[float | None, float | None, int]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None, None, 0
    return float(mean(clean)), float(stdev(clean)) if len(clean) > 1 else 0.0, len(clean)


def summarize(
    final_root: Path,
    baseline_csv: Path,
    output_dir: Path,
    *,
    allow_diagnostic: bool = False,
) -> dict[str, Any]:
    final_rows, diagnostic = _load_final(final_root, allow_diagnostic=allow_diagnostic)
    baseline_rows = _load_baseline(baseline_csv)
    baseline_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        name = str(row.get("dataset", ""))
        if name:
            baseline_by_dataset[name.casefold().replace(" ", "_")].append(row)

    long_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    external_seen: set[tuple[str, str, str]] = set()
    for row in final_rows:
        dataset_id = str(row["dataset_id"])
        variant = str(row["evaluation_variant"])
        scope = str(row.get("comparison_scope", ""))
        grouped[(dataset_id, variant, scope)].append(row)
    for (dataset_id, variant, scope), rows in sorted(grouped.items()):
        for metric in METRICS:
            values = [value for value in (_float_metric(row, metric) for row in rows) if value is not None]
            value_mean, value_std, count = _aggregate(values)
            long_rows.append(
                {
                    "dataset_id": dataset_id,
                    "underlying_dataset_id": _underlying(dataset_id),
                    "comparison_scope": scope,
                    "method": f"V19_{variant}",
                    "method_type": "fresh_postfreeze_v19",
                    "metric": metric,
                    "mean": value_mean,
                    "std": value_std,
                    "n": count,
                    "source": str(final_root),
                    "protocol_note": "fresh post-freeze V19 run; labels only for benchmark K and post-fit metrics",
                }
            )

        if scope == COMPARABLE_SCOPE:
            names = set()
            for row in rows:
                names.update(_display_names(row))
            external: list[dict[str, Any]] = []
            for name in names:
                external.extend(baseline_by_dataset.get(name.casefold().replace(" ", "_"), []))
            unique_external = {(str(row.get("model")), str(row.get("dataset"))): row for row in external}
            for (method, _dataset), baseline in sorted(unique_external.items()):
                for metric in METRICS:
                    external_key = (dataset_id, method, metric)
                    if external_key in external_seen:
                        continue
                    external_seen.add(external_key)
                    value = baseline.get(metric.upper())
                    numeric = None if value in (None, "") else float(value)
                    long_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "underlying_dataset_id": _underlying(dataset_id),
                            "comparison_scope": scope,
                            "method": method,
                            "method_type": "archived_external_reference",
                            "metric": metric,
                            "mean": numeric,
                            "std": None,
                            "n": 1 if numeric is not None else 0,
                            "source": str(baseline_csv),
                            "protocol_note": "archived baseline CSV; not a fresh matched V19 rerun",
                        }
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    fields = [
        "dataset_id", "underlying_dataset_id", "comparison_scope", "method",
        "method_type", "metric", "mean", "std", "n", "source", "protocol_note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(long_rows)

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        by_dataset[str(row["dataset_id"])].append(row)
    report = [
        "# V19 post-freeze comparison",
        "",
        f"- Final root: `{final_root}`",
        f"- Archived baseline source: `{baseline_csv}`",
        "- V19 fit remains label-free; labels are used only for benchmark K and post-fit metrics.",
        f"- Selection status: {'no_go diagnostic' if diagnostic else 'proxy_supported final'}.",
        "- Only `archived_sota_bridge_eligible` layers are joined to archived external baselines.",
        "- Missing external rows remain missing; no zero imputation is performed.",
        "",
        "## Dataset-level V19 means",
        "",
        "| Dataset | Scope | Variant | ARI mean±std | NMI mean±std | ACC mean±std | n |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    v19_rows = [row for row in long_rows if row["method_type"] == "fresh_postfreeze_v19"]
    index: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in v19_rows:
        index[(str(row["dataset_id"]), str(row["comparison_scope"]), str(row["method"]))][str(row["metric"])] = row
    for key in sorted(index):
        dataset_id, scope, method = key
        values = index[key]
        def fmt(metric: str) -> str:
            row = values.get(metric, {})
            if row.get("mean") is None:
                return "NA"
            return f"{float(row['mean']):.4f}±{float(row['std'] or 0.0):.4f}"
        n = values.get("ari", {}).get("n", 0)
        report.append(f"| {dataset_id} | {scope} | {method} | {fmt('ari')} | {fmt('nmi')} | {fmt('acc')} | {n} |")
    report.extend([
        "",
        "## External reference rows",
        "",
        "See `comparison.csv` for the long-form table. External values retain their archived provenance and are not interpreted as fresh matched SOTA runs.",
        "",
    ])
    md_path = output_dir / "comparison.md"
    _write_text(md_path, "\n".join(report))
    result = {
        "status": "completed",
        "final_protocol_id": FINAL_PROTOCOL_ID,
        "final_root": str(final_root),
        "baseline_source": str(baseline_csv),
        "rows": len(long_rows),
        "datasets": sorted({str(row["dataset_id"]) for row in final_rows}),
        "labels_used_during_fit": False,
        "variant_selection_uses_labels_or_outcomes": False,
        "selection_status": "no_go" if diagnostic else "proxy_supported",
        "no_go": diagnostic,
        "diagnostic": diagnostic,
        "outputs": {"comparison_csv": str(csv_path), "comparison_md": str(md_path)},
    }
    _write_text(output_dir / "comparison_summary.json", json.dumps(result, ensure_ascii=True, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize V19 final results against archived baselines")
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, default=ROOT / "result" / "baseline_comparison" / "summary.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-diagnostic",
        action="store_true",
        help="summarize a no_go post-freeze run with an explicit diagnostic label",
    )
    args = parser.parse_args()
    result = summarize(
        args.final_root,
        args.baseline_csv,
        args.output_dir,
        allow_diagnostic=bool(args.allow_diagnostic),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
