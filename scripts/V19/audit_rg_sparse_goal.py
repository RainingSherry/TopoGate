#!/usr/bin/env python
"""Audit the preregistered sparse/high-dimensional RG goal end to end."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _load_extension(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read(path)
    if payload.get("audit_ok") is not True:
        raise RuntimeError(f"extension audit is incomplete: {path}")
    return payload


def _load_baselines(paths: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for path in paths:
        if not path.is_file():
            continue
        payload = _read(path)
        if payload.get("status") != "completed":
            raise RuntimeError(f"baseline audit is incomplete: {path}")
        for row in payload.get("rows", []):
            if row.get("status") != "completed":
                continue
            dataset_id = str(row["dataset_id"])
            method = str(row["method"])
            result.setdefault(dataset_id, {})[method] = row
    return result


def audit(extension_paths: list[Path], baseline_paths: list[Path], output: Path) -> dict[str, Any]:
    extensions = [payload for path in extension_paths if (payload := _load_extension(path)) is not None]
    if not extensions:
        raise RuntimeError("no completed extension summary was provided")
    datasets: dict[str, dict[str, Any]] = {}
    for payload in extensions:
        for row in payload.get("datasets", []):
            dataset_id = str(row["dataset_id"])
            if dataset_id in datasets:
                raise RuntimeError(f"duplicate dataset across extension panels: {dataset_id}")
            datasets[dataset_id] = row
    baselines = _load_baselines(baseline_paths)
    rows: list[dict[str, Any]] = []
    for dataset_id, extension in sorted(datasets.items()):
        winner = bool(extension.get("promotion_rg_win_by_mean_ari") is True)
        method_rows = baselines.get(dataset_id, {})
        method_aris = {
            method: float(row.get("metrics", {}).get("ari"))
            for method, row in method_rows.items()
            if row.get("metrics", {}).get("ari") is not None
        }
        best_method = max(method_aris, key=method_aris.get) if method_aris else None
        best_ari = method_aris.get(best_method) if best_method is not None else None
        rg_ari = float(extension["rg_ari_mean"])
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": extension.get("dataset"),
                "panel": extension.get("comparison_scope"),
                "rg_ari_mean": rg_ari,
                "scmae_ari_mean": float(extension["scmae_ari_mean"]),
                "delta_ari_mean": float(extension["delta_ari_mean"]),
                "rg_win_scmae": winner,
                "baseline_methods_completed": sorted(method_aris),
                "best_baseline_method": best_method,
                "best_baseline_ari": best_ari,
                "rg_win_best_baseline": bool(winner and best_ari is not None and rg_ari > best_ari),
                "baseline_complete_for_winner": bool(not winner or set(method_aris) == {"AHDPC", "DPC_GFNN", "GCC"}),
            }
        )
    winners = [row for row in rows if row["rg_win_scmae"]]
    missing_baseline = [row["dataset_id"] for row in winners if not row["baseline_complete_for_winner"]]
    goal_met = len(winners) >= 5
    result = {
        "status": "completed" if goal_met and not missing_baseline else ("insufficient_rg_wins" if not goal_met else "incomplete_compute"),
        "extension_panels": len(extensions),
        "n_datasets": len(rows),
        "rg_wins_over_scmae": len(winners),
        "minimum_required_rg_wins": 5,
        "goal_met": goal_met,
        "rg_wins_over_best_baseline": sum(bool(row["rg_win_best_baseline"]) for row in rows),
        "missing_baseline_for_winners": missing_baseline,
        "baseline_methods": ["AHDPC", "DPC_GFNN", "GCC"],
        "datasets": rows,
        "extension_summaries": [str(path.resolve()) for path in extension_paths if path.is_file()],
        "baseline_summaries": [str(path.resolve()) for path in baseline_paths if path.is_file()],
    }
    _write(output / "goal_audit.json", result)
    with (output / "goal_dataset_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["dataset_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-summary", type=Path, action="append", required=True)
    parser.add_argument("--baseline-summary", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.extension_summary, args.baseline_summary, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "n_datasets", "rg_wins_over_scmae", "rg_wins_over_best_baseline", "goal_met")}, ensure_ascii=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
