"""C1 structural replay of the closed B1 corruption panel.

This script performs no model fitting and never loads labels.  It reuses the
audited S0 H0 matrices, applies the read-only B1 corruption functions to make
structural snapshots, and joins the already-published B1 post-fit ARI/L_rec
rows only as an external comparison column.  The output is therefore a
mechanism audit, not a second B1 performance result.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import protocol
from .corruption_library import residual_proxy
from .mechanism_diagnostics import combined_diagnostics


OLD_ARMS = (
    "C_clean_no_corruption",
    "C0_MatchedRandom",
    "C1_ValueOnly",
    "C2_SupportOnly",
    "C3_MixedMatched",
    "C4_StaticHard",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_b1_rows() -> dict[tuple[str, str, int], dict[str, Any]]:
    path = protocol.B1_ROOT / "b1_run_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"closed B1 compact summary is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["dataset"], row["arm"], int(row["seed"])): row for row in rows}


def _load_h0(dataset: str) -> np.ndarray:
    path = protocol.H0_ROOT / dataset / "H0.npy"
    if not path.exists():
        raise FileNotFoundError(f"audited S0 H0 is missing: {path}")
    h0 = np.asarray(np.load(path), dtype=np.float32)
    if h0.ndim != 2 or not np.isfinite(h0).all():
        raise ValueError(f"invalid H0 for {dataset}: {h0.shape}")
    return h0


def run(output_dir: Path, *, seeds: tuple[int, ...] = protocol.PRIMARY_SEEDS) -> dict[str, Any]:
    protocol.validate_contract()
    b1 = _load_b1_rows()
    rows: list[dict[str, Any]] = []
    for dataset in protocol.DEVELOPMENT_PANEL:
        clean = _load_h0(dataset)
        for arm in OLD_ARMS:
            for seed in seeds:
                from scripts.adaptive_corruption_probe.b1_corruption_library import corrupt_h0

                rng = np.random.default_rng(int(seed))
                static_residual = residual_proxy(clean) if arm == "C4_StaticHard" else None
                corrupted, old_stats = corrupt_h0(clean, arm, rng, static_residual=static_residual)
                diagnostics = combined_diagnostics(clean, corrupted)
                b1_row = b1.get((dataset, arm, int(seed)), {})
                rows.append(
                    {
                        "dataset": dataset,
                        "role": protocol.ROLE_BY_DATASET[dataset],
                        "old_b1_arm": arm,
                        "seed": int(seed),
                        "status": "completed_valid",
                        "labels_used_for_structure": False,
                        "b1_metric_provenance": "closed_B1_post_fit_compact_summary" if b1_row else "missing",
                        "b1_ARI_after_fit": float(b1_row["ARI"]) if b1_row else "",
                        "b1_L_rec_after_fit": float(b1_row["L_rec"]) if b1_row else "",
                        **{f"b1_{key}": value for key, value in old_stats.items()},
                        **diagnostics,
                    }
                )

    stamp = _timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_stamp = output_dir / f"c1_structural_rows_{stamp}.csv"
    csv_latest = output_dir / "c1_structural_rows.csv"
    _write_csv(csv_stamp, rows)
    csv_latest.write_bytes(csv_stamp.read_bytes())
    summary = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "C1_mechanism_localization",
        "status": "completed_valid",
        "rows": len(rows),
        "datasets": list(protocol.DEVELOPMENT_PANEL),
        "arms": list(OLD_ARMS),
        "seeds": list(seeds),
        "fit_runs": 0,
        "labels_loaded": False,
        "b1_metrics_reused": bool(b1),
        "b1_metrics_are_post_fit": True,
        "structural_source": "S0_H0_replay_with_closed_B1_corruption_function",
        "static_hard_score_source": "label_free_column_median_MAD_proxy_for_structural_audit_only",
        "publication_scope": "compact_diagnostics_only; no arrays, labels, embeddings, predictions, weights or logs",
        "csv_timestamped": str(csv_stamp),
        "csv_latest": str(csv_latest),
    }
    json_stamp = output_dir / f"c1_structural_summary_{stamp}.json"
    json_latest = output_dir / "c1_structural_summary.json"
    _write_json(json_stamp, summary)
    json_latest.write_bytes(json_stamp.read_bytes())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=protocol.RESULT_ROOT / "C1_mechanism_audit")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()

