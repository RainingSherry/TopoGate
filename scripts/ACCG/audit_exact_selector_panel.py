#!/usr/bin/env python3
"""Run the small W5 exact-selector audit over a frozen seed panel."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_single_audit():
    path = ROOT / "scripts/ACCG/audit_exact_selector.py"
    spec = importlib.util.spec_from_file_location("accg_single_exact_selector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load selector audit: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_audit


def run_panel(seeds: list[int], rows: int) -> dict[str, object]:
    run_audit = _load_single_audit()
    audits = [run_audit(int(seed), int(rows)) for seed in seeds]
    total_rows = sum(len(audit["rows"]) for audit in audits)
    feasible_rows = sum(int(audit["exact_feasible_rows"]) for audit in audits)
    feasible_greedy = [
        not row["greedy_infeasible"]
        for audit in audits
        for row in audit["rows"]
        if row["exact_feasible"]
    ]
    hardness_gaps = [
        row["hardness_gap_exact_minus_greedy"]
        for audit in audits
        for row in audit["rows"]
        if row["exact_feasible"]
    ]
    return {
        "protocol": "accg_small_w5_exact_gap_audit_panel_v2",
        "seeds": [int(seed) for seed in seeds],
        "rows_per_seed": int(rows),
        "rows": int(total_rows),
        "exact_feasible_rows": int(feasible_rows),
        "greedy_feasible_given_exact_rate": float(sum(feasible_greedy) / len(feasible_greedy))
        if feasible_greedy
        else float("nan"),
        "mean_hardness_gap": float(sum(hardness_gaps) / len(hardness_gaps))
        if hardness_gaps
        else float("nan"),
        "seed_audits": audits,
        "labels_used": False,
        "formal_training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_panel(args.seeds, args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": payload["rows"],
                "exact_feasible_rows": payload["exact_feasible_rows"],
                "formal_training_started": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
