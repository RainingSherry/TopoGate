"""Finalize exploratory readout examples from all three completed seed outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from readout_analysis import DATASETS, select_exploratory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--case-dir", type=Path, nargs=3, required=True,
                        help="Three completed non-smoke per-case output directories")
    parser.add_argument("--out", type=Path, required=True,
                        help="Unique dataset-level output directory")
    args = parser.parse_args()
    result = select_exploratory(args.dataset, args.case_dir, args.out)
    print(json.dumps({"status": result["status"], "dataset": args.dataset,
                      "selection_kind": result["selection_kind"],
                      "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
