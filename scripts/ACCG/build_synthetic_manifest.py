#!/usr/bin/env python3
"""Generate the frozen W0-W5 synthetic inputs without launching training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.ACCG_action_constrained_gate.synthetic import SyntheticConfig, write_panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/synthetic_contract.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7, 2025, 2026])
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    config = SyntheticConfig(**payload)
    config.validate()
    manifest = write_panel(args.output_dir, config, seeds=tuple(int(seed) for seed in args.seeds))
    print(
        json.dumps(
            {
                "status": "generated_not_run",
                "records": len(manifest["records"]),
                "formal_training_started": False,
                "manifest": str((args.output_dir / "manifest.json").resolve()),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
