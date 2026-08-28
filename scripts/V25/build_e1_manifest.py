#!/usr/bin/env python3
"""Build the gated V25 E1 pilot/confirmation manifest without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
A2_DEFAULT = ROOT / "result" / "V25_systematic_mechanism_study" / "A2" / "A2_decision.json"
OUT_DEFAULT = ROOT / "result" / "V25_systematic_mechanism_study" / "E1"
SEEDS = (42, 123, 7)
PILOT = (
    ("cnae9", "shared_text", ROOT / "datasets/cnae9.npz"),
    ("Mouse_retina", "clubench_bridge", ROOT / "datasets/Mouse_retina.npz"),
    ("sms_spam_collection", "shared_text", ROOT / "datasets/sms_spam_collection.npz"),
)
CONFIRMATION = (
    ("Baron Human", "clubench_bridge", ROOT / "datasets/Baron Human.npz"),
    ("Campbell", "clubench_bridge", ROOT / "datasets/Campbell.npz"),
    ("hate_speech", "shared_text", ROOT / "datasets/hate_speech.npz"),
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_jobs(phase: str, rows: tuple[tuple[str, str, Path], ...], output_root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for dataset, protocol, path in rows:
        for seed in SEEDS:
            run_key = f"v25_e1_{phase}::{dataset}::{seed}"
            jobs.append(
                {
                    "run_key": run_key,
                    "phase": phase,
                    "dataset": dataset,
                    "input_protocol": protocol,
                    "source_path": str(path.resolve()),
                    "source_sha256": sha256_file(path),
                    "seed": seed,
                    "arms": ["N", "R", "T"],
                    "output_dir": str((output_root / phase / dataset.replace(" ", "_") / f"seed{seed}").resolve()),
                    "primary_readout": "clean_embedding_known_k_kmeans",
                    "K_source": "benchmark_oracle_from_y",
                    "labels_used_during_fit": False,
                    "selection_uses_labels_or_outcomes": False,
                    "status": "queued_manifest_only",
                }
            )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2", type=Path, default=A2_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--phase", choices=("pilot", "confirmation", "both"), default="both")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision = json.loads(args.a2.read_text(encoding="utf-8")).get("decision")
    if decision != "retain_e1":
        raise SystemExit(f"A2 decision is {decision!r}; E1 manifest generation is vetoed")
    phases: list[tuple[str, tuple[tuple[str, str, Path], ...]]] = []
    if args.phase in {"pilot", "both"}:
        phases.append(("pilot", PILOT))
    if args.phase in {"confirmation", "both"}:
        phases.append(("confirmation", CONFIRMATION))
    args.out.mkdir(parents=True, exist_ok=True)
    phase_payload: dict[str, Any] = {}
    for phase, rows in phases:
        panel_jobs = build_jobs(phase, rows, args.out)
        arm_jobs: list[dict[str, Any]] = []
        for panel in panel_jobs:
            for arm in ("N", "R", "T"):
                arm_jobs.append(
                    panel
                    | {
                        "run_key": f"{panel['run_key']}::{arm}",
                        "panel_run_key": panel["run_key"],
                        "arm": arm,
                        "execution_unit": "shared_three_arm_panel",
                    }
                )
        phase_payload[phase] = {
            "expected_panel_jobs": len(panel_jobs),
            "expected_arm_jobs": len(arm_jobs),
            "jobs": arm_jobs,
        }
    payload = {
        "manifest_id": "v25_e1_gated_manifest_v1",
        "protocol_id": "v25_e1_v21_matched_nrt_v1",
        "a2_decision": decision,
        "generated_without_e1_outcomes": True,
        "seeds": list(SEEDS),
        "arms": ["N", "R", "T"],
        # Immutable manifest policy label retained for compatibility with the
        # launcher contracts created by the completed E1 runs.  Phase audit
        # result files use the phase-agnostic `phase_gate` field.
        "pilot_gate": "at least 2/3 datasets have seed-stable material I or S; signs may differ",
        "phases": phase_payload,
    }
    (args.out / "e1_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_id": payload["manifest_id"], "phases": {key: {"panel_jobs": value["expected_panel_jobs"], "arm_jobs": value["expected_arm_jobs"]} for key, value in payload["phases"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
