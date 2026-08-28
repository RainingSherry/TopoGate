#!/usr/bin/env python3
"""Run one V25 E1 matched N/R/T panel for a dataset and seed.

This is the only new-training entry point currently authorized by V25 A2.  It
uses the V21 input adapter and keeps labels outside the fit function; labels
are loaded by this outer runner only for known-K readout metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_A2_DECISION = ROOT / "result" / "V25_systematic_mechanism_study" / "A2" / "A2_decision.json"

from methods.TopoGate.V21_assignment_adversarial_gate.input_adapter import load_npz, prepare_dual_input
from methods.TopoGate.V25_systematic_mechanism_study.e1_protocol import E1Config, run_e1


ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_e1_authorization(path: Path) -> tuple[dict[str, Any], str]:
    """Require the immutable A2 decision before any formal E1 computation."""
    if not path.is_file():
        raise ValueError(f"A2 decision is required before E1: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid A2 decision file: {path}") from exc
    if not isinstance(payload, dict) or payload.get("decision") != "retain_e1":
        decision = payload.get("decision") if isinstance(payload, dict) else None
        raise ValueError(f"E1 is vetoed unless A2 decision is retain_e1; got {decision!r}")
    return payload, _sha256(path)


def _resolve_n_clusters(loaded: Any, explicit_n_clusters: int | None) -> tuple[int, str]:
    """Resolve K without guessing; labels are metadata for the outer readout only."""
    if explicit_n_clusters is not None:
        n_clusters = int(explicit_n_clusters)
        source = "explicit_n_clusters"
    elif loaded.labels is not None:
        n_clusters = int(np.unique(np.asarray(loaded.labels)).size)
        source = "benchmark_oracle_from_y"
    else:
        raise ValueError("--n-clusters is required when the NPZ has no labels")
    if n_clusters <= 1:
        raise ValueError("n_clusters must be greater than one")
    return n_clusters, source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-protocol", choices=("clubench_bridge", "shared_text", "scRNA_count"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a2-decision", type=Path, default=DEFAULT_A2_DECISION)
    parser.add_argument("--seed", type=int, choices=(42, 123, 7), default=42)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _a2_payload, a2_sha256 = _require_e1_authorization(args.a2_decision)
    if args.device == "cuda":
        if args.gpu not in ALLOWED_GPUS:
            raise ValueError(f"CUDA requires a physical GPU in {sorted(ALLOWED_GPUS)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:0")
    else:
        if args.gpu is not None:
            raise ValueError("--gpu cannot be used with --device cpu")
        threads = max(1, int(os.environ.get("TOPOGATE_CPU_THREADS", "1")))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(2, threads)))
        device = torch.device("cpu")

    loaded = load_npz(args.data)
    n_clusters, k_source = _resolve_n_clusters(loaded, args.n_clusters)
    prepared = prepare_dual_input(loaded.X, dataset_name=args.dataset_name, input_protocol=args.input_protocol)
    config_values: dict[str, Any] = {}
    if args.epochs is not None:
        config_values["epochs"] = int(args.epochs)
        config_values["warmup_epochs"] = int(args.warmup_epochs if args.warmup_epochs is not None else max(1, args.epochs // 2))
    if args.warmup_epochs is not None:
        config_values["warmup_epochs"] = int(args.warmup_epochs)
    if args.batch_size is not None:
        config_values["batch_size"] = int(args.batch_size)
    if args.hidden_size is not None:
        config_values["hidden_size"] = int(args.hidden_size)
    config = E1Config(**config_values)
    result = run_e1(
        prepared.X_model,
        prepared.X_graph,
        n_clusters=n_clusters,
        config=config,
        seed=args.seed,
        device=device,
        evaluation_labels=loaded.labels,
        output_dir=args.output_dir,
    )
    _write_json(
        args.output_dir / "runner_profile.json",
        {
            "dataset": args.dataset_name,
            "data_path": str(args.data.resolve()),
            "input_protocol": args.input_protocol,
            "seed": args.seed,
            "n_clusters": n_clusters,
            "K_source": k_source,
            "a2_decision_path": str(args.a2_decision.resolve()),
            "a2_decision_sha256": a2_sha256,
            "a2_decision": _a2_payload["decision"],
            "labels_loaded_by_outer_runner": loaded.labels is not None,
            "labels_used_during_fit": False,
            "preprocess_profile": prepared.profile,
            "selected_feature_indices": int(prepared.selected_feature_indices.size),
        },
    )
    print(json.dumps({"status": result["status"], "pairs": result["pairs"], "audit": result["audit"]}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
