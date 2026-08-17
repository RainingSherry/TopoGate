from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from methods.TopoGate.V21_assignment_adversarial_gate.input_adapter import load_npz, prepare_dual_input
from methods.TopoGate.V21_assignment_adversarial_gate.trainer import ALLOWED_PHYSICAL_GPUS

from .config import load_config
from .protocol import run_constrained_from_branchpoint, run_matched_panel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one matched ACCG N/R/T_s/T_c panel")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-protocol", choices=("clubench_bridge", "shared_text", "scRNA_count"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 7), default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="engineering-only override")
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument(
        "--branchpoint-from",
        type=Path,
        default=None,
        help="reuse N/R/T_s and the shared branchpoint; run only this config's T_c arm",
    )
    return parser.parse_args()


def run_one(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cuda":
        if args.gpu not in ALLOWED_PHYSICAL_GPUS:
            raise ValueError(f"CUDA requires a physical GPU in {sorted(ALLOWED_PHYSICAL_GPUS)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:0")
    else:
        if args.gpu is not None:
            raise ValueError("--gpu cannot be used with --device cpu")
        threads = max(1, int(os.environ.get("TOPOGATE_CPU_THREADS", "1")))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(2, threads)))
        device = torch.device("cpu")
    config = load_config(args.config)
    if args.epochs is not None:
        values = config.to_dict()
        values["v21"]["epochs"] = int(args.epochs)
        values["v21"]["warmup_epochs"] = int(
            args.warmup_epochs if args.warmup_epochs is not None else max(1, args.epochs // 2)
        )
        from .config import ACCGConfig, FeatureConstraintConfig
        from methods.TopoGate.V25_systematic_mechanism_study.e1_protocol import E1Config

        config = ACCGConfig(
            protocol_id=values["protocol_id"],
            variant=values["variant"],
            v21=E1Config(**values["v21"]),
            constraint=FeatureConstraintConfig(**values["constraint"]),
        )
        config.validate()
    elif args.warmup_epochs is not None:
        raise ValueError("--warmup-epochs requires --epochs")
    loaded = load_npz(args.data)
    if args.n_clusters is not None:
        n_clusters = int(args.n_clusters)
        k_source = "explicit_n_clusters"
    elif loaded.labels is not None:
        n_clusters = int(np.unique(loaded.labels).size)
        k_source = "benchmark_oracle_from_y"
    else:
        raise ValueError("--n-clusters is required when labels are unavailable")
    prepared = prepare_dual_input(loaded.X, dataset_name=args.dataset_name, input_protocol=args.input_protocol)
    if args.branchpoint_from is None:
        result = run_matched_panel(
            prepared.X_model,
            prepared.X_graph,
            n_clusters=n_clusters,
            config=config,
            seed=args.seed,
            device=device,
            evaluation_labels=loaded.labels,
            output_dir=args.output_dir,
        )
    else:
        result = run_constrained_from_branchpoint(
            prepared.X_model,
            prepared.X_graph,
            n_clusters=n_clusters,
            config=config,
            seed=args.seed,
            branchpoint_path=args.branchpoint_from,
            device=device,
            evaluation_labels=loaded.labels,
            output_dir=args.output_dir,
        )
    runner = {
        "status": result["status"],
        "dataset": args.dataset_name,
        "dataset_path": str(args.data.resolve()),
        "dataset_sha256": _sha256(args.data),
        "input_protocol": args.input_protocol,
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "seed": int(args.seed),
        "n_clusters": int(n_clusters),
        "K_source": k_source,
        "labels_loaded_by_outer_runner": loaded.labels is not None,
        "labels_used_during_fit": False,
        "preprocess_profile": prepared.profile,
        "variant": config.variant,
        "branchpoint_reused": args.branchpoint_from is not None,
        "reused_from": None if args.branchpoint_from is None else str(args.branchpoint_from.resolve()),
        "evidence_level": "engineering_smoke" if config.v21.epochs < 10 else "experiment",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runner_profile.json").write_text(
        json.dumps(runner, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    np.save(args.output_dir / "selected_feature_indices.npy", prepared.selected_feature_indices)
    if loaded.labels is not None:
        _unique, encoded = np.unique(np.asarray(loaded.labels).astype(str), return_inverse=True)
        np.save(args.output_dir / "labels_true.npy", encoded.astype(np.int64))
    return runner


if __name__ == "__main__":
    print(json.dumps(run_one(parse_args()), indent=2, ensure_ascii=True))
