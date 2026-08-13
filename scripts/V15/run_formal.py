#!/usr/bin/env python3
"""Launcher for the registered V15 paired confirmation matrix.

The launcher is deliberately explicit: every variant, condition, seed, source
path, and K protocol is written to a manifest. It does not select datasets or
variants from observed ARI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V15_counterfactual_gate.config import load_config
from methods.TopoGate.V15_counterfactual_gate.run import fit_v15, load_npz
from scripts.V15.run_corruption_diagnostics import corrupt


DATASETS = (
    "Mouse_retina",
    "cnae9",
    "imdb",
    "sms_spam_collection",
    "secom",
    "enron",
    "reuters",
    "20newsgroups",
    "cifar10",
    "CIFAR10_CLIP",
    "labeled_faces_in_the_wild",
    "flickr_material_database",
    "ISOLET",
    "olivetti_faces",
    "mnist64",
    "seeds",
)
VARIANTS = (
    "self_only",
    "union_uniform",
    "direct_counterfactual",
    "direct_local_consensus",
    "counterfactual_learned",
    "forced_topk",
    "shuffled_utility",
)

VARIANT_OVERRIDES: dict[str, dict[str, Any]] = {
    "direct_counterfactual": {
        "gate_mode": "direct_counterfactual",
        "utility_target_mode": "operator_aligned",
        "counterfactual_distill_weight": 0.0,
        "final_prediction_source": "gate_readout",
    },
    "direct_local_consensus": {
        # Same exact sparsemax/readout operator as direct_counterfactual; only
        # the detached utility reference changes to leave-one-candidate-out
        # local consensus. This isolates the target mechanism.
        "gate_mode": "direct_counterfactual",
        "utility_target_mode": "local_consensus",
        "counterfactual_distill_weight": 0.0,
        "final_prediction_source": "gate_readout",
    },
    "counterfactual_learned": {
        "gate_mode": "counterfactual_learned",
        "utility_target_mode": "local_consensus",
        "lambda_gate": 0.5,
        "final_prediction_source": "gate_readout",
    },
    "self_only": {
        "gate_mode": "self_only",
        "counterfactual_distill_weight": 0.0,
        "final_prediction_source": "gate_readout",
    },
    "union_uniform": {"gate_mode": "union_uniform", "final_prediction_source": "gate_readout"},
    "forced_topk": {"gate_mode": "forced_topk", "final_prediction_source": "gate_readout"},
    "shuffled_utility": {"gate_mode": "shuffled_utility", "final_prediction_source": "gate_readout"},
}


def _hash_input(X: Any) -> str:
    digest = hashlib.sha256()
    if hasattr(X, "tocsr"):
        matrix = X.tocsr()
        for value in (matrix.data, matrix.indices, matrix.indptr, np.asarray(matrix.shape, dtype=np.int64)):
            digest.update(np.asarray(value).tobytes())
    else:
        digest.update(np.asarray(X, dtype=np.float32).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-root", type=Path, default=ROOT / "result" / "V15" / "formal")
    parser.add_argument("--config", type=Path, default=ROOT / "methods" / "TopoGate" / "V15_counterfactual_gate" / "configs" / "topogate_v15.yaml")
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--variant", action="append", choices=VARIANTS, default=None)
    parser.add_argument("--condition", choices=("clean", "compound"), default="clean")
    parser.add_argument("--seeds", default="42,123,7")
    parser.add_argument("--feature-fraction", type=float, default=0.2)
    parser.add_argument("--row-fraction", type=float, default=0.1)
    parser.add_argument("--noise-scale", type=float, default=0.2)
    parser.add_argument("--graph-replacement-fraction", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    names = tuple(args.dataset) if args.dataset else DATASETS
    variants = tuple(args.variant) if args.variant else VARIANTS
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    overrides: dict[str, Any] = {}
    for value in args.overrides:
        if "=" not in value:
            raise ValueError(f"override must be key=value: {value}")
        key, raw = value.split("=", 1)
        overrides[key] = yaml.safe_load(raw)
    output_root = args.output_root.resolve()
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for dataset in names:
        path = args.dataset_root / f"{dataset}.npz"
        if not path.exists():
            failures.append({"dataset": dataset, "error": f"missing dataset: {path}"})
            continue
        try:
            X, y = load_npz(path)
            if y is None:
                failures.append({"dataset": dataset, "error": "labels are required for benchmark K in this launcher"})
                continue
            K = int(np.unique(y).size)
            for seed in seeds:
                corrupted, corruption_meta, _ = (
                    corrupt(
                        X,
                        "compound",
                        seed,
                        feature_fraction=args.feature_fraction,
                        row_fraction=args.row_fraction,
                        noise_scale=args.noise_scale,
                    )
                    if args.condition == "compound"
                    else (X, {"mode": "clean", "changed": False}, np.zeros(int(X.shape[0]), dtype=np.uint8))
                )
                for variant in variants:
                    save_dir = output_root / f"{dataset}__{args.condition}__{variant}__seed{seed}"
                    record = {
                        "dataset": dataset,
                        "condition": args.condition,
                        "variant": variant,
                        "seed": seed,
                        "path": str(path.resolve()),
                        "K": K,
                        "k_protocol": "benchmark_oracle_from_y",
                        "labels_used_during_fit": False,
                        "input_sha256": _hash_input(corrupted),
                        "output": str(save_dir),
                    }
                    records.append(record)
                    if args.dry_run:
                        continue
                    mode_overrides = {
                        **overrides,
                        "seed": seed,
                        **VARIANT_OVERRIDES.get(variant, {"gate_mode": variant}),
                    }
                    if args.condition == "compound" and "graph_replacement_fraction" not in overrides:
                        mode_overrides["graph_replacement_fraction"] = args.graph_replacement_fraction
                    config = load_config(args.config, mode_overrides)
                    fit_v15(
                        corrupted,
                        K,
                        y,
                        config=config,
                        save_dir=save_dir,
                        dataset_name=dataset,
                        source_path=path,
                        k_protocol="benchmark_oracle_from_y",
                        run_metadata={
                            "stage": "Stage3",
                            "condition": args.condition,
                            "variant": variant,
                            "corruption": {
                                **corruption_meta,
                                "graph_replacement_fraction": mode_overrides.get(
                                    "graph_replacement_fraction",
                                    0.0,
                                ),
                            },
                            "labels_used_during_fit": False,
                        },
                    )
        except Exception as exc:
            failures.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
    manifest = {
        "stage": "Stage3",
        "condition": args.condition,
        "variants": list(variants),
        "seeds": seeds,
        "graph_replacement_fraction": args.graph_replacement_fraction
        if args.condition == "compound"
        else 0.0,
        "records": records,
        "failures": failures,
        "dry_run": args.dry_run,
        "labels_used_during_fit": False,
        "note": "Paired confirmation matrix; do not select variants from these outputs post hoc.",
    }
    if not args.dry_run:
        (output_root / "formal_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "failures": len(failures), "dry_run": args.dry_run, "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
