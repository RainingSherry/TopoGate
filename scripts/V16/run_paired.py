from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.TopoGate.V16_predictive_graph_gate.config import V16Config, load_config
from methods.TopoGate.V16_predictive_graph_gate.gate import assignment_readout, summarize_gate
from methods.TopoGate.V16_predictive_graph_gate.run import fit_v16, write_domain_status
from methods.TopoGate.V16_predictive_graph_gate.sparse import TheoryDomainError, load_npz_matrix
from scripts.V16.stress import apply_compound_stress


DEFAULT_VARIANTS = [
    "self_only",
    "fixed_predictive_graph",
    "V16_predictive_gate",
    "shuffled_support",
    "output_disabled",
]
VARIANTS = DEFAULT_VARIANTS


def _dump(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(y, prediction)),
        "nmi": float(normalized_mutual_info_score(y, prediction)),
        "ami": float(adjusted_mutual_info_score(y, prediction)),
    }


def _write_readout(
    reference_dir: Path,
    output_dir: Path,
    variant: str,
    q_self: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_valid: np.ndarray,
    support: np.ndarray,
    embedding_self: np.ndarray,
    y: np.ndarray,
    config: V16Config,
    base_summary: dict,
) -> None:
    q_out, pi, scores = assignment_readout(
        q_self,
        candidate_indices,
        candidate_valid,
        support,
        variant=variant,
        temperature=config.gate_temperature,
        seed=config.seed,
    )
    predictions = np.argmax(q_out, axis=1).astype(np.int64)
    summary = dict(base_summary)
    summary["variant"] = variant
    summary["paired_reference"] = str(reference_dir)
    summary["metrics"] = _metrics(y, predictions)
    summary["gate"] = {
        **summarize_gate(pi),
        "positive_score_rate": float(np.mean(scores[candidate_valid] > 0.0)) if candidate_valid.any() else 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "cluster_probabilities.npy", q_out)
    np.save(output_dir / "embedding_self.npy", embedding_self)
    np.save(output_dir / "embedding_final.npy", embedding_self)
    np.save(output_dir / "labels_true.npy", y)
    np.savez_compressed(
        output_dir / "gate_diagnostics.npz",
        candidate_indices=candidate_indices,
        candidate_valid=candidate_valid,
        support=support,
        gate_scores=scores,
        probabilities_self=q_self,
        probabilities_final=q_out,
        pi=pi,
    )
    config_dict = config.to_dict()
    config_dict["variant"] = variant
    _dump(config_dict, output_dir / "resolved_config.json")
    _dump(summary["metrics"], output_dir / "metrics.json")
    _dump(summary, output_dir / "summary.json")


def run_one(
    path: Path,
    output_root: Path,
    seed: int,
    config: V16Config,
    *,
    condition: str = "clean",
) -> None:
    X, input_storage = load_npz_matrix(path)
    with np.load(path, allow_pickle=False) as data:
        y_raw = np.asarray(data["y"])
    _, y = np.unique(y_raw, return_inverse=True)
    dataset = path.stem
    condition_root = output_root / dataset / condition
    if condition == "compound":
        X, stress_metadata, _ = apply_compound_stress(X, seed)
    elif condition == "clean":
        stress_metadata = {"mode": "clean", "changed": False}
    else:
        raise ValueError(f"unknown condition: {condition}")
    run_metadata = {"condition": condition, "stress": stress_metadata}
    reference_dir = condition_root / "V16_predictive_gate" / f"seed{seed}"
    try:
        _, base_summary = fit_v16(
            X,
            int(np.unique(y).size),
            y,
            config=config,
            save_dir=reference_dir,
            dataset_name=dataset,
            source_path=path,
            k_protocol="benchmark_oracle_from_y",
            input_storage=input_storage,
            run_metadata=run_metadata,
        )
    except TheoryDomainError as exc:
        for variant in VARIANTS:
            variant_config = replace(config, variant=variant)
            write_domain_status(
                save_dir=condition_root / variant / f"seed{seed}",
                config=variant_config,
                dataset_name=dataset,
                source_path=path,
                n_clusters=int(np.unique(y).size),
                k_protocol="benchmark_oracle_from_y",
                profile=exc.profile,
                run_metadata=run_metadata,
            )
        return
    with np.load(reference_dir / "gate_diagnostics.npz") as diagnostics:
        q_self = np.asarray(diagnostics["probabilities_self"])
        candidate_indices = np.asarray(diagnostics["candidate_indices"])
        candidate_valid = np.asarray(diagnostics["candidate_valid"])
        support = np.asarray(diagnostics["support"])
    embedding_self = np.load(reference_dir / "embedding_self.npy")
    for variant in VARIANTS:
        if variant == "V16_predictive_gate":
            continue
        _write_readout(
            reference_dir,
            condition_root / variant / f"seed{seed}",
            variant,
            q_self,
            candidate_indices,
            candidate_valid,
            support,
            embedding_self,
            y,
            config,
            base_summary,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="V16 paired readouts with one Stage-A/graph fit per seed")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--save_root", default="/tmp/v16_paired")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--condition", choices=("clean", "compound"), default="clean")
    args = parser.parse_args()
    overrides = {"variant": "V16_predictive_gate", "no_cuda": bool(args.no_cuda)}
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    for dataset in args.datasets:
        path = Path(args.data_root) / dataset
        for seed in args.seeds:
            config = load_config(args.config, {**overrides, "seed": int(seed)})
            run_one(path, Path(args.save_root), int(seed), config, condition=args.condition)


if __name__ == "__main__":
    main()
