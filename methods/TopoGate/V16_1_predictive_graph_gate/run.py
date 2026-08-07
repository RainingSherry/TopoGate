from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import sklearn
import torch
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

from .config import V16_1Config, load_config
from .gate import assignment_readout, cross_fitted_predictive_support, summarize_gate
from .graph import build_candidate_graph, candidate_recurrence, consensus_graph
from .sparse import (
    DenseNPZReference,
    TheoryDomainError,
    load_npz_matrix,
    prepare_counts,
    repeated_splits,
    summarize_split_views,
)
from .trainer import DeviceUnavailableError, resolve_device, train_stage_a


def _json_dump(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _encode_labels(y: np.ndarray | None) -> np.ndarray | None:
    if y is None:
        return None
    values = np.asarray(y).reshape(-1)
    _, encoded = np.unique(values, return_inverse=True)
    return encoded.astype(np.int64)


def write_domain_status(
    *,
    save_dir: str | Path,
    config: V16_1Config,
    dataset_name: str,
    source_path: str | Path | None,
    n_clusters: int,
    k_protocol: str,
    profile: dict[str, Any],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "method": "TopoGate",
        "version": "V16.1",
        "variant": config.variant,
        "status": "theory_domain_not_supported",
        "dataset": dataset_name,
        "source_path": None if source_path is None else str(Path(source_path).resolve()),
        "seed": int(config.seed),
        "n_samples": int(profile.get("n", 0)),
        "n_features": int(profile.get("d", 0)),
        "n_clusters": int(n_clusters),
        "K": int(n_clusters),
        "k_protocol": k_protocol,
        "benchmark_oracle_from_y": k_protocol == "benchmark_oracle_from_y",
        "labels_used_during_fit": False,
        "theory_certificate": profile,
        "metrics": {},
        "run_metadata": run_metadata or {},
        "condition": (run_metadata or {}).get("condition", "clean"),
        "output_files": {},
    }
    _json_dump(config.to_dict(), output / "resolved_config.json")
    _json_dump({}, output / "metrics.json")
    _json_dump(summary, output / "summary.json")
    return summary


def write_environment_status(
    *,
    save_dir: str | Path,
    config: V16_1Config,
    dataset_name: str,
    source_path: str | Path | None,
    n_clusters: int,
    k_protocol: str,
    error: Exception,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "method": "TopoGate",
        "version": "V16.1",
        "variant": config.variant,
        "status": "environment_error",
        "error": str(error),
        "dataset": dataset_name,
        "source_path": None if source_path is None else str(Path(source_path).resolve()),
        "seed": int(config.seed),
        "n_clusters": int(n_clusters),
        "K": int(n_clusters),
        "k_protocol": k_protocol,
        "benchmark_oracle_from_y": k_protocol == "benchmark_oracle_from_y",
        "labels_used_during_fit": False,
        "metrics": {},
        "run_metadata": run_metadata or {},
        "output_files": {},
    }
    _json_dump(config.to_dict(), output / "resolved_config.json")
    _json_dump({}, output / "metrics.json")
    _json_dump(summary, output / "summary.json")
    return summary


def fit_v16_1(
    X: np.ndarray | sp.spmatrix | DenseNPZReference,
    n_clusters: int,
    y: np.ndarray | None = None,
    *,
    config: V16_1Config | None = None,
    save_dir: str | Path | None = None,
    dataset_name: str = "adhoc",
    source_path: str | Path | None = None,
    k_protocol: str = "explicit_n_clusters",
    input_storage: str | None = None,
    count_semantics: str | None = None,
    semantics_source: str | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit V16.1 without allowing labels into Stage A, graph, or gate."""
    config = config or V16_1Config()
    if int(n_clusters) <= 0:
        raise ValueError("n_clusters must be positive")
    if y is not None and np.asarray(y).reshape(-1).size != int(X.shape[0]):
        raise ValueError("y must have one entry per input row")
    prepared = prepare_counts(
        X,
        enforce_domain=config.enforce_domain,
        count_semantics=count_semantics,
        semantics_source=semantics_source,
        min_feature_dim=config.min_feature_dim,
        min_zero_fraction=config.min_zero_fraction,
        min_median_nnz=config.min_median_nnz,
        max_empty_fraction=config.max_empty_fraction,
        input_storage=input_storage,
        require_sparse_input=config.require_sparse_input,
        input_policy=config.input_policy,
    )
    splits = repeated_splits(prepared.counts, config.thinning_fraction, config.support_repeats, config.seed)
    split_profile = summarize_split_views(splits)
    prepared.profile["count_split"] = split_profile
    if not split_profile["has_nonempty_heldout"]:
        profile = dict(prepared.profile)
        profile["domain_reasons"] = list(profile.get("domain_reasons", [])) + ["heldout_view_empty"]
        profile["theory_domain"] = "theory_domain_not_supported"
        raise TheoryDomainError(profile)
    if prepared.n_samples < int(n_clusters):
        profile = dict(prepared.profile)
        profile["domain_reasons"] = list(profile.get("domain_reasons", [])) + ["insufficient_samples_for_clusters"]
        profile["theory_domain"] = "theory_domain_not_supported"
        raise TheoryDomainError(profile)
    device = resolve_device(config)
    stage_a = train_stage_a(prepared, int(n_clusters), config, device)
    split_graphs = [build_candidate_graph(view_a, config.graph_k) for view_a, _ in splits]
    graph = consensus_graph(
        split_graphs,
        k=config.graph_k,
        min_repeats=config.consensus_min_repeats,
    )
    support, support_profile = cross_fitted_predictive_support(
        splits,
        graph.indices,
        graph.valid,
        smoothing=config.smoothing,
    )
    q_out, pi, gate_scores = assignment_readout(
        stage_a.probabilities,
        graph.indices,
        graph.valid,
        support,
        variant=config.variant,
        temperature=config.gate_temperature,
        seed=config.seed,
    )
    predictions = np.argmax(q_out, axis=1).astype(np.int64)
    encoded_y = _encode_labels(y)
    metrics: dict[str, Any] = {}
    if encoded_y is not None:
        metrics = {
            "ari": float(adjusted_rand_score(encoded_y, predictions)),
            "nmi": float(normalized_mutual_info_score(encoded_y, predictions)),
            "ami": float(adjusted_mutual_info_score(encoded_y, predictions)),
        }
    gate_summary = summarize_gate(pi)
    diagnostics = {
        "candidate_indices": graph.indices,
        "candidate_valid": graph.valid,
        "candidate_similarity": graph.similarity,
        "support": support,
        "gate_scores": gate_scores,
        "probabilities_self": stage_a.probabilities,
        "probabilities_final": q_out,
        "pi": pi,
    }
    summary: dict[str, Any] = {
        "method": "TopoGate",
        "version": "V16.1",
        "variant": config.variant,
        "dataset": dataset_name,
        "source_path": None if source_path is None else str(Path(source_path).resolve()),
        "seed": int(config.seed),
        "n_samples": int(prepared.n_samples),
        "n_features": int(prepared.n_features),
        "n_clusters": int(n_clusters),
        "K": int(n_clusters),
        "k_protocol": k_protocol,
        "benchmark_oracle_from_y": k_protocol == "benchmark_oracle_from_y",
        "labels_used_during_fit": False,
        "theory_certificate": prepared.profile,
        "graph_profile": {
            **graph.profile,
            "split_candidate_recurrence": candidate_recurrence(split_graphs),
            "posthoc_edge_purity": None if encoded_y is None else graph.edge_purity(encoded_y),
            "posthoc_candidate_recall": None if encoded_y is None else graph.recall(encoded_y),
            "posthoc_recall_definition": "budget_normalized_same_label_coverage",
        },
        "support_profile": support_profile,
        "readout_space": "assignment",
        "embedding_final_semantics": "topology_disabled_latent_equal_embedding_self",
        "gate": {
            **gate_summary,
            "positive_score_rate": float(np.mean(gate_scores[graph.valid] > 0.0)) if graph.valid.any() else 0.0,
        },
        "metrics": metrics,
        "run_metadata": run_metadata or {},
        "condition": (run_metadata or {}).get("condition", "clean"),
        "stage_a": {"seconds": stage_a.seconds, "history": stage_a.history},
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
        },
        "output_files": {
            "predictions": "predictions.npy",
            "labels_true": "labels_true.npy" if encoded_y is not None else None,
            "cluster_probabilities": "cluster_probabilities.npy",
            "embedding_self": "embedding_self.npy",
            "embedding_final": "embedding_final.npy",
            "gate_diagnostics": "gate_diagnostics.npz",
        },
    }
    if save_dir is not None:
        output = Path(save_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "predictions.npy", predictions)
        np.save(output / "cluster_probabilities.npy", q_out)
        np.save(output / "embedding_self.npy", stage_a.embedding)
        np.save(output / "embedding_final.npy", stage_a.embedding)
        if encoded_y is not None:
            np.save(output / "labels_true.npy", encoded_y)
        np.savez_compressed(output / "gate_diagnostics.npz", **diagnostics)
        _json_dump(config.to_dict(), output / "resolved_config.json")
        _json_dump(metrics, output / "metrics.json")
        _json_dump(summary, output / "summary.json")
    return predictions, summary


def run_v16_1(
    X: np.ndarray | sp.spmatrix | DenseNPZReference,
    n_clusters: int,
    y: np.ndarray | None = None,
    *,
    config_path: str | Path | None = None,
    save_dir: str | Path | None = None,
    dataset_name: str = "adhoc",
    seed: int = 42,
    source_path: str | Path | None = None,
    input_storage: str | None = None,
    count_semantics: str | None = None,
    semantics_source: str | None = None,
    **overrides: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = load_config(config_path, {**overrides, "seed": int(seed)})
    return fit_v16_1(
        X,
        n_clusters,
        y,
        config=config,
        save_dir=save_dir,
        dataset_name=dataset_name,
        source_path=source_path,
        input_storage=input_storage,
        count_semantics=count_semantics,
        semantics_source=semantics_source,
    )


def _load_npz(path: Path) -> tuple[np.ndarray | sp.csr_matrix | DenseNPZReference, np.ndarray | None, str]:
    X, storage = load_npz_matrix(path)
    with np.load(path, allow_pickle=False) as data:
        y = np.asarray(data["y"]) if "y" in data.files else None
    return X, y, storage


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoGate V16.1 predictive topology gate")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--count-semantics", default=None)
    parser.add_argument("--semantics-source", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    path = Path(args.data_path)
    X, y, input_storage = _load_npz(path)
    if args.n_clusters is None:
        if y is None:
            raise ValueError("--n_clusters is required when labels are absent")
        n_clusters = int(np.unique(y).size)
        k_protocol = "benchmark_oracle_from_y"
    else:
        n_clusters = int(args.n_clusters)
        k_protocol = "explicit_n_clusters"
    overrides = {
        key: value
        for key, value in {
            "variant": args.variant,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "no_cuda": True if args.no_cuda else None,
        }.items()
        if value is not None
    }
    config = load_config(args.config_path, {**overrides, "seed": int(args.seed)})
    try:
        _, summary = fit_v16_1(
            X,
            n_clusters,
            y,
            config=config,
            save_dir=args.save_dir,
            dataset_name=args.dataset_name or path.stem,
            source_path=path,
            k_protocol=k_protocol,
            input_storage=input_storage,
            count_semantics=args.count_semantics,
            semantics_source=args.semantics_source,
        )
    except TheoryDomainError as exc:
        summary = write_domain_status(
            save_dir=args.save_dir,
            config=config,
            dataset_name=args.dataset_name or path.stem,
            source_path=path,
            n_clusters=n_clusters,
            k_protocol=k_protocol,
            profile=exc.profile,
        )
    except DeviceUnavailableError as exc:
        summary = write_environment_status(
            save_dir=args.save_dir,
            config=config,
            dataset_name=args.dataset_name or path.stem,
            source_path=path,
            n_clusters=n_clusters,
            k_protocol=k_protocol,
            error=exc,
        )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
