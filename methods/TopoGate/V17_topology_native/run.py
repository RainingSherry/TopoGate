from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

from .config import V17Config, load_config
from .model import TopologyState, fit_topology, readout_topology


def _write_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _encode_labels(y: np.ndarray | None) -> tuple[np.ndarray | None, list[str] | None]:
    if y is None:
        return None, None
    values = np.asarray(y).reshape(-1)
    unique, encoded = np.unique(values, return_inverse=True)
    return encoded.astype(np.int64), [str(value) for value in unique.tolist()]


def _posthoc_metrics(labels_true: np.ndarray | None, predictions: np.ndarray) -> dict[str, float]:
    if labels_true is None:
        return {}
    return {
        "ari": float(adjusted_rand_score(labels_true, predictions)),
        "nmi": float(normalized_mutual_info_score(labels_true, predictions)),
        "ami": float(adjusted_mutual_info_score(labels_true, predictions)),
    }


def _save_topology(topology: TopologyState, output: Path) -> None:
    sp.save_npz(output / "coefficient_matrix.npz", topology.relation.coefficients, compressed=True)
    sp.save_npz(output / "affinity_matrix.npz", topology.affinity, compressed=True)
    np.savez_compressed(
        output / "candidate_graph.npz",
        indices=topology.candidates.indices,
        similarity=topology.candidates.similarity,
        valid=topology.candidates.valid,
        view_count=topology.candidates.view_count,
        candidate_coefficients=topology.relation.candidate_coefficients,
        solver_iterations=topology.relation.iterations,
    )


def fit_v17(
    X: np.ndarray | sp.spmatrix,
    n_clusters: int,
    y: np.ndarray | None = None,
    *,
    config: V17Config | None = None,
    save_dir: str | Path | None = None,
    dataset_name: str = "adhoc",
    source_path: str | Path | None = None,
    k_protocol: str = "explicit_n_clusters",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit V17 while keeping labels and K outside topology estimation."""
    config = config or V17Config()
    if int(n_clusters) <= 0:
        raise ValueError("n_clusters must be positive")
    if y is not None and np.asarray(y).reshape(-1).size != int(X.shape[0]):
        raise ValueError("y must have one entry per input row")
    topology = fit_topology(X, config)
    spectral = readout_topology(topology, int(n_clusters), config)
    labels_true, label_values = _encode_labels(y)
    metrics = _posthoc_metrics(labels_true, spectral.labels)
    summary: dict[str, Any] = {
        "method": "TopoGate",
        "version": "V17-reference",
        "status": spectral.profile["status"],
        "dataset": dataset_name,
        "source_path": None if source_path is None else str(Path(source_path).resolve()),
        "seed": int(config.seed),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_clusters": int(n_clusters),
        "K": int(n_clusters),
        "k_protocol": k_protocol,
        "benchmark_oracle_from_y": k_protocol == "benchmark_oracle_from_y",
        "labels_used_during_fit": False,
        "K_used_in_input_adapter": False,
        "K_used_in_candidate_graph": False,
        "K_used_in_relation_solver": False,
        "K_used_in_spectral_readout": True,
        "label_values": label_values,
        "topology": topology.profile,
        "spectral": spectral.profile,
        "metrics": metrics,
        "output_semantics": {
            "predictions": "normalized spectral partition; -1 means topology abstention",
            "embedding_final": "normalized spectral embedding derived only from affinity_matrix",
            "coefficient_matrix": "shared sparse self-expression C and exact-zero edge gate",
            "affinity_matrix": "abs(C)+abs(C.T)",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "output_files": {
            "resolved_config": "resolved_config.json",
            "predictions": "predictions.npy",
            "labels_true": "labels_true.npy" if labels_true is not None else None,
            "embedding_final": "embedding_final.npy",
            "abstained_mask": "abstained_mask.npy",
            "coefficient_matrix": "coefficient_matrix.npz",
            "affinity_matrix": "affinity_matrix.npz",
            "candidate_graph": "candidate_graph.npz",
            "metrics": "metrics.json",
            "summary": "summary.json",
        },
    }
    if save_dir is not None:
        output = Path(save_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "predictions.npy", spectral.labels)
        np.save(output / "embedding_final.npy", spectral.embedding)
        np.save(output / "abstained_mask.npy", spectral.abstained)
        if labels_true is not None:
            np.save(output / "labels_true.npy", labels_true)
        _save_topology(topology, output)
        _write_json(config.to_dict(), output / "resolved_config.json")
        _write_json(metrics, output / "metrics.json")
        _write_json(summary, output / "summary.json")
    return spectral.labels, summary


def _load_npz(path: Path) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        csr_keys = {"data", "indices", "indptr", "shape"}
        if csr_keys.issubset(payload.files):
            shape = tuple(int(value) for value in np.asarray(payload["shape"]).reshape(-1))
            X: np.ndarray | sp.csr_matrix = sp.csr_matrix(
                (
                    np.asarray(payload["data"]),
                    np.asarray(payload["indices"], dtype=np.int64),
                    np.asarray(payload["indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
        elif "x" in payload.files:
            X = np.asarray(payload["x"])
        elif "X" in payload.files:
            X = np.asarray(payload["X"])
        else:
            raise ValueError(f"NPZ has no supported input matrix: {path}")
        y = np.asarray(payload["y"]) if "y" in payload.files else None
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoGate V17 topology-native reference solver")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-mode", choices=["auto", "count", "nonnegative", "continuous"], default=None)
    args = parser.parse_args()
    path = Path(args.data_path)
    X, y = _load_npz(path)
    if args.n_clusters is None:
        if y is None:
            raise ValueError("--n-clusters is required when labels are absent")
        n_clusters = int(np.unique(y).size)
        k_protocol = "benchmark_oracle_from_y"
    else:
        n_clusters = int(args.n_clusters)
        k_protocol = "explicit_n_clusters"
    config = load_config(
        args.config,
        {"seed": int(args.seed), "input_mode": args.input_mode},
    )
    _, summary = fit_v17(
        X,
        n_clusters,
        y,
        config=config,
        save_dir=args.save_dir,
        dataset_name=args.dataset_name or path.stem,
        source_path=path,
        k_protocol=k_protocol,
    )
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
