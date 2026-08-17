"""Run the conditional S2 SimpleCut opportunity-confirmation matrix.

S2 is deliberately narrower than a model search.  It reuses the exact selected
graphs from the completed S1 Spectral matrix and trains one small full-graph MLP
with a normalized-cut objective.  Labels are used only to construct the two
diagnostic oracle graphs and for known-K/post-fit metrics; ``train_simplecut``
receives only H0, W, and the seed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.representation_consumer_probe.protocol import (  # noqa: E402
    CONFIG,
    FORBIDDEN_GPU_IDS,
    LEGAL_GPU_POOL,
    STRESS_DATASETS,
    jsonable,
    sha256_array,
    sha256_file,
)
from scripts.representation_consumer_probe.s1_opportunity import (  # noqa: E402
    SEEDS,
    _accuracy_by_optimal_mapping,
    _artifact_hash_manifest,
    _metric_summary,
    _verify_artifact_hashes,
    graph_diagnostics,
    graph_hash,
)


S0_ROOT = ROOT / "result/representation_consumer_probe/S0_freeze"
S1_ROOT = ROOT / "result/representation_consumer_probe/S1_oracle_v2"
DEFAULT_OUTPUT = ROOT / "result/representation_consumer_probe/S2_simple_cut"
S2_PROTOCOL_ID = "representation_consumer_probe_s2_opportunity_simplecut_v1"
S1_PROTOCOL_ID = "representation_consumer_probe_s1_opportunity_spectral_v2"
S2_DATASETS: tuple[str, ...] = ("Baron Human", "Mouse_retina")
S2_ARMS: tuple[str, ...] = ("R", "O_pool", "O_full")
S2_SEEDS: tuple[int, ...] = tuple(SEEDS)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _configure_gpu(physical_gpu: int) -> torch.device:
    physical_gpu = int(physical_gpu)
    if physical_gpu in FORBIDDEN_GPU_IDS or physical_gpu not in LEGAL_GPU_POOL:
        raise ValueError(
            f"GPU {physical_gpu} is not legal; allowed={list(LEGAL_GPU_POOL)}, "
            f"forbidden={list(FORBIDDEN_GPU_IDS)}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    elif visible.split(",") != [str(physical_gpu)]:
        raise RuntimeError(
            f"S2 requires one explicit physical GPU; CUDA_VISIBLE_DEVICES={visible!r}, "
            f"requested={physical_gpu}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S2 GPU execution requires exactly one visible CUDA device")
    return torch.device("cuda:0")


def _sparse_to_torch(graph: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = sp.coo_matrix(graph, dtype=np.float32)
    indices = torch.tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.long, device=device
    )
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, dtype=torch.float32, device=device
    ).coalesce()


class SimpleCutEncoder(torch.nn.Module):
    """The frozen small MLP used only by the S2 mechanism probe."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = CONFIG.encoder_dims) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(v) for v in hidden_dims]]
        layers: list[torch.nn.Module] = []
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(torch.nn.Linear(left, right))
            if index < len(dims) - 2:
                layers.append(torch.nn.ReLU())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        return self.network(h0)


def simplecut_loss(
    z: torch.Tensor,
    graph: torch.Tensor,
    degrees: torch.Tensor,
    *,
    v_min: float = CONFIG.v_min,
    lambda_orth: float = CONFIG.lambda_orth,
    lambda_var: float = CONFIG.lambda_var,
    eps: float = CONFIG.loss_eps,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute normalized graph energy plus non-collapse penalties."""
    weighted_z = degrees[:, None] * z
    lz = weighted_z - torch.sparse.mm(graph, z)
    denominator = torch.sum(z * weighted_z).clamp_min(float(eps))
    cut = torch.sum(z * lz) / denominator
    degree_sum = degrees.sum().clamp_min(float(eps))
    gram = (z.T @ weighted_z) / degree_sum
    identity = torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
    orth = torch.sum((gram - identity) ** 2)
    variance = torch.var(z, dim=0, unbiased=False)
    var_penalty = torch.sum(torch.relu(float(v_min) - variance) ** 2)
    total = cut + float(lambda_orth) * orth + float(lambda_var) * var_penalty
    return total, {
        "loss": total,
        "L_cut": cut,
        "L_orth": orth,
        "L_var": var_penalty,
        "variance_min": variance.min(),
    }


def train_simplecut(
    h0: np.ndarray,
    selected_graph: sp.spmatrix,
    *,
    seed: int,
    device: torch.device,
    epochs: int = CONFIG.epochs,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any]]:
    """Fit SimpleCut without accepting labels or K."""
    _seed_everything(seed)
    values = np.asarray(h0, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("H0 must be a two-dimensional array")
    graph = _sparse_to_torch(selected_graph, device)
    degrees = torch.sparse.sum(graph, dim=1).to_dense()
    inputs = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    model = SimpleCutEncoder(values.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(CONFIG.learning_rate), weight_decay=float(CONFIG.weight_decay)
    )
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        z = model(inputs)
        total, parts = simplecut_loss(z, graph, degrees)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite SimpleCut loss at epoch {epoch}")
        total.backward()
        optimizer.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(parts["loss"].detach().cpu()),
                "L_cut": float(parts["L_cut"].detach().cpu()),
                "L_orth": float(parts["L_orth"].detach().cpu()),
                "L_var": float(parts["L_var"].detach().cpu()),
                "variance_min": float(parts["variance_min"].detach().cpu()),
            }
        )
    model.eval()
    with torch.no_grad():
        embedding = model(inputs).detach().cpu().numpy().astype(np.float32, copy=False)
        final_z = model(inputs)
        final_total, final_parts = simplecut_loss(final_z, graph, degrees)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    fit_meta = {
        "labels_vector_used_in_fit": False,
        "K_used_in_representation": False,
        "input": "H0",
        "graph_input": "selected_graph_W",
        "epochs": int(epochs),
        "encoder_dims": [int(v) for v in CONFIG.encoder_dims],
        "decoder": None,
        "optimizer": CONFIG.optimizer,
        "learning_rate": float(CONFIG.learning_rate),
        "weight_decay": float(CONFIG.weight_decay),
        "final_loss": float(final_total.detach().cpu()),
        "final_L_cut": float(final_parts["L_cut"].detach().cpu()),
        "final_L_orth": float(final_parts["L_orth"].detach().cpu()),
        "final_L_var": float(final_parts["L_var"].detach().cpu()),
        "peak_memory_bytes": peak_memory,
        "device": str(device),
    }
    return embedding, history, fit_meta


def _effective_rank(embedding: np.ndarray) -> float:
    values = np.asarray(embedding, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(values.shape[0], 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    probabilities = eigenvalues / total
    nonzero = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(nonzero * np.log(nonzero))))


def _representation_diagnostics(embedding: np.ndarray, graph: sp.spmatrix) -> dict[str, Any]:
    values = np.asarray(embedding, dtype=np.float64)
    w = sp.csr_matrix(graph, dtype=np.float64)
    degrees = np.asarray(w.sum(axis=1)).ravel()
    lz = degrees[:, None] * values - w.dot(values)
    energy = float(np.sum(values * lz) / max(float(np.sum(values * values)), CONFIG.loss_eps))
    std = np.std(values, axis=0)
    pairwise_sample = values[: min(values.shape[0], 2048)]
    distances = np.linalg.norm(pairwise_sample[:, None, :] - pairwise_sample[None, :, :], axis=2)
    upper = distances[np.triu_indices(distances.shape[0], k=1)] if distances.shape[0] > 1 else np.array([])
    return {
        "effective_rank": _effective_rank(values),
        "dimension_std_min": float(std.min()) if std.size else 0.0,
        "dimension_std_median": float(np.median(std)) if std.size else 0.0,
        "low_variance_dimension_ratio": float(np.mean(std <= 1e-6)) if std.size else 1.0,
        "mean_pairwise_distance_sample": float(np.mean(upper)) if upper.size else 0.0,
        "pairwise_distance_cv_sample": float(np.std(upper) / max(np.mean(upper), 1e-8))
        if upper.size
        else 0.0,
        "graph_energy": energy,
        "finite": bool(np.isfinite(values).all() and np.isfinite(energy)),
    }


def _load_dataset_meta(dataset: str) -> dict[str, Any]:
    decision = json.loads((S0_ROOT / "s0_decision.json").read_text(encoding="utf-8"))
    if decision.get("status") != "adapter_not_estimable":
        raise RuntimeError(f"S2 requires adapter_not_estimable S0, got {decision.get('status')}")
    rows = json.loads((S0_ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))
    row = next((value for value in rows if value.get("dataset") == dataset), None)
    if row is None:
        raise KeyError(f"missing S0 dataset row: {dataset}")
    source = Path(row["source_path"])
    if not source.exists() or sha256_file(source) != row["source_sha256"]:
        raise RuntimeError(f"source preflight mismatch: {dataset}")
    h0_path = Path(row["H0_path"])
    h0 = np.asarray(np.load(h0_path, allow_pickle=False), dtype=np.float32)
    if sha256_array(h0) != row["H0_sha256"]:
        raise RuntimeError(f"H0 hash mismatch: {dataset}")
    with np.load(source, allow_pickle=True) as archive:
        labels = np.asarray(archive["y"]).reshape(-1)
    if labels.size != h0.shape[0]:
        raise RuntimeError(f"H0/labels mismatch: {dataset}")
    return {
        "dataset": dataset,
        "source": source,
        "source_sha256": row["source_sha256"],
        "h0": h0,
        "h0_sha256": sha256_array(h0),
        "labels": labels,
        "K": int(np.unique(labels).size),
    }


def _load_s1_graph(dataset: str, arm: str, seed: int) -> tuple[sp.csr_matrix, sp.csr_matrix, dict[str, Any]]:
    run_dir = S1_ROOT / dataset / f"seed{seed}" / arm
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing S1 source summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed_valid" or summary.get("protocol_id") != S1_PROTOCOL_ID:
        raise RuntimeError(f"S1 source is not valid: {run_dir}")
    selected = sp.load_npz(run_dir / "selected_graph.npz").tocsr().astype(np.float32)
    directed_path = run_dir / "directed_graph.npz"
    directed = sp.load_npz(directed_path).tocsr().astype(np.float32) if directed_path.exists() else selected
    actual_hash = graph_hash(selected)
    if actual_hash != summary.get("graph_hash"):
        raise RuntimeError(f"S1 graph hash mismatch: {run_dir}")
    return directed, selected, {
        "source_run": str(run_dir.resolve()),
        "source_protocol_id": S1_PROTOCOL_ID,
        "source_graph_hash": actual_hash,
    }


def _audit_contract_valid(run_dir: Path, dataset: str, arm: str, seed: int) -> bool:
    summary_path = run_dir / "summary.json"
    audit_path = run_dir / "audit.json"
    config_path = run_dir / "resolved_config.json"
    if not summary_path.exists() or not audit_path.exists() or not config_path.exists():
        return False
    if not _verify_artifact_hashes(run_dir):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return bool(
            summary.get("status") == "completed_valid"
            and summary.get("protocol_id") == S2_PROTOCOL_ID
            and summary.get("dataset") == dataset
            and summary.get("arm") == arm
            and int(summary.get("seed")) == int(seed)
            and audit.get("audit_ok") is True
            and audit.get("protocol_id") == S2_PROTOCOL_ID
            and audit.get("dataset") == dataset
            and audit.get("arm") == arm
            and int(audit.get("seed")) == int(seed)
            and config.get("protocol_id") == S2_PROTOCOL_ID
            and config.get("dataset") == dataset
            and config.get("arm") == arm
            and int(config.get("seed")) == int(seed)
            and config.get("labels_used_during_fit") is False
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _run_one(
    meta: dict[str, Any],
    arm: str,
    seed: int,
    run_dir: Path,
    *,
    device: torch.device,
    physical_gpu: int,
) -> dict[str, Any]:
    dataset = str(meta["dataset"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if _audit_contract_valid(run_dir, dataset, arm, seed):
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary["execution_status"] = "reused"
        return summary
    summary: dict[str, Any] = {
        "project_id": CONFIG.project_id,
        "protocol_id": S2_PROTOCOL_ID,
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "consumer": "SimpleCut",
        "status": "incomplete_compute",
        "execution_status": "queued",
        "source_path": str(meta["source"]),
        "source_sha256": meta["source_sha256"],
        "H0_sha256": meta["h0_sha256"],
        "K": int(meta["K"]),
        "K_source": CONFIG.k_source,
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "labels_vector_used_in_fit": False,
        "K_used_in_representation": False,
        "K_used_in_readout": True,
        "execution_device": "cuda",
        "physical_gpu": int(physical_gpu),
    }
    try:
        directed, selected, graph_meta = _load_s1_graph(dataset, arm, seed)
        start = time.perf_counter()
        embedding, history, fit_meta = train_simplecut(
            meta["h0"], selected, seed=seed, device=device, epochs=CONFIG.epochs
        )
        elapsed = time.perf_counter() - start
        model = KMeans(n_clusters=int(meta["K"]), n_init=20, random_state=int(seed))
        predictions = model.fit_predict(embedding).astype(np.int64, copy=False)
        benchmark_labels = np.asarray(meta["labels"]).reshape(-1)
        metrics = _metric_summary(benchmark_labels, predictions)
        graph_metrics = graph_diagnostics(selected, benchmark_labels)
        graph_metrics.update({"graph_condition": arm, "directed_edge_count": int(directed.nnz)})
        graph_metrics["s1_source_graph_hash"] = graph_meta["source_graph_hash"]
        np.save(run_dir / "embedding.npy", embedding)
        np.save(run_dir / "predictions.npy", predictions)
        np.save(run_dir / "labels_true.npy", benchmark_labels)
        sp.save_npz(run_dir / "directed_graph.npz", directed, compressed=True)
        sp.save_npz(run_dir / "selected_graph.npz", selected, compressed=True)
        _write_history(run_dir / "training_metrics.csv", history)
        if arm.startswith("O_"):
            _write_json(
                run_dir / "oracle_manifest.json",
                {
                    "labels_used": True,
                    "purpose": "diagnostic_only",
                    "method_claim": False,
                    "arm": arm,
                    "source_graph": "reused_from_S1_H0_positive_cosine",
                    "weights_changed_by_oracle": False,
                    "s1_source_graph_hash": graph_meta["source_graph_hash"],
                },
            )
        summary.update(
            {
                "status": "completed_valid",
                "execution_status": "completed",
                "metrics": metrics,
                "graph_hash": graph_hash(selected),
                "s1_source": graph_meta,
                "graph_diagnostics": graph_metrics,
                "fit_metadata": fit_meta,
                "representation_diagnostics": _representation_diagnostics(embedding, selected),
                "elapsed_seconds": float(elapsed),
                "artifacts": {
                    "embedding": str((run_dir / "embedding.npy").resolve()),
                    "predictions": str((run_dir / "predictions.npy").resolve()),
                    "labels_true": str((run_dir / "labels_true.npy").resolve()),
                    "training_metrics": str((run_dir / "training_metrics.csv").resolve()),
                },
            }
        )
    except Exception as exc:  # preserve incomplete computation and continue the matrix
        summary.update(
            {
                "status": "incomplete_compute",
                "execution_status": "incomplete_compute",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    audit = {
        "audit_ok": summary.get("status") == "completed_valid",
        "protocol_id": S2_PROTOCOL_ID,
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "labels_used_in_oracle_graph": bool(arm.startswith("O_")),
        "K_source": CONFIG.k_source,
        "K_used_in_representation": False,
        "s1_graph_reused": True,
    }
    _write_json(run_dir / "audit.json", audit)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "project_id": CONFIG.project_id,
            "protocol_id": S2_PROTOCOL_ID,
            "dataset": dataset,
            "arm": arm,
            "seed": int(seed),
            "K": int(meta["K"]),
            "K_source": CONFIG.k_source,
            "consumer": "SimpleCut",
            "encoder_dims": [int(v) for v in CONFIG.encoder_dims],
            "epochs": int(CONFIG.epochs),
            "learning_rate": float(CONFIG.learning_rate),
            "weight_decay": float(CONFIG.weight_decay),
            "lambda_orth": float(CONFIG.lambda_orth),
            "lambda_var": float(CONFIG.lambda_var),
            "v_min": float(CONFIG.v_min),
            "labels_used_during_fit": False,
            "K_used_in_representation": False,
            "K_used_in_readout": True,
            "s1_graph_protocol_id": S1_PROTOCOL_ID,
            "legal_gpu_pool": list(LEGAL_GPU_POOL),
            "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
            "physical_gpu": int(physical_gpu),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "execution_device": "cuda",
        },
    )
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "artifact_hashes.json", _artifact_hash_manifest(run_dir))
    return summary


def _effect(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(np.mean(array)) if array.size else 0.0
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "values": array.tolist(),
        "mean": mean,
        "std": std,
        "positive_seed_count": int(np.sum(array > 0.0)),
        "negative_seed_count": int(np.sum(array < 0.0)),
        "materiality_delta": 0.03,
        "material_positive": bool(array.size == len(S2_SEEDS) and mean >= 0.03 and np.sum(array > 0.0) >= 2),
        "classification": "material_positive" if mean >= 0.03 and np.sum(array > 0.0) >= 2 else "observed_small",
    }


def _aggregate_dataset(dataset: str, summaries: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    h_pool = _effect(
        summaries["O_pool"][seed]["metrics"]["ARI"] - summaries["R"][seed]["metrics"]["ARI"]
        for seed in S2_SEEDS
    )
    h_full = _effect(
        summaries["O_full"][seed]["metrics"]["ARI"] - summaries["R"][seed]["metrics"]["ARI"]
        for seed in S2_SEEDS
    )
    candidate_gap = _effect(
        summaries["O_full"][seed]["metrics"]["ARI"]
        - summaries["O_pool"][seed]["metrics"]["ARI"]
        for seed in S2_SEEDS
    )
    arms = {
        arm: {
            "ARI": _effect(summaries[arm][seed]["metrics"]["ARI"] for seed in S2_SEEDS),
            "NMI": _effect(summaries[arm][seed]["metrics"]["NMI"] for seed in S2_SEEDS),
            "ACC": _effect(summaries[arm][seed]["metrics"]["ACC"] for seed in S2_SEEDS),
        }
        for arm in S2_ARMS
    }
    return {
        "dataset": dataset,
        "consumer": "SimpleCut",
        "seed_count": len(S2_SEEDS),
        "arms": arms,
        "H_pool": h_pool,
        "H_full": h_full,
        "C_matched_budget_candidate_gap": candidate_gap,
        "within_pool_opportunity": "present" if h_pool["material_positive"] else "absent",
        "full_opportunity": "present" if h_full["material_positive"] else "absent",
        "candidate_gap": "present" if candidate_gap["material_positive"] else "absent",
        "purpose": "conditional_confirmation_of_spectral_negative_or_near_threshold",
        "S_graph_estimable": False,
    }


def summarize(output_dir: Path, datasets: tuple[str, ...] = S2_DATASETS) -> dict[str, Any]:
    manifest_rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    for dataset in datasets:
        by_arm: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in S2_ARMS}
        for seed in S2_SEEDS:
            for arm in S2_ARMS:
                run_dir = output_dir / dataset / f"seed{seed}" / arm
                valid = _audit_contract_valid(run_dir, dataset, arm, seed)
                status = "completed_valid" if valid else "incomplete_compute"
                manifest_rows.append(
                    {
                        "run_key": f"{S2_PROTOCOL_ID}::{dataset}::{arm}::{seed}",
                        "dataset": dataset,
                        "arm": arm,
                        "seed": int(seed),
                        "output_dir": str(run_dir.resolve()),
                        "status": status,
                        "labels_used_during_fit": False,
                    }
                )
                if valid:
                    by_arm[arm][seed] = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if all(len(by_arm[arm]) == len(S2_SEEDS) for arm in S2_ARMS):
            aggregates[dataset] = _aggregate_dataset(dataset, by_arm)
        else:
            aggregates[dataset] = {
                "dataset": dataset,
                "status": "incomplete_compute",
                "completed_rows": {arm: sorted(by_arm[arm]) for arm in S2_ARMS},
            }
    _write_json(output_dir / "s2_manifest.json", manifest_rows)
    _write_json(output_dir / "s2_dataset_aggregates.json", aggregates)
    completed = sum(row["status"] == "completed_valid" for row in manifest_rows)
    expected = len(datasets) * len(S2_ARMS) * len(S2_SEEDS)
    summary = {
        "project_id": CONFIG.project_id,
        "protocol_id": S2_PROTOCOL_ID,
        "status": "completed_valid" if completed == expected else "incomplete_compute",
        "datasets": list(datasets),
        "arms": list(S2_ARMS),
        "seeds": list(S2_SEEDS),
        "expected_run_count": expected,
        "completed_valid_run_count": completed,
        "incomplete_run_count": expected - completed,
        "consumer": "SimpleCut",
        "purpose": "conditional_opportunity_confirmation_only",
        "labels_used_during_fit": False,
        "K_used_in_representation": False,
        "K_used_in_readout": True,
        "S_graph_estimable": False,
        "dataset_aggregates": aggregates,
        "note": "S2 can distinguish a Spectral relaxation miss from absent opportunity; it does not estimate selector or TopoGate gain.",
    }
    _write_json(output_dir / "s2_summary.json", summary)
    _write_json(output_dir / "artifact_hashes.json", _artifact_hash_manifest(output_dir))
    return summary


def run(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    datasets: tuple[str, ...] = S2_DATASETS,
    physical_gpu: int = 2,
) -> dict[str, Any]:
    if not set(datasets).issubset(set(S2_DATASETS)):
        raise ValueError(f"S2 is restricted to {S2_DATASETS}")
    device = _configure_gpu(physical_gpu)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "resolved_config.json",
        {
            "project_id": CONFIG.project_id,
            "protocol_id": S2_PROTOCOL_ID,
            "s1_source": str(S1_ROOT.resolve()),
            "s0_source": str(S0_ROOT.resolve()),
            "datasets": list(datasets),
            "arms": list(S2_ARMS),
            "seeds": list(S2_SEEDS),
            "consumer": "SimpleCut",
            "encoder_dims": [int(v) for v in CONFIG.encoder_dims],
            "epochs": int(CONFIG.epochs),
            "learning_rate": float(CONFIG.learning_rate),
            "weight_decay": float(CONFIG.weight_decay),
            "lambda_orth": float(CONFIG.lambda_orth),
            "lambda_var": float(CONFIG.lambda_var),
            "v_min": float(CONFIG.v_min),
            "labels_used_during_fit": False,
            "K_used_in_representation": False,
            "K_used_in_readout": True,
            "legal_gpu_pool": list(LEGAL_GPU_POOL),
            "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
            "physical_gpu": int(physical_gpu),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "execution_device": "cuda",
            "oracle_non_tuning": True,
        },
    )
    for dataset in datasets:
        meta = _load_dataset_meta(dataset)
        for seed in S2_SEEDS:
            for arm in S2_ARMS:
                _run_one(
                    meta,
                    arm,
                    seed,
                    output_dir / dataset / f"seed{seed}" / arm,
                    device=device,
                    physical_gpu=physical_gpu,
                )
        del meta
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summarize(output_dir, datasets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", action="append", choices=S2_DATASETS)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    selected = tuple(args.dataset) if args.dataset else S2_DATASETS
    result = run(args.output_dir, datasets=selected, physical_gpu=int(args.gpu))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed_valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
