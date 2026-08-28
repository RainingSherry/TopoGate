from __future__ import annotations

import json
import os
import platform
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import V18Config
from .graph import CandidateGraph, build_candidate_graph, shuffle_candidate_graph
from .relation import (
    EdgeGate,
    SparseRelation,
    affinity_from_coefficients,
    group_huber,
    initialize_relation_fista,
    relation_profile,
)
from .scmae import MaskedAutoencoder, masked_view
from .spectral import ReadoutResult, leiden_readout, normalized_spectral_readout


VARIANTS = {
    "scmae_only",
    "latent_candidate_spectral",
    "latent_C_exactzero",
    "latent_GW_frozen",
    "v18_full",
    "v18_shuffled_E0",
    "v18_no_recurrence",
    "v18_no_stability",
    "v18_mask04",
    "v18_leiden",
}


@dataclass(frozen=True)
class V18Result:
    predictions: np.ndarray
    embedding: np.ndarray
    abstained: np.ndarray
    latent_final: np.ndarray
    latent_mae: np.ndarray
    candidates: CandidateGraph | None
    coefficients: sp.csr_matrix | None
    affinity: sp.csr_matrix | None
    summary: dict[str, Any]


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    # torch.manual_seed() also seeds all visible CUDA devices. Use an explicit
    # CPU generator and seed only the selected CUDA device for the project GPU
    # isolation contract.
    cpu_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    torch.random.set_rng_state(cpu_generator.get_state())
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(config: V18Config) -> torch.device:
    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V18 requested CUDA but CUDA is unavailable")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            physical_ids = {item.strip() for item in visible.split(",") if item.strip()}
            if physical_ids.intersection({"0", "7"}):
                raise RuntimeError("V18 CUDA_VISIBLE_DEVICES includes forbidden physical GPU 0 or 7")
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return _resolve_device(replace(config, device="cuda"))
    return torch.device("cpu")


def _fixed_views(X: np.ndarray, config: V18Config) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tensor = torch.as_tensor(X, dtype=torch.float32)
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for view_index in range(config.n_views):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.seed) + 1009 * (view_index + 1))
        corrupted, mask = masked_view(tensor, config.mask_ratio, generator)
        values.append(corrupted.numpy().astype(np.float32, copy=False))
        masks.append(mask.numpy().astype(np.float32, copy=False))
    return values, masks


@torch.no_grad()
def _encode_numpy(model: MaskedAutoencoder, arrays: list[np.ndarray], device: torch.device, batch_size: int) -> list[np.ndarray]:
    model.eval()
    output: list[np.ndarray] = []
    for array in arrays:
        chunks: list[np.ndarray] = []
        for start in range(0, array.shape[0], batch_size):
            batch = torch.as_tensor(array[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model.encode(batch).detach().cpu().numpy().astype(np.float32))
        output.append(np.concatenate(chunks, axis=0) if chunks else np.zeros((0, model.hidden_size), dtype=np.float32))
    return output


def _train_mae(model: MaskedAutoencoder, X: np.ndarray, config: V18Config, device: torch.device) -> list[dict[str, float]]:
    model.train()
    loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32)), batch_size=config.batch_size,
                        shuffle=True, drop_last=False, generator=torch.Generator().manual_seed(config.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_mae)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs_mae):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.seed) + 7001 + epoch)
        total = 0.0
        batches = 0
        for (x_cpu,) in loader:
            x = x_cpu.to(device)
            corrupted, mask = masked_view(x_cpu, config.mask_ratio, generator)
            optimizer.zero_grad(set_to_none=True)
            _, loss, parts = model.loss_mask(corrupted.to(device), x, mask.to(device), return_parts=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite scMAE loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        history.append({"stage": "mae", "epoch": epoch + 1, "mae_loss": total / max(1, batches)})
    return history


def _normalize_edge_features(graph: CandidateGraph) -> tuple[np.ndarray, dict[str, Any]]:
    values = graph.features.copy().astype(np.float32)
    valid = graph.valid
    means = np.zeros(values.shape[-1], dtype=np.float32)
    scales = np.ones(values.shape[-1], dtype=np.float32)
    for feature_index in range(values.shape[-1]):
        observed = values[..., feature_index][valid]
        if observed.size:
            means[feature_index] = float(np.mean(observed))
            scales[feature_index] = max(float(np.std(observed)), 1e-6)
            values[..., feature_index][valid] = (observed - means[feature_index]) / scales[feature_index]
    values[~valid] = 0.0
    return values, {"means": means.tolist(), "scales": scales.tolist(), "normalization": "valid-edge-zscore"}


def _topology_terms(
    H_views: list[torch.Tensor],
    indices: torch.Tensor,
    valid: torch.Tensor,
    features: torch.Tensor,
    relation: SparseRelation,
    gate_module: EdgeGate,
    *,
    rows: torch.Tensor,
    temperature: float,
    lambda_gate: float,
    lambda_w: float,
    lambda_l2: float,
    huber_delta: float,
    sample_gate: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    local_features = features[rows]
    gate, expected, _ = gate_module(local_features, temperature=temperature, sample=sample_gate)
    local_valid = valid[rows]
    gate = gate * local_valid.to(dtype=gate.dtype)
    expected = expected * local_valid.to(dtype=expected.dtype)
    local_w = relation.W[rows]
    coefficients = gate * local_w * local_valid.to(dtype=local_w.dtype)
    donors = indices[rows].clamp_min(0)
    reconstruction_losses: list[torch.Tensor] = []
    for H in H_views:
        target = H[rows]
        donor_values = H[donors]
        reconstruction = torch.sum(coefficients.unsqueeze(-1) * donor_values, dim=1)
        reconstruction_losses.append(group_huber(target - reconstruction, huber_delta))
    reconstruction_loss = torch.stack(reconstruction_losses).mean() if reconstruction_losses else coefficients.sum() * 0.0
    valid_count = local_valid.sum().clamp_min(1)
    open_loss = expected.sum() / valid_count
    l1_loss = torch.abs(local_w[local_valid]).mean() if torch.any(local_valid) else local_w.sum() * 0.0
    l2_loss = local_w[local_valid].square().mean() if torch.any(local_valid) else local_w.sum() * 0.0
    topology = reconstruction_loss + float(lambda_w) * l1_loss + 0.5 * float(lambda_l2) * l2_loss
    topology = topology + float(lambda_gate) * open_loss
    return topology, {
        "self_expression": reconstruction_loss.detach(),
        "gate_open": open_loss.detach(),
        "w_l1": l1_loss.detach(),
        "w_l2": l2_loss.detach(),
    }, coefficients


def _proximal_relation(relation: SparseRelation, *, step: float, lambda_w: float) -> None:
    with torch.no_grad():
        valid = relation.valid
        relation.W[valid] = torch.sign(relation.W[valid]) * torch.relu(torch.abs(relation.W[valid]) - float(step) * float(lambda_w))
        relation.W[~valid] = 0.0


def _train_gate_stage(
    model: MaskedAutoencoder,
    relation: SparseRelation,
    gate_module: EdgeGate,
    graph: CandidateGraph,
    z_views: list[np.ndarray],
    features_np: np.ndarray,
    config: V18Config,
    device: torch.device,
    *,
    epochs: int,
    stage: str,
) -> list[dict[str, float]]:
    del model
    H_views = [F.normalize(torch.as_tensor(z, dtype=torch.float32, device=device), dim=1) for z in z_views]
    indices = torch.as_tensor(graph.indices, dtype=torch.long, device=device)
    valid = torch.as_tensor(graph.valid, dtype=torch.bool, device=device)
    features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    rows = torch.arange(graph.n_nodes, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam([
        {"params": list(gate_module.parameters()), "lr": config.lr_gate},
        {"params": [relation.W], "lr": config.lr_relation},
    ])
    history: list[dict[str, float]] = []
    gate_module.train()
    for epoch in range(epochs):
        fraction = (epoch + 1) / max(1, epochs)
        temperature = config.gate_temperature_start + fraction * (config.gate_temperature_end - config.gate_temperature_start)
        lambda_gate = config.lambda_gate * fraction
        optimizer.zero_grad(set_to_none=True)
        topology, parts, _ = _topology_terms(
            H_views, indices, valid, features, relation, gate_module, rows=rows,
            temperature=temperature, lambda_gate=lambda_gate, lambda_w=config.lambda_w,
            lambda_l2=config.lambda_l2, huber_delta=config.huber_delta, sample_gate=True,
        )
        if not torch.isfinite(topology):
            raise FloatingPointError(f"non-finite topology loss in {stage}")
        topology.backward()
        optimizer.step()
        _proximal_relation(relation, step=config.lr_relation, lambda_w=config.lambda_w)
        history.append({
            "stage": stage, "epoch": epoch + 1, "topology_loss": float(topology.detach().cpu()),
            "self_expression": float(parts["self_expression"].cpu()),
            "gate_open": float(parts["gate_open"].cpu()), "temperature": float(temperature),
            "lambda_gate": float(lambda_gate),
        })
    return history


def _variance_floor(z: torch.Tensor, target_std: float = 0.05) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.sum() * 0.0
    return torch.relu(float(target_std) - torch.std(z, dim=0)).mean()


def _train_joint(
    model: MaskedAutoencoder,
    relation: SparseRelation,
    gate_module: EdgeGate,
    graph: CandidateGraph,
    view_inputs: list[np.ndarray],
    view_masks: list[np.ndarray],
    z0_views: list[np.ndarray],
    X: np.ndarray,
    features_np: np.ndarray,
    config: V18Config,
    device: torch.device,
) -> list[dict[str, float]]:
    dataset = TensorDataset(torch.arange(X.shape[0], dtype=torch.long))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    indices = torch.as_tensor(graph.indices, dtype=torch.long, device=device)
    valid = torch.as_tensor(graph.valid, dtype=torch.bool, device=device)
    features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    z0 = [torch.as_tensor(z, dtype=torch.float32, device=device) for z in z0_views]
    parameters: list[dict[str, Any]] = [
        {"params": list(model.parameters()), "lr": config.lr_encoder_joint},
        {"params": list(gate_module.parameters()), "lr": config.lr_gate},
        {"params": [relation.W], "lr": config.lr_relation},
    ]
    optimizer = torch.optim.Adam(parameters)
    history: list[dict[str, float]] = []
    model.train()
    gate_module.train()
    for epoch in range(config.epochs_joint):
        fraction = (epoch + 1) / max(1, config.epochs_joint)
        temperature = config.gate_temperature_start + fraction * (config.gate_temperature_end - config.gate_temperature_start)
        lambda_gate = config.lambda_gate * fraction
        totals = {"loss": 0.0, "mae": 0.0, "topology": 0.0, "anchor": 0.0, "variance": 0.0, "open": 0.0}
        batches = 0
        for (rows_cpu,) in loader:
            rows = rows_cpu.to(device)
            x = torch.as_tensor(X[rows_cpu.numpy()], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            gate, expected, _ = gate_module(features[rows], temperature=temperature, sample=True)
            local_valid = valid[rows]
            gate = gate * local_valid.to(gate.dtype)
            expected = expected * local_valid.to(expected.dtype)
            local_w = relation.W[rows]
            coefficients = gate * local_w * local_valid.to(local_w.dtype)
            mae_terms: list[torch.Tensor] = []
            topology_terms: list[torch.Tensor] = []
            anchor_terms: list[torch.Tensor] = []
            variance_terms: list[torch.Tensor] = []
            for view_index, (inputs_np, mask_np) in enumerate(zip(view_inputs, view_masks, strict=True)):
                corrupted = torch.as_tensor(inputs_np[rows_cpu.numpy()], dtype=torch.float32, device=device)
                mask = torch.as_tensor(mask_np[rows_cpu.numpy()], dtype=torch.float32, device=device)
                z_target, mae_loss = model.loss_mask(corrupted, x, mask)
                donors = indices[rows].clamp_min(0)
                donor_input = torch.as_tensor(inputs_np[donors.detach().cpu().numpy()], dtype=torch.float32, device=device)
                donor_z = model.encode(donor_input.reshape(-1, X.shape[1])).reshape(rows.shape[0], graph.width, -1)
                target_h = F.normalize(z_target, dim=1)
                donor_h = F.normalize(donor_z, dim=2)
                reconstructed = torch.sum(coefficients.unsqueeze(-1) * donor_h, dim=1)
                topology_terms.append(group_huber(target_h - reconstructed, config.huber_delta))
                mae_terms.append(mae_loss)
                anchor_terms.append(F.mse_loss(z_target, z0[view_index][rows]))
                variance_terms.append(_variance_floor(z_target))
            mae_loss = torch.stack(mae_terms).mean()
            self_expression = torch.stack(topology_terms).mean()
            open_loss = expected.sum() / local_valid.sum().clamp_min(1)
            w_l1 = torch.abs(local_w[local_valid]).mean() if torch.any(local_valid) else local_w.sum() * 0.0
            w_l2 = local_w[local_valid].square().mean() if torch.any(local_valid) else local_w.sum() * 0.0
            topology = self_expression + config.lambda_w * w_l1 + 0.5 * config.lambda_l2 * w_l2 + lambda_gate * open_loss
            anchor = torch.stack(anchor_terms).mean()
            variance = torch.stack(variance_terms).mean()
            total = mae_loss + config.lambda_topo * topology + config.lambda_anchor * anchor + config.lambda_var * variance
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite joint V18 loss")
            total.backward()
            optimizer.step()
            _proximal_relation(relation, step=config.lr_relation, lambda_w=config.lambda_w)
            for name, value in (("loss", total), ("mae", mae_loss), ("topology", topology), ("anchor", anchor), ("variance", variance), ("open", open_loss)):
                totals[name] += float(value.detach().cpu())
            batches += 1
        history.append({"stage": "joint", "epoch": epoch + 1, **{key: value / max(1, batches) for key, value in totals.items()},
                        "temperature": float(temperature), "lambda_gate": float(lambda_gate)})
    return history


def _slot_to_sparse(graph: CandidateGraph, values: np.ndarray, epsilon: float) -> sp.csr_matrix:
    keep = graph.valid & (np.abs(values) > float(epsilon))
    rows, slots = np.where(keep)
    cols = graph.indices[rows, slots]
    matrix = sp.csr_matrix((values[rows, slots], (rows, cols)), shape=(graph.n_nodes, graph.n_nodes), dtype=np.float32)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return matrix


def _save_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _readout(affinity: sp.csr_matrix, n_clusters: int | None, config: V18Config, variant: str) -> ReadoutResult:
    if variant == "v18_leiden":
        return leiden_readout(affinity, resolution=config.leiden_resolution)
    if n_clusters is None:
        raise ValueError(f"n_clusters is required for {variant}")
    return normalized_spectral_readout(affinity, n_clusters, seed=config.seed,
                                       n_init=config.spectral_n_init, degree_epsilon=config.degree_epsilon)


def fit_v18(
    X: np.ndarray,
    n_clusters: int | None,
    *,
    config: V18Config | None = None,
    variant: str = "v18_full",
    save_dir: str | Path | None = None,
    dataset_name: str = "adhoc",
    source_path: str | Path | None = None,
) -> V18Result:
    """Fit V18 without labels; benchmark labels are handled by the runner after return."""
    config = config or V18Config()
    if variant not in VARIANTS:
        raise ValueError(f"unknown V18 variant: {variant}")
    if variant == "v18_leiden" and n_clusters is not None and n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if variant != "v18_leiden" and (n_clusters is None or n_clusters <= 0):
        raise ValueError(f"n_clusters is required and must be positive for {variant}")
    device = _resolve_device(config)
    _set_seed(config.seed, device)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    effective_config = replace(config, mask_ratio=0.40) if variant == "v18_mask04" else config
    model = MaskedAutoencoder(X.shape[1], effective_config.hidden_size, effective_config.dropout,
                              effective_config.masked_data_weight, effective_config.mask_loss_weight).to(device)
    mae_history = _train_mae(model, X, effective_config, device)
    view_inputs, view_masks = _fixed_views(X, effective_config)
    z0_views = _encode_numpy(model, view_inputs, device, effective_config.batch_size)
    latent_mae = _encode_numpy(model, [X], device, effective_config.batch_size)[0]
    history = list(mae_history)
    if variant == "scmae_only":
        latent_final = latent_mae
        readout = KMeans(n_clusters=n_clusters, n_init=effective_config.spectral_n_init, random_state=effective_config.seed).fit_predict(latent_final)
        result = V18Result(
            readout.astype(np.int64), latent_final, np.zeros(X.shape[0], dtype=bool), latent_final, latent_mae,
            None, None, None,
            {"status": "ok", "protocol_id": effective_config.protocol_id, "variant": variant,
             "device": str(device), "n_clusters": int(n_clusters), "K_used_only_in_readout": True,
             "labels_used_during_fit": False, "history": history},
        )
        return _persist_result(result, effective_config, dataset_name, source_path, save_dir, n_clusters)

    graph = build_candidate_graph(tuple(z0_views), k=effective_config.candidate_k, width=effective_config.candidate_width)
    if variant == "v18_shuffled_E0":
        graph = shuffle_candidate_graph(graph, seed=effective_config.seed + 17, views=tuple(z0_views))
    features_np, feature_profile = _normalize_edge_features(graph)
    if variant == "v18_no_recurrence":
        features_np[..., 3] = 0.0
    if variant == "v18_no_stability":
        features_np[..., 4] = 0.0
    initial_w = initialize_relation_fista(tuple(z0_views), graph, lambda_l1=effective_config.lambda_w,
                                          lambda_l2=effective_config.lambda_l2, max_iter=effective_config.solver_max_iter,
                                          tolerance=effective_config.solver_tolerance)
    relation = SparseRelation(graph, initial_w).to(device)
    gate_module = EdgeGate(
        len(graph.profile["feature_names"]), init_bias=effective_config.gate_init_bias,
        gamma=effective_config.gate_gamma, zeta=effective_config.gate_zeta,
    ).to(device)

    if variant == "latent_candidate_spectral":
        slot_values = np.where(graph.valid, graph.features[..., 0], 0.0).astype(np.float32)
        coefficients = _slot_to_sparse(graph, slot_values, effective_config.coefficient_epsilon)
        affinity = affinity_from_coefficients(coefficients)
    elif variant == "latent_C_exactzero":
        slot_values = np.where(graph.valid & (np.abs(initial_w) > effective_config.coefficient_epsilon), initial_w, 0.0)
        coefficients = _slot_to_sparse(graph, slot_values, effective_config.coefficient_epsilon)
        affinity = affinity_from_coefficients(coefficients)
    else:
        if variant in {"latent_GW_frozen", "v18_full", "v18_shuffled_E0", "v18_no_recurrence", "v18_no_stability", "v18_mask04", "v18_leiden"}:
            history.extend(_train_gate_stage(model, relation, gate_module, graph, z0_views, features_np, effective_config, device,
                                             epochs=effective_config.epochs_gate, stage="gate_frozen"))
        if variant in {"v18_full", "v18_shuffled_E0", "v18_no_recurrence", "v18_no_stability", "v18_mask04", "v18_leiden"}:
            history.extend(_train_joint(model, relation, gate_module, graph, view_inputs, view_masks, z0_views, X,
                                        features_np, effective_config, device))
        model.eval()
        final_views = _encode_numpy(model, view_inputs, device, effective_config.batch_size)
        latent_final = _encode_numpy(model, [X], device, effective_config.batch_size)[0]
        features_t = torch.as_tensor(features_np, dtype=torch.float32, device=device)
        gate_module.eval()
        with torch.no_grad():
            hard_gate, expected_gate, _ = gate_module(features_t, temperature=effective_config.gate_temperature_end, sample=False)
        slot_values = (hard_gate * relation.W).detach().cpu().numpy().astype(np.float32)
        slot_values[~graph.valid] = 0.0
        coefficients = _slot_to_sparse(graph, slot_values, effective_config.coefficient_epsilon)
        affinity = affinity_from_coefficients(coefficients)
        gate_stats = {
            "expected_open_rate": float(expected_gate[torch.as_tensor(graph.valid, device=device)].mean().cpu()) if graph.n_edges else 0.0,
            "hard_open_rate": float(hard_gate[torch.as_tensor(graph.valid, device=device)].mean().cpu()) if graph.n_edges else 0.0,
            "zero_outgoing_row_fraction": float(np.mean(np.diff(coefficients.indptr) == 0)),
            "temperature_final": float(effective_config.gate_temperature_end),
        }

    if variant in {"latent_candidate_spectral", "latent_C_exactzero"}:
        latent_final = latent_mae
        gate_stats = {"expected_open_rate": None, "hard_open_rate": None, "zero_outgoing_row_fraction": float(np.mean(np.diff(coefficients.indptr) == 0))}
    readout = _readout(affinity, n_clusters, effective_config, variant)
    summary = {
        "status": readout.profile.get("status", "ok"),
        "protocol_id": effective_config.protocol_id,
        "variant": variant,
        "dataset": dataset_name,
        "source_path": None if source_path is None else str(Path(source_path).resolve()),
        "seed": int(effective_config.seed),
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]), "hidden_size": int(effective_config.hidden_size),
        "n_clusters": None if variant == "v18_leiden" else int(n_clusters),
        "K_used_only_in_readout": variant != "v18_leiden", "labels_used_during_fit": False,
        "device": str(device), "graph": graph.profile, "feature_normalization": feature_profile,
        "relation": relation_profile(graph, slot_values, epsilon=effective_config.coefficient_epsilon),
        "gate": gate_stats, "readout": readout.profile, "history": history,
        "output_semantics": {"predictions": "spectral or Leiden labels; -1 means topology abstention",
                              "latent_final": "final scMAE encoder output on clean X",
                              "coefficient_matrix": "candidate-restricted C = hard G elementwise W",
                              "affinity_matrix": "abs(C)+abs(C.T)"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
    }
    result = V18Result(readout.labels, readout.embedding, readout.abstained, latent_final, latent_mae,
                       graph, coefficients, affinity, summary)
    return _persist_result(result, effective_config, dataset_name, source_path, save_dir, n_clusters)


def _persist_result(result: V18Result, config: V18Config, dataset_name: str, source_path: str | Path | None,
                    save_dir: str | Path | None, n_clusters: int | None) -> V18Result:
    if save_dir is None:
        return result
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "predictions.npy", result.predictions)
    np.save(output / "embedding_final.npy", result.embedding)
    np.save(output / "latent_final.npy", result.latent_final)
    np.save(output / "latent_mae.npy", result.latent_mae)
    np.save(output / "abstained_mask.npy", result.abstained)
    if result.candidates is not None:
        np.savez_compressed(output / "candidate_graph.npz", indices=result.candidates.indices,
                            features=result.candidates.features, valid=result.candidates.valid)
        np.savez_compressed(output / "gate_relation_slots.npz", indices=result.candidates.indices,
                            valid=result.candidates.valid)
    if result.coefficients is not None:
        sp.save_npz(output / "coefficient_matrix.npz", result.coefficients, compressed=True)
    if result.affinity is not None:
        sp.save_npz(output / "affinity_matrix.npz", result.affinity, compressed=True)
    _save_json(config.to_dict(), output / "resolved_config.json")
    _save_json(result.summary, output / "summary.json")
    return result
