from __future__ import annotations

import numpy as np
import torch

from .graph import NeighborGraph


def compute_node_gate(
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    gate_mode: str,
    gate_min: float,
    gate_max: float,
    beta_mutual: float,
    beta_snn: float,
    beta_perturb: float,
    beta_uncertainty: float,
    uncertainty: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reproduce the original analytic topology node gate without labels."""
    del edge_weights  # Kept in the signature for numerical/API conformance with RG.
    n_samples, k = graph.indices.shape
    if gate_mode == "none" or k == 0:
        gate = np.zeros(n_samples, dtype=np.float32)
        perturbation = np.zeros(n_samples, dtype=np.float32)
    elif gate_mode == "constant":
        gate = np.full(n_samples, float(gate_max), dtype=np.float32)
        perturbation = np.zeros(n_samples, dtype=np.float32)
    else:
        mutual_ratio = graph.mutual.mean(axis=1).astype(np.float32)
        snn_average = graph.snn.mean(axis=1).astype(np.float32)
        perturbation = 1.0 - np.sum(graph.probs * graph.similarity, axis=1)
        unsupervised_uncertainty = (
            np.zeros(n_samples, dtype=np.float32)
            if uncertainty is None
            else np.asarray(uncertainty, dtype=np.float32)
        )
        logits = (
            float(beta_mutual) * mutual_ratio
            + float(beta_snn) * snn_average
            - float(beta_perturb) * perturbation
            - float(beta_uncertainty) * unsupervised_uncertainty
        )
        sigmoid = 1.0 / (1.0 + np.exp(-logits))
        gate = float(gate_min) + (float(gate_max) - float(gate_min)) * sigmoid
        gate = gate.astype(np.float32)
    sample_weight = np.clip(gate / max(float(gate_max), 1e-8), 0.0, 1.0).astype(np.float32)
    summary = {
        "gate_mode": gate_mode,
        "gate_min": float(gate_min),
        "gate_max": float(gate_max),
        "mean_node_gate": float(np.mean(gate)) if gate.size else 0.0,
        "min_node_gate": float(np.min(gate)) if gate.size else 0.0,
        "max_node_gate": float(np.max(gate)) if gate.size else 0.0,
        "fraction_gate_lt_0p01": float(np.mean(gate < 0.01)) if gate.size else 1.0,
        "fraction_gate_gt_90pct_max": (
            float(np.mean(gate > 0.9 * float(gate_max))) if gate.size else 0.0
        ),
        "uncertainty_enabled": bool(uncertainty is not None),
        "uncertainty_source": "disabled" if uncertainty is None else "unsupervised",
        "mean_perturb_proxy": float(np.mean(perturbation)) if perturbation.size else 0.0,
    }
    return gate, sample_weight, summary


def make_pseudo_batch(
    data_np: np.ndarray,
    batch_indices: np.ndarray,
    batch_x: torch.Tensor,
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    node_gate: np.ndarray,
    mix_neighbors: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Sample reliability-weighted neighbors using the original current estimator."""
    if graph.indices.shape[1] == 0 or int(mix_neighbors) <= 0:
        zeros = torch.zeros(batch_x.shape[0], dtype=batch_x.dtype, device=batch_x.device)
        return batch_x.detach(), zeros, {"mean_node_gate": 0.0, "mean_perturb_norm": 0.0}
    batch_size = int(batch_indices.shape[0])
    k = int(graph.indices.shape[1])
    sampled_count = max(1, min(int(mix_neighbors), k))
    sampled = np.empty((batch_size, sampled_count), dtype=np.int64)
    weights = np.empty((batch_size, sampled_count), dtype=np.float32)
    for position, sample in enumerate(batch_indices):
        row = graph.indices[sample]
        probs = edge_weights[sample]
        choices = rng.choice(
            row.shape[0],
            size=sampled_count,
            replace=True,
            p=probs / np.clip(probs.sum(), 1e-12, None),
        )
        sampled[position] = row[choices]
        selected = probs[choices].astype(np.float32, copy=False)
        weights[position] = selected / max(float(selected.sum()), 1e-12)
    neighbor_expression = data_np[sampled]
    neighbor_mean = np.sum(neighbor_expression * weights[:, :, None], axis=1).astype(np.float32)
    gate = np.asarray(node_gate[batch_indices], dtype=np.float32)
    anchor = data_np[batch_indices]
    mixed = (1.0 - gate[:, None]) * anchor + gate[:, None] * neighbor_mean
    perturbation = np.linalg.norm(neighbor_mean - anchor, axis=1) / (
        np.linalg.norm(anchor, axis=1) + 1e-6
    )
    x_prime = torch.as_tensor(mixed, dtype=batch_x.dtype, device=batch_x.device)
    sample_weight = torch.as_tensor(
        np.clip(
            gate / max(float(np.max(node_gate)) if node_gate.size else 1.0, 1e-8),
            0.0,
            1.0,
        ),
        dtype=batch_x.dtype,
        device=batch_x.device,
    )
    return x_prime.detach(), sample_weight, {
        "mean_node_gate": float(np.mean(gate)),
        "mean_perturb_norm": float(np.mean(perturbation)),
        "fraction_zero_gate": float(np.mean(gate <= 0.0)),
    }
