from __future__ import annotations

import numpy as np
import torch

from methods.TopoGate.learnable_gate.neighbor_graph import NeighborGraph


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
    n_cells, k = graph.indices.shape
    if gate_mode == "none" or k == 0:
        gate = np.zeros(n_cells, dtype=np.float32)
        perturb = np.zeros(n_cells, dtype=np.float32)
    elif gate_mode == "constant":
        gate = np.full(n_cells, float(gate_max), dtype=np.float32)
        perturb = np.zeros(n_cells, dtype=np.float32)
    else:
        mutual_ratio = graph.mutual.mean(axis=1).astype(np.float32)
        snn_avg = graph.snn.mean(axis=1).astype(np.float32)
        perturb = 1.0 - np.sum(graph.probs * graph.similarity, axis=1)
        unc = np.zeros(n_cells, dtype=np.float32) if uncertainty is None else uncertainty.astype(np.float32)
        logits = (
            float(beta_mutual) * mutual_ratio
            + float(beta_snn) * snn_avg
            - float(beta_perturb) * perturb
            - float(beta_uncertainty) * unc
        )
        sig = 1.0 / (1.0 + np.exp(-logits))
        gate = float(gate_min) + (float(gate_max) - float(gate_min)) * sig
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
        "fraction_gate_gt_90pct_max": float(np.mean(gate > 0.9 * float(gate_max))) if gate.size else 0.0,
        "uncertainty_enabled": bool(uncertainty is not None),
        "uncertainty_source": "disabled" if uncertainty is None else "unsupervised",
        "mean_perturb_proxy": float(np.mean(perturb)) if perturb.size else 0.0,
    }
    return gate, sample_weight, summary


def make_pseudo_batch_binary(
    data_np: np.ndarray,
    batch_indices: np.ndarray,
    batch_x: torch.Tensor,
    mix_mode: str,
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    mix_neighbors: int,
    rng: np.random.Generator,
    random_neighbors: np.ndarray | None = None,
    far_neighbors: np.ndarray | None = None,
    neighbor_estimator: str = "current",
    router_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Binary routing version of make_pseudo_batch.

    x' = anchor + r * (neighbor_mean - anchor)
       = (1-r)*anchor + r*neighbor_mean

    where r ∈ [0,1] is the BinaryRouter output (soft during training,
    hard {0,1} during inference).

    When r=0: x' = anchor (pure self-reconstruction)
    When r=1: x' = mixed (topology-aware neighbor blending)

    router_tensor: (batch_size,) torch tensor from BinaryRouter.
        If provided, routing decisions are computed in torch with gradient.
        sample_weight = router_tensor (so nodes that route to anchor
        contribute zero pseudo-loss).
    """
    if neighbor_estimator not in {"current", "uniform_sample", "full"}:
        raise ValueError(f"Unknown neighbor_estimator: {neighbor_estimator!r}")
    use_torch_router = router_tensor is not None
    if use_torch_router:
        r_t = router_tensor.to(dtype=batch_x.dtype, device=batch_x.device).reshape(-1)
        if r_t.shape[0] != batch_x.shape[0]:
            raise ValueError(
                f"router_tensor must be (batch_size,), got {tuple(r_t.shape)}"
            )

    if mix_mode == "none" or graph.indices.shape[1] == 0 or int(mix_neighbors) <= 0:
        zeros = torch.zeros(batch_x.shape[0], dtype=batch_x.dtype, device=batch_x.device)
        return batch_x.detach(), zeros, {"mean_router": 0.0, "mean_perturb_norm": 0.0}

    bsz = int(batch_indices.shape[0])
    k = int(graph.indices.shape[1])
    m = max(1, min(int(mix_neighbors), k))

    if mix_mode in {"random", "far", "fixed"}:
        neighbor_mean = np.empty((bsz, data_np.shape[1]), dtype=np.float32)
        for pos, cell in enumerate(batch_indices):
            if mix_mode == "random" and random_neighbors is not None:
                row = random_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "far" and far_neighbors is not None:
                row = far_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "mutual":
                mask = graph.mutual[cell]
                row = graph.indices[cell][mask] if np.any(mask) else graph.indices[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            else:
                row = graph.indices[cell]
                probs = edge_weights[cell] if mix_mode == "reliability" else graph.probs[cell]
            normalized = probs / np.clip(probs.sum(), 1e-12, None)
            neighbor_mean[pos] = np.sum(data_np[row] * normalized[:, None], axis=0).astype(np.float32)
    else:
        sampled = np.empty((bsz, m), dtype=np.int64)
        weights = np.empty((bsz, m), dtype=np.float32)
        for pos, cell in enumerate(batch_indices):
            if mix_mode == "random" and random_neighbors is not None:
                row = random_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "far" and far_neighbors is not None:
                row = far_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "mutual":
                mask = graph.mutual[cell]
                row = graph.indices[cell][mask] if np.any(mask) else graph.indices[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            else:
                row = graph.indices[cell]
                probs = edge_weights[cell] if mix_mode == "reliability" else graph.probs[cell]
            choices = rng.choice(row.shape[0], size=m, replace=True, p=probs / np.clip(probs.sum(), 1e-12, None))
            sampled[pos] = row[choices]
            picked = probs[choices].astype(np.float32, copy=False)
            if neighbor_estimator == "current":
                weights[pos] = picked / max(float(picked.sum()), 1e-12)
            else:
                weights[pos] = 1.0 / float(m)
        neighbor_expr = data_np[sampled]
        neighbor_mean = np.sum(neighbor_expr * weights[:, :, None], axis=1).astype(np.float32)

    anchor_np = data_np[batch_indices]

    if use_torch_router:
        anchor_t = torch.as_tensor(anchor_np, dtype=batch_x.dtype, device=batch_x.device)
        neighbor_mean_t = torch.as_tensor(neighbor_mean, dtype=batch_x.dtype, device=batch_x.device)
        # x' = anchor + r * (neighbor - anchor) = (1-r)*anchor + r*neighbor
        mixed_t = anchor_t + r_t.unsqueeze(1) * (neighbor_mean_t - anchor_t)
        r_used_np = r_t.detach().cpu().float().numpy()
    else:
        # Fallback: all nodes route to mixed (no routing decision)
        mixed_np = neighbor_mean.astype(np.float32)
        mixed_t = torch.as_tensor(mixed_np, dtype=batch_x.dtype, device=batch_x.device)
        r_used_np = np.ones(bsz, dtype=np.float32)

    perturb = np.linalg.norm(neighbor_mean - anchor_np, axis=1) / (
        np.linalg.norm(anchor_np, axis=1) + 1e-6
    )

    # sample_weight = routing probability; nodes that route to anchor contribute 0
    sample_weight = torch.as_tensor(
        np.clip(r_used_np, 0.0, 1.0),
        dtype=batch_x.dtype,
        device=batch_x.device,
    )

    info = {
        "mean_router": float(np.mean(r_used_np)),
        "mean_perturb_norm": float(np.mean(perturb)),
        "fraction_routed_to_anchor": float(np.mean(r_used_np <= 0.0)),
        "fraction_routed_to_mixed": float(np.mean(r_used_np > 0.0)),
    }
    return mixed_t, sample_weight, info


def make_pseudo_batch(
    data_np: np.ndarray,
    batch_indices: np.ndarray,
    batch_x: torch.Tensor,
    mix_mode: str,
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    node_gate: np.ndarray,
    mix_neighbors: int,
    rng: np.random.Generator,
    random_neighbors: np.ndarray | None = None,
    far_neighbors: np.ndarray | None = None,
    neighbor_estimator: str = "current",
    gate_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build pseudo-mixed batch.

    node_gate: full per-cell (n_cells,) gate array (used for `max` references).
    gate_tensor: optional (batch_size,) torch tensor (with grad) gate values.
        If provided, the mix `(1-g)*anchor + g*neighbor` is computed in torch so
        that gradients flow back through the gate tensor.  The numpy `node_gate`
        is still used for `mean` reference when mix_mode is 'random/far/fixed'.
    """
    if neighbor_estimator not in {"current", "uniform_sample", "full"}:
        raise ValueError(f"Unknown neighbor_estimator: {neighbor_estimator!r}")
    use_torch_gate = gate_tensor is not None
    if use_torch_gate:
        gate_t = gate_tensor.to(dtype=batch_x.dtype, device=batch_x.device).reshape(-1)
        if gate_t.shape[0] != batch_x.shape[0]:
            raise ValueError(
                f"gate_tensor must be (batch_size,), got {tuple(gate_t.shape)}"
            )
    if mix_mode == "none" or graph.indices.shape[1] == 0 or int(mix_neighbors) <= 0:
        zeros = torch.zeros(batch_x.shape[0], dtype=batch_x.dtype, device=batch_x.device)
        return batch_x.detach(), zeros, {"mean_node_gate": 0.0, "mean_perturb_norm": 0.0}

    bsz = int(batch_indices.shape[0])
    k = int(graph.indices.shape[1])
    m = max(1, min(int(mix_neighbors), k))
    gate = np.asarray(node_gate[batch_indices], dtype=np.float32)
    if mix_mode in {"random", "far", "fixed"}:
        gate = np.maximum(gate, float(np.mean(node_gate)) if node_gate.size else 0.1).astype(np.float32)
        neighbor_mean = np.empty((bsz, data_np.shape[1]), dtype=np.float32)
        for pos, cell in enumerate(batch_indices):
            if mix_mode == "random" and random_neighbors is not None:
                row = random_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "far" and far_neighbors is not None:
                row = far_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "mutual":
                mask = graph.mutual[cell]
                row = graph.indices[cell][mask] if np.any(mask) else graph.indices[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            else:
                row = graph.indices[cell]
                probs = edge_weights[cell] if mix_mode == "reliability" else graph.probs[cell]
            normalized = probs / np.clip(probs.sum(), 1e-12, None)
            neighbor_mean[pos] = np.sum(data_np[row] * normalized[:, None], axis=0).astype(np.float32)
    else:
        sampled = np.empty((bsz, m), dtype=np.int64)
        weights = np.empty((bsz, m), dtype=np.float32)
        for pos, cell in enumerate(batch_indices):
            if mix_mode == "random" and random_neighbors is not None:
                row = random_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "far" and far_neighbors is not None:
                row = far_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "mutual":
                mask = graph.mutual[cell]
                row = graph.indices[cell][mask] if np.any(mask) else graph.indices[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            else:
                row = graph.indices[cell]
                probs = edge_weights[cell] if mix_mode == "reliability" else graph.probs[cell]
            choices = rng.choice(row.shape[0], size=m, replace=True, p=probs / np.clip(probs.sum(), 1e-12, None))
            sampled[pos] = row[choices]
            picked = probs[choices].astype(np.float32, copy=False)
            if neighbor_estimator == "current":
                weights[pos] = picked / max(float(picked.sum()), 1e-12)
            else:
                weights[pos] = 1.0 / float(m)

        neighbor_expr = data_np[sampled]
        neighbor_mean = np.sum(neighbor_expr * weights[:, :, None], axis=1).astype(np.float32)
    if mix_mode in {"random", "far", "fixed"} and not use_torch_gate:
        gate = np.maximum(gate, float(np.mean(node_gate)) if node_gate.size else 0.1).astype(np.float32)
    anchor_np = data_np[batch_indices]
    if use_torch_gate:
        anchor_t = torch.as_tensor(anchor_np, dtype=batch_x.dtype, device=batch_x.device)
        neighbor_mean_t = torch.as_tensor(neighbor_mean, dtype=batch_x.dtype, device=batch_x.device)
        if mix_mode in {"random", "far", "fixed"}:
            mean_full = float(np.mean(node_gate)) if node_gate.size else 0.1
            gate_t = torch.maximum(gate_t, torch.tensor(mean_full, dtype=gate_t.dtype, device=gate_t.device))
        mixed_t = (1.0 - gate_t).unsqueeze(1) * anchor_t + gate_t.unsqueeze(1) * neighbor_mean_t
        gate_used_np = gate_t.detach().cpu().float().numpy()
    else:
        mixed = (1.0 - gate[:, None]) * anchor_np + gate[:, None] * neighbor_mean
        mixed_t = torch.as_tensor(mixed, dtype=batch_x.dtype, device=batch_x.device)
        gate_used_np = gate
    anchor_for_perturb = anchor_np
    perturb = np.linalg.norm(neighbor_mean - anchor_for_perturb, axis=1) / (np.linalg.norm(anchor_for_perturb, axis=1) + 1e-6)
    x_prime = mixed_t
    sample_weight = torch.as_tensor(
        np.clip(gate_used_np / max(float(np.max(node_gate)) if node_gate.size else 1.0, 1e-8), 0, 1),
        dtype=batch_x.dtype,
        device=batch_x.device,
    )
    info = {
        "mean_node_gate": float(np.mean(gate_used_np)),
        "mean_perturb_norm": float(np.mean(perturb)),
        "fraction_zero_gate": float(np.mean(gate_used_np <= 0.0)),
    }
    return x_prime, sample_weight, info
