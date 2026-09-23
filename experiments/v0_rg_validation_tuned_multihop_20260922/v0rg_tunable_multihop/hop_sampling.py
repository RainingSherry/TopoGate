from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from .config import HOP_VARIANTS, V0_RGConfig
from .graph import NeighborGraph


@dataclass(frozen=True)
class FixedCandidatePlan:
    """Variable-width, label-free candidate rows for a fixed-hop experiment.

    The original :class:`NeighborGraph` is never replaced.  This object only
    controls which samples enter the auxiliary reconstruction view.
    """

    indices: tuple[np.ndarray, ...]
    probs: tuple[np.ndarray, ...]
    hop_labels: tuple[np.ndarray, ...]
    draw_count: int
    variant: str
    profile: dict

    def __post_init__(self) -> None:
        if self.variant == "H1":
            raise ValueError("H1 must use the legacy sampler directly")
        if not self.indices or len(self.indices) != len(self.probs) or len(self.indices) != len(self.hop_labels):
            raise ValueError("candidate rows must be non-empty and aligned")
        if int(self.draw_count) < 1:
            raise ValueError("draw_count must be positive")
        for nodes, probabilities, labels in zip(self.indices, self.probs, self.hop_labels):
            if nodes.ndim != 1 or probabilities.ndim != 1 or labels.ndim != 1:
                raise ValueError("candidate rows must be one-dimensional")
            if nodes.size == 0 or nodes.size != probabilities.size or nodes.size != labels.size:
                raise ValueError("candidate row has an invalid width")
            if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
                raise ValueError("candidate probabilities must be finite and positive")
            if not np.isclose(float(probabilities.sum()), 1.0, rtol=1e-6, atol=1e-8):
                raise ValueError("candidate probabilities must sum to one")


def _transition_matrix(graph: NeighborGraph, edge_weights: np.ndarray) -> csr_matrix:
    n_samples, k = graph.indices.shape
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), k)
    return csr_matrix(
        (
            np.asarray(edge_weights, dtype=np.float64).reshape(-1),
            (rows, np.asarray(graph.indices, dtype=np.int64).reshape(-1)),
        ),
        shape=(n_samples, n_samples),
        dtype=np.float64,
    )


def _directed_exact_shells(indices: np.ndarray, source: int, max_hop: int) -> dict[int, np.ndarray]:
    """Return directed shortest-path shells without padding or self nodes."""
    n_samples = int(indices.shape[0])
    seen = np.zeros(n_samples, dtype=bool)
    seen[int(source)] = True
    frontier = np.asarray([source], dtype=np.int64)
    shells: dict[int, np.ndarray] = {}
    for hop in range(1, int(max_hop) + 1):
        reachable = indices[frontier].reshape(-1) if frontier.size else np.empty(0, dtype=np.int64)
        reachable = reachable[reachable != int(source)]
        if reachable.size:
            new_nodes = np.unique(reachable[~seen[reachable]])
        else:
            new_nodes = np.empty(0, dtype=np.int64)
        seen[new_nodes] = True
        shells[hop] = new_nodes.astype(np.int64, copy=False)
        frontier = new_nodes
    return shells


def _rank_shell(nodes: np.ndarray, scores: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray, float]:
    if nodes.size == 0:
        return nodes.astype(np.int64), scores.astype(np.float64), 1.0
    if np.any(~np.isfinite(scores)) or np.any(scores <= 0.0):
        raise ValueError("exact-shell path scores must be finite and positive")
    order = np.lexsort((nodes, -scores))
    keep = order[: min(int(limit), nodes.size)]
    retained = float(scores[keep].sum() / scores.sum())
    return nodes[keep].astype(np.int64, copy=False), scores[keep].astype(np.float64, copy=False), retained


def resolve_path_variant(hop_variant: str) -> tuple[int, int]:
    """Resolve legacy and expanded path variants into depth and shell width.

    ``H12`` and ``H123`` remain first-class aliases so that their historical
    plans are bitwise reproducible by the generalized implementation.
    """
    legacy = {"H12": (2, 1), "H123": (3, 1)}
    if hop_variant in legacy:
        return legacy[hop_variant]
    matched = re.fullmatch(r"H(2|3|4|5|6|8)_B(1|2|4)", str(hop_variant))
    if matched is None:
        raise ValueError("path variants must be H12, H123, or H{2,3,4,5,6,8}_B{1,2,4}")
    return int(matched.group(1)), int(matched.group(2))


def build_exact_hop_plan(
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    hop_variant: str,
    mix_neighbors: int,
    high_hop_probability_multiplier: float = 1.0,
) -> FixedCandidatePlan:
    """Build fixed path candidates from exact shells of the frozen graph."""
    max_hop, shell_width_multiplier = resolve_path_variant(hop_variant)
    if graph.indices.shape != edge_weights.shape:
        raise ValueError("edge_weights must align with the frozen one-hop graph")
    n_samples, base_k = graph.indices.shape
    if base_k == 0:
        raise ValueError("a multihop plan requires a non-empty one-hop graph")
    transition = _transition_matrix(graph, edge_weights)
    draw_count = max(1, min(int(mix_neighbors), int(base_k)))
    rows: list[np.ndarray] = []
    row_probs: list[np.ndarray] = []
    row_hops: list[np.ndarray] = []
    shell_counts: dict[int, list[int]] = {hop: [] for hop in range(1, max_hop + 1)}
    retained_mass: dict[int, list[float]] = {hop: [] for hop in range(2, max_hop + 1)}

    for source in range(n_samples):
        shells = _directed_exact_shells(graph.indices, source, max_hop)
        selected: dict[int, tuple[np.ndarray, np.ndarray]] = {
            1: (
                np.asarray(graph.indices[source], dtype=np.int64).copy(),
                np.asarray(edge_weights[source], dtype=np.float64).copy(),
            )
        }
        shell_counts[1].append(int(selected[1][0].size))
        walk = transition.getrow(source)
        for hop in range(2, max_hop + 1):
            walk = (walk @ transition).tocsr()
            nodes = shells[hop]
            score_map = dict(zip(walk.indices.tolist(), walk.data.tolist()))
            scores = np.asarray([score_map.get(int(node), 0.0) for node in nodes], dtype=np.float64)
            picked_nodes, picked_scores, retained = _rank_shell(
                nodes, scores, int(base_k) * int(shell_width_multiplier)
            )
            selected[hop] = (picked_nodes, picked_scores)
            shell_counts[hop].append(int(picked_nodes.size))
            retained_mass[hop].append(retained)

        available = [hop for hop in range(1, max_hop + 1) if selected[hop][0].size]
        if not available:
            raise RuntimeError("the original one-hop graph unexpectedly produced an empty candidate row")
        shell_multipliers = np.asarray(
            [1.0 if hop == 1 else float(high_hop_probability_multiplier) for hop in available],
            dtype=np.float64,
        )
        shell_multipliers /= np.clip(float(shell_multipliers.sum()), 1e-12, None)
        row_nodes: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for hop, shell_mass in zip(available, shell_multipliers):
            nodes, score = selected[hop]
            normalized = score / np.clip(float(score.sum()), 1e-12, None)
            normalized_sum = float(normalized.sum())
            if not np.isclose(normalized_sum, 1.0, rtol=1e-10, atol=1e-12):
                if normalized_sum <= 0.0 or not np.isfinite(normalized_sum):
                    raise ValueError("exact-shell scores cannot be normalized")
                normalized = normalized / normalized_sum
            row_nodes.append(nodes)
            probabilities.append(shell_mass * normalized)
            labels.append(np.full(nodes.size, hop, dtype=np.int8))
        rows.append(np.concatenate(row_nodes))
        row_probs.append(np.concatenate(probabilities))
        row_hops.append(np.concatenate(labels))

    profile = {
        "variant": hop_variant,
        "construction": "directed_exact_shortest_path_shells",
        "path_score": "frozen_edge_weight_transition_power",
        "base_neighbor_k": int(base_k),
        "max_hop": int(max_hop),
        "shell_width_multiplier": int(shell_width_multiplier),
        "high_hop_probability_multiplier": float(high_hop_probability_multiplier),
        "fixed_draw_count": int(draw_count),
        "mean_candidate_count": float(np.mean([row.size for row in rows])),
        "max_candidate_count": int(max(row.size for row in rows)),
        "shell_mean_selected_count": {str(hop): float(np.mean(values)) for hop, values in shell_counts.items()},
        "shell_nonempty_fraction": {str(hop): float(np.mean(np.asarray(values) > 0)) for hop, values in shell_counts.items()},
        "path_mass_retained_after_topk": {
            str(hop): float(np.mean(values)) for hop, values in retained_mass.items()
        },
        "global_padding": False,
        "node_gate_recomputed": False,
        "auxiliary_weight_recomputed": False,
    }
    return FixedCandidatePlan(tuple(rows), tuple(row_probs), tuple(row_hops), draw_count, hop_variant, profile)


def _direct_candidate_weights(
    graph: NeighborGraph,
    source: int,
    nodes: np.ndarray,
    config: V0_RGConfig,
    one_hop_sets: list[set[int]],
) -> np.ndarray:
    similarity = np.asarray(graph.embedding[nodes] @ graph.embedding[source], dtype=np.float64)
    distance = 1.0 - similarity
    scaled = similarity / max(float(config.tau), 1e-12)
    scaled -= float(np.max(scaled))
    base_probability = np.exp(scaled)
    reliability = np.ones_like(base_probability)
    mode = config.edge_reliability_mode
    if mode in {"sim", "sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
        reliability *= np.exp(float(config.gamma_sim) * similarity)
    if mode in {"sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
        mutual = np.asarray([source in one_hop_sets[int(node)] for node in nodes], dtype=np.float64)
        reliability *= 1.0 + float(config.gamma_mutual) * mutual
    if mode in {"sim_mutual_snn", "sim_mutual_snn_distance"}:
        source_set = one_hop_sets[source]
        snn = np.asarray(
            [
                len(source_set.intersection(one_hop_sets[int(node)]))
                / float(max(1, len(source_set.union(one_hop_sets[int(node)]))))
                for node in nodes
            ],
            dtype=np.float64,
        )
        reliability *= 1.0 + float(config.gamma_snn) * snn
    if mode == "sim_mutual_snn_distance":
        reliability *= np.exp(-float(config.gamma_distance) * distance)
    score = base_probability * np.clip(reliability, 1e-6, 1e6)
    return score / np.clip(float(score.sum()), 1e-12, None)


def build_kwide_plan(
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    reference_plan: FixedCandidatePlan,
    config: V0_RGConfig,
    hop_variant: str,
) -> FixedCandidatePlan:
    """Build a direct-geometric candidate control with per-row matched width."""
    if hop_variant not in {"Kwide2", "Kwide3"}:
        raise ValueError("build_kwide_plan supports Kwide2 and Kwide3 only")
    n_samples, base_k = graph.indices.shape
    target_counts = np.asarray([row.size for row in reference_plan.indices], dtype=np.int64)
    if target_counts.shape != (n_samples,) or np.any(target_counts < base_k):
        raise ValueError("reference plan must preserve every original one-hop candidate")
    nearest = NearestNeighbors(n_neighbors=min(int(target_counts.max()) + 1, n_samples), metric="cosine")
    nearest.fit(graph.embedding)
    _, proposed = nearest.kneighbors(graph.embedding)
    one_hop_sets = [set(map(int, row)) for row in graph.indices]
    rows: list[np.ndarray] = []
    row_probs: list[np.ndarray] = []
    row_hops: list[np.ndarray] = []
    for source in range(n_samples):
        target_count = int(target_counts[source])
        if target_count == base_k:
            nodes = np.asarray(graph.indices[source], dtype=np.int64).copy()
            probabilities = np.asarray(edge_weights[source], dtype=np.float64).copy()
        else:
            candidates = proposed[source][proposed[source] != source]
            nodes = np.asarray(candidates[:target_count], dtype=np.int64)
            if nodes.size != target_count:
                raise RuntimeError("direct-geometric control could not supply the matched candidate count")
            probabilities = _direct_candidate_weights(graph, source, nodes, config, one_hop_sets)
        rows.append(nodes)
        row_probs.append(probabilities)
        row_hops.append(np.zeros(nodes.size, dtype=np.int8))
    profile = {
        "variant": hop_variant,
        "construction": "direct_cosine_knn_matched_per_row_candidate_count",
        "reference_variant": reference_plan.variant,
        "base_neighbor_k": int(base_k),
        "fixed_draw_count": int(reference_plan.draw_count),
        "mean_candidate_count": float(np.mean(target_counts)),
        "max_candidate_count": int(target_counts.max()),
        "global_padding": False,
        "node_gate_recomputed": False,
        "auxiliary_weight_recomputed": False,
    }
    return FixedCandidatePlan(tuple(rows), tuple(row_probs), tuple(row_hops), reference_plan.draw_count, hop_variant, profile)


def estimate_neighbors_from_plan(
    data: np.ndarray,
    batch_indices: np.ndarray,
    plan: FixedCandidatePlan,
    rng: np.random.Generator,
    estimator: str,
) -> tuple[np.ndarray, dict]:
    """Apply the legacy estimator to variable-width fixed-hop candidate rows."""
    if estimator not in {"current", "uniform_sample", "full"}:
        raise ValueError("unknown neighbor estimator")
    output = np.empty((len(batch_indices), data.shape[1]), dtype=np.float32)
    mass = defaultdict(float)
    draw_hits = defaultdict(int)
    total_draws = 0
    known_hops = sorted({int(hop) for row in plan.hop_labels for hop in np.unique(row)})
    per_sample_mass = {hop: np.zeros(len(batch_indices), dtype=np.float64) for hop in known_hops}
    for position, source in enumerate(batch_indices):
        nodes = plan.indices[int(source)]
        probabilities = plan.probs[int(source)]
        labels = plan.hop_labels[int(source)]
        if estimator == "full":
            output[position] = np.sum(data[nodes] * probabilities[:, None], axis=0)
            for hop in known_hops:
                contribution = float(probabilities[labels == hop].sum())
                mass[hop] += contribution
                per_sample_mass[hop][position] = contribution
            continue
        choices = rng.choice(nodes.size, size=plan.draw_count, replace=True, p=probabilities)
        selected_probability = probabilities[choices]
        weights = (
            selected_probability / max(float(selected_probability.sum()), 1e-12)
            if estimator == "current"
            else np.full(plan.draw_count, 1.0 / float(plan.draw_count), dtype=np.float64)
        )
        output[position] = np.sum(data[nodes[choices]] * weights[:, None], axis=0)
        selected_hops = labels[choices]
        for hop in known_hops:
            mask = selected_hops == hop
            contribution = float(weights[mask].sum())
            mass[hop] += contribution
            per_sample_mass[hop][position] = contribution
            draw_hits[hop] += int(mask.sum())
        total_draws += int(plan.draw_count)
    normalizer = max(1, len(batch_indices))
    return output, {
        "actual_sampling_mass": {str(hop): float(mass[hop] / normalizer) for hop in known_hops},
        "draw_fraction": {
            str(hop): (None if estimator == "full" else float(draw_hits[hop] / max(1, total_draws)))
            for hop in known_hops
        },
        "_per_sample_actual_sampling_mass": {str(hop): values for hop, values in per_sample_mass.items()},
    }


def make_pseudo_batch_from_plan(
    data_np: np.ndarray,
    batch_indices: np.ndarray,
    batch_x,
    node_gate: np.ndarray,
    plan: FixedCandidatePlan,
    rng: np.random.Generator,
    neighbor_estimator: str = "current",
    auxiliary_weighting: str = "gate",
    mix_amplitude_multiplier: float = 1.0,
):
    """Construct the auxiliary view while preserving V0_RG's mixing semantics."""
    import torch

    if auxiliary_weighting not in {"gate", "uniform"}:
        raise ValueError("unknown auxiliary_weighting")
    neighbor_mean, usage = estimate_neighbors_from_plan(
        data_np, batch_indices, plan, rng, neighbor_estimator
    )
    gate = np.asarray(node_gate[batch_indices], dtype=np.float32)
    anchor = data_np[batch_indices]
    mix_gate = np.clip(gate * float(mix_amplitude_multiplier), 0.0, 1.0).astype(np.float32)
    mixed = (1.0 - mix_gate[:, None]) * anchor + mix_gate[:, None] * neighbor_mean
    perturbation = np.linalg.norm(neighbor_mean - anchor, axis=1) / (
        np.linalg.norm(anchor, axis=1) + 1e-6
    )
    actual_displacement = np.linalg.norm(mixed - anchor, axis=1)
    sample_weight = torch.as_tensor(
        np.clip(gate / max(float(np.max(node_gate)) if node_gate.size else 1.0, 1e-8), 0.0, 1.0),
        dtype=batch_x.dtype,
        device=batch_x.device,
    )
    if auxiliary_weighting == "uniform":
        sample_weight = torch.ones_like(sample_weight)
    per_sample_mass = usage.pop("_per_sample_actual_sampling_mass")
    gate_weighted_mass = {
        hop: float(np.mean(mix_gate * values)) for hop, values in per_sample_mass.items()
    }
    return torch.as_tensor(mixed, dtype=batch_x.dtype, device=batch_x.device).detach(), sample_weight, {
        "mean_node_gate": float(np.mean(gate)),
        "mean_mix_strength": float(np.mean(mix_gate)),
        "mix_amplitude_multiplier": float(mix_amplitude_multiplier),
        "mean_perturb_norm": float(np.mean(perturbation)),
        "mean_actual_displacement": float(np.mean(actual_displacement)),
        "fraction_zero_gate": float(np.mean(gate <= 0.0)),
        "hop_usage": usage,
        "gate_weighted_hop_mass": gate_weighted_mass,
    }
