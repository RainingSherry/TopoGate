"""Label-free shell-prior transformations and frozen-graph reconstruction.

The geometric arm keeps the EXACT candidate rows (node identity, order, hop
labels) and the within-shell transition-power weights of the historical plan;
only the per-shell mass changes.  Masses are renormalized over the shells that
are NON-EMPTY for that anchor, exactly as ``build_exact_hop_plan`` does for q.

    flat_q     : w_1 = 1, w_h = q (h >= 2)           -- historical construction
    geometric  : w_h = (1 - alpha) ** (h - 1)        -- truncated PPR-style decay

This is a geometric shell-prior control, NOT a PPR/APPNP implementation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from common import canonical_sha256, sha256_bytes

GEOMETRIC_ALPHA = 0.2
SENSITIVITY_ALPHAS = (0.1, 0.2, 0.5)
WITHIN_TOL = 1e-10


def resolve_ok(hop_variant: str) -> bool:
    from common import resolve_path_variant
    return resolve_path_variant(hop_variant) is not None


def shell_masses(available: Iterable[int], prior: str, q: float = 1.0,
                 alpha: float = GEOMETRIC_ALPHA) -> np.ndarray:
    hops = np.asarray(list(available), dtype=np.int64)
    if hops.size == 0 or np.any(hops < 1):
        raise ValueError("available shells must be a non-empty list of hops >= 1")
    if prior == "flat_q":
        if not q > 0:
            raise ValueError("q must be positive")
        weights = np.where(hops == 1, 1.0, float(q))
    elif prior == "geometric":
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        weights = (1.0 - float(alpha)) ** (hops - 1).astype(np.float64)
    else:
        raise ValueError(f"unknown shell prior {prior!r}")
    weights = weights.astype(np.float64)
    return weights / weights.sum()


def row_decomposition(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict[int, float], np.ndarray]:
    """Return (available hops ascending, shell mass by hop, within-shell weights)."""
    available = np.unique(labels.astype(np.int64))
    within = np.empty_like(probs, dtype=np.float64)
    mass = {}
    for hop in available:
        mask = labels == hop
        total = float(probs[mask].sum())
        if not total > 0:
            raise ValueError("empty shell mass in a non-empty shell")
        mass[int(hop)] = total
        within[mask] = probs[mask] / total
    return available, mass, within


def reshape_plan(reference, prior: str, q: float, alpha: float = GEOMETRIC_ALPHA):
    """Build a plan with identical candidates/within weights and a new shell prior."""
    from v0rg_tunable_multihop.hop_sampling import FixedCandidatePlan

    rows = []
    max_flat_error = 0.0
    for nodes, probs, labels in zip(reference.indices, reference.probs, reference.hop_labels):
        available, mass, within = row_decomposition(np.asarray(probs, dtype=np.float64), labels)
        # Consistency: the reference must itself be flat_q with the locked q.
        flat = shell_masses(available, "flat_q", q=q)
        observed = np.asarray([mass[int(h)] for h in available])
        max_flat_error = max(max_flat_error, float(np.max(np.abs(flat - observed))))
        target = shell_masses(available, prior, q=q, alpha=alpha)
        lookup = dict(zip(available.tolist(), target.tolist()))
        new = within * np.asarray([lookup[int(h)] for h in labels], dtype=np.float64)
        new = new / new.sum()
        rows.append(new)
    if max_flat_error > 1e-9:
        raise ValueError(f"reference plan is not the locked flat-q prior (max error {max_flat_error:.3g})")
    profile = dict(reference.profile)
    profile.update({"shell_prior": prior, "geometric_alpha": float(alpha) if prior == "geometric" else None,
                    "locked_q": float(q), "candidate_rows_shared_with_reference": True,
                    "within_shell_weights_shared_with_reference": True,
                    "reference_flat_q_reconstruction_max_error": max_flat_error,
                    "prior_normalized_over": "nonempty_shells_per_anchor"})
    # Same ndarray objects for nodes and labels: candidate identity by construction.
    return FixedCandidatePlan(reference.indices, tuple(rows), reference.hop_labels,
                              reference.draw_count, reference.variant, profile)


def verify_same_candidates(a, b) -> dict:
    if len(a.indices) != len(b.indices) or a.draw_count != b.draw_count:
        raise AssertionError("plans differ in row count or draw count")
    max_within = 0.0
    for row, (na, nb, la, lb, pa, pb) in enumerate(zip(a.indices, b.indices, a.hop_labels, b.hop_labels,
                                                       a.probs, b.probs)):
        if not (np.array_equal(na, nb) and np.array_equal(la, lb)):
            raise AssertionError(f"candidate identity differs on row {row}")
        _, _, wa = row_decomposition(np.asarray(pa, dtype=np.float64), la)
        _, _, wb = row_decomposition(np.asarray(pb, dtype=np.float64), lb)
        max_within = max(max_within, float(np.max(np.abs(wa - wb))))
    if max_within > WITHIN_TOL:
        raise AssertionError(f"within-shell weights differ (max {max_within:.3g})")
    return {"rows": len(a.indices), "max_within_shell_abs_diff": max_within}


def plan_arrays(plan) -> dict[str, np.ndarray]:
    counts = np.asarray([len(r) for r in plan.indices], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return {"offsets": offsets,
            "nodes": np.concatenate(plan.indices).astype(np.int64),
            "probs": np.concatenate(plan.probs).astype(np.float64),
            "hops": np.concatenate(plan.hop_labels).astype(np.int8),
            "draw_count": np.asarray([plan.draw_count], dtype=np.int64)}


def h1_plan_arrays(graph_indices: np.ndarray, edge_weights: np.ndarray, draw_count: int) -> dict[str, np.ndarray]:
    n, k = graph_indices.shape
    probs = edge_weights / np.clip(edge_weights.sum(axis=1, keepdims=True), 1e-12, None)
    return {"offsets": np.arange(0, n * k + 1, k, dtype=np.int64),
            "nodes": graph_indices.reshape(-1).astype(np.int64),
            "probs": probs.reshape(-1).astype(np.float64),
            "hops": np.ones(n * k, dtype=np.int8),
            "draw_count": np.asarray([draw_count], dtype=np.int64)}


def plan_fingerprints(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    candidate = sha256_bytes(arrays["offsets"].tobytes() + arrays["nodes"].tobytes() + arrays["hops"].tobytes())
    return {"candidate_sha256": candidate,
            "probability_sha256": sha256_bytes(arrays["probs"].tobytes()),
            "draw_count": int(arrays["draw_count"][0])}


def expected_shell_mass(arrays: dict[str, np.ndarray], max_hop: int = 8) -> dict[str, float]:
    """Mean over anchors of the plan's shell probability (prior expectation, not usage)."""
    n = len(arrays["offsets"]) - 1
    anchors = np.repeat(np.arange(n), np.diff(arrays["offsets"]))
    result = {}
    for hop in range(1, max_hop + 1):
        mask = arrays["hops"] == hop
        per_anchor = np.bincount(anchors[mask], weights=arrays["probs"][mask], minlength=n)
        result[str(hop)] = float(per_anchor.mean())
    return result


def source_prior_statistics(reference, q: float, alphas=SENSITIVITY_ALPHAS) -> dict:
    """Training-free alpha sensitivity over the exact same candidate shells."""
    out = {"flat_q": expected_shell_mass(plan_arrays(reference))}
    for alpha in alphas:
        plan = reshape_plan(reference, "geometric", q, alpha)
        arrays = plan_arrays(plan)
        stats = expected_shell_mass(arrays)
        n = len(arrays["offsets"]) - 1
        anchors = np.repeat(np.arange(n), np.diff(arrays["offsets"]))
        far = arrays["hops"] >= 2
        far_mass = np.bincount(anchors[far], weights=arrays["probs"][far], minlength=n)
        ess = 1.0 / np.bincount(anchors, weights=arrays["probs"] ** 2, minlength=n)
        out[f"geometric_alpha_{alpha}"] = {"shell_mass": stats, "far_mass_mean": float(far_mass.mean()),
                                           "far_mass_p10": float(np.quantile(far_mass, .1)),
                                           "far_mass_p90": float(np.quantile(far_mass, .9)),
                                           "effective_candidates_mean": float(ess.mean())}
    return out


def build_frozen_graph(fit_X: np.ndarray, config, seed: int) -> dict:
    """Replicate trainer.fit_predict graph/gate construction (lines before training).

    Verified against the trainer's own diagnostics in ``run_factorial refcheck``.
    """
    from v0rg_tunable_multihop.graph import build_pca_knn_graph, compute_edge_reliability
    from v0rg_tunable_multihop.mixing import compute_node_gate

    data = np.ascontiguousarray(np.asarray(fit_X, dtype=np.float32))
    gate_min, gate_max = config.gate_bounds()
    graph = build_pca_knn_graph(data, k=min(int(config.neighbor_k), int(data.shape[0]) - 1),
                                pca_dim=config.knn_pca_dim, tau=config.tau, seed=int(seed))
    _rel, edge_weights, _summary = compute_edge_reliability(
        graph, mode=config.edge_reliability_mode, gamma_sim=config.gamma_sim,
        gamma_mutual=config.gamma_mutual, gamma_snn=config.gamma_snn,
        gamma_distance=config.gamma_distance)
    node_gate, _, _ = compute_node_gate(
        graph, edge_weights=edge_weights, gate_mode="topology", gate_min=gate_min, gate_max=gate_max,
        beta_mutual=config.beta_mutual, beta_snn=config.beta_snn, beta_perturb=config.beta_perturb,
        beta_uncertainty=config.beta_uncertainty, uncertainty=None)
    return {"graph": graph, "edge_weights": edge_weights, "node_gate": node_gate}


def arm_config(locked_multihop_config: dict, arm: str) -> dict:
    """Frozen run config for an arm; H1 differs from the locked config ONLY in hop fields."""
    from v0rg_tunable_multihop.config import V0_RGConfig
    from dataclasses import asdict

    base = V0_RGConfig(**locked_multihop_config)
    if arm == "H1_anchor":
        base = replace(base, hop_variant="H1", high_hop_probability_multiplier=1.0)
    return asdict(base)


def config_difference(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


__all__ = ["shell_masses", "reshape_plan", "verify_same_candidates", "plan_arrays", "h1_plan_arrays",
           "plan_fingerprints", "expected_shell_mass", "source_prior_statistics", "build_frozen_graph",
           "arm_config", "config_difference", "canonical_sha256", "GEOMETRIC_ALPHA", "SENSITIVITY_ALPHAS"]
