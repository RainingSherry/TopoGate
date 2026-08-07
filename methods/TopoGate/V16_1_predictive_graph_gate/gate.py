from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def sparsemax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must be a 2D array")
    order = np.sort(values, axis=-1)[:, ::-1]
    cssv = np.cumsum(order, axis=-1) - 1.0
    positions = np.arange(1, values.shape[1] + 1, dtype=np.float64)[None, :]
    active = order - cssv / positions > 0.0
    count = np.maximum(active.sum(axis=-1), 1).astype(np.int64)
    tau = np.take_along_axis(cssv, count[:, None] - 1, axis=1) / count[:, None]
    return np.maximum(values - tau, 0.0).astype(np.float32)


def abstaining_sparsemax(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if scores.shape != valid.shape:
        raise ValueError("scores and valid must have identical shapes")
    active = valid & (scores > 0.0)
    masked = np.where(active, scores, -1e9)
    augmented = np.concatenate([np.zeros((scores.shape[0], 1), dtype=np.float32), masked], axis=1)
    probabilities = sparsemax(augmented)
    has_positive = active.any(axis=1)
    probabilities[~has_positive] = 0.0
    probabilities[~has_positive, 0] = 1.0
    return probabilities


def cross_fitted_predictive_support(
    split_views: list[tuple[sp.spmatrix, sp.spmatrix]],
    candidate_indices: np.ndarray,
    valid: np.ndarray,
    *,
    smoothing: float = 1e-3,
    block_size: int = 128,
    exchange_views: bool = True,
) -> tuple[np.ndarray, dict]:
    """Score final candidate edges using donor/background A and anchor B.

    ``split_views`` contains ``(view_a, view_b)`` pairs.  Donor profiles and
    the global background are estimated exclusively from the first view of an
    evaluation, while the anchor likelihood is evaluated exclusively on the
    second.  With the fixed default, both ``(A, B)`` and ``(B, A)`` are scored
    for every split and the edge-wise median is taken over all evaluations.
    This makes the two thinning roles exchangeable without using labels or
    changing the candidate graph.  The reported repeat count remains the
    number of original splits.
    """
    if not split_views:
        raise ValueError("at least one split view is required")
    indices = np.asarray(candidate_indices, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    if indices.shape != valid.shape or indices.ndim != 2:
        raise ValueError("candidate_indices and valid must be identical 2D shapes")
    n, width = indices.shape
    evaluations: list[tuple[sp.spmatrix, sp.spmatrix]] = []
    for view_a, view_b in split_views:
        evaluations.append((view_a, view_b))
        if exchange_views:
            evaluations.append((view_b, view_a))
    all_support: list[np.ndarray] = []
    positive_rates: list[float] = []
    for view_a, view_b in evaluations:
        donor_counts = sp.csr_matrix(view_a, dtype=np.float64)
        anchor_counts = sp.csr_matrix(view_b, dtype=np.float64)
        if donor_counts.shape != anchor_counts.shape or donor_counts.shape[0] != n:
            raise ValueError("split views and candidate graph must have matching shapes")
        d = int(donor_counts.shape[1])
        global_counts = np.asarray(donor_counts.sum(axis=0)).ravel()
        alpha = float(smoothing)
        background_denominator = float(global_counts.sum() + alpha * d)
        background_prob = (global_counts + alpha) / max(background_denominator, alpha)
        log_background = np.log(np.clip(background_prob, 1e-300, None))
        prior_mass = alpha * background_prob
        log_prior = np.log(np.clip(prior_mass, 1e-300, None))
        donor_delta = donor_counts.copy()
        donor_delta.data = np.log(
            np.clip(donor_delta.data + prior_mass[donor_delta.indices], 1e-300, None)
        ) - log_prior[donor_delta.indices]
        donor_totals = np.asarray(donor_counts.sum(axis=1)).ravel()
        support = np.zeros((n, width), dtype=np.float32)
        for block_start in range(0, n, int(block_size)):
            block_end = min(block_start + int(block_size), n)
            block_rows = np.arange(block_start, block_end, dtype=np.int64)
            anchor = anchor_counts[block_rows]
            anchor_totals = np.asarray(anchor.sum(axis=1)).ravel()
            base_log_likelihood = np.asarray(anchor.multiply(log_background).sum(axis=1)).ravel()
            prior_log_likelihood = np.asarray(anchor.multiply(log_prior).sum(axis=1)).ravel()
            block_valid = valid[block_rows]
            block_donors = np.where(block_valid, indices[block_rows], 0).astype(np.int64)
            anchor_repeated = sp.kron(anchor, np.ones((width, 1), dtype=np.float64), format="csr")
            donor_rows = donor_delta[block_donors.reshape(-1)]
            overlap = np.asarray(anchor_repeated.multiply(donor_rows).sum(axis=1)).ravel()
            repeated_anchor_totals = np.repeat(anchor_totals, width)
            repeated_prior = np.repeat(prior_log_likelihood, width)
            donor_log_likelihood = (
                repeated_prior
                + overlap
                - repeated_anchor_totals * np.log(np.maximum(donor_totals[block_donors.reshape(-1)] + alpha, alpha))
            )
            block_support = (
                donor_log_likelihood - np.repeat(base_log_likelihood, width)
            ) / np.maximum(repeated_anchor_totals, 1.0)
            support[block_rows] = np.where(block_valid, block_support.reshape(-1, width), 0.0).astype(np.float32)
        all_support.append(support)
        positive_rates.append(float(np.mean(support[valid] > 0.0)) if valid.any() else 0.0)
    stacked = np.stack(all_support, axis=0)
    median_support = np.median(stacked, axis=0).astype(np.float32)
    row_positive = np.any(median_support > 0.0, axis=1)
    return median_support, {
        "support_definition": "cross_fitted_per_token_log_likelihood_ratio",
        "donor_profile_view": "A_then_B" if exchange_views else "A",
        "anchor_evaluation_view": "B_then_A" if exchange_views else "B",
        "support_repeats": int(len(split_views)),
        "support_evaluations": int(len(evaluations)),
        "view_exchange": bool(exchange_views),
        "positive_support_rate": float(np.mean(median_support[valid] > 0.0)) if valid.any() else 0.0,
        "positive_support_row_rate": float(np.mean(row_positive)) if n else 0.0,
        "repeat_positive_support_rates": positive_rates,
        "support_median": float(np.median(median_support[valid])) if valid.any() else 0.0,
        "support_p90_abs": float(np.quantile(np.abs(median_support[valid]), 0.9)) if valid.any() else 0.0,
    }


def summarize_gate(pi: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(pi, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("pi must be a non-empty 2D probability matrix")
    null_mass = probabilities[:, 0]
    edge = probabilities[:, 1:]
    edge_mass = edge.sum(axis=1) if edge.shape[1] else np.zeros(probabilities.shape[0], dtype=np.float32)
    conditional = np.divide(
        edge,
        edge_mass[:, None],
        out=np.zeros_like(edge),
        where=edge_mass[:, None] > 0.0,
    )
    entropy = (
        -(conditional * np.log(np.clip(conditional, 1e-8, None))).sum(axis=1)
        if edge.shape[1]
        else np.zeros(probabilities.shape[0], dtype=np.float32)
    )
    effective = np.where(edge_mass > 0.0, np.exp(entropy), 1.0)
    return {
        "null_mass": float(np.mean(null_mass)),
        "edge_mass": float(np.mean(edge_mass)),
        "conditional_edge_entropy": float(np.mean(entropy)),
        "effective_neighbors": float(np.mean(effective)),
    }


def shuffle_support(scores: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.asarray(scores, dtype=np.float32).copy()
    for i in range(out.shape[0]):
        positions = np.flatnonzero(valid[i])
        if positions.size > 1:
            out[i, positions] = out[i, rng.permutation(positions)]
    return out


def assignment_readout(
    q_self: np.ndarray,
    candidate_indices: np.ndarray,
    valid: np.ndarray,
    support: np.ndarray,
    *,
    variant: str,
    temperature: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_self = np.asarray(q_self, dtype=np.float32)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    support = np.asarray(support, dtype=np.float32)
    if candidates.shape != valid.shape or candidates.shape != support.shape:
        raise ValueError("candidate_indices, valid, and support must have identical shapes")
    if variant in {"self_only", "output_disabled"}:
        pi = np.zeros((q_self.shape[0], candidates.shape[1] + 1), dtype=np.float32)
        pi[:, 0] = 1.0
        return q_self.copy(), pi, np.zeros_like(support, dtype=np.float32)
    if variant == "fixed_predictive_graph":
        scores = valid.astype(np.float32)
    elif variant == "shuffled_support":
        scores = shuffle_support(support, valid, seed)
    elif variant == "V16_1_predictive_gate":
        scores = support.copy()
    else:
        raise ValueError(f"unknown V16.1 variant: {variant}")
    scores = scores / float(temperature)
    pi = abstaining_sparsemax(scores, valid)
    q_out = pi[:, :1] * q_self
    for position in range(candidates.shape[1]):
        donors = candidates[:, position]
        edge_q = np.zeros_like(q_self)
        good = valid[:, position]
        edge_q[good] = q_self[donors[good]]
        q_out += pi[:, position + 1 : position + 2] * edge_q
    return q_out, pi, scores
