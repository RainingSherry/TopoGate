from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def sparsemax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
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


def predictive_support(
    heldout_views: list,
    candidate_indices: np.ndarray,
    valid: np.ndarray,
    smoothing: float = 1e-3,
    block_size: int = 128,
) -> tuple[np.ndarray, dict]:
    """Compute held-out donor-vs-background count support.

    The candidate graph is never used to form a target.  Each donor is scored
    against an independent count view, and repeated splits are median-aggregated.
    """
    if not heldout_views:
        raise ValueError("at least one held-out view is required")
    indices = np.asarray(candidate_indices, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    n, width = indices.shape
    all_support: list[np.ndarray] = []
    positive_rates: list[float] = []
    for view in heldout_views:
        counts = view.tocsr().astype(np.float64)
        d = int(counts.shape[1])
        global_counts = np.asarray(counts.sum(axis=0)).ravel()
        global_denominator = float(global_counts.sum() + smoothing * d)
        base_logp = np.log(global_counts + smoothing) - np.log(max(global_denominator, smoothing))
        log_alpha = float(np.log(smoothing))
        # Store only the nonzero correction to log(alpha).  This lets each
        # candidate row use sparse multiplication; absent donor features still
        # contribute the exact log(alpha) background term.
        donor_delta = counts.copy()
        donor_delta.data = np.log(donor_delta.data + smoothing) - log_alpha
        support = np.zeros((n, width), dtype=np.float32)
        for block_start in range(0, n, int(block_size)):
            block_rows = np.arange(block_start, min(block_start + int(block_size), n), dtype=np.int64)
            anchor = counts[block_rows]
            anchor_sums = np.asarray(anchor.sum(axis=1)).ravel()
            base_risk = -np.asarray(anchor.multiply(base_logp).sum(axis=1)).ravel()
            block_valid = valid[block_rows]
            block_donors = np.where(block_valid, indices[block_rows], 0).astype(np.int64)
            # The Kronecker repeat makes one sparse anchor row per candidate;
            # the elementwise product then evaluates all edge overlaps in a
            # block without creating a dense feature matrix.
            anchor_repeated = sp.kron(anchor, np.ones((width, 1), dtype=np.float64), format="csr")
            donor_rows = donor_delta[block_donors.reshape(-1)]
            overlap = np.asarray(anchor_repeated.multiply(donor_rows).sum(axis=1)).ravel()
            donor_totals = np.asarray(counts[block_donors.reshape(-1)].sum(axis=1)).ravel()
            repeated_anchor_sums = np.repeat(anchor_sums, width)
            donor_log_likelihood = (
                repeated_anchor_sums * log_alpha
                + overlap
                - repeated_anchor_sums * np.log(np.maximum(donor_totals + smoothing * d, smoothing))
            )
            block_support = (np.repeat(base_risk, width) + donor_log_likelihood).reshape(-1, width)
            support[block_rows] = np.where(block_valid, block_support, 0.0).astype(np.float32)
        all_support.append(support)
        positive_rates.append(float(np.mean(support[valid] > 0.0)) if valid.any() else 0.0)
    stacked = np.stack(all_support, axis=0)
    median_support = np.median(stacked, axis=0).astype(np.float32)
    return median_support, {
        "support_repeats": int(len(heldout_views)),
        "positive_support_rate": float(np.mean(median_support[valid] > 0.0)) if valid.any() else 0.0,
        "repeat_positive_support_rates": positive_rates,
        "support_median": float(np.median(median_support[valid])) if valid.any() else 0.0,
        "support_p90_abs": float(np.quantile(np.abs(median_support[valid]), 0.9)) if valid.any() else 0.0,
    }


def summarize_gate(pi: np.ndarray) -> dict[str, float]:
    """Summarize null mass and conditional edge usage.

    Effective neighbors is defined on the edge distribution conditional on
    taking a non-null edge.  Null abstention is reported separately instead of
    being folded into the entropy.
    """
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
    if variant in {"self_only", "output_disabled"}:
        pi = np.zeros((q_self.shape[0], candidates.shape[1] + 1), dtype=np.float32)
        pi[:, 0] = 1.0
        return q_self.copy(), pi, np.zeros_like(support, dtype=np.float32)
    if variant == "fixed_predictive_graph":
        scores = valid.astype(np.float32)
    elif variant == "shuffled_support":
        scores = shuffle_support(support, valid, seed)
    elif variant == "V16_predictive_gate":
        scores = np.asarray(support, dtype=np.float32).copy()
    else:
        raise ValueError(f"unknown V16 variant: {variant}")
    scores = scores / float(temperature)
    pi = abstaining_sparsemax(scores, valid)
    q_out = pi[:, :1] * q_self
    for p in range(candidates.shape[1]):
        donors = candidates[:, p]
        edge_q = np.zeros_like(q_self)
        good = valid[:, p]
        edge_q[good] = q_self[donors[good]]
        q_out += pi[:, p + 1 : p + 2] * edge_q
    return q_out, pi, scores
