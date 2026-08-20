"""Exact-budget support/value corruption arms and the simple label oracle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from . import protocol


@dataclass(frozen=True)
class OracleScores:
    class_probabilities: np.ndarray
    class_sizes: np.ndarray
    labels: np.ndarray
    metadata: dict[str, Any]

    def scores_for_row(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        label = int(self.labels[int(row)])
        own = self.class_probabilities[label]
        other_mask = np.arange(self.class_probabilities.shape[0]) != label
        other = np.average(self.class_probabilities[other_mask], axis=0, weights=self.class_sizes[other_mask])
        return own - other, other - own


def build_simple_label_oracle(x: sp.csr_matrix, y: np.ndarray) -> OracleScores:
    """Build class-conditional support scores without involving model state.

    Sources are features characteristic of the sample's true class; destinations
    are absent features characteristic of another class.  The score is an
    intentionally simple label oracle, not a deployable selector.
    """
    support = x.copy().tocsr()
    support.data = np.ones_like(support.data, dtype=np.float32)
    classes = np.unique(y)
    if not np.array_equal(classes, np.arange(classes.size)):
        raise ValueError("V26 oracle requires contiguous encoded labels")
    class_sizes = np.bincount(y, minlength=int(classes.size)).astype(np.float32)
    probabilities = np.empty((classes.size, x.shape[1]), dtype=np.float32)
    for row, label in enumerate(classes):
        counts = np.asarray(support[y == label].sum(axis=0)).ravel().astype(np.float32)
        probabilities[row] = (counts + 1.0) / (class_sizes[int(label)] + 2.0)
    return OracleScores(
        class_probabilities=probabilities,
        class_sizes=class_sizes,
        labels=np.asarray(y, dtype=np.int64),
        metadata={
            "kind": "class_conditional_support_profile_v1",
            "smoothing": "Laplace(alpha=1)",
            "other_class_aggregation": "class-size-weighted mean over all non-own classes",
            "labels_used_for_mask_selection": True,
            "model_state_used": False,
            "n_classes": int(classes.size),
        },
    )


def _seed(seed: int, epoch: int, row: int) -> int:
    return int((seed * 1_000_003 + epoch * 9_176 + row * 97_409) % (2**63 - 1))


def _pair_count(active_count: int, width: int) -> int:
    inactive_count = int(width - active_count)
    return int(min(int(np.ceil(protocol.CORRUPTION_RATE * active_count)), active_count // 2, inactive_count))


def _top(candidates: np.ndarray, scores: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    jitter = rng.uniform(0.0, 1e-12, size=candidates.size)
    order = np.argsort(-(scores[candidates] + jitter), kind="stable")
    return np.asarray(candidates[order[:count]], dtype=np.int64)


def corrupt_batch(
    clean: np.ndarray,
    global_rows: np.ndarray,
    *,
    arm: str,
    seed: int,
    epoch: int,
    oracle: OracleScores | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a row-wise swap corruption while preserving every row value multiset."""
    value = np.asarray(clean, dtype=np.float32)
    if arm == "CLEAN":
        return value.copy(), {"pair_count_total": 0, "changed_coordinate_total": 0, "support_crossing_total": 0}
    if arm not in protocol.ARMS:
        raise ValueError(f"unknown V26 arm: {arm}")
    if arm == "O_LABEL_ORACLE" and oracle is None:
        raise ValueError("label oracle scores are required for O_LABEL_ORACLE")
    corrupted = value.copy()
    pair_total = 0
    changed_total = 0
    crossing_total = 0
    width = value.shape[1]
    for local_row, global_row in enumerate(np.asarray(global_rows, dtype=np.int64)):
        rng = np.random.default_rng(_seed(seed, epoch, int(global_row)))
        active = np.flatnonzero(value[local_row] != 0.0)
        inactive = np.flatnonzero(value[local_row] == 0.0)
        pair_count = _pair_count(int(active.size), width)
        if pair_count <= 0:
            continue
        if arm == "P0_RANDOM":
            picked = rng.choice(np.arange(width, dtype=np.int64), size=2 * pair_count, replace=False)
            source, destination = picked[:pair_count], picked[pair_count:]
        elif arm == "P1_SUPPORT_PRESERVE":
            picked = rng.choice(active, size=2 * pair_count, replace=False)
            source, destination = picked[:pair_count], picked[pair_count:]
        elif arm == "P2_SUPPORT_TARGET":
            source = rng.choice(active, size=pair_count, replace=False)
            destination = rng.choice(inactive, size=pair_count, replace=False)
        else:
            assert oracle is not None
            source_score, destination_score = oracle.scores_for_row(int(global_row))
            source = _top(active, source_score, pair_count, rng)
            destination = _top(inactive, destination_score, pair_count, rng)
        before_source = corrupted[local_row, source].copy()
        before_destination = corrupted[local_row, destination].copy()
        corrupted[local_row, source] = before_destination
        corrupted[local_row, destination] = before_source
        pair_total += int(pair_count)
        changed_total += int(np.count_nonzero(before_source != before_destination) * 2)
        crossing_total += int(np.count_nonzero((before_source == 0.0) != (before_destination == 0.0)))
    return corrupted, {
        "pair_count_total": int(pair_total),
        "changed_coordinate_total": int(changed_total),
        "support_crossing_total": int(crossing_total),
        "value_multiset_preserved": True,
    }
