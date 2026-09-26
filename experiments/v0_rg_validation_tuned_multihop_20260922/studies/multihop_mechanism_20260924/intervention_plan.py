"""Frozen, label-free source interventions for the multihop mechanism study.

This module imports the historical sampler but never edits its source tree.
Every source replacement keeps the probability of its original candidate slot.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib

import numpy as np
from sklearn.neighbors import NearestNeighbors

from v0rg_tunable_multihop.hop_sampling import (
    FixedCandidatePlan, _direct_candidate_weights, _directed_exact_shells,
)


ARMS = frozenset({"reference", "h1_matched_amplitude", "replace_far_le4",
                  "kwide_countmatched", "far_shuffle", "rescue50", "rescue100",
                  "mixed_target"})


def _actual_hops(graph, anchor: int, candidates: np.ndarray, max_hop: int = 8) -> np.ndarray:
    shells = _directed_exact_shells(graph.indices, anchor, max_hop)
    lookup = {int(node): hop for hop, nodes in shells.items() for node in nodes}
    return np.asarray([lookup.get(int(node), max_hop + 1) for node in candidates], dtype=np.int8)


def _row_seed(seed: int, row: int, salt: str) -> int:
    data = f"{seed}|{row}|{salt}".encode("ascii")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def _make_plan(reference: FixedCandidatePlan, indices: list[np.ndarray],
               hops: list[np.ndarray], arm: str, extras: dict) -> FixedCandidatePlan:
    for row, (old, new, labels, probs) in enumerate(zip(reference.indices, indices, hops, reference.probs)):
        if old.shape != new.shape or labels.shape != old.shape:
            raise AssertionError(f"candidate slot count changed on row {row}")
        if np.any(new == row):
            raise AssertionError(f"self-source introduced on row {row}")
        if not np.array_equal(probs, reference.probs[row]):
            raise AssertionError("source intervention changed probabilities")
    profile = dict(reference.profile)
    original_unique = [len(np.unique(row)) / len(row) for row in reference.indices]
    result_unique = [len(np.unique(row)) / len(row) for row in indices]
    def effective_sources(rows):
        values = []
        for nodes, probabilities in zip(rows, reference.probs):
            _, inverse = np.unique(nodes, return_inverse=True)
            mass = np.bincount(inverse, weights=probabilities)
            values.append(float(1.0 / np.sum(mass * mass)))
        return float(np.mean(values))
    profile.update({"intervention": arm, "original_hop_variant": reference.variant,
                    "slot_probability_preserved": True, "draw_count_preserved": True,
                    "original_unique_source_fraction": float(np.mean(original_unique)),
                    "intervened_unique_source_fraction": float(np.mean(result_unique)),
                    "original_effective_source_count_mean": effective_sources(reference.indices),
                    "intervened_effective_source_count_mean": effective_sources(indices),
                    **extras})
    return FixedCandidatePlan(tuple(indices), reference.probs, tuple(hops),
                              reference.draw_count, reference.variant, profile)


def _replace_far_le4(reference: FixedCandidatePlan, graph, seed: int,
                     rescue_fraction: float) -> tuple[FixedCandidatePlan, dict]:
    rows, labels, restored, far_total = [], [], 0, 0
    for anchor, (nodes, hops) in enumerate(zip(reference.indices, reference.hop_labels)):
        near_positions = np.flatnonzero(hops <= 4)
        far_positions = np.flatnonzero(hops >= 5)
        if near_positions.size == 0:
            raise ValueError(f"row {anchor} has no <=4-hop donor")
        new_nodes = nodes.copy()
        new_hops = hops.copy()
        donor_rng = np.random.default_rng(_row_seed(seed, anchor, "near-donors"))
        donors = donor_rng.permutation(near_positions)
        rescue_rng = np.random.default_rng(_row_seed(seed, anchor, "rescue-order"))
        rescue_order = rescue_rng.permutation(far_positions)
        # Nested restoration sets: rescue0 subset rescue50 subset rescue100.
        keep_count = int(np.floor(float(rescue_fraction) * len(far_positions)))
        keep = set(int(x) for x in rescue_order[:keep_count])
        for order, position in enumerate(far_positions):
            if int(position) in keep:
                restored += 1
                continue
            donor_position = int(donors[order % len(donors)])
            new_nodes[position] = nodes[donor_position]
            new_hops[position] = hops[donor_position]
        far_total += len(far_positions)
        rows.append(new_nodes)
        labels.append(new_hops)
    arm = "rescue50" if rescue_fraction > 0 else "replace_far_le4"
    plan = _make_plan(reference, rows, labels, arm,
                      {"far_slot_count": far_total, "restored_far_slot_count": restored,
                       "restored_far_slot_fraction": restored / max(1, far_total),
                       "restoration_rule": "floor(half_far_slots_per_row)_sha256_seeded_nested"})
    return plan, {"original_hops": reference.hop_labels,
                  "restored_far_slot_count": restored, "far_slot_count": far_total}


def _kwide(reference: FixedCandidatePlan, graph, edge_weights, config) -> tuple[FixedCandidatePlan, dict]:
    n = len(reference.indices)
    counts = np.asarray([len(row) for row in reference.indices], dtype=np.int64)
    if int(counts.max()) >= n:
        raise ValueError("matched Kwide count requires fewer candidates than rows")
    search = NearestNeighbors(n_neighbors=int(counts.max()) + 1, metric="cosine")
    search.fit(graph.embedding)
    _dist, proposed = search.kneighbors(graph.embedding)
    one_hop_sets = [set(map(int, row)) for row in graph.indices]
    rows, probs, hops = [], [], []
    for anchor in range(n):
        target = int(counts[anchor])
        nearest = proposed[anchor][proposed[anchor] != anchor]
        nodes = np.asarray(nearest[:target], dtype=np.int64)
        if len(nodes) != target or anchor in nodes:
            raise RuntimeError("Kwide count mismatch or self-source")
        weights = _direct_candidate_weights(graph, anchor, nodes, config, one_hop_sets)
        rows.append(nodes)
        probs.append(weights)
        hops.append(_actual_hops(graph, anchor, nodes))
    profile = dict(reference.profile)
    profile.update({"intervention": "kwide_countmatched",
                    "construction": "direct_cosine_knn_same_per_row_candidate_count",
                    "candidate_counts_preserved": True, "draw_count_preserved": True,
                    "slot_probability_preserved": False,
                    "actual_hop_code_9": ">8_or_unreachable",
                    "original_hop_variant": reference.variant})
    plan = FixedCandidatePlan(tuple(rows), tuple(probs), tuple(hops),
                              reference.draw_count, reference.variant, profile)
    return plan, {"original_hops": reference.hop_labels}


def _quantile_bins(values: np.ndarray, count: int = 4) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    return np.quantile(finite, np.linspace(0, 1, count + 1)[1:-1])


def _far_shuffle(reference: FixedCandidatePlan, graph, feature_nonzero: np.ndarray,
                 seed: int) -> tuple[FixedCandidatePlan, dict]:
    """Shuffle far identities within exact 5–8-hop shells when possible."""
    n = len(reference.indices)
    feature_nonzero = np.asarray(feature_nonzero)
    if feature_nonzero.shape != (n,):
        raise ValueError("feature_nonzero must have one value per sample")
    indegree = np.bincount(graph.indices.ravel(), minlength=n)
    degree_bin = np.searchsorted(_quantile_bins(indegree), indegree, side="right")
    sparse_bin = np.searchsorted(_quantile_bins(feature_nonzero), feature_nonzero, side="right")
    # Quantiles come only from graph positions and are frozen before any label access.
    sample_distances = []
    for anchor in np.linspace(0, n - 1, min(n, 256), dtype=np.int64):
        far = reference.indices[anchor][reference.hop_labels[anchor] >= 5]
        if far.size:
            sample_distances.extend((1.0 - graph.embedding[far] @ graph.embedding[anchor]).tolist())
    distance_edges = _quantile_bins(np.asarray(sample_distances or [0.0, 0.5, 1.0]))
    rows, hops_out = [], []
    changed = distance_match = degree_match = sparse_match = hop_match = 0
    no_far = no_alternative = reused_original_candidate = 0
    fallback_counts: dict[str, int] = defaultdict(int)
    for anchor, (nodes, labels) in enumerate(zip(reference.indices, reference.hop_labels)):
        far_positions = np.flatnonzero(labels >= 5)
        if not far_positions.size:
            no_far += 1
            rows.append(nodes.copy()); hops_out.append(labels.copy())
            continue
        rng = np.random.default_rng(_row_seed(seed, anchor, "far-shuffle"))
        new_nodes = nodes.copy()
        new_labels = labels.copy()
        original_set = set(map(int, nodes))
        used_donors: set[int] = set()
        shells = _directed_exact_shells(graph.indices, anchor, 8)
        # Bounded feature-only donor pool; all choices retain an actual far shell.
        candidates = []
        candidate_hops = []
        for hop in range(5, 9):
            pool = shells[hop]
            if pool.size > 512:
                pool = rng.choice(pool, size=512, replace=False)
            candidates.extend(map(int, pool))
            candidate_hops.extend([hop] * len(pool))
        candidate_array = np.asarray(candidates, dtype=np.int64)
        candidate_hops_array = np.asarray(candidate_hops, dtype=np.int8)
        donor_hop_lookup = dict(zip(candidates, candidate_hops))
        donor_distance = (1.0 - graph.embedding[candidate_array] @ graph.embedding[anchor]
                          if candidate_array.size else np.empty(0, dtype=np.float32))
        donor_distance_bin = np.searchsorted(distance_edges, donor_distance, side="right")
        groups: dict[tuple, list[int]] = defaultdict(list)
        order = rng.permutation(len(candidate_array))
        for j in order:
            donor = int(candidate_array[j]); hop = int(candidate_hops_array[j])
            d = int(degree_bin[donor]); s = int(sparse_bin[donor]); b = int(donor_distance_bin[j])
            for tier in ((0, 1) if donor not in original_set else (1,)):
                for key in ((hop, d, s, b), (hop, d, s, -1),
                            (hop, -1, -1, b), (hop, -1, -1, -1),
                            (-1, -1, -1, -1)):
                    groups[(tier,) + key].append(donor)

        def take(key: tuple, original: int) -> int | None:
            group = groups[key]
            skipped = []
            while group:
                donor = group.pop()
                if donor == original:
                    skipped.append(donor)
                    continue
                if donor in used_donors:
                    continue
                group.extend(skipped)
                return donor
            group.extend(skipped)
            return None

        far_distances = 1.0 - graph.embedding[nodes[far_positions]] @ graph.embedding[anchor]
        for offset, position in enumerate(far_positions):
            original = int(nodes[position])
            hop = int(labels[position]); d = int(degree_bin[original]); s = int(sparse_bin[original])
            b = int(np.searchsorted(distance_edges, far_distances[offset], side="right"))
            choices = (
                ("exact_unselected", (0, hop, d, s, b)),
                ("distance_relaxed", (0, hop, d, s, -1)),
                ("attributes_relaxed", (0, hop, -1, -1, b)),
                ("same_shell", (0, hop, -1, -1, -1)),
                ("exact_selected", (1, hop, d, s, b)),
                ("selected_distance_relaxed", (1, hop, d, s, -1)),
                ("selected_attributes_relaxed", (1, hop, -1, -1, b)),
                ("selected_same_shell", (1, hop, -1, -1, -1)),
                ("other_far_shell", (1, -1, -1, -1, -1)),
            )
            donor = None
            for tier_name, key in choices:
                donor = take(key, original)
                if donor is not None:
                    fallback_counts[tier_name] += 1
                    break
            if donor is None:
                no_alternative += 1
                continue
            used_donors.add(donor)
            new_nodes[position] = donor
            donor_hop = int(donor_hop_lookup[donor])
            new_labels[position] = donor_hop
            changed += 1
            hop_match += int(donor_hop == hop)
            degree_match += int(degree_bin[donor] == d)
            sparse_match += int(sparse_bin[donor] == s)
            donor_b = int(np.searchsorted(distance_edges,
                           1.0 - graph.embedding[donor] @ graph.embedding[anchor], side="right"))
            distance_match += int(donor_b == b)
            reused_original_candidate += int(donor in original_set)
        rows.append(new_nodes)
        hops_out.append(new_labels)
    plan = _make_plan(reference, rows, hops_out, "far_shuffle",
                      {"far_slots_shuffled": changed,
                       "far_slots_without_alternative": no_alternative,
                       "same_exact_hop_fraction": hop_match / max(1, changed),
                       "indegree_bin_matched_fraction": degree_match / max(1, changed),
                       "nonzero_bin_matched_fraction": sparse_match / max(1, changed),
                       "distance_bin_matched_fraction": distance_match / max(1, changed),
                       "reused_original_candidate_fraction": reused_original_candidate / max(1, changed),
                       "matching_tier_counts": dict(fallback_counts),
                       "rows_without_far_candidates": no_far,
                       "matching": "same_directed_shell_then_indegree_input_nonzero_and_source_distance_quartiles",
                       "actual_hop_code_9": ">8_or_unreachable",
                       "distance_bin_edges": distance_edges.tolist()})
    return plan, {"original_hops": reference.hop_labels}


def transform_plan(reference: FixedCandidatePlan, graph, edge_weights, config,
                   arm: str, seed: int, feature_nonzero: np.ndarray | None = None
                   ) -> tuple[FixedCandidatePlan, dict]:
    if arm not in ARMS:
        raise ValueError(f"unknown intervention arm {arm!r}")
    if arm in {"reference", "rescue100", "mixed_target"}:
        return reference, {"original_hops": reference.hop_labels}
    if arm == "replace_far_le4":
        return _replace_far_le4(reference, graph, seed, 0.0)
    if arm == "rescue50":
        return _replace_far_le4(reference, graph, seed, 0.5)
    if arm == "kwide_countmatched":
        return _kwide(reference, graph, edge_weights, config)
    if arm == "far_shuffle":
        if feature_nonzero is None:
            raise ValueError("far_shuffle requires feature_nonzero")
        return _far_shuffle(reference, graph, feature_nonzero, seed)
    raise ValueError("H1 matched amplitude uses the original one-hop sampler")
