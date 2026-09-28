"""t-order neighborhood expansion for V0_RG graphs (Top1 multihop experiment).

Design (2026-09-18):
- S_t(i) = { v : dist(i, v) <= t, v != i } computed by BFS propagation over the
  1-hop kNN directed adjacency defined by ``NeighborGraph.indices``.
- For t == 1 the input graph is returned unchanged (bit-for-bit identity), so a
  hop-1 run reproduces the original pipeline exactly.
- For t >= 2 the neighbor table is re-ordered per row: original 1-hop neighbors
  first (original order), then farther hops by cosine similarity (descending),
  truncated to ``cap`` columns; rows shorter than ``cap`` are padded with unused
  nodes (lowest index first) and counted as undersized.
- mutual / SNN statistics are recomputed against the ORIGINAL 1-hop
  neighborhoods (edge reliability keeps its "local structure similarity"
  semantics independent of the hop order); similarity/probs are recomputed for
  the new edge set.
- Pure numpy, no dependencies on sklearn (embedding already exists in graph).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .graph import NeighborGraph


def hop_reachable_sets(indices: np.ndarray, t: int) -> list[np.ndarray]:
    """For each sample, the sorted array of nodes reachable within <= t hops.

    ``indices``: (n, k) int64 1-hop neighbor table (no self loops).
    Returns ``S_t`` sets excluding the sample itself.
    """
    n = int(indices.shape[0])
    adj = [set(int(v) for v in row if int(v) != i) for i, row in enumerate(indices)]
    out: list[np.ndarray] = []
    steps = max(0, int(t) - 1)
    for i in range(n):
        reached = set(adj[i])
        front = set(adj[i])
        for _ in range(steps):
            nxt: set[int] = set()
            for v in front:
                nxt |= adj[v]
            nxt.discard(i)
            reached |= nxt
            front = nxt
            if not front:
                break
        out.append(np.array(sorted(reached), dtype=np.int64))
    return out


def build_hop_graph(graph: NeighborGraph, t: int, cap: int = 48) -> NeighborGraph:
    """Return a graph whose neighbor table is the t-hop expansion of ``graph``.

    t <= 1 returns the exact same object (bit-for-bit identity).
    """
    if t is None or int(t) <= 1:
        return graph
    t = int(t)
    cap = int(cap)
    indices = graph.indices
    n, k1 = indices.shape
    if n <= 1:
        return graph

    # Cosine similarities between every pair in the (normalized) PCA embedding.
    embedding = np.asarray(graph.embedding, dtype=np.float32)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    u = embedding / np.maximum(norms, 1e-12)
    sim_all = np.asarray(u @ u.T, dtype=np.float32)  # (n, n)

    tau = float(graph.profile.get("tau", 0.2))
    sets = hop_reachable_sets(indices, t)
    one_hop_sets = [set(int(v) for v in row) for row in indices]
    kp = min(cap, n - 1)

    out_idx = np.empty((n, kp), dtype=np.int64)
    out_sim = np.empty((n, kp), dtype=np.float32)
    undersized = 0
    for i in range(n):
        s = sets[i]
        in_s = set(int(v) for v in s)
        # 1-hop neighbors first, preserving the original row order.
        one = [int(v) for v in indices[i] if int(v) in in_s]
        # Higher-hop candidates, sorted by cosine similarity (descending).
        rest = [int(v) for v in s if int(v) not in one_hop_sets[i]]
        rest.sort(key=lambda v: -float(sim_all[i, v]))
        ordered = (one + rest)[:kp]
        if len(ordered) < kp:
            undersized += 1
            used = set(ordered)
            for v in range(n):
                if v == i or v in used:
                    continue
                ordered.append(v)
                if len(ordered) >= kp:
                    break
        # Safety: still short only if n - 1 < kp (kp <= n - 1 guarantees fill).
        ordered = ordered[:kp]
        out_idx[i] = np.asarray(ordered, dtype=np.int64)
        out_sim[i] = np.asarray([sim_all[i, v] for v in ordered], dtype=np.float32)

    distance = (1.0 - out_sim).astype(np.float32)
    scaled = out_sim / max(tau, 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled).astype(np.float32)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

    # mutual / SNN recomputed against the ORIGINAL 1-hop neighborhoods.
    mutual = np.zeros_like(out_idx, dtype=bool)
    snn = np.zeros_like(out_sim, dtype=np.float32)
    for i in range(n):
        n1_i = one_hop_sets[i]
        for pos in range(kp):
            j = int(out_idx[i, pos])
            n1_j = one_hop_sets[j]
            mutual[i, pos] = i in n1_j
            union = len(n1_i | n1_j)
            inter = len(n1_i & n1_j)
            snn[i, pos] = inter / float(max(1, union))

    profile = dict(graph.profile)
    profile.update(
        {
            "neighbor_k": int(kp),
            "hop_order": int(t),
            "hop_cap": int(cap),
            "hop_undersized_rows": int(undersized),
            "fraction_undersized": float(undersized / n),
            "mean_hop_similarity": float(np.mean(out_sim)),
            "mean_mutual_ratio": float(np.mean(mutual)),
            "mean_snn": float(np.mean(snn)),
        }
    )
    return replace(
        graph,
        indices=out_idx,
        probs=probs.astype(np.float32),
        similarity=out_sim,
        distance=distance,
        mutual=mutual,
        snn=snn.astype(np.float32),
        profile=profile,
    )
