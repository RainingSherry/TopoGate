#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = next(parent for parent in [CURRENT_DIR, *CURRENT_DIR.parents] if (parent / "methods" / "DeepLearning" / "scMAE_family.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.DeepLearning import scMAE_family as family
from experimental_retired_models.NeighborMix_scMAE.model import AutoEncoder as _BaseAutoEncoder
from methods.shared_utils import ensure_dir, save_json, sanitize_anndata_for_write


METHOD_CHOICES = [
    "beta_control",
    "nm_scmae_nomix",
    "neighbormix_scmae",
    "random_pseudo_gate_p0.5",
    "random_edge_dropout_keep0.5",
    "random_beta_uniform_0.1",
    "mutual_knn_neighbormix",
    "snn_neighbormix",
    "consensus_neighbormix_threshold0.4",
    "global_random_neighbor_control",
]


@dataclass
class NeighborGraph:
    indices: np.ndarray
    probs: np.ndarray
    similarity: np.ndarray
    distance: np.ndarray
    mutual: np.ndarray
    snn: np.ndarray
    consensus: np.ndarray
    fallback_mask: np.ndarray
    embedding: np.ndarray
    profile: dict


class AutoEncoder(_BaseAutoEncoder):
    """NeighborMix scMAE AutoEncoder with optional per-sample pseudo loss weights."""

    def loss_mask_weighted(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        mask_loss_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._check_expression_shape(x, "x")
        self._check_expression_shape(y, "y")
        self._check_expression_shape(mask, "mask")
        if x.shape != y.shape or x.shape != mask.shape:
            raise ValueError("x, y, and mask must have identical shapes.")

        mask = mask.to(dtype=x.dtype, device=x.device)
        y = y.to(dtype=x.dtype, device=x.device)
        latent, mask_logits, reconstruction = self.forward_mask(x)
        raw_mse = F.mse_loss(reconstruction, y, reduction="none")
        weights = mask * self.masked_data_weight + (1.0 - mask) * (1.0 - self.masked_data_weight)
        weighted_mse = weights * raw_mse
        if self.normalize_reconstruction_by_weight:
            rec_per = weighted_mse.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
        else:
            rec_per = weighted_mse.mean(dim=1)
        rec_per = (1.0 - self.mask_loss_weight) * rec_per
        mask_per = F.binary_cross_entropy_with_logits(mask_logits, mask, reduction="none").mean(dim=1)
        mask_per = self.mask_loss_weight * mask_per
        total_per = rec_per + float(mask_loss_scale) * mask_per

        if sample_weight is None:
            loss = total_per.mean()
        else:
            w = sample_weight.to(dtype=x.dtype, device=x.device).view(-1)
            loss = (total_per * w).sum() / w.sum().clamp_min(1e-8)
        parts = {
            "reconstruction_loss": rec_per.mean().detach(),
            "mask_loss": mask_per.mean().detach(),
            "total_loss": loss.detach(),
            "mask_positive_rate": mask.mean().detach(),
            "per_sample_loss": total_per.detach(),
        }
        return latent, loss, parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeighborMix beta mechanism and pseudo-branch mechanism ablation")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--ablation_method", required=True, choices=METHOD_CHOICES)
    parser.add_argument("--method_name", default=None)
    parser.add_argument("--variant_name", default=None)
    parser.add_argument("--beta_mode", default="fixed", choices=["fixed", "uniform", "bernoulli", "truncated_normal", "beta_distribution"])
    parser.add_argument("--beta_fixed", type=float, default=None)
    parser.add_argument("--beta_max", type=float, default=None)
    parser.add_argument("--beta_mean", type=float, default=0.05)
    parser.add_argument("--beta_std", type=float, default=0.02)
    parser.add_argument("--beta_p", type=float, default=0.5)
    parser.add_argument("--beta_alpha", type=float, default=None)
    parser.add_argument("--beta_beta", type=float, default=None)
    parser.add_argument("--beta_concentration", type=float, default=16.0)
    parser.add_argument("--target_mode", default="anchor", choices=["anchor", "mixed"])
    parser.add_argument("--noise_mode", default="local_mix", choices=["local_mix", "global_mix", "gaussian_matched"])
    parser.add_argument("--bad_edge_ratio", type=float, default=0.0)
    parser.add_argument("--oracle_neighbor", default="none", choices=["none", "same_label", "cross_label"])
    parser.add_argument("--label_key", default="auto")
    parser.add_argument("--input_mode", default="auto", choices=["auto", "raw", "log1p"])
    parser.add_argument("--n_top_genes", type=int, default=1000)
    parser.add_argument("--target_sum", type=float, default=10000.0)
    parser.add_argument("--scale_input", type=family.str2bool, default=True)
    parser.add_argument("--n_clusters", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--masked_data_weight", type=float, default=0.75)
    parser.add_argument("--mask_loss_weight", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--pseudo_weight", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--neighbor_k", type=int, default=5)
    parser.add_argument("--mix_neighbors", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--knn_pca_dim", type=int, default=50)
    parser.add_argument("--consensus_threshold", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--save_h5ad", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    return parser.parse_args()


def make_loaders(data_np: np.ndarray, labels: np.ndarray, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    dataset = family.IndexedExpressionDataset(data_np, labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=max(batch_size * 4, 512),
        shuffle=False,
        drop_last=False,
    )
    return train_loader, eval_loader


def _pca_embedding(data_np: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    data = np.asarray(data_np, dtype=np.float32)
    n_cells, n_genes = data.shape
    dim = max(1, min(int(pca_dim), n_genes, max(1, n_cells - 1)))
    if min(data.shape) > 1 and dim < min(data.shape):
        emb = PCA(n_components=dim, random_state=seed).fit_transform(data)
    else:
        emb = data
    return np.nan_to_num(np.asarray(emb, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _rank_embedding(data_np: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    data = np.asarray(data_np, dtype=np.float32)
    order = np.argsort(data, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    col_ranks = np.arange(data.shape[0], dtype=np.float32)[:, None]
    np.put_along_axis(ranks, order, col_ranks, axis=0)
    if data.shape[0] > 1:
        ranks /= float(data.shape[0] - 1)
    return _pca_embedding(ranks, pca_dim=pca_dim, seed=seed)


def _knn_from_embedding(
    embedding: np.ndarray,
    k: int,
    tau: float,
    metric: str,
    source: str,
    pca_dim: Optional[int] = None,
) -> NeighborGraph:
    emb = np.asarray(embedding, dtype=np.float32)
    n_cells = int(emb.shape[0])
    if k <= 0 or n_cells <= 1:
        return _empty_graph(n_cells, source=source)

    k_eff = min(int(k), n_cells - 1)
    fit_emb = normalize(emb, axis=1).astype(np.float32) if metric == "cosine" else emb
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(fit_emb)
    distances, indices = nn.kneighbors(fit_emb)
    indices = indices[:, 1 : k_eff + 1].astype(np.int64, copy=False)
    distances = distances[:, 1 : k_eff + 1].astype(np.float32, copy=False)

    if metric == "cosine":
        similarity = (1.0 - distances).astype(np.float32)
    else:
        scale = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
        similarity = np.exp(-distances / max(scale, 1e-8)).astype(np.float32)

    scaled = similarity / max(float(tau), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled).astype(np.float32)
    probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-12, None)
    mutual, snn = _mutual_and_snn(indices)
    consensus = np.ones_like(similarity, dtype=np.float32)
    fallback = np.zeros_like(indices, dtype=bool)
    profile = _graph_profile(
        indices=indices,
        probs=probs,
        similarity=similarity,
        mutual=mutual,
        snn=snn,
        consensus=consensus,
        fallback_mask=fallback,
        source=source,
        pca_dim=pca_dim,
    )
    return NeighborGraph(indices, probs.astype(np.float32), similarity, distances, mutual, snn, consensus, fallback, fit_emb, profile)


def _mutual_and_snn(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_cells, k = indices.shape
    mutual = np.zeros_like(indices, dtype=bool)
    snn = np.zeros_like(indices, dtype=np.float32)
    sets = [set(row.tolist()) for row in indices]
    for i in range(n_cells):
        set_i = sets[i]
        for pos, j in enumerate(indices[i]):
            if 0 <= int(j) < n_cells:
                mutual[i, pos] = i in sets[int(j)]
                union = set_i.union(sets[int(j)])
                snn[i, pos] = len(set_i.intersection(sets[int(j)])) / float(max(1, len(union)))
    return mutual, snn


def _graph_profile(
    indices: np.ndarray,
    probs: np.ndarray,
    similarity: np.ndarray,
    mutual: np.ndarray,
    snn: np.ndarray,
    consensus: np.ndarray,
    fallback_mask: np.ndarray,
    source: str,
    pca_dim: Optional[int],
) -> dict:
    n_cells, k = indices.shape
    if k == 0:
        return {
            "neighbor_backend": source,
            "neighbor_k": 0,
            "effective_neighbor_count": 0.0,
            "edge_keep_rate": 0.0,
            "neighbor_similarity_mean": 0.0,
            "neighbor_similarity_std": 0.0,
            "mutual_ratio": 0.0,
            "snn_mean": 0.0,
            "snn_std": 0.0,
            "consensus_mean": 0.0,
            "consensus_std": 0.0,
            "fallback_rate": 0.0,
            "hubness_mean": 0.0,
            "hubness_max": 0.0,
            "hubness_p95": 0.0,
            "knn_pca_dim": pca_dim,
        }
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)
    effective = np.exp(entropy)
    indegree = np.bincount(indices.reshape(-1), minlength=n_cells).astype(np.float32)
    keep = probs > 0
    return {
        "neighbor_backend": source,
        "neighbor_k": int(k),
        "effective_neighbor_count": float(np.mean(effective)),
        "edge_keep_rate": float(np.mean(keep)),
        "neighbor_similarity_mean": float(np.mean(similarity)),
        "neighbor_similarity_std": float(np.std(similarity)),
        "mutual_ratio": float(np.mean(mutual)),
        "snn_mean": float(np.mean(snn)),
        "snn_std": float(np.std(snn)),
        "consensus_mean": float(np.mean(consensus)),
        "consensus_std": float(np.std(consensus)),
        "fallback_rate": float(np.mean(fallback_mask)),
        "fallback_cell_rate": float(np.mean(np.any(fallback_mask, axis=1))),
        "hubness_mean": float(np.mean(indegree)),
        "hubness_max": float(np.max(indegree)),
        "hubness_p95": float(np.percentile(indegree, 95)),
        "knn_pca_dim": pca_dim,
    }


def _empty_graph(n_cells: int, source: str) -> NeighborGraph:
    empty_i = np.zeros((n_cells, 0), dtype=np.int64)
    empty_f = np.zeros((n_cells, 0), dtype=np.float32)
    empty_b = np.zeros((n_cells, 0), dtype=bool)
    return NeighborGraph(
        empty_i,
        empty_f,
        empty_f,
        empty_f,
        empty_b,
        empty_f,
        empty_f,
        empty_b,
        np.zeros((n_cells, 0), dtype=np.float32),
        _graph_profile(empty_i, empty_f, empty_f, empty_b, empty_f, empty_f, empty_b, source, None),
    )


def build_vanilla_graph(data_np: np.ndarray, k: int, pca_dim: int, tau: float, seed: int) -> NeighborGraph:
    emb = _pca_embedding(data_np, pca_dim=pca_dim, seed=seed)
    return _knn_from_embedding(emb, k=k, tau=tau, metric="cosine", source="pca_cosine_knn", pca_dim=pca_dim)


def build_mutual_graph(data_np: np.ndarray, k: int, pca_dim: int, tau: float, seed: int) -> NeighborGraph:
    base = build_vanilla_graph(data_np, k=k, pca_dim=pca_dim, tau=tau, seed=seed)
    if base.indices.shape[1] == 0:
        return base
    n_cells, k_eff = base.indices.shape
    out = np.zeros_like(base.indices)
    sim = np.zeros_like(base.similarity)
    dist = np.zeros_like(base.distance)
    snn = np.zeros_like(base.snn)
    consensus = np.zeros_like(base.consensus)
    fallback = np.zeros_like(base.fallback_mask)
    for i in range(n_cells):
        chosen: list[int] = []
        slots: list[int] = []
        for pos, j in enumerate(base.indices[i].tolist()):
            if base.mutual[i, pos]:
                chosen.append(j)
                slots.append(pos)
        for pos, j in enumerate(base.indices[i].tolist()):
            if len(chosen) >= k_eff:
                break
            if j not in chosen:
                chosen.append(j)
                slots.append(pos)
                fallback[i, len(chosen) - 1] = True
        for out_pos, src_pos in enumerate(slots[:k_eff]):
            out[i, out_pos] = base.indices[i, src_pos]
            sim[i, out_pos] = base.similarity[i, src_pos]
            dist[i, out_pos] = base.distance[i, src_pos]
            snn[i, out_pos] = base.snn[i, src_pos]
            consensus[i, out_pos] = 1.0 if base.mutual[i, src_pos] else 0.0
    scaled = sim / max(float(tau), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled).astype(np.float32)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    mutual, snn_recomputed = _mutual_and_snn(out)
    profile = _graph_profile(out, probs, sim, mutual, snn_recomputed, consensus, fallback, "mutual_knn_with_topk_fallback", pca_dim)
    profile["fallback_slot_rate"] = float(np.mean(fallback))
    return NeighborGraph(out, probs, sim, dist, mutual, snn_recomputed, consensus, fallback, base.embedding, profile)


def build_snn_graph(data_np: np.ndarray, k: int, pca_dim: int, tau: float, seed: int) -> NeighborGraph:
    base = build_vanilla_graph(data_np, k=k, pca_dim=pca_dim, tau=tau, seed=seed)
    if base.indices.shape[1] == 0:
        return base
    raw = base.snn.astype(np.float32)
    empty_rows = raw.sum(axis=1) <= 1e-12
    raw[empty_rows] = base.probs[empty_rows]
    probs = raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-12, None)
    fallback = np.zeros_like(base.fallback_mask)
    fallback[empty_rows, :] = True
    profile = _graph_profile(base.indices, probs, base.similarity, base.mutual, base.snn, base.snn, fallback, "snn_jaccard_weighted_knn", pca_dim)
    profile["snn_zero_row_fallback_rate"] = float(np.mean(empty_rows))
    return NeighborGraph(base.indices, probs.astype(np.float32), base.similarity, base.distance, base.mutual, base.snn, base.snn, fallback, base.embedding, profile)


def build_global_random_graph(data_np: np.ndarray, k: int, pca_dim: int, tau: float, seed: int) -> NeighborGraph:
    base = build_vanilla_graph(data_np, k=max(1, k), pca_dim=pca_dim, tau=tau, seed=seed)
    n_cells = int(data_np.shape[0])
    if k <= 0 or n_cells <= 1:
        return _empty_graph(n_cells, "global_random")
    k_eff = min(int(k), n_cells - 1)
    rng = np.random.default_rng(seed + 9403)
    all_idx = np.arange(n_cells, dtype=np.int64)
    indices = np.zeros((n_cells, k_eff), dtype=np.int64)
    for i in range(n_cells):
        candidates = all_idx[all_idx != i]
        indices[i] = rng.choice(candidates, size=k_eff, replace=candidates.size < k_eff)
    emb = base.embedding
    row = emb[np.arange(n_cells)[:, None]]
    nbr = emb[indices]
    similarity = np.sum(row * nbr, axis=2).astype(np.float32) if emb.shape[1] else np.zeros((n_cells, k_eff), dtype=np.float32)
    distance = (1.0 - similarity).astype(np.float32)
    probs = np.full((n_cells, k_eff), 1.0 / float(k_eff), dtype=np.float32)
    mutual, snn = _mutual_and_snn(indices)
    consensus = np.zeros_like(similarity, dtype=np.float32)
    fallback = np.zeros_like(indices, dtype=bool)
    profile = _graph_profile(indices, probs, similarity, mutual, snn, consensus, fallback, "global_random_excluding_self", pca_dim)
    return NeighborGraph(indices, probs, similarity, distance, mutual, snn, consensus, fallback, emb, profile)


def _graph_from_indices(
    indices: np.ndarray,
    embedding: np.ndarray,
    tau: float,
    source: str,
    pca_dim: Optional[int],
    fallback_mask: Optional[np.ndarray] = None,
    consensus: Optional[np.ndarray] = None,
) -> NeighborGraph:
    indices = np.asarray(indices, dtype=np.int64)
    n_cells, k_eff = indices.shape
    if k_eff == 0:
        return _empty_graph(n_cells, source)
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.shape[1] == 0:
        similarity = np.zeros((n_cells, k_eff), dtype=np.float32)
    else:
        row = emb[np.arange(n_cells)[:, None]]
        nbr = emb[indices]
        similarity = np.sum(row * nbr, axis=2).astype(np.float32)
    distance = (1.0 - similarity).astype(np.float32)
    scaled = similarity / max(float(tau), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled).astype(np.float32)
    probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-12, None)
    mutual, snn = _mutual_and_snn(indices)
    if fallback_mask is None:
        fallback_mask = np.zeros_like(indices, dtype=bool)
    if consensus is None:
        consensus = np.ones_like(similarity, dtype=np.float32)
    profile = _graph_profile(indices, probs, similarity, mutual, snn, consensus, fallback_mask, source, pca_dim)
    return NeighborGraph(indices, probs.astype(np.float32), similarity, distance, mutual, snn, consensus, fallback_mask, emb, profile)


def build_oracle_graph(
    data_np: np.ndarray,
    labels: np.ndarray,
    k: int,
    pca_dim: int,
    tau: float,
    seed: int,
    mode: str,
) -> NeighborGraph:
    n_cells = int(data_np.shape[0])
    if k <= 0 or n_cells <= 1:
        return _empty_graph(n_cells, f"oracle_{mode}")
    k_eff = min(int(k), n_cells - 1)
    pool_k = min(max(k_eff * 20, 50), n_cells - 1)
    base = build_vanilla_graph(data_np, k=pool_k, pca_dim=pca_dim, tau=tau, seed=seed)
    rng = np.random.default_rng(seed + 6619)
    labels = np.asarray(labels)
    all_idx = np.arange(n_cells, dtype=np.int64)
    by_label = {value: all_idx[labels == value] for value in np.unique(labels)}
    cross_by_label = {value: all_idx[labels != value] for value in np.unique(labels)}
    out = np.zeros((n_cells, k_eff), dtype=np.int64)
    fallback = np.zeros((n_cells, k_eff), dtype=bool)
    for i in range(n_cells):
        if mode == "same_label":
            valid_pool = by_label[labels[i]]
            mask = labels[base.indices[i]] == labels[i]
        elif mode == "cross_label":
            valid_pool = cross_by_label[labels[i]]
            mask = labels[base.indices[i]] != labels[i]
        else:
            raise ValueError(f"Unknown oracle neighbor mode: {mode}")
        valid_pool = valid_pool[valid_pool != i]
        chosen = [int(j) for j in base.indices[i][mask].tolist() if int(j) != i]
        if len(chosen) < k_eff:
            existing = set(chosen)
            candidates = np.asarray([int(j) for j in valid_pool.tolist() if int(j) not in existing], dtype=np.int64)
            need = k_eff - len(chosen)
            if candidates.size == 0:
                candidates = all_idx[all_idx != i]
            extra = rng.choice(candidates, size=need, replace=candidates.size < need).astype(np.int64)
            chosen.extend([int(j) for j in extra.tolist()])
            fallback[i, len(chosen) - need : k_eff] = True
        out[i] = np.asarray(chosen[:k_eff], dtype=np.int64)
    graph = _graph_from_indices(out, base.embedding, tau=tau, source=f"oracle_{mode}_pca_cosine", pca_dim=pca_dim, fallback_mask=fallback)
    graph.profile["oracle_neighbor"] = mode
    graph.profile["label_leakage_diagnostic"] = True
    return graph


def inject_bad_edges(graph: NeighborGraph, labels: np.ndarray, bad_edge_ratio: float, seed: int, tau: float, pca_dim: int) -> NeighborGraph:
    ratio = float(bad_edge_ratio)
    if ratio <= 0.0 or graph.indices.shape[1] == 0:
        graph.profile["bad_edge_ratio_requested"] = max(0.0, ratio)
        graph.profile["bad_edge_ratio_observed"] = 0.0
        return graph
    ratio = min(1.0, ratio)
    labels = np.asarray(labels)
    n_cells, k_eff = graph.indices.shape
    rng = np.random.default_rng(seed + 8831 + int(round(ratio * 1000)))
    all_idx = np.arange(n_cells, dtype=np.int64)
    cross_by_label = {value: all_idx[labels != value] for value in np.unique(labels)}
    out = graph.indices.copy()
    fallback = graph.fallback_mask.copy()
    replaced = np.zeros_like(out, dtype=bool)
    for i in range(n_cells):
        replace_mask = rng.random(k_eff) < ratio
        if not np.any(replace_mask):
            continue
        pool = cross_by_label[labels[i]]
        pool = pool[pool != i]
        if pool.size == 0:
            pool = all_idx[all_idx != i]
        draw = rng.choice(pool, size=int(np.sum(replace_mask)), replace=pool.size < int(np.sum(replace_mask))).astype(np.int64)
        out[i, replace_mask] = draw
        fallback[i, replace_mask] = True
        replaced[i, replace_mask] = True
    consensus = graph.consensus.copy()
    consensus[replaced] = 0.0
    out_graph = _graph_from_indices(
        out,
        graph.embedding,
        tau=tau,
        source=f"{graph.profile.get('neighbor_backend', 'knn')}_bad_edge_injected_{ratio:g}",
        pca_dim=pca_dim,
        fallback_mask=fallback,
        consensus=consensus,
    )
    out_graph.profile["bad_edge_ratio_requested"] = ratio
    out_graph.profile["bad_edge_ratio_observed"] = float(np.mean(replaced))
    out_graph.profile["label_leakage_diagnostic"] = True
    return out_graph


def build_consensus_graph(
    data_np: np.ndarray,
    k: int,
    pca_dim: int,
    tau: float,
    seed: int,
    threshold: float,
) -> NeighborGraph:
    cos = build_vanilla_graph(data_np, k=k, pca_dim=pca_dim, tau=tau, seed=seed)
    if cos.indices.shape[1] == 0:
        return cos
    emb = _pca_embedding(data_np, pca_dim=pca_dim, seed=seed)
    euc = _knn_from_embedding(emb, k=k, tau=tau, metric="euclidean", source="pca_euclidean_knn", pca_dim=pca_dim)
    rank = _knn_from_embedding(_rank_embedding(data_np, pca_dim=pca_dim, seed=seed), k=k, tau=tau, metric="cosine", source="rank_cosine_knn", pca_dim=pca_dim)
    n_cells, k_eff = cos.indices.shape
    out = np.zeros_like(cos.indices)
    sim = np.zeros_like(cos.similarity)
    dist = np.zeros_like(cos.distance)
    consensus = np.zeros_like(cos.consensus)
    fallback = np.zeros_like(cos.fallback_mask)
    source_rows = [cos.indices, euc.indices, rank.indices]
    cos_rank = [{int(j): pos for pos, j in enumerate(cos.indices[i])} for i in range(n_cells)]
    for i in range(n_cells):
        counts: dict[int, int] = {}
        for rows in source_rows:
            for j in rows[i].tolist():
                counts[int(j)] = counts.get(int(j), 0) + 1
        for pos, j in enumerate(cos.indices[i].tolist()):
            if cos.mutual[i, pos]:
                counts[int(j)] = counts.get(int(j), 0) + 1
            if cos.snn[i, pos] > 0.0:
                counts[int(j)] = counts.get(int(j), 0) + 1
        candidates = []
        for j, count in counts.items():
            score = count / 5.0
            rank_pos = cos_rank[i].get(j, k_eff + 10)
            candidates.append((score, -rank_pos, j))
        candidates.sort(reverse=True)
        chosen: list[tuple[int, float, bool]] = []
        for score, _neg_rank, j in candidates:
            if score >= float(threshold):
                chosen.append((j, score, False))
            if len(chosen) >= k_eff:
                break
        chosen_ids = {j for j, _score, _fb in chosen}
        for pos, j in enumerate(cos.indices[i].tolist()):
            if len(chosen) >= k_eff:
                break
            if int(j) not in chosen_ids:
                chosen.append((int(j), counts.get(int(j), 0) / 5.0, True))
                chosen_ids.add(int(j))
        for pos, (j, score, is_fallback) in enumerate(chosen[:k_eff]):
            out[i, pos] = j
            src_pos = cos_rank[i].get(j)
            if src_pos is not None:
                sim[i, pos] = cos.similarity[i, src_pos]
                dist[i, pos] = cos.distance[i, src_pos]
            else:
                sim[i, pos] = float(cos.embedding[i] @ cos.embedding[j])
                dist[i, pos] = 1.0 - sim[i, pos]
            consensus[i, pos] = score
            fallback[i, pos] = is_fallback
    raw = np.clip(consensus, 1e-3, None)
    raw = raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-12, None)
    mutual, snn = _mutual_and_snn(out)
    source = f"multi_metric_consensus_threshold_{threshold:g}"
    profile = _graph_profile(out, raw.astype(np.float32), sim, mutual, snn, consensus, fallback, source, pca_dim)
    profile["consensus_threshold"] = float(threshold)
    return NeighborGraph(out, raw.astype(np.float32), sim, dist, mutual, snn, consensus, fallback, cos.embedding, profile)


def build_graph_for_method(args: argparse.Namespace, data_np: np.ndarray, labels: np.ndarray) -> NeighborGraph:
    method = args.ablation_method
    if method == "nm_scmae_nomix":
        return _empty_graph(data_np.shape[0], "none")
    if args.oracle_neighbor != "none":
        graph = build_oracle_graph(data_np, labels, args.neighbor_k, args.knn_pca_dim, args.tau, args.seed, args.oracle_neighbor)
        return inject_bad_edges(graph, labels, args.bad_edge_ratio, args.seed, args.tau, args.knn_pca_dim)
    if method == "mutual_knn_neighbormix":
        graph = build_mutual_graph(data_np, args.neighbor_k, args.knn_pca_dim, args.tau, args.seed)
    elif method == "snn_neighbormix":
        graph = build_snn_graph(data_np, args.neighbor_k, args.knn_pca_dim, args.tau, args.seed)
    elif method == "consensus_neighbormix_threshold0.4":
        graph = build_consensus_graph(data_np, args.neighbor_k, args.knn_pca_dim, args.tau, args.seed, args.consensus_threshold)
    elif method == "global_random_neighbor_control" or args.noise_mode == "global_mix":
        graph = build_global_random_graph(data_np, args.neighbor_k, args.knn_pca_dim, args.tau, args.seed)
    else:
        graph = build_vanilla_graph(data_np, args.neighbor_k, args.knn_pca_dim, args.tau, args.seed)
    return inject_bad_edges(graph, labels, args.bad_edge_ratio, args.seed, args.tau, args.knn_pca_dim)


def method_config(method: str, args: argparse.Namespace) -> dict:
    beta = 1.0 - float(args.alpha) if args.beta_fixed is None else float(args.beta_fixed)
    beta_max = beta if args.beta_max is None else float(args.beta_max)
    cfg = {
        "pseudo_enabled": method != "nm_scmae_nomix",
        "pseudo_gate_p": 1.0,
        "edge_dropout_keep": 1.0,
        "beta_mode": args.beta_mode,
        "beta_fixed": beta,
        "beta_max": beta_max,
        "beta_mean": float(args.beta_mean),
        "beta_std": float(args.beta_std),
        "beta_p": float(args.beta_p),
        "beta_alpha": args.beta_alpha,
        "beta_beta": args.beta_beta,
        "beta_concentration": float(args.beta_concentration),
        "target_mode": args.target_mode,
        "noise_mode": args.noise_mode,
        "bad_edge_ratio": float(args.bad_edge_ratio),
        "oracle_neighbor": args.oracle_neighbor,
        "mask_loss_scale": 1.0,
    }
    if method == "random_pseudo_gate_p0.5":
        cfg["pseudo_gate_p"] = 0.5
    elif method == "random_edge_dropout_keep0.5":
        cfg["edge_dropout_keep"] = 0.5
    elif method == "random_beta_uniform_0.1":
        cfg["beta_mode"] = "uniform"
        cfg["beta_max"] = 0.1
    elif method == "neighbormix_scmae":
        cfg["beta_mode"] = "fixed"
        cfg["beta_fixed"] = 1.0 - float(args.alpha)
        cfg["beta_max"] = cfg["beta_fixed"]
    return cfg


def sample_beta(cfg: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    mode = str(cfg["beta_mode"])
    if mode == "fixed":
        return np.full(n, float(cfg["beta_fixed"]), dtype=np.float32)
    if mode == "uniform":
        return rng.uniform(0.0, float(cfg["beta_max"]), size=n).astype(np.float32)
    if mode == "bernoulli":
        high = float(cfg["beta_max"])
        p = min(1.0, max(0.0, float(cfg["beta_p"])))
        return (rng.random(n) < p).astype(np.float32) * np.float32(high)
    if mode == "truncated_normal":
        beta_max = float(cfg["beta_max"]) if cfg.get("beta_max") is not None else 1.0
        values = rng.normal(float(cfg["beta_mean"]), float(cfg["beta_std"]), size=n)
        return np.clip(values, 0.0, beta_max).astype(np.float32)
    if mode == "beta_distribution":
        beta_max = float(cfg["beta_max"]) if cfg.get("beta_max") is not None else 0.1
        alpha = cfg.get("beta_alpha")
        beta_param = cfg.get("beta_beta")
        if alpha is None or beta_param is None:
            mean_frac = float(cfg["beta_mean"]) / max(beta_max, 1e-12)
            mean_frac = min(1.0 - 1e-6, max(1e-6, mean_frac))
            concentration = max(float(cfg["beta_concentration"]), 2.0)
            alpha = mean_frac * concentration
            beta_param = (1.0 - mean_frac) * concentration
        return (rng.beta(float(alpha), float(beta_param), size=n) * beta_max).astype(np.float32)
    raise ValueError(f"Unsupported beta_mode: {mode}")


def sample_neighbor_mean(
    data_np: np.ndarray,
    graph: NeighborGraph,
    batch_indices: np.ndarray,
    mix_neighbors: int,
    rng: np.random.Generator,
    edge_keep_prob: float,
) -> tuple[np.ndarray, dict]:
    if graph.indices.shape[1] == 0 or int(mix_neighbors) <= 0:
        return data_np[batch_indices].astype(np.float32), {
            "edge_keep_rate": 0.0,
            "effective_neighbor_count": 0.0,
            "mean_sampled_weight_max": 0.0,
        }
    bsz = int(batch_indices.shape[0])
    k = int(graph.indices.shape[1])
    m = max(1, min(int(mix_neighbors), k))
    sampled = np.zeros((bsz, m), dtype=np.int64)
    weights = np.zeros((bsz, m), dtype=np.float32)
    keep_rates = []
    effective_counts = []
    for pos, cell in enumerate(batch_indices):
        row = graph.indices[cell]
        probs = graph.probs[cell].astype(np.float32, copy=True)
        if edge_keep_prob < 1.0:
            keep = rng.random(k) < float(edge_keep_prob)
            if not np.any(keep):
                keep[int(np.argmax(probs))] = True
            probs = probs * keep.astype(np.float32)
            keep_rates.append(float(np.mean(keep)))
        else:
            keep_rates.append(1.0)
        probs = probs / np.clip(probs.sum(), 1e-12, None)
        choices = rng.choice(k, size=m, replace=True, p=probs)
        picked = probs[choices].astype(np.float32, copy=False)
        weights[pos] = picked / max(float(picked.sum()), 1e-12)
        sampled[pos] = row[choices]
        entropy = -float(np.sum(weights[pos] * np.log(np.clip(weights[pos], 1e-12, None))))
        effective_counts.append(math.exp(entropy))
    neighbor_expr = data_np[sampled]
    neighbor_mean = np.sum(neighbor_expr * weights[:, :, None], axis=1).astype(np.float32)
    info = {
        "edge_keep_rate": float(np.mean(keep_rates)),
        "effective_neighbor_count": float(np.mean(effective_counts)),
        "mean_sampled_weight_max": float(np.mean(np.max(weights, axis=1))),
    }
    return neighbor_mean, info


def label_diagnostics(labels: np.ndarray, graph: NeighborGraph) -> tuple[dict, pd.DataFrame]:
    if graph.indices.shape[1] == 0:
        return {
            "same_label_edge_ratio": 0.0,
            "same_label_edge_ratio_weighted": 0.0,
            "minority_class_neighbor_purity": 0.0,
        }, pd.DataFrame()
    same = labels[:, None] == labels[graph.indices]
    weighted = float(np.sum(graph.probs * same) / np.clip(np.sum(graph.probs), 1e-12, None))
    rows = []
    counts = pd.Series(labels).value_counts().sort_index()
    minority_threshold = float(np.percentile(counts.to_numpy(), 25)) if len(counts) else 0.0
    minority_values = set(counts[counts <= minority_threshold].index.tolist())
    minority_scores = []
    for value, count in counts.items():
        mask = labels == value
        if not np.any(mask):
            continue
        purity = float(np.mean(same[mask]))
        weighted_purity = float(np.sum(graph.probs[mask] * same[mask]) / np.clip(np.sum(graph.probs[mask]), 1e-12, None))
        rows.append(
            {
                "label": int(value),
                "n_cells": int(count),
                "same_label_edge_ratio": purity,
                "same_label_edge_ratio_weighted": weighted_purity,
                "is_minority_q25": bool(value in minority_values),
            }
        )
        if value in minority_values:
            minority_scores.append(weighted_purity)
    summary = {
        "same_label_edge_ratio": float(np.mean(same)),
        "same_label_edge_ratio_weighted": weighted,
        "minority_class_neighbor_purity": float(np.mean(minority_scores)) if minority_scores else 0.0,
    }
    return summary, pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.gpu in {0, 7} and not args.no_cuda:
        raise ValueError("GPU 0 and 7 are forbidden for this experiment.")
    family.set_seed(args.seed)
    save_dir = Path(ensure_dir(args.save_dir))
    method = args.ablation_method
    args.method_name = args.method_name or method
    args.variant_name = args.variant_name or method
    save_json(vars(args), str(save_dir / "args.json"))

    device = family.get_device(args.gpu, args.no_cuda)
    print(f"Using device: {device}; method={method}; seed={args.seed}", flush=True)

    bundle = family.load_scmae_dataset(
        file_path=args.data_path,
        input_mode=args.input_mode,
        n_top_genes=args.n_top_genes,
        target_sum=args.target_sum,
        scale_input=args.scale_input,
        label_key=args.label_key,
        seed=args.seed,
    )
    data_np = bundle.data
    labels = bundle.labels
    dataset_name = args.dataset_name or Path(args.data_path).stem
    n_clusters = int(args.n_clusters) if args.n_clusters and args.n_clusters > 0 else int(len(np.unique(labels)))
    save_json(bundle.profile, str(save_dir / "dataset_profile.json"))
    save_json(bundle.preprocess_config, str(save_dir / "preprocess_config.json"))
    np.save(save_dir / "gene_names.npy", bundle.gene_names.astype(str))

    graph = build_graph_for_method(args, data_np, labels)
    label_summary, per_class_edges = label_diagnostics(labels, graph)
    graph.profile.update(label_summary)
    save_json(graph.profile, str(save_dir / "neighbor_diagnostics.json"))
    pd.DataFrame([graph.profile]).to_csv(save_dir / "neighbor_diagnostics.csv", index=False)
    if len(per_class_edges):
        per_class_edges.to_csv(save_dir / "per_class_neighbor_purity.csv", index=False)
    if graph.indices.shape[1] > 0:
        np.save(save_dir / "neighbor_indices.npy", graph.indices)
        np.save(save_dir / "neighbor_probs.npy", graph.probs)
        np.save(save_dir / "neighbor_similarity.npy", graph.similarity)
        np.save(save_dir / "neighbor_snn.npy", graph.snn)
        np.save(save_dir / "neighbor_consensus.npy", graph.consensus)

    train_loader, eval_loader = make_loaders(data_np, labels, args.batch_size, args.seed)
    model = AutoEncoder(
        num_genes=data_np.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        masked_data_weight=args.masked_data_weight,
        mask_loss_weight=args.mask_loss_weight,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cfg = method_config(method, args)
    pseudo_enabled = bool(cfg["pseudo_enabled"]) and graph.indices.shape[1] > 0 and float(args.pseudo_weight) > 0.0
    rng = np.random.default_rng(args.seed + 20260616)

    history = {
        "loss": [],
        "real_loss": [],
        "pseudo_loss": [],
        "pseudo_branch_activation_rate": [],
        "edge_keep_rate": [],
        "effective_neighbor_count_sampled": [],
        "beta_mean": [],
        "alpha_mean": [],
        "perturbation_norm_mean": [],
        "perturbation_norm_std": [],
        "perturbation_norm_p95": [],
        "real_mask_rate": [],
        "pseudo_mask_rate": [],
        "method_config": cfg,
    }

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        totals = {key: 0.0 for key, value in history.items() if isinstance(value, list)}
        n_batches = 0
        for idx_t, x_cpu, _ in train_loader:
            idx_np = idx_t.numpy().astype(np.int64, copy=False)
            x = x_cpu.to(device)
            x_corrupt, real_mask = family.apply_scmae_noise(x, args.mask_ratio)
            _, real_loss, real_parts = model.loss_mask_weighted(x_corrupt, x, real_mask, mask_loss_scale=1.0)
            loss = real_loss
            pseudo_loss = torch.zeros((), dtype=real_loss.dtype, device=device)
            pseudo_mask_rate = 0.0
            activation_rate = 0.0
            edge_keep_rate = 0.0
            effective_sampled = 0.0
            beta_np = np.zeros(idx_np.shape[0], dtype=np.float32)
            perturb = np.zeros(idx_np.shape[0], dtype=np.float32)

            if pseudo_enabled:
                neighbor_mean, sample_info = sample_neighbor_mean(
                    data_np=data_np,
                    graph=graph,
                    batch_indices=idx_np,
                    mix_neighbors=args.mix_neighbors,
                    rng=rng,
                    edge_keep_prob=float(cfg["edge_dropout_keep"]),
                )
                beta_np = sample_beta(cfg, idx_np.shape[0], rng)
                anchor = data_np[idx_np].astype(np.float32, copy=False)
                mixed_np = (1.0 - beta_np[:, None]) * anchor + beta_np[:, None] * neighbor_mean
                perturb = np.linalg.norm(mixed_np - anchor, axis=1).astype(np.float32)
                if cfg["noise_mode"] == "gaussian_matched":
                    direction = rng.normal(0.0, 1.0, size=anchor.shape).astype(np.float32)
                    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
                    direction = direction / np.clip(direction_norm, 1e-12, None)
                    mixed_np = anchor + direction * perturb[:, None]
                    perturb = np.linalg.norm(mixed_np - anchor, axis=1).astype(np.float32)
                x_prime = torch.as_tensor(mixed_np, dtype=x.dtype, device=device)
                xp_corrupt, pseudo_mask = family.apply_scmae_noise(x_prime, args.mask_ratio)
                pseudo_target = x if cfg["target_mode"] == "anchor" else x_prime
                sample_weight = torch.ones(idx_np.shape[0], dtype=x.dtype, device=device)
                if float(cfg["pseudo_gate_p"]) < 1.0:
                    gate_np = (rng.random(idx_np.shape[0]) < float(cfg["pseudo_gate_p"])).astype(np.float32)
                    activation_rate = float(np.mean(gate_np))
                    if activation_rate > 0.0:
                        sample_weight = torch.as_tensor(gate_np, dtype=x.dtype, device=device)
                    else:
                        sample_weight = torch.zeros(idx_np.shape[0], dtype=x.dtype, device=device)
                else:
                    activation_rate = 1.0
                if float(sample_weight.sum().detach().cpu()) > 0.0:
                    _, pseudo_loss_raw, pseudo_parts = model.loss_mask_weighted(
                        xp_corrupt,
                        pseudo_target,
                        pseudo_mask,
                        sample_weight=sample_weight,
                        mask_loss_scale=float(cfg["mask_loss_scale"]),
                    )
                    if float(cfg["pseudo_gate_p"]) < 1.0:
                        pseudo_loss = pseudo_loss_raw * sample_weight.mean()
                    else:
                        pseudo_loss = pseudo_loss_raw
                    pseudo_mask_rate = float(pseudo_parts["mask_positive_rate"].detach().cpu())
                    loss = loss + float(args.pseudo_weight) * pseudo_loss
                edge_keep_rate = float(sample_info["edge_keep_rate"])
                effective_sampled = float(sample_info["effective_neighbor_count"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["real_loss"] += float(real_loss.detach().cpu())
            totals["pseudo_loss"] += float(pseudo_loss.detach().cpu())
            totals["pseudo_branch_activation_rate"] += activation_rate
            totals["edge_keep_rate"] += edge_keep_rate
            totals["effective_neighbor_count_sampled"] += effective_sampled
            totals["beta_mean"] += float(np.mean(beta_np)) if beta_np.size else 0.0
            totals["alpha_mean"] += float(np.mean(1.0 - beta_np)) if beta_np.size else 1.0
            totals["perturbation_norm_mean"] += float(np.mean(perturb)) if perturb.size else 0.0
            totals["perturbation_norm_std"] += float(np.std(perturb)) if perturb.size else 0.0
            totals["perturbation_norm_p95"] += float(np.percentile(perturb, 95)) if perturb.size else 0.0
            totals["real_mask_rate"] += float(real_mask.mean().detach().cpu())
            totals["pseudo_mask_rate"] += pseudo_mask_rate
            n_batches += 1

        for key, value in totals.items():
            history[key].append(value / max(1, n_batches))
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} loss={history['loss'][-1]:.4f} "
                f"real={history['real_loss'][-1]:.4f} pseudo={history['pseudo_loss'][-1]:.4f} "
                f"act={history['pseudo_branch_activation_rate'][-1]:.3f} "
                f"beta={history['beta_mean'][-1]:.3f}",
                flush=True,
            )

    embedding, labels_out = family.extract_embedding(model, eval_loader, device)
    np.save(save_dir / "embedding_final.npy", embedding.astype(np.float32))
    np.save(save_dir / "embeddings_base.npy", embedding.astype(np.float32))
    np.save(save_dir / "labels.npy", labels_out.astype(np.int64))
    save_json(history, str(save_dir / "training_history.json"))

    if args.save_model:
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "neighbor_profile": graph.profile,
                "gene_names": bundle.gene_names.astype(str),
            },
            save_dir / "model.pt",
        )

    extra = {
        "variant": args.variant_name,
        "ablation_method": method,
        "neighbor_backend": graph.profile.get("neighbor_backend", ""),
        "pseudo_weight": float(args.pseudo_weight if pseudo_enabled else 0.0),
        "pseudo_gate_p": float(cfg["pseudo_gate_p"]),
        "edge_dropout_keep": float(cfg["edge_dropout_keep"]),
        "alpha_mean": float(np.mean(history["alpha_mean"])) if history["alpha_mean"] else 1.0,
        "beta_mean": float(np.mean(history["beta_mean"])) if history["beta_mean"] else 0.0,
        "beta_mode": cfg["beta_mode"],
        "beta_fixed": cfg["beta_fixed"],
        "beta_max": cfg["beta_max"],
        "beta_target_mean": cfg["beta_mean"],
        "beta_std": cfg["beta_std"],
        "beta_p": cfg["beta_p"],
        "target_mode": cfg["target_mode"],
        "noise_mode": cfg["noise_mode"],
        "bad_edge_ratio": float(args.bad_edge_ratio),
        "oracle_neighbor": args.oracle_neighbor,
    }
    result = family.write_kmeans_known_k_outputs(
        output_dir=save_dir,
        dataset=dataset_name,
        method=args.method_name,
        seed=args.seed,
        embedding=embedding,
        labels=labels_out,
        n_clusters=n_clusters,
        extra=extra,
    )
    save_json(result["fixed"], str(save_dir / "metrics.json"))

    final_diag = {
        **graph.profile,
        "dataset": dataset_name,
        "seed": int(args.seed),
        "method": method,
        "pseudo_branch_activation_rate": float(np.mean(history["pseudo_branch_activation_rate"])) if history["pseudo_branch_activation_rate"] else 0.0,
        "edge_keep_rate_observed": float(np.mean(history["edge_keep_rate"])) if history["edge_keep_rate"] else graph.profile.get("edge_keep_rate", 0.0),
        "effective_neighbor_count_sampled": float(np.mean(history["effective_neighbor_count_sampled"])) if history["effective_neighbor_count_sampled"] else graph.profile.get("effective_neighbor_count", 0.0),
        "perturbation_norm_mean": float(np.mean(history["perturbation_norm_mean"])) if history["perturbation_norm_mean"] else 0.0,
        "perturbation_norm_std": float(np.mean(history["perturbation_norm_std"])) if history["perturbation_norm_std"] else 0.0,
        "perturbation_norm_p95": float(np.mean(history["perturbation_norm_p95"])) if history["perturbation_norm_p95"] else 0.0,
        "alpha_mean_observed": float(np.mean(history["alpha_mean"])) if history["alpha_mean"] else 1.0,
        "beta_mean_observed": float(np.mean(history["beta_mean"])) if history["beta_mean"] else 0.0,
        "beta_mode": cfg["beta_mode"],
        "beta_fixed": cfg["beta_fixed"],
        "beta_max": cfg["beta_max"],
        "beta_target_mean": cfg["beta_mean"],
        "beta_std": cfg["beta_std"],
        "beta_p": cfg["beta_p"],
        "target_mode": cfg["target_mode"],
        "noise_mode": cfg["noise_mode"],
        "bad_edge_ratio_requested": float(args.bad_edge_ratio),
        "bad_edge_ratio_observed": float(graph.profile.get("bad_edge_ratio_observed", 0.0)),
        "oracle_neighbor": args.oracle_neighbor,
        "label_leakage_diagnostic": bool(graph.profile.get("label_leakage_diagnostic", False)),
    }
    save_json(final_diag, str(save_dir / "neighbor_diagnostics_final.json"))
    pd.DataFrame([final_diag]).to_csv(save_dir / "neighbor_diagnostics_final.csv", index=False)

    summary = {
        "dataset": dataset_name,
        "method": args.method_name,
        "variant": args.variant_name,
        "ablation_method": method,
        "seed": int(args.seed),
        "n_cells": int(data_np.shape[0]),
        "n_genes": int(data_np.shape[1]),
        "n_clusters": int(n_clusters),
        "pseudo_enabled": bool(pseudo_enabled),
        "fixed_metrics": result["fixed"],
        "neighbor_diagnostics": final_diag,
        "label_leakage": bool(final_diag["label_leakage_diagnostic"]),
        "interpretation_note": "Pseudo branch reconstructs anchor cells by default; oracle and bad-edge modes are diagnostic and marked as label leakage.",
    }
    save_json(summary, str(save_dir / "summary.json"))

    if args.save_h5ad:
        bundle.adata.obsm[f"X_{method}"] = embedding
        bundle.adata.uns[method] = summary
        sanitize_anndata_for_write(bundle.adata)
        bundle.adata.write_h5ad(save_dir / f"adata_{method}.h5ad", compression="gzip")

    print(f"Results saved to: {save_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
