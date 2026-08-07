#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = next(parent for parent in [CURRENT_DIR, *CURRENT_DIR.parents] if (parent / "methods" / "DeepLearning" / "scMAE_family.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.DeepLearning import scMAE_family as family
from methods.TopoGate.static_gate.diagnostics import (
    embedding_geometry,
    mapped_predictions,
    per_cell_type_metrics,
)
from methods.TopoGate.static_gate.mixing import compute_node_gate, make_pseudo_batch
from methods.TopoGate.static_gate.model import AutoEncoder
from methods.TopoGate.static_gate.neighbor_graph import (
    build_far_neighbors,
    build_pca_knn_graph,
    build_random_neighbors,
    compute_edge_reliability,
)
from methods.shared_utils import ensure_dir, save_json, sanitize_anndata_for_write


def str2bool(value):
    return family.str2bool(value)


def parse_args():
    parser = argparse.ArgumentParser(description="TopoGate runner (Topology-Aware Pseudo-Perturbation scMAE)")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--method_name", default="TopoGate")
    parser.add_argument("--variant_name", default="topogate_full")
    parser.add_argument("--label_key", default="auto")
    parser.add_argument("--input_mode", default="auto", choices=["auto", "raw", "log1p"])
    parser.add_argument("--n_top_genes", type=int, default=1000)
    parser.add_argument("--target_sum", type=float, default=10000.0)
    parser.add_argument("--scale_input", type=str2bool, default=True)
    parser.add_argument("--n_clusters", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--masked_data_weight", type=float, default=0.75)
    parser.add_argument("--mask_loss_weight", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--mix_mode", default="reliability", choices=["none", "fixed", "mutual", "reliability", "random", "far", "attention"])
    parser.add_argument("--allow_attention_phase2", type=str2bool, default=False)
    parser.add_argument("--gate_mode", default="topology", choices=["none", "constant", "topology", "learned"])
    parser.add_argument("--edge_reliability_mode", default="sim_mutual_snn_distance")
    parser.add_argument("--neighbor_k", type=int, default=10)
    parser.add_argument("--mix_neighbors", type=int, default=4)
    parser.add_argument("--neighbor_estimator", default="current", choices=["current", "uniform_sample", "full"])
    parser.add_argument("--stress_bad_edge_ratio", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--knn_pca_dim", type=int, default=50)
    parser.add_argument("--pseudo_weight", type=float, default=0.3)
    parser.add_argument("--gate_max", type=float, default=0.15)
    parser.add_argument("--gate_min", type=float, default=0.0)
    parser.add_argument("--gamma_sim", type=float, default=1.0)
    parser.add_argument("--gamma_mutual", type=float, default=1.0)
    parser.add_argument("--gamma_snn", type=float, default=1.0)
    parser.add_argument("--gamma_distance", type=float, default=1.0)
    parser.add_argument("--beta_mutual", type=float, default=1.0)
    parser.add_argument("--beta_snn", type=float, default=1.0)
    parser.add_argument("--beta_perturb", type=float, default=2.0)
    parser.add_argument("--beta_uncertainty", type=float, default=1.0)
    parser.add_argument("--contrast_weight", type=float, default=0.0)
    parser.add_argument("--contrast_temperature", type=float, default=0.5)
    parser.add_argument("--contrast_start_epoch", type=int, default=20)
    parser.add_argument("--contrast_neighbor_positive_weight", type=float, default=0.0)
    parser.add_argument("--contrast_hard_negative_weight", type=float, default=0.0)
    parser.add_argument("--contrast_projection_dim", type=int, default=0)
    parser.add_argument("--contrast_min_negatives", type=int, default=16)
    parser.add_argument("--contrast_partition_mode", default="kmeans", choices=["none", "kmeans"])
    parser.add_argument("--save_neighbor_diagnostics", type=str2bool, default=True)
    parser.add_argument("--save_gate_diagnostics", type=str2bool, default=True)
    parser.add_argument("--save_embedding_geometry", type=str2bool, default=True)
    parser.add_argument("--save_per_cell_type_metrics", type=str2bool, default=True)
    parser.add_argument("--skip_eval", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_save_h5ad", action="store_true")
    parser.add_argument("--lightweight_outputs", action="store_true")
    return parser.parse_args()


def make_loaders(data_np: np.ndarray, labels: np.ndarray, batch_size: int, seed: int):
    dataset = family.IndexedExpressionDataset(data_np, labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator)
    eval_loader = DataLoader(dataset, batch_size=max(batch_size * 4, 512), shuffle=False, drop_last=False)
    return train_loader, eval_loader


def build_contrast_partitions(graph, n_clusters: int, seed: int, mode: str) -> tuple[np.ndarray | None, dict]:
    if mode == "none" or n_clusters <= 1 or graph.embedding.shape[0] <= 1:
        return None, {"contrast_partition_mode": mode, "enabled": False}
    from sklearn.cluster import KMeans

    n_partitions = max(2, min(int(n_clusters), int(graph.embedding.shape[0])))
    labels = KMeans(n_clusters=n_partitions, n_init=10, random_state=int(seed)).fit_predict(graph.embedding)
    values, counts = np.unique(labels, return_counts=True)
    return labels.astype(np.int64), {
        "contrast_partition_mode": mode,
        "enabled": True,
        "n_partitions": int(values.size),
        "min_partition_size": int(counts.min()) if counts.size else 0,
        "max_partition_size": int(counts.max()) if counts.size else 0,
    }


def safe_contrastive_loss(
    model: AutoEncoder,
    anchor_latent: torch.Tensor,
    mixed_latent: torch.Tensor,
    batch_indices: np.ndarray,
    graph,
    edge_weights: np.ndarray,
    partition_labels: np.ndarray | None,
    temperature: float,
    neighbor_positive_weight: float,
    hard_negative_weight: float,
    min_negatives: int,
) -> tuple[torch.Tensor, dict]:
    device = anchor_latent.device
    dtype = anchor_latent.dtype
    batch_size = int(anchor_latent.shape[0])
    zero = torch.zeros((), dtype=dtype, device=device)
    empty_stats = {
        "contrast_loss": 0.0,
        "mean_positives": 0.0,
        "mean_negatives": 0.0,
        "mean_hard_negatives": 0.0,
        "valid_batch_fraction": 0.0,
        "excluded_knn_fraction": 0.0,
        "excluded_reverse_knn_fraction": 0.0,
        "excluded_mutual_knn_fraction": 0.0,
        "excluded_same_partition_fraction": 0.0,
        "invalid_reason": "batch_too_small" if batch_size <= 1 else "no_valid_rows",
    }
    if batch_size <= 1:
        return zero, empty_stats

    idx = np.asarray(batch_indices, dtype=np.int64)
    batch_pos = {int(cell): pos for pos, cell in enumerate(idx.tolist())}
    eye_np = np.eye(batch_size, dtype=bool)
    knn_np = np.zeros((batch_size, batch_size), dtype=bool)
    mutual_np = np.zeros((batch_size, batch_size), dtype=bool)
    reliability_np = np.zeros((batch_size, batch_size), dtype=np.float32)
    if graph.indices.shape[1] > 0:
        for row_pos, cell in enumerate(idx.tolist()):
            cell = int(cell)
            for local_pos, nb in enumerate(graph.indices[cell].tolist()):
                col_pos = batch_pos.get(int(nb))
                if col_pos is None:
                    continue
                knn_np[row_pos, col_pos] = True
                mutual_np[row_pos, col_pos] = bool(graph.mutual[cell, local_pos])
                reliability_np[row_pos, col_pos] = float(edge_weights[cell, local_pos])
    reverse_knn_np = knn_np.T

    if partition_labels is not None and idx.size and len(partition_labels) > int(idx.max()):
        partitions = np.asarray(partition_labels, dtype=np.int64)[idx]
        same_partition_np = partitions[:, None] == partitions[None, :]
    else:
        same_partition_np = np.zeros((batch_size, batch_size), dtype=bool)
    different_partition_np = ~same_partition_np if partition_labels is not None else ~eye_np

    neighbor_positive_np = knn_np & mutual_np & same_partition_np & (~eye_np)
    pos_weight_np = np.where(eye_np, 1.0, 0.0).astype(np.float32)
    if float(neighbor_positive_weight) > 0.0:
        pos_weight_np = pos_weight_np + neighbor_positive_np.astype(np.float32) * float(neighbor_positive_weight)

    hard_negative_np = knn_np & different_partition_np & (~eye_np)
    safe_negative_np = different_partition_np & (~eye_np) & (~knn_np) & (~reverse_knn_np)
    neg_weight_np = safe_negative_np.astype(np.float32)
    if float(hard_negative_weight) > 0.0:
        neg_weight_np = neg_weight_np + hard_negative_np.astype(np.float32) * float(hard_negative_weight)

    valid_np = (pos_weight_np.sum(axis=1) > 0.0) & (neg_weight_np.sum(axis=1) >= max(1, int(min_negatives)))
    if not np.any(valid_np):
        stats = dict(empty_stats)
        stats.update(
            {
                "mean_positives": float(pos_weight_np.sum(axis=1).mean()),
                "mean_negatives": float(neg_weight_np.sum(axis=1).mean()),
                "mean_hard_negatives": float(hard_negative_np.sum(axis=1).mean()),
                "excluded_knn_fraction": float(knn_np.mean()),
                "excluded_reverse_knn_fraction": float(reverse_knn_np.mean()),
                "excluded_mutual_knn_fraction": float(mutual_np.mean()),
                "excluded_same_partition_fraction": float((same_partition_np & ~eye_np).mean()),
            }
        )
        return zero, stats

    candidate_weight_np = pos_weight_np + neg_weight_np
    z_anchor = F.normalize(model.contrast_projection(anchor_latent), dim=1, p=2)
    z_mixed = F.normalize(model.contrast_projection(mixed_latent), dim=1, p=2)
    logits = torch.matmul(z_anchor, z_mixed.t()) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits)

    pos_weight = torch.as_tensor(pos_weight_np, dtype=dtype, device=device)
    candidate_weight = torch.as_tensor(candidate_weight_np, dtype=dtype, device=device)
    valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
    numerator = (exp_logits * pos_weight).sum(dim=1).clamp_min(1e-12)
    denominator = (exp_logits * candidate_weight).sum(dim=1).clamp_min(1e-12)
    row_loss = -torch.log(numerator / denominator)
    loss = row_loss[valid].mean()
    if not torch.isfinite(loss):
        stats = dict(empty_stats)
        stats["invalid_reason"] = "non_finite_loss"
        return zero, stats

    stats = {
        "contrast_loss": float(loss.detach().cpu()),
        "mean_positives": float(pos_weight_np[valid_np].sum(axis=1).mean()),
        "mean_negatives": float(neg_weight_np[valid_np].sum(axis=1).mean()),
        "mean_hard_negatives": float(hard_negative_np[valid_np].sum(axis=1).mean()),
        "valid_batch_fraction": float(np.mean(valid_np)),
        "excluded_knn_fraction": float(knn_np.mean()),
        "excluded_reverse_knn_fraction": float(reverse_knn_np.mean()),
        "excluded_mutual_knn_fraction": float(mutual_np.mean()),
        "excluded_same_partition_fraction": float((same_partition_np & ~eye_np).mean()),
        "mean_batch_edge_reliability": float(reliability_np[knn_np].mean()) if np.any(knn_np) else 0.0,
        "invalid_reason": "",
    }
    return loss, stats


def main():
    args = parse_args()
    if not 0.0 <= float(args.stress_bad_edge_ratio) <= 1.0:
        raise ValueError("--stress_bad_edge_ratio must be between 0 and 1")
    contrast_variant = args.variant_name in {
        "rg_neighbormix_scmae_contrast_safe",
        "rg_nm_v1_reliability_contrast_safe",
        "topogate_contrast_safe",
    }
    if contrast_variant:
        if float(args.contrast_weight) <= 0.0:
            args.contrast_weight = 0.01
        if int(args.contrast_projection_dim) <= 0:
            args.contrast_projection_dim = 64
        args.contrast_temperature = max(float(args.contrast_temperature), 1e-6)
        args.contrast_start_epoch = max(1, int(args.contrast_start_epoch))
    else:
        args.contrast_weight = 0.0
        args.contrast_projection_dim = 0
    if args.lightweight_outputs:
        args.save_neighbor_diagnostics = False
    if args.mix_mode == "attention" and not args.allow_attention_phase2:
        raise ValueError("--mix_mode attention is reserved for phase 2. Pass --allow_attention_phase2 true only after phase-1 criteria pass.")
    if args.hidden_dim and not args.hidden_size:
        args.hidden_size = args.hidden_dim
    if args.latent_dim:
        args.hidden_size = args.latent_dim
    if args.mix_mode == "none":
        args.gate_mode = "none"
        args.pseudo_weight = 0.0
        args.neighbor_k = 0
    if args.mix_mode == "fixed":
        args.gate_mode = "constant"
        args.edge_reliability_mode = "none"

    family.set_seed(args.seed)
    rng = np.random.default_rng(args.seed + 3089)
    save_dir = Path(ensure_dir(args.save_dir))
    save_json(vars(args), str(save_dir / "args.json"))
    device = family.get_device(args.gpu, args.no_cuda)
    print(f"Using device: {device}", flush=True)

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
    if not args.lightweight_outputs:
        np.save(save_dir / "gene_names.npy", bundle.gene_names.astype(str))

    graph = build_pca_knn_graph(
        data_np,
        k=args.neighbor_k,
        pca_dim=args.knn_pca_dim,
        tau=args.tau,
        seed=args.seed,
        labels=labels if args.stress_bad_edge_ratio > 0.0 else None,
        stress_bad_edge_ratio=args.stress_bad_edge_ratio,
    )
    graph.profile["neighbor_estimator"] = args.neighbor_estimator
    edge_reliability, edge_weights, edge_summary = compute_edge_reliability(
        graph,
        mode=args.edge_reliability_mode,
        gamma_sim=args.gamma_sim,
        gamma_mutual=args.gamma_mutual,
        gamma_snn=args.gamma_snn,
        gamma_distance=args.gamma_distance,
    )
    node_gate, normalized_gate, gate_summary = compute_node_gate(
        graph,
        edge_weights=edge_weights,
        gate_mode=args.gate_mode,
        gate_min=args.gate_min,
        gate_max=args.gate_max,
        beta_mutual=args.beta_mutual,
        beta_snn=args.beta_snn,
        beta_perturb=args.beta_perturb,
        beta_uncertainty=args.beta_uncertainty,
        uncertainty=None,
    )
    random_neighbors = build_random_neighbors(data_np.shape[0], max(1, min(args.mix_neighbors, max(1, data_np.shape[0] - 1))), rng, graph.indices)
    far_neighbors = build_far_neighbors(graph.embedding, max(1, min(args.mix_neighbors, max(1, data_np.shape[0] - 1))), rng)
    perturb_proxy = (1.0 - np.sum(graph.probs * graph.similarity, axis=1)).astype(np.float32) if graph.probs.size else np.zeros(data_np.shape[0], dtype=np.float32)
    contrast_partitions, contrast_partition_summary = build_contrast_partitions(
        graph=graph,
        n_clusters=n_clusters,
        seed=args.seed,
        mode=args.contrast_partition_mode if contrast_variant else "none",
    )

    if args.save_neighbor_diagnostics:
        np.save(save_dir / "neighbor_indices.npy", graph.indices)
        np.save(save_dir / "neighbor_base_probs.npy", graph.probs)
        np.save(save_dir / "neighbor_similarity.npy", graph.similarity)
        np.save(save_dir / "neighbor_distance.npy", graph.distance)
        save_json(graph.profile, str(save_dir / "neighbor_graph_profile.json"))
        save_json(graph.profile, str(save_dir / "train_knn_diagnostics.json"))
    if not args.lightweight_outputs:
        np.save(save_dir / "edge_reliability.npy", edge_reliability)
        np.save(save_dir / "node_gate.npy", node_gate)
        np.save(save_dir / "pseudo_perturbation.npy", perturb_proxy)
    if contrast_partitions is not None and not args.lightweight_outputs:
        np.save(save_dir / "contrast_partition_labels.npy", contrast_partitions.astype(np.int64))
    save_json({"edge_reliability_mode": args.edge_reliability_mode, **edge_summary}, str(save_dir / "edge_weight_summary.json"))
    save_json({"edge_reliability_mode": args.edge_reliability_mode, **edge_summary}, str(save_dir / "neighbor_reliability_summary.json"))
    save_json(gate_summary, str(save_dir / "gate_summary.json"))
    save_json(contrast_partition_summary, str(save_dir / "contrast_partition_summary.json"))
    save_json(
        {
            "mean_pseudo_perturbation": float(np.mean(perturb_proxy)) if perturb_proxy.size else 0.0,
            "p95_pseudo_perturbation": float(np.percentile(perturb_proxy, 95)) if perturb_proxy.size else 0.0,
        },
        str(save_dir / "pseudo_perturbation_summary.json"),
    )

    train_loader, eval_loader = make_loaders(data_np, labels, args.batch_size, args.seed)
    model = AutoEncoder(
        num_genes=data_np.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        masked_data_weight=args.masked_data_weight,
        mask_loss_weight=args.mask_loss_weight,
        contrast_projection_dim=args.contrast_projection_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pseudo_enabled = args.mix_mode != "none" and float(args.pseudo_weight) > 0 and int(args.mix_neighbors) > 0
    contrast_enabled = bool(contrast_variant and pseudo_enabled and float(args.contrast_weight) > 0.0)

    history = {
        "loss": [],
        "real_loss": [],
        "real_reconstruction_loss": [],
        "real_mask_loss": [],
        "pseudo_loss": [],
        "pseudo_reconstruction_loss": [],
        "pseudo_mask_loss": [],
        "contrast_loss": [],
        "contrast_mean_positives": [],
        "contrast_mean_negatives": [],
        "contrast_hard_negatives": [],
        "contrast_valid_batch_fraction": [],
        "contrast_excluded_knn_fraction": [],
        "contrast_excluded_reverse_knn_fraction": [],
        "contrast_excluded_mutual_knn_fraction": [],
        "contrast_excluded_same_partition_fraction": [],
        "mean_node_gate": [],
        "mean_pseudo_perturbation": [],
        "real_mask_rate": [],
        "pseudo_mask_rate": [],
        "mix_mode": args.mix_mode,
        "pseudo_enabled": bool(pseudo_enabled),
        "contrast_enabled": bool(contrast_enabled),
    }

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        totals = {key: 0.0 for key in history if isinstance(history[key], list)}
        n_batches = 0
        for idx_t, x_cpu, _ in train_loader:
            idx_np = idx_t.numpy().astype(np.int64, copy=False)
            x = x_cpu.to(device)
            x_corrupt, real_mask = family.apply_scmae_noise(x, args.mask_ratio)
            _, real_loss, real_parts = model.loss_mask_weighted(x_corrupt, x, real_mask, mask_loss_scale=1.0)
            loss = real_loss
            pseudo_loss = torch.zeros((), dtype=real_loss.dtype, device=device)
            pseudo_parts = {"reconstruction_loss": pseudo_loss, "mask_loss": pseudo_loss, "mask_positive_rate": pseudo_loss}
            contrast_loss_value = torch.zeros((), dtype=real_loss.dtype, device=device)
            contrast_stats = {
                "mean_positives": 0.0,
                "mean_negatives": 0.0,
                "mean_hard_negatives": 0.0,
                "valid_batch_fraction": 0.0,
                "excluded_knn_fraction": 0.0,
                "excluded_reverse_knn_fraction": 0.0,
                "excluded_mutual_knn_fraction": 0.0,
                "excluded_same_partition_fraction": 0.0,
            }
            mix_info = {"mean_node_gate": 0.0, "mean_perturb_norm": 0.0}
            if pseudo_enabled:
                x_prime, sample_weight, mix_info = make_pseudo_batch(
                    data_np=data_np,
                    batch_indices=idx_np,
                    batch_x=x,
                    mix_mode=args.mix_mode,
                    graph=graph,
                    edge_weights=edge_weights,
                    node_gate=node_gate,
                    mix_neighbors=args.mix_neighbors,
                    rng=rng,
                    random_neighbors=random_neighbors,
                    far_neighbors=far_neighbors,
                    neighbor_estimator=args.neighbor_estimator,
                )
                xp_corrupt, pseudo_mask = family.apply_scmae_noise(x_prime, args.mask_ratio)
                _, pseudo_loss, pseudo_parts = model.loss_mask_weighted(
                    xp_corrupt,
                    x,
                    pseudo_mask,
                    sample_weight=sample_weight,
                    mask_loss_scale=1.0,
                )
                loss = loss + float(args.pseudo_weight) * pseudo_loss
                if contrast_enabled and epoch >= int(args.contrast_start_epoch):
                    anchor_latent = model.encoder(x)
                    mixed_latent = model.encoder(x_prime)
                    contrast_loss_value, contrast_stats = safe_contrastive_loss(
                        model=model,
                        anchor_latent=anchor_latent,
                        mixed_latent=mixed_latent,
                        batch_indices=idx_np,
                        graph=graph,
                        edge_weights=edge_weights,
                        partition_labels=contrast_partitions,
                        temperature=args.contrast_temperature,
                        neighbor_positive_weight=args.contrast_neighbor_positive_weight,
                        hard_negative_weight=args.contrast_hard_negative_weight,
                        min_negatives=args.contrast_min_negatives,
                    )
                    loss = loss + float(args.contrast_weight) * contrast_loss_value

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["real_loss"] += float(real_loss.detach().cpu())
            totals["real_reconstruction_loss"] += float(real_parts["reconstruction_loss"].cpu())
            totals["real_mask_loss"] += float(real_parts["mask_loss"].cpu())
            totals["pseudo_loss"] += float(pseudo_loss.detach().cpu())
            totals["pseudo_reconstruction_loss"] += float(pseudo_parts["reconstruction_loss"].cpu())
            totals["pseudo_mask_loss"] += float(pseudo_parts["mask_loss"].cpu())
            totals["contrast_loss"] += float(contrast_loss_value.detach().cpu())
            totals["contrast_mean_positives"] += float(contrast_stats.get("mean_positives", 0.0))
            totals["contrast_mean_negatives"] += float(contrast_stats.get("mean_negatives", 0.0))
            totals["contrast_hard_negatives"] += float(contrast_stats.get("mean_hard_negatives", 0.0))
            totals["contrast_valid_batch_fraction"] += float(contrast_stats.get("valid_batch_fraction", 0.0))
            totals["contrast_excluded_knn_fraction"] += float(contrast_stats.get("excluded_knn_fraction", 0.0))
            totals["contrast_excluded_reverse_knn_fraction"] += float(contrast_stats.get("excluded_reverse_knn_fraction", 0.0))
            totals["contrast_excluded_mutual_knn_fraction"] += float(contrast_stats.get("excluded_mutual_knn_fraction", 0.0))
            totals["contrast_excluded_same_partition_fraction"] += float(contrast_stats.get("excluded_same_partition_fraction", 0.0))
            totals["mean_node_gate"] += float(mix_info["mean_node_gate"])
            totals["mean_pseudo_perturbation"] += float(mix_info["mean_perturb_norm"])
            totals["real_mask_rate"] += float(real_mask.mean().detach().cpu())
            totals["pseudo_mask_rate"] += float(pseudo_parts["mask_positive_rate"].cpu()) if pseudo_enabled else 0.0
            n_batches += 1

        for key, value in totals.items():
            history[key].append(value / max(1, n_batches))
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} loss={history['loss'][-1]:.4f} "
                f"real={history['real_loss'][-1]:.4f} pseudo={history['pseudo_loss'][-1]:.4f} "
                f"contrast={history['contrast_loss'][-1]:.4f} gate={history['mean_node_gate'][-1]:.4f}",
                flush=True,
            )

    embedding, labels_out = family.extract_embedding(model, eval_loader, device)
    save_json(history, str(save_dir / "training_history.json"))
    if not args.lightweight_outputs:
        np.save(save_dir / "embedding_final.npy", embedding.astype(np.float32))
        np.save(save_dir / "embeddings_base.npy", embedding.astype(np.float32))
        np.save(save_dir / "labels.npy", labels_out.astype(np.int64))
        family.save_embedding_h5(save_dir / "embedding.h5", embedding, labels_out)
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "gene_names": bundle.gene_names.astype(str),
                "neighbor_graph_profile": graph.profile,
                "edge_weight_summary": edge_summary,
                "gate_summary": gate_summary,
                "contrast_partition_summary": contrast_partition_summary,
                "contrast_enabled": bool(contrast_enabled),
            },
            save_dir / "model.pt",
        )

    contrast_diag = {
        "contrast_enabled": bool(contrast_enabled),
        "contrast_variant": bool(contrast_variant),
        "contrast_weight": float(args.contrast_weight),
        "contrast_temperature": float(args.contrast_temperature),
        "contrast_start_epoch": int(args.contrast_start_epoch),
        "contrast_neighbor_positive_weight": float(args.contrast_neighbor_positive_weight),
        "contrast_hard_negative_weight": float(args.contrast_hard_negative_weight),
        "contrast_projection_dim": int(args.contrast_projection_dim) if contrast_enabled else 0,
        "contrast_min_negatives": int(args.contrast_min_negatives),
        "contrast_partition_mode": args.contrast_partition_mode if contrast_variant else "none",
        "contrast_partition_summary": contrast_partition_summary,
        "mean_contrast_loss": float(np.mean(history["contrast_loss"])) if history["contrast_loss"] else 0.0,
        "final_contrast_loss": float(history["contrast_loss"][-1]) if history["contrast_loss"] else 0.0,
        "mean_positives": float(np.mean(history["contrast_mean_positives"])) if history["contrast_mean_positives"] else 0.0,
        "mean_negatives": float(np.mean(history["contrast_mean_negatives"])) if history["contrast_mean_negatives"] else 0.0,
        "mean_hard_negatives": float(np.mean(history["contrast_hard_negatives"])) if history["contrast_hard_negatives"] else 0.0,
        "mean_valid_batch_fraction": float(np.mean(history["contrast_valid_batch_fraction"])) if history["contrast_valid_batch_fraction"] else 0.0,
        "mean_excluded_knn_fraction": float(np.mean(history["contrast_excluded_knn_fraction"])) if history["contrast_excluded_knn_fraction"] else 0.0,
        "mean_excluded_reverse_knn_fraction": float(np.mean(history["contrast_excluded_reverse_knn_fraction"])) if history["contrast_excluded_reverse_knn_fraction"] else 0.0,
        "mean_excluded_mutual_knn_fraction": float(np.mean(history["contrast_excluded_mutual_knn_fraction"])) if history["contrast_excluded_mutual_knn_fraction"] else 0.0,
        "mean_excluded_same_partition_fraction": float(np.mean(history["contrast_excluded_same_partition_fraction"])) if history["contrast_excluded_same_partition_fraction"] else 0.0,
    }
    save_json(contrast_diag, str(save_dir / "contrast_diagnostics.json"))

    result = None
    pred = None
    if not args.skip_eval:
        result = family.write_kmeans_known_k_outputs(
            output_dir=save_dir,
            dataset=dataset_name,
            method=args.method_name,
            seed=args.seed,
            embedding=embedding,
            labels=labels_out,
            n_clusters=n_clusters,
            extra={
                "variant": args.variant_name,
                "mix_mode": args.mix_mode,
                "gate_mode": args.gate_mode,
                "edge_reliability_mode": args.edge_reliability_mode,
                "neighbor_estimator": args.neighbor_estimator,
                "stress_bad_edge_ratio": float(args.stress_bad_edge_ratio),
                "label_leakage_diagnostic": bool(args.stress_bad_edge_ratio > 0.0),
                "gate_max": float(args.gate_max),
                "pseudo_weight": float(args.pseudo_weight),
                "contrast_enabled": bool(contrast_enabled),
                "contrast_weight": float(args.contrast_weight),
                "contrast_temperature": float(args.contrast_temperature),
                "contrast_valid_batch_fraction": contrast_diag["mean_valid_batch_fraction"],
            },
        )
        pred = result["preds"]["kmeans_known_k"]
        save_json(result["fixed"], str(save_dir / "metrics.json"))
    else:
        save_json({}, str(save_dir / "metrics.json"))

    if args.save_embedding_geometry:
        pd.DataFrame([embedding_geometry(embedding, labels_out)]).to_csv(save_dir / "embedding_geometry_summary.csv", index=False)
    if pred is not None and args.save_per_cell_type_metrics:
        pct = per_cell_type_metrics(labels_out, pred)
        pct.to_csv(save_dir / "per_cell_type_metrics.csv", index=False)
        mapped = mapped_predictions(labels_out, pred)
        pd.crosstab(pd.Series(labels_out, name="true"), pd.Series(pred, name="pred")).to_csv(save_dir / "confusion_matrix_raw.csv")
        pd.crosstab(pd.Series(labels_out, name="true"), pd.Series(mapped, name="mapped")).to_csv(save_dir / "confusion_matrix_mapped.csv")
        rare = pct.groupby("is_rare_lt_50")[["precision", "recall", "f1"]].mean().reset_index()
        rare.to_csv(save_dir / "rare_cell_effect_summary.csv", index=False)

    summary = {
        "dataset": dataset_name,
        "method": args.method_name,
        "variant": args.variant_name,
        "seed": int(args.seed),
        "n_cells": int(data_np.shape[0]),
        "n_genes": int(data_np.shape[1]),
        "n_clusters": int(n_clusters),
        "mix_mode": args.mix_mode,
        "neighbor_estimator": args.neighbor_estimator,
        "stress_bad_edge_ratio": float(args.stress_bad_edge_ratio),
        "pseudo_enabled": bool(pseudo_enabled),
        "edge_weight_summary": edge_summary,
        "gate_summary": gate_summary,
        "contrast_enabled": bool(contrast_enabled),
        "contrast_diagnostics": contrast_diag,
        "lightweight_outputs": bool(args.lightweight_outputs),
        "fixed_metrics": result["fixed"] if result is not None else {},
        "label_leakage": bool(args.stress_bad_edge_ratio > 0.0),
        "label_leakage_diagnostic": bool(args.stress_bad_edge_ratio > 0.0),
    }
    save_json(summary, str(save_dir / "summary.json"))

    if not args.no_save_h5ad:
        bundle.adata.obsm["X_scvicar"] = embedding
        bundle.adata.uns["scvicar"] = summary
        sanitize_anndata_for_write(bundle.adata)
        bundle.adata.write_h5ad(save_dir / "adata_scvicar.h5ad", compression="gzip")

    print(f"Results saved to: {save_dir}", flush=True)


if __name__ == "__main__":
    main()
