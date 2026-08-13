from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, TensorDataset

from .config import V19Config
from .graph import (
    NeighborGraph,
    build_far_neighbors,
    build_pca_knn_graph,
    build_random_neighbors,
    compute_edge_reliability,
)
from .mixing import compute_node_gate, make_pseudo_batch
from .model import WeightedAutoEncoder, apply_scmae_noise


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if resolved.index is None:
            raise ValueError("CUDA device must include a logical index, for example cuda:0")
        torch.cuda.set_device(resolved)
    return resolved


def _seed_runtime(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.random.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.default_generators[int(device.index)].manual_seed(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a device-local stream so pseudo and real masks cannot perturb one another."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _empty_graph(n_samples: int) -> NeighborGraph:
    empty_i = np.zeros((n_samples, 0), dtype=np.int64)
    empty_f = np.zeros((n_samples, 0), dtype=np.float32)
    return NeighborGraph(
        indices=empty_i,
        probs=empty_f,
        similarity=empty_f,
        distance=empty_f,
        embedding=np.zeros((n_samples, 0), dtype=np.float32),
        mutual=empty_f.astype(bool),
        snn=empty_f,
        profile={
            "neighbor_k": 0,
            "graph_enabled": False,
            "label_leakage_diagnostic": False,
        },
    )


@torch.no_grad()
def _extract_embedding(
    model: WeightedAutoEncoder,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(X, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=max(int(batch_size) * 4, 512),
        shuffle=False,
        drop_last=False,
    )
    rows = [model.feature(batch[0].to(device)).detach().cpu().numpy() for batch in loader]
    embedding = np.concatenate(rows, axis=0).astype(np.float32)
    return np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)


def _neighbor_overlap(
    reference_indices: np.ndarray,
    embedding: np.ndarray,
) -> float:
    """Measure X-only local-neighborhood preservation in the final embedding."""
    if reference_indices.ndim != 2 or reference_indices.shape[1] == 0:
        return 0.0
    n_samples = int(embedding.shape[0])
    k = min(int(reference_indices.shape[1]), max(1, n_samples - 1))
    normalized_embedding = normalize(
        np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0), axis=1
    ).astype(np.float32)
    nearest = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nearest.fit(normalized_embedding)
    _, raw_indices = nearest.kneighbors(normalized_embedding)
    overlaps = []
    for sample in range(n_samples):
        candidate = [int(value) for value in raw_indices[sample] if int(value) != sample]
        candidate_set = set(candidate[:k])
        reference_set = set(int(value) for value in reference_indices[sample, :k])
        overlaps.append(len(candidate_set.intersection(reference_set)) / float(k))
    return float(np.mean(overlaps)) if overlaps else 0.0


@torch.no_grad()
def _evaluate_unsupervised_views(
    model: WeightedAutoEncoder,
    data_np: np.ndarray,
    clean_embedding: np.ndarray,
    graph: NeighborGraph,
    batch_size: int,
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate masked recovery and latent stability without labels or K."""
    model.eval()
    n_samples = int(data_np.shape[0])
    view_cosines: list[float] = []
    total_loss = 0.0
    total_count = 0
    for view_id in range(2):
        view_rng = np.random.default_rng(int(seed) + 100_003 + view_id * 7_919)
        latent_rows: list[np.ndarray] = []
        for start in range(0, n_samples, max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), n_samples)
            clean = np.asarray(data_np[start:stop], dtype=np.float32)
            replacement_indices = (np.arange(start, stop, dtype=np.int64) + view_id + 1) % n_samples
            replacement = np.asarray(data_np[replacement_indices], dtype=np.float32)
            selected = view_rng.random(clean.shape) < float(mask_ratio)
            corrupted = np.where(selected, replacement, clean).astype(np.float32, copy=False)
            effective_mask = (corrupted != clean).astype(np.float32, copy=False)
            batch = torch.as_tensor(corrupted, dtype=torch.float32, device=device)
            target = torch.as_tensor(clean, dtype=torch.float32, device=device)
            mask = torch.as_tensor(effective_mask, dtype=torch.float32, device=device)
            latent, loss, _parts = model.loss_mask_weighted(batch, target, mask)
            latent_rows.append(latent.detach().cpu().numpy().astype(np.float32, copy=False))
            total_loss += float(loss.detach().cpu()) * int(stop - start)
            total_count += int(stop - start)
        view_embedding = np.concatenate(latent_rows, axis=0)
        clean_norm = np.linalg.norm(clean_embedding, axis=1)
        view_norm = np.linalg.norm(view_embedding, axis=1)
        cosine = np.sum(clean_embedding * view_embedding, axis=1) / np.clip(
            clean_norm * view_norm, 1e-8, None
        )
        view_cosines.append(float(np.mean(np.nan_to_num(cosine, nan=0.0))))
    latent_std = float(np.mean(np.std(clean_embedding, axis=0)))
    return {
        "eval_mask_loss": float(total_loss / max(1, total_count * 2)),
        "latent_view_cosine_mean": float(np.mean(view_cosines)) if view_cosines else 0.0,
        "latent_view_cosine_std": float(np.std(view_cosines)) if view_cosines else 0.0,
        "input_neighbor_overlap": _neighbor_overlap(graph.indices, clean_embedding),
        "latent_mean_feature_std": latent_std,
    }


def _evaluation_graph(
    data_np: np.ndarray,
    config: V19Config,
    seed: int,
) -> NeighborGraph:
    """Build a diagnostic-only graph from held-out rows, never the training graph."""
    if data_np.shape[0] < 3:
        return _empty_graph(int(data_np.shape[0]))
    return build_pca_knn_graph(
        data_np,
        k=min(int(config.neighbor_k), int(data_np.shape[0]) - 1),
        pca_dim=int(config.knn_pca_dim),
        tau=float(config.tau),
        seed=int(seed),
    )


def fit_predict(
    X: np.ndarray,
    *,
    n_clusters: int | None,
    config: V19Config,
    seed: int,
    device: str | torch.device,
    evaluate_unsupervised: bool = False,
    fit_X: np.ndarray | None = None,
    evaluation_X: np.ndarray | None = None,
    evaluation_mask_ratio: float | None = None,
    evaluation_graph_config: V19Config | None = None,
    precomputed_graph_embeddings: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Fit V19 without labels; optional K is used only by the final KMeans readout.

    ``fit_X`` and ``evaluation_X`` are an opt-in protocol for label-free tuning:
    the model and RG graph are fitted on ``fit_X`` while corruption, stability,
    and neighborhood diagnostics are measured on ``evaluation_X`` rows.  The
    optional evaluation arguments keep those diagnostics paired when a tuning
    candidate changes the training mask or graph hyperparameters.  The default
    path is unchanged for the formal benchmark runner.
    """
    data_np = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    if data_np.ndim != 2 or data_np.shape[0] < 2 or data_np.shape[1] == 0:
        raise ValueError("X must contain at least two samples and one feature")
    if not np.all(np.isfinite(data_np)):
        raise ValueError("X contains non-finite values after preprocessing")
    if n_clusters is not None and (int(n_clusters) <= 0 or int(n_clusters) > data_np.shape[0]):
        raise ValueError("n_clusters must be in [1, n_samples]")
    fit_data_np = data_np if fit_X is None else np.ascontiguousarray(np.asarray(fit_X, dtype=np.float32))
    if fit_data_np.ndim != 2 or fit_data_np.shape[0] < 2 or fit_data_np.shape[1] != data_np.shape[1]:
        raise ValueError("fit_X must have at least two rows and the same feature width as X")
    if not np.all(np.isfinite(fit_data_np)):
        raise ValueError("fit_X contains non-finite values")
    evaluation_np = None if evaluation_X is None else np.ascontiguousarray(np.asarray(evaluation_X, dtype=np.float32))
    if evaluation_np is not None:
        if evaluation_np.ndim != 2 or evaluation_np.shape[0] < 2 or evaluation_np.shape[1] != data_np.shape[1]:
            raise ValueError("evaluation_X must have at least two rows and the same feature width as X")
        if not np.all(np.isfinite(evaluation_np)):
            raise ValueError("evaluation_X contains non-finite values")
    if evaluation_mask_ratio is not None and not 0.0 <= float(evaluation_mask_ratio) < 1.0:
        raise ValueError("evaluation_mask_ratio must be in [0, 1)")
    if evaluation_graph_config is not None and evaluation_graph_config.variant not in {"rg_full", "scmae_only"}:
        raise ValueError("evaluation_graph_config must be a V19Config")
    runtime_device = _resolve_device(device)
    _seed_runtime(int(seed), runtime_device)
    rng = np.random.default_rng(int(seed) + 3089)
    graph_enabled = config.variant == "rg_full"

    if graph_enabled:
        graph = build_pca_knn_graph(
            fit_data_np,
            k=min(int(config.neighbor_k), int(fit_data_np.shape[0]) - 1),
            pca_dim=config.knn_pca_dim,
            tau=config.tau,
            seed=int(seed),
            precomputed_embedding=(
                None
                if precomputed_graph_embeddings is None
                else precomputed_graph_embeddings.get(int(config.knn_pca_dim))
            ),
        )
        edge_reliability, edge_weights, edge_summary = compute_edge_reliability(
            graph,
            mode="sim_mutual_snn_distance",
            gamma_sim=config.gamma_sim,
            gamma_mutual=config.gamma_mutual,
            gamma_snn=config.gamma_snn,
            gamma_distance=config.gamma_distance,
        )
        node_gate, _, gate_summary = compute_node_gate(
            graph,
            edge_weights=edge_weights,
            gate_mode="topology",
            gate_min=config.gate_min,
            gate_max=config.gate_max,
            beta_mutual=config.beta_mutual,
            beta_snn=config.beta_snn,
            beta_perturb=config.beta_perturb,
            beta_uncertainty=config.beta_uncertainty,
            uncertainty=None,
        )
        # Preserve the original RG RNG stream before reliability-neighbor sampling.
        rng_k = max(1, min(config.mix_neighbors, fit_data_np.shape[0] - 1))
        build_random_neighbors(fit_data_np.shape[0], rng_k, rng, exclude=graph.indices)
        build_far_neighbors(graph.embedding, rng_k, rng)
    else:
        graph = _empty_graph(fit_data_np.shape[0])
        edge_reliability = np.zeros((fit_data_np.shape[0], 0), dtype=np.float32)
        edge_weights = np.zeros((fit_data_np.shape[0], 0), dtype=np.float32)
        node_gate = np.zeros(fit_data_np.shape[0], dtype=np.float32)
        edge_summary = {
            "edge_weight_entropy": 0.0,
            "effective_neighbor_count": 0.0,
            "max_edge_weight_mean": 0.0,
            "max_edge_weight_p95": 0.0,
            "fraction_effective_neighbors_lt_2": 1.0,
        }
        gate_summary = {
            "gate_mode": "none",
            "gate_min": 0.0,
            "gate_max": 0.0,
            "mean_node_gate": 0.0,
            "min_node_gate": 0.0,
            "max_node_gate": 0.0,
            "fraction_gate_lt_0p01": 1.0,
            "fraction_gate_gt_90pct_max": 0.0,
            "uncertainty_enabled": False,
            "uncertainty_source": "disabled",
            "mean_perturb_proxy": 0.0,
        }

    indices = torch.arange(fit_data_np.shape[0], dtype=torch.long)
    tensor = torch.as_tensor(fit_data_np, dtype=torch.float32)
    loader_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    train_loader = DataLoader(
        TensorDataset(indices, tensor),
        batch_size=int(config.batch_size),
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
        num_workers=int(config.num_workers),
    )
    model = WeightedAutoEncoder(
        num_genes=fit_data_np.shape[1],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    pseudo_enabled = graph_enabled and float(config.pseudo_weight) > 0.0
    real_noise_generator = _make_torch_generator(runtime_device, int(seed) + 400_003)
    pseudo_noise_generator = _make_torch_generator(runtime_device, int(seed) + 500_003)
    history: dict[str, Any] = {
        "loss": [],
        "real_loss": [],
        "real_reconstruction_loss": [],
        "real_mask_loss": [],
        "pseudo_loss": [],
        "pseudo_reconstruction_loss": [],
        "pseudo_mask_loss": [],
        "mean_node_gate": [],
        "mean_pseudo_perturbation": [],
        "real_mask_rate": [],
        "pseudo_mask_rate": [],
        "variant": config.variant,
        "mix_mode": "reliability" if graph_enabled else "none",
        "pseudo_enabled": bool(pseudo_enabled),
    }

    tracked = [key for key, value in history.items() if isinstance(value, list)]
    for _epoch in range(1, int(config.epochs) + 1):
        model.train()
        totals = {key: 0.0 for key in tracked}
        n_batches = 0
        for batch_indices_tensor, batch_cpu in train_loader:
            batch_indices = batch_indices_tensor.numpy().astype(np.int64, copy=False)
            batch = batch_cpu.to(runtime_device)
            corrupted, real_mask = apply_scmae_noise(
                batch,
                config.mask_ratio,
                generator=real_noise_generator,
            )
            _, real_loss, real_parts = model.loss_mask_weighted(corrupted, batch, real_mask)
            loss = real_loss
            pseudo_loss = torch.zeros((), dtype=batch.dtype, device=runtime_device)
            pseudo_parts = {
                "reconstruction_loss": pseudo_loss,
                "mask_loss": pseudo_loss,
                "mask_positive_rate": pseudo_loss,
            }
            mix_info = {"mean_node_gate": 0.0, "mean_perturb_norm": 0.0}
            if pseudo_enabled:
                pseudo_batch, sample_weight, mix_info = make_pseudo_batch(
                    data_np=fit_data_np,
                    batch_indices=batch_indices,
                    batch_x=batch,
                    graph=graph,
                    edge_weights=edge_weights,
                    node_gate=node_gate,
                    mix_neighbors=config.mix_neighbors,
                    rng=rng,
                )
                pseudo_corrupted, pseudo_mask = apply_scmae_noise(
                    pseudo_batch,
                    config.mask_ratio,
                    generator=pseudo_noise_generator,
                )
                _, pseudo_loss, pseudo_parts = model.loss_mask_weighted(
                    pseudo_corrupted,
                    batch,
                    pseudo_mask,
                    sample_weight=sample_weight,
                )
                loss = loss + float(config.pseudo_weight) * pseudo_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["real_loss"] += float(real_loss.detach().cpu())
            totals["real_reconstruction_loss"] += float(real_parts["reconstruction_loss"].cpu())
            totals["real_mask_loss"] += float(real_parts["mask_loss"].cpu())
            totals["pseudo_loss"] += float(pseudo_loss.detach().cpu())
            totals["pseudo_reconstruction_loss"] += float(
                pseudo_parts["reconstruction_loss"].cpu()
            )
            totals["pseudo_mask_loss"] += float(pseudo_parts["mask_loss"].cpu())
            totals["mean_node_gate"] += float(mix_info["mean_node_gate"])
            totals["mean_pseudo_perturbation"] += float(mix_info["mean_perturb_norm"])
            totals["real_mask_rate"] += float(real_mask.mean().detach().cpu())
            totals["pseudo_mask_rate"] += (
                float(pseudo_parts["mask_positive_rate"].cpu()) if pseudo_enabled else 0.0
            )
            n_batches += 1
        for key in tracked:
            history[key].append(totals[key] / max(1, n_batches))

    embedding = _extract_embedding(model, data_np, config.batch_size, runtime_device)
    predictions = None
    if n_clusters is not None:
        predictions = KMeans(
            n_clusters=int(n_clusters),
            n_init=int(config.kmeans_n_init),
            random_state=int(seed),
        ).fit_predict(embedding).astype(np.int64)
    perturbation_proxy = (
        (1.0 - np.sum(graph.probs * graph.similarity, axis=1)).astype(np.float32)
        if graph.probs.size
        else np.zeros(data_np.shape[0], dtype=np.float32)
    )
    evaluation_graph = graph
    diagnostic_data = data_np
    diagnostic_embedding = embedding
    if evaluation_np is not None:
        diagnostic_data = evaluation_np
        diagnostic_embedding = _extract_embedding(
            model,
            evaluation_np,
            config.batch_size,
            runtime_device,
        )
        evaluation_graph = _evaluation_graph(
            evaluation_np,
            evaluation_graph_config or config,
            int(seed) + 700_003,
        )
    unsupervised_diagnostics = (
        _evaluate_unsupervised_views(
            model,
            diagnostic_data,
            diagnostic_embedding,
            evaluation_graph,
            config.batch_size,
            config.mask_ratio if evaluation_mask_ratio is None else float(evaluation_mask_ratio),
            int(seed) + (800_003 if evaluation_np is not None else 0),
            runtime_device,
        )
        if evaluate_unsupervised
        else {}
    )
    if evaluation_np is not None:
        unsupervised_diagnostics.update(
            {
                "validation_protocol": "held_out_rows",
                "fit_n_samples": int(fit_data_np.shape[0]),
                "evaluation_n_samples": int(evaluation_np.shape[0]),
                "evaluation_graph_profile": evaluation_graph.profile,
                "evaluation_mask_ratio": (
                    float(config.mask_ratio)
                    if evaluation_mask_ratio is None
                    else float(evaluation_mask_ratio)
                ),
            }
        )
    diagnostics: dict[str, Any] = {
        "neighbor_indices": graph.indices,
        "neighbor_base_probs": graph.probs,
        "neighbor_similarity": graph.similarity,
        "neighbor_distance": graph.distance,
        "edge_reliability": edge_reliability,
        "edge_weights": edge_weights,
        "node_gate": node_gate,
        "pseudo_perturbation": perturbation_proxy,
        "graph_profile": graph.profile,
        "edge_summary": edge_summary,
        "gate_summary": gate_summary,
        "training_history": history,
        "unsupervised_diagnostics": unsupervised_diagnostics,
        "core_summary": {
            "variant": config.variant,
            "seed": int(seed),
            "device": str(runtime_device),
            "n_samples": int(data_np.shape[0]),
            "fit_n_samples": int(fit_data_np.shape[0]),
            "evaluation_n_samples": int(evaluation_np.shape[0]) if evaluation_np is not None else None,
            "n_features": int(data_np.shape[1]),
            "n_clusters": int(n_clusters) if n_clusters is not None else None,
            "graph_enabled": bool(graph_enabled),
            "pseudo_enabled": bool(pseudo_enabled),
            "mix_mode": "reliability" if graph_enabled else "none",
            "gate_mode": "topology" if graph_enabled else "none",
            "edge_reliability_mode": "sim_mutual_snn_distance" if graph_enabled else "none",
            "contrast_enabled": False,
            "K_used_only_in_readout": n_clusters is not None,
            "readout_enabled": n_clusters is not None,
            "labels_used_during_fit": False,
            "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
    }
    return predictions, embedding, diagnostics
