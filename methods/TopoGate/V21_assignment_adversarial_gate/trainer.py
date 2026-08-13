from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import V21Config
from .graph import build_svd_knn_graph, compute_topology_statistics
from .model import (
    FeatureGate,
    StudentTClusterHead,
    V21AutoEncoder,
    coverage_concentration,
    cyclic_donor,
    information_maximization_loss,
    jensen_shannon_divergence,
    random_bernoulli_mask,
    random_topk_mask,
    straight_through_changeable_topk,
)


ALLOWED_PHYSICAL_GPUS = frozenset({1, 2, 3, 4, 5, 6})


def resolve_device(device: str | torch.device, gpu: int | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cpu":
        return torch.device("cpu")
    if gpu is None:
        raise ValueError("CUDA V21 runs require an explicit physical --gpu in 1..6")
    if gpu is not None and int(gpu) not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(f"physical GPU {gpu} is forbidden; allowed={sorted(ALLOWED_PHYSICAL_GPUS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda:0")


def seed_all(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.random.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("V21 requires an indexed logical CUDA device")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(int(seed))


def _set_requires_grad(module: torch.nn.Module | None, enabled: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _grad_norm(module: torch.nn.Module) -> float:
    values = [p.grad.detach().norm().item() for p in module.parameters() if p.grad is not None]
    return float(np.sqrt(np.sum(np.square(values)))) if values else 0.0


def _batch_indices(
    n_samples: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> list[torch.Tensor]:
    order = torch.randperm(n_samples, generator=generator, device=device)
    return [order[start : min(n_samples, start + batch_size)] for start in range(0, n_samples, batch_size)]


def _sample_random_reconstruction_mask(
    shape: tuple[int, int],
    *,
    mask_ratio: float,
    mode: str,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if mode == "bernoulli":
        return random_bernoulli_mask(shape, mask_ratio, device=device, generator=generator)
    if mode == "topk":
        k = max(1, min(shape[1], int(round(mask_ratio * shape[1]))))
        return random_topk_mask(shape, k, device=device, generator=generator)
    raise ValueError(f"unsupported random mask mode: {mode}")


def _clean_embedding(
    model: V21AutoEncoder,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model.encode(batch).cpu().numpy())
    model.train(was_training)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _clean_probabilities(
    model: V21AutoEncoder,
    head: StudentTClusterHead,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    head.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(head(model.encode(batch)).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _assignment_mask_from_scores(
    scores: torch.Tensor,
    batch: torch.Tensor,
    donor: torch.Tensor,
    *,
    config: V21Config,
    generator: torch.Generator,
    gumbel_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eligible = (donor - batch).abs() > float(config.assignment_change_epsilon)
    mask_st, hard, budgets = straight_through_changeable_topk(
        scores,
        eligible,
        config.assignment_mask_ratio,
        generator=generator,
        gumbel_scale=gumbel_scale,
        tau_ste=config.tau_ste,
    )
    return mask_st, hard, budgets, eligible


def fit_v21(
    X_model: np.ndarray,
    X_graph: Any | None,
    *,
    n_clusters: int | None,
    config: V21Config,
    seed: int,
    device: str | torch.device,
    stats_cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Fit V21 without receiving labels.

    ``n_clusters`` is required only by the two differentiable assignment-head
    variants. The runner records whether it was explicit or benchmark-known-K.
    """

    config.validate()
    if config.uses_cluster_head and (n_clusters is None or int(n_clusters) <= 1):
        raise ValueError("cluster-head variants require n_clusters > 1")
    if config.uses_topology_gate and X_graph is None:
        raise ValueError("topology_assignment_adversarial requires X_graph")
    runtime_device = torch.device(device)
    seed_all(seed, runtime_device)
    X_np = np.ascontiguousarray(np.asarray(X_model, dtype=np.float32))
    if X_np.ndim != 2 or X_np.shape[0] == 0 or X_np.shape[1] == 0:
        raise ValueError("X_model must be a non-empty 2D matrix")
    n_samples, n_features = X_np.shape
    if config.uses_cluster_head and n_samples < int(n_clusters):
        raise ValueError("n_samples must be at least n_clusters")

    graph_profile: dict[str, Any]
    stats_profile: dict[str, Any]
    stats: np.ndarray | None = None
    if config.uses_topology_gate:
        graph = build_svd_knn_graph(
            X_graph,
            neighbor_k=config.neighbor_k,
            svd_target=config.graph_svd_target,
            svd_min_dim=config.graph_svd_min_dim,
            svd_max_dim=config.graph_svd_max_dim,
            seed=seed,
        )
        stats, stats_profile = compute_topology_statistics(
            X_np,
            graph,
            block_size=config.stats_block_size,
            cache_dir=stats_cache_dir,
            cache_dtype=config.stats_cache_dtype,
            clip=config.stats_clip,
        )
        graph_profile = graph.profile | {"enabled": True}
    else:
        reason = config.variant
        graph_profile = {"enabled": False, "reason": reason}
        stats_profile = {"enabled": False, "reason": reason}

    model = V21AutoEncoder(
        num_genes=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(runtime_device)
    head = (
        StudentTClusterHead(
            int(n_clusters),
            config.hidden_size,
            config.cluster_alpha,
            config.cluster_distance_reduction,
        ).to(runtime_device)
        if config.uses_cluster_head
        else None
    )
    gate = FeatureGate(config.gate_hidden).to(runtime_device) if config.uses_topology_gate else None
    parameter_groups: list[dict[str, Any]] = [{"params": model.parameters(), "lr": float(config.lr)}]
    if head is not None:
        parameter_groups.append({"params": head.parameters(), "lr": float(config.cluster_lr)})
    optimizer = torch.optim.Adam(parameter_groups)
    gate_optimizer = torch.optim.Adam(gate.parameters(), lr=float(config.gate_lr)) if gate is not None else None

    batch_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 101)
    reconstruction_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 202)
    assignment_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 303)
    gate_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 404)
    history: list[dict[str, float]] = []
    cluster_initialised = False
    adversarial_step = 0
    gate_updates = 0
    gate_nonzero_updates = 0

    for epoch in range(config.epochs):
        if head is not None and not cluster_initialised and epoch >= config.warmup_epochs:
            warm_embedding = _clean_embedding(model, X_np, config.batch_size, runtime_device)
            head.initialise(warm_embedding, seed=seed, n_init=config.cluster_n_init)
            cluster_initialised = True

        model.train()
        if head is not None:
            head.train()
        if gate is not None:
            gate.train()
        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "mask_loss": 0.0,
            "random_requested_mask_rate": 0.0,
            "random_effective_mask_rate": 0.0,
            "assignment_divergence": 0.0,
            "infomax_loss": 0.0,
            "assignment_eligible_rate": 0.0,
            "assignment_selected_rate": 0.0,
            "assignment_effective_rate": 0.0,
            "assignment_effective_given_selected": 0.0,
            "assignment_budget_fill": 0.0,
        }
        gate_totals = {"gate_loss": 0.0, "gate_divergence": 0.0, "gate_coverage": 0.0}
        batches = 0
        epoch_gate_updates = 0
        for batch_ids in _batch_indices(n_samples, config.batch_size, batch_rng, runtime_device):
            row_ids = batch_ids.detach().cpu().numpy()
            batch = torch.as_tensor(X_np[row_ids], dtype=torch.float32, device=runtime_device)
            stat_batch = None
            if stats is not None:
                stat_batch = torch.as_tensor(
                    np.asarray(stats[row_ids], dtype=np.float32),
                    dtype=torch.float32,
                    device=runtime_device,
                )

            requested = _sample_random_reconstruction_mask(
                (batch.shape[0], n_features),
                mask_ratio=config.mask_ratio,
                mode=config.random_mask_mode,
                device=runtime_device,
                generator=reconstruction_rng,
            )
            reconstruction_donor = cyclic_donor(batch, generator=reconstruction_rng)
            changed = (reconstruction_donor != batch).to(dtype=batch.dtype)
            effective = requested * changed
            corrupted = batch + requested * (reconstruction_donor - batch)
            training_mask = effective if config.mask_target_mode == "effective" else requested

            _set_requires_grad(model, True)
            _set_requires_grad(head, True)
            _set_requires_grad(gate, False)
            optimizer.zero_grad(set_to_none=True)
            _, parts = model.loss_encoder(corrupted, batch, training_mask)
            total_loss = parts["loss"]
            assignment_divergence = total_loss.new_zeros(())
            infomax = total_loss.new_zeros(())
            eligible = torch.zeros_like(batch, dtype=torch.bool)
            assignment_hard = torch.zeros_like(batch)
            budgets = torch.zeros(batch.shape[0], dtype=torch.long, device=runtime_device)

            if head is not None and cluster_initialised:
                assignment_donor = cyclic_donor(batch, generator=assignment_rng)
                if gate is None:
                    scores = torch.rand(batch.shape, dtype=batch.dtype, device=runtime_device, generator=assignment_rng)
                    mask_st, assignment_hard, budgets, eligible = _assignment_mask_from_scores(
                        scores,
                        batch,
                        assignment_donor,
                        config=config,
                        generator=assignment_rng,
                        gumbel_scale=0.0,
                    )
                else:
                    if stat_batch is None:
                        raise RuntimeError("topology Gate requires topology statistics")
                    with torch.no_grad():
                        scores = gate(stat_batch)
                        mask_st, assignment_hard, budgets, eligible = _assignment_mask_from_scores(
                            scores,
                            batch,
                            assignment_donor,
                            config=config,
                            generator=assignment_rng,
                            gumbel_scale=config.gumbel_scale,
                        )
                assignment_corrupted = batch + assignment_hard * (assignment_donor - batch)
                q_clean = head(model.encode(batch))
                q_assignment = head(model.encode(assignment_corrupted))
                assignment_divergence = jensen_shannon_divergence(q_clean.detach(), q_assignment)
                infomax = information_maximization_loss(q_clean)
                total_loss = (
                    total_loss
                    + float(config.assignment_weight) * assignment_divergence
                    + float(config.infomax_weight) * infomax
                )

            total_loss.backward()
            optimizer.step()
            totals["loss"] += float(total_loss.detach().cpu())
            totals["reconstruction_loss"] += float(parts["reconstruction_loss"].cpu())
            totals["mask_loss"] += float(parts["mask_loss"].cpu())
            totals["random_requested_mask_rate"] += float(requested.mean().cpu())
            totals["random_effective_mask_rate"] += float(effective.mean().cpu())
            totals["assignment_divergence"] += float(assignment_divergence.detach().cpu())
            totals["infomax_loss"] += float(infomax.detach().cpu())
            if cluster_initialised:
                selected = assignment_hard.sum()
                effective_selected = (assignment_hard * eligible.to(batch.dtype)).sum()
                totals["assignment_eligible_rate"] += float(eligible.to(batch.dtype).mean().cpu())
                totals["assignment_selected_rate"] += float(assignment_hard.mean().cpu())
                totals["assignment_effective_rate"] += float((assignment_hard * eligible.to(batch.dtype)).mean().cpu())
                totals["assignment_effective_given_selected"] += float(
                    (effective_selected / selected.clamp_min(1.0)).detach().cpu()
                )
                totals["assignment_budget_fill"] += float(
                    (selected / budgets.sum().clamp_min(1).to(batch.dtype)).detach().cpu()
                )

            if gate is not None and cluster_initialised:
                adversarial_step += 1
                if adversarial_step % config.gate_update_every == 0:
                    if stat_batch is None or gate_optimizer is None or head is None:
                        raise RuntimeError("invalid topology Gate training state")
                    gate_updates += 1
                    epoch_gate_updates += 1
                    model.eval()
                    head.eval()
                    _set_requires_grad(model, False)
                    _set_requires_grad(head, False)
                    _set_requires_grad(gate, True)
                    gate_optimizer.zero_grad(set_to_none=True)
                    gate_donor = cyclic_donor(batch, generator=gate_rng)
                    gate_scores = gate(stat_batch)
                    gate_mask_st, _gate_hard, _gate_budgets, gate_eligible = _assignment_mask_from_scores(
                        gate_scores,
                        batch,
                        gate_donor,
                        config=config,
                        generator=gate_rng,
                        gumbel_scale=config.gumbel_scale,
                    )
                    with torch.no_grad():
                        q_reference = head(model.encode(batch))
                    gate_corrupted = batch + gate_mask_st * (gate_donor - batch)
                    q_gate = head(model.encode(gate_corrupted))
                    gate_divergence = jensen_shannon_divergence(q_reference, q_gate)
                    gate_coverage = coverage_concentration(gate_mask_st, gate_eligible)
                    gate_loss = -gate_divergence + float(config.gate_coverage_weight) * gate_coverage
                    gate_loss.backward()
                    gate_grad = _grad_norm(gate)
                    if gate_grad > 0.0 and np.isfinite(gate_grad):
                        gate_nonzero_updates += 1
                    gate_optimizer.step()
                    gate_totals["gate_loss"] += float(gate_loss.detach().cpu())
                    gate_totals["gate_divergence"] += float(gate_divergence.detach().cpu())
                    gate_totals["gate_coverage"] += float(gate_coverage.detach().cpu())
                    _set_requires_grad(model, True)
                    _set_requires_grad(head, True)
                    model.train()
                    head.train()
            batches += 1

        row = {key: value / max(1, batches) for key, value in totals.items()}
        row.update({key: value / max(1, epoch_gate_updates) for key, value in gate_totals.items()})
        row.update(
            {
                "epoch": float(epoch + 1),
                "cluster_head_active": float(cluster_initialised),
                "gate_updates_epoch": float(epoch_gate_updates),
            }
        )
        history.append(row)

    embedding = _clean_embedding(model, X_np, config.batch_size, runtime_device)
    probabilities = None
    if head is not None:
        if not cluster_initialised:
            raise RuntimeError("cluster head was never initialised")
        probabilities = _clean_probabilities(model, head, X_np, config.batch_size, runtime_device)
    diagnostics = {
        "graph_profile": graph_profile,
        "stats_profile": stats_profile,
        "history": history,
        "variant_contract": {
            "random_scmae_reconstruction": True,
            "cluster_head": head is not None,
            "assignment_consistency": head is not None,
            "topology_gate": gate is not None,
            "assignment_budget_scope": config.assignment_budget_scope if head is not None else None,
        },
        "mask_target_mode": config.mask_target_mode,
        "random_mask_mode": config.random_mask_mode,
        "cluster_head_initialised": bool(cluster_initialised),
        "gate_updates": int(gate_updates),
        "gate_nonzero_update_rate": float(gate_nonzero_updates / max(1, gate_updates)),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "cluster_head_parameter_count": 0 if head is None else int(sum(p.numel() for p in head.parameters())),
        "gate_parameter_count": 0 if gate is None else int(sum(p.numel() for p in gate.parameters())),
        "labels_used_during_fit": False,
        "K_used_during_fit": bool(head is not None),
        "model": model,
        "cluster_head": head,
        "gate": gate,
    }
    return embedding, probabilities, diagnostics


def fit_scmae_only(
    X_model: np.ndarray,
    *,
    config: V21Config,
    seed: int,
    device: str | torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.variant != "scmae_only":
        raise ValueError("fit_scmae_only requires variant='scmae_only'")
    embedding, probabilities, diagnostics = fit_v21(
        X_model,
        None,
        n_clusters=None,
        config=config,
        seed=seed,
        device=device,
    )
    if probabilities is not None:
        raise RuntimeError("scmae_only unexpectedly produced cluster probabilities")
    return embedding, diagnostics
