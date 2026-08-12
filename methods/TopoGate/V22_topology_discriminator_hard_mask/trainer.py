from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from .config import V22Config
from .graph import build_svd_knn_graph, compute_topology_statistics
from .model import (
    CoordinateDiscriminator,
    CoordinateGate,
    V22AutoEncoder,
    coverage_concentration,
    cyclic_donor,
    random_topk_mask,
    straight_through_topk,
)


ALLOWED_PHYSICAL_GPUS = frozenset({1, 2, 3, 4, 5, 6})


def resolve_device(device: str | torch.device, gpu: int | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cpu":
        if gpu is not None:
            raise ValueError("--gpu cannot be used with --device cpu")
        return torch.device("cpu")
    if gpu is None:
        raise ValueError("CUDA V22 runs require an explicit physical --gpu in 1..6")
    if int(gpu) not in ALLOWED_PHYSICAL_GPUS:
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
            raise ValueError("V22 requires an indexed logical CUDA device")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(int(seed))


def _set_requires_grad(module: torch.nn.Module | None, enabled: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _grad_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    values = [p.grad.detach().norm().item() for p in module.parameters() if p.grad is not None]
    return float(np.sqrt(np.sum(np.square(values)))) if values else 0.0


def _loss_grad_norm(loss: torch.Tensor, module: torch.nn.Module | None) -> float:
    """Measure a component's gradient without consuming the shared graph."""
    if module is None or not loss.requires_grad:
        return 0.0
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [gradient.detach().norm().item() for gradient in gradients if gradient is not None]
    return float(np.sqrt(np.sum(np.square(values)))) if values else 0.0


def _batch_indices(n_samples: int, batch_size: int, generator: torch.Generator, device: torch.device) -> list[torch.Tensor]:
    order = torch.randperm(n_samples, generator=generator, device=device)
    return [order[start : min(n_samples, start + batch_size)] for start in range(0, n_samples, batch_size)]


def _clean_embedding(model: V22AutoEncoder, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    was_training = model.training
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model.encode(batch).cpu().numpy())
    model.train(was_training)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _pair_coordinates(
    mask: torch.Tensor,
    *,
    per_row: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Sample coordinate ids while preserving matched row/feature pairs."""
    row_ids: list[torch.Tensor] = []
    feature_ids: list[torch.Tensor] = []
    for row in range(mask.shape[0]):
        candidates = torch.nonzero(mask[row] > 0.0, as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        if candidates.numel() > int(per_row):
            order = torch.randperm(candidates.numel(), device=mask.device, generator=generator)[: int(per_row)]
            candidates = candidates[order]
        row_ids.append(torch.full((candidates.numel(),), row, dtype=torch.long, device=mask.device))
        feature_ids.append(candidates.to(dtype=torch.long))
    if not row_ids:
        return None
    return torch.cat(row_ids), torch.cat(feature_ids)


def _topology_input(
    stats: np.ndarray | None,
    row_ids: np.ndarray,
    n_features: int,
    device: torch.device,
) -> torch.Tensor:
    if stats is None:
        zeros = torch.zeros((row_ids.shape[0], n_features, 4), dtype=torch.float32, device=device)
        zeros[:, :, 3] = 1.0
        return zeros
    return torch.as_tensor(np.asarray(stats[row_ids], dtype=np.float32), dtype=torch.float32, device=device)


def _pair_inputs(
    latent: torch.Tensor,
    candidate: torch.Tensor,
    topology: torch.Tensor,
    row_ids: torch.Tensor,
    feature_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context = latent[row_ids]
    values = candidate[row_ids, feature_ids]
    topology_pairs = topology[row_ids, feature_ids]
    return context, feature_ids, topology_pairs, values


def _safe_mean(value: torch.Tensor, fallback: float = 0.0) -> float:
    if value.numel() == 0:
        return float(fallback)
    result = float(value.detach().mean().cpu())
    return result if np.isfinite(result) else float(fallback)


def _mask_profile(counts: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(counts, dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return {
            "selected_total": 0,
            "unique_feature_count": 0,
            "unique_feature_fraction": 0.0,
            "top10_mass": 0.0,
            "coverage_entropy": 0.0,
        }
    probabilities = values / total
    positive = probabilities[probabilities > 0.0]
    top10 = np.sort(probabilities)[-min(10, probabilities.size) :]
    return {
        "selected_total": int(total),
        "unique_feature_count": int(np.count_nonzero(values > 0.0)),
        "unique_feature_fraction": float(np.mean(values > 0.0)),
        "top10_mass": float(top10.sum()),
        "coverage_entropy": float(-(positive * np.log(positive)).sum()),
    }


def _reuse_topology_statistics(
    cache_dir: str | Path,
    *,
    n_samples: int,
    n_features: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Reuse a complete statistics memmap from the same explicit run directory.

    The byte-size check is deliberately strict: a partially written or differently
    shaped cache must fail rather than silently changing the topology signal.
    Callers expose reuse as an explicit recovery option and record that fact.
    """
    path = Path(cache_dir) / "topology_statistics.dat"
    expected_bytes = int(n_samples) * int(n_features) * 4 * np.dtype("float32").itemsize
    if not path.is_file():
        raise FileNotFoundError(f"topology cache is missing: {path}")
    actual_bytes = int(path.stat().st_size)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"topology cache size mismatch for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
    stats = np.memmap(
        path,
        mode="r",
        dtype=np.float32,
        shape=(int(n_samples), int(n_features), 4),
    )
    stats_profile = {
        "storage": "memmap",
        "path": str(path.resolve()),
        "dtype": "float32",
        "shape": [int(n_samples), int(n_features), 4],
        "cache_reused": True,
        "cache_validation": "exact_byte_shape",
        "support_is_label_free": True,
    }
    graph_profile = {
        "enabled": True,
        "cache_reused": True,
        "cache_path": str(path.resolve()),
        "graph_profile_available": False,
    }
    return stats, stats_profile, graph_profile


def fit_v22(
    X_model: np.ndarray,
    X_graph: Any | None,
    X_support: Any | None,
    *,
    config: V22Config,
    seed: int,
    device: str | torch.device,
    stats_cache_dir: str | Path | None = None,
    reuse_topology_cache: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit V22 without receiving labels.

    The discriminator is active from the first epoch for every discriminator
    variant.  Gate updates use the detached/frozen discriminator and model
    parameters; only the Gate receives gradients in that phase.
    """
    config.validate()
    runtime_device = torch.device(device)
    seed_all(seed, runtime_device)
    X_np = np.ascontiguousarray(np.asarray(X_model, dtype=np.float32))
    if X_np.ndim != 2 or X_np.shape[0] == 0 or X_np.shape[1] == 0:
        raise ValueError("X_model must be a non-empty 2D matrix")
    n_samples, n_features = X_np.shape

    graph_profile: dict[str, Any]
    stats_profile: dict[str, Any]
    stats: np.ndarray | None = None
    if config.uses_topology_gate:
        if X_graph is None:
            raise ValueError("topology V22 requires X_graph")
        if reuse_topology_cache:
            if stats_cache_dir is None:
                raise ValueError("reuse_topology_cache requires stats_cache_dir")
            stats, stats_profile, graph_profile = _reuse_topology_statistics(
                stats_cache_dir,
                n_samples=n_samples,
                n_features=n_features,
            )
        else:
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
                support_matrix=X_support,
                block_size=config.stats_block_size,
                cache_dir=stats_cache_dir,
                clip=config.stats_clip,
            )
            graph_profile = graph.profile | {"enabled": True}
    else:
        graph_profile = {"enabled": False, "reason": config.variant}
        stats_profile = {"enabled": False, "reason": config.variant}

    model = V22AutoEncoder(
        num_genes=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(runtime_device)
    discriminator = (
        CoordinateDiscriminator(
            n_features=n_features,
            context_dim=config.hidden_size,
            hidden_size=config.discriminator_hidden,
            coordinate_embedding_dim=config.coordinate_embedding_dim,
            topology_dim=config.discriminator_topology_dim,
        ).to(runtime_device)
        if config.uses_discriminator
        else None
    )
    gate = (
        CoordinateGate(
            n_features=n_features,
            hidden_size=config.gate_hidden,
            coordinate_embedding_dim=config.coordinate_embedding_dim,
        ).to(runtime_device)
        if config.uses_gate
        else None
    )
    model_optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    discriminator_optimizer = (
        torch.optim.Adam(discriminator.parameters(), lr=float(config.discriminator_lr))
        if discriminator is not None
        else None
    )
    gate_optimizer = torch.optim.Adam(gate.parameters(), lr=float(config.gate_lr)) if gate is not None else None

    batch_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 101)
    reconstruction_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 202)
    adversarial_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 303)
    gate_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 404)
    bce = torch.nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    discriminator_steps = 0
    discriminator_nonzero_steps = 0
    gate_updates = 0
    gate_nonzero_updates = 0
    adversarial_feature_counts = np.zeros(n_features, dtype=np.float64)
    adversarial_effective_feature_counts = np.zeros(n_features, dtype=np.float64)
    random_feature_counts = np.zeros(n_features, dtype=np.float64)
    random_effective_feature_counts = np.zeros(n_features, dtype=np.float64)
    gate_feature_counts = np.zeros(n_features, dtype=np.float64)
    gate_effective_feature_counts = np.zeros(n_features, dtype=np.float64)
    gate_keep_feature_counts = np.zeros(n_features, dtype=np.float64)
    gate_keep_effective_feature_counts = np.zeros(n_features, dtype=np.float64)

    for epoch in range(config.epochs):
        model.train()
        if discriminator is not None:
            discriminator.train()
        if gate is not None:
            gate.train()
        totals: dict[str, float] = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "mask_loss": 0.0,
            "discriminator_loss": 0.0,
            "generator_adversarial_loss": 0.0,
            "gate_reconstruction_loss": 0.0,
            "gate_loss": 0.0,
            "gate_reward": 0.0,
            "gate_coverage": 0.0,
            "random_requested_mask_rate": 0.0,
            "random_effective_mask_rate": 0.0,
            "adversarial_selected_rate": 0.0,
            "adversarial_effective_rate": 0.0,
            "discriminator_pair_count": 0.0,
            "discriminator_real_accuracy": 0.0,
            "discriminator_gate_fake_accuracy": 0.0,
            "discriminator_scmae_fake_accuracy": 0.0,
            "discriminator_confusion_pair_count": 0.0,
            "discriminator_real_abs_value_mean": 0.0,
            "discriminator_fake_abs_value_mean": 0.0,
            "discriminator_real_nonzero_rate": 0.0,
            "discriminator_fake_nonzero_rate": 0.0,
            "discriminator_value_low_accuracy": 0.0,
            "discriminator_value_mid_accuracy": 0.0,
            "discriminator_value_high_accuracy": 0.0,
            "discriminator_value_matched_accuracy": 0.0,
            "gate_grad_norm": 0.0,
            "gate_grad_reconstruction_norm": 0.0,
            "gate_grad_discriminator_norm": 0.0,
        }
        batches = 0
        epoch_discriminator_steps = 0
        epoch_gate_updates = 0
        for batch_ids in _batch_indices(n_samples, config.batch_size, batch_rng, runtime_device):
            row_ids_np = batch_ids.detach().cpu().numpy()
            batch = torch.as_tensor(X_np[row_ids_np], dtype=torch.float32, device=runtime_device)
            topology = _topology_input(stats, row_ids_np, n_features, runtime_device)

            random_requested = random_topk_mask(
                (batch.shape[0], n_features),
                config.random_mask_ratio,
                device=runtime_device,
                generator=reconstruction_rng,
            )
            random_donor = cyclic_donor(batch, generator=reconstruction_rng)
            random_changed = (random_donor - batch).abs() > float(config.assignment_change_epsilon)
            random_effective = random_requested * random_changed.to(dtype=batch.dtype)
            random_corrupted = batch + random_requested * (random_donor - batch)
            random_feature_counts += random_requested.detach().sum(dim=0).cpu().numpy()
            random_effective_feature_counts += random_effective.detach().sum(dim=0).cpu().numpy()

            _set_requires_grad(model, True)
            _set_requires_grad(discriminator, False)
            _set_requires_grad(gate, False)
            model_optimizer.zero_grad(set_to_none=True)
            _, base_parts = model.loss_encoder(random_corrupted, batch, random_effective)
            total_loss = base_parts["loss"]
            discriminator_loss = total_loss.new_zeros(())
            generator_adversarial_loss = total_loss.new_zeros(())
            gate_reconstruction_loss = total_loss.new_zeros(())
            adversarial_hard = torch.zeros_like(batch)
            adversarial_effective = torch.zeros_like(batch)

            if discriminator is not None:
                adversarial_donor = cyclic_donor(batch, generator=adversarial_rng)
                adversarial_changed = (adversarial_donor - batch).abs() > float(config.assignment_change_epsilon)
                if config.variant == "scmae_plus_discriminator_random_mask":
                    adversarial_requested = random_topk_mask(
                        (batch.shape[0], n_features),
                        config.adversarial_mask_ratio,
                        device=runtime_device,
                        generator=adversarial_rng,
                    )
                    adversarial_hard = adversarial_requested
                else:
                    if gate is None:
                        raise RuntimeError("gate variant has no Gate module")
                    with torch.no_grad():
                        gate_scores = gate(topology)
                        if config.gate_reward_mode == "cooperative_keep":
                            keep_ratio = 1.0 - float(config.adversarial_mask_ratio)
                            _st, keep_hard, _budgets = straight_through_topk(
                                gate_scores,
                                adversarial_changed,
                                keep_ratio,
                                generator=adversarial_rng,
                                gumbel_scale=0.0,
                            )
                            # The Gate selects the coordinates to keep visible;
                            # only the complementary changed coordinates are reconstructed.
                            adversarial_hard = adversarial_changed.to(dtype=batch.dtype) - keep_hard
                        else:
                            _st, adversarial_hard, _budgets = straight_through_topk(
                                gate_scores,
                                adversarial_changed,
                                config.adversarial_mask_ratio,
                                generator=adversarial_rng,
                                gumbel_scale=0.0,
                            )
                adversarial_effective = adversarial_hard * adversarial_changed.to(dtype=batch.dtype)
                adversarial_feature_counts += adversarial_hard.detach().sum(dim=0).cpu().numpy()
                adversarial_effective_feature_counts += adversarial_effective.detach().sum(dim=0).cpu().numpy()
                adversarial_corrupted = batch + adversarial_hard * (adversarial_donor - batch)
                adversarial_latent, _mask_logits, adversarial_reconstruction = model.forward_mask(adversarial_corrupted)
                pairs = _pair_coordinates(
                    adversarial_effective,
                    per_row=config.discriminator_coordinates_per_row,
                    generator=adversarial_rng,
                )
                if pairs is not None:
                    pair_rows, pair_features = pairs
                    context, feature_ids, topology_pairs, fake_values = _pair_inputs(
                        adversarial_latent,
                        adversarial_reconstruction,
                        topology,
                        pair_rows,
                        pair_features,
                    )
                    real_values = batch[pair_rows, pair_features]
                    _set_requires_grad(discriminator, True)
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    real_logits = discriminator(context.detach(), feature_ids, topology_pairs.detach(), real_values.detach())
                    fake_logits = discriminator(context.detach(), feature_ids, topology_pairs.detach(), fake_values.detach())
                    discriminator_loss = 0.5 * (bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits)))
                    discriminator_loss.backward()
                    discriminator_optimizer.step()
                    discriminator_steps += 1
                    epoch_discriminator_steps += 1
                    if np.isfinite(float(discriminator_loss.detach().cpu())):
                        discriminator_nonzero_steps += 1
                    _set_requires_grad(discriminator, False)

                    # The discriminator is frozen in this phase, but its output
                    # remains connected to the generator's latent/reconstruction.
                    fake_logits_for_generator = discriminator(context, feature_ids, topology_pairs, fake_values)
                    generator_adversarial_loss = bce(
                        fake_logits_for_generator,
                        torch.ones_like(fake_logits_for_generator),
                    )
                    gate_reconstruction_loss = (
                        (adversarial_reconstruction - batch).square() * adversarial_effective
                    ).sum() / adversarial_effective.sum().clamp_min(1.0)
                    total_loss = (
                        total_loss
                        + float(config.lambda_adversarial) * generator_adversarial_loss
                        + float(config.lambda_gate_reconstruction) * gate_reconstruction_loss
                    )

            total_loss.backward()
            model_optimizer.step()

            # Diagnostic only: compare the trained D on Gate-style fake
            # coordinates with the same-coordinate random scMAE fake.  This
            # does not feed back into any optimizer and exposes a corruption
            # style shortcut before formal experiments are attempted.
            d_real_accuracy = total_loss.new_zeros(())
            d_gate_fake_accuracy = total_loss.new_zeros(())
            d_scmae_fake_accuracy = total_loss.new_zeros(())
            d_confusion_pairs = 0
            d_real_abs_value_mean = total_loss.new_zeros(())
            d_fake_abs_value_mean = total_loss.new_zeros(())
            d_real_nonzero_rate = total_loss.new_zeros(())
            d_fake_nonzero_rate = total_loss.new_zeros(())
            d_value_low_accuracy = total_loss.new_zeros(())
            d_value_mid_accuracy = total_loss.new_zeros(())
            d_value_high_accuracy = total_loss.new_zeros(())
            d_value_matched_accuracy = total_loss.new_zeros(())
            if discriminator is not None and pairs is not None:
                with torch.no_grad():
                    d_real_accuracy = (real_logits.detach() > 0.0).to(dtype=total_loss.dtype).mean()
                    d_gate_fake_accuracy = (fake_logits.detach() < 0.0).to(dtype=total_loss.dtype).mean()
                    real_abs = real_values.detach().abs()
                    fake_abs = fake_values.detach().abs()
                    d_real_abs_value_mean = real_abs.mean()
                    d_fake_abs_value_mean = fake_abs.mean()
                    d_real_nonzero_rate = (real_abs > float(config.assignment_change_epsilon)).to(dtype=total_loss.dtype).mean()
                    d_fake_nonzero_rate = (fake_abs > float(config.assignment_change_epsilon)).to(dtype=total_loss.dtype).mean()
                    value_thresholds = torch.quantile(real_abs, torch.tensor([0.1, 0.9], device=real_abs.device))
                    paired_labels = torch.cat(
                        [torch.ones_like(real_logits.detach()), torch.zeros_like(fake_logits.detach())]
                    )
                    paired_predictions = torch.cat([real_logits.detach(), fake_logits.detach()]) > 0.0
                    paired_magnitudes = torch.cat([real_abs, real_abs])
                    low = paired_magnitudes <= value_thresholds[0]
                    high = paired_magnitudes > value_thresholds[1]
                    mid = ~(low | high)
                    if bool(low.any()):
                        d_value_low_accuracy = (paired_predictions[low] == paired_labels[low].bool()).to(
                            dtype=total_loss.dtype
                        ).mean()
                    if bool(mid.any()):
                        d_value_mid_accuracy = (paired_predictions[mid] == paired_labels[mid].bool()).to(
                            dtype=total_loss.dtype
                        ).mean()
                    if bool(high.any()):
                        d_value_high_accuracy = (paired_predictions[high] == paired_labels[high].bool()).to(
                            dtype=total_loss.dtype
                        ).mean()
                    common_abs = 0.5 * (real_abs + fake_abs)
                    real_value_matched = torch.sign(real_values.detach()) * common_abs
                    fake_value_matched = torch.sign(fake_values.detach()) * common_abs
                    matched_real_logits = discriminator(
                        context.detach(), feature_ids, topology_pairs.detach(), real_value_matched
                    )
                    matched_fake_logits = discriminator(
                        context.detach(), feature_ids, topology_pairs.detach(), fake_value_matched
                    )
                    matched_logits = torch.cat([matched_real_logits, matched_fake_logits])
                    matched_labels = torch.cat(
                        [torch.ones_like(matched_real_logits), torch.zeros_like(matched_fake_logits)]
                    )
                    d_value_matched_accuracy = (
                        ((matched_logits > 0.0) == matched_labels.bool()).to(dtype=total_loss.dtype).mean()
                    )
                    random_latent_eval, _random_logits_eval, random_reconstruction_eval = model.forward_mask(random_corrupted)
                    random_pairs = _pair_coordinates(
                        random_effective,
                        per_row=config.discriminator_coordinates_per_row,
                        generator=reconstruction_rng,
                    )
                    if random_pairs is not None:
                        random_rows, random_features = random_pairs
                        random_context, random_ids, random_topology, random_fake_values = _pair_inputs(
                            random_latent_eval,
                            random_reconstruction_eval,
                            topology,
                            random_rows,
                            random_features,
                        )
                        random_real_values = batch[random_rows, random_features]
                        random_real_logits = discriminator(
                            random_context,
                            random_ids,
                            random_topology,
                            random_real_values,
                        )
                        random_fake_logits = discriminator(
                            random_context,
                            random_ids,
                            random_topology,
                            random_fake_values,
                        )
                        d_scmae_fake_accuracy = (random_fake_logits < 0.0).to(dtype=total_loss.dtype).mean()
                        d_confusion_pairs = int(random_fake_logits.numel())

            # Gate is a separate frozen-model optimisation problem.  The ST
            # mask carries the score gradient without allowing Gate to update D.
            gate_loss = total_loss.new_zeros(())
            gate_reward = total_loss.new_zeros(())
            gate_coverage = total_loss.new_zeros(())
            gate_grad_reconstruction_norm = 0.0
            gate_grad_discriminator_norm = 0.0
            if gate is not None and discriminator is not None:
                if batches % config.gate_update_every == 0:
                    _set_requires_grad(model, False)
                    _set_requires_grad(discriminator, False)
                    _set_requires_grad(gate, True)
                    gate_optimizer.zero_grad(set_to_none=True)
                    gate_scores = gate(topology)
                    gate_donor = cyclic_donor(batch, generator=gate_rng)
                    gate_eligible = (gate_donor - batch).abs() > float(config.assignment_change_epsilon)
                    if config.gate_reward_mode == "cooperative_keep":
                        keep_ratio = 1.0 - float(config.adversarial_mask_ratio)
                        keep_mask_st, keep_hard, _gate_budgets = straight_through_topk(
                            gate_scores,
                            gate_eligible,
                            keep_ratio,
                            generator=gate_rng,
                            gumbel_scale=1.0,
                        )
                        gate_mask_st = gate_eligible.to(dtype=batch.dtype) - keep_mask_st
                        gate_hard = gate_eligible.to(dtype=batch.dtype) - keep_hard
                        gate_keep_feature_counts += keep_hard.detach().sum(dim=0).cpu().numpy()
                        gate_keep_effective_feature_counts += (
                            (keep_hard * gate_eligible.to(dtype=batch.dtype)).detach().sum(dim=0).cpu().numpy()
                        )
                        coverage_mask = keep_mask_st
                    else:
                        gate_mask_st, gate_hard, _gate_budgets = straight_through_topk(
                            gate_scores,
                            gate_eligible,
                            config.adversarial_mask_ratio,
                            generator=gate_rng,
                            gumbel_scale=1.0,
                        )
                        coverage_mask = gate_mask_st
                    gate_effective = gate_hard * gate_eligible.to(dtype=batch.dtype)
                    gate_feature_counts += gate_hard.detach().sum(dim=0).cpu().numpy()
                    gate_effective_feature_counts += gate_effective.detach().sum(dim=0).cpu().numpy()
                    gate_corrupted = batch + gate_mask_st * (gate_donor - batch)
                    gate_latent, _gate_logits, gate_reconstruction = model.forward_mask(gate_corrupted)
                    gate_pairs = _pair_coordinates(
                        gate_effective,
                        per_row=config.discriminator_coordinates_per_row,
                        generator=gate_rng,
                    )
                    if gate_pairs is not None:
                        pair_rows, pair_features = gate_pairs
                        context, feature_ids, topology_pairs, fake_values = _pair_inputs(
                            gate_latent,
                            gate_reconstruction,
                            topology,
                            pair_rows,
                            pair_features,
                        )
                        real_values = batch[pair_rows, pair_features]
                        real_logits = discriminator(context, feature_ids, topology_pairs, real_values)
                        fake_logits = discriminator(context, feature_ids, topology_pairs, fake_values)
                        if config.gate_reward_mode == "cooperative_keep":
                            cooperative_reconstruction = (
                                (real_values - fake_values).square()
                            ).mean()
                            cooperative_adversarial = bce(fake_logits, torch.ones_like(fake_logits))
                            cooperative_discriminator_term = (
                                float(config.lambda_adversarial) * cooperative_adversarial
                            )
                            gate_grad_reconstruction_norm = _loss_grad_norm(cooperative_reconstruction, gate)
                            gate_grad_discriminator_norm = _loss_grad_norm(cooperative_discriminator_term, gate)
                            gate_reward = -(
                                cooperative_reconstruction
                                + cooperative_discriminator_term
                            )
                        else:
                            stability = topology_pairs[:, 3].clamp(0.0, 1.0)
                            if config.gate_reward_mode == "reconstruction_error_control":
                                difficulty = (real_values - fake_values).square().clamp(
                                    0.0, float(config.gate_reward_clip)
                                )
                            else:
                                difficulty = (real_logits - fake_logits).clamp(0.0, float(config.gate_reward_clip))
                            gate_reward = (difficulty * stability).mean()
                        gate_coverage = coverage_concentration(coverage_mask, gate_eligible)
                        gate_loss = -gate_reward + float(config.lambda_gate_coverage) * gate_coverage
                        gate_loss.backward()
                        totals["gate_grad_norm"] += _grad_norm(gate)
                        if _grad_norm(gate) > 0.0:
                            gate_nonzero_updates += 1
                        gate_optimizer.step()
                        gate_updates += 1
                        epoch_gate_updates += 1
                    _set_requires_grad(gate, False)
                    _set_requires_grad(model, True)
                    if discriminator is not None:
                        _set_requires_grad(discriminator, True)

            totals["loss"] += float(total_loss.detach().cpu())
            totals["reconstruction_loss"] += float(base_parts["reconstruction_loss"].cpu())
            totals["mask_loss"] += float(base_parts["mask_loss"].cpu())
            totals["discriminator_loss"] += float(discriminator_loss.detach().cpu())
            totals["generator_adversarial_loss"] += float(generator_adversarial_loss.detach().cpu())
            totals["gate_reconstruction_loss"] += float(gate_reconstruction_loss.detach().cpu())
            totals["gate_loss"] += float(gate_loss.detach().cpu())
            totals["gate_reward"] += float(gate_reward.detach().cpu())
            totals["gate_coverage"] += float(gate_coverage.detach().cpu())
            totals["random_requested_mask_rate"] += _safe_mean(random_requested)
            totals["random_effective_mask_rate"] += _safe_mean(random_effective)
            totals["adversarial_selected_rate"] += _safe_mean(adversarial_hard)
            totals["adversarial_effective_rate"] += _safe_mean(adversarial_effective)
            totals["discriminator_pair_count"] += float(0 if pairs is None else pair_features.numel()) if discriminator is not None else 0.0
            totals["discriminator_real_accuracy"] += float(d_real_accuracy.detach().cpu())
            totals["discriminator_gate_fake_accuracy"] += float(d_gate_fake_accuracy.detach().cpu())
            totals["discriminator_scmae_fake_accuracy"] += float(d_scmae_fake_accuracy.detach().cpu())
            totals["discriminator_confusion_pair_count"] += float(d_confusion_pairs)
            totals["discriminator_real_abs_value_mean"] += float(d_real_abs_value_mean.detach().cpu())
            totals["discriminator_fake_abs_value_mean"] += float(d_fake_abs_value_mean.detach().cpu())
            totals["discriminator_real_nonzero_rate"] += float(d_real_nonzero_rate.detach().cpu())
            totals["discriminator_fake_nonzero_rate"] += float(d_fake_nonzero_rate.detach().cpu())
            totals["discriminator_value_low_accuracy"] += float(d_value_low_accuracy.detach().cpu())
            totals["discriminator_value_mid_accuracy"] += float(d_value_mid_accuracy.detach().cpu())
            totals["discriminator_value_high_accuracy"] += float(d_value_high_accuracy.detach().cpu())
            totals["discriminator_value_matched_accuracy"] += float(d_value_matched_accuracy.detach().cpu())
            totals["gate_grad_reconstruction_norm"] += float(gate_grad_reconstruction_norm)
            totals["gate_grad_discriminator_norm"] += float(gate_grad_discriminator_norm)
            batches += 1

        row = {key: value / max(1, batches) for key, value in totals.items()}
        row.update(
            {
                "epoch": float(epoch + 1),
                "discriminator_steps_epoch": float(epoch_discriminator_steps),
                "gate_updates_epoch": float(epoch_gate_updates),
            }
        )
        history.append(row)

    embedding = _clean_embedding(model, X_np, config.batch_size, runtime_device)
    diagnostics = {
        "graph_profile": graph_profile,
        "stats_profile": stats_profile,
        "history": history,
        "variant_contract": {
            "scmae_backbone": True,
            "discriminator": discriminator is not None,
            "coordinate_matched_real_fake": discriminator is not None,
            "mask_or_hint_passed_to_discriminator": False,
            "gate": gate is not None,
            "topology_gate": config.uses_topology_gate,
            "gate_reward_mode": config.gate_reward_mode if gate is not None else None,
            "gate_semantics": (
                "cooperative_keep_complementary_mask"
                if config.gate_reward_mode == "cooperative_keep"
                else "adversarial_hard_mask"
            ),
            "clean_embedding_kmeans_primary": True,
        },
        "labels_used_during_fit": False,
        "K_used_during_fit": False,
        "discriminator_steps": int(discriminator_steps),
        "discriminator_finite_step_rate": float(discriminator_nonzero_steps / max(1, discriminator_steps)),
        "gate_updates": int(gate_updates),
        "gate_nonzero_update_rate": float(gate_nonzero_updates / max(1, gate_updates)),
        "adversarial_mask_profile": _mask_profile(adversarial_feature_counts),
        "adversarial_effective_mask_profile": _mask_profile(adversarial_effective_feature_counts),
        "random_mask_profile": _mask_profile(random_feature_counts),
        "random_effective_mask_profile": _mask_profile(random_effective_feature_counts),
        "gate_mask_profile": _mask_profile(gate_feature_counts),
        "gate_effective_mask_profile": _mask_profile(gate_effective_feature_counts),
        "gate_keep_profile": _mask_profile(gate_keep_feature_counts),
        "gate_keep_effective_profile": _mask_profile(gate_keep_effective_feature_counts),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "discriminator_parameter_count": 0 if discriminator is None else int(sum(p.numel() for p in discriminator.parameters())),
        "gate_parameter_count": 0 if gate is None else int(sum(p.numel() for p in gate.parameters())),
        "model": model,
        "discriminator": discriminator,
        "gate": gate,
    }
    return embedding, diagnostics


def fit_scmae_only(
    X_model: np.ndarray,
    *,
    config: V22Config,
    seed: int,
    device: str | torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.variant != "scmae_only":
        raise ValueError("fit_scmae_only requires variant='scmae_only'")
    return fit_v22(X_model, None, None, config=config, seed=seed, device=device)
