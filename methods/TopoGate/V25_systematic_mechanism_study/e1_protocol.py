"""V25 E1: V21 matched N/R/T selection-policy protocol.

This module is deliberately separate from the historical V21 runner.  It
reuses the audited V21 model and topology primitives but introduces the
prospective matching contract: one branchpoint, one schedule, one selection
noise tensor, and three arms (None, matched Random, Topology).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from methods.TopoGate.V21_assignment_adversarial_gate.graph import (
    build_svd_knn_graph,
    compute_topology_statistics,
)
from methods.TopoGate.V21_assignment_adversarial_gate.model import (
    FeatureGate,
    StudentTClusterHead,
    V21AutoEncoder,
    coverage_concentration,
    information_maximization_loss,
    jensen_shannon_divergence,
)


class E1Arm(str, Enum):
    NONE = "N"
    RANDOM = "R"
    TOPOLOGY = "T"


@dataclass(frozen=True)
class E1Config:
    protocol_id: str = "v25_e1_v21_matched_nrt_v1"
    epochs: int = 80
    warmup_epochs: int = 40
    batch_size: int = 256
    hidden_size: int = 128
    dropout: float = 0.0
    lr: float = 1e-3
    cluster_lr: float = 1e-3
    gate_lr: float = 2.5e-4
    mask_ratio: float = 0.4
    masked_data_weight: float = 0.75
    mask_loss_weight: float = 0.7
    mask_target_mode: str = "effective"
    assignment_mask_ratio: float = 0.4
    assignment_change_epsilon: float = 0.0
    assignment_weight: float = 0.1
    infomax_weight: float = 0.05
    cluster_alpha: float = 1.0
    cluster_distance_reduction: str = "sum"
    cluster_n_init: int = 20
    gate_hidden: int = 64
    gate_coverage_weight: float = 0.01
    gumbel_scale: float = 1.0
    tau_ste: float = 0.5
    graph_svd_target: float = 0.95
    graph_svd_min_dim: int = 50
    graph_svd_max_dim: int = 500
    neighbor_k: int = 20
    stats_block_size: int = 1024
    stats_cache_dtype: str = "float32"
    stats_clip: float = 5.0
    kmeans_n_init: int = 20
    schedule_seed_offset: int = 9101
    # `None` preserves PyTorch's default for ordinary inputs.  The V21
    # decoder is quadratic in feature count; for very high-dimensional
    # holdouts, disabling Adam's foreach workspace is a resource-only
    # implementation choice that leaves the Adam update, loss, and schedule
    # unchanged.
    adam_foreach: bool | None = None
    adam_foreach_disable_feature_threshold: int = 10000

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.hidden_size <= 0:
            raise ValueError("epochs, batch_size, and hidden_size must be positive")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs)")
        if not 0 < self.mask_ratio < 1 or not 0 < self.assignment_mask_ratio <= 1:
            raise ValueError("mask ratios are outside their valid ranges")
        if self.mask_target_mode not in {"requested", "effective"}:
            raise ValueError("mask_target_mode must be requested or effective")
        if self.cluster_distance_reduction not in {"mean", "sum"}:
            raise ValueError("cluster_distance_reduction must be mean or sum")
        if min(self.lr, self.cluster_lr, self.gate_lr) <= 0:
            raise ValueError("learning rates must be positive")
        if self.assignment_weight < 0 or self.infomax_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.gumbel_scale < 0 or self.tau_ste <= 0:
            raise ValueError("invalid selection noise parameters")
        if self.kmeans_n_init <= 0 or self.cluster_n_init <= 0:
            raise ValueError("k-means initialization counts must be positive")
        if self.adam_foreach is not None and not isinstance(self.adam_foreach, bool):
            raise ValueError("adam_foreach must be true, false, or null")
        if self.adam_foreach_disable_feature_threshold <= 0:
            raise ValueError("adam_foreach_disable_feature_threshold must be positive")


@dataclass(frozen=True)
class ScheduleEntry:
    epoch: int
    step: int
    batch_ids: tuple[int, ...]
    reconstruction_seed: int
    reconstruction_offset: int
    assignment_offset: int
    selection_noise_seed: int


@dataclass
class ScheduleBundle:
    warmup: list[ScheduleEntry]
    post_branch: list[ScheduleEntry]
    hashes: dict[str, str]
    batch_rng_state: torch.Tensor


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.random.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def _hash_array_update(digest: Any, array: np.ndarray | torch.Tensor) -> None:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    digest.update(np.ascontiguousarray(array).tobytes())


def _capture_rng(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _batch_entries(n_samples: int, batch_size: int, epochs: int, seed: int) -> tuple[list[ScheduleEntry], torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    entries: list[ScheduleEntry] = []
    plan_rng = np.random.default_rng(int(seed) + 17)
    for epoch in range(int(epochs)):
        permutation = torch.randperm(n_samples, generator=generator)
        for step, start in enumerate(range(0, n_samples, int(batch_size))):
            batch = permutation[start : min(n_samples, start + int(batch_size))]
            batch_size_actual = int(batch.numel())
            entries.append(
                ScheduleEntry(
                    epoch=epoch,
                    step=step,
                    batch_ids=tuple(int(v) for v in batch.tolist()),
                    reconstruction_seed=int(plan_rng.integers(0, 2**31 - 1)),
                    reconstruction_offset=0 if batch_size_actual <= 1 else int(plan_rng.integers(1, batch_size_actual)),
                    assignment_offset=0 if batch_size_actual <= 1 else int(plan_rng.integers(1, batch_size_actual)),
                    selection_noise_seed=int(plan_rng.integers(0, 2**31 - 1)),
                )
            )
    return entries, generator.get_state()


def _schedule_hashes(entries: Iterable[ScheduleEntry]) -> dict[str, str]:
    batch = hashlib.sha256()
    donor = hashlib.sha256()
    noise = hashlib.sha256()
    for entry in entries:
        _hash_array_update(batch, np.asarray(entry.batch_ids, dtype=np.int64))
        donor.update(f"{entry.reconstruction_offset}:{entry.assignment_offset}".encode())
        noise.update(f"{entry.selection_noise_seed}".encode())
    return {
        "batch_permutation_hash": batch.hexdigest(),
        "donor_schedule_hash": donor.hexdigest(),
        "selection_noise_seed_hash": noise.hexdigest(),
    }


def _make_schedule(n_samples: int, config: E1Config, seed: int) -> ScheduleBundle:
    entries, batch_state = _batch_entries(
        n_samples,
        config.batch_size,
        config.epochs,
        int(seed) + config.schedule_seed_offset,
    )
    warmup_count = sum(1 for entry in entries if entry.epoch < config.warmup_epochs)
    warmup = entries[:warmup_count]
    post_branch = entries[warmup_count:]
    all_hashes = _schedule_hashes(entries)
    return ScheduleBundle(warmup=warmup, post_branch=post_branch, hashes=all_hashes, batch_rng_state=batch_state)


def _random_generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def _materialize_schedule(
    X: torch.Tensor,
    entry: ScheduleEntry,
    config: E1Config,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = _batch_on_device(X, entry.batch_ids, device)
    if batch.shape[0] <= 1:
        reconstruction_donor = batch.clone()
        assignment_donor = batch.clone()
    else:
        reconstruction_donor = torch.roll(batch, shifts=entry.reconstruction_offset, dims=0)
        assignment_donor = torch.roll(batch, shifts=entry.assignment_offset, dims=0)
    mask_rng = _random_generator(entry.reconstruction_seed, device)
    requested = (torch.rand(batch.shape, generator=mask_rng, device=device) < float(config.mask_ratio)).to(batch.dtype)
    changed = (reconstruction_donor != batch).to(batch.dtype)
    effective = requested * changed
    corrupted = batch + requested * (reconstruction_donor - batch)
    eligible = (assignment_donor - batch).abs() > float(config.assignment_change_epsilon)
    noise_rng = _random_generator(entry.selection_noise_seed, device)
    uniform = torch.rand(batch.shape, generator=noise_rng, device=device, dtype=batch.dtype).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    return {
        "batch": batch,
        "reconstruction_donor": reconstruction_donor,
        "assignment_donor": assignment_donor,
        "requested": requested,
        "effective": effective,
        "corrupted": corrupted,
        "training_mask": effective if config.mask_target_mode == "effective" else requested,
        "eligible": eligible,
        "gumbel": gumbel,
    }


def _selection_from_logits(
    logits: torch.Tensor,
    eligible: torch.Tensor,
    gumbel: torch.Tensor,
    config: E1Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.shape != eligible.shape or logits.shape != gumbel.shape:
        raise ValueError("selection tensors must have the same shape")
    eligible = eligible.to(torch.bool)
    counts = eligible.sum(dim=1)
    budgets = torch.ceil(counts.to(logits.dtype) * float(config.assignment_mask_ratio)).to(torch.long)
    budgets = torch.minimum(budgets, counts)
    max_budget = int(budgets.max().item()) if budgets.numel() else 0
    if max_budget == 0:
        zero = logits * 0.0
        return zero, torch.zeros_like(logits), budgets
    noisy = logits + float(config.gumbel_scale) * gumbel
    masked = noisy.masked_fill(~eligible, -torch.inf)
    top_values, top_indices = torch.topk(masked, k=max_budget, dim=1, largest=True, sorted=True)
    ranks = torch.arange(max_budget, device=logits.device)[None, :] < budgets[:, None]
    hard = torch.zeros_like(logits).scatter(1, top_indices, ranks.to(logits.dtype))
    threshold = top_values.gather(1, (budgets - 1).clamp_min(0)[:, None])
    valid = budgets.gt(0)[:, None]
    soft = torch.sigmoid((noisy - threshold.detach()) / float(config.tau_ste))
    soft = soft * eligible.to(logits.dtype) * valid.to(logits.dtype)
    return hard + soft - soft.detach(), hard, budgets


def _flatten_grads(grads: Iterable[torch.Tensor | None], params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for grad, parameter in zip(grads, params):
        values.append(torch.zeros_like(parameter).reshape(-1) if grad is None else grad.detach().reshape(-1))
    return torch.cat(values) if values else torch.zeros(1)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = a.norm() * b.norm()
    if float(denominator) <= 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denominator)


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _batch_on_device(X: torch.Tensor, batch_ids: Iterable[int], device: torch.device) -> torch.Tensor:
    """Fetch a batch from host-backed data and move only that batch to the model."""
    ids = torch.as_tensor(tuple(int(value) for value in batch_ids), dtype=torch.long, device=X.device)
    batch = X.index_select(0, ids)
    if batch.device != device:
        batch = batch.to(device)
    return batch


def _stats_batch_on_device(
    stats: np.ndarray | torch.Tensor,
    batch_ids: Iterable[int],
    device: torch.device,
) -> torch.Tensor:
    """Fetch topology statistics without materializing the full tensor on GPU."""
    ids_tuple = tuple(int(value) for value in batch_ids)
    if isinstance(stats, torch.Tensor):
        ids = torch.as_tensor(ids_tuple, dtype=torch.long, device=stats.device)
        batch = stats.index_select(0, ids)
        return batch if batch.device == device else batch.to(device)
    batch = np.asarray(stats)[np.asarray(ids_tuple, dtype=np.int64)]
    return torch.as_tensor(batch, dtype=torch.float32, device=device)


def _hash_array(value: np.ndarray | torch.Tensor, *, block_rows: int = 256) -> str:
    """Hash array bytes in bounded blocks, preserving the contiguous byte order."""
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    digest = hashlib.sha256()
    if array.ndim == 0:
        _hash_array_update(digest, array)
    else:
        for start in range(0, array.shape[0], int(block_rows)):
            _hash_array_update(digest, array[start : start + int(block_rows)])
    return digest.hexdigest()


def _clean_embedding(model: V21AutoEncoder, X: torch.Tensor, batch_size: int) -> np.ndarray:
    was_training = model.training
    model.eval()
    values: list[np.ndarray] = []
    device = _model_device(model)
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = X[start : start + batch_size]
            if batch.device != device:
                batch = batch.to(device)
            values.append(model.encode(batch).detach().cpu().numpy())
    model.train(was_training)
    return np.concatenate(values, axis=0).astype(np.float32, copy=False)


def _readout(embedding: np.ndarray, n_clusters: int, seed: int, n_init: int) -> np.ndarray:
    if int(n_init) <= 0:
        raise ValueError("k-means n_init must be positive")
    return KMeans(n_clusters=int(n_clusters), n_init=int(n_init), random_state=int(seed)).fit_predict(embedding).astype(np.int64)


def _metrics(
    embedding: np.ndarray,
    n_clusters: int,
    seed: int,
    labels: np.ndarray | None,
    *,
    kmeans_n_init: int,
) -> dict[str, Any]:
    predictions = _readout(embedding, n_clusters, seed, kmeans_n_init)
    output: dict[str, Any] = {
        "readout": "clean_embedding_known_k_kmeans",
        "labels_used_for_fit": False,
        "labels_used_for_readout": False,
        "n_clusters": int(n_clusters),
        "unique_predicted_clusters": int(np.unique(predictions).size),
    }
    if labels is not None:
        encoded = np.asarray(labels).astype(str)
        output.update(
            {
                "ari": float(adjusted_rand_score(encoded, predictions)),
                "nmi": float(normalized_mutual_info_score(encoded, predictions)),
            }
        )
    return {"predictions": predictions, "metrics": output}


def _resolve_adam_options(n_features: int, config: E1Config, device: torch.device) -> dict[str, bool]:
    if config.adam_foreach is not None:
        return {"foreach": bool(config.adam_foreach), "fused": False}
    if device.type == "cuda" and int(n_features) >= config.adam_foreach_disable_feature_threshold:
        # Fused Adam evaluates the denominator in its CUDA kernel rather than
        # allocating a feature-sized sqrt workspace.  The optimizer remains
        # Adam; this is only a memory-bounded implementation path.
        return {"foreach": False, "fused": True}
    return {}


def _resolve_adam_foreach(n_features: int, config: E1Config, device: torch.device) -> bool | None:
    """Compatibility helper used by tests and audit tooling."""
    return _resolve_adam_options(n_features, config, device).get("foreach")


def _resolve_adam_fused(n_features: int, config: E1Config, device: torch.device) -> bool:
    return bool(_resolve_adam_options(n_features, config, device).get("fused", False))


def _build_components(X: torch.Tensor, n_clusters: int, config: E1Config, seed: int, device: torch.device) -> dict[str, Any]:
    model = V21AutoEncoder(
        num_genes=X.shape[1],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(device)
    head = StudentTClusterHead(
        n_clusters,
        config.hidden_size,
        config.cluster_alpha,
        config.cluster_distance_reduction,
    ).to(device)
    gate = FeatureGate(config.gate_hidden).to(device)
    # The threshold is deliberately tied to the frozen input adapter, not to
    # a result or dataset outcome.  It only avoids the extra CUDA foreach
    # workspace for the quadratic V21 decoder at extreme width.
    optimizer_options = _resolve_adam_options(X.shape[1], config, device)
    optimizer_foreach = optimizer_options.get("foreach")
    optimizer_fused = optimizer_options.get("fused", False)
    optimizer_kwargs: dict[str, Any] = dict(optimizer_options)
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": float(config.lr)},
            {"params": head.parameters(), "lr": float(config.cluster_lr)},
        ],
        **optimizer_kwargs,
    )
    gate_optimizer = torch.optim.Adam(gate.parameters(), lr=float(config.gate_lr), **optimizer_kwargs)
    return {
        "model": model,
        "head": head,
        "gate": gate,
        "optimizer": optimizer,
        "gate_optimizer": gate_optimizer,
        "optimizer_foreach": optimizer_foreach,
        "optimizer_fused": optimizer_fused,
    }


def _loss_for_arm(
    arm: E1Arm,
    components: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    stats_batch: torch.Tensor,
    config: E1Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    model: V21AutoEncoder = components["model"]
    head: StudentTClusterHead = components["head"]
    gate: FeatureGate = components["gate"]
    _, parts = model.loss_encoder(tensors["corrupted"], tensors["batch"], tensors["training_mask"])
    base = parts["loss"]
    q_clean = head(model.encode(tensors["batch"]))
    infomax = information_maximization_loss(q_clean)
    js = base.new_zeros(())
    assignment_mask = torch.zeros_like(tensors["batch"])
    hard = torch.zeros_like(tensors["batch"])
    budgets = torch.zeros(tensors["batch"].shape[0], dtype=torch.long, device=tensors["batch"].device)
    if arm is not E1Arm.NONE:
        if arm is E1Arm.RANDOM:
            logits = torch.zeros_like(tensors["batch"])
        else:
            logits = gate(stats_batch)
        assignment_mask, hard, budgets = _selection_from_logits(logits, tensors["eligible"], tensors["gumbel"], config)
        assignment_corrupted = tensors["batch"] + hard * (tensors["assignment_donor"] - tensors["batch"])
        q_assignment = head(model.encode(assignment_corrupted))
        js = jensen_shannon_divergence(q_clean.detach(), q_assignment)
    total = base + float(config.infomax_weight) * infomax + float(config.assignment_weight) * js
    return total, {
        "base": base,
        "infomax": infomax,
        "js": js,
        "q_clean": q_clean,
        "assignment_mask": assignment_mask,
        "hard": hard,
        "budgets": budgets,
    }, {"assignment_forward": arm is not E1Arm.NONE, "js_forward": arm is not E1Arm.NONE}


def _initialise_head(components: dict[str, Any], X: torch.Tensor, config: E1Config, seed: int) -> None:
    embedding = _clean_embedding(components["model"], X, config.batch_size)
    components["head"].initialise(embedding, seed=int(seed), n_init=config.cluster_n_init)


def _run_warmup(
    components: dict[str, Any], X: torch.Tensor, entries: list[ScheduleEntry], config: E1Config, device: torch.device
) -> None:
    model: V21AutoEncoder = components["model"]
    optimizer: torch.optim.Optimizer = components["optimizer"]
    model.train()
    for entry in entries:
        tensors = _materialize_schedule(X, entry, config, device)
        optimizer.zero_grad(set_to_none=True)
        _, parts = model.loss_encoder(tensors["corrupted"], tensors["batch"], tensors["training_mask"])
        parts["loss"].backward()
        optimizer.step()


def _snapshot_to_cpu(value: Any) -> Any:
    """Copy a state tree to host memory without a second GPU-sized clone."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _release_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _state_for_save(components: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": _snapshot_to_cpu(components["model"].state_dict()),
        "head": _snapshot_to_cpu(components["head"].state_dict()),
        "gate": _snapshot_to_cpu(components["gate"].state_dict()),
        "optimizer": _snapshot_to_cpu(components["optimizer"].state_dict()),
        "gate_optimizer": _snapshot_to_cpu(components["gate_optimizer"].state_dict()),
    }


def _load_state(components: dict[str, Any], state: dict[str, Any]) -> None:
    components["model"].load_state_dict(state["model"])
    components["head"].load_state_dict(state["head"])
    components["gate"].load_state_dict(state["gate"])
    components["optimizer"].load_state_dict(state["optimizer"])
    components["gate_optimizer"].load_state_dict(state["gate_optimizer"])


def _gradient_probe(
    losses: dict[str, torch.Tensor], components: dict[str, Any]
) -> dict[str, float]:
    params = list(components["model"].parameters()) + list(components["head"].parameters())
    base_grads = torch.autograd.grad(losses["base"], params, retain_graph=True, allow_unused=True)
    assignment_grads = torch.autograd.grad(losses["js"], params, retain_graph=True, allow_unused=True)
    infomax_grads = torch.autograd.grad(losses["infomax"], params, retain_graph=True, allow_unused=True)
    g_base = _flatten_grads(base_grads, params)
    g_assignment = _flatten_grads(assignment_grads, params)
    g_infomax = _flatten_grads(infomax_grads, params)
    g_sum = g_assignment + g_infomax
    return {
        "cos_base_assignment": _cosine(g_base, g_assignment),
        "cos_base_infomax": _cosine(g_base, g_infomax),
        "cos_base_assignment_plus_infomax": _cosine(g_base, g_sum),
        "norm_base": float(g_base.norm()),
        "norm_assignment": float(g_assignment.norm()),
        "norm_infomax": float(g_infomax.norm()),
        "norm_assignment_plus_infomax": float(g_sum.norm()),
    }


def _gate_update(
    components: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    stats_batch: torch.Tensor,
    config: E1Config,
) -> float:
    model: V21AutoEncoder = components["model"]
    head: StudentTClusterHead = components["head"]
    gate: FeatureGate = components["gate"]
    optimizer: torch.optim.Optimizer = components["gate_optimizer"]
    model.eval()
    head.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    for parameter in gate.parameters():
        parameter.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    scores = gate(stats_batch)
    mask_st, _hard, _budgets = _selection_from_logits(scores, tensors["eligible"], tensors["gumbel"], config)
    with torch.no_grad():
        q_reference = head(model.encode(tensors["batch"]))
    corrupted = tensors["batch"] + mask_st * (tensors["assignment_donor"] - tensors["batch"])
    q_gate = head(model.encode(corrupted))
    divergence = jensen_shannon_divergence(q_reference, q_gate)
    coverage = coverage_concentration(mask_st, tensors["eligible"])
    loss = -divergence + float(config.gate_coverage_weight) * coverage
    loss.backward()
    optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    model.train()
    head.train()
    return float(loss.detach().cpu())


def _arm_train(
    arm: E1Arm,
    state: dict[str, Any],
    X: torch.Tensor,
    stats: torch.Tensor,
    schedule: ScheduleBundle,
    n_clusters: int,
    config: E1Config,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    components = _build_components(X, n_clusters, config, seed, device)
    _load_state(components, state)
    model: V21AutoEncoder = components["model"]
    head: StudentTClusterHead = components["head"]
    gate: FeatureGate = components["gate"]
    optimizer: torch.optim.Optimizer = components["optimizer"]
    history: list[dict[str, Any]] = []
    assignment_forward_calls = 0
    js_forward_calls = 0
    shadow_assignment_calls = 0
    gradient_probe: dict[str, dict[str, float]] = {}
    selected_feature_counts = torch.zeros(X.shape[1], dtype=torch.float64, device=device)
    eligible_not_selected_feature_counts = torch.zeros(X.shape[1], dtype=torch.float64, device=device)
    coordinate_metric_sums = {
        "topology_deviation": {"selected": 0.0, "eligible_not_selected": 0.0},
        "topology_dispersion": {"selected": 0.0, "eligible_not_selected": 0.0},
        "donor_change_magnitude": {"selected": 0.0, "eligible_not_selected": 0.0},
    }
    coordinate_counts = {"total": 0, "selected": 0, "eligible": 0, "eligible_not_selected": 0}
    selection_hashes = {
        "eligible": hashlib.sha256(),
        "budget": hashlib.sha256(),
        "noise": hashlib.sha256(),
        "donor": hashlib.sha256(),
    }
    for step_index, entry in enumerate(schedule.post_branch):
        tensors = _materialize_schedule(X, entry, config, device)
        stats_batch = _stats_batch_on_device(stats, entry.batch_ids, device)
        _hash_array_update(selection_hashes["eligible"], tensors["eligible"])
        _hash_array_update(selection_hashes["donor"], tensors["reconstruction_donor"])
        _hash_array_update(selection_hashes["donor"], tensors["assignment_donor"])
        shadow_assignment_calls += 1
        model.train()
        head.train()
        gate.train()
        optimizer.zero_grad(set_to_none=True)
        total, losses, counters = _loss_for_arm(arm, components, tensors, stats_batch, config)
        assignment_forward_calls += int(counters["assignment_forward"])
        js_forward_calls += int(counters["js_forward"])
        if arm is E1Arm.NONE:
            # N replays donor/eligibility/noise only for audit.  It never builds
            # an assignment corruption and never evaluates JS.
            eligible_count = tensors["eligible"].sum(dim=1)
            budgets = torch.ceil(eligible_count.to(tensors["batch"].dtype) * float(config.assignment_mask_ratio)).to(torch.long)
            budgets = torch.minimum(budgets, eligible_count)
            _hash_array_update(selection_hashes["budget"], budgets)
            _hash_array_update(selection_hashes["noise"], tensors["gumbel"])
        else:
            _hash_array_update(selection_hashes["budget"], losses["budgets"])
            _hash_array_update(selection_hashes["noise"], tensors["gumbel"])
        if arm is E1Arm.TOPOLOGY:
            hard_mask = losses["hard"].detach().to(torch.bool)
            other_mask = tensors["eligible"].detach().to(torch.bool) & ~hard_mask
            selected_feature_counts += hard_mask.to(torch.float64).sum(dim=0)
            eligible_not_selected_feature_counts += other_mask.to(torch.float64).sum(dim=0)
            coordinate_counts["total"] += int(hard_mask.numel())
            coordinate_counts["eligible"] += int(tensors["eligible"].sum().detach().cpu())
            coordinate_counts["selected"] += int(hard_mask.sum().detach().cpu())
            coordinate_counts["eligible_not_selected"] += int(other_mask.sum().detach().cpu())
            metric_values = {
                "topology_deviation": stats_batch[:, :, 0],
                "topology_dispersion": stats_batch[:, :, 1],
                "donor_change_magnitude": (tensors["assignment_donor"] - tensors["batch"]).abs(),
            }
            for name, values in metric_values.items():
                coordinate_metric_sums[name]["selected"] += float(values[hard_mask].detach().sum().cpu())
                coordinate_metric_sums[name]["eligible_not_selected"] += float(values[other_mask].detach().sum().cpu())
        if arm is E1Arm.TOPOLOGY and len(gradient_probe) < 3:
            gradient_probe[f"T{len(gradient_probe)}"] = _gradient_probe(losses, components)
        total.backward()
        optimizer.step()
        if arm is E1Arm.TOPOLOGY:
            _gate_update(components, tensors, stats_batch, config)
        history.append(
            {
                "step": int(step_index),
                "epoch": int(entry.epoch),
                "loss": float(total.detach().cpu()),
                "base_loss": float(losses["base"].detach().cpu()),
                "infomax_loss": float(losses["infomax"].detach().cpu()),
                "assignment_js": float(losses["js"].detach().cpu()),
                "eligible_rate": float(tensors["eligible"].to(torch.float32).mean().cpu()),
                "selected_rate": float(losses["hard"].mean().cpu()),
                "effective_budget": int(losses["budgets"].sum().detach().cpu()),
            }
        )

    embedding = _clean_embedding(model, X, config.batch_size)
    readout = _metrics(embedding, n_clusters, seed, None, kmeans_n_init=config.kmeans_n_init)
    feature_audit = None
    if arm is E1Arm.TOPOLOGY:
        X_np = X.detach().cpu().numpy().astype(np.float64)
        feature_audit = {
            "statistical_unit": "dataset_seed_summary",
            "selection_snapshot": "training_time_topology_gate_policy",
            "coordinate_distribution_is_descriptive_only": True,
            "coordinate_count": coordinate_counts["total"],
            "eligible_coordinate_count": coordinate_counts["eligible"],
            "selected_coordinate_count": coordinate_counts["selected"],
            "eligible_not_selected_coordinate_count": coordinate_counts["eligible_not_selected"],
            "coordinate_metric_sums": coordinate_metric_sums,
            "selected_feature_counts": selected_feature_counts.detach().cpu().numpy(),
            "eligible_not_selected_feature_counts": eligible_not_selected_feature_counts.detach().cpu().numpy(),
            "feature_metrics": {
                "model_variance": X_np.var(axis=0),
                "model_zero_fraction": (X_np == 0).mean(axis=0),
                "model_support_frequency": (X_np != 0).mean(axis=0),
            },
        }
    return {
        "arm": arm.value,
        "status": "completed",
        "history": history,
        "embedding": embedding,
        "predictions": readout["predictions"],
        "metrics": readout["metrics"],
        "gradient_probe": gradient_probe,
        "feature_audit": feature_audit,
        "audit": {
            "optimizer": "Adam",
            "optimizer_foreach": components.get("optimizer_foreach"),
            "optimizer_fused": components.get("optimizer_fused", False),
            "assignment_forward_calls": assignment_forward_calls,
            "js_forward_calls": js_forward_calls,
            "shadow_assignment_calls": shadow_assignment_calls,
            "none_assignment_forward_forbidden": arm is E1Arm.NONE,
            "none_js_forward_forbidden": arm is E1Arm.NONE,
            "eligible_schedule_hash": selection_hashes["eligible"].hexdigest(),
            "budget_schedule_hash": selection_hashes["budget"].hexdigest(),
            "selection_noise_hash": selection_hashes["noise"].hexdigest(),
            "donor_schedule_hash": selection_hashes["donor"].hexdigest(),
        },
        "checkpoint": {
            "model": _snapshot_to_cpu(model.state_dict()),
            "head": _snapshot_to_cpu(head.state_dict()),
            "gate": _snapshot_to_cpu(gate.state_dict()),
            "optimizer": _snapshot_to_cpu(optimizer.state_dict()),
            "gate_optimizer": _snapshot_to_cpu(components["gate_optimizer"].state_dict()),
        },
    }


def _one_step(
    arm: E1Arm,
    state: dict[str, Any],
    X: torch.Tensor,
    stats: torch.Tensor,
    entry: ScheduleEntry,
    n_clusters: int,
    config: E1Config,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    components = _build_components(X, n_clusters, config, seed, device)
    _load_state(components, state)
    tensors = _materialize_schedule(X, entry, config, device)
    stats_batch = _stats_batch_on_device(stats, entry.batch_ids, device)
    components["optimizer"].zero_grad(set_to_none=True)
    total, losses, _ = _loss_for_arm(arm, components, tensors, stats_batch, config)
    total.backward()
    components["optimizer"].step()
    embedding = _clean_embedding(components["model"], X, config.batch_size)
    readout = _metrics(embedding, n_clusters, seed, None, kmeans_n_init=config.kmeans_n_init)
    return {"arm": arm.value, "metrics": readout["metrics"], "predictions": readout["predictions"], "loss": float(total.detach().cpu())}


def _pair_delta(left: dict[str, Any], right: dict[str, Any], key: str = "ari") -> float | None:
    a = left.get("metrics", {}).get(key)
    b = right.get("metrics", {}).get(key)
    return None if a is None or b is None else float(a) - float(b)


def run_e1(
    X_model: np.ndarray,
    X_graph: Any,
    *,
    n_clusters: int,
    config: E1Config,
    seed: int,
    device: str | torch.device = "cpu",
    evaluation_labels: np.ndarray | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one dataset/seed N/R/T E1 panel with no labels in fit."""
    config.validate()
    if n_clusters <= 1:
        raise ValueError("n_clusters must be greater than one")
    runtime_device = torch.device(device)
    _seed_all(seed, runtime_device)
    X_np = np.ascontiguousarray(np.asarray(X_model, dtype=np.float32))
    if X_np.ndim != 2 or X_np.shape[0] < n_clusters or not np.isfinite(X_np).all():
        raise ValueError("X_model must be finite 2D data with at least n_clusters rows")
    # Keep the complete high-dimensional matrix on host memory when the model
    # runs on CUDA.  Only scheduled batches move to the device; this preserves
    # the V21 computation while avoiding a full dense X/statistics GPU copy.
    data_device = torch.device("cpu") if runtime_device.type == "cuda" else runtime_device
    X = torch.as_tensor(X_np, dtype=torch.float32, device=data_device)
    graph = build_svd_knn_graph(
        X_graph,
        neighbor_k=config.neighbor_k,
        svd_target=config.graph_svd_target,
        svd_min_dim=min(config.graph_svd_min_dim, max(1, X_np.shape[0] - 1)),
        svd_max_dim=min(config.graph_svd_max_dim, max(1, X_np.shape[0] - 1)),
        seed=seed,
    )
    stats_np, stats_profile = compute_topology_statistics(
        X_np,
        graph,
        block_size=config.stats_block_size,
        cache_dir=(Path(output_dir) / "cache") if output_dir is not None else None,
        cache_dtype=config.stats_cache_dtype,
        clip=config.stats_clip,
    )
    # `stats_np` may be a memmap.  Keep it host-backed for CUDA runs and fetch
    # only the current batch in `_stats_batch_on_device`.
    stats: np.ndarray | torch.Tensor = stats_np
    schedule = _make_schedule(X_np.shape[0], config, seed)
    components = _build_components(X, n_clusters, config, seed, runtime_device)
    _run_warmup(components, X, schedule.warmup, config, runtime_device)
    _initialise_head(components, X, config, seed)
    branch_state = _state_for_save(components)
    branch_rng = _capture_rng(runtime_device)
    branchpoint = {
        "epoch": config.warmup_epochs,
        "head_initialised": bool(components["head"].initialised),
        "schedule_hashes": schedule.hashes,
        "batch_permutation_state": schedule.batch_rng_state,
        "rng": branch_rng,
        "model_state": branch_state,
        "topology_statistics_hash": _hash_array(stats_np),
        "graph_profile": graph.profile,
        "stats_profile": stats_profile,
    }
    optimizer_foreach = components.get("optimizer_foreach")
    optimizer_fused = components.get("optimizer_fused", False)
    # The serialized branchpoint is host-backed.  Drop the warmup model and
    # optimizer before constructing the first arm so high-dimensional Adam
    # state is never duplicated on the CUDA device.
    del components
    _release_cuda_cache(runtime_device)
    arms: dict[str, Any] = {}
    for arm in (E1Arm.NONE, E1Arm.RANDOM, E1Arm.TOPOLOGY):
        _restore_rng(branch_rng, runtime_device)
        arms[arm.value] = _arm_train(arm, branch_state, X, stats, schedule, n_clusters, config, seed, runtime_device)
        _release_cuda_cache(runtime_device)
    one_step: dict[str, Any] = {}
    for arm in (E1Arm.NONE, E1Arm.RANDOM, E1Arm.TOPOLOGY):
        _restore_rng(branch_rng, runtime_device)
        one_step[arm.value] = _one_step(arm, branch_state, X, stats, schedule.post_branch[0], n_clusters, config, seed, runtime_device)
        _release_cuda_cache(runtime_device)

    # Benchmark labels enter only after every arm and one-step counterfactual
    # has finished fitting.  They never reach the model, graph, Gate, loss, or
    # K-means fitting paths above.  Inject the readout-only metrics before
    # constructing the paired estimands so I/S are complete when labels exist.
    if evaluation_labels is not None:
        encoded = np.asarray(evaluation_labels).astype(str)
        if encoded.shape[0] != X_np.shape[0]:
            raise ValueError("evaluation_labels must have one entry per sample")
        for section in (arms, one_step):
            for item in section.values():
                item["metrics"].update(
                    {
                        "ari": float(adjusted_rand_score(encoded, item["predictions"])),
                        "nmi": float(normalized_mutual_info_score(encoded, item["predictions"])),
                        "labels_used_after_fit_only": True,
                    }
                )
    pairs = {
        "I_full_ARI": _pair_delta(arms[E1Arm.RANDOM.value], arms[E1Arm.NONE.value]),
        "S_full_ARI": _pair_delta(arms[E1Arm.TOPOLOGY.value], arms[E1Arm.RANDOM.value]),
        "I_1step_ARI": _pair_delta(one_step[E1Arm.RANDOM.value], one_step[E1Arm.NONE.value]),
        "S_1step_ARI": _pair_delta(one_step[E1Arm.TOPOLOGY.value], one_step[E1Arm.RANDOM.value]),
    }
    audit = {
        "protocol_id": config.protocol_id,
        "labels_used_during_fit": False,
        "K_used_during_fit": True,
        "K_source": "caller_outer_evaluation",
        "data_device": str(data_device),
        "model_device": str(runtime_device),
        "host_backed_streaming": bool(runtime_device.type == "cuda"),
        "topology_statistics_storage": stats_profile.get("storage"),
        "optimizer": "Adam",
        "optimizer_foreach": optimizer_foreach,
        "optimizer_fused": optimizer_fused,
        "optimizer_foreach_threshold": int(config.adam_foreach_disable_feature_threshold),
        "arm_names": [arm.value for arm in (E1Arm.NONE, E1Arm.RANDOM, E1Arm.TOPOLOGY)],
        "donor_schedule_hash": arms[E1Arm.RANDOM.value]["audit"]["donor_schedule_hash"],
        "eligible_schedule_hash": arms[E1Arm.RANDOM.value]["audit"]["eligible_schedule_hash"],
        "budget_schedule_hash": arms[E1Arm.RANDOM.value]["audit"]["budget_schedule_hash"],
        "selection_noise_hash": arms[E1Arm.RANDOM.value]["audit"]["selection_noise_hash"],
        "topology_statistics_hash": branchpoint["topology_statistics_hash"],
        "TR_shared_schedule_hashes": {
            "donor": arms[E1Arm.TOPOLOGY.value]["audit"]["donor_schedule_hash"] == arms[E1Arm.RANDOM.value]["audit"]["donor_schedule_hash"],
            "eligible": arms[E1Arm.TOPOLOGY.value]["audit"]["eligible_schedule_hash"] == arms[E1Arm.RANDOM.value]["audit"]["eligible_schedule_hash"],
            "budget": arms[E1Arm.TOPOLOGY.value]["audit"]["budget_schedule_hash"] == arms[E1Arm.RANDOM.value]["audit"]["budget_schedule_hash"],
            "selection_noise": arms[E1Arm.TOPOLOGY.value]["audit"]["selection_noise_hash"] == arms[E1Arm.RANDOM.value]["audit"]["selection_noise_hash"],
        },
        "none_contract": {
            "assignment_forward_calls": arms[E1Arm.NONE.value]["audit"]["assignment_forward_calls"],
            "js_forward_calls": arms[E1Arm.NONE.value]["audit"]["js_forward_calls"],
            "shadow_assignment_calls": arms[E1Arm.NONE.value]["audit"]["shadow_assignment_calls"],
        },
        "branchpoint": {
            "epoch": branchpoint["epoch"],
            "head_initialised": branchpoint["head_initialised"],
            "warmup_branchpoint_before_first_assignment": True,
        },
    }
    result = {
        "status": "completed",
        "protocol_id": config.protocol_id,
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "labels_used_during_fit": False,
        "branchpoint": branchpoint,
        "schedule": {"hashes": schedule.hashes, "warmup_entries": len(schedule.warmup), "post_branch_entries": len(schedule.post_branch)},
        "pairs": pairs,
        "audit": audit,
        "arms": arms,
        "one_step": one_step,
    }
    if output_dir is not None:
        _write_result(Path(output_dir), result, config)
    return result


def _write_result(out: Path, result: dict[str, Any], config: E1Config) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    branchpoint = result["branchpoint"]
    branchpoint_for_torch = dict(branchpoint)
    torch.save(branchpoint_for_torch, out / "branchpoint.pt")
    branchpoint_meta = {key: value for key, value in branchpoint.items() if key not in {"model_state", "rng", "batch_permutation_state"}}
    (out / "branchpoint_metadata.json").write_text(json.dumps(branchpoint_meta, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (out / "schedule_manifest.json").write_text(json.dumps(result["schedule"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "pairs" ).mkdir(exist_ok=True)
    (out / "pairs" / "N_R.json").write_text(json.dumps({"I_full_ARI": result["pairs"]["I_full_ARI"], "I_1step_ARI": result["pairs"]["I_1step_ARI"]}, indent=2) + "\n", encoding="utf-8")
    (out / "pairs" / "T_R.json").write_text(json.dumps({"S_full_ARI": result["pairs"]["S_full_ARI"], "S_1step_ARI": result["pairs"]["S_1step_ARI"]}, indent=2) + "\n", encoding="utf-8")
    for arm_name, arm in result["arms"].items():
        arm_out = out / arm_name
        arm_out.mkdir(exist_ok=True)
        np.save(arm_out / "embedding_final.npy", arm["embedding"])
        np.save(arm_out / "predictions.npy", arm["predictions"])
        (arm_out / "metrics.json").write_text(json.dumps(arm["metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (arm_out / "history.json").write_text(json.dumps(arm["history"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (arm_out / "audit.json").write_text(json.dumps(arm["audit"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (arm_out / "gradient_probe.json").write_text(json.dumps(arm["gradient_probe"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if arm.get("feature_audit") is not None:
            audit = arm["feature_audit"]
            np.savez_compressed(
                arm_out / "feature_selection_counts.npz",
                selected_feature_counts=np.asarray(audit["selected_feature_counts"], dtype=np.float64),
                eligible_not_selected_feature_counts=np.asarray(audit["eligible_not_selected_feature_counts"], dtype=np.float64),
                **{name: np.asarray(value, dtype=np.float64) for name, value in audit["feature_metrics"].items()},
            )
            feature_json = {
                key: value
                for key, value in audit.items()
                if key not in {"selected_feature_counts", "eligible_not_selected_feature_counts", "feature_metrics"}
            }
            feature_json["feature_metric_names"] = sorted(audit["feature_metrics"])
            (arm_out / "feature_audit_label_free.json").write_text(json.dumps(feature_json, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        torch.save(arm["checkpoint"], arm_out / "checkpoint.pt")
    (out / "one_step.json").write_text(json.dumps({key: {"arm": value["arm"], "metrics": value["metrics"], "loss": value["loss"]} for key, value in result["one_step"].items()}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(result["audit"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key not in {"branchpoint", "arms", "one_step"}}
    summary["status"] = "completed"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
