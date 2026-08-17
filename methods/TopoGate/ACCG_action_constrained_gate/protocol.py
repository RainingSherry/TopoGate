"""Matched N/R/T_s/T_c protocol for ACCG.

The shared branchpoint and schedule are inherited from the audited V25 E1
implementation. ACCG adds only a label-free feature-conditional action
constraint to the T_c selection policy.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from methods.TopoGate.V25_systematic_mechanism_study import e1_protocol as e1

from .calibration import EpsilonCalibration, calibrate_epsilon
from .config import ACCGConfig
from .feature_model import CrossFittedFeatureModel, fit_cross_fitted_feature_model
from .selector import SelectionResult, select_action, straight_through_mask
from .torch_energy import TorchFeatureConstraint


class ACCGArm(str, Enum):
    NONE = "N"
    RANDOM = "R"
    TOPOLOGY = "T_s"
    CONSTRAINED = "T_c"


def _hash_array_update(digest: Any, array: np.ndarray | torch.Tensor) -> None:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    digest.update(np.ascontiguousarray(array).tobytes())


def _load_reusable_branchpoint(
    branchpoint_path: str | Path,
    *,
    config: ACCGConfig,
    seed: int,
    n_clusters: int,
    X_model: np.ndarray,
) -> tuple[Path, dict[str, Any]]:
    source = Path(branchpoint_path)
    if source.is_dir():
        source = source / "branchpoint.pt"
    if not source.is_file():
        raise FileNotFoundError(f"main ACCG branchpoint is missing: {source}")
    resolved_path = source.parent / "resolved_config.json"
    if not resolved_path.is_file():
        raise ValueError("reused branchpoint is missing its resolved main config")
    resolved_main = json.loads(resolved_path.read_text(encoding="utf-8"))
    if resolved_main.get("variant") != "accg_joint":
        raise ValueError("ablations must reuse the canonical accg_joint branchpoint")
    if resolved_main.get("v21") != asdict(config.v21):
        raise ValueError("ablation V21 settings differ from the canonical main panel")
    main_constraint = resolved_main.get("constraint", {})
    expected_main = {
        "selector_mode": "joint",
        "graph_control": "real",
        "infeasible_fallback": "least_violation",
    }
    observed_main = {key: main_constraint.get(key) for key in expected_main}
    if observed_main != expected_main:
        raise ValueError("reused branchpoint does not belong to the frozen canonical main policy")
    branchpoint = torch.load(source, map_location="cpu", weights_only=False)
    expected_identity = {
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "model_input_shape": [int(value) for value in X_model.shape],
        "model_input_hash": e1._hash_array(X_model),
    }
    for key, expected in expected_identity.items():
        if branchpoint.get(key) != expected:
            raise ValueError(f"reused branchpoint {key} does not match the requested panel")
    if int(branchpoint.get("epoch", -1)) != int(config.v21.warmup_epochs):
        raise ValueError("reused branchpoint warmup epoch does not match the ablation config")
    branch_state = branchpoint.get("model_state")
    branch_rng = branchpoint.get("rng")
    if not isinstance(branch_state, dict) or not isinstance(branch_rng, dict):
        raise ValueError("reused branchpoint is missing model or RNG state")
    return source, branchpoint


def _selection_for_accg(
    logits: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    *,
    row_ids: np.ndarray,
    z_all: np.ndarray,
    epsilon: np.ndarray,
    feature_model: CrossFittedFeatureModel,
    config: ACCGConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SelectionResult]:
    v21 = config.v21
    noisy = logits.detach().cpu().numpy().astype(np.float64) + float(v21.gumbel_scale) * tensors[
        "gumbel"
    ].detach().cpu().numpy().astype(np.float64)
    donor_offset_values = tensors["assignment_donor"].detach().cpu().numpy().astype(np.float32)
    donor_z = feature_model.transform.apply(donor_offset_values).astype(np.float64)
    selection = select_action(
        noisy,
        tensors["eligible"].detach().cpu().numpy(),
        z_all[row_ids],
        donor_z,
        row_ids=row_ids,
        epsilon=epsilon[row_ids],
        model=feature_model,
        mask_ratio=v21.assignment_mask_ratio,
        selector_mode=config.constraint.selector_mode,
        greedy_passes=config.constraint.selector_greedy_passes,
        pair_lookahead=config.constraint.selector_pair_lookahead,
        fallback=config.constraint.infeasible_fallback,
    )
    hard = torch.as_tensor(selection.hard_mask, dtype=logits.dtype, device=logits.device)
    budgets = torch.as_tensor(selection.budgets, dtype=torch.long, device=logits.device)
    mask_st = straight_through_mask(
        logits,
        tensors["eligible"],
        tensors["gumbel"],
        hard,
        budgets,
        gumbel_scale=v21.gumbel_scale,
        tau=v21.tau_ste,
    )
    return mask_st, hard, budgets, selection


def _audit_action(
    hard: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    *,
    row_ids: np.ndarray,
    z_all: np.ndarray,
    feature_model: CrossFittedFeatureModel,
) -> dict[str, np.ndarray]:
    donor = feature_model.transform.apply(tensors["assignment_donor"].detach().cpu().numpy()).astype(np.float64)
    masks = hard.detach().cpu().numpy().astype(np.bool_)
    delta = np.zeros(masks.shape[0], dtype=np.float64)
    clean = np.zeros_like(delta)
    action = np.zeros_like(delta)
    for local, row in enumerate(row_ids):
        delta[local], clean[local], action[local] = feature_model.fold_for_row(int(row)).action_delta(
            z_all[int(row)], donor[local], masks[local]
        )
    return {"joint_delta": delta, "clean_energy": clean, "action_energy": action}


def _loss_for_arm(
    arm: ACCGArm,
    components: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    stats_batch: torch.Tensor,
    *,
    row_ids: np.ndarray,
    z_all: np.ndarray,
    epsilon: np.ndarray,
    feature_model: CrossFittedFeatureModel,
    config: ACCGConfig,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, bool]]:
    v21 = config.v21
    model: V21AutoEncoder = components["model"]
    head: StudentTClusterHead = components["head"]
    gate: FeatureGate = components["gate"]
    _, parts = model.loss_encoder(tensors["corrupted"], tensors["batch"], tensors["training_mask"])
    base = parts["loss"]
    q_clean = head(model.encode(tensors["batch"]))
    infomax = information_maximization_loss(q_clean)
    js = base.new_zeros(())
    mask_st = torch.zeros_like(tensors["batch"])
    hard = torch.zeros_like(tensors["batch"])
    budgets = torch.zeros(tensors["batch"].shape[0], dtype=torch.long, device=tensors["batch"].device)
    selection: SelectionResult | None = None
    structural: dict[str, np.ndarray] = {
        "joint_delta": np.zeros(tensors["batch"].shape[0]),
        "clean_energy": np.zeros(tensors["batch"].shape[0]),
        "action_energy": np.zeros(tensors["batch"].shape[0]),
    }
    if arm is not ACCGArm.NONE:
        if arm is ACCGArm.RANDOM:
            logits = torch.zeros_like(tensors["batch"])
            mask_st, hard, budgets = e1._selection_from_logits(logits, tensors["eligible"], tensors["gumbel"], v21)
        elif arm is ACCGArm.TOPOLOGY:
            with torch.no_grad():
                logits = gate(stats_batch)
                mask_st, hard, budgets = e1._selection_from_logits(logits, tensors["eligible"], tensors["gumbel"], v21)
        else:
            with torch.no_grad():
                logits = gate(stats_batch)
                mask_st, hard, budgets, selection = _selection_for_accg(
                    logits,
                    tensors,
                    row_ids=row_ids,
                    z_all=z_all,
                    epsilon=epsilon,
                    feature_model=feature_model,
                    config=config,
                )
        assignment_corrupted = tensors["batch"] + hard * (tensors["assignment_donor"] - tensors["batch"])
        q_assignment = head(model.encode(assignment_corrupted))
        js = jensen_shannon_divergence(q_clean.detach(), q_assignment)
        if selection is None:
            structural = _audit_action(
                hard,
                tensors,
                row_ids=row_ids,
                z_all=z_all,
                feature_model=feature_model,
            )
        else:
            structural = {
                "joint_delta": selection.joint_delta,
                "clean_energy": selection.clean_energy,
                "action_energy": selection.action_energy,
            }
    total = base + float(v21.infomax_weight) * infomax + float(v21.assignment_weight) * js
    return total, {
        "base": base,
        "infomax": infomax,
        "js": js,
        "q_clean": q_clean,
        "mask_st": mask_st,
        "hard": hard,
        "budgets": budgets,
        "selection": selection,
        "structural": structural,
    }, {"assignment_forward": arm is not ACCGArm.NONE, "js_forward": arm is not ACCGArm.NONE}


def _gate_update_accg(
    components: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    stats_batch: torch.Tensor,
    *,
    row_ids: np.ndarray,
    z_all: np.ndarray,
    epsilon: np.ndarray,
    feature_model: CrossFittedFeatureModel,
    torch_constraint: TorchFeatureConstraint,
    config: ACCGConfig,
) -> dict[str, float]:
    v21 = config.v21
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
    mask_st, hard, _budgets, selection = _selection_for_accg(
        scores,
        tensors,
        row_ids=row_ids,
        z_all=z_all,
        epsilon=epsilon,
        feature_model=feature_model,
        config=config,
    )
    with torch.no_grad():
        q_reference = head(model.encode(tensors["batch"]))
    corrupted = tensors["batch"] + mask_st * (tensors["assignment_donor"] - tensors["batch"])
    q_gate = head(model.encode(corrupted))
    divergence = jensen_shannon_divergence(q_reference, q_gate)
    coverage = coverage_concentration(mask_st, tensors["eligible"])
    structural_delta = torch_constraint.joint_delta(
        tensors["batch"], tensors["assignment_donor"], mask_st, hard, row_ids
    )
    epsilon_tensor = torch.as_tensor(epsilon[row_ids], dtype=structural_delta.dtype, device=structural_delta.device)
    barrier = torch.relu(structural_delta - epsilon_tensor).mean()
    loss = -divergence + float(v21.gate_coverage_weight) * coverage + float(config.constraint.barrier_weight) * barrier
    loss.backward()
    grad_values = [parameter.grad.detach().norm().item() for parameter in gate.parameters() if parameter.grad is not None]
    grad_norm = float(np.sqrt(np.sum(np.square(grad_values)))) if grad_values else 0.0
    optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    model.train()
    head.train()
    return {
        "loss": float(loss.detach().cpu()),
        "divergence": float(divergence.detach().cpu()),
        "coverage": float(coverage.detach().cpu()),
        "barrier": float(barrier.detach().cpu()),
        "grad_norm": grad_norm,
        "hard_joint_delta_mean": float(np.mean(selection.joint_delta)),
    }


def _arm_train(
    arm: ACCGArm,
    state: dict[str, Any],
    X: torch.Tensor,
    stats: np.ndarray | torch.Tensor,
    schedule: e1.ScheduleBundle,
    n_clusters: int,
    *,
    z_all: np.ndarray,
    epsilon_calibration: EpsilonCalibration,
    feature_model: CrossFittedFeatureModel,
    config: ACCGConfig,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    v21 = config.v21
    components = e1._build_components(X, n_clusters, v21, seed, device)
    e1._load_state(components, state)
    model: V21AutoEncoder = components["model"]
    head: StudentTClusterHead = components["head"]
    gate: FeatureGate = components["gate"]
    optimizer: torch.optim.Optimizer = components["optimizer"]
    torch_constraint = TorchFeatureConstraint(feature_model)
    history: list[dict[str, Any]] = []
    assignment_forward_calls = 0
    js_forward_calls = 0
    shadow_assignment_calls = 0
    gradient_probe: dict[str, dict[str, float]] = {}
    selection_hashes = {name: hashlib.sha256() for name in ("eligible", "budget", "noise", "donor", "hard")}
    structural_sums = {
        "joint_delta": 0.0,
        "clean_energy": 0.0,
        "action_energy": 0.0,
        "constraint_infeasible": 0.0,
        "constraint_violated": 0.0,
        "fallback_count": 0.0,
        "safe_selected_count": 0.0,
        "selected_count": 0.0,
        "budget": 0.0,
    }
    structural_rows = 0
    gate_rows: list[dict[str, float]] = []
    trace_row_ids: list[np.ndarray] = []
    trace_masks: list[np.ndarray] = []
    trace_limit = int(config.constraint.selector_audit_rows)
    traced_rows = 0
    for step_index, entry in enumerate(schedule.post_branch):
        row_ids = np.asarray(entry.batch_ids, dtype=np.int64)
        tensors = e1._materialize_schedule(X, entry, v21, device)
        stats_batch = e1._stats_batch_on_device(stats, entry.batch_ids, device)
        _hash_array_update(selection_hashes["eligible"], tensors["eligible"])
        _hash_array_update(selection_hashes["donor"], tensors["reconstruction_donor"])
        _hash_array_update(selection_hashes["donor"], tensors["assignment_donor"])
        _hash_array_update(selection_hashes["noise"], tensors["gumbel"])
        shadow_assignment_calls += 1
        model.train()
        head.train()
        gate.train()
        optimizer.zero_grad(set_to_none=True)
        total, losses, counters = _loss_for_arm(
            arm,
            components,
            tensors,
            stats_batch,
            row_ids=row_ids,
            z_all=z_all,
            epsilon=epsilon_calibration.epsilon,
            feature_model=feature_model,
            config=config,
        )
        assignment_forward_calls += int(counters["assignment_forward"])
        js_forward_calls += int(counters["js_forward"])
        if arm is ACCGArm.NONE:
            eligible_count = tensors["eligible"].sum(dim=1)
            budgets = torch.ceil(eligible_count.to(tensors["batch"].dtype) * float(v21.assignment_mask_ratio)).to(torch.long)
            budgets = torch.minimum(budgets, eligible_count)
            _hash_array_update(selection_hashes["budget"], budgets)
        else:
            _hash_array_update(selection_hashes["budget"], losses["budgets"])
            _hash_array_update(selection_hashes["hard"], losses["hard"])
        if arm in {ACCGArm.TOPOLOGY, ACCGArm.CONSTRAINED} and len(gradient_probe) < 3:
            gradient_probe[f"P{len(gradient_probe)}"] = e1._gradient_probe(losses, components)
        total.backward()
        optimizer.step()
        if arm is ACCGArm.TOPOLOGY:
            gate_rows.append({"loss": e1._gate_update(components, tensors, stats_batch, v21)})
        elif arm is ACCGArm.CONSTRAINED:
            gate_rows.append(
                _gate_update_accg(
                    components,
                    tensors,
                    stats_batch,
                    row_ids=row_ids,
                    z_all=z_all,
                    epsilon=epsilon_calibration.epsilon,
                    feature_model=feature_model,
                    torch_constraint=torch_constraint,
                    config=config,
                )
            )
        selection = losses["selection"]
        if arm is not ACCGArm.NONE:
            structural_rows += int(row_ids.size)
            for name in ("joint_delta", "clean_energy", "action_energy"):
                structural_sums[name] += float(np.sum(losses["structural"][name]))
            if selection is not None:
                structural_sums["constraint_infeasible"] += float(np.sum(selection.constraint_infeasible))
                structural_sums["constraint_violated"] += float(np.sum(selection.constraint_violated))
                structural_sums["fallback_count"] += float(np.sum(selection.fallback_counts))
                structural_sums["safe_selected_count"] += float(np.sum(selection.safe_selected_counts))
                structural_sums["selected_count"] += float(np.sum(selection.selected_counts))
                structural_sums["budget"] += float(np.sum(selection.budgets))
            else:
                structural_sums["selected_count"] += float(losses["hard"].sum().detach().cpu())
                structural_sums["budget"] += float(losses["budgets"].sum().detach().cpu())
            if arm in {ACCGArm.TOPOLOGY, ACCGArm.CONSTRAINED} and traced_rows < trace_limit:
                take = min(trace_limit - traced_rows, int(row_ids.size))
                trace_row_ids.append(row_ids[:take].copy())
                trace_masks.append(losses["hard"][:take].detach().cpu().numpy().astype(np.bool_))
                traced_rows += take
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
                "joint_structural_delta": float(np.mean(losses["structural"]["joint_delta"])),
                "constraint_infeasible_rate": 0.0
                if selection is None
                else float(np.mean(selection.constraint_infeasible)),
            }
        )
    embedding = e1._clean_embedding(model, X, v21.batch_size)
    readout = e1._metrics(embedding, n_clusters, seed, None, kmeans_n_init=v21.kmeans_n_init)
    denominator = max(1, structural_rows)
    structural_audit = {
        "rows": int(structural_rows),
        "joint_delta_mean": structural_sums["joint_delta"] / denominator,
        "clean_energy_mean": structural_sums["clean_energy"] / denominator,
        "action_energy_mean": structural_sums["action_energy"] / denominator,
        "constraint_infeasible_rate": structural_sums["constraint_infeasible"] / denominator,
        "constraint_violation_rate": structural_sums["constraint_violated"] / denominator,
        "fallback_per_row": structural_sums["fallback_count"] / denominator,
        "safe_selected_per_row": structural_sums["safe_selected_count"] / denominator,
        "selected_per_row": structural_sums["selected_count"] / denominator,
        "budget_fill": structural_sums["selected_count"] / max(1.0, structural_sums["budget"]),
        "epsilon_scope": config.constraint.epsilon_scope,
        "epsilon_mean": float(np.mean(epsilon_calibration.epsilon)),
    }
    gate_audit = {
        "updates": len(gate_rows),
        "loss_mean": float(np.mean([row["loss"] for row in gate_rows])) if gate_rows else 0.0,
        "barrier_mean": float(np.mean([row.get("barrier", 0.0) for row in gate_rows])) if gate_rows else 0.0,
        "nonzero_grad_rate": float(np.mean([row.get("grad_norm", 0.0) > 0.0 for row in gate_rows])) if gate_rows else 0.0,
    }
    return {
        "arm": arm.value,
        "status": "completed",
        "history": history,
        "embedding": embedding,
        "predictions": readout["predictions"],
        "metrics": readout["metrics"],
        "gradient_probe": gradient_probe,
        "structural_audit": structural_audit,
        "gate_audit": gate_audit,
        "selection_trace": {
            "row_ids": np.concatenate(trace_row_ids) if trace_row_ids else np.empty(0, dtype=np.int64),
            "hard_masks": np.concatenate(trace_masks, axis=0)
            if trace_masks
            else np.empty((0, X.shape[1]), dtype=np.bool_),
            "sampling": "first scheduled post-branch rows up to selector_audit_rows",
        },
        "audit": {
            "optimizer": "Adam",
            "optimizer_foreach": components.get("optimizer_foreach"),
            "optimizer_fused": components.get("optimizer_fused", False),
            "assignment_forward_calls": assignment_forward_calls,
            "js_forward_calls": js_forward_calls,
            "shadow_assignment_calls": shadow_assignment_calls,
            "eligible_schedule_hash": selection_hashes["eligible"].hexdigest(),
            "budget_schedule_hash": selection_hashes["budget"].hexdigest(),
            "selection_noise_hash": selection_hashes["noise"].hexdigest(),
            "donor_schedule_hash": selection_hashes["donor"].hexdigest(),
            "hard_action_hash": selection_hashes["hard"].hexdigest(),
            "labels_used_during_fit": False,
        },
        "checkpoint": {
            "model": e1._snapshot_to_cpu(model.state_dict()),
            "head": e1._snapshot_to_cpu(head.state_dict()),
            "gate": e1._snapshot_to_cpu(gate.state_dict()),
            "optimizer": e1._snapshot_to_cpu(optimizer.state_dict()),
            "gate_optimizer": e1._snapshot_to_cpu(components["gate_optimizer"].state_dict()),
        },
    }


def _pair_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    a = left.get("metrics", {}).get(key)
    b = right.get("metrics", {}).get(key)
    return None if a is None or b is None else float(a) - float(b)


def run_matched_panel(
    X_model: np.ndarray,
    X_graph: Any,
    *,
    n_clusters: int,
    config: ACCGConfig,
    seed: int,
    device: str | torch.device = "cpu",
    evaluation_labels: np.ndarray | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one shared-branchpoint N/R/T_s/T_c panel without labels in fit."""

    config.validate()
    if n_clusters <= 1:
        raise ValueError("n_clusters must be greater than one")
    runtime_device = torch.device(device)
    e1._seed_all(seed, runtime_device)
    X_np = np.ascontiguousarray(np.asarray(X_model, dtype=np.float32))
    if X_np.ndim != 2 or X_np.shape[0] < n_clusters or not np.isfinite(X_np).all():
        raise ValueError("X_model must be finite 2D data with at least n_clusters rows")
    feature_model = fit_cross_fitted_feature_model(X_np, config=config.constraint, seed=seed)
    z_all = feature_model.transform_matrix(X_np).astype(np.float64)
    epsilon_calibration = calibrate_epsilon(
        X_np,
        feature_model,
        mask_ratio=config.v21.assignment_mask_ratio,
        config=config.constraint,
        seed=seed,
    )
    data_device = torch.device("cpu") if runtime_device.type == "cuda" else runtime_device
    X = torch.as_tensor(X_np, dtype=torch.float32, device=data_device)
    graph = build_svd_knn_graph(
        X_graph,
        neighbor_k=config.v21.neighbor_k,
        svd_target=config.v21.graph_svd_target,
        svd_min_dim=min(config.v21.graph_svd_min_dim, max(1, X_np.shape[0] - 1)),
        svd_max_dim=min(config.v21.graph_svd_max_dim, max(1, X_np.shape[0] - 1)),
        seed=seed,
    )
    stats_np, stats_profile = compute_topology_statistics(
        X_np,
        graph,
        block_size=config.v21.stats_block_size,
        cache_dir=(Path(output_dir) / "cache") if output_dir is not None else None,
        cache_dtype=config.v21.stats_cache_dtype,
        clip=config.v21.stats_clip,
    )
    schedule = e1._make_schedule(X_np.shape[0], config.v21, seed)
    components = e1._build_components(X, n_clusters, config.v21, seed, runtime_device)
    e1._run_warmup(components, X, schedule.warmup, config.v21, runtime_device)
    e1._initialise_head(components, X, config.v21, seed)
    branch_state = e1._state_for_save(components)
    branch_rng = e1._capture_rng(runtime_device)
    branchpoint = {
        "protocol_id": config.protocol_id,
        "variant": config.variant,
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "model_input_shape": [int(value) for value in X_np.shape],
        "model_input_hash": e1._hash_array(X_np),
        "epoch": config.v21.warmup_epochs,
        "head_initialised": bool(components["head"].initialised),
        "schedule_hashes": schedule.hashes,
        "topology_statistics_hash": e1._hash_array(stats_np),
        "sample_graph_profile": graph.profile,
        "stats_profile": stats_profile,
        "feature_model_profile": feature_model.profile,
        "epsilon_profile": epsilon_calibration.profile,
        "model_state": branch_state,
        "rng": branch_rng,
        "batch_permutation_state": schedule.batch_rng_state,
    }
    del components
    e1._release_cuda_cache(runtime_device)
    arms: dict[str, Any] = {}
    for arm in (ACCGArm.NONE, ACCGArm.RANDOM, ACCGArm.TOPOLOGY, ACCGArm.CONSTRAINED):
        e1._restore_rng(branch_rng, runtime_device)
        arms[arm.value] = _arm_train(
            arm,
            branch_state,
            X,
            stats_np,
            schedule,
            n_clusters,
            z_all=z_all,
            epsilon_calibration=epsilon_calibration,
            feature_model=feature_model,
            config=config,
            seed=seed,
            device=runtime_device,
        )
        e1._release_cuda_cache(runtime_device)
    if evaluation_labels is not None:
        encoded = np.asarray(evaluation_labels).astype(str)
        if encoded.shape[0] != X_np.shape[0]:
            raise ValueError("evaluation_labels must have one entry per sample")
        for item in arms.values():
            item["metrics"].update(
                {
                    "ari": float(adjusted_rand_score(encoded, item["predictions"])),
                    "nmi": float(normalized_mutual_info_score(encoded, item["predictions"])),
                    "labels_used_after_fit_only": True,
                }
            )
    pairs = {
        "I_R_minus_N_ARI": _pair_delta(arms[ACCGArm.RANDOM.value], arms[ACCGArm.NONE.value], "ari"),
        "S_Ts_minus_R_ARI": _pair_delta(arms[ACCGArm.TOPOLOGY.value], arms[ACCGArm.RANDOM.value], "ari"),
        "C_Tc_minus_Ts_ARI": _pair_delta(arms[ACCGArm.CONSTRAINED.value], arms[ACCGArm.TOPOLOGY.value], "ari"),
        "C_Tc_minus_N_ARI": _pair_delta(arms[ACCGArm.CONSTRAINED.value], arms[ACCGArm.NONE.value], "ari"),
    }
    reference = arms[ACCGArm.RANDOM.value]["audit"]
    matching = {}
    for arm in (ACCGArm.TOPOLOGY, ACCGArm.CONSTRAINED):
        audit = arms[arm.value]["audit"]
        matching[arm.value] = {
            "donor": audit["donor_schedule_hash"] == reference["donor_schedule_hash"],
            "eligible": audit["eligible_schedule_hash"] == reference["eligible_schedule_hash"],
            "budget": audit["budget_schedule_hash"] == reference["budget_schedule_hash"],
            "selection_noise": audit["selection_noise_hash"] == reference["selection_noise_hash"],
        }
    audit = {
        "protocol_id": config.protocol_id,
        "labels_used_during_fit": False,
        "K_used_during_fit": True,
        "K_source": "caller_outer_evaluation",
        "arm_names": [arm.value for arm in ACCGArm],
        "matched_schedule": matching,
        "joint_action_primary": config.constraint.selector_mode == "joint",
        "exact_budget_primary": config.constraint.infeasible_fallback == "least_violation",
        "feature_graph_cross_fitted": True,
        "outcomes_used_for_epsilon": False,
    }
    result = {
        "status": "completed",
        "protocol_id": config.protocol_id,
        "variant": config.variant,
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "labels_used_during_fit": False,
        "branchpoint": branchpoint,
        "schedule": {
            "hashes": schedule.hashes,
            "warmup_entries": len(schedule.warmup),
            "post_branch_entries": len(schedule.post_branch),
        },
        "pairs": pairs,
        "audit": audit,
        "arms": arms,
    }
    if output_dir is not None:
        _write_result(Path(output_dir), result, config, epsilon_calibration)
    return result


def run_constrained_from_branchpoint(
    X_model: np.ndarray,
    X_graph: Any,
    *,
    n_clusters: int,
    config: ACCGConfig,
    seed: int,
    branchpoint_path: str | Path,
    device: str | torch.device = "cpu",
    evaluation_labels: np.ndarray | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run only an ACCG control arm while reusing the main panel branchpoint."""

    config.validate()
    runtime_device = torch.device(device)
    e1._seed_all(seed, runtime_device)
    X_np = np.ascontiguousarray(np.asarray(X_model, dtype=np.float32))
    if X_np.ndim != 2 or X_np.shape[0] < n_clusters or not np.isfinite(X_np).all():
        raise ValueError("X_model must be finite 2D data with at least n_clusters rows")
    source, branchpoint = _load_reusable_branchpoint(
        branchpoint_path,
        config=config,
        seed=seed,
        n_clusters=n_clusters,
        X_model=X_np,
    )
    branch_state = branchpoint.get("model_state")
    branch_rng = branchpoint.get("rng")
    feature_model = fit_cross_fitted_feature_model(X_np, config=config.constraint, seed=seed)
    z_all = feature_model.transform_matrix(X_np).astype(np.float64)
    epsilon_calibration = calibrate_epsilon(
        X_np,
        feature_model,
        mask_ratio=config.v21.assignment_mask_ratio,
        config=config.constraint,
        seed=seed,
    )
    data_device = torch.device("cpu") if runtime_device.type == "cuda" else runtime_device
    X = torch.as_tensor(X_np, dtype=torch.float32, device=data_device)
    graph = build_svd_knn_graph(
        X_graph,
        neighbor_k=config.v21.neighbor_k,
        svd_target=config.v21.graph_svd_target,
        svd_min_dim=min(config.v21.graph_svd_min_dim, max(1, X_np.shape[0] - 1)),
        svd_max_dim=min(config.v21.graph_svd_max_dim, max(1, X_np.shape[0] - 1)),
        seed=seed,
    )
    stats_np, stats_profile = compute_topology_statistics(
        X_np,
        graph,
        block_size=config.v21.stats_block_size,
        cache_dir=(Path(output_dir) / "cache") if output_dir is not None else None,
        cache_dtype=config.v21.stats_cache_dtype,
        clip=config.v21.stats_clip,
    )
    stats_hash = e1._hash_array(stats_np)
    if branchpoint.get("topology_statistics_hash") != stats_hash:
        raise ValueError("recomputed topology statistics do not match the reused branchpoint")
    schedule = e1._make_schedule(X_np.shape[0], config.v21, seed)
    if branchpoint.get("schedule_hashes") != schedule.hashes:
        raise ValueError("recomputed schedule does not match the reused branchpoint")
    e1._restore_rng(branch_rng, runtime_device)
    arm = _arm_train(
        ACCGArm.CONSTRAINED,
        branch_state,
        X,
        stats_np,
        schedule,
        n_clusters,
        z_all=z_all,
        epsilon_calibration=epsilon_calibration,
        feature_model=feature_model,
        config=config,
        seed=seed,
        device=runtime_device,
    )
    if evaluation_labels is not None:
        encoded = np.asarray(evaluation_labels).astype(str)
        if encoded.shape[0] != X_np.shape[0]:
            raise ValueError("evaluation_labels must have one entry per sample")
        arm["metrics"].update(
            {
                "ari": float(adjusted_rand_score(encoded, arm["predictions"])),
                "nmi": float(normalized_mutual_info_score(encoded, arm["predictions"])),
                "labels_used_after_fit_only": True,
            }
        )
    result = {
        "status": "completed",
        "protocol_id": config.protocol_id,
        "variant": config.variant,
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "labels_used_during_fit": False,
        "reused_from": str(source.parent.resolve()),
        "reused_controls": ["N", "R", "T_s"],
        "branchpoint": {
            "epoch": branchpoint["epoch"],
            "schedule_hashes": schedule.hashes,
            "topology_statistics_hash": stats_hash,
            "sample_graph_profile": graph.profile,
            "stats_profile": stats_profile,
            "feature_model_profile": feature_model.profile,
            "epsilon_profile": epsilon_calibration.profile,
        },
        "audit": {
            "branchpoint_reused": True,
            "controls_recomputed": False,
            "labels_used_during_fit": False,
            "joint_action_primary": config.constraint.selector_mode == "joint",
        },
        "arms": {ACCGArm.CONSTRAINED.value: arm},
    }
    if output_dir is not None:
        _write_ablation_result(Path(output_dir), result, config, epsilon_calibration)
    return result


def _write_result(out: Path, result: dict[str, Any], config: ACCGConfig, epsilon: EpsilonCalibration) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    branchpoint = result["branchpoint"]
    torch.save(branchpoint, out / "branchpoint.pt")
    branch_meta = {key: value for key, value in branchpoint.items() if key not in {"model_state", "rng", "batch_permutation_state"}}
    (out / "branchpoint_metadata.json").write_text(
        json.dumps(branch_meta, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8"
    )
    np.save(out / "epsilon_per_sample.npy", epsilon.epsilon)
    np.save(out / "epsilon_null_deltas.npy", epsilon.sampled_deltas)
    (out / "epsilon_calibration.json").write_text(
        json.dumps(epsilon.profile, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (out / "schedule_manifest.json").write_text(
        json.dumps(result["schedule"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    pairs_dir = out / "pairs"
    pairs_dir.mkdir(exist_ok=True)
    (pairs_dir / "primary.json").write_text(
        json.dumps(result["pairs"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    for arm_name, arm in result["arms"].items():
        arm_out = out / arm_name
        arm_out.mkdir(exist_ok=True)
        np.save(arm_out / "embedding_final.npy", arm["embedding"])
        np.save(arm_out / "predictions.npy", arm["predictions"])
        (arm_out / "metrics.json").write_text(json.dumps(arm["metrics"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        (arm_out / "history.json").write_text(json.dumps(arm["history"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        (arm_out / "audit.json").write_text(json.dumps(arm["audit"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        (arm_out / "structural_audit.json").write_text(
            json.dumps(arm["structural_audit"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        (arm_out / "gate_audit.json").write_text(json.dumps(arm["gate_audit"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        (arm_out / "gradient_probe.json").write_text(
            json.dumps(arm["gradient_probe"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            arm_out / "selection_trace.npz",
            row_ids=arm["selection_trace"]["row_ids"],
            hard_masks=arm["selection_trace"]["hard_masks"],
        )
        torch.save(arm["checkpoint"], arm_out / "checkpoint.pt")
    (out / "audit.json").write_text(json.dumps(result["audit"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key not in {"branchpoint", "arms"}}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8"
    )


def _write_ablation_result(
    out: Path,
    result: dict[str, Any],
    config: ACCGConfig,
    epsilon: EpsilonCalibration,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    np.save(out / "epsilon_per_sample.npy", epsilon.epsilon)
    np.save(out / "epsilon_null_deltas.npy", epsilon.sampled_deltas)
    (out / "epsilon_calibration.json").write_text(
        json.dumps(epsilon.profile, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    arm = result["arms"][ACCGArm.CONSTRAINED.value]
    arm_out = out / ACCGArm.CONSTRAINED.value
    arm_out.mkdir(exist_ok=True)
    np.save(arm_out / "embedding_final.npy", arm["embedding"])
    np.save(arm_out / "predictions.npy", arm["predictions"])
    for name in ("metrics", "audit", "structural_audit", "gate_audit", "gradient_probe"):
        (arm_out / f"{name}.json").write_text(
            json.dumps(arm[name], indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
    (arm_out / "history.json").write_text(
        json.dumps(arm["history"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        arm_out / "selection_trace.npz",
        row_ids=arm["selection_trace"]["row_ids"],
        hard_masks=arm["selection_trace"]["hard_masks"],
    )
    torch.save(arm["checkpoint"], arm_out / "checkpoint.pt")
    summary = {key: value for key, value in result.items() if key not in {"arms"}}
    summary["T_c_metrics"] = arm["metrics"]
    summary["T_c_structural_audit"] = arm["structural_audit"]
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8"
    )
