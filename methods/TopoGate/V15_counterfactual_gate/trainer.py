from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from .config import V15Config
from .graph import (
    CandidateGraph,
    build_candidate_graph,
    refresh_latent_graph,
    replace_candidate_edges,
    restrict_candidate_scope,
)
from .model import V15Model, abstaining_sparsemax, ema_update, make_teacher
from .sparse import PreparedInput, apply_mask, sparse_reconstruction_per_sample


# ``direct_*`` modes are exact counterfactual readouts.  The learned mode uses
# the same detached target but fits an amortized scorer on the training split;
# keeping the sets explicit prevents an ablation from silently changing its
# readout semantics.
_EXACT_COUNTERFACTUAL_MODES = {"direct_target", "direct_counterfactual"}


@dataclass
class TrainingResult:
    embedding: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    history: list[dict]
    graph_history: list[dict]
    graph: CandidateGraph
    train_seconds: float
    gate_diagnostics: dict[str, np.ndarray]
    teacher_diagnostics: dict[str, np.ndarray]
    teacher_selection: dict[str, object]
    cluster_frequency_ema: np.ndarray


def kl_per_sample(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    target = target.clamp_min(1e-8)
    prediction = prediction.clamp_min(1e-8)
    return torch.sum(target * (torch.log(target) - torch.log(prediction)), dim=1)


def jsd_per_sample(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("JSD inputs must have identical shapes")
    first = first.clamp_min(1e-8)
    second = second.clamp_min(1e-8)
    midpoint = 0.5 * (first + second)
    return 0.5 * kl_per_sample(first, midpoint) + 0.5 * kl_per_sample(second, midpoint)


def align_teacher_assignments(
    latent_probabilities: torch.Tensor,
    raw_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Align raw-view component ids to latent ids without benchmark labels."""
    if latent_probabilities.shape != raw_probabilities.shape or latent_probabilities.ndim != 2:
        raise ValueError("teacher views must have identical [N, K] shapes")
    latent = latent_probabilities.clamp_min(1e-8)
    raw = raw_probabilities.clamp_min(1e-8)
    latent = latent / latent.sum(dim=1, keepdim=True).clamp_min(1e-8)
    raw = raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
    overlap = (latent.transpose(0, 1) @ raw).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-overlap)
    permutation = np.empty(latent.shape[1], dtype=np.int64)
    permutation[row_ind] = col_ind
    return raw[:, torch.as_tensor(permutation, dtype=torch.long, device=raw.device)].detach()


def build_teacher_reference(
    latent_probabilities: torch.Tensor,
    raw_probabilities: torch.Tensor,
    *,
    mode: str,
    raw_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine independent assignments into one detached reference target.

    The arithmetic mixture keeps raw and latent views as one soft reference
    rather than creating a second reliability head. Disagreement is retained
    as a certificate and naturally produces a softer target.
    """
    if latent_probabilities.shape != raw_probabilities.shape or latent_probabilities.ndim != 2:
        raise ValueError("teacher reference views must have identical [N, K] shapes")
    latent = latent_probabilities.clamp_min(1e-8)
    raw = raw_probabilities.clamp_min(1e-8)
    latent = latent / latent.sum(dim=1, keepdim=True).clamp_min(1e-8)
    raw = raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
    # KMeans component ids are arbitrary in each view. Align raw columns to
    # latent columns using only their assignment overlap, never benchmark y.
    raw = align_teacher_assignments(latent, raw)
    if mode == "latent":
        reference = latent
        disagreement = torch.zeros(latent.shape[0], dtype=latent.dtype, device=latent.device)
        agreement = torch.ones_like(disagreement)
    elif mode == "raw":
        reference = raw
        disagreement = torch.zeros(raw.shape[0], dtype=raw.dtype, device=raw.device)
        agreement = torch.ones_like(disagreement)
    elif mode == "consensus":
        weight = float(raw_weight)
        reference = (1.0 - weight) * latent + weight * raw
        reference = reference / reference.sum(dim=1, keepdim=True).clamp_min(1e-8)
        disagreement = jsd_per_sample(latent, raw)
        agreement = torch.exp(-disagreement / float(temperature)).clamp(0.0, 1.0)
    else:
        raise ValueError(f"unknown teacher reference mode: {mode}")
    return reference.detach(), agreement.detach(), disagreement.detach()


def assignment_margin(probabilities: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 assignment probability, per sample."""
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("assignment probabilities must be [N,K] with K >= 2")
    top = torch.topk(probabilities, k=2, dim=1).values
    return top[:, 0] - top[:, 1]


def view_local_assignment_quality(
    probabilities: np.ndarray | torch.Tensor,
    neighbor_indices: np.ndarray,
) -> dict[str, float]:
    """Score a view without labels or cluster-id alignment.

    The hard local agreement is invariant to arbitrary prototype identities and
    exposes a geometrically stable but semantically fragmented view. A very
    collapsed assignment is explicitly penalized, while legitimate class
    imbalance is not forced toward a uniform prior.
    """
    q = probabilities.detach().cpu().numpy() if isinstance(probabilities, torch.Tensor) else np.asarray(probabilities)
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2 or neighbor_indices.ndim != 2 or q.shape[0] != neighbor_indices.shape[0]:
        raise ValueError("probabilities and neighbor_indices must have compatible [N,*] shapes")
    q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-8, None)
    labels = q.argmax(axis=1)
    valid = (neighbor_indices >= 0) & (neighbor_indices < q.shape[0])
    if not np.any(valid):
        return {"local_agreement": 0.0, "confidence": 0.0, "effective_clusters": 0.0, "score": -1.0}
    rows = np.broadcast_to(np.arange(q.shape[0])[:, None], neighbor_indices.shape)
    local_agreement = float(np.mean((labels[rows[valid]] == labels[neighbor_indices[valid]]).astype(np.float32)))
    confidence = float(np.mean(np.sort(q, axis=1)[:, -1] - np.sort(q, axis=1)[:, -2])) if q.shape[1] > 1 else 1.0
    marginal = q.mean(axis=0)
    marginal_entropy = float(-(marginal * np.log(np.clip(marginal, 1e-8, None))).sum())
    effective_clusters = float(np.exp(marginal_entropy))
    # A view that activates only a small fraction of requested clusters can
    # look locally coherent while being useless as a global teacher.  The
    # 60%-of-K floor rejects that partial collapse without forcing balanced
    # ground-truth class frequencies.
    collapse_penalty = 1.0 if effective_clusters < max(1.5, 0.6 * q.shape[1]) else 0.0
    score = local_agreement + 0.05 * confidence - collapse_penalty
    return {
        "local_agreement": local_agreement,
        "confidence": confidence,
        "effective_clusters": effective_clusters,
        "score": float(score),
    }


def _zero_preserving_robust_scale(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Scale a batch globally without shifting the meaningful zero point."""
    selected = values[valid].abs()
    if selected.numel() == 0:
        return torch.ones((), dtype=values.dtype, device=values.device)
    # Exact zero is the abstention boundary, not evidence that the effect-size
    # scale is tiny. Excluding exact/negligible zeros prevents a sparse utility
    # batch from amplifying numerical noise to the clipping boundary.
    nonzero = selected[selected > 1e-8]
    if nonzero.numel() == 0:
        return torch.ones((), dtype=values.dtype, device=values.device)
    scale = nonzero.median()
    return (1.4826 * scale).clamp_min(1e-3).detach()


def _normalise_edge_feature(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Scale a detached edge diagnostic without moving its zero point."""
    if values.shape != valid.shape:
        raise ValueError("edge feature and validity mask must have identical shapes")
    scale = _zero_preserving_robust_scale(values, valid)
    return (values / scale).clamp(-8.0, 8.0).detach()


def _assemble_utility_features(
    graph_features: np.ndarray | torch.Tensor,
    teacher_assignment_agreement: torch.Tensor,
    semantic_help: torch.Tensor,
    reconstruction_damage: torch.Tensor,
    valid: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build the scorer input shared by training and final evaluation."""
    raw_features = torch.as_tensor(graph_features, dtype=torch.float32, device=device)
    if raw_features.ndim != 3 or raw_features.shape[2] < 6:
        raise ValueError("graph features must have shape [B, M, >=6]")
    features = raw_features[:, :, :6].clone()
    if teacher_assignment_agreement.shape != valid.shape:
        raise ValueError("teacher agreement shape must match valid edges")
    features[:, :, 5] = teacher_assignment_agreement.detach()
    return features


def latent_variance_floor_loss(z: torch.Tensor, floor: float) -> torch.Tensor:
    """Penalize collapsed latent dimensions without decorrelating semantics."""
    if z.ndim != 2 or z.shape[0] < 2:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(float(floor) - std).mean()


def latent_view_consistency_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("latent views must have identical shapes")
    return (1.0 - F.cosine_similarity(first, second, dim=1)).mean()


def _uniform_edge_distribution(valid: torch.Tensor) -> torch.Tensor:
    """Return a null-free uniform union baseline, with null for empty rows."""
    counts = valid.sum(dim=1, keepdim=True)
    edge_mass = valid.to(dtype=torch.float32) / counts.clamp_min(1).to(dtype=torch.float32)
    output = torch.cat([torch.zeros_like(counts, dtype=torch.float32), edge_mass], dim=1)
    empty = counts.squeeze(1).eq(0)
    if torch.any(empty):
        output[empty, 0] = 1.0
    return output


def _forced_topk_distribution(scores: torch.Tensor, valid: torch.Tensor, topk: int) -> torch.Tensor:
    """Force a fixed number of valid neighbours for the registered baseline."""
    masked = scores.masked_fill(~valid, -1e9)
    k = min(int(topk), scores.shape[1])
    selected = torch.zeros_like(valid)
    if k > 0:
        selected.scatter_(1, torch.topk(masked, k=k, dim=1).indices, True)
        selected &= valid
    counts = selected.sum(dim=1, keepdim=True)
    output = torch.cat(
        [torch.zeros_like(counts, dtype=torch.float32), selected.to(dtype=torch.float32) / counts.clamp_min(1)],
        dim=1,
    )
    empty = counts.squeeze(1).eq(0)
    if torch.any(empty):
        output[empty, 0] = 1.0
    return output


def _shuffle_scores_within_valid(
    scores: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Break edge-utility correspondence while preserving each row's scores."""
    if scores.shape != valid.shape:
        raise ValueError("scores and valid must have identical shapes")
    shuffled = scores.clone()
    for row in range(scores.shape[0]):
        positions = torch.where(valid[row])[0]
        if positions.numel() <= 1:
            continue
        permutation = positions[torch.randperm(positions.numel(), device=scores.device)]
        shuffled[row, positions] = scores[row, permutation]
    return shuffled


def counterfactual_components(
    q_teacher: torch.Tensor,
    q_self: torch.Tensor,
    q_edge: torch.Tensor,
    rec_self: torch.Tensor,
    rec_edge: torch.Tensor,
    *,
    valid: torch.Tensor,
    q_self_second: torch.Tensor | None = None,
    q_edge_second: torch.Tensor | None = None,
    rec_self_second: torch.Tensor | None = None,
    rec_edge_second: torch.Tensor | None = None,
    stability_weight: float = 0.0,
    confidence_weight: float = 0.0,
    opportunity_temperature: float = 0.0,
    edge_agreement: torch.Tensor | None = None,
    agreement_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q_teacher.ndim != 2 or q_self.ndim != 2 or q_edge.ndim != 3 or rec_edge.ndim != 2:
        raise ValueError("q_edge must be [B,M,K] and rec_edge must be [B,M]")
    batch, candidates, clusters = q_edge.shape
    if q_teacher.shape != (batch, clusters) or q_self.shape != (batch, clusters):
        raise ValueError("q_teacher and q_self must be [B,K] matching q_edge")
    if rec_self.shape != (batch,) or rec_edge.shape != (batch, candidates):
        raise ValueError("rec_self must be [B] and rec_edge must be [B,M]")
    if valid.shape != (batch, candidates):
        raise ValueError("valid must be [B,M]")
    q_teacher_edge = (
        q_teacher[:, None, :]
        .expand(-1, candidates, -1)
        .reshape(-1, clusters)
    )
    semantic = kl_per_sample(q_teacher, q_self)[:, None] - kl_per_sample(
        q_teacher_edge,
        q_edge.reshape(-1, clusters),
    ).reshape(batch, candidates)
    damage = rec_edge - rec_self[:, None]

    if edge_agreement is not None:
        if edge_agreement.shape != valid.shape:
            raise ValueError("edge_agreement must have shape [B,M]")
        # Uniform assignment is the zero-information baseline. The detached
        # agreement term makes a same-component donor useful even when the
        # single probe does not flip the teacher argmax.
        agreement_help = edge_agreement - (1.0 / float(clusters))
        semantic = semantic + float(agreement_weight) * agreement_help

    second_args = (q_self_second, q_edge_second, rec_self_second, rec_edge_second)
    if any(value is not None for value in second_args):
        if not all(value is not None for value in second_args):
            raise ValueError("all second-view counterfactual arguments are required together")
        assert q_self_second is not None
        assert q_edge_second is not None
        assert rec_self_second is not None
        assert rec_edge_second is not None
        if q_self_second.shape != q_self.shape or q_edge_second.shape != q_edge.shape:
            raise ValueError("second-view assignment shapes must match first-view shapes")
        if rec_self_second.shape != rec_self.shape or rec_edge_second.shape != rec_edge.shape:
            raise ValueError("second-view reconstruction shapes must match first-view shapes")
        semantic_second = kl_per_sample(q_teacher, q_self_second)[:, None] - kl_per_sample(
            q_teacher_edge,
            q_edge_second.reshape(-1, clusters),
        ).reshape(batch, candidates)
        semantic = 0.5 * (semantic + semantic_second)
        if stability_weight > 0.0:
            self_stability = jsd_per_sample(q_self, q_self_second)[:, None]
            edge_stability = jsd_per_sample(
                q_edge.reshape(-1, clusters),
                q_edge_second.reshape(-1, clusters),
            ).reshape(batch, candidates)
            semantic = semantic + float(stability_weight) * (self_stability - edge_stability)
        if confidence_weight > 0.0:
            self_margin = 0.5 * (assignment_margin(q_self) + assignment_margin(q_self_second))
            edge_margin = 0.5 * (
                assignment_margin(q_edge.reshape(-1, clusters)).reshape(batch, candidates)
                + assignment_margin(q_edge_second.reshape(-1, clusters)).reshape(batch, candidates)
            )
            semantic = semantic + float(confidence_weight) * (edge_margin - self_margin[:, None])
        if opportunity_temperature > 0.0:
            residual = 0.5 * (
                kl_per_sample(q_teacher, q_self)
                + kl_per_sample(q_teacher, q_self_second)
            )
            opportunity = residual / (residual + float(opportunity_temperature))
            semantic = semantic * opportunity[:, None]
        damage = 0.5 * (
            damage
            + rec_edge_second
            - rec_self_second[:, None]
        )
    return semantic, damage


def counterfactual_utility(
    q_teacher: torch.Tensor,
    q_self: torch.Tensor,
    q_edge: torch.Tensor,
    rec_self: torch.Tensor,
    rec_edge: torch.Tensor,
    *,
    lambda_rec: float,
    clip: float,
    valid: torch.Tensor,
    q_self_second: torch.Tensor | None = None,
    q_edge_second: torch.Tensor | None = None,
    rec_self_second: torch.Tensor | None = None,
    rec_edge_second: torch.Tensor | None = None,
    stability_weight: float = 0.0,
    confidence_weight: float = 0.0,
    opportunity_temperature: float = 0.0,
    edge_agreement: torch.Tensor | None = None,
    agreement_weight: float = 0.0,
) -> torch.Tensor:
    """Compute detached zero-preserving utility relative to the self branch."""
    semantic, damage = counterfactual_components(
        q_teacher,
        q_self,
        q_edge,
        rec_self,
        rec_edge,
        valid=valid,
        q_self_second=q_self_second,
        q_edge_second=q_edge_second,
        rec_self_second=rec_self_second,
        rec_edge_second=rec_edge_second,
        stability_weight=stability_weight,
        confidence_weight=confidence_weight,
        opportunity_temperature=opportunity_temperature,
        edge_agreement=edge_agreement,
        agreement_weight=agreement_weight,
    )
    semantic_scale = _zero_preserving_robust_scale(semantic, valid)
    damage_scale = _zero_preserving_robust_scale(damage, valid)
    utility = semantic / semantic_scale - float(lambda_rec) * damage / damage_scale
    utility = utility.clamp(-float(clip), float(clip))
    return utility.masked_fill(~valid, -float(clip)).detach()


def stability_gain_utility(
    q_self: torch.Tensor,
    q_edge: torch.Tensor,
    rec_self: torch.Tensor,
    rec_edge: torch.Tensor,
    *,
    q_self_second: torch.Tensor,
    q_edge_second: torch.Tensor,
    rec_self_second: torch.Tensor,
    rec_edge_second: torch.Tensor,
    lambda_rec: float,
    clip: float,
    valid: torch.Tensor,
    confidence_weight: float = 0.0,
    relative_baseline: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate edge utility from an independent augmentation response.

    The target is permutation-invariant and does not require a teacher cluster
    identity: an edge is useful when it reduces assignment instability across
    two masked views, improves the assignment margin, and does not damage sparse
    reconstruction. All returned tensors are detached from the gate/student.
    """
    if q_self.ndim != 2 or q_edge.ndim != 3 or q_self_second.shape != q_self.shape:
        raise ValueError("q_self/q_edge views have incompatible shapes")
    if q_edge_second.shape != q_edge.shape:
        raise ValueError("q_edge views have incompatible shapes")
    batch, candidates, clusters = q_edge.shape
    if rec_self.shape != (batch,) or rec_self_second.shape != (batch,):
        raise ValueError("reconstruction self terms must be [B]")
    if rec_edge.shape != (batch, candidates) or rec_edge_second.shape != rec_edge.shape:
        raise ValueError("reconstruction edge terms must be [B,M]")
    if valid.shape != (batch, candidates):
        raise ValueError("valid must be [B,M]")
    self_instability = jsd_per_sample(q_self, q_self_second)
    edge_instability = jsd_per_sample(
        q_edge.reshape(-1, clusters), q_edge_second.reshape(-1, clusters)
    ).reshape(batch, candidates)
    semantic = self_instability[:, None] - edge_instability
    if confidence_weight > 0.0:
        self_margin = 0.5 * (assignment_margin(q_self) + assignment_margin(q_self_second))
        edge_margin = 0.5 * (
            assignment_margin(q_edge.reshape(-1, clusters)).reshape(batch, candidates)
            + assignment_margin(q_edge_second.reshape(-1, clusters)).reshape(batch, candidates)
        )
        semantic = semantic + float(confidence_weight) * (edge_margin - self_margin[:, None])
    damage = 0.5 * (
        rec_edge - rec_self[:, None] + rec_edge_second - rec_self_second[:, None]
    )
    if relative_baseline:
        # A donor can improve consistency merely by averaging representations.
        # Subtract the non-negative median candidate response so the gate learns
        # edge-specific benefit above that generic smoothing background.
        invalid_fill = torch.full_like(semantic, float("inf"))
        semantic_median = torch.median(torch.where(valid, semantic, invalid_fill), dim=1).values
        damage_median = torch.median(torch.where(valid, damage, invalid_fill), dim=1).values
        semantic = semantic - semantic_median.clamp_min(0.0)[:, None]
        damage = damage - damage_median.clamp_min(0.0)[:, None]
    semantic_scale = _zero_preserving_robust_scale(semantic, valid)
    damage_scale = _zero_preserving_robust_scale(damage, valid)
    utility = semantic / semantic_scale - float(lambda_rec) * damage / damage_scale
    utility = utility.clamp(-float(clip), float(clip)).masked_fill(~valid, -float(clip)).detach()
    return utility, semantic.detach(), damage.detach()


def operator_aligned_utility(
    q_reference: torch.Tensor,
    q_self_first: torch.Tensor,
    q_edge_first: torch.Tensor,
    q_self_second: torch.Tensor,
    q_edge_second: torch.Tensor,
    rec_self: torch.Tensor,
    rec_edge: torch.Tensor,
    *,
    lambda_rec: float,
    stability_weight: float,
    relative_baseline: bool,
    reference_mode: str,
    reference_temperature: float,
    min_gain: float = 0.0,
    clip: float,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score the exact counterfactual operator used by the final readout.

    The clean topology-disabled teacher is a detached reference.  Each
    treatment is compared to the *same masked anchor view* used in the final
    transport.  Averaging two deterministic masked views makes utility an
    expected denoising benefit rather than a lucky single-mask response.
    """
    if q_reference.ndim != 2 or q_self_first.ndim != 2 or q_edge_first.ndim != 3:
        raise ValueError("operator-aligned assignments have invalid ranks")
    if q_self_second.shape != q_self_first.shape or q_edge_second.shape != q_edge_first.shape:
        raise ValueError("operator-aligned views must have matching shapes")
    batch, candidates, clusters = q_edge_first.shape
    if q_reference.shape != (batch, clusters) or q_self_first.shape != (batch, clusters):
        raise ValueError("reference/self assignments do not match edge assignments")
    if rec_self.shape != (batch,) or rec_edge.shape != (batch, candidates):
        raise ValueError("reconstruction terms do not match operator assignments")
    if valid.shape != (batch, candidates):
        raise ValueError("valid must match candidate assignments")
    q_ref_edge = q_reference[:, None, :].expand(-1, candidates, -1)
    teacher_first = kl_per_sample(q_reference, q_self_first)[:, None] - kl_per_sample(
        q_ref_edge.reshape(-1, clusters), q_edge_first.reshape(-1, clusters)
    ).reshape(batch, candidates)
    teacher_second = kl_per_sample(q_reference, q_self_second)[:, None] - kl_per_sample(
        q_ref_edge.reshape(-1, clusters), q_edge_second.reshape(-1, clusters)
    ).reshape(batch, candidates)
    cross_first_reference = q_self_second[:, None, :].expand(-1, candidates, -1)
    cross_second_reference = q_self_first[:, None, :].expand(-1, candidates, -1)
    cross_first = kl_per_sample(q_self_second, q_self_first)[:, None] - kl_per_sample(
        cross_first_reference.reshape(-1, clusters), q_edge_first.reshape(-1, clusters)
    ).reshape(batch, candidates)
    cross_second = kl_per_sample(q_self_first, q_self_second)[:, None] - kl_per_sample(
        cross_second_reference.reshape(-1, clusters), q_edge_second.reshape(-1, clusters)
    ).reshape(batch, candidates)
    if reference_mode == "teacher":
        first, second = teacher_first, teacher_second
    elif reference_mode == "cross_view":
        first, second = cross_first, cross_second
    elif reference_mode == "hybrid":
        first = 0.5 * (teacher_first + cross_first)
        second = 0.5 * (teacher_second + cross_second)
    elif reference_mode == "adaptive":
        self_consensus = 0.5 * (q_self_first + q_self_second)
        teacher_advantage = assignment_margin(q_reference) - assignment_margin(self_consensus)
        teacher_trust = torch.sigmoid(
            teacher_advantage / max(float(reference_temperature), 1e-4)
        )[:, None]
        first = teacher_trust * teacher_first + (1.0 - teacher_trust) * cross_first
        second = teacher_trust * teacher_second + (1.0 - teacher_trust) * cross_second
    else:
        raise ValueError(f"unknown operator utility reference mode: {reference_mode}")
    # A noisy edge should not be accepted because it happened to help one
    # particular feature mask.  Use a two-view lower-confidence bound: mean
    # counterfactual benefit minus disagreement between the two interventions.
    semantic = 0.5 * (first + second) - float(stability_weight) * 0.5 * torch.abs(first - second)
    damage = rec_edge - rec_self[:, None]
    if relative_baseline:
        # Generic neighbour averaging often improves every masked anchor a
        # little.  It is not edge-specific evidence.  Subtract only a positive
        # per-anchor median, preserving zero as the abstention boundary and
        # never manufacturing positive utility when all candidates are bad.
        invalid_fill = torch.full_like(semantic, float("inf"))
        semantic_median = torch.median(torch.where(valid, semantic, invalid_fill), dim=1).values
        damage_median = torch.median(torch.where(valid, damage, invalid_fill), dim=1).values
        semantic = semantic - semantic_median.clamp_min(0.0)[:, None]
        damage = damage - damage_median.clamp_min(0.0)[:, None]
    semantic_scale = _zero_preserving_robust_scale(semantic, valid)
    damage_scale = _zero_preserving_robust_scale(damage, valid)
    utility = (
        semantic / semantic_scale
        - float(lambda_rec) * damage / damage_scale
        - float(min_gain)
    )
    utility = utility.clamp(-float(clip), float(clip)).masked_fill(~valid, -float(clip))
    return utility.detach(), semantic.detach(), damage.detach()


def local_consensus_utility(
    q_self_first: torch.Tensor,
    q_edge_first: torch.Tensor,
    q_self_second: torch.Tensor,
    q_edge_second: torch.Tensor,
    q_donor: torch.Tensor,
    rec_self: torch.Tensor,
    rec_edge: torch.Tensor,
    *,
    lambda_rec: float,
    stability_weight: float,
    confidence_weight: float,
    relative_baseline: bool,
    min_gain: float = 0.0,
    clip: float,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate edge utility against a leave-one-candidate-out consensus.

    The tested donor is removed from the local reference before its effect is
    scored.  This makes the target an exogenous graph-consensus signal rather
    than a teacher target that can be improved by copying the same donor.  A
    useful edge reduces the assignment divergence to the remaining candidate
    population in two independent masked probes and increases assignment
    margin without damaging sparse reconstruction.
    """
    if q_self_first.ndim != 2 or q_edge_first.ndim != 3:
        raise ValueError("local consensus assignments have invalid ranks")
    if q_self_second.shape != q_self_first.shape or q_edge_second.shape != q_edge_first.shape:
        raise ValueError("local consensus probe views must have matching shapes")
    if q_donor.shape != q_edge_first.shape:
        raise ValueError("q_donor must match q_edge shape")
    batch, candidates, clusters = q_edge_first.shape
    if rec_self.shape != (batch,) or rec_edge.shape != (batch, candidates):
        raise ValueError("reconstruction terms do not match local consensus candidates")
    if valid.shape != (batch, candidates):
        raise ValueError("valid must match local consensus candidates")

    donor = q_donor.clamp_min(1e-8)
    donor = donor / donor.sum(dim=2, keepdim=True).clamp_min(1e-8)
    valid_float = valid.to(dtype=donor.dtype)
    counts = valid_float.sum(dim=1, keepdim=True)
    denominator = (counts - 1.0).clamp_min(1.0)
    donor_sum = (donor * valid_float[:, :, None]).sum(dim=1, keepdim=True)
    # When a row has only one candidate, the candidate cannot be judged against
    # another edge; use the masked self assignment and let its utility collapse
    # to the null boundary after the zero-preserving normalization.
    peer = (donor_sum - donor) / denominator[:, :, None]
    singleton = counts <= 1.0
    peer = torch.where(singleton[:, :, None], q_self_first[:, None, :], peer)
    peer = peer.clamp_min(1e-8)
    peer = peer / peer.sum(dim=2, keepdim=True).clamp_min(1e-8)

    self_first_expanded = q_self_first[:, None, :].expand(-1, candidates, -1)
    self_second_expanded = q_self_second[:, None, :].expand(-1, candidates, -1)
    self_peer_first = jsd_per_sample(
        self_first_expanded.reshape(-1, clusters), peer.reshape(-1, clusters)
    ).reshape(batch, candidates)
    self_peer_second = jsd_per_sample(
        self_second_expanded.reshape(-1, clusters), peer.reshape(-1, clusters)
    ).reshape(batch, candidates)
    edge_peer_first = jsd_per_sample(
        q_edge_first.reshape(-1, clusters), peer.reshape(-1, clusters)
    ).reshape(batch, candidates)
    edge_peer_second = jsd_per_sample(
        q_edge_second.reshape(-1, clusters), peer.reshape(-1, clusters)
    ).reshape(batch, candidates)
    first = self_peer_first - edge_peer_first
    second = self_peer_second - edge_peer_second
    semantic = 0.5 * (first + second)
    if stability_weight > 0.0:
        semantic = semantic - float(stability_weight) * 0.5 * torch.abs(first - second)
    if confidence_weight > 0.0:
        self_margin = 0.5 * (
            assignment_margin(q_self_first) + assignment_margin(q_self_second)
        )
        edge_margin = 0.5 * (
            assignment_margin(q_edge_first.reshape(-1, clusters)).reshape(batch, candidates)
            + assignment_margin(q_edge_second.reshape(-1, clusters)).reshape(batch, candidates)
        )
        semantic = semantic + float(confidence_weight) * (edge_margin - self_margin[:, None])

    # ``rec_edge`` is already the mean over the two probe views at the call
    # site.  Keep the zero boundary and scale identical to operator-aligned
    # utility; introducing another 0.5 here would silently halve the
    # reconstruction penalty.
    damage = rec_edge - rec_self[:, None]
    if relative_baseline:
        invalid_fill = torch.full_like(semantic, float("inf"))
        semantic_median = torch.median(torch.where(valid, semantic, invalid_fill), dim=1).values
        damage_median = torch.median(torch.where(valid, damage, invalid_fill), dim=1).values
        semantic = semantic - semantic_median.clamp_min(0.0)[:, None]
        damage = damage - damage_median.clamp_min(0.0)[:, None]
    semantic_scale = _zero_preserving_robust_scale(semantic, valid)
    damage_scale = _zero_preserving_robust_scale(damage, valid)
    utility = (
        semantic / semantic_scale
        - float(lambda_rec) * damage / damage_scale
        - float(min_gain)
    )
    utility = utility.clamp(-float(clip), float(clip)).masked_fill(~valid, -float(clip))
    return utility.detach(), semantic.detach(), damage.detach()


def clean_output_utility(
    q_reference: torch.Tensor,
    q_self: torch.Tensor,
    q_edge: torch.Tensor,
    *,
    min_gain: float,
    clip: float,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Score the exact clean assignment intervention used by the readout."""
    if q_reference.shape != q_self.shape or q_edge.ndim != 3:
        raise ValueError("clean output assignments have incompatible shapes")
    if q_edge.shape[0] != q_self.shape[0] or q_edge.shape[2] != q_self.shape[1]:
        raise ValueError("clean edge assignments do not match self/reference")
    if valid.shape != q_edge.shape[:2]:
        raise ValueError("valid must match clean edge assignments")
    batch, candidates, clusters = q_edge.shape
    reference_edges = q_reference[:, None, :].expand(-1, candidates, -1)
    semantic = kl_per_sample(q_reference, q_self)[:, None] - kl_per_sample(
        reference_edges.reshape(-1, clusters),
        q_edge.reshape(-1, clusters),
    ).reshape(batch, candidates)
    scale = _zero_preserving_robust_scale(semantic, valid)
    utility = semantic / scale - float(min_gain)
    return utility.clamp(-float(clip), float(clip)).masked_fill(~valid, -float(clip)).detach()


class V15Trainer:
    """Independent V15 trainer. Labels are intentionally absent from this class."""

    def __init__(
        self,
        data: PreparedInput,
        n_clusters: int,
        config: V15Config,
        device: torch.device,
    ):
        if n_clusters <= 1 or n_clusters > data.n_samples:
            raise ValueError("n_clusters must be between 2 and n_samples")
        self.data = data
        self.n_clusters = int(n_clusters)
        self.config = config
        self.device = device
        self.model = V15Model(
            input_dim=data.n_features,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            n_clusters=n_clusters,
            dropout=config.dropout,
            student_t_nu=config.student_t_nu,
            cluster_normalize_latent=config.cluster_normalize_latent,
            cluster_cosine_temperature=config.cluster_cosine_temperature,
            cluster_head=config.cluster_head,
            prototype_separation_weight=config.prototype_separation_weight,
            prototype_separation_margin=config.prototype_separation_margin,
        ).to(device)
        self.teacher = make_teacher(self.model).to(device)
        self.cluster_frequency_ema = torch.full(
            (self.n_clusters,),
            1.0 / self.n_clusters,
            dtype=torch.float32,
            device=self.device,
        )
        self.graph: CandidateGraph | None = None
        self.raw_view_probabilities: torch.Tensor | None = None
        self.teacher_selection: dict[str, object] = {
            "mode": config.teacher_reference_mode,
            "selected_view": "latent",
            "raw_quality": None,
            "latent_quality": None,
        }
        # ``quality_auto`` freezes the view choice after teacher calibration;
        # keeping the selected masked reference fixed avoids a moving target
        # whose apparent quality could oscillate with the EMA model.
        self._quality_augmented_reference: torch.Tensor | None = None
        self._finalizing = False
        self._selection_allowed = False
        self._backbone_frozen = False
        split_rng = np.random.default_rng(config.seed + 1501)
        order = split_rng.permutation(data.n_samples)
        holdout_count = min(data.n_samples - 1, max(1, int(round(data.n_samples * config.utility_holdout_fraction))))
        self.utility_train_mask = np.ones(data.n_samples, dtype=bool)
        self.utility_train_mask[order[:holdout_count]] = False

    @torch.no_grad()
    def full_embeddings(self, model: V15Model | None = None) -> np.ndarray:
        active = self.model if model is None else model
        active.eval()
        output: list[np.ndarray] = []
        for start in range(0, self.data.n_samples, self.config.batch_size):
            indices = np.arange(start, min(start + self.config.batch_size, self.data.n_samples), dtype=np.int64)
            x = self.data.get(indices, self.device)
            output.append(active.encode(x).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(output, axis=0)

    @torch.no_grad()
    def full_probabilities(self, model: V15Model | None = None) -> np.ndarray:
        active = self.model if model is None else model
        active.eval()
        output: list[np.ndarray] = []
        for start in range(0, self.data.n_samples, self.config.batch_size):
            indices = np.arange(start, min(start + self.config.batch_size, self.data.n_samples), dtype=np.int64)
            x = self.data.get(indices, self.device)
            output.append(active.assignments(active.encode(x)).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(output, axis=0)

    @torch.no_grad()
    def full_probabilities_masked(self, model: V15Model, seed: int) -> np.ndarray:
        model.eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        output: list[np.ndarray] = []
        for start in range(0, self.data.n_samples, self.config.batch_size):
            indices = np.arange(start, min(start + self.config.batch_size, self.data.n_samples), dtype=np.int64)
            x = self.data.get(indices, self.device)
            corrupted, _ = apply_mask(
                x,
                self.config.mask_ratio,
                generator=generator,
                strategy=self.config.mask_strategy,
            )
            output.append(model.assignments(model.encode(corrupted)).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(output, axis=0)

    def _build_graph(self, latent_embedding: np.ndarray) -> CandidateGraph:
        graph = build_candidate_graph(
            self.data,
            k_raw=self.config.k_raw,
            k_latent=self.config.k_latent,
            candidate_cap=self.config.candidate_cap,
            raw_svd_dim=self.config.raw_svd_dim,
            latent_embedding=latent_embedding,
            latent_graph_dim=self.config.latent_graph_dim,
            seed=self.config.seed,
        )
        graph = replace_candidate_edges(graph, self.config.graph_replacement_fraction, self.config.seed)
        return restrict_candidate_scope(graph, self.config.candidate_scope)

    def _build_raw_view_teacher(self, raw_embedding: np.ndarray) -> torch.Tensor:
        """Create a label-free sparse-view assignment target.

        This target is deliberately derived from the raw sparse SVD view, not
        from the current gate or EMA latent. It prevents a geometrically stable
        but semantically wrong latent teacher from becoming self-confirming.
        """
        values = np.asarray(raw_embedding, dtype=np.float32)
        values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.config.n_init,
            random_state=self.config.seed,
        )
        km.fit(values)
        centres = np.asarray(km.cluster_centers_, dtype=np.float32)
        centres /= np.clip(np.linalg.norm(centres, axis=1, keepdims=True), 1e-8, None)
        logits = values @ centres.T
        probabilities = torch.softmax(
            torch.as_tensor(logits, dtype=torch.float32, device=self.device)
            / self.config.raw_view_cluster_temperature,
            dim=1,
        )
        return probabilities.detach()

    @torch.no_grad()
    def _build_teacher_reference(
        self,
        latent_probabilities: torch.Tensor,
        augmented_probabilities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.raw_view_probabilities is not None
        if self.config.teacher_reference_mode in {"auto", "quality_auto"}:
            assert self.graph is not None
            raw = self.raw_view_probabilities
            raw_aligned = align_teacher_assignments(latent_probabilities, raw)
            if self.config.teacher_reference_mode == "quality_auto":
                # During fitting, quality selection must not become a moving
                # teacher target. The final detached readout performs the
                # selection once on the mature teacher; training uses the
                # stable consensus reference below.
                if not self._finalizing:
                    return build_teacher_reference(
                        latent_probabilities,
                        self.raw_view_probabilities,
                        mode="consensus",
                        raw_weight=self.config.teacher_reference_raw_weight,
                        temperature=self.config.teacher_reference_temperature,
                    )
                if augmented_probabilities is None:
                    augmented_probabilities = self._quality_augmented_reference
                if self.teacher_selection.get("raw_quality") is None:
                    if not self._selection_allowed:
                        disagreement = jsd_per_sample(latent_probabilities, raw_aligned)
                        return (
                            latent_probabilities.detach(),
                            torch.ones_like(disagreement),
                            disagreement.detach(),
                        )
                    # Use the active candidate scope for all three views. This
                    # makes teacher selection answer the same local question
                    # that the gate will answer, without labels or a second
                    # reliability head.
                    neighbors = self.graph.indices
                    latent_quality = view_local_assignment_quality(
                        latent_probabilities, neighbors
                    )
                    raw_quality = view_local_assignment_quality(raw_aligned, neighbors)
                    qualities: dict[str, dict[str, float]] = {
                        "latent": latent_quality,
                        "raw": raw_quality,
                    }
                    if augmented_probabilities is not None:
                        qualities["augmented"] = view_local_assignment_quality(
                            augmented_probabilities, neighbors
                        )
                    selected_view = max(
                        qualities,
                        key=lambda name: qualities[name]["score"],
                    )
                    self.teacher_selection = {
                        "mode": "quality_auto",
                        "selected_view": selected_view,
                        "raw_quality": raw_quality,
                        "latent_quality": latent_quality,
                        "augmented_quality": qualities.get("augmented"),
                    }
                    if selected_view == "augmented" and augmented_probabilities is not None:
                        self._quality_augmented_reference = augmented_probabilities.detach()
                selected_view = str(self.teacher_selection["selected_view"])
                if selected_view == "raw":
                    reference = raw_aligned
                elif selected_view == "augmented":
                    reference = self._quality_augmented_reference
                    if reference is None:
                        reference = augmented_probabilities
                    if reference is None:
                        reference = latent_probabilities
                else:
                    reference = latent_probabilities
                disagreement = jsd_per_sample(latent_probabilities, reference)
                agreement = torch.ones_like(disagreement)
                return reference.detach(), agreement.detach(), disagreement.detach()
            # Freeze the view choice after the first graph/teacher snapshot so
            # cluster semantics cannot oscillate as the EMA representation
            # changes during gate training.
            if self.teacher_selection.get("raw_quality") is None:
                if not self._selection_allowed:
                    # During topology-disabled calibration the latent teacher
                    # is only a temporary target. Do not record a selection
                    # until its assignment geometry has matured.
                    disagreement = jsd_per_sample(latent_probabilities, raw_aligned)
                    return latent_probabilities.detach(), torch.ones_like(disagreement), disagreement.detach()
                raw_quality = view_local_assignment_quality(raw, self.graph.raw_indices)
                latent_quality = view_local_assignment_quality(latent_probabilities, self.graph.latent_indices)
                selected_view = "raw" if raw_quality["score"] > latent_quality["score"] else "latent"
                self.teacher_selection = {
                    "mode": "auto",
                    "selected_view": selected_view,
                    "raw_quality": raw_quality,
                    "latent_quality": latent_quality,
                }
            selected_view = str(self.teacher_selection["selected_view"])
            reference = raw_aligned if selected_view == "raw" else latent_probabilities
            disagreement = jsd_per_sample(latent_probabilities, raw_aligned)
            # The selected view is the semantic target. Disagreement is saved
            # as a certificate, but it must not downweight every target row.
            agreement = torch.ones_like(disagreement)
            return reference.detach(), agreement.detach(), disagreement.detach()
        return build_teacher_reference(
            latent_probabilities,
            self.raw_view_probabilities,
            mode=self.config.teacher_reference_mode,
            raw_weight=self.config.teacher_reference_raw_weight,
            temperature=self.config.teacher_reference_temperature,
        )

    def _predict_gate_distribution(
        self,
        edge_scores: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the registered gate/control semantics in one place."""
        if self.config.gate_mode in {"self_only", "output_disabled"}:
            predicted_pi = torch.zeros(
                (edge_scores.shape[0], edge_scores.shape[1] + 1),
                dtype=edge_scores.dtype,
                device=edge_scores.device,
            )
            predicted_pi[:, 0] = 1.0
            return predicted_pi
        if self.config.gate_mode == "union_uniform":
            return _uniform_edge_distribution(valid).to(dtype=edge_scores.dtype)
        if self.config.gate_mode == "forced_topk":
            return _forced_topk_distribution(edge_scores, valid, self.config.forced_topk)
        if self.config.gate_mode == "shuffled_utility":
            return abstaining_sparsemax(_shuffle_scores_within_valid(edge_scores, valid), valid)
        return abstaining_sparsemax(edge_scores, valid)

    def _apply_gate_opportunity(
        self,
        predicted_pi: torch.Tensor,
        q_self: torch.Tensor,
    ) -> torch.Tensor:
        """Limit topology transport to anchors that have assignment opportunity."""
        if self.config.gate_opportunity_mode == "none" or predicted_pi.shape[1] <= 1:
            return predicted_pi
        entropy = -(q_self.clamp_min(1e-8) * torch.log(q_self.clamp_min(1e-8))).sum(dim=1)
        opportunity = entropy / np.log(max(2, q_self.shape[1]))
        opportunity = opportunity.clamp(0.0, 1.0)[:, None]
        edge_mass = predicted_pi[:, 1:] * opportunity
        null_mass = 1.0 - edge_mass.sum(dim=1, keepdim=True)
        return torch.cat([null_mass.clamp_min(0.0), edge_mass], dim=1)

    def _output_alpha_for_batch(
        self,
        features: torch.Tensor,
        edge_mass: torch.Tensor,
    ) -> float | torch.Tensor:
        if not self.config.output_consensus_scaling:
            return self.config.output_alpha
        # Feature slot 2 is raw-only=-1, both=0, latent-only=+1. Supported by
        # both views can tolerate stronger transport; single-view edges use a
        # half-strength residual without introducing a forced neighbor.
        consensus = (features[:, :, 2].abs() < 0.5).to(dtype=edge_mass.dtype)
        total = edge_mass.sum(dim=1)
        consensus_mass = (edge_mass * consensus).sum(dim=1) / total.clamp_min(1e-8)
        factor = 0.5 + 0.5 * consensus_mass
        return float(self.config.output_alpha) * factor

    @torch.no_grad()
    def _full_gated_outputs(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Evaluate the same candidate/scorer/output path used during training."""
        assert self.graph is not None
        self.model.eval()
        self.teacher.eval()
        self._finalizing = True
        teacher_z = torch.as_tensor(self.full_embeddings(self.teacher), dtype=torch.float32, device=self.device)
        q_teacher_latent_full = self.teacher.assignments(teacher_z)
        augmented_reference = self._quality_augmented_reference
        if self.config.teacher_reference_mode == "quality_auto" and augmented_reference is None:
            augmented_reference = torch.as_tensor(
                self.full_probabilities_masked(self.teacher, self.config.seed + 2501),
                dtype=torch.float32,
                device=self.device,
            )
        q_teacher_full, reference_agreement_full, _ = self._build_teacher_reference(
            q_teacher_latent_full,
            augmented_reference,
        )
        probe_generator = torch.Generator(device=self.device)
        probe_generator.manual_seed(int(self.config.seed) + 4501)
        embedding_parts: list[np.ndarray] = []
        probability_parts: list[np.ndarray] = []
        pi_parts: list[np.ndarray] = []
        utility_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        gate_valid_parts: list[np.ndarray] = []
        self_prediction_parts: list[np.ndarray] = []
        edge_prediction_parts: list[np.ndarray] = []
        self_embedding_parts: list[np.ndarray] = []
        self_assignment_parts: list[np.ndarray] = []
        edge_assignment_parts: list[np.ndarray] = []
        edge_embedding_parts: list[np.ndarray] = []
        transport_embedding_parts: list[np.ndarray] = []
        for start in range(0, self.data.n_samples, self.config.batch_size):
            indices = np.arange(start, min(start + self.config.batch_size, self.data.n_samples), dtype=np.int64)
            x = self.data.get(indices, self.device)
            z_self = self.model.encode(x)
            probe_views: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for _ in range(2 * self.config.utility_probe_pairs):
                probe_corrupted, probe_mask = apply_mask(
                    x,
                    self.config.mask_ratio,
                    generator=probe_generator,
                    strategy=self.config.mask_strategy,
                )
                probe_zero_mask = (x == 0.0) & (
                    torch.rand_like(x) < self.config.zero_sample_ratio
                )
                probe_views.append((probe_corrupted, probe_mask, probe_zero_mask))
            (
                _utility_target,
                donors,
                valid,
                teacher_assignment_agreement,
                semantic_help,
                reconstruction_damage,
                q_probe_self,
                q_probe_edge,
            ) = self._target_for_probe_views(
                indices,
                probe_views,
                q_teacher_latent_full,
                teacher_z,
                q_teacher_full,
                reference_agreement_full,
            )
            q_self = self.model.assignments(z_self)
            q_donor_teacher = self.teacher.assignments(donors.reshape(-1, donors.shape[-1])).reshape(
                donors.shape[0], donors.shape[1], -1
            )
            z_edge_clean = z_self[:, None, :] + self.config.probe_alpha * (
                donors.detach() - z_self[:, None, :]
            )
            q_edge_clean = self.model.assignments(
                z_edge_clean.reshape(-1, z_edge_clean.shape[-1])
            ).reshape(z_edge_clean.shape[0], z_edge_clean.shape[1], -1)
            q_edge_clean = (
                (1.0 - self.config.output_alpha) * q_self[:, None, :]
                + self.config.output_alpha * q_edge_clean
            ).clamp_min(1e-8)
            q_edge_clean = q_edge_clean / q_edge_clean.sum(dim=2, keepdim=True)
            features = _assemble_utility_features(
                self.graph.features[indices],
                teacher_assignment_agreement,
                semantic_help,
                reconstruction_damage,
                valid,
                self.device,
            )
            direct_modes = {
                "direct_target",
                "direct_counterfactual",
                "counterfactual_learned",
                "self_only",
                "union_uniform",
                "forced_topk",
                "shuffled_utility",
                "output_disabled",
            }
            use_direct_utility = (
                self.config.utility_target_mode in {"operator_aligned", "local_consensus"}
                and self.config.gate_mode in direct_modes
            )
            use_amortized_utility = self.config.gate_mode == "counterfactual_learned"
            if (
                use_direct_utility
                and not use_amortized_utility
                and self.config.utility_target_mode == "operator_aligned"
                and self.config.direct_utility_source == "clean_output"
            ):
                q_reference = q_teacher_full[
                    torch.as_tensor(indices, dtype=torch.long, device=self.device)
                ]
                utility_hat = clean_output_utility(
                    q_reference,
                    q_self,
                    q_edge_clean,
                    min_gain=self.config.utility_min_gain,
                    clip=self.config.utility_clip,
                    valid=valid,
                )
            elif use_amortized_utility:
                utility_hat = self.model.utility(features)
            else:
                utility_hat = _utility_target if use_direct_utility else self.model.utility(features)
            edge_scores = utility_hat / max(self.config.gate_temperature, 1e-4)
            gate_valid = valid & (
                teacher_assignment_agreement >= float(self.config.gate_teacher_agreement_floor)
            )
            predicted_pi = self._predict_gate_distribution(edge_scores, gate_valid)
            # Counterfactual masks estimate whether an edge is useful; they are
            # not the final null branch.  The exported operator must start from
            # the clean student so exact abstention is genuinely identical to
            # the topology-disabled output.
            q_output_self = q_self
            predicted_pi = self._apply_gate_opportunity(predicted_pi, q_output_self)
            edge_mass = predicted_pi[:, 1:]
            output_alpha = self._output_alpha_for_batch(features, edge_mass)
            z_transport = self.model.mix_latent(
                z_self,
                donors,
                edge_mass,
                output_alpha,
            )
            if self.config.output_mode == "logit":
                self_logits = self.model.assignment_logits(z_self)
                donor_logits = self.model.assignment_logits(donors.reshape(-1, donors.shape[-1])).reshape(
                    donors.shape[0], donors.shape[1], -1
                )
                q_edge = F.softmax(
                    self_logits[:, None, :] + self.config.output_alpha * (donor_logits - self_logits[:, None, :]),
                    dim=2,
                )
                q_out = self.model.mix_assignments(
                    z_self,
                    donors,
                    edge_mass,
                    output_alpha,
                )
                # Logit transport is retained as an assignment-only control.
                # The exported embedding remains the self branch in this
                # explicit mode, while z_transport is persisted for audit.
                z_out = z_self
            elif self.config.output_mode == "probability":
                q_edge = (
                    (1.0 - self.config.output_alpha) * q_self[:, None, :]
                    + self.config.output_alpha * q_donor_teacher
                ).clamp_min(1e-8)
                q_edge = q_edge / q_edge.sum(dim=2, keepdim=True)
                q_out = self.model.mix_probabilities(
                    q_self,
                    q_donor_teacher,
                    edge_mass,
                    output_alpha,
                )
                z_out = z_self
            elif self.config.output_mode == "assignment":
                q_edge = q_edge_clean
                q_out = self.model.mix_assignment_output(q_self, q_edge, predicted_pi)
                if isinstance(output_alpha, torch.Tensor):
                    alpha_view = output_alpha.to(device=z_self.device, dtype=z_self.dtype).reshape(-1, 1, 1)
                else:
                    alpha_view = float(output_alpha)
                z_edge_assignment = (1.0 - alpha_view) * z_self[:, None, :] + alpha_view * z_edge_clean
                z_transport = self.model.mix_assignment_embedding(
                    z_self,
                    z_edge_assignment,
                    predicted_pi,
                )
                z_out = z_transport
            else:
                z_edge = z_self[:, None, :] + self.config.output_alpha * (donors - z_self[:, None, :])
                q_edge = self.model.assignments(z_edge.reshape(-1, z_edge.shape[-1])).reshape(
                    z_edge.shape[0], z_edge.shape[1], -1
                )
                z_out = z_transport
                q_out = self.model.assignments(z_out)
            embedding_parts.append(z_out.detach().cpu().numpy().astype(np.float32))
            self_embedding_parts.append(z_self.detach().cpu().numpy().astype(np.float32))
            transport_embedding_parts.append(z_transport.detach().cpu().numpy().astype(np.float32))
            probability_parts.append(q_out.detach().cpu().numpy().astype(np.float32))
            pi_parts.append(predicted_pi.detach().cpu().numpy().astype(np.float32))
            utility_parts.append(utility_hat.detach().cpu().numpy().astype(np.float32))
            feature_parts.append(features.detach().cpu().numpy().astype(np.float32))
            gate_valid_parts.append(gate_valid.detach().cpu().numpy().astype(bool))
            self_prediction_parts.append(q_output_self.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            edge_prediction_parts.append(q_edge.argmax(dim=2).detach().cpu().numpy().astype(np.int64))
            self_assignment_parts.append(q_output_self.detach().cpu().numpy().astype(np.float32))
            edge_assignment_parts.append(q_edge.detach().cpu().numpy().astype(np.float32))
            edge_embedding_parts.append(z_edge_clean.detach().cpu().numpy().astype(np.float32))
        return (
            np.concatenate(embedding_parts, axis=0),
            np.concatenate(probability_parts, axis=0),
            np.concatenate(pi_parts, axis=0),
            np.concatenate(utility_parts, axis=0),
            np.concatenate(feature_parts, axis=0),
            np.concatenate(gate_valid_parts, axis=0),
            np.concatenate(self_prediction_parts, axis=0),
            np.concatenate(edge_prediction_parts, axis=0),
            np.concatenate(self_embedding_parts, axis=0),
            np.concatenate(self_assignment_parts, axis=0),
            np.concatenate(edge_assignment_parts, axis=0),
            np.concatenate(edge_embedding_parts, axis=0),
            np.concatenate(transport_embedding_parts, axis=0),
        )

    def _target_for_batch(
        self,
        batch_indices: np.ndarray,
        eval_corrupted: torch.Tensor,
        eval_mask: torch.Tensor,
        eval_zero_mask: torch.Tensor,
        q_teacher_full: torch.Tensor,
        teacher_z_full: torch.Tensor,
        q_reference_full: torch.Tensor,
        reference_agreement_full: torch.Tensor,
        eval_corrupted_second: torch.Tensor | None = None,
        eval_mask_second: torch.Tensor | None = None,
        eval_zero_mask_second: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert self.graph is not None
        graph_indices = self.graph.indices[batch_indices]
        valid_np = self.graph.valid[batch_indices]
        valid = torch.as_tensor(valid_np, dtype=torch.bool, device=self.device)
        fallback = np.repeat(batch_indices[:, None], graph_indices.shape[1], axis=1)
        safe_indices = np.where(valid_np, graph_indices, fallback)
        donors = teacher_z_full[torch.as_tensor(safe_indices, dtype=torch.long, device=self.device)]
        with torch.no_grad():
            batch_tensor = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
            q_teacher_latent = q_teacher_full[batch_tensor]
            q_teacher = q_reference_full[batch_tensor]
            reference_agreement = reference_agreement_full[batch_tensor]
            x_target = self.data.get(batch_indices, self.device)
            z_self_eval = self.teacher.encode(eval_corrupted)
            q_self_teacher = self.teacher.assignments(z_self_eval)
            q_donor = self.teacher.assignments(donors.reshape(-1, donors.shape[-1])).reshape(
                donors.shape[0], donors.shape[1], -1
            )
            # Use permutation-invariant view-specific assignment affinity for
            # edge validity. Raw-only edges are judged in the raw view,
            # latent-only edges in the latent view, and consensus edges use
            # both. This avoids aligning an actually wrong teacher to another
            # wrong teacher merely because their component ids agree.
            assert self.raw_view_probabilities is not None
            raw_anchor = self.raw_view_probabilities[batch_tensor]
            raw_donor = self.raw_view_probabilities[
                torch.as_tensor(safe_indices, dtype=torch.long, device=self.device)
            ]
            raw_affinity = (raw_anchor[:, None, :] * raw_donor).sum(dim=2)
            latent_affinity = (q_teacher_latent[:, None, :] * q_donor).sum(dim=2)
            source = torch.as_tensor(
                self.graph.features[batch_indices, :, 2], dtype=torch.float32, device=self.device
            )
            teacher_assignment_agreement = torch.where(
                source < -0.5,
                raw_affinity,
                torch.where(source > 0.5, latent_affinity, 0.5 * (raw_affinity + latent_affinity)),
            )
            z_probe = z_self_eval[:, None, :] + self.config.probe_alpha * (donors - z_self_eval[:, None, :])
            q_edge = self.teacher.assignments(z_probe.reshape(-1, z_probe.shape[-1])).reshape(
                z_probe.shape[0], z_probe.shape[1], -1
            )
            # This is the masked-probe assignment intervention used by the
            # masked-probe and local-consensus targets.  The clean-output
            # operator-aligned target is built explicitly below so its target
            # and final readout retain the same operator semantics.
            q_edge_aligned = (
                (1.0 - self.config.output_alpha) * q_self_teacher[:, None, :]
                + self.config.output_alpha * q_edge
            ).clamp_min(1e-8)
            q_edge_aligned = q_edge_aligned / q_edge_aligned.sum(dim=2, keepdim=True)
            batch, candidates, clusters = q_edge_aligned.shape
            rec_self = sparse_reconstruction_per_sample(
                self.teacher.autoencoder.decode(z_self_eval),
                x_target,
                eval_mask,
                masked_weight=self.config.masked_weight,
                visible_weight=self.config.visible_weight,
                zero_weight=self.config.zero_weight,
                zero_mask=eval_zero_mask,
            )
            rec_probe = self.teacher.autoencoder.decode(z_probe.reshape(-1, z_probe.shape[-1]))
            rec_edge = sparse_reconstruction_per_sample(
                rec_probe,
                x_target[:, None, :].expand(-1, z_probe.shape[1], -1).reshape(-1, x_target.shape[1]),
                eval_mask[:, None, :].expand(-1, z_probe.shape[1], -1).reshape(-1, x_target.shape[1]),
                masked_weight=self.config.masked_weight,
                visible_weight=self.config.visible_weight,
                zero_weight=self.config.zero_weight,
                zero_mask=eval_zero_mask[:, None, :]
                .expand(-1, z_probe.shape[1], -1)
                .reshape(-1, x_target.shape[1]),
            ).reshape(z_probe.shape[:2])
            q_self_teacher_second = None
            q_edge_second = None
            q_edge_second_aligned = None
            rec_self_second = None
            rec_edge_second = None
            if eval_corrupted_second is not None:
                if eval_mask_second is None or eval_zero_mask_second is None:
                    raise ValueError("second-view masks are required with eval_corrupted_second")
                z_self_eval_second = self.teacher.encode(eval_corrupted_second)
                q_self_teacher_second = self.teacher.assignments(z_self_eval_second)
                z_probe_second = z_self_eval_second[:, None, :] + self.config.probe_alpha * (
                    donors - z_self_eval_second[:, None, :]
                )
                q_edge_second = self.teacher.assignments(
                    z_probe_second.reshape(-1, z_probe_second.shape[-1])
                ).reshape(z_probe_second.shape[0], z_probe_second.shape[1], -1)
                q_edge_second_aligned = (
                    (1.0 - self.config.output_alpha) * q_self_teacher_second[:, None, :]
                    + self.config.output_alpha * q_edge_second
                ).clamp_min(1e-8)
                q_edge_second_aligned = q_edge_second_aligned / q_edge_second_aligned.sum(dim=2, keepdim=True)
                rec_self_second = sparse_reconstruction_per_sample(
                    self.teacher.autoencoder.decode(z_self_eval_second),
                    x_target,
                    eval_mask_second,
                    masked_weight=self.config.masked_weight,
                    visible_weight=self.config.visible_weight,
                    zero_weight=self.config.zero_weight,
                    zero_mask=eval_zero_mask_second,
                )
                rec_probe_second = self.teacher.autoencoder.decode(
                    z_probe_second.reshape(-1, z_probe_second.shape[-1])
                )
                rec_edge_second = sparse_reconstruction_per_sample(
                    rec_probe_second,
                    x_target[:, None, :]
                    .expand(-1, z_probe_second.shape[1], -1)
                    .reshape(-1, x_target.shape[1]),
                    eval_mask_second[:, None, :]
                    .expand(-1, z_probe_second.shape[1], -1)
                    .reshape(-1, x_target.shape[1]),
                    masked_weight=self.config.masked_weight,
                    visible_weight=self.config.visible_weight,
                    zero_weight=self.config.zero_weight,
                    zero_mask=eval_zero_mask_second[:, None, :]
                    .expand(-1, z_probe_second.shape[1], -1)
                    .reshape(-1, x_target.shape[1]),
                ).reshape(z_probe_second.shape[:2])
            if (
                self.config.utility_target_mode == "operator_aligned"
                and self.config.direct_utility_source == "clean_output"
            ):
                # The exported assignment readout starts from the clean
                # student.  Build the detached target with the same clean
                # self-plus-edge operator; masked probes remain available as
                # the explicit ``direct_utility_source=masked_probe`` mode.
                z_self_clean = teacher_z_full[batch_tensor]
                z_edge_clean = z_self_clean[:, None, :] + self.config.probe_alpha * (
                    donors - z_self_clean[:, None, :]
                )
                q_self_clean = self.teacher.assignments(z_self_clean)
                q_edge_clean = self.teacher.assignments(
                    z_edge_clean.reshape(-1, z_edge_clean.shape[-1])
                ).reshape(z_edge_clean.shape[0], z_edge_clean.shape[1], -1)
                q_edge_clean = (
                    (1.0 - self.config.output_alpha) * q_self_clean[:, None, :]
                    + self.config.output_alpha * q_edge_clean
                ).clamp_min(1e-8)
                q_edge_clean = q_edge_clean / q_edge_clean.sum(dim=2, keepdim=True)
                utility = clean_output_utility(
                    q_teacher,
                    q_self_clean,
                    q_edge_clean,
                    min_gain=self.config.utility_min_gain,
                    clip=self.config.utility_clip,
                    valid=valid,
                )
                reference_edges = q_teacher[:, None, :].expand(-1, candidates, -1)
                semantic_help = (
                    kl_per_sample(q_teacher, q_self_clean)[:, None]
                    - kl_per_sample(
                        reference_edges.reshape(-1, clusters),
                        q_edge_clean.reshape(-1, clusters),
                    ).reshape(batch, candidates)
                )
                reconstruction_damage = torch.zeros_like(utility)
            elif self.config.utility_target_mode == "operator_aligned":
                if (
                    q_self_teacher_second is None
                    or q_edge_second_aligned is None
                    or rec_self_second is None
                    or rec_edge_second is None
                ):
                    raise ValueError("operator-aligned utility requires the second probe view")
                utility, semantic_help, reconstruction_damage = operator_aligned_utility(
                    q_teacher,
                    q_self_teacher,
                    q_edge_aligned,
                    q_self_teacher_second,
                    q_edge_second_aligned,
                    0.5 * (rec_self + rec_self_second),
                    0.5 * (rec_edge + rec_edge_second),
                    lambda_rec=self.config.utility_lambda_rec,
                    stability_weight=self.config.utility_stability_weight,
                    relative_baseline=self.config.utility_relative_baseline,
                    reference_mode=self.config.utility_reference_mode,
                    reference_temperature=self.config.utility_reference_temperature,
                    min_gain=self.config.utility_min_gain,
                    clip=self.config.utility_clip,
                    valid=valid,
                )
            elif self.config.utility_target_mode == "local_consensus":
                if (
                    q_self_teacher_second is None
                    or q_edge_second_aligned is None
                    or rec_self_second is None
                    or rec_edge_second is None
                ):
                    raise ValueError("local consensus utility requires the second probe view")
                utility, semantic_help, reconstruction_damage = local_consensus_utility(
                    q_self_teacher,
                    q_edge_aligned,
                    q_self_teacher_second,
                    q_edge_second_aligned,
                    q_donor,
                    0.5 * (rec_self + rec_self_second),
                    0.5 * (rec_edge + rec_edge_second),
                    lambda_rec=self.config.utility_lambda_rec,
                    stability_weight=self.config.utility_stability_weight,
                    confidence_weight=self.config.utility_confidence_weight,
                    relative_baseline=self.config.utility_relative_baseline,
                    min_gain=self.config.utility_min_gain,
                    clip=self.config.utility_clip,
                    valid=valid,
                )
            elif self.config.utility_target_mode == "stability":
                utility, semantic_help, reconstruction_damage = stability_gain_utility(
                    q_self_teacher,
                    q_edge,
                    rec_self,
                    rec_edge,
                    q_self_second=q_self_teacher_second,
                    q_edge_second=q_edge_second,
                    rec_self_second=rec_self_second,
                    rec_edge_second=rec_edge_second,
                    lambda_rec=self.config.utility_lambda_rec,
                    clip=self.config.utility_clip,
                    valid=valid,
                    confidence_weight=self.config.utility_confidence_weight,
                    relative_baseline=self.config.utility_relative_baseline,
                )
            else:
                semantic_help, reconstruction_damage = counterfactual_components(
                    q_teacher,
                    q_self_teacher,
                    q_edge,
                    rec_self,
                    rec_edge,
                    valid=valid,
                    q_self_second=q_self_teacher_second,
                    q_edge_second=q_edge_second,
                    rec_self_second=rec_self_second,
                    rec_edge_second=rec_edge_second,
                    stability_weight=self.config.utility_stability_weight,
                    confidence_weight=self.config.utility_confidence_weight,
                    opportunity_temperature=self.config.utility_opportunity_temperature,
                    edge_agreement=teacher_assignment_agreement,
                    agreement_weight=self.config.utility_assignment_agreement_weight,
                )
                utility = counterfactual_utility(
                    q_teacher,
                    q_self_teacher,
                    q_edge,
                    rec_self,
                    rec_edge,
                    lambda_rec=self.config.utility_lambda_rec,
                    clip=self.config.utility_clip,
                    valid=valid,
                    q_self_second=q_self_teacher_second,
                    q_edge_second=q_edge_second,
                    rec_self_second=rec_self_second,
                    rec_edge_second=rec_edge_second,
                    stability_weight=self.config.utility_stability_weight,
                    confidence_weight=self.config.utility_confidence_weight,
                    opportunity_temperature=self.config.utility_opportunity_temperature,
                    edge_agreement=teacher_assignment_agreement,
                    agreement_weight=self.config.utility_assignment_agreement_weight,
                )
        return (
            utility,
            donors,
            valid,
            teacher_assignment_agreement,
            semantic_help.detach(),
            reconstruction_damage.detach(),
            (0.5 * (q_self_teacher + q_self_teacher_second)).detach()
            if q_self_teacher_second is not None
            else q_self_teacher.detach(),
            (0.5 * (q_edge_aligned + q_edge_second_aligned)).detach()
            if q_edge_second_aligned is not None
            else q_edge_aligned.detach(),
        )

    def _target_for_probe_views(
        self,
        batch_indices: np.ndarray,
        probe_views: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        q_teacher_full: torch.Tensor,
        teacher_z_full: torch.Tensor,
        q_reference_full: torch.Tensor,
        reference_agreement_full: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Average independent two-view counterfactual estimates.

        Each pair retains the bidirectional lower-confidence-bound utility from
        ``operator_aligned_utility``. Averaging pairs estimates expected edge
        benefit under the masking distribution without adding a learned gate.
        """
        expected_views = 2 * int(self.config.utility_probe_pairs)
        if len(probe_views) != expected_views:
            raise ValueError(f"expected {expected_views} probe views, got {len(probe_views)}")
        pair_results = []
        for offset in range(0, len(probe_views), 2):
            first, second = probe_views[offset], probe_views[offset + 1]
            pair_results.append(
                self._target_for_batch(
                    batch_indices,
                    first[0],
                    first[1],
                    first[2],
                    q_teacher_full,
                    teacher_z_full,
                    q_reference_full,
                    reference_agreement_full,
                    second[0],
                    second[1],
                    second[2],
                )
            )
        valid = pair_results[0][2]
        if any(not torch.equal(result[2], valid) for result in pair_results[1:]):
            raise ValueError("candidate validity changed between counterfactual probe pairs")
        q_self = torch.stack([result[6] for result in pair_results], dim=0).mean(dim=0)
        q_edge = torch.stack([result[7] for result in pair_results], dim=0).mean(dim=0)
        q_self = q_self / q_self.sum(dim=1, keepdim=True).clamp_min(1e-8)
        q_edge = q_edge / q_edge.sum(dim=2, keepdim=True).clamp_min(1e-8)
        return (
            torch.stack([result[0] for result in pair_results], dim=0).mean(dim=0).detach(),
            pair_results[0][1],
            valid,
            torch.stack([result[3] for result in pair_results], dim=0).mean(dim=0).detach(),
            torch.stack([result[4] for result in pair_results], dim=0).mean(dim=0).detach(),
            torch.stack([result[5] for result in pair_results], dim=0).mean(dim=0).detach(),
            q_self.detach(),
            q_edge.detach(),
        )

    def _pretrain_backbone(
        self,
        optimizer: torch.optim.Optimizer,
        history: list[dict],
        generator: np.random.Generator,
    ) -> None:
        """Warm the topology-disabled anchor encoder before teacher creation."""
        for epoch in range(1, self.config.teacher_pretrain_epochs + 1):
            self.model.train()
            order = generator.permutation(self.data.n_samples)
            rows: list[dict[str, float]] = []
            for start in range(0, self.data.n_samples, self.config.batch_size):
                batch_indices = order[start : start + self.config.batch_size]
                x = self.data.get(batch_indices, self.device)
                first_view, first_mask = apply_mask(
                    x,
                    self.config.mask_ratio,
                    strategy=self.config.mask_strategy,
                )
                second_view, _ = apply_mask(
                    x,
                    self.config.mask_ratio,
                    strategy=self.config.mask_strategy,
                )
                zero_mask = (x == 0.0) & (torch.rand_like(x) < self.config.zero_sample_ratio)
                z_first, reconstruction, mask_logits = self.model.autoencoder(first_view)
                z_second = self.model.autoencoder.encode(second_view)
                rec = sparse_reconstruction_per_sample(
                    reconstruction,
                    x,
                    first_mask,
                    masked_weight=self.config.masked_weight,
                    visible_weight=self.config.visible_weight,
                    zero_weight=self.config.zero_weight,
                    zero_mask=zero_mask,
                ).mean()
                mask_loss = F.binary_cross_entropy_with_logits(mask_logits, first_mask, reduction="none").mean()
                view_loss = latent_view_consistency_loss(z_first, z_second)
                variance_loss = latent_variance_floor_loss(
                    torch.cat([z_first, z_second], dim=0),
                    self.config.teacher_variance_floor,
                )
                total = (
                    rec
                    + self.config.mask_prediction_weight * mask_loss
                    + self.config.teacher_view_consistency_weight * view_loss
                    + self.config.teacher_variance_weight * variance_loss
                )
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.autoencoder.parameters(), max_norm=5.0)
                optimizer.step()
                rows.append(
                    {
                        "loss": float(total.detach().cpu()),
                        "loss_rec": float(rec.detach().cpu()),
                        "loss_view": float(view_loss.detach().cpu()),
                        "loss_variance": float(variance_loss.detach().cpu()),
                    }
                )
            history.append(
                {
                    "epoch": epoch,
                    "phase": "teacher_pretrain",
                    **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
                }
            )

    def fit(self) -> TrainingResult:
        started = time.perf_counter()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        history: list[dict] = []
        graph_history: list[dict] = []
        generator = np.random.default_rng(self.config.seed)
        self._pretrain_backbone(optimizer, history, generator)
        initial_z = self.full_embeddings(self.model)
        self.model.cluster_head.initialise(initial_z, self.config.seed, self.config.n_init)
        self.teacher = make_teacher(self.model).to(self.device)
        teacher_z = self.full_embeddings(self.teacher)
        teacher_probabilities_initial = self.full_probabilities(self.teacher)
        self.graph = self._build_graph(teacher_z)
        self.raw_view_probabilities = self._build_raw_view_teacher(self.graph.raw_embedding)
        diagnostic_parts: dict[str, list[np.ndarray]] = {
            "anchor_indices": [],
            "utility_target": [],
            "utility_hat": [],
            "predicted_pi": [],
            "valid": [],
            "features": [],
            "gate_valid": [],
            "semantic_help": [],
            "reconstruction_damage": [],
            "reference_agreement": [],
            "reference_disagreement": [],
            "independent_cluster_gain": [],
            "probe_self_prediction": [],
            "probe_edge_prediction": [],
            "train_anchor": [],
        }

        for epoch in range(self.config.teacher_pretrain_epochs + 1, self.config.epochs + 1):
            assert self.graph is not None
            distill_active = bool(
                self.config.counterfactual_distill_weight > 0.0
                and epoch >= self.config.counterfactual_distill_start_epoch
            )
            self._selection_allowed = epoch >= (
                self.config.teacher_pretrain_epochs + self.config.teacher_selection_warmup_epochs
            )
            self.model.train()
            if self.config.freeze_backbone_after_teacher and self._backbone_frozen:
                self.model.autoencoder.eval()
                self.model.cluster_head.eval()
            self.teacher.eval()
            if (
                not distill_active
                and epoch > 1
                and epoch % self.config.graph_refresh_interval == 0
            ):
                teacher_z = self.full_embeddings(self.teacher)
                self.graph = refresh_latent_graph(
                    self.graph,
                    teacher_z,
                    self.config.k_latent,
                    self.config.candidate_cap,
                    self.config.latent_graph_dim,
                    self.config.seed,
                )
                self.graph = replace_candidate_edges(
                    self.graph,
                    self.config.graph_replacement_fraction,
                    self.config.seed + epoch,
                )
                self.graph = restrict_candidate_scope(self.graph, self.config.candidate_scope)
                graph_history.append({"epoch": epoch, **self.graph.profile})
            with torch.no_grad():
                teacher_z = torch.as_tensor(self.full_embeddings(self.teacher), dtype=torch.float32, device=self.device)
                q_teacher_latent_full = self.teacher.assignments(teacher_z)
                raw_view_aligned_full = align_teacher_assignments(
                    q_teacher_latent_full,
                    self.raw_view_probabilities,
                )
                q_reference_full, reference_agreement_full, reference_disagreement_full = (
                    self._build_teacher_reference(q_teacher_latent_full)
                )
            selection_pending = (
                self.config.teacher_reference_mode == "auto"
                and self.teacher_selection.get("raw_quality") is None
            )
            if (
                self.config.freeze_backbone_after_teacher
                and not selection_pending
                and not self._backbone_frozen
            ):
                # Freeze only after the topology-disabled calibration has
                # selected a mature teacher/view. The utility scorer remains
                # trainable for the registered frozen-backbone ablation.
                for parameter in self.model.autoencoder.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.model.cluster_head.parameters():
                    parameter.requires_grad_(False)
                self._backbone_frozen = True
            order = generator.permutation(self.data.n_samples)
            epoch_rows: list[dict] = []
            for start in range(0, self.data.n_samples, self.config.batch_size):
                batch_indices = order[start : start + self.config.batch_size]
                x = self.data.get(batch_indices, self.device)
                batch_tensor = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
                reference_agreement = reference_agreement_full[batch_tensor]
                reference_disagreement = reference_disagreement_full[batch_tensor]
                corrupted, mask = apply_mask(x, self.config.mask_ratio, strategy=self.config.mask_strategy)
                probe_views: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
                for _ in range(2 * self.config.utility_probe_pairs):
                    probe_corrupted, probe_mask = apply_mask(
                        x,
                        self.config.mask_ratio,
                        strategy=self.config.mask_strategy,
                    )
                    probe_zero_mask = (x == 0.0) & (
                        torch.rand_like(x) < self.config.zero_sample_ratio
                    )
                    probe_views.append((probe_corrupted, probe_mask, probe_zero_mask))
                zero_mask = (x == 0.0) & (torch.rand_like(x) < self.config.zero_sample_ratio)
                z_self, reconstruction, mask_logits = self.model.autoencoder(corrupted)
                z_eval_student = self.model.encode(probe_views[0][0])
                q_self = self.model.assignments(z_self)
                q_probe_self = q_self.detach()
                q_probe_edge = q_probe_self[:, None, :].expand(
                    -1, self.graph.n_candidates, -1
                )
                train_anchor = torch.as_tensor(
                    self.utility_train_mask[batch_indices],
                    dtype=torch.bool,
                    device=self.device,
                )
                use_amortized_utility = self.config.gate_mode == "counterfactual_learned"
                if self.config.gate_mode == "self_only" or selection_pending:
                    candidates = self.graph.n_candidates
                    donors = z_self.detach()[:, None, :].expand(-1, candidates, -1)
                    valid = torch.zeros((z_self.shape[0], candidates), dtype=torch.bool, device=self.device)
                    utility_target = torch.full(
                        valid.shape,
                        -float(self.config.utility_clip),
                        dtype=z_self.dtype,
                        device=self.device,
                    )
                    semantic_help = torch.zeros_like(utility_target)
                    reconstruction_damage = torch.zeros_like(utility_target)
                    features = torch.zeros(
                        (z_self.shape[0], candidates, 6),
                        dtype=z_self.dtype,
                        device=self.device,
                    )
                    utility_hat = torch.zeros_like(utility_target)
                    gate_valid = valid
                else:
                    (
                        utility_target,
                        donors,
                        valid,
                        teacher_assignment_agreement,
                        semantic_help,
                        reconstruction_damage,
                        q_probe_self,
                        q_probe_edge,
                    ) = self._target_for_probe_views(
                        batch_indices,
                        probe_views,
                        q_teacher_latent_full,
                        teacher_z,
                        q_reference_full,
                        reference_agreement_full,
                    )
                    features = _assemble_utility_features(
                        self.graph.features[batch_indices],
                        teacher_assignment_agreement,
                        semantic_help,
                        reconstruction_damage,
                        valid,
                        self.device,
                    )
                    direct_modes = {
                        "direct_target",
                        "direct_counterfactual",
                        "counterfactual_learned",
                        "union_uniform",
                        "forced_topk",
                        "shuffled_utility",
                        "output_disabled",
                    }
                    use_direct_utility = (
                        self.config.utility_target_mode in {"operator_aligned", "local_consensus"}
                        and self.config.gate_mode in direct_modes
                    )
                    utility_hat = (
                        self.model.utility(features)
                        if use_amortized_utility
                        else utility_target
                        if use_direct_utility
                        else self.model.utility(features)
                    )
                    gate_valid = valid & (
                        teacher_assignment_agreement >= float(self.config.gate_teacher_agreement_floor)
                    )
                if self.config.gate_mode == "self_only" or selection_pending:
                    use_direct_utility = False
                edge_scores = utility_hat / max(self.config.gate_temperature, 1e-4)
                predicted_pi = self._predict_gate_distribution(edge_scores, gate_valid)
                q_output_self = q_probe_self if use_direct_utility else q_self
                predicted_pi = self._apply_gate_opportunity(predicted_pi, q_output_self)
                edge_mass = predicted_pi[:, 1:]
                output_alpha = self._output_alpha_for_batch(features, edge_mass)
                if self.config.output_mode == "logit":
                    z_out = z_self
                    q_out = self.model.mix_assignments(
                        z_self,
                        donors,
                        edge_mass.detach(),
                        output_alpha,
                    )
                elif self.config.output_mode == "probability":
                    q_donor_teacher = self.teacher.assignments(
                        donors.reshape(-1, donors.shape[-1])
                    ).reshape(donors.shape[0], donors.shape[1], -1)
                    q_edge = (
                        (1.0 - self.config.output_alpha) * q_self[:, None, :]
                        + self.config.output_alpha * q_donor_teacher
                    ).clamp_min(1e-8)
                    q_edge = q_edge / q_edge.sum(dim=2, keepdim=True)
                    z_out = z_self
                    q_out = self.model.mix_probabilities(
                        q_self,
                        q_donor_teacher,
                        edge_mass.detach(),
                        output_alpha,
                    )
                elif self.config.output_mode == "assignment":
                    if use_direct_utility:
                        # The exact counterfactual target is evaluated on the
                        # detached teacher probes, so there is no student
                        # latent whose assignment head produced q_probe_*.
                        # Keep the clean student embedding for this training
                        # diagnostic; the exported readout is assembled in
                        # ``_full_gated_outputs`` from the matching clean
                        # assignment transport below.
                        z_out = z_self
                        q_out = self.model.mix_assignment_output(
                            q_probe_self,
                            q_probe_edge,
                            predicted_pi.detach(),
                        )
                    else:
                        z_edge_student = z_self[:, None, :] + self.config.probe_alpha * (
                            donors.detach() - z_self[:, None, :]
                        )
                        q_edge_student = self.model.assignments(
                            z_edge_student.reshape(-1, z_edge_student.shape[-1])
                        ).reshape(z_edge_student.shape[0], z_edge_student.shape[1], -1)
                        q_edge_student = (
                            (1.0 - self.config.output_alpha) * q_self[:, None, :]
                            + self.config.output_alpha * q_edge_student
                        ).clamp_min(1e-8)
                        q_edge_student = q_edge_student / q_edge_student.sum(dim=2, keepdim=True)
                        q_out = self.model.mix_assignment_output(
                            q_self,
                            q_edge_student,
                            predicted_pi.detach(),
                        )
                        # Keep the embedding export/readout on the same
                        # null-plus-edge transport as q_out.  Using z_self
                        # here silently discarded the selected topology during
                        # joint training even though q_out used it.
                        z_out = self.model.mix_assignment_embedding(
                            z_self,
                            z_edge_student,
                            predicted_pi.detach(),
                        )
                else:
                    z_out = self.model.mix_latent(z_self, donors, edge_mass.detach(), output_alpha)
                    q_out = self.model.assignments(z_out)
                q_teacher = q_reference_full[batch_tensor]
                assert self.raw_view_probabilities is not None
                q_raw_view = raw_view_aligned_full[batch_tensor]
                rec_per = sparse_reconstruction_per_sample(
                    reconstruction,
                    x,
                    mask,
                    masked_weight=self.config.masked_weight,
                    visible_weight=self.config.visible_weight,
                    zero_weight=self.config.zero_weight,
                    zero_mask=zero_mask,
                )
                mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask, reduction="none").mean(dim=1)
                loss_rec = rec_per.mean() + self.config.mask_prediction_weight * mask_loss.mean()
                loss_view = latent_view_consistency_loss(z_self, z_eval_student)
                loss_variance = latent_variance_floor_loss(
                    torch.cat([z_self, z_eval_student], dim=0),
                    self.config.teacher_variance_floor,
                )
                frequency_target = (
                    (1.0 - self.config.cluster_frequency_uniform_mix) * self.cluster_frequency_ema
                    + self.config.cluster_frequency_uniform_mix
                    * torch.full_like(self.cluster_frequency_ema, 1.0 / self.n_clusters)
                )
                cluster_prediction = q_self if self.config.gate_training_mode == "detached" else q_out
                loss_cluster = (
                    (reference_agreement * kl_per_sample(q_teacher.detach(), cluster_prediction)).mean()
                    + 0.5 * (reference_agreement * kl_per_sample(q_teacher.detach(), q_self)).mean()
                    + self.model.cluster_head.prior_loss(
                        cluster_prediction,
                        self.config.dirichlet_strength,
                        frequency_target,
                        self.config.cluster_frequency_weight,
                    )
                )
                loss_raw_view = (
                    kl_per_sample(q_raw_view, cluster_prediction).mean()
                    + 0.5 * kl_per_sample(q_raw_view, q_self).mean()
                )
                with torch.no_grad():
                    self.cluster_frequency_ema.mul_(self.config.cluster_frequency_decay).add_(
                        cluster_prediction.detach().mean(dim=0),
                        alpha=1.0 - self.config.cluster_frequency_decay,
                    )
                # ExactCF targets are available for every anchor, but only the
                # pre-drawn utility-training split may update the scorer. The
                # complementary split is a genuine held-out diagnostic.
                fit_valid = valid & train_anchor[:, None]
                if use_direct_utility and not use_amortized_utility:
                    utility_loss = torch.zeros((), device=self.device)
                    sign_loss = torch.zeros((), device=self.device)
                elif torch.any(fit_valid):
                    utility_loss = F.smooth_l1_loss(utility_hat[fit_valid], utility_target[fit_valid])
                    sign_target = (utility_target[fit_valid] > 0.0).to(dtype=utility_hat.dtype)
                    sign_loss = F.binary_cross_entropy_with_logits(
                        utility_hat[fit_valid] / max(self.config.utility_temperature, 1e-4),
                        sign_target,
                    )
                else:
                    utility_loss = torch.zeros((), device=self.device)
                    sign_loss = torch.zeros((), device=self.device)
                holdout_valid = valid & ~train_anchor[:, None]
                if use_amortized_utility and torch.any(holdout_valid):
                    holdout_utility_loss = F.smooth_l1_loss(
                        utility_hat[holdout_valid], utility_target[holdout_valid]
                    )
                else:
                    holdout_utility_loss = torch.zeros((), device=self.device)
                loss_gate = utility_loss + self.config.utility_sign_weight * sign_loss
                if distill_active and use_direct_utility:
                    student_probe_probabilities = [
                        self.model.assignments(self.model.encode(view[0]))
                        for view in probe_views
                    ]
                    distill_per_anchor = torch.stack(
                        [
                            kl_per_sample(q_out.detach(), prediction)
                            for prediction in student_probe_probabilities
                        ]
                    ).mean(dim=0)
                    topology_mass = predicted_pi[:, 1:].sum(dim=1).detach()
                    if torch.any(topology_mass > 0.0):
                        loss_counterfactual_distill = (
                            topology_mass * distill_per_anchor
                        ).sum() / topology_mass.sum().clamp_min(1e-8)
                    else:
                        loss_counterfactual_distill = torch.zeros((), device=self.device)
                else:
                    topology_mass = torch.zeros(z_self.shape[0], device=self.device)
                    loss_counterfactual_distill = torch.zeros((), device=self.device)
                gate_epoch = epoch - self.config.teacher_pretrain_epochs
                gate_ramp = min(
                    1.0,
                    max(0.0, (gate_epoch - self.config.warmup_epochs) / max(1, self.config.warmup_epochs)),
                )
                total = (
                    loss_rec
                    + self.config.teacher_view_consistency_weight * loss_view
                    + self.config.teacher_variance_weight * loss_variance
                    + self.config.lambda_cluster * loss_cluster
                    + self.config.raw_view_cluster_weight * loss_raw_view
                    + self.config.lambda_gate * gate_ramp * loss_gate
                    + self.config.counterfactual_distill_weight
                    * gate_ramp
                    * loss_counterfactual_distill
                )
                optimizer.zero_grad(set_to_none=True)
                if total.requires_grad:
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    optimizer.step()
                # Mature the topology-disabled EMA before transfer, then freeze
                # it so the counterfactual target cannot learn its own output.
                if not distill_active:
                    ema_update(self.teacher, self.model, self.config.ema_decay)
                if epoch == self.config.epochs:
                    with torch.no_grad():
                        certificate_views: list[
                            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                        ] = []
                        for _ in range(2 * self.config.utility_probe_pairs):
                            certificate_view, certificate_mask = apply_mask(
                                x,
                                self.config.mask_ratio,
                                strategy=self.config.mask_strategy,
                            )
                            certificate_zero = (x == 0.0) & (
                                torch.rand_like(x) < self.config.zero_sample_ratio
                            )
                            certificate_views.append(
                                (certificate_view, certificate_mask, certificate_zero)
                            )
                        (
                            independent_cluster_gain,
                            _certificate_donors,
                            certificate_valid,
                            _certificate_agreement,
                            _certificate_semantic,
                            _certificate_damage,
                            q_certificate_self,
                            q_certificate_edge,
                        ) = self._target_for_probe_views(
                            batch_indices,
                            certificate_views,
                            q_teacher_latent_full,
                            teacher_z,
                            q_reference_full,
                            reference_agreement_full,
                        )
                        independent_cluster_gain = independent_cluster_gain.masked_fill(
                            ~certificate_valid,
                            0.0,
                        )
                    diagnostic_parts["anchor_indices"].append(np.asarray(batch_indices, dtype=np.int64))
                    diagnostic_parts["utility_target"].append(utility_target.detach().cpu().numpy().astype(np.float32))
                    diagnostic_parts["utility_hat"].append(utility_hat.detach().cpu().numpy().astype(np.float32))
                    diagnostic_parts["predicted_pi"].append(predicted_pi.detach().cpu().numpy().astype(np.float32))
                    diagnostic_parts["valid"].append(valid.detach().cpu().numpy().astype(bool))
                    diagnostic_parts["features"].append(features.detach().cpu().numpy().astype(np.float32))
                    diagnostic_parts["semantic_help"].append(semantic_help.detach().cpu().numpy().astype(np.float32))
                    diagnostic_parts["reconstruction_damage"].append(
                        reconstruction_damage.detach().cpu().numpy().astype(np.float32)
                    )
                    diagnostic_parts["reference_agreement"].append(
                        reference_agreement.detach().cpu().numpy().astype(np.float32)
                    )
                    diagnostic_parts["reference_disagreement"].append(
                        reference_disagreement.detach().cpu().numpy().astype(np.float32)
                    )
                    diagnostic_parts["independent_cluster_gain"].append(
                        independent_cluster_gain.detach().cpu().numpy().astype(np.float32)
                    )
                    diagnostic_parts["probe_self_prediction"].append(
                        q_certificate_self.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                    )
                    diagnostic_parts["probe_edge_prediction"].append(
                        q_certificate_edge.argmax(dim=2).detach().cpu().numpy().astype(np.int64)
                    )
                    diagnostic_parts["train_anchor"].append(train_anchor.detach().cpu().numpy().astype(bool))
                    diagnostic_parts["gate_valid"].append(gate_valid.detach().cpu().numpy().astype(bool))
                epoch_rows.append(
                    {
                        "loss": float(total.detach().cpu()),
                        "loss_rec": float(loss_rec.detach().cpu()),
                        "loss_cluster": float(loss_cluster.detach().cpu()),
                        "loss_raw_view": float(loss_raw_view.detach().cpu()),
                        "loss_gate": float(loss_gate.detach().cpu()),
                        "loss_view": float(loss_view.detach().cpu()),
                        "loss_variance": float(loss_variance.detach().cpu()),
                        "loss_gate_regression": float(utility_loss.detach().cpu()),
                        "loss_gate_sign": float(sign_loss.detach().cpu()),
                        "loss_gate_holdout": float(holdout_utility_loss.detach().cpu()),
                        "loss_counterfactual_distill": float(loss_counterfactual_distill.detach().cpu()),
                        "distill_active": float(distill_active),
                        "distill_topology_mass": float(topology_mass.mean().detach().cpu()),
                        "utility_positive_rate": float((utility_target[valid] > 0).float().mean().detach().cpu()) if torch.any(valid) else 0.0,
                        "null_mass": float(predicted_pi[:, 0].mean().detach().cpu()),
                        "edge_mass": float(predicted_pi[:, 1:].sum(dim=1).mean().detach().cpu()),
                        "effective_neighbors": float(torch.exp(-(predicted_pi[:, 1:] * torch.log(predicted_pi[:, 1:].clamp_min(1e-8))).sum(dim=1)).mean().detach().cpu()),
                        "cluster_frequency_min": float(self.cluster_frequency_ema.min().detach().cpu()),
                        "cluster_frequency_max": float(self.cluster_frequency_ema.max().detach().cpu()),
                    }
                )
            aggregate = {
                "epoch": epoch,
                "phase": "joint",
                "gate_ramp": gate_ramp,
                "distill_active": bool(distill_active),
                **{
                    key: float(np.mean([row[key] for row in epoch_rows]))
                    for key in epoch_rows[0]
                    if key not in {"epoch", "distill_active"}
                },
            }
            history.append(aggregate)

        (
            embedding,
            probabilities,
            final_pi,
            final_utility,
            final_features,
            final_gate_valid,
            final_self_prediction,
            final_edge_prediction,
            final_self_embedding,
            final_q_self,
            final_q_edge,
            final_edge_embedding,
            final_transport_embedding,
        ) = self._full_gated_outputs()
        gate_readout_embedding = embedding
        gate_readout_probabilities = probabilities
        student_embedding = self.full_embeddings(self.model)
        student_probabilities = self.full_probabilities(self.model)
        if self.config.final_prediction_source == "student_clean":
            embedding = student_embedding
            probabilities = student_probabilities
        else:
            embedding = gate_readout_embedding
            probabilities = gate_readout_probabilities
        predictions = probabilities.argmax(axis=1).astype(np.int64)
        gate_diagnostics = {
            key: np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.float32)
            for key, parts in diagnostic_parts.items()
        }
        anchor_indices = gate_diagnostics["anchor_indices"].astype(np.int64, copy=False)
        if anchor_indices.size:
            canonical_order = np.argsort(anchor_indices)
            for key, values in gate_diagnostics.items():
                if values.shape[0] == canonical_order.shape[0]:
                    gate_diagnostics[key] = values[canonical_order]
        gate_diagnostics["final_predicted_pi"] = final_pi
        gate_diagnostics["final_utility_hat"] = final_utility
        gate_diagnostics["final_utility_features"] = final_features
        gate_diagnostics["final_gate_valid"] = final_gate_valid
        gate_diagnostics["final_probe_self_prediction"] = final_self_prediction
        gate_diagnostics["final_probe_edge_prediction"] = final_edge_prediction
        gate_diagnostics["final_embedding_self"] = final_self_embedding
        gate_diagnostics["final_q_self"] = final_q_self
        gate_diagnostics["final_q_edge"] = final_q_edge
        gate_diagnostics["final_edge_embedding"] = final_edge_embedding
        gate_diagnostics["final_embedding_transport"] = final_transport_embedding
        gate_diagnostics["final_gate_readout_probabilities"] = gate_readout_probabilities
        gate_diagnostics["final_student_probabilities"] = student_probabilities
        teacher_embedding = self.full_embeddings(self.teacher)
        teacher_probabilities_final = self.full_probabilities(self.teacher)
        teacher_probabilities_final_tensor = torch.as_tensor(
            teacher_probabilities_final,
            dtype=torch.float32,
            device=self.device,
        )
        augmented_reference = self._quality_augmented_reference
        if self.config.teacher_reference_mode == "quality_auto" and augmented_reference is None:
            augmented_reference = torch.as_tensor(
                self.full_probabilities_masked(self.teacher, self.config.seed + 2501),
                dtype=torch.float32,
                device=self.device,
            )
        teacher_probabilities_reference, teacher_reference_agreement, teacher_reference_disagreement = (
            self._build_teacher_reference(teacher_probabilities_final_tensor, augmented_reference)
        )
        teacher_probabilities_raw_aligned = align_teacher_assignments(
            teacher_probabilities_final_tensor,
            self.raw_view_probabilities,
        )
        teacher_probabilities_augmented = self.full_probabilities_masked(
            self.teacher,
            self.config.seed + 2501,
        )
        shuffled_rng = np.random.default_rng(self.config.seed + 2502)
        teacher_probabilities_shuffled = teacher_probabilities_final[
            shuffled_rng.permutation(self.data.n_samples)
        ]
        return TrainingResult(
            embedding=embedding,
            probabilities=probabilities,
            predictions=predictions,
            history=history,
            graph_history=graph_history,
            graph=self.graph,
            train_seconds=float(time.perf_counter() - started),
            gate_diagnostics=gate_diagnostics,
            teacher_diagnostics={
                "embedding": teacher_embedding,
                "probabilities_clean": teacher_probabilities_final,
                "probabilities_augmented": teacher_probabilities_augmented,
                "probabilities_epoch0": teacher_probabilities_initial,
                "probabilities_epoch_last": teacher_probabilities_final,
                "probabilities_shuffled": teacher_probabilities_shuffled,
                "probabilities_raw_view": self.raw_view_probabilities.detach().cpu().numpy().astype(np.float32),
                "probabilities_raw_aligned": teacher_probabilities_raw_aligned.cpu().numpy().astype(np.float32),
                "probabilities_reference": teacher_probabilities_reference.cpu().numpy().astype(np.float32),
                "reference_agreement": teacher_reference_agreement.cpu().numpy().astype(np.float32),
                "reference_disagreement": teacher_reference_disagreement.cpu().numpy().astype(np.float32),
            },
            teacher_selection=dict(self.teacher_selection),
            cluster_frequency_ema=self.cluster_frequency_ema.detach().cpu().numpy().astype(np.float32),
        )
