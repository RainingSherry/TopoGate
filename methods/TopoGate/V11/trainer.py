"""Training loop for TopoGate V11.

The objective has three named terms:

``L = L_rec + lambda_cls L_cls + lambda_graph L_graph``

``L_rec`` contains the real and optional topology-mixture reconstruction
views, ``L_cls`` is confidence-filtered soft-mixture self-training, and
``L_graph`` fits a self/null-plus-edge posterior.  The V11.2 counterfactual
target uses a frozen teacher-defined topology proposal and opens the graph
only when the proposal improves both paired reconstruction risk and paired
clean-assignment risk under the same feature intervention.  The historical
self-referential reconstruction target remains available as an explicit
ablation. Labels are never accepted by this class.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import V11Config
from .graph import (
    CandidateGraph,
    build_candidate_graph,
    edge_recurrence_against,
    graph_change_fraction,
)
from .model import TopoGateV11, ema_update, make_teacher
from .tda import H0Persistence, candidate_prior_from_h0, compute_h0_persistence


@dataclass
class TrainingResult:
    embedding: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    history: list[dict]
    graph_history: list[dict]
    train_seconds: float


@dataclass(frozen=True)
class CounterfactualGateTarget:
    """Detached supervision for the self/null versus topology posterior."""

    topology_help: torch.Tensor
    target_mixture: torch.Tensor
    edge_conditional: torch.Tensor
    reconstruction_help: torch.Tensor
    cluster_help: torch.Tensor
    reconstruction_improvement: torch.Tensor
    cluster_improvement: torch.Tensor


def corrupt_batch(
    x: torch.Tensor,
    ratio: float,
    mask: torch.Tensor | None = None,
    donor_indices: torch.Tensor | None = None,
    replacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        mask = (torch.rand_like(x) < float(ratio)).to(x.dtype)
    else:
        mask = mask.to(dtype=x.dtype, device=x.device)
    if replacement is not None:
        replacement = replacement.to(dtype=x.dtype, device=x.device)
        if replacement.shape != x.shape:
            raise ValueError("replacement must have the same shape as x")
    elif donor_indices is not None:
        donor_indices = donor_indices.to(dtype=torch.long, device=x.device).view(-1)
        if donor_indices.shape[0] != x.shape[0]:
            raise ValueError("donor_indices must contain one row index per sample")
        replacement = x[donor_indices]
    elif x.shape[0] <= 1:
        replacement = x
    else:
        replacement = x[torch.randperm(x.shape[0], device=x.device)]
    return torch.where(mask.bool(), replacement, x), mask


def kl_per_sample(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    target = target.clamp_min(1e-8)
    prediction = prediction.clamp_min(1e-8)
    return torch.sum(target * (torch.log(target) - torch.log(prediction)), dim=1)


def sharpen_assignments(q: torch.Tensor, frequency: torch.Tensor, temperature: float) -> torch.Tensor:
    power = 1.0 / max(float(temperature), 1e-3)
    target = q.clamp_min(1e-8).pow(power)
    target = target / frequency.clamp_min(1e-6)[None, :]
    return target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _relative_positive_risk_help(
    anchor_risk: torch.Tensor,
    probe_risk: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a positive paired risk reduction to a dimensionless confidence.

    Dividing by the anchor risk makes reconstruction and assignment risks
    comparable without asserting that their raw numerical units are equal.
    Equal or worse probes receive exactly zero evidence.
    """
    if anchor_risk.shape == probe_risk.shape:
        aligned_anchor = anchor_risk
    elif (
        anchor_risk.ndim + 1 == probe_risk.ndim
        and anchor_risk.shape == probe_risk.shape[:-1]
    ):
        aligned_anchor = anchor_risk[..., None]
    else:
        raise ValueError("anchor risk must match probe risk or its leading dimensions")
    improvement = F.relu(aligned_anchor.detach() - probe_risk.detach())
    relative = improvement / aligned_anchor.detach().clamp_min(1e-6)
    help_value = 1.0 - torch.exp(
        -relative / max(float(temperature), 1e-6)
    )
    return help_value.clamp(0.0, 1.0), improvement


def counterfactual_semantic_target(
    reconstruction_anchor_risk: torch.Tensor,
    reconstruction_probe_risk: torch.Tensor,
    cluster_anchor_risk: torch.Tensor,
    cluster_probe_risk: torch.Tensor,
    edge_target: torch.Tensor,
    confidence: torch.Tensor,
    reconstruction_temperature: float,
    cluster_temperature: float,
    semantic_help_combiner: str = "geometric_mean",
) -> CounterfactualGateTarget:
    """Construct a conservative topology target from per-edge risk tests.

    The geometric mean is an AND-like evidence combiner: the topology channel
    receives exactly zero mass for an edge if either the reconstruction or
    assignment probe fails to improve.  ``edge_target`` is an exogenous
    teacher-defined candidate prior, never the learned mixture being trained.
    All inputs are detached because this object is a teacher target; gradients
    must flow only through the learned mixture.
    """
    reconstruction_help, reconstruction_improvement = _relative_positive_risk_help(
        reconstruction_anchor_risk,
        reconstruction_probe_risk,
        reconstruction_temperature,
    )
    cluster_help, cluster_improvement = _relative_positive_risk_help(
        cluster_anchor_risk,
        cluster_probe_risk,
        cluster_temperature,
    )
    if confidence.ndim != 1 or confidence.shape[0] != reconstruction_help.shape[0]:
        raise ValueError("confidence must contain one value per paired risk")
    if edge_target.shape != reconstruction_help.shape:
        raise ValueError("edge_target must be [batch, candidates]")

    detached_edges = edge_target.detach().clamp_min(0.0)
    detached_edges = detached_edges / detached_edges.sum(dim=1, keepdim=True).clamp_min(1e-8)
    product = (reconstruction_help * cluster_help).clamp_min(0.0)
    if semantic_help_combiner == "geometric_mean":
        joint_help = torch.sqrt(product)
    elif semantic_help_combiner == "harmonic_mean":
        joint_help = (
            2.0 * product
            / (reconstruction_help + cluster_help).clamp_min(1e-8)
        )
    elif semantic_help_combiner == "minimum":
        joint_help = torch.minimum(reconstruction_help, cluster_help)
    elif semantic_help_combiner == "product":
        joint_help = product
    else:
        raise ValueError(
            "semantic_help_combiner must be geometric_mean, harmonic_mean, "
            "minimum, or product"
        )
    joint_help = joint_help.clamp(0.0, 1.0)
    scored_edges = detached_edges * joint_help
    support = scored_edges.sum(dim=1)
    topology_help = (confidence.detach().clamp(0.0, 1.0) * support).clamp(0.0, 1.0)
    conditional_edges = torch.where(
        support[:, None] > 1e-8,
        scored_edges / support[:, None].clamp_min(1e-8),
        detached_edges,
    )
    target_mixture = torch.cat(
        [
            (1.0 - topology_help)[:, None],
            topology_help[:, None] * conditional_edges,
        ],
        dim=1,
    )
    return CounterfactualGateTarget(
        topology_help=topology_help,
        target_mixture=target_mixture,
        edge_conditional=conditional_edges,
        reconstruction_help=reconstruction_help,
        cluster_help=cluster_help,
        reconstruction_improvement=reconstruction_improvement,
        cluster_improvement=cluster_improvement,
    )


def trusted_edge_alignment(
    anchor: torch.Tensor,
    teacher_neighbours: torch.Tensor,
    edge_mass: torch.Tensor,
    sample_trust: torch.Tensor,
    valid: torch.Tensor | None = None,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Risk-weighted soft contrastive alignment over trusted candidate edges.

    ``edge_mass`` is normally a detached, counterfactual edge target rather
    than the learned gate posterior.  The target therefore cannot be made true
    by changing the gate itself; it only supplies a label-free positive-edge
    distribution for the student representation.  The denominator includes
    every valid candidate, so incompatible candidates remain negatives.  The
    graph posterior is still trained separately by ``graph_loss``.
    """
    if teacher_neighbours.ndim != 3 or anchor.ndim != 2:
        raise ValueError("anchor and teacher_neighbours must be [B,D] and [B,K,D]")
    if edge_mass.shape != teacher_neighbours.shape[:2]:
        raise ValueError("edge_mass shape must match teacher_neighbours[:2]")
    if sample_trust.numel() != anchor.shape[0]:
        raise ValueError("sample_trust must contain one value per anchor")
    if valid is None:
        valid = torch.ones_like(edge_mass, dtype=torch.bool)
    else:
        if valid.shape != edge_mass.shape:
            raise ValueError("valid shape must match edge_mass")
        valid = valid.to(dtype=torch.bool, device=edge_mass.device)
    positive_mass = edge_mass.to(dtype=anchor.dtype, device=anchor.device).clamp_min(0.0)
    positive_mass = positive_mass.masked_fill(~valid, 0.0)
    conditional = positive_mass / positive_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
    logits = F.cosine_similarity(anchor[:, None, :], teacher_neighbours, dim=2)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~valid, -1e9)
    per_sample = -torch.sum(
        conditional.detach() * F.log_softmax(logits, dim=1), dim=1
    )
    # Keep the evidence mass in the objective.  Dividing by ``trust.sum``
    # would make a 4%-gate batch contribute the same full-strength geometry
    # loss as a 90%-gate batch, allowing weak counterfactual evidence to
    # overwhelm the clean objective.  A sample mean gives the intended
    # evidence-weighted curriculum and makes a zero-evidence batch exactly
    # zero.
    trust = sample_trust.to(dtype=per_sample.dtype, device=per_sample.device).view(-1).detach()
    return torch.mean(trust * per_sample)


class V11Trainer:
    def __init__(
        self,
        X: np.ndarray,
        raw_embedding: np.ndarray,
        n_clusters: int,
        config: V11Config,
        device: torch.device,
    ) -> None:
        if n_clusters < 1 or n_clusters > X.shape[0]:
            raise ValueError("n_clusters must lie in [1, n_samples]")
        self.cfg = config
        self.device = device
        self.X_np = np.asarray(X, dtype=np.float32)
        self.raw_embedding = np.asarray(raw_embedding, dtype=np.float32)
        self.X_cpu = torch.as_tensor(self.X_np, dtype=torch.float32)
        self.n_clusters = int(n_clusters)
        self._knn_options = {
            "knn_backend": config.knn_backend,
            "knn_exact_max_nodes": config.knn_exact_max_nodes,
            "knn_hnsw_m": config.knn_hnsw_m,
            "knn_hnsw_ef_search": config.knn_hnsw_ef_search,
        }
        self.student = TopoGateV11(
            input_dim=X.shape[1],
            hidden_dim=config.hidden_size,
            latent_dim=config.latent_size,
            n_clusters=n_clusters,
            dropout=config.dropout,
            null_bias=config.gate_initial_null_bias,
            student_t_nu=config.student_t_nu,
            cluster_logit_normalization=config.cluster_logit_normalization,
            cluster_scale_floor_ratio=config.cluster_scale_floor_ratio,
            cluster_assignment_kernel=config.cluster_assignment_kernel,
        ).to(device)
        self.teacher = make_teacher(self.student)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        # Strict NoMix must not materialise a candidate graph or any graph
        # state.  Topology-enabled variants construct their first consensus
        # graph only after the representation warm-up.
        self.graph: CandidateGraph | None = None
        # A previous *latent-only* graph is intentionally distinct from the
        # raw/latent candidate union.  This avoids rewarding an edge merely
        # because raw neighbours are always inserted into the union.
        self.previous_latent_graph: CandidateGraph | None = None
        self.edge_recurrence: np.ndarray | None = None
        self.temporal_target_available = False
        self.tda_persistence: H0Persistence | None = None
        if self.cfg.tda_prior_mode in {"h0_mst", "h0_early_mst", "fixed_filtration"}:
            raw_graph = build_candidate_graph(
                self.raw_embedding,
                None,
                self.cfg.neighbor_k,
                self.cfg.candidate_k,
                **self._knn_options,
            )
            if raw_graph.raw_knn_indices is None:
                raise RuntimeError("raw kNN indices were not retained for TDA")
            self.tda_persistence = compute_h0_persistence(
                self.raw_embedding,
                raw_graph.raw_knn_indices,
                scale_mode=self.cfg.tda_scale_mode,
                scale_quantile=self.cfg.tda_scale_quantile,
                scale_floor=self.cfg.tda_scale_floor,
            )
        self.cluster_frequency = torch.full(
            (n_clusters,), 1.0 / n_clusters, dtype=torch.float32, device=device
        )
        self.history: list[dict] = []
        self.graph_history: list[dict] = []
        self._loader_calls = 0

    def _loader(self, shuffle: bool) -> DataLoader:
        indices = torch.arange(self.X_cpu.shape[0], dtype=torch.long)
        self._loader_calls += 1
        generator = torch.Generator().manual_seed(self.cfg.seed + self._loader_calls)
        return DataLoader(
            TensorDataset(indices, self.X_cpu),
            batch_size=min(self.cfg.batch_size, self.X_cpu.shape[0]),
            shuffle=shuffle,
            drop_last=False,
            generator=generator if shuffle else None,
        )

    @torch.no_grad()
    def _full_embeddings(self, model: TopoGateV11) -> np.ndarray:
        was_training = model.training
        model.eval()
        chunks = []
        for _, x_cpu in self._loader(shuffle=False):
            chunks.append(model.encode(x_cpu.to(self.device)).cpu().numpy())
        if was_training:
            model.train()
        return np.concatenate(chunks, axis=0).astype(np.float32)

    @torch.no_grad()
    def _full_probabilities(self, model: TopoGateV11) -> np.ndarray:
        was_training = model.training
        model.eval()
        chunks = []
        for _, x_cpu in self._loader(shuffle=False):
            chunks.append(model.assignments(x_cpu.to(self.device)).cpu().numpy())
        if was_training:
            model.train()
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _initialise_clusters(self) -> None:
        embedding = self._full_embeddings(self.student)
        self.student.cluster_head.initialise(embedding, self.cfg.seed)
        self.teacher.load_state_dict(self.student.state_dict())
        probabilities = self._full_probabilities(self.teacher)
        frequency = probabilities.mean(axis=0)
        self.cluster_frequency = torch.as_tensor(
            frequency / np.clip(frequency.sum(), 1e-8, None),
            dtype=torch.float32,
            device=self.device,
        )

    def _refresh_graph(self, epoch: int) -> None:
        source_model = self.teacher if self.cfg.use_teacher else self.student
        # Every topology-enabled variant receives the same initial
        # input–EMA-latent consensus graph. ``use_dynamic_graph`` controls
        # subsequent refreshes in ``fit``; it must not silently turn the
        # fixed-graph ablation into a different raw-only candidate graph.
        latent = self._full_embeddings(source_model)
        updated = build_candidate_graph(
            self.raw_embedding,
            latent,
            self.cfg.neighbor_k,
            self.cfg.candidate_k,
            **self._knn_options,
        )
        if self.cfg.tda_prior_mode != "none":
            updated.tda_prior = candidate_prior_from_h0(
                self.tda_persistence,
                updated.indices,
                updated.valid,
                mode=self.cfg.tda_prior_mode,
                seed=self.cfg.seed,
            )
        latent_only = build_candidate_graph(
            latent,
            None,
            self.cfg.neighbor_k,
            self.cfg.candidate_k,
            **self._knn_options,
        )
        # Construction is not drift: no earlier candidate set exists at the
        # first topology phase.
        change = 0.0 if self.graph is None else graph_change_fraction(self.graph, updated)
        recurrence = edge_recurrence_against(updated, self.previous_latent_graph)
        temporal_available = self.previous_latent_graph is not None
        self.graph = updated
        self.edge_recurrence = recurrence
        self.temporal_target_available = temporal_available
        self.previous_latent_graph = latent_only
        self._refresh_cluster_frequency(source_model)
        self.graph_history.append(
            {
                "epoch": int(epoch),
                "source": updated.source,
                "knn_backend": updated.knn_backend,
                "edge_change_fraction": float(change),
                "mean_raw_similarity": float(updated.raw_similarity[updated.valid].mean()),
                "mean_mutual": float(updated.mutual[updated.valid].mean()),
                "mean_snn": float(updated.snn[updated.valid].mean()),
                "tda_prior_mode": self.cfg.tda_prior_mode,
                "tda_scale": (
                    float(self.tda_persistence.scale)
                    if self.tda_persistence is not None
                    else None
                ),
                "tda_h0_merge_count": (
                    int(self.tda_persistence.merge_count)
                    if self.tda_persistence is not None
                    else None
                ),
                "mean_tda_prior": (
                    float(updated.tda_prior[updated.valid].mean())
                    if updated.tda_prior is not None and np.any(updated.valid)
                    else 0.0
                ),
                "tda_prior_nonzero_fraction": (
                    float(np.mean(updated.tda_prior[updated.valid] > 0.0))
                    if updated.tda_prior is not None and np.any(updated.valid)
                    else 0.0
                ),
                "temporal_target_available": bool(temporal_available),
                "mean_temporal_recurrence": (
                    float(recurrence[updated.valid].mean()) if temporal_available else None
                ),
            }
        )

    def _refresh_cluster_frequency(self, source_model: TopoGateV11 | None = None) -> None:
        """Refresh assignment statistics independently of graph construction.

        NoMix, static-graph, and dynamic-graph variants must receive the same
        cluster-frequency updates; otherwise ``use_topology=false`` silently
        changes both topology and the DEC-style target correction.
        """
        if not self.cfg.use_cluster_head:
            return
        reference = source_model or (self.teacher if self.cfg.use_teacher else self.student)
        probabilities = self._full_probabilities(reference)
        frequency = probabilities.mean(axis=0)
        self.cluster_frequency = torch.as_tensor(
            frequency / np.clip(frequency.sum(), 1e-8, None),
            dtype=torch.float32,
            device=self.device,
        )

    def _mask_ratio(self, epoch: int) -> float:
        progress = (epoch - 1) / max(1, self.cfg.epochs - 1)
        return float(self.cfg.mask_ratio + progress * (self.cfg.mask_ratio_end - self.cfg.mask_ratio))

    def _ramp(self, epoch: int) -> float:
        if epoch <= self.cfg.warmup_epochs:
            return 0.0
        return min(1.0, (epoch - self.cfg.warmup_epochs) / max(1, self.cfg.ramp_epochs))

    @torch.no_grad()
    def _paired_reference_risk(
        self,
        anchor: torch.Tensor,
        probe: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        donor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return comparable anchor/probe losses for the risk gate.

        Both views share the precise feature intervention (mask and donor-row
        permutation), and both are decoded by the same deterministic reference
        network. Consequently an unchanged probe has exactly zero estimated
        improvement; this is a required null calibration, not a regularizer.
        """
        reference = self.teacher if self.cfg.use_teacher else self.student
        was_training = reference.training
        reference.eval()
        try:
            donor_indices = donor_indices.to(dtype=torch.long, device=anchor.device).view(-1)
            replacement = anchor[donor_indices]
            anchor_corrupt, _ = corrupt_batch(
                anchor, ratio=0.0, mask=mask, replacement=replacement
            )
            probe_corrupt, _ = corrupt_batch(
                probe, ratio=0.0, mask=mask, replacement=replacement
            )
            _, anchor_per, _ = reference.autoencoder.masked_loss(
                anchor_corrupt,
                target,
                mask,
                self.cfg.reconstruction_distribution,
                self.cfg.masked_data_weight,
                self.cfg.mask_prediction_weight,
                self.cfg.student_t_nu,
            )
            _, probe_per, _ = reference.autoencoder.masked_loss(
                probe_corrupt,
                target,
                mask,
                self.cfg.reconstruction_distribution,
                self.cfg.masked_data_weight,
                self.cfg.mask_prediction_weight,
                self.cfg.student_t_nu,
            )
        finally:
            reference.train(was_training)
        return anchor_per, probe_per

    @torch.no_grad()
    def _paired_reference_edge_risks(
        self,
        anchor: torch.Tensor,
        probes: torch.Tensor,
        target: torch.Tensor,
        clean_assignments: torch.Tensor,
        mask: torch.Tensor,
        donor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate each candidate under the exact same corruption values.

        ``probes[i, j]`` is a one-edge topology intervention for anchor ``i``.
        Its masked coordinates receive the *raw-anchor* donor values used by
        ``anchor[i]`` rather than values from another probe.  This makes the
        per-edge risk difference attributable to the topology intervention,
        not to a different random or view-dependent corruption.
        """
        if probes.ndim != 3 or probes.shape[0] != anchor.shape[0] or probes.shape[2] != anchor.shape[1]:
            raise ValueError("probes must be [batch, candidates, features]")
        if target.shape != anchor.shape or mask.shape != anchor.shape:
            raise ValueError("target and mask must match the anchor shape")
        if clean_assignments.shape != (anchor.shape[0], self.n_clusters):
            raise ValueError("clean_assignments must be [batch, n_clusters]")

        batch, candidates, width = probes.shape
        donor_indices = donor_indices.to(dtype=torch.long, device=anchor.device).view(-1)
        if donor_indices.numel() != batch:
            raise ValueError("donor_indices must contain one index per anchor")
        replacement = anchor[donor_indices]
        flat_probes = probes.reshape(batch * candidates, width)
        flat_mask = mask[:, None, :].expand(batch, candidates, width).reshape(batch * candidates, width)
        flat_replacement = replacement[:, None, :].expand(batch, candidates, width).reshape(
            batch * candidates, width
        )
        flat_target = target[:, None, :].expand(batch, candidates, width).reshape(batch * candidates, width)

        reference = self.teacher if self.cfg.use_teacher else self.student
        was_training = reference.training
        reference.eval()
        try:
            anchor_corrupt, _ = corrupt_batch(
                anchor, ratio=0.0, mask=mask, replacement=replacement
            )
            probe_corrupt, _ = corrupt_batch(
                flat_probes, ratio=0.0, mask=flat_mask, replacement=flat_replacement
            )
            anchor_z, anchor_reconstruction_risk, _ = reference.autoencoder.masked_loss(
                anchor_corrupt,
                target,
                mask,
                self.cfg.reconstruction_distribution,
                self.cfg.masked_data_weight,
                self.cfg.mask_prediction_weight,
                self.cfg.student_t_nu,
            )
            probe_z, probe_reconstruction_risk, _ = reference.autoencoder.masked_loss(
                probe_corrupt,
                flat_target,
                flat_mask,
                self.cfg.reconstruction_distribution,
                self.cfg.masked_data_weight,
                self.cfg.mask_prediction_weight,
                self.cfg.student_t_nu,
            )
            anchor_assignments = reference.cluster_head(anchor_z)
            probe_assignments = reference.cluster_head(probe_z).reshape(
                batch, candidates, self.n_clusters
            )
            anchor_cluster_risk = kl_per_sample(clean_assignments.detach(), anchor_assignments)
            flat_clean_assignments = clean_assignments[:, None, :].expand(
                batch, candidates, self.n_clusters
            ).reshape(batch * candidates, self.n_clusters)
            probe_cluster_risk = kl_per_sample(
                flat_clean_assignments,
                probe_assignments.reshape(batch * candidates, self.n_clusters),
            ).reshape(batch, candidates)
        finally:
            reference.train(was_training)
        return (
            anchor_reconstruction_risk,
            probe_reconstruction_risk.reshape(batch, candidates),
            anchor_cluster_risk,
            probe_cluster_risk,
        )

    def _graph_batch(
        self,
        batch_indices: torch.Tensor,
        x: torch.Tensor,
        z_clean: torch.Tensor,
        q_teacher: torch.Tensor,
        confidence: torch.Tensor,
        real_per: torch.Tensor,
        mask: torch.Tensor,
        donor_indices: torch.Tensor,
        ramp: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        graph = self.graph
        if graph is None:
            raise RuntimeError("topology batch requested before candidate graph construction")
        idx_np = batch_indices.cpu().numpy()
        neighbour_idx_np = graph.indices[idx_np]
        neighbour_idx = torch.as_tensor(neighbour_idx_np, dtype=torch.long)
        valid = torch.as_tensor(graph.valid[idx_np], dtype=torch.bool, device=self.device)
        recurrence = (
            self.edge_recurrence
            if self.edge_recurrence is not None
            else np.zeros_like(graph.raw_similarity, dtype=np.float32)
        )
        temporal_recurrence = torch.as_tensor(
            recurrence[idx_np], dtype=x.dtype, device=self.device
        )
        neighbour_x = self.X_cpu[neighbour_idx].to(self.device)
        batch, width, dim = neighbour_x.shape
        neighbour_flat = neighbour_x.reshape(batch * width, dim)
        z_neighbour = self.student.encode(neighbour_flat).reshape(batch, width, -1)

        with torch.no_grad():
            reference = self.teacher if self.cfg.use_teacher else self.student
            was_training = reference.training
            reference.eval()
            try:
                z_teacher_anchor = reference.encode(x)
                z_teacher_neighbour = reference.encode(neighbour_flat).reshape(batch, width, -1)
                q_teacher_neighbour = (
                    reference.cluster_head(z_teacher_neighbour.reshape(batch * width, -1)).reshape(
                        batch, width, self.n_clusters
                    )
                    if self.cfg.use_cluster_head
                    else None
                )
            finally:
                reference.train(was_training)

        raw_similarity = torch.as_tensor(
            graph.raw_similarity[idx_np], dtype=x.dtype, device=self.device
        )
        mutual = torch.as_tensor(graph.mutual[idx_np], dtype=x.dtype, device=self.device)
        snn = torch.as_tensor(graph.snn[idx_np], dtype=x.dtype, device=self.device)
        tda_prior = torch.as_tensor(
            graph.tda_prior[idx_np]
            if graph.tda_prior is not None
            else np.zeros_like(graph.raw_similarity[idx_np], dtype=np.float32),
            dtype=x.dtype,
            device=self.device,
        )
        latent_similarity = F.cosine_similarity(z_clean[:, None, :], z_neighbour, dim=2)
        latent_distance = (z_clean[:, None, :] - z_neighbour).square().sum(dim=2).sqrt()
        scale = latent_distance.detach().masked_fill(~valid, float("nan")).nanmedian(dim=1).values
        scale = torch.nan_to_num(scale, nan=1.0).clamp_min(1e-4)
        local_distance = latent_distance / scale[:, None]
        anchor_stability = F.cosine_similarity(z_clean, z_teacher_anchor, dim=1)
        neighbour_stability = F.cosine_similarity(z_neighbour, z_teacher_neighbour, dim=2)
        stability = 0.5 * (anchor_stability[:, None] + neighbour_stability)
        edge_features = torch.stack(
            [raw_similarity, latent_similarity, local_distance, mutual, snn, stability], dim=2
        )

        prior_score = (
            float(self.cfg.raw_prior_weight) * raw_similarity
            + float(self.cfg.latent_prior_weight) * latent_similarity.detach()
            + float(self.cfg.tda_prior_weight) * tda_prior
        ) / max(float(self.cfg.raw_prior_temperature), 1e-4)
        prior_score = prior_score.masked_fill(~valid, -1e9)
        raw_prior = F.softmax(prior_score, dim=1)
        if self.cfg.use_cluster_head and q_teacher_neighbour is not None:
            agreement = torch.sum(q_teacher[:, None, :] * q_teacher_neighbour, dim=2).clamp_min(1e-6)
        else:
            agreement = ((latent_similarity.detach() + 1.0) * 0.5).clamp_min(1e-6)
        edge_target = raw_prior * agreement
        edge_target = edge_target.masked_fill(~valid, 0.0)
        edge_target = edge_target / edge_target.sum(dim=1, keepdim=True).clamp_min(1e-8)

        entropy_norm = -torch.sum(q_teacher.clamp_min(1e-8) * torch.log(q_teacher.clamp_min(1e-8)), dim=1)
        entropy_norm = entropy_norm / max(math.log(max(self.n_clusters, 2)), 1e-8)
        edge_entropy = -torch.sum(raw_prior.clamp_min(1e-8) * torch.log(raw_prior.clamp_min(1e-8)), dim=1)
        edge_entropy = edge_entropy / max(math.log(max(width, 2)), 1e-8)
        local_agreement = torch.sum(raw_prior * agreement, dim=1)
        node_features = torch.stack(
            [confidence, entropy_norm, raw_prior.max(dim=1).values, edge_entropy, local_agreement], dim=1
        )
        mixture = self.student.topology(
            edge_features,
            node_features,
            valid,
            ramp=ramp,
            temperature=self.cfg.edge_temperature,
            use_edge_reliability=self.cfg.use_edge_reliability,
        )

        if self.cfg.topology_path == "input_mix":
            # Keep the two interventions identifiable. The clean mixed
            # embedding measures topology alone, whereas the masked mixed view
            # trains denoising reconstruction.
            mixed = mixture[:, :1] * x + torch.sum(mixture[:, 1:, None] * neighbour_x, dim=1)
            mixed_z_clean = self.student.encode(mixed)
            mixed_corrupt, _ = corrupt_batch(
                mixed, ratio=0.0, mask=mask, donor_indices=donor_indices
            )
            _, mixed_per, _ = self.student.autoencoder.masked_loss(
                mixed_corrupt,
                x,
                mask,
                self.cfg.reconstruction_distribution,
                self.cfg.masked_data_weight,
                self.cfg.mask_prediction_weight,
                self.cfg.student_t_nu,
            )
        else:
            # The residual path never feeds a neighbour-interpolated vector to
            # the decoder. It preserves the topology gate as a selector of a
            # trusted assignment residual rather than as an input corruption.
            mixed_z_clean = z_clean
            mixed_per = torch.zeros_like(real_per)

        zero_node = torch.zeros_like(confidence)
        zero_edge = torch.zeros_like(edge_target)
        risk_anchor_per = torch.zeros_like(real_per)
        probe_per = torch.zeros_like(real_per)
        anchor_cluster_risk = torch.zeros_like(real_per)
        probe_cluster_risk = torch.zeros_like(real_per)
        reconstruction_help = zero_edge
        cluster_help = zero_edge
        reconstruction_improvement = zero_edge
        cluster_improvement = zero_edge
        assignment_edge_target = edge_target.detach()
        target_mixture: torch.Tensor | None = None

        if self.cfg.gate_target_source == "paired_risk":
            # Historical V11 ablation: one probe is induced by the current
            # learned edge conditional, so it is intentionally not used by the
            # new counterfactual-semantic method.
            edge_conditional = mixture[:, 1:] / mixture[:, 1:].sum(dim=1, keepdim=True).clamp_min(1e-8)
            neighbour_mean = torch.sum(edge_conditional[:, :, None] * neighbour_x, dim=1)
            probe = x + float(self.cfg.graph_probe_strength) * (neighbour_mean - x)
            with torch.no_grad():
                if self.cfg.risk_target_mode == "paired_ema_eval":
                    risk_anchor_per, probe_per = self._paired_reference_risk(
                        x, probe, x, mask, donor_indices
                    )
                else:
                    # Compatibility-only ablation for the pre-paired-risk V11.
                    risk_anchor_per = real_per.detach()
                    replacement = x[donor_indices.to(dtype=torch.long, device=x.device)]
                    probe_corrupt, _ = corrupt_batch(
                        probe, ratio=0.0, mask=mask, replacement=replacement
                    )
                    _, probe_per, _ = self.student.autoencoder.masked_loss(
                        probe_corrupt,
                        x,
                        mask,
                        self.cfg.reconstruction_distribution,
                        self.cfg.masked_data_weight,
                        self.cfg.mask_prediction_weight,
                        self.cfg.student_t_nu,
                    )
                if self.cfg.use_cluster_head:
                    reference = self.teacher if self.cfg.use_teacher else self.student
                    was_training = reference.training
                    reference.eval()
                    try:
                        q_probe = reference.cluster_head(reference.encode(probe))
                    finally:
                        reference.train(was_training)
                    probe_agreement = torch.sum(q_teacher * q_probe, dim=1)
                else:
                    probe_agreement = torch.ones_like(confidence)
            improvement = F.relu(risk_anchor_per.detach() - probe_per.detach())
            risk_help = 1.0 - torch.exp(
                -improvement / max(float(self.cfg.gate_risk_temperature), 1e-4)
            )
            topology_help = (
                risk_help
                * confidence.detach()
                * local_agreement.detach()
                * probe_agreement.detach()
            ).clamp(0.0, 1.0)
            gate_evidence = risk_help

        elif self.cfg.gate_target_source == "counterfactual_semantic":
            # Candidate-wise intervention is defined without the learned gate:
            # every edge receives the same alpha-strength probe before the
            # teacher evaluates its paired reconstruction and assignment risk.
            edge_probes = x[:, None, :] + float(self.cfg.graph_probe_strength) * (
                neighbour_x - x[:, None, :]
            )
            with torch.no_grad():
                (
                    risk_anchor_per,
                    edge_probe_risk,
                    anchor_cluster_risk,
                    edge_cluster_risk,
                ) = self._paired_reference_edge_risks(
                    x,
                    edge_probes,
                    x,
                    q_teacher,
                    mask,
                    donor_indices,
                )
                counterfactual = counterfactual_semantic_target(
                    risk_anchor_per,
                    edge_probe_risk,
                    anchor_cluster_risk,
                    edge_cluster_risk,
                    edge_target,
                    confidence,
                    self.cfg.gate_risk_temperature,
                    self.cfg.gate_cluster_risk_temperature,
                    self.cfg.semantic_help_combiner,
                )
            topology_help = counterfactual.topology_help
            target_mixture = counterfactual.target_mixture
            assignment_edge_target = counterfactual.edge_conditional
            reconstruction_help = counterfactual.reconstruction_help
            cluster_help = counterfactual.cluster_help
            reconstruction_improvement = counterfactual.reconstruction_improvement
            cluster_improvement = counterfactual.cluster_improvement
            joint_edge_help = torch.sqrt((reconstruction_help * cluster_help).clamp_min(0.0))
            gate_evidence = torch.sum(edge_target.detach() * joint_edge_help, dim=1)
            risk_help = gate_evidence
            improvement = torch.sum(edge_target.detach() * reconstruction_improvement, dim=1)
            probe_per = torch.sum(edge_target.detach() * edge_probe_risk, dim=1)
            probe_cluster_risk = torch.sum(edge_target.detach() * edge_cluster_risk, dim=1)

        else:
            # A separate graph-refresh event supplies the supervision. Current
            # gate features do not contain this temporal recurrence, avoiding
            # a self-confirming target.
            temporal_score = raw_prior * agreement.detach() * temporal_recurrence
            temporal_score = temporal_score.masked_fill(~valid, 0.0)
            support = temporal_score.sum(dim=1)
            if self.temporal_target_available:
                topology_help = (
                    float(self.cfg.temporal_gate_max) * confidence.detach() * support
                ).clamp(min=0.0, max=float(self.cfg.temporal_gate_max))
                assignment_edge_target = temporal_score / support[:, None].clamp_min(1e-8)
            else:
                topology_help = torch.zeros_like(confidence)
            risk_help = zero_node
            improvement = zero_node
            gate_evidence = support

        if target_mixture is None:
            target_mixture = torch.cat(
                [
                    (1.0 - topology_help)[:, None],
                    topology_help[:, None] * assignment_edge_target.detach(),
                ],
                dim=1,
            )
        if q_teacher_neighbour is None:
            topology_assignment_target = q_teacher.detach()
        else:
            # This is a conditional trusted-neighbour teacher target. The
            # total topology mass is returned separately, so a null gate adds
            # exactly zero residual loss rather than another self-distillation
            # term.
            topology_assignment_target = torch.sum(
                assignment_edge_target[:, :, None].detach() * q_teacher_neighbour.detach(), dim=1
            )
        graph_per = kl_per_sample(target_mixture, mixture)
        graph_loss = graph_per.mean()
        edge_consistency_loss = trusted_edge_alignment(
            z_clean,
            z_teacher_neighbour,
            assignment_edge_target.detach(),
            topology_help,
            valid=valid,
            temperature=self.cfg.edge_alignment_temperature,
        )
        diagnostics = {
            "mean_topology_gate": float((1.0 - mixture[:, 0]).detach().mean().cpu()),
            "mean_target_topology_gate": float(topology_help.detach().mean().cpu()),
            "mean_gate_evidence": float(gate_evidence.detach().mean().cpu()),
            "mean_risk_help": float(risk_help.detach().mean().cpu()),
            "mean_risk_improvement": float(improvement.detach().mean().cpu()),
            "mean_reconstruction_help": float(reconstruction_help.detach().mean().cpu()),
            "mean_cluster_help": float(cluster_help.detach().mean().cpu()),
            "mean_reconstruction_improvement": float(
                reconstruction_improvement.detach().mean().cpu()
            ),
            "mean_cluster_improvement": float(cluster_improvement.detach().mean().cpu()),
            "mean_reference_anchor_risk": float(risk_anchor_per.detach().mean().cpu()),
            "mean_reference_probe_risk": float(probe_per.detach().mean().cpu()),
            "mean_reference_anchor_cluster_risk": float(
                anchor_cluster_risk.detach().mean().cpu()
            ),
            "mean_reference_probe_cluster_risk": float(
                probe_cluster_risk.detach().mean().cpu()
            ),
            "mean_temporal_recurrence": float(
                temporal_recurrence[valid].detach().mean().cpu()
            ) if self.temporal_target_available and torch.any(valid) else 0.0,
            "mean_edge_entropy": float(edge_entropy.detach().mean().cpu()),
            "mean_tda_prior": float(tda_prior[valid].detach().mean().cpu())
            if torch.any(valid)
            else 0.0,
            "mean_teacher_agreement": float(local_agreement.detach().mean().cpu()),
            "mean_edge_consistency": float(edge_consistency_loss.detach().cpu()),
        }
        return (
            mixed_z_clean,
            mixed_per,
            topology_assignment_target,
            topology_help.detach(),
            graph_loss,
            edge_consistency_loss,
            diagnostics,
        )

    def fit(self) -> TrainingResult:
        started = time.time()
        clusters_initialised = False
        for epoch in range(1, self.cfg.epochs + 1):
            if not clusters_initialised and epoch > self.cfg.warmup_epochs:
                self._initialise_clusters()
                if self.cfg.use_topology:
                    self._refresh_graph(epoch)
                clusters_initialised = True
            elif clusters_initialised:
                refresh_due = (
                    self.cfg.graph_refresh_interval > 0
                    and (epoch - self.cfg.warmup_epochs - 1) % self.cfg.graph_refresh_interval == 0
                    and epoch > self.cfg.warmup_epochs + 1
                )
                if refresh_due:
                    if self.cfg.use_topology and self.cfg.use_dynamic_graph:
                        self._refresh_graph(epoch)
                    else:
                        self._refresh_cluster_frequency()

            ramp = self._ramp(epoch)
            mask_ratio = self._mask_ratio(epoch)
            self.student.train()
            accum = {
                "loss": 0.0,
                "rec": 0.0,
                "real_rec": 0.0,
                "mixed_rec": 0.0,
                "cls": 0.0,
                "clean_cls": 0.0,
                "mixed_cls": 0.0,
                "topology_cls": 0.0,
                "prior_cls": 0.0,
                "selected_fraction": 0.0,
                "graph": 0.0,
                "edge_consistency": 0.0,
                "tda_prior": 0.0,
                "gate": 0.0,
                "target_gate": 0.0,
                "gate_evidence": 0.0,
                "temporal_recurrence": 0.0,
                "risk_help": 0.0,
                "risk_improvement": 0.0,
                "reconstruction_help": 0.0,
                "cluster_help": 0.0,
                "reconstruction_improvement": 0.0,
                "cluster_improvement": 0.0,
                "reference_anchor_risk": 0.0,
                "reference_probe_risk": 0.0,
                "reference_anchor_cluster_risk": 0.0,
                "reference_probe_cluster_risk": 0.0,
            }
            batches = 0
            for batch_indices, x_cpu in self._loader(shuffle=True):
                x = x_cpu.to(self.device)
                if x.shape[0] <= 1:
                    donor_indices = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
                else:
                    donor_indices = torch.randperm(x.shape[0], device=self.device)
                corrupted, mask = corrupt_batch(
                    x, mask_ratio, donor_indices=donor_indices
                )
                _, real_per, _ = self.student.autoencoder.masked_loss(
                    corrupted,
                    x,
                    mask,
                    self.cfg.reconstruction_distribution,
                    self.cfg.masked_data_weight,
                    self.cfg.mask_prediction_weight,
                    self.cfg.student_t_nu,
                )
                z_clean = self.student.encode(x)
                real_rec_loss = real_per.mean()
                mixed_rec_loss = torch.zeros((), device=self.device)
                rec_loss = real_rec_loss
                cluster_loss = torch.zeros((), device=self.device)
                graph_loss = torch.zeros((), device=self.device)
                edge_consistency_loss = torch.zeros((), device=self.device)
                mixed_per = torch.zeros_like(real_per)
                clean_consistency = torch.zeros((), device=self.device)
                mixed_consistency = torch.zeros((), device=self.device)
                topology_consistency = torch.zeros((), device=self.device)
                prior_loss = torch.zeros((), device=self.device)
                topology_assignment_target: torch.Tensor | None = None
                topology_residual_mass: torch.Tensor | None = None
                selected_fraction = 0.0
                gate_mean = 0.0
                target_gate_mean = 0.0
                tda_prior_mean = 0.0
                gate_evidence_mean = 0.0
                temporal_recurrence_mean = 0.0
                risk_help_mean = 0.0
                risk_improvement_mean = 0.0
                reconstruction_help_mean = 0.0
                cluster_help_mean = 0.0
                reconstruction_improvement_mean = 0.0
                cluster_improvement_mean = 0.0
                reference_anchor_risk_mean = 0.0
                reference_probe_risk_mean = 0.0
                reference_anchor_cluster_risk_mean = 0.0
                reference_probe_cluster_risk_mean = 0.0

                if clusters_initialised:
                    with torch.no_grad():
                        reference = self.teacher if self.cfg.use_teacher else self.student
                        was_training = reference.training
                        reference.eval()
                        try:
                            q_teacher = reference.cluster_head(reference.encode(x))
                            confidence = q_teacher.max(dim=1).values
                        finally:
                            reference.train(was_training)
                    threshold = max(
                        float(self.cfg.confidence_threshold),
                        float(torch.quantile(confidence, float(self.cfg.confidence_quantile)).cpu()),
                    )
                    selected = confidence >= threshold
                    if not torch.any(selected):
                        selected[torch.argmax(confidence)] = True
                    selected_fraction = float(selected.to(torch.float32).mean().detach().cpu())

                    if self.cfg.use_topology:
                        (
                            mixed_z,
                            mixed_per,
                            topology_assignment_target,
                            topology_residual_mass,
                            graph_loss,
                            edge_consistency_loss,
                            graph_diag,
                        ) = self._graph_batch(
                            batch_indices,
                            x,
                            z_clean,
                            q_teacher,
                            confidence,
                            real_per,
                            mask,
                            donor_indices,
                            ramp,
                        )
                        gate_mean = graph_diag["mean_topology_gate"]
                        target_gate_mean = graph_diag["mean_target_topology_gate"]
                        tda_prior_mean = graph_diag["mean_tda_prior"]
                        gate_evidence_mean = graph_diag["mean_gate_evidence"]
                        temporal_recurrence_mean = graph_diag["mean_temporal_recurrence"]
                        risk_help_mean = graph_diag["mean_risk_help"]
                        risk_improvement_mean = graph_diag["mean_risk_improvement"]
                        reconstruction_help_mean = graph_diag["mean_reconstruction_help"]
                        cluster_help_mean = graph_diag["mean_cluster_help"]
                        reconstruction_improvement_mean = graph_diag[
                            "mean_reconstruction_improvement"
                        ]
                        cluster_improvement_mean = graph_diag["mean_cluster_improvement"]
                        reference_anchor_risk_mean = graph_diag["mean_reference_anchor_risk"]
                        reference_probe_risk_mean = graph_diag["mean_reference_probe_risk"]
                        reference_anchor_cluster_risk_mean = graph_diag[
                            "mean_reference_anchor_cluster_risk"
                        ]
                        reference_probe_cluster_risk_mean = graph_diag[
                            "mean_reference_probe_cluster_risk"
                        ]
                        if self.cfg.topology_path == "input_mix" and self.cfg.use_mixed_reconstruction:
                            mixed_rec_loss = mixed_per.mean()
                            rec_loss = rec_loss + ramp * float(self.cfg.mix_reconstruction_weight) * mixed_rec_loss
                    else:
                        mixed_z = z_clean

                    if self.cfg.use_cluster_head:
                        q_student = self.student.cluster_head(z_clean)
                        target = sharpen_assignments(
                            q_teacher.detach(), self.cluster_frequency, self.cfg.cluster_temperature
                        )
                        clean_kl = kl_per_sample(target, q_student)
                        if self.cfg.use_topology:
                            if self.cfg.topology_path == "input_mix":
                                mixed_q = self.student.cluster_head(mixed_z)
                                topology_kl = kl_per_sample(target, mixed_q)
                                topology_weight = float(self.cfg.mixed_cluster_weight)
                            else:
                                if topology_assignment_target is None:
                                    raise RuntimeError("residual topology target was not constructed")
                                if topology_residual_mass is None:
                                    raise RuntimeError("residual topology mass was not constructed")
                                topology_kl = topology_residual_mass * kl_per_sample(
                                    topology_assignment_target.detach(), q_student
                                )
                                topology_weight = float(self.cfg.residual_assignment_weight)
                        else:
                            topology_kl = torch.zeros_like(clean_kl)
                            topology_weight = 0.0
                        selected_weight = selected.to(clean_kl.dtype)
                        selected_count = selected_weight.sum().clamp_min(1.0)
                        clean_consistency = (clean_kl * selected_weight).sum() / selected_count
                        topology_consistency = (
                            topology_kl * selected_weight
                        ).sum() / selected_count
                        if self.cfg.topology_path == "input_mix":
                            mixed_consistency = topology_consistency
                        consistency = (
                            float(self.cfg.clean_cluster_weight) * clean_consistency
                            + topology_weight * topology_consistency
                        )
                        prior_loss = self.student.cluster_head.mixture_prior_loss(
                            q_student, self.cfg.dirichlet_strength
                        )
                        cluster_loss = consistency + prior_loss

                total = rec_loss
                if clusters_initialised and self.cfg.use_cluster_head:
                    total = total + ramp * float(self.cfg.cluster_weight) * cluster_loss
                if clusters_initialised and self.cfg.use_topology and self.cfg.use_graph_prior:
                    total = total + ramp * float(self.cfg.graph_weight) * graph_loss
                if clusters_initialised and self.cfg.use_topology and self.cfg.use_edge_consistency:
                    total = total + ramp * float(self.cfg.edge_consistency_weight) * edge_consistency_loss

                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.gradient_clip)
                self.optimizer.step()
                if self.cfg.use_teacher:
                    ema_update(self.teacher, self.student, self.cfg.ema_decay)

                accum["loss"] += float(total.detach().cpu())
                accum["rec"] += float(rec_loss.detach().cpu())
                accum["real_rec"] += float(real_rec_loss.detach().cpu())
                accum["mixed_rec"] += float(mixed_rec_loss.detach().cpu())
                accum["cls"] += float(cluster_loss.detach().cpu())
                accum["clean_cls"] += float(clean_consistency.detach().cpu())
                accum["mixed_cls"] += float(mixed_consistency.detach().cpu())
                accum["topology_cls"] += float(topology_consistency.detach().cpu())
                accum["prior_cls"] += float(prior_loss.detach().cpu())
                accum["selected_fraction"] += float(selected_fraction)
                accum["graph"] += float(graph_loss.detach().cpu())
                accum["edge_consistency"] += float(edge_consistency_loss.detach().cpu())
                accum["tda_prior"] += float(tda_prior_mean)
                accum["gate"] += float(gate_mean)
                accum["target_gate"] += float(target_gate_mean)
                accum["gate_evidence"] += float(gate_evidence_mean)
                accum["temporal_recurrence"] += float(temporal_recurrence_mean)
                accum["risk_help"] += float(risk_help_mean)
                accum["risk_improvement"] += float(risk_improvement_mean)
                accum["reconstruction_help"] += float(reconstruction_help_mean)
                accum["cluster_help"] += float(cluster_help_mean)
                accum["reconstruction_improvement"] += float(reconstruction_improvement_mean)
                accum["cluster_improvement"] += float(cluster_improvement_mean)
                accum["reference_anchor_risk"] += float(reference_anchor_risk_mean)
                accum["reference_probe_risk"] += float(reference_probe_risk_mean)
                accum["reference_anchor_cluster_risk"] += float(
                    reference_anchor_cluster_risk_mean
                )
                accum["reference_probe_cluster_risk"] += float(
                    reference_probe_cluster_risk_mean
                )
                batches += 1

            row = {
                "epoch": epoch,
                "ramp": float(ramp),
                "mask_ratio": float(mask_ratio),
                **{key: value / max(1, batches) for key, value in accum.items()},
            }
            self.history.append(row)
            if epoch == 1 or epoch == self.cfg.epochs or epoch % 10 == 0:
                topology_weight_log = (
                    float(self.cfg.mixed_cluster_weight)
                    if self.cfg.topology_path == "input_mix"
                    else float(self.cfg.residual_assignment_weight)
                )
                print(
                    f"[V11] epoch {epoch:03d}/{self.cfg.epochs} "
                    f"loss={row['loss']:.4f} rec={row['rec']:.4f} "
                    f"cls={row['cls']:.4f}[{self.cfg.clean_cluster_weight:.2f}*"
                    f"{row['clean_cls']:.3f}+{topology_weight_log:.2f}*"
                    f"{row['topology_cls']:.3f}] graph={row['graph']:.4f} "
                    f"edge={row['edge_consistency']:.4f} "
                    f"gate={row['gate']:.3f}/{row['target_gate']:.3f} "
                    f"temporal={row['temporal_recurrence']:.3f}",
                    flush=True,
                )

        if not clusters_initialised:
            self._initialise_clusters()
            if self.cfg.use_topology:
                self._refresh_graph(self.cfg.epochs)
        source_model = self.teacher if self.cfg.use_teacher else self.student
        embedding = self._full_embeddings(source_model)
        probabilities = self._full_probabilities(source_model)
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)
        return TrainingResult(
            embedding=embedding,
            probabilities=probabilities,
            predictions=predictions,
            history=self.history,
            graph_history=self.graph_history,
            train_seconds=float(time.time() - started),
        )
