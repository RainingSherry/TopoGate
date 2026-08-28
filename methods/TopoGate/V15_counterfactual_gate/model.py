from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Differentiable sparse probability projection with exact zeros."""
    if logits.numel() == 0:
        return logits
    z = logits.transpose(dim, -1)
    original_shape = z.shape
    z = z.reshape(-1, z.shape[-1])
    zs = torch.sort(z, dim=1, descending=True).values
    cssv = torch.cumsum(zs, dim=1) - 1.0
    positions = torch.arange(1, zs.shape[1] + 1, device=z.device, dtype=z.dtype)[None, :]
    support = zs - cssv / positions > 0
    support_count = support.sum(dim=1).clamp_min(1).long()
    tau = cssv.gather(1, support_count[:, None] - 1) / support_count[:, None].to(dtype=z.dtype)
    output = torch.relu(z - tau)
    return output.reshape(original_shape).transpose(dim, -1)


def abstaining_sparsemax(
    edge_scores: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Project edge scores onto one null branch plus candidate edges.

    The null branch has score zero. Rows with no strictly positive valid edge
    are explicitly assigned ``[1, 0, ..., 0]``; vanilla sparsemax would spread
    mass over tied zero-valued edges and would therefore violate abstention.
    """
    if dim != -1:
        raise ValueError("abstaining_sparsemax currently expects the last dimension")
    if edge_scores.ndim < 1:
        raise ValueError("edge_scores must have at least one dimension")
    if valid is None:
        valid = torch.ones_like(edge_scores, dtype=torch.bool)
    if valid.shape != edge_scores.shape:
        raise ValueError("valid must have the same shape as edge_scores")
    # Negative utility is an abstention signal, not a candidate to be
    # averaged with a positive edge. Restrict the sparsemax support to
    # strictly positive valid utilities.
    active = valid & (edge_scores > 0.0)
    masked = edge_scores.masked_fill(~active, -1e9)
    scores = torch.cat([torch.zeros_like(masked[..., :1]), masked], dim=-1)
    projected = sparsemax(scores, dim=-1)
    has_positive = (valid & (edge_scores > 0.0)).any(dim=-1)
    null_only = torch.zeros_like(projected)
    null_only[..., 0] = 1.0
    return torch.where(has_positive[..., None], projected, null_only)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class V15AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.mask_predictor = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z), self.mask_predictor(z)


class StudentTHead(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        latent_dim: int,
        nu: float = 4.0,
        normalize_latent: bool = True,
        cosine_temperature: float = 0.1,
    ):
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.latent_dim = int(latent_dim)
        self.nu = float(nu)
        self.normalize_latent = bool(normalize_latent)
        self.cosine_temperature = float(cosine_temperature)
        self.centres = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.02)
        self.log_scales = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.prior_logits = nn.Parameter(torch.zeros(n_clusters))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(z), dim=1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        values = F.normalize(z, dim=1) if self.normalize_latent else z
        centres = F.normalize(self.centres, dim=1) if self.normalize_latent else self.centres
        if self.normalize_latent:
            distance = (1.0 - values @ centres.transpose(0, 1)).clamp_min(0.0)
            distance = distance / float(self.cosine_temperature)
        else:
            scales = F.softplus(self.log_scales).clamp_min(1e-3)
            delta = (values[:, None, :] - centres[None, :, :]) / scales[None, :, :]
            distance = delta.square().mean(dim=2)
        logits = -0.5 * (self.nu + 1.0) * torch.log1p(distance / max(self.nu, 1e-3))
        logits = logits + F.log_softmax(self.prior_logits, dim=0)[None, :]
        return logits

    @torch.no_grad()
    def initialise(self, embeddings: np.ndarray, seed: int, n_init: int = 10) -> None:
        if self.normalize_latent:
            embeddings = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
        km = KMeans(n_clusters=self.n_clusters, n_init=n_init, random_state=seed)
        labels = km.fit_predict(np.asarray(embeddings, dtype=np.float32))
        centres = torch.as_tensor(km.cluster_centers_, dtype=self.centres.dtype, device=self.centres.device)
        global_scale = np.std(embeddings, axis=0) + 1e-2
        scales: list[np.ndarray] = []
        counts: list[int] = []
        for cluster in range(self.n_clusters):
            members = embeddings[labels == cluster]
            counts.append(max(1, int(members.shape[0])))
            scales.append(np.std(members, axis=0) + 1e-2 if members.shape[0] >= 2 else global_scale)
        scales_t = torch.as_tensor(np.stack(scales), dtype=self.centres.dtype, device=self.centres.device)
        self.centres.copy_(centres)
        self.log_scales.copy_(torch.log(torch.expm1(scales_t.clamp(1e-2, 20.0))))
        prior = torch.as_tensor(counts, dtype=self.centres.dtype, device=self.centres.device)
        self.prior_logits.copy_(torch.log(prior / prior.sum()))

    def prior_loss(
        self,
        q: torch.Tensor,
        strength: float,
        frequency_target: torch.Tensor | None = None,
        frequency_weight: float = 0.0,
    ) -> torch.Tensor:
        prior = F.softmax(self.prior_logits, dim=0).clamp_min(1e-8)
        marginal = q.mean(dim=0).clamp_min(1e-8)
        divergence = torch.sum(marginal * (torch.log(marginal) - torch.log(prior)))
        correction = torch.zeros((), dtype=q.dtype, device=q.device)
        if frequency_target is not None and frequency_weight > 0.0:
            target = frequency_target.detach().to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
            target = target / target.sum().clamp_min(1e-8)
            correction = float(frequency_weight) * torch.sum(target * (torch.log(target) - torch.log(marginal)))
        return divergence - float(strength) * torch.log(prior).sum() + correction


class SphericalPrototypeHead(nn.Module):
    """A fixed-temperature spherical prototype assignment head.

    The head deliberately has no learnable class prior or per-cluster scale.
    Those degrees of freedom let a weak teacher explain away poor geometry by
    becoming a nearly uniform (or single-cluster) assignment.  Confidence is
    controlled only by the normalized prototype geometry and a fixed
    temperature; the marginal-balance term is kept separate in ``prior_loss``.
    """

    def __init__(
        self,
        n_clusters: int,
        latent_dim: int,
        temperature: float = 0.1,
        separation_weight: float = 0.1,
        separation_margin: float = 0.0,
    ):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.n_clusters = int(n_clusters)
        self.latent_dim = int(latent_dim)
        self.temperature = float(temperature)
        self.separation_weight = float(separation_weight)
        self.separation_margin = float(separation_margin)
        self.centres = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.02)
        self.register_buffer("latent_mean", torch.zeros(latent_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(z), dim=1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        # The MAE latent often carries a large common offset. Normalising
        # before removing it makes every sample point in nearly the same
        # direction and destroys spherical prototype geometry.
        values = F.normalize(z - self.latent_mean[None, :], dim=1)
        centres = F.normalize(self.centres, dim=1)
        logits = values @ centres.transpose(0, 1)
        return logits / self.temperature

    @torch.no_grad()
    def initialise(self, embeddings: np.ndarray, seed: int, n_init: int = 10) -> None:
        values = np.asarray(embeddings, dtype=np.float32)
        mean = np.nan_to_num(values.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        self.latent_mean.copy_(torch.as_tensor(mean, dtype=self.latent_mean.dtype, device=self.latent_mean.device))
        values = values - mean[None, :]
        values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        km = KMeans(n_clusters=self.n_clusters, n_init=n_init, random_state=seed)
        km.fit(values)
        centres = np.asarray(km.cluster_centers_, dtype=np.float32)
        centres /= np.clip(np.linalg.norm(centres, axis=1, keepdims=True), 1e-8, None)
        self.centres.copy_(torch.as_tensor(centres, dtype=self.centres.dtype, device=self.centres.device))

    def prior_loss(
        self,
        q: torch.Tensor,
        strength: float,
        frequency_target: torch.Tensor | None = None,
        frequency_weight: float = 0.0,
    ) -> torch.Tensor:
        """Minimize conditional assignment entropy while balancing prototypes."""
        q = q.clamp_min(1e-8)
        marginal = q.mean(dim=0).clamp_min(1e-8)
        conditional_entropy = -(q * torch.log(q)).sum(dim=1).mean()
        marginal_entropy = -(marginal * torch.log(marginal)).sum()
        information_loss = conditional_entropy - marginal_entropy

        if frequency_target is None:
            target = torch.full_like(marginal, 1.0 / self.n_clusters)
        else:
            target = frequency_target.detach().to(device=q.device, dtype=q.dtype).clamp_min(1e-8)
            target = target / target.sum().clamp_min(1e-8)
        balance = torch.sum(target * (torch.log(target) - torch.log(marginal)))

        centres = F.normalize(self.centres, dim=1)
        similarity = centres @ centres.transpose(0, 1)
        off_diagonal = ~torch.eye(self.n_clusters, dtype=torch.bool, device=q.device)
        separation = F.relu(similarity[off_diagonal] - self.separation_margin).square().mean()
        # A full-strength mutual-information objective forces an approximately
        # uniform cluster marginal.  That is incompatible with naturally
        # imbalanced sparse corpora (for example spam or rare-cell data) and
        # is unnecessary once the detached teacher supplies the semantic
        # assignment target.  ``strength`` is therefore the actual weight of
        # this optional anti-collapse regularizer.
        return (
            float(strength) * information_loss
            + float(frequency_weight) * balance
            + self.separation_weight * separation
        )


class UtilityScorer(nn.Module):
    def __init__(self, feature_dim: int = 6, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class V15Model(nn.Module):
    """Anchor MAE, configurable cluster head, and one edge utility scorer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_clusters: int,
        dropout: float,
        student_t_nu: float,
        cluster_normalize_latent: bool,
        cluster_cosine_temperature: float,
        cluster_head: str = "spherical_prototype",
        prototype_separation_weight: float = 0.1,
        prototype_separation_margin: float = 0.0,
    ):
        super().__init__()
        self.autoencoder = V15AutoEncoder(input_dim, hidden_dim, latent_dim, dropout)
        if cluster_head == "spherical_prototype":
            self.cluster_head = SphericalPrototypeHead(
                n_clusters,
                latent_dim,
                temperature=cluster_cosine_temperature,
                separation_weight=prototype_separation_weight,
                separation_margin=prototype_separation_margin,
            )
        elif cluster_head == "student_t":
            self.cluster_head = StudentTHead(
                n_clusters,
                latent_dim,
                student_t_nu,
                normalize_latent=cluster_normalize_latent,
                cosine_temperature=cluster_cosine_temperature,
            )
        else:
            raise ValueError(f"unknown cluster_head: {cluster_head}")
        # The scorer sees retrieval/semantic-support features only. Detached
        # utility components stay targets, preventing target leakage through
        # the feature tensor.
        self.utility_scorer = UtilityScorer(6, hidden_dim=min(32, hidden_dim), dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def assignments(self, z: torch.Tensor) -> torch.Tensor:
        return self.cluster_head(z)

    def assignment_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cluster_head.logits(z)

    def utility(self, features: torch.Tensor) -> torch.Tensor:
        return self.utility_scorer(features)

    def mix_latent(
        self,
        z_self: torch.Tensor,
        donors: torch.Tensor,
        edge_mass: torch.Tensor,
        alpha: float | torch.Tensor,
    ) -> torch.Tensor:
        if donors.shape[1] == 0:
            return z_self
        delta = donors - z_self[:, None, :]
        if isinstance(alpha, torch.Tensor):
            alpha_view = alpha.to(device=z_self.device, dtype=z_self.dtype).reshape(-1, 1, 1)
        else:
            alpha_view = float(alpha)
        return z_self + alpha_view * torch.sum(edge_mass[:, :, None] * delta, dim=1)

    def mix_assignments(
        self,
        z_self: torch.Tensor,
        donors: torch.Tensor,
        edge_mass: torch.Tensor,
        alpha: float | torch.Tensor,
    ) -> torch.Tensor:
        """Transport topology in prototype-logit space, staying on-manifold."""
        self_logits = self.assignment_logits(z_self)
        if donors.shape[1] == 0:
            return F.softmax(self_logits, dim=1)
        donor_logits = self.assignment_logits(donors.reshape(-1, donors.shape[-1])).reshape(
            donors.shape[0], donors.shape[1], -1
        )
        if isinstance(alpha, torch.Tensor):
            alpha_view = alpha.to(device=z_self.device, dtype=z_self.dtype).reshape(-1, 1)
        else:
            alpha_view = float(alpha)
        mixed = self_logits + alpha_view * torch.sum(
            edge_mass[:, :, None] * (donor_logits - self_logits[:, None, :]),
            dim=1,
        )
        return F.softmax(mixed, dim=1)

    def mix_probabilities(
        self,
        q_self: torch.Tensor,
        q_donor: torch.Tensor,
        edge_mass: torch.Tensor,
        alpha: float | torch.Tensor,
    ) -> torch.Tensor:
        """Transport detached donor assignments on the probability simplex.

        This branch avoids moving sparse high-dimensional latent coordinates or
        re-evaluating teacher donors through a potentially mismatched student
        head. ``alpha`` is bounded by configuration, so the self residual
        remains non-negative and the result is a valid convex assignment.
        """
        if q_self.ndim != 2 or q_donor.ndim != 3 or edge_mass.ndim != 2:
            raise ValueError("q_self, q_donor and edge_mass must be [B,K], [B,M,K], [B,M]")
        if q_donor.shape[:2] != edge_mass.shape or q_donor.shape[0] != q_self.shape[0]:
            raise ValueError("probability transport shapes are inconsistent")
        if isinstance(alpha, torch.Tensor):
            alpha_view = alpha.to(device=q_self.device, dtype=q_self.dtype).reshape(-1, 1)
        else:
            alpha_view = torch.as_tensor(float(alpha), device=q_self.device, dtype=q_self.dtype)
        mass = edge_mass.sum(dim=1, keepdim=True)
        transported = torch.sum(edge_mass[:, :, None] * q_donor, dim=1)
        mixed = (1.0 - alpha_view * mass) * q_self + alpha_view * transported
        return mixed.clamp_min(1e-8) / mixed.clamp_min(1e-8).sum(dim=1, keepdim=True)

    @staticmethod
    def mix_assignment_output(
        q_self: torch.Tensor,
        q_edge: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        """Mix counterfactual assignments with an exact null/self branch.

        ``pi[:, 0]`` is the abstention mass. Unlike latent donor transport,
        this operation cannot move a sample outside the cluster-assignment
        simplex and makes the null semantics explicit in the primary output.
        """
        if q_self.ndim != 2 or q_edge.ndim != 3 or pi.ndim != 2:
            raise ValueError("q_self, q_edge and pi must be [B,K], [B,M,K], [B,M+1]")
        if q_edge.shape[0] != q_self.shape[0] or pi.shape != (q_self.shape[0], q_edge.shape[1] + 1):
            raise ValueError("assignment transport shapes are inconsistent")
        if not torch.isfinite(q_self).all() or not torch.isfinite(q_edge).all() or not torch.isfinite(pi).all():
            raise ValueError("assignment transport inputs must be finite")
        null_mass = pi[:, :1].clamp_min(0.0)
        edge_mass = pi[:, 1:].clamp_min(0.0)
        mixed = null_mass * q_self + torch.sum(edge_mass[:, :, None] * q_edge, dim=1)
        return mixed.clamp_min(1e-8) / mixed.clamp_min(1e-8).sum(dim=1, keepdim=True)

    @staticmethod
    def mix_assignment_embedding(
        z_self: torch.Tensor,
        z_edge: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the assignment readout gate to latent vectors as well.

        The assignment output uses one null branch plus candidate edge mass.
        Keeping this operator identical in latent space makes ``embedding``
        and ``probabilities`` describe the same readout intervention; in
        particular, null-only rows are exactly the clean self embedding.
        """
        if z_self.ndim != 2 or z_edge.ndim != 3 or pi.ndim != 2:
            raise ValueError("z_self, z_edge and pi must be [B,D], [B,M,D], [B,M+1]")
        if z_edge.shape[0] != z_self.shape[0] or pi.shape != (z_self.shape[0], z_edge.shape[1] + 1):
            raise ValueError("assignment embedding transport shapes are inconsistent")
        if not torch.isfinite(z_self).all() or not torch.isfinite(z_edge).all() or not torch.isfinite(pi).all():
            raise ValueError("assignment embedding transport inputs must be finite")
        null_mass = pi[:, :1].clamp_min(0.0)
        edge_mass = pi[:, 1:].clamp_min(0.0)
        return null_mass * z_self + torch.sum(edge_mass[:, :, None] * z_edge, dim=1)


@torch.no_grad()
def make_teacher(student: V15Model) -> V15Model:
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher: V15Model, student: V15Model, decay: float) -> None:
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(float(decay)).add_(student_parameter, alpha=1.0 - float(decay))
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)
