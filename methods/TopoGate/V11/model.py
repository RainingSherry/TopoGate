"""Neural components for TopoGate V11.

The topology mixer is a conditional mixture-of-experts distribution over one
explicit null/self expert and all candidate-neighbour experts.  Consequently,
``1 - weight_self`` is the node-level topology gate and the remaining weights
are the edge-level reliabilities; the two quantities cannot silently disagree.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


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


class V11AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder_block = ResidualBlock(hidden_dim, dropout)
        self.encoder_out = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.mask_predictor = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder_block(self.encoder_stem(x))
        return self.encoder_out(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decoder(z), self.mask_predictor(z)

    @staticmethod
    def reconstruction_per_sample(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        distribution: str,
        masked_data_weight: float,
        student_t_nu: float,
    ) -> torch.Tensor:
        if distribution == "gaussian":
            element = (prediction - target).square()
        elif distribution == "student_t":
            nu = max(float(student_t_nu), 1e-3)
            element = 0.5 * (nu + 1.0) * torch.log1p((prediction - target).square() / nu)
        elif distribution == "bernoulli":
            element = F.binary_cross_entropy_with_logits(
                prediction, target.clamp(0.0, 1.0), reduction="none"
            )
        elif distribution == "poisson":
            if torch.any(target < 0):
                raise ValueError("Poisson reconstruction requires non-negative targets")
            element = F.poisson_nll_loss(
                prediction, target, log_input=True, full=False, reduction="none"
            )
        else:
            raise ValueError(f"unknown reconstruction distribution: {distribution}")
        weights = mask * float(masked_data_weight) + (1.0 - mask) * (1.0 - float(masked_data_weight))
        return (element * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)

    def masked_loss(
        self,
        corrupted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        distribution: str,
        masked_data_weight: float,
        mask_prediction_weight: float,
        student_t_nu: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, prediction, mask_logits = self(corrupted)
        rec = self.reconstruction_per_sample(
            prediction, target, mask, distribution, masked_data_weight, student_t_nu
        )
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask, reduction="none").mean(dim=1)
        total = rec + float(mask_prediction_weight) * mask_loss
        return z, total, prediction


class StudentTMixtureHead(nn.Module):
    """Student-t responsibilities with a learnable diagonal metric.

    The head is used for discriminative soft assignments rather than as a
    standalone generative likelihood objective.  ``diagonal_product`` retains
    the independent-coordinate evidence model with optional dimensional
    tempering.  ``radial`` uses the DEC-style robust Student-t assignment
    kernel over the mean diagonal Mahalanobis distance, avoiding the product
    density's exponential growth in logit gaps as latent width increases.
    """

    def __init__(
        self,
        n_clusters: int,
        latent_dim: int,
        degrees_of_freedom: float = 4.0,
        logit_normalization: str = "sqrt_dim",
        scale_floor_ratio: float = 0.0,
        assignment_kernel: str = "diagonal_product",
    ):
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.latent_dim = int(latent_dim)
        self.nu = float(degrees_of_freedom)
        self.logit_normalization = str(logit_normalization)
        self.scale_floor_ratio = float(scale_floor_ratio)
        self.assignment_kernel = str(assignment_kernel)
        if self.logit_normalization not in {"none", "sqrt_dim", "mean_dim"}:
            raise ValueError("unknown cluster logit normalization")
        if not 0.0 <= self.scale_floor_ratio <= 1.0:
            raise ValueError("scale_floor_ratio must be in [0, 1]")
        if self.assignment_kernel not in {"diagonal_product", "radial"}:
            raise ValueError("unknown Student-t assignment kernel")
        self.centres = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.02)
        self.log_scales = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.prior_logits = nn.Parameter(torch.zeros(n_clusters))
        self.register_buffer("initialised", torch.tensor(False))
        self.register_buffer("initial_scales", torch.ones(n_clusters, latent_dim))

    def effective_scales(self) -> torch.Tensor:
        """Return the diagonal metric scales after the optional warm-up floor."""
        scale = F.softplus(self.log_scales).clamp_min(1e-3)
        if bool(self.initialised) and self.scale_floor_ratio > 0.0:
            scale = torch.maximum(scale, self.initial_scales * self.scale_floor_ratio)
        return scale

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scale = self.effective_scales()
        delta = (z[:, None, :] - self.centres[None, :, :]) / scale[None, :, :]
        log_prior = F.log_softmax(self.prior_logits, dim=0)[None, :]
        if self.assignment_kernel == "radial":
            # A robust, dimension-normalised Student-t assignment kernel.  It
            # is intentionally a clustering kernel, not a claimed density;
            # unlike a coordinate product it does not make confidence depend
            # exponentially on latent width.
            distance = delta.square().mean(dim=2)
            log_kernel = -0.5 * (self.nu + 1.0) * torch.log1p(distance / self.nu)
            return F.softmax(log_kernel + log_prior, dim=1)

        log_kernel = -0.5 * (self.nu + 1.0) * torch.log1p(delta.square() / self.nu)
        log_density = log_kernel.sum(dim=2) - torch.log(scale).sum(dim=1)[None, :]
        if self.logit_normalization == "sqrt_dim":
            evidence_scale = math.sqrt(float(self.latent_dim))
        elif self.logit_normalization == "mean_dim":
            evidence_scale = float(self.latent_dim)
        else:
            evidence_scale = 1.0
        # Temper only the evidence; mixture priors remain interpretable as
        # priors instead of being accidentally weakened by latent width.
        return F.softmax(log_density / evidence_scale + log_prior, dim=1)

    @torch.no_grad()
    def initialise(self, embeddings: np.ndarray, seed: int, n_init: int = 20) -> None:
        km = KMeans(n_clusters=self.n_clusters, n_init=n_init, random_state=seed)
        labels = km.fit_predict(embeddings)
        centres = torch.as_tensor(km.cluster_centers_, dtype=self.centres.dtype, device=self.centres.device)
        scales = []
        counts = []
        global_scale = np.std(embeddings, axis=0) + 1e-2
        for cluster in range(self.n_clusters):
            members = embeddings[labels == cluster]
            counts.append(max(1, int(members.shape[0])))
            if members.shape[0] >= 2:
                scales.append(np.std(members, axis=0) + 1e-2)
            else:
                scales.append(global_scale)
        scales_t = torch.as_tensor(np.stack(scales), dtype=self.centres.dtype, device=self.centres.device)
        self.centres.copy_(centres)
        # Invert softplus without moving into its saturated tail.
        self.log_scales.copy_(torch.log(torch.expm1(scales_t.clamp(1e-2, 20.0))))
        self.initial_scales.copy_(scales_t.clamp_min(1e-3))
        prior = torch.as_tensor(counts, dtype=self.centres.dtype, device=self.centres.device)
        self.prior_logits.copy_(torch.log(prior / prior.sum()))
        self.initialised.fill_(True)

    def mixture_prior_loss(self, q: torch.Tensor, dirichlet_strength: float) -> torch.Tensor:
        pi = F.softmax(self.prior_logits, dim=0).clamp_min(1e-8)
        q_mean = q.mean(dim=0).clamp_min(1e-8)
        marginal_kl = torch.sum(q_mean * (torch.log(q_mean) - torch.log(pi)))
        # Symmetric Dirichlet(alpha=1+epsilon): weakly discourages zero priors
        # without forcing balanced clusters.
        dirichlet = -float(dirichlet_strength) * torch.log(pi).sum()
        return marginal_kl + dirichlet


class TopologyMixture(nn.Module):
    """Conditional distribution over self/null and candidate-neighbour experts."""

    def __init__(self, edge_feature_dim: int, node_feature_dim: int, hidden_dim: int, null_bias: float):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.LayerNorm(edge_feature_dim),
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.null_net = nn.Sequential(
            nn.LayerNorm(node_feature_dim),
            nn.Linear(node_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.null_net[-1].bias, float(null_bias))

    def forward(
        self,
        edge_features: torch.Tensor,
        node_features: torch.Tensor,
        valid: torch.Tensor,
        ramp: float,
        temperature: float,
        use_edge_reliability: bool,
    ) -> torch.Tensor:
        batch, width, _ = edge_features.shape
        if use_edge_reliability:
            edge_logits = self.edge_net(edge_features).squeeze(-1)
        else:
            edge_logits = torch.zeros(batch, width, dtype=edge_features.dtype, device=edge_features.device)
        edge_logits = edge_logits.masked_fill(~valid, -1e9)
        null_logits = self.null_net(node_features).squeeze(-1)
        # The topology experts do not exist during warmup.  Adding log(ramp)
        # gives a continuous curriculum and an exact conceptual null expert.
        log_ramp = math.log(max(float(ramp), 1e-8))
        edge_logits = edge_logits + log_ramp
        logits = torch.cat([null_logits[:, None], edge_logits], dim=1)
        return F.softmax(logits / max(float(temperature), 1e-4), dim=1)


class TopoGateV11(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_clusters: int,
        dropout: float,
        null_bias: float,
        student_t_nu: float,
        cluster_logit_normalization: str = "sqrt_dim",
        cluster_scale_floor_ratio: float = 0.0,
        cluster_assignment_kernel: str = "diagonal_product",
    ):
        super().__init__()
        self.autoencoder = V11AutoEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.cluster_head = StudentTMixtureHead(
            n_clusters,
            latent_dim,
            student_t_nu,
            logit_normalization=cluster_logit_normalization,
            scale_floor_ratio=cluster_scale_floor_ratio,
            assignment_kernel=cluster_assignment_kernel,
        )
        self.topology = TopologyMixture(edge_feature_dim=6, node_feature_dim=5, hidden_dim=64, null_bias=null_bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def assignments(self, x_or_z: torch.Tensor, encoded: bool = False) -> torch.Tensor:
        z = x_or_z if encoded else self.encode(x_or_z)
        return self.cluster_head(z)


@torch.no_grad()
def make_teacher(student: TopoGateV11) -> TopoGateV11:
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(float(decay)).add_(student_parameter, alpha=1.0 - float(decay))
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)
