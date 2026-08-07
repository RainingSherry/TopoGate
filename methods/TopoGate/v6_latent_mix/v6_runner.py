"""v6 runner — isolated TopoGate training loop with latent-space mix.

This module is a thin, self-contained training loop for the v6 latent-space
mix variant.  It is structurally similar to the run_npz training loop in
`methods/TopoGate/learnable_gate/run_npz.py:558-717` but with one key
difference:

  - The pseudo branch mixes in *latent* space (z_anchor, z_neighbor) instead
    of in *input* space (x_anchor, x_neighbor_mean).

No edits to learnable_gate/, static_gate/, NeighborMix_scMAE/model.py, or any
baseline — only imports.

What is reused (and how)
------------------------
  - `model.encoder`, `model.mask_predictor`, `model.decoder`            → via MicroMAEEncoder
  - `model.loss_mask_weighted`                                            → for the real branch
  - `LearnableGate`                                                       → for gate parameter parity
  - `build_gate_stats_tensor`                                             → for the per-node stats
  - `compute_edge_reliability`                                            → for static edge weights
  - `compute_node_gate`                                                   → for `node_gate` (used
                                                                            for sample_weight when
                                                                            neighbour_estimator is
                                                                            uniform_sample; here we
                                                                            always sample from the
                                                                            reliability weights)
  - `build_pca_knn_graph`, `build_random_neighbors`, `build_far_neighbors`
  - `apply_mask_noise` (re-defined here as a small private helper, identical
                        to the one in run_npz).

What is new
-----------
  - `LatentMixer` (from latent_mixer.py) — gate computation + mix step.
  - `MicroMAEEncoder` (from micro_encoder.py) — exposes encode/decode_from_latent.

CLI surface
-----------
`parse_v6_args()` returns the same argument names as `run_npz.parse_args()`
plus a small set of v6-specific flags.  `run_v6(...)` is the high-level entry
point that wraps training + embedding extraction + KMeans + metrics, mirroring
`run_topogate(...)` in run_npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Block BLAS / MKL / OpenMP thread leakage before any heavy import
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = next(
    p for p in [CURRENT_DIR, *CURRENT_DIR.parents]
    if (p / "methods" / "DeepLearning" / "scMAE_family.py").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.learnable_gate.model import AutoEncoder  # noqa: E402
from methods.TopoGate.learnable_gate.neighbor_graph import (  # noqa: E402
    build_pca_knn_graph,
    build_random_neighbors,
    build_far_neighbors,
    compute_edge_reliability,
)
from methods.TopoGate.learnable_gate.learnable_gate import (  # noqa: E402
    LearnableGate,
    build_gate_stats_tensor,
)
from methods.TopoGate.v6_latent_mix.latent_mixer import LatentMixer  # noqa: E402
from methods.TopoGate.v6_latent_mix.micro_encoder import MicroMAEEncoder  # noqa: E402


# ──────────────────────────────────────────────
#  Argparse
# ──────────────────────────────────────────────

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got {v!r}")


def parse_v6_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TopoGate v6 (latent space mix)")
    # Core (mirror run_npz)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--method_name", default="TopoGate")
    parser.add_argument("--variant_name", default="v6_latent_mix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")

    # Preprocessing
    parser.add_argument("--input_mode", default="raw", choices=["raw", "log1p"])
    parser.add_argument("--scale_input", type=str2bool, default=True)

    # Model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--masked_data_weight", type=float, default=0.75)
    parser.add_argument("--mask_loss_weight", type=float, default=0.7)

    # Training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # TopoGate topology (mirror run_npz defaults)
    parser.add_argument("--neighbor_k", type=int, default=5)
    parser.add_argument("--mix_neighbors", type=int, default=4)
    parser.add_argument("--mix_mode", default="reliability",
                        choices=["none", "fixed", "mutual", "reliability", "random", "far"])
    parser.add_argument("--neighbor_estimator", default="current",
                        choices=["current", "uniform_sample", "full"])
    parser.add_argument("--gate_mode", default="learned",
                        choices=["none", "constant", "topology", "learned"])
    parser.add_argument("--gate_max", type=float, default=0.5,
                        help="v6 default extended upper bound (learnable_gate uses 0.15).")
    parser.add_argument("--gate_min", type=float, default=0.0)
    parser.add_argument("--pseudo_weight", type=float, default=0.3)
    parser.add_argument("--edge_reliability_mode", default="sim_mutual_snn_distance")
    parser.add_argument("--knn_pca_dim", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--gamma_sim", type=float, default=1.0)
    parser.add_argument("--gamma_mutual", type=float, default=1.0)
    parser.add_argument("--gamma_snn", type=float, default=1.0)
    parser.add_argument("--gamma_distance", type=float, default=1.0)
    parser.add_argument("--beta_mutual", type=float, default=1.0)
    parser.add_argument("--beta_snn", type=float, default=1.0)
    parser.add_argument("--beta_perturb", type=float, default=2.0)
    parser.add_argument("--beta_uncertainty", type=float, default=1.0)

    # LearnableGate init mode
    parser.add_argument("--init_beta_mutual", type=float, default=0.0)
    parser.add_argument("--init_beta_snn", type=float, default=0.0)
    parser.add_argument("--init_beta_perturb", type=float, default=0.0)
    parser.add_argument("--init_beta_uncertainty", type=float, default=0.0)
    parser.add_argument("--learned_gate_init_mode", type=str, default="zero",
                        choices=["zero", "v1_default"])
    parser.add_argument("--learnable_gate_max", type=str2bool, default=False)
    parser.add_argument("--gate_max_min", type=float, default=0.05)
    parser.add_argument("--gate_max_max", type=float, default=1.0)
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0)
    parser.add_argument("--enhanced_stats", type=int, default=4, choices=[4, 6])

    # Schedule (parity with run_npz.py — required for fair comparison)
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of epochs where schedule_t = 0 (gate = static v1 fallback).")
    parser.add_argument("--ramp_epochs", type=int, default=10,
                        help="Number of epochs over which t linearly ramps from 0 to 1.")
    parser.add_argument("--freeze_mae_after_epoch", type=int, default=1000000000,
                        help="If > 0 and < epochs, MAE encoder/decoder is frozen after this "
                             "epoch (so gate params keep optimising alone). Default 1e9 = disabled.")

    # v6-specific
    parser.add_argument("--latent_consistency_weight", type=float, default=0.0,
                        help="Weight for ||z_mixed - z_anchor||^2 auxiliary loss. "
                             "Phase 1 default 0 (off).")

    # I/O
    parser.add_argument("--lightweight_outputs", action="store_true")

    return parser.parse_args()


# ──────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────

def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    X = data.get("X", data.get("x", data.get("data")))
    y = data.get("y", data.get("labels", data.get("label", None)))
    if X is None:
        raise ValueError(f"npz at {path!r} must contain 'X'/'x'/'data' key.")
    X = np.asarray(X, dtype=np.float64)
    if y is not None:
        y = np.asarray(y).ravel()
    return X, y


def load_compressed(path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    import zlib
    with open(os.path.join(path, "data.bin"), "rb") as f:
        data = np.array(json.loads(zlib.decompress(f.read()).decode("utf8")))
    label_path = os.path.join(path, "label.bin")
    if os.path.exists(label_path):
        with open(label_path, "rb") as f:
            labels = np.array(json.loads(zlib.decompress(f.read()).decode("utf8")))
    else:
        labels = None
    return data.astype(np.float64), labels


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    if path.suffix == ".npz":
        return load_npz(str(path))
    elif path.is_dir():
        return load_compressed(str(path))
    else:
        raise ValueError(f"Unsupported data path: {path!r}")


# ──────────────────────────────────────────────
#  Masked noise (identical to run_npz.apply_mask_noise)
# ──────────────────────────────────────────────

def apply_mask_noise(x: torch.Tensor, mask_ratio) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(mask_ratio, torch.Tensor):
        ratio_val = float(mask_ratio.detach().cpu())
    else:
        ratio_val = float(mask_ratio)
    should_swap = torch.bernoulli(ratio_val * torch.ones_like(x))
    if x.shape[0] <= 1:
        replacement = x
    else:
        replacement = x[torch.randperm(x.shape[0], device=x.device)]
    corrupted = torch.where(should_swap.bool(), replacement, x)
    mask = (corrupted != x).float()
    return corrupted, mask


# ──────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────

def linear_assignment(cost_matrix):
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.column_stack([row_ind, col_ind])


def align_labels(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(np.unique(y_pred)) == 1:
        return y_pred
    cm = confusion_matrix(y_true, y_pred)
    aligned = np.zeros_like(y_pred)
    for r, c in linear_assignment(-cm):
        aligned[y_pred == c] = r
    return aligned


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_aligned = align_labels(y_true, y_pred)
    return {
        "acc": float(accuracy_score(y_true, y_aligned)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_aligned, average="macro", zero_division=0)),
        "fmi": float(fowlkes_mallows_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
    }


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    with open(path, "w") as f:
        json.dump(obj, f, indent=4, default=conv)


# ──────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────

def get_device(gpu: int, no_cuda: bool) -> torch.device:
    if no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    forbidden = {0, 7}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        ids = [i.strip() for i in visible.split(",") if i.strip()]
        if any(i in forbidden for i in ids):
            raise ValueError("CUDA_VISIBLE_DEVICES includes forbidden GPU 0 or 7.")
        if len(ids) == 1:
            return torch.device("cuda:0")
        if str(gpu) in ids:
            return torch.device(f"cuda:{ids.index(str(gpu))}")
        if 0 <= gpu < len(ids):
            return torch.device(f"cuda:{gpu}")
        raise ValueError(f"GPU {gpu} not in CUDA_VISIBLE_DEVICES={visible!r}.")
    if gpu in forbidden:
        raise ValueError("Physical GPU 0 and GPU 7 are forbidden. Use 1-6.")
    return torch.device(f"cuda:{gpu}")


# ──────────────────────────────────────────────
#  Neighbour sampling (numpy-only — no torch grad needed here)
# ──────────────────────────────────────────────

def sample_neighbours_for_batch(
    batch_indices: np.ndarray,
    graph_indices: np.ndarray,
    graph_probs: np.ndarray,
    edge_weights: np.ndarray,
    mix_mode: str,
    mix_neighbors: int,
    random_neighbors: np.ndarray | None,
    far_neighbors: np.ndarray | None,
    rng: np.random.Generator,
    node_gate_for_max: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick one weighted-mean neighbour per anchor cell.

    Returns:
        sampled_idx: (bsz, mix_neighbors) int64
        sampled_weights: (bsz, mix_neighbors) float32
    """
    bsz = int(batch_indices.shape[0])
    k = int(graph_indices.shape[1])
    if mix_mode == "none" or k == 0 or int(mix_neighbors) <= 0:
        return np.zeros((bsz, 0), dtype=np.int64), np.zeros((bsz, 0), dtype=np.float32)
    m = max(1, min(int(mix_neighbors), k))
    sampled = np.empty((bsz, m), dtype=np.int64)
    weights = np.empty((bsz, m), dtype=np.float32)
    for pos, cell in enumerate(batch_indices):
        if mix_mode == "random" and random_neighbors is not None:
            row = random_neighbors[cell]
            probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
        elif mix_mode == "far" and far_neighbors is not None:
            row = far_neighbors[cell]
            probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
        elif mix_mode == "mutual":
            mask = graph_probs[cell]  # use probs as a placeholder mask array
            # The real mutual mask lives on the graph itself; fall back to the
            # full row when the mask shape does not match.
            row = graph_indices[cell]
            if mask.shape == row.shape:
                row = row[mask.astype(bool)]
            probs = np.full(row.shape[0], 1.0 / max(row.shape[0], 1), dtype=np.float32)
        else:
            row = graph_indices[cell]
            probs = edge_weights[cell] if mix_mode == "reliability" else graph_probs[cell]
        if probs.sum() <= 0:
            probs = np.full(row.shape[0], 1.0 / max(row.shape[0], 1), dtype=np.float32)
        normalized = probs / probs.sum()
        choices = rng.choice(row.shape[0], size=m, replace=True, p=normalized)
        sampled[pos] = row[choices]
        picked = probs[choices].astype(np.float32, copy=False)
        weights[pos] = picked / max(float(picked.sum()), 1e-12)
    return sampled, weights


# ──────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────

def train_and_evaluate(args: argparse.Namespace) -> Dict:
    """Train TopoGate with v6 latent-space mix and return summary dict."""

    # Seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    X_raw, y_raw = load_data(args.data_path)
    dataset_name = args.dataset_name or Path(args.data_path).stem

    # Preprocess
    if args.input_mode == "log1p" and np.nanmax(X_raw) <= 30:
        X_raw = np.log1p(X_raw)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_np = scaler.fit_transform(X_raw).astype(np.float32)

    if y_raw is None:
        if args.n_clusters is None:
            raise ValueError("n_clusters required when y is absent.")
        y_np = None
        n_clusters = args.n_clusters
    else:
        le = LabelEncoder()
        y_np = le.fit_transform(np.asarray(y_raw).ravel()).astype(np.int64)
        n_clusters = args.n_clusters if args.n_clusters is not None else len(np.unique(y_np))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), str(save_dir / "args.json"))

    device = get_device(args.gpu, args.no_cuda)
    print(
        f"[{dataset_name}] v6_latent_mix  device={device}  "
        f"n={X_np.shape[0]}  d={X_np.shape[1]}  K={n_clusters}",
        flush=True,
    )

    # Topology
    graph = build_pca_knn_graph(
        X_np, k=args.neighbor_k,
        pca_dim=min(args.knn_pca_dim, X_np.shape[1]),
        tau=args.tau, seed=args.seed,
    )
    _, edge_weights, edge_summary = compute_edge_reliability(
        graph, mode=args.edge_reliability_mode,
        gamma_sim=args.gamma_sim, gamma_mutual=args.gamma_mutual,
        gamma_snn=args.gamma_snn, gamma_distance=args.gamma_distance,
    )
    rng = np.random.default_rng(args.seed + 3089)
    random_neighbors = build_random_neighbors(
        X_np.shape[0], max(1, min(args.mix_neighbors, X_np.shape[0] - 1)),
        rng, graph.indices,
    )
    far_neighbors = build_far_neighbors(
        graph.embedding,
        max(1, min(args.mix_neighbors, X_np.shape[0] - 1)),
        rng,
    )

    # Per-node stats tensor (4 or 6)
    stats_tensor = build_gate_stats_tensor(
        graph.indices, graph.mutual, graph.snn, graph.probs, graph.similarity,
        uncertainty=None, device=device,
        enhanced_stats=int(args.enhanced_stats),
    )

    # LatentMixer (gate computation + mix step)
    if args.learned_gate_init_mode == "v1_default":
        init_m, init_s, init_p, init_u = (
            args.beta_mutual, args.beta_snn, args.beta_perturb, args.beta_uncertainty,
        )
    else:
        init_m = init_s = init_p = init_u = 0.0
    latent_mixer = LatentMixer(
        gate_min=args.gate_min,
        gate_max=args.gate_max,
        init_beta_mutual=init_m,
        init_beta_snn=init_s,
        init_beta_perturb=init_p,
        init_beta_uncertainty=init_u,
        # v6-patch: parity with run_npz.py
        learnable_gate_max=bool(args.learnable_gate_max),
        gate_max_min=float(args.gate_max_min),
        gate_max_max=float(args.gate_max_max),
        enhanced_stats=int(args.enhanced_stats),
        latent_consistency_weight=float(args.latent_consistency_weight),
    ).to(device)

    # Pre-compute v1-style static gate for schedule fallback (run_npz.py parity).
    # Used during warmup_epochs + ramped-out during ramp_epochs so that β does not
    # see gradient before the model has converged on the manifold.
    from methods.TopoGate.learnable_gate.mixing import compute_node_gate
    static_gate_np, _, _ = compute_node_gate(
        graph, edge_weights=edge_weights,
        gate_mode="topology" if args.gate_mode != "learned" else "topology",
        gate_min=args.gate_min, gate_max=args.gate_max,
        beta_mutual=args.beta_mutual, beta_snn=args.beta_snn,
        beta_perturb=args.beta_perturb, beta_uncertainty=args.beta_uncertainty,
        uncertainty=None,
    )
    static_gate_tensor = torch.as_tensor(static_gate_np, dtype=torch.float32, device=device)
    learned_gate_static_np = static_gate_np.astype(np.float32).copy()

    # Model
    model = AutoEncoder(
        num_genes=X_np.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        masked_data_weight=args.masked_data_weight,
        mask_loss_weight=args.mask_loss_weight,
    ).to(device)
    micro_encoder = MicroMAEEncoder(model).to(device)

    # Two param groups: MAE + gate (with lr multiplier)
    mae_params = list(model.parameters())
    gate_params = list(latent_mixer.parameters())
    optimizer = torch.optim.Adam([
        {"params": mae_params, "lr": args.lr},
        {"params": gate_params, "lr": args.lr * float(args.gate_lr_multiplier)},
    ])

    # DataLoaders
    from torch.utils.data import DataLoader, Dataset

    class _Dataset(Dataset):
        def __init__(self, X):
            self.X = torch.as_tensor(X.astype(np.float32))

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return int(idx), self.X[idx]

    train_dataset = _Dataset(X_np)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, generator=gen
    )
    eval_loader = DataLoader(
        train_dataset, batch_size=max(args.batch_size * 4, 512), shuffle=False
    )

    pseudo_enabled = args.mix_mode != "none" and float(args.pseudo_weight) > 0
    beta_history: List[Dict] = []
    mae_param_ids = {id(p) for p in mae_params}   # for freeze_mae_after_epoch

    # Training loop
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Schedule t — parity with run_npz.py.  Default 10/10 means beta only
        # starts seeing MAE gradient from epoch 11 onward, ramps to full strength
        # by epoch 20.  Crucial: without this, init β=0 → sigmoid(0)=0.5 →
        # gate immediately saturates (we saw this in Phase-1 smoke; see
        # CHANGELOG_errors.md 2026-07-26 v6 first-pass).
        if pseudo_enabled:
            t = max(0.0, min(1.0, (epoch - args.warmup_epochs) / max(1, args.ramp_epochs)))
        else:
            t = 1.0
        mae_frozen = (
            pseudo_enabled
            and args.freeze_mae_after_epoch >= 0
            and epoch > args.freeze_mae_after_epoch
        )

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for idx_t, x in train_loader:
            idx_np = idx_t.cpu().numpy()
            x = x.to(device)

            # ── Real branch: standard MAE on (masked anchor, anchor) ──
            x_corrupt, real_mask = apply_mask_noise(x, args.mask_ratio)
            _, real_loss, _ = model.loss_mask_weighted(x_corrupt, x, real_mask)
            loss = real_loss

            # ── Pseudo branch: latent-space mix ──
            pseudo_loss = torch.zeros((), dtype=real_loss.dtype, device=device)
            aux_loss = torch.zeros((), dtype=real_loss.dtype, device=device)
            if pseudo_enabled:
                # 1. Sample neighbours (numpy)
                sampled_idx, _ = sample_neighbours_for_batch(
                    batch_indices=idx_np,
                    graph_indices=graph.indices,
                    graph_probs=graph.probs,
                    edge_weights=edge_weights,
                    mix_mode=args.mix_mode,
                    mix_neighbors=args.mix_neighbors,
                    random_neighbors=random_neighbors,
                    far_neighbors=far_neighbors,
                    rng=rng,
                    node_gate_for_max=None,
                )
                if sampled_idx.shape[1] == 0:
                    pseudo_loss = torch.zeros((), dtype=real_loss.dtype, device=device)
                else:
                    # 2. Build a per-cell neighbour x_n via weighted mean over the m picks
                    x_neighbor = X_np[sampled_idx]  # (bsz, m, num_genes)
                    # weights from sampling (already normalised); reduce to mean
                    x_neighbor = x_neighbor.mean(axis=1)  # (bsz, num_genes)
                    x_neighbor_t = torch.as_tensor(x_neighbor, dtype=x.dtype, device=device)

                    # 3. Mask both anchor and neighbour, encode both to latent
                    x_n_corrupt, _ = apply_mask_noise(x_neighbor_t, args.mask_ratio)
                    z_anchor, _, _ = model.forward_mask(x_corrupt)         # (bsz, hidden)
                    z_neighbor = micro_encoder.encode(x_n_corrupt)          # (bsz, hidden)

                    # 4. Latent mix (with schedule: static_gate_during_warmup, ramp, then dyn)
                    batch_stats = stats_tensor[idx_t]
                    if t < 1.0:
                        sg = static_gate_tensor[idx_t]
                        z_mixed, mix_info = latent_mixer(
                            z_anchor, z_neighbor, batch_stats,
                            static_gate=sg, schedule_t=t,
                        )
                    else:
                        z_mixed, mix_info = latent_mixer(
                            z_anchor, z_neighbor, batch_stats,
                        )

                    # 5. Decode back to input space
                    recon = micro_encoder.decode_from_latent(z_mixed)        # (bsz, num_genes)

                    # 6. MAE reconstruction loss against anchor (target = x, mask = real_mask)
                    raw_mse = F.mse_loss(recon, x, reduction="none")
                    w_mask = real_mask * model.masked_data_weight + (1.0 - real_mask) * (1.0 - model.masked_data_weight)
                    weighted = w_mask * raw_mse
                    if bool(getattr(model, "normalize_reconstruction_by_weight", False)):
                        rec = weighted.sum(dim=1) / w_mask.sum(dim=1).clamp_min(1e-8)
                    else:
                        rec = weighted.mean(dim=1)
                    pseudo_loss = (1.0 - model.mask_loss_weight) * rec.mean()
                    if float(args.latent_consistency_weight) > 0.0:
                        aux_loss = torch.as_tensor(
                            mix_info["latent_consistency_loss"],
                            dtype=real_loss.dtype, device=device,
                        ) * float(args.latent_consistency_weight)
                loss = loss + float(args.pseudo_weight) * pseudo_loss + aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # v6-patch: parity with run_npz.py — freeze MAE params if asked.
            if mae_frozen:
                for pg in optimizer.param_groups:
                    if not pg["params"]:
                        continue
                    for p in pg["params"]:
                        if p.grad is not None and id(p) in mae_param_ids:
                            p.grad = None
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        # beta history (every epoch, now with schedule_t and mae_frozen).
        beta_history.append({
            "epoch": epoch,
            "schedule_t": float(t),
            "mae_frozen": bool(mae_frozen),
            **latent_mixer.beta_snapshot(),
        })

        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:03d}/{args.epochs}  "
                f"loss={epoch_loss / max(1, n_batches):.4f}  "
                f"sched_t={t:.2f}"
                f"{'  [MAE-frozen]' if mae_frozen else ''}",
                flush=True,
            )

    train_time = time.time() - t0

    # Embedding extraction
    model.eval()
    embeddings = []
    with torch.no_grad():
        for _, x in eval_loader:
            z = model.feature(x.to(device))
            embeddings.append(z.detach().cpu().numpy())
    embedding = np.concatenate(embeddings, axis=0).astype(np.float32)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

    # KMeans + metrics
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=args.seed)
    pred = km.fit_predict(embedding)
    if y_np is not None:
        metrics = compute_metrics(y_np, pred)
        metrics_str = f"ACC={metrics['acc']:.4f}  NMI={metrics['nmi']:.4f}  ARI={metrics['ari']:.4f}"
    else:
        metrics = {}
        metrics_str = "ACC=N/A  NMI=N/A  ARI=N/A"

    # Save outputs (mirror run_npz layout)
    np.save(save_dir / "embedding_final.npy", embedding)
    np.save(save_dir / "labels.npy", y_np.astype(np.int64) if y_np is not None else pred.astype(np.int64))
    if not args.lightweight_outputs:
        np.save(save_dir / "embeddings_base.npy", embedding)
    save_json(metrics, str(save_dir / "metrics.json"))
    summary = {
        "dataset": dataset_name,
        "method": args.method_name,
        "variant": args.variant_name,
        "seed": int(args.seed),
        "n_samples": int(X_np.shape[0]),
        "n_features": int(X_np.shape[1]),
        "n_clusters": int(n_clusters),
        "mix_mode": args.mix_mode,
        "pseudo_weight": float(args.pseudo_weight),
        "gate_mode": args.gate_mode,
        "train_seconds": float(train_time),
        "edge_reliability_summary": edge_summary,
        "gate_summary": {
            "gate_mode": args.gate_mode,
            "gate_min": float(args.gate_min),
            "gate_max": float(args.gate_max),
            "learned_gate_init_mode": args.learned_gate_init_mode,
            "init_beta_mutual": float(init_m),
            "init_beta_snn": float(init_s),
            "init_beta_perturb": float(init_p),
            "init_beta_uncertainty": float(init_u),
            "learnable_gate_max": bool(args.learnable_gate_max),
            "warmup_epochs": int(args.warmup_epochs),
            "ramp_epochs": int(args.ramp_epochs),
            "freeze_mae_after_epoch": int(args.freeze_mae_after_epoch),
            "static_gate_mean": float(np.mean(learned_gate_static_np)) if learned_gate_static_np.size else 0.0,
            "static_gate_max": float(np.max(learned_gate_static_np)) if learned_gate_static_np.size else 0.0,
        },
        "learned_gate_final_beta": latent_mixer.beta_snapshot(),
        "learned_gate_beta_history": beta_history,
        "latent_consistency_weight": float(args.latent_consistency_weight),
    }
    save_json(summary, str(save_dir / "summary.json"))

    print(
        f"[{dataset_name}] v6 done  {metrics_str}  time={train_time:.1f}s",
        flush=True,
    )
    return summary


def main():
    args = parse_v6_args()
    train_and_evaluate(args)


# ──────────────────────────────────────────────
#  High-level wrapper (mirrors run_topogate)
# ──────────────────────────────────────────────

def run_v6(X, n_clusters, y=None, gpu=4, variant="v6_latent_mix", save_dir=None,
           seed=42, return_metrics=False, **overrides):
    """Top-level v6 entry point.  Mirrors `run_topogate` for direct comparison."""
    import tempfile
    import shutil
    import io
    import contextlib

    cli_args = dict(overrides)
    # Decide data_path / save_dir (use placeholders for the build phase)
    tmp_npz = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp_npz.close()
    save_kwargs = {"X": np.asarray(X, dtype=np.float64)}
    if y is not None:
        save_kwargs["y"] = np.asarray(y).ravel()
    np.savez(tmp_npz.name, **save_kwargs)

    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix=f"topogate_v6_{variant}_")
        _cleanup_tmp = True
    else:
        save_dir = str(save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _cleanup_tmp = False

    full_argv = ["v6_runner.py"]
    for k, v in cli_args.items():
        full_argv += [f"--{k}", str(v)]
    full_argv += [
        "--data_path", tmp_npz.name,
        "--save_dir", save_dir,
        "--dataset_name", "adhoc",
        "--variant_name", variant,
        "--method_name", "TopoGate",
        "--n_clusters", str(int(n_clusters)),
        "--seed", str(int(seed)),
        "--gpu", str(int(gpu)),
    ]

    saved_argv = sys.argv
    sys.argv = full_argv
    buf = io.StringIO()
    try:
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            summary = train_and_evaluate(parse_v6_args())
        elapsed = time.time() - t0
    finally:
        sys.argv = saved_argv

    metrics_path = Path(save_dir) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    emb_path = Path(save_dir) / "embedding_final.npy"
    if emb_path.exists():
        embedding = np.load(emb_path)
        km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(seed))
        pred_labels = km.fit_predict(embedding)
    else:
        pred_labels = np.array([], dtype=np.int64)

    try:
        os.unlink(tmp_npz.name)
    except OSError:
        pass
    if _cleanup_tmp:
        shutil.rmtree(save_dir, ignore_errors=True)

    if return_metrics:
        return pred_labels, elapsed, metrics
    return pred_labels, elapsed


if __name__ == "__main__":
    raise SystemExit(main())