#!/usr/bin/env python
"""
TopoGate runner for generic tabular / npz datasets.
=================================================

Accepts .npz files (X, y) or the compressed binary format used by ToPoGate's
dataset/reader.py.  Produces the same output layout as TopoGate/run.py so that
all downstream paper-evaluation code works unchanged.

Outputs (same layout as TopoGate/run.py):
    embedding_final.npy  — learned embedding (n_samples, hidden_size)
    labels.npy           — integer-encoded ground-truth labels
    metrics.json         — ACC/NMI/ARI/F1/... after Hungarian-aligned KMeans
    summary.json         — runtime + method metadata
    <plus all TopoGate training diagnostics>

Usage:
    python run_npz.py --data_path <path/to/data.npz> --save_dir <out_dir> --seed 42 --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = next(
    p for p in [CURRENT_DIR, *CURRENT_DIR.parents]
    if (p / "methods" / "DeepLearning" / "scMAE_family.py").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.learnable_gate.model import AutoEncoder
from methods.TopoGate.learnable_gate.neighbor_graph import (
    build_pca_knn_graph,
    build_random_neighbors,
    build_far_neighbors,
    compute_edge_reliability,
)
from methods.TopoGate.learnable_gate.mixing import compute_node_gate, make_pseudo_batch, make_pseudo_batch_binary
from methods.TopoGate.learnable_gate.learnable_gate import LearnableGate, build_gate_stats_tensor
from methods.TopoGate.learnable_gate.learnable_edge_reliability import (
    LearnableEdgeReliability,
    edge_weights_to_numpy,
)
from methods.TopoGate.learnable_gate.binary_router import BinaryRouter
from methods.TopoGate.learnable_gate.uncertainty import compute_mc_dropout_uncertainty
from methods.shared_utils import ensure_dir


# ──────────────────────────────────────────────
#  Adaptive PCA helper
# ──────────────────────────────────────────────

def select_adaptive_pca_dim(X: np.ndarray, max_dim: int = 200,
                             var_threshold: float = 0.95,
                             min_dim: int = 10,
                             seed: int = 0) -> int:
    """Auto-select PCA dim to retain at least var_threshold (default 95%) variance.

    Args:
        X: (n, d) raw feature matrix (should be scaled before calling).
        max_dim: Upper bound on the selected dim.
        var_threshold: Fraction of cumulative variance to retain.
        min_dim: Lower bound on the selected dim.
        seed: Random seed for PCA.
    Returns:
        int: selected PCA dimension.
    """
    actual_max = min(max_dim, X.shape[0] - 1, X.shape[1])
    if actual_max < min_dim:
        return actual_max
    pca = PCA(n_components=actual_max, random_state=seed)
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_needed = int(np.searchsorted(cumvar, var_threshold)) + 1
    n_needed = max(min_dim, min(n_needed, max_dim, X.shape[1], X.shape[0] - 1))
    return n_needed


# ──────────────────────────────────────────────
#  Data loading helpers
# ──────────────────────────────────────────────

def load_npz(path: str):
    """Load X, y from .npz file."""
    data = np.load(path)
    X = data.get("X", data.get("x", data.get("data")))
    y = data.get("y", data.get("labels", data.get("label", None)))
    if X is None:
        raise ValueError(f"npz at {path!r} must contain 'X'/'x'/'data' key.")
    X = np.asarray(X, dtype=np.float64)
    if y is not None:
        y = np.asarray(y).ravel()
    return X, y


def load_compressed(path: str):
    """Load from ToPoGate's compressed binary reader path."""
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


def load_data(path: str):
    """Auto-detect format and load (X, y)."""
    path = Path(path)
    if path.suffix == ".npz":
        return load_npz(str(path))
    elif path.is_dir():
        return load_compressed(str(path))
    else:
        raise ValueError(f"Unsupported data path: {path!r}")


class TensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None):
        self.X = torch.as_tensor(X.astype(np.float32))
        self.y = None if y is None else torch.as_tensor(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        y_val = self.y[idx] if self.y is not None else torch.tensor(0, dtype=torch.int64)
        return int(idx), self.X[idx], y_val


# ──────────────────────────────────────────────
#  Argument parsing
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


def parse_args():
    parser = argparse.ArgumentParser(description="TopoGate (npz / tabular)")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--method_name", default="TopoGate")
    parser.add_argument("--variant_name", default="topogate_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Number of clusters (auto-detected from y if not given)")
    parser.add_argument("--gpu", type=int, default=1)

    # Preprocessing
    parser.add_argument("--n_top_features", type=int, default=0,
        help="HVF: number of top high-variance features to keep before PCA. "
             "0=disabled (use all features). Recommended: 1000-2000 for d>5000.")
    parser.add_argument("--knn_pca_mode", type=str, default="fixed",
        choices=["fixed", "adaptive"],
        help="fixed: use knn_pca_dim directly. "
             "adaptive: auto-select dim to retain at least 95 percent variance "
             "(capped at knn_pca_dim).")
    parser.add_argument("--input_mode", default="raw", choices=["raw", "log1p"])
    parser.add_argument("--scale_input", type=str2bool, default=True)

    # Model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--masked_data_weight", type=float, default=0.75)
    parser.add_argument("--mask_loss_weight", type=float, default=0.7)

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # TopoGate topology
    parser.add_argument("--neighbor_k", type=int, default=10)
    parser.add_argument("--mix_neighbors", type=int, default=4)
    parser.add_argument("--mix_mode", default="reliability",
                        choices=["none", "fixed", "mutual", "reliability", "random", "far"])
    parser.add_argument("--risk_adaptive_mix", type=str2bool, default=False,
                        help="V12: scale each node's learned topology gate by an unsupervised local-risk proxy.")
    parser.add_argument("--risk_adaptive_temperature", type=float, default=1.0,
                        help="V12 temperature for the local-risk attenuation; must be positive.")
    parser.add_argument("--neighbor_estimator", default="current",
                        choices=["current", "uniform_sample", "full"])
    parser.add_argument("--gate_mode", default="topology",
                        choices=["none", "constant", "topology", "learned", "binary"])
    parser.add_argument("--gate_max", type=float, default=0.15)
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

    # learnable_gate learnable gate (only used when gate_mode == 'learned')
    parser.add_argument("--init_beta_mutual", type=float, default=0.0)
    parser.add_argument("--init_beta_snn", type=float, default=0.0)
    parser.add_argument("--init_beta_perturb", type=float, default=0.0)
    parser.add_argument("--init_beta_uncertainty", type=float, default=0.0)
    parser.add_argument("--learned_gate_init_mode", type=str, default="zero",
                        choices=["zero", "v1_default", "nomix"])
    parser.add_argument("--learnable_gate_max", type=str2bool, default=False,
                        help="v3: promote gate_max to a learnable parameter "
                             "(initialised at --gate_max, range [0.05, 1.0]).")
    parser.add_argument("--gate_max_min", type=float, default=0.05,
                        help="Floor for learnable gate_max.")
    parser.add_argument("--gate_max_max", type=float, default=1.0,
                        help="Ceiling for learnable gate_max.")
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0,
                        help="v3: multiplier for the gate parameter group's lr. "
                             "Defaults to 10x. This decouples the gate learning rate "
                             "from the small pseudo-loss channel (pseudo_weight times "
                             "gate_max), so the gate can compete with the MAE loss.")
    # Direction B: BinaryRouter
    parser.add_argument("--router_init_temp", type=float, default=5.0,
                        help="Direction B: initial Gumbel-Softmax temperature. "
                             "Higher = softer routing. Default 5.0.")
    parser.add_argument("--router_temp_min", type=float, default=0.01,
                        help="Direction B: minimum temperature after cool-down. Default 0.01.")
    parser.add_argument("--router_warmup_epochs", type=int, default=20,
                        help="Direction B: warmup epochs before temperature cools. Default 20.")
    parser.add_argument("--router_ramp_epochs", type=int, default=10,
                        help="Direction B: ramp epochs for temperature cool-down. Default 10.")
    # v3: LearnableEdgeReliability (promote the 4 gamma to nn.Parameter)
    parser.add_argument("--learnable_gamma", type=str2bool, default=False,
                        help="v3: promote the 4 gamma coefficients to learnable nn.Parameter. "
                             "Only effective when edge_reliability_mode != 'none'.")
    parser.add_argument("--gamma_reg_weight", type=float, default=1e-4,
                        help="v3: L2 regularisation weight on the 4 gamma.")
    # v3: EnhancedTopologyFeatures (extend stats from 4 → 6 with degree/clustering)
    parser.add_argument("--enhanced_stats", type=int, default=4, choices=[4, 6],
                        help="v3: number of per-node stats passed to the gate. 6 "
                             "adds degree_norm and clustering_coeff on top of "
                             "mutual/snn/perturb/uncertainty.")
    # v3: AdaptiveMaskRatio (promote mask_ratio to a learnable parameter)
    parser.add_argument("--learnable_mask_ratio", type=str2bool, default=False,
                        help="v3: make mask_ratio a learnable parameter initialised "
                             "at --mask_ratio and clamped to [0.1, 0.6].")
    parser.add_argument("--mask_ratio_min", type=float, default=0.1,
                        help="v3: floor for the learnable mask_ratio.")
    parser.add_argument("--mask_ratio_max", type=float, default=0.6,
                        help="v3: ceiling for the learnable mask_ratio.")
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--ramp_epochs", type=int, default=10)
    parser.add_argument(
        "--use_beta_scale_schedule",
        type=str2bool,
        default=False,
        help="Legacy experiment switch. False preserves the V9 gate equation; "
             "True enables the later nomix beta-scale curriculum.",
    )
    # Freeze the MAE encoder after this epoch number so that LearnableGate β
    # can settle without chasing a moving target.  Set to a large value
    # (e.g. 10**9) to disable freezing entirely.  Only effective when
    # gate_mode == 'learned'.
    parser.add_argument("--freeze_mae_after_epoch", type=int, default=10**9,
                        help="Freeze the MAE encoder+decoder after this epoch "
                             "(β keeps updating).  Default 1e9 = disabled.")

    # I/O
    parser.add_argument("--lightweight_outputs", action="store_true")
    parser.add_argument(
        "--legacy_labels_output",
        type=str2bool,
        default=True,
        help="Keep the historical overloaded labels.npy output. New protocol "
             "runners should set false and use predictions.npy/labels_true.npy.",
    )
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


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
#  Masked noise (same as scMAE_family.py)
# ──────────────────────────────────────────────

def apply_mask_noise(x: torch.Tensor, mask_ratio) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply row-shuffle noise to x with given mask_ratio.

    mask_ratio can be a float (constant for the call) or a torch.Tensor scalar
    (e.g. a learnable parameter).  The function uses .item() to extract the
    Python float so torch.bernoulli receives a Python scalar.
    """
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


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_aligned = align_labels(y_true, y_pred)
    return {
        "acc":           float(accuracy_score(y_true, y_aligned)),
        "nmi":           float(normalized_mutual_info_score(y_true, y_pred)),
        "ari":           float(adjusted_rand_score(y_true, y_pred)),
        "f1_macro":      float(f1_score(y_true, y_aligned, average="macro", zero_division=0)),
        "fmi":           float(fowlkes_mallows_score(y_true, y_pred)),
        "v_measure":     float(v_measure_score(y_true, y_pred)),
        "homogeneity":   float(homogeneity_score(y_true, y_pred)),
        "completeness":  float(completeness_score(y_true, y_pred)),
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
#  Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

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

    # Input scaling is an explicit protocol switch.  The historical V9
    # configs default to True; paper-matched AHDPC comparisons can disable it
    # for rows whose published preprocessing is raw.
    if args.scale_input:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_np = scaler.fit_transform(X_raw).astype(np.float32)
    else:
        X_np = X_raw.astype(np.float32, copy=False)

    # Labels
    if y_raw is None:
        # Fully unsupervised path: TopoGate training is self-supervised and does not
        # need labels. n_clusters must be provided via --n_clusters.
        y_np = None
        if args.n_clusters is None:
            raise ValueError(
                f"{args.data_path!r} has no labels and --n_clusters was not provided. "
                "TopoGate is unsupervised; either provide labels in the .npz or pass "
                "--n_clusters explicitly."
            )
        n_clusters = args.n_clusters
    else:
        le = LabelEncoder()
        y_np = le.fit_transform(np.asarray(y_raw).ravel()).astype(np.int64)
        n_clusters = args.n_clusters if args.n_clusters is not None else len(np.unique(y_np))

    save_dir = Path(ensure_dir(args.save_dir))
    save_json(vars(args), str(save_dir / "args.json"))

    # ── HVF feature selection ────────────────────────────────────────────────
    # Remove low-variance / noisy features before PCA to improve kNN quality.
    # HVF must happen BEFORE adaptive PCA dimension selection.
    if args.n_top_features > 0 and args.n_top_features < X_np.shape[1]:
        original_d = X_np.shape[1]
        var = np.var(X_np, axis=0)
        hvf_idx = np.argsort(var)[-args.n_top_features:]
        X_np = X_np[:, hvf_idx]
        # Re-scale after feature selection (mean=0, std=1 per selected feature)
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_np = scaler.fit_transform(X_np).astype(np.float32)
        print(f"[HVF] Reduced from {original_d} to {X_np.shape[1]} dims "
              f"(top {args.n_top_features} high-variance features)")
    # ───────────────────────────────────────────────────────────────────────

    device = get_device(args.gpu, args.no_cuda)
    print(f"[{dataset_name}] device={device}  n={X_np.shape[0]}  d={X_np.shape[1]}  K={n_clusters}", flush=True)

    # ── Determine kNN PCA dim ────────────────────────────────────────────────
    knn_pca_dim = min(args.knn_pca_dim, X_np.shape[1])
    if args.knn_pca_mode == "adaptive":
        actual_dim = select_adaptive_pca_dim(X_np, max_dim=knn_pca_dim, seed=args.seed)
        print(f"[Adaptive PCA] Selected dim={actual_dim} (retains ≥95% variance, "
              f"upper_bound={knn_pca_dim})")
        knn_pca_dim = actual_dim
    else:
        print(f"[Fixed PCA] Using knn_pca_dim={knn_pca_dim}")

    # Build topology graph
    graph = build_pca_knn_graph(
        X_np, k=args.neighbor_k,
        pca_dim=knn_pca_dim,
        tau=args.tau, seed=args.seed,
        n_top_features=0,  # HVF already done above
    )
    # v3: LearnableEdgeReliability (optional).  When enabled, replaces the
    # argparse-fixed 4 gammas with nn.Parameter.
    learnable_edge_module = None
    if args.learnable_gamma and args.edge_reliability_mode != "none":
        learnable_edge_module = LearnableEdgeReliability(
            mode=args.edge_reliability_mode,
            init_gamma_sim=args.gamma_sim,
            init_gamma_mutual=args.gamma_mutual,
            init_gamma_snn=args.gamma_snn,
            init_gamma_distance=args.gamma_distance,
            reg_weight=args.gamma_reg_weight,
        ).to(device)
        edge_rel_t, edge_weights_t = learnable_edge_module(graph)
        edge_weights = edge_weights_to_numpy(edge_weights_t)
        from methods.TopoGate.learnable_gate.learnable_edge_reliability import summarize_edge_weights_torch
        edge_summary = summarize_edge_weights_torch(edge_weights_t)
    else:
        _, edge_weights, edge_summary = compute_edge_reliability(
            graph, mode=args.edge_reliability_mode,
            gamma_sim=args.gamma_sim, gamma_mutual=args.gamma_mutual,
            gamma_snn=args.gamma_snn, gamma_distance=args.gamma_distance,
        )
    node_gate, _, gate_summary = compute_node_gate(
        graph, edge_weights=edge_weights,
        gate_mode=args.gate_mode if args.gate_mode != "learned" else "none",
        gate_min=args.gate_min, gate_max=args.gate_max,
        beta_mutual=args.beta_mutual, beta_snn=args.beta_snn,
        beta_perturb=args.beta_perturb, beta_uncertainty=args.beta_uncertainty,
        uncertainty=None,
    )
    # Gate module placeholders — will be initialised after model + uncertainty are ready
    learned_gate_module = None
    learned_gate_stats = None
    learned_gate_static = None
    binary_router_module = None
    binary_router_stats = None
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

    # V12: compute an unsupervised local-risk attenuation from the frozen graph.
    # The proxy does not inspect y: it combines disagreement in mutual-neighbor
    # support with the existing perturbation statistic.  High-risk nodes retain
    # the self/feature branch and receive less topology mixing; low-risk nodes
    # keep the original V9 gate.  This is deliberately configuration-gated so
    # V9 remains bit-for-bit reachable through risk_adaptive_mix=false.
    risk_scale_np = np.ones(X_np.shape[0], dtype=np.float32)
    risk_summary = {
        "enabled": bool(args.risk_adaptive_mix),
        "temperature": float(args.risk_adaptive_temperature),
        "mean_local_risk": 0.0,
        "mean_topology_trust": 1.0,
        "p10_topology_trust": 1.0,
        "p90_topology_trust": 1.0,
    }
    if args.risk_adaptive_temperature <= 0:
        raise ValueError("risk_adaptive_temperature must be positive")
    if args.risk_adaptive_mix:
        mutual_support = np.asarray(graph.mutual.mean(axis=1), dtype=np.float32)
        perturb_proxy = (1.0 - np.sum(graph.probs * graph.similarity, axis=1)).astype(np.float32)
        local_risk = 0.5 * (1.0 - np.clip(mutual_support, 0.0, 1.0)) + 0.5 * np.clip(perturb_proxy, 0.0, 1.0)
        risk_scale_np = np.exp(-local_risk / float(args.risk_adaptive_temperature)).astype(np.float32)
        risk_summary = {
            "enabled": True,
            "temperature": float(args.risk_adaptive_temperature),
            "mean_local_risk": float(np.mean(local_risk)),
            "mean_topology_trust": float(np.mean(risk_scale_np)),
            "p10_topology_trust": float(np.percentile(risk_scale_np, 10)),
            "p90_topology_trust": float(np.percentile(risk_scale_np, 90)),
        }

    # DataLoaders
    dataset = TensorDataset(X_np, y_np)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, generator=gen)
    eval_loader  = DataLoader(dataset, batch_size=max(args.batch_size * 4, 512), shuffle=False)

    # Model
    model = AutoEncoder(
        num_genes=X_np.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        masked_data_weight=args.masked_data_weight,
        mask_loss_weight=args.mask_loss_weight,
    ).to(device)

    # ── MC Dropout uncertainty ───────────────────────────────────────────────
    # Compute structural instability per node using the untrained encoder's latent
    # variance.  This replaces the previously hardcoded uncertainty=None, making
    # the 4th topology stat meaningful for the LearnableGate.
    uncertainty_np = None
    if args.gate_mode == "learned" and args.enhanced_stats >= 4:
        print(f"[Uncertainty] Computing MC dropout uncertainty (n_passes=5)...")
        model.eval()
        with torch.no_grad():
            all_unc = []
            for idx_batch, x_batch, _ in eval_loader:
                x_t = x_batch.to(device)
                unc = compute_mc_dropout_uncertainty(model, x_t, n_passes=5, device=device)
                all_unc.append(unc)
        uncertainty_np = np.concatenate(all_unc)  # (n,)
        u_mean = float(np.mean(uncertainty_np))
        u_std = float(np.std(uncertainty_np))
        print(f"[Uncertainty] done. mean={u_mean:.4f} std={u_std:.4f} "
              f"range=[{float(np.min(uncertainty_np)):.4f}, {float(np.max(uncertainty_np)):.4f}]")
        model.train()
    # ───────────────────────────────────────────────────────────────────────

    # ── LearnableGate initialisation (requires model + uncertainty) ──────────
    if args.gate_mode == "learned":
        if args.learned_gate_init_mode == "v1_default":
            init_m, init_s, init_p, init_u = (
                args.beta_mutual, args.beta_snn, args.beta_perturb, args.beta_uncertainty
            )
        elif args.learned_gate_init_mode == "nomix":
            # nomix_init: all betas start at -1.5 → gate ≈ 0.018 (≈ NoMix)
            # gate = gate_min + (gate_max - gate_min) * sigmoid(betas * stats)
            # with stats≈1: sigmoid(-1.5) ≈ 0.018 → gate ≈ 0.15 * 0.018 ≈ 0.0027
            # This is NOT exactly zero, but close to NoMix. Gradient is strong:
            # sigmoid'(-1.5) ≈ 0.018 (vs 3e-7 at -5.0 — 50,000x improvement).
            init_m = init_s = init_p = init_u = -1.5
        else:
            init_m = init_s = init_p = init_u = 0.0
        learned_gate_module = LearnableGate(
            gate_min=args.gate_min, gate_max=args.gate_max,
            init_beta_mutual=init_m, init_beta_snn=init_s,
            init_beta_perturb=init_p, init_beta_uncertainty=init_u,
            learnable_gate_max=bool(args.learnable_gate_max),
            gate_max_min=args.gate_max_min,
            gate_max_max=args.gate_max_max,
            enhanced_stats=int(args.enhanced_stats),
        ).to(device)
        learned_gate_stats = build_gate_stats_tensor(
            graph.indices, graph.mutual, graph.snn, graph.probs, graph.similarity,
            uncertainty=uncertainty_np, device=device,
            enhanced_stats=int(args.enhanced_stats),
        )
        learned_gate_static = node_gate.copy()
        gate_summary = {
            **gate_summary,
            "learned_gate_init_mode": args.learned_gate_init_mode,
            "warmup_epochs": int(args.warmup_epochs),
            "ramp_epochs": int(args.ramp_epochs),
            "init_beta_mutual": float(init_m),
            "init_beta_snn": float(init_s),
            "init_beta_perturb": float(init_p),
            "init_beta_uncertainty": float(init_u),
            "uncertainty_computed": uncertainty_np is not None,
            "use_beta_scale_schedule": bool(args.use_beta_scale_schedule),
        }
    # ── BinaryRouter initialisation (requires model + uncertainty) ──────────
    if args.gate_mode == "binary":
        binary_router_module = BinaryRouter(
            temperature_init=float(args.router_init_temp),
            temperature_min=float(args.router_temp_min),
            warmup_epochs=int(args.router_warmup_epochs),
            ramp_epochs=int(args.router_ramp_epochs),
            enhanced_stats=int(args.enhanced_stats),
            init_beta_mutual=float(args.init_beta_mutual),
            init_beta_snn=float(args.init_beta_snn),
            init_beta_perturb=float(args.init_beta_perturb),
            init_beta_uncertainty=float(args.init_beta_uncertainty),
        ).to(device)
        binary_router_stats = build_gate_stats_tensor(
            graph.indices, graph.mutual, graph.snn, graph.probs, graph.similarity,
            uncertainty=uncertainty_np, device=device,
            enhanced_stats=int(args.enhanced_stats),
        )
        gate_summary = {
            **gate_summary,
            "gate_mode": "binary",
            "router_temperature_init": float(args.router_init_temp),
            "router_temperature_min": float(args.router_temp_min),
            "router_warmup_epochs": int(args.router_warmup_epochs),
            "router_ramp_epochs": int(args.router_ramp_epochs),
        }
    # ───────────────────────────────────────────────────────────────────────

    # Two param-groups so we can freeze the MAE encoder/decoder without
    # touching the LearnableGate β params.  Group 0 = MAE, Group 1 = gates.
    mae_params = list(model.parameters())
    gate_params = list(learned_gate_module.parameters()) if learned_gate_module is not None else []
    binary_router_params = list(binary_router_module.parameters()) if binary_router_module is not None else []
    edge_params = list(learnable_edge_module.parameters()) if learnable_edge_module is not None else []
    gate_params = gate_params + binary_router_params  # both get the amplified lr
    # v3: AdaptiveMaskRatio — promote mask_ratio to a learnable scalar parameter
    # initialised at args.mask_ratio, clamped to [mask_ratio_min, mask_ratio_max].
    learnable_mask_ratio_module = None
    mask_ratio_param = None
    mask_params = []
    if args.learnable_mask_ratio:
        import torch.nn as nn
        learnable_mask_ratio_module = nn.Module()
        # initial value: logit((init - min) / (max - min))
        with torch.no_grad():
            span = max(args.mask_ratio_max - args.mask_ratio_min, 1e-6)
            p0 = max(min((args.mask_ratio - args.mask_ratio_min) / span, 1.0 - 1e-4), 1e-4)
            raw0 = float(np.log(p0 / (1.0 - p0)))
        learnable_mask_ratio_module.mask_ratio_raw = nn.Parameter(torch.tensor(raw0))
        mask_params = list(learnable_mask_ratio_module.parameters())
        def mask_ratio_value():
            span = max(args.mask_ratio_max - args.mask_ratio_min, 1e-6)
            return args.mask_ratio_min + span * torch.sigmoid(
                learnable_mask_ratio_module.mask_ratio_raw
            )
        mask_ratio_param = mask_ratio_value()
    optimizer = torch.optim.Adam([
        {"params": mae_params, "lr": args.lr},
        # v3: amplify gate group lr (default 10x).  The β/gate_max gradient
        # travels through pseudo_weight(0.3) * gate(<0.15) ≈ 4.5% of the loss
        # signal, so without amplification it cannot compete with MAE loss.
        {"params": gate_params, "lr": args.lr * float(args.gate_lr_multiplier)},
        # v3: LearnableEdgeReliability params share the MAE lr (they affect
        # the pseudo branch via edge_weights, which gets ~30% gradient).
        {"params": edge_params, "lr": args.lr},
        # v3: AdaptiveMaskRatio shares the MAE lr.  Its gradient flows through
        # both real_loss and pseudo_loss (every apply_mask_noise call).
        {"params": mask_params, "lr": args.lr},
    ])
    # Easier approach: just track which params belong to MAE so we can zero out
    # their gradients after freeze_mae_after_epoch.  Using requires_grad toggle
    # also works but PyTorch prints a redundant lr=0 warning.
    mae_param_ids = {id(p) for p in mae_params}

    pseudo_enabled = args.mix_mode != "none" and float(args.pseudo_weight) > 0

    # Beta history for post-hoc analysis.  Recorded every epoch.  Only
    # active when gate_mode == 'learned'.
    beta_history = []

    # Training loop
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if learned_gate_module is not None:
            t = max(0.0, min(1.0, (epoch - args.warmup_epochs) / max(1, args.ramp_epochs)))
            # Keep V9 reproducible by leaving beta_scale=1 unless the legacy
            # nomix-warmup experiment explicitly opts into the later schedule.
            # Note: beta_scale=0 also makes d(gate)/d(beta)=0; beta parameters do
            # not learn from the pseudo branch during that interval.
            if not args.use_beta_scale_schedule:
                beta_scale_val = 1.0
            elif epoch < args.warmup_epochs:
                beta_scale_val = 0.0
            elif epoch < args.warmup_epochs + args.ramp_epochs:
                beta_scale_val = (epoch - args.warmup_epochs) / max(1, args.ramp_epochs)
            else:
                beta_scale_val = 1.0
            learned_gate_module.beta_scale.fill_(beta_scale_val)
        else:
            t = 1.0
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        # ── MAE freezing ────────────────────────────────────────────────────
        # After freeze_mae_after_epoch, the encoder/decoder no longer update.
        # Implemented by zeroing out gradients of MAE params after loss.backward()
        # but BEFORE optimizer.step().  We track this with `_mae_frozen`.
        # BinaryRouter is also frozen at this point (like learned_gate_module).
        mae_frozen = (
            (learned_gate_module is not None or binary_router_module is not None)
            and args.freeze_mae_after_epoch >= 0
            and epoch > args.freeze_mae_after_epoch
        )
        for idx_t, x_cpu, _ in train_loader:
            idx_np = idx_t.numpy().astype(np.int64)
            x = x_cpu.to(device)

            # Real branch
            x_corrupt, real_mask = apply_mask_noise(x, mask_ratio_param if mask_ratio_param is not None else args.mask_ratio)
            _, real_loss, _ = model.loss_mask_weighted(x_corrupt, x, real_mask)
            loss = real_loss

            # Pseudo branch
            pseudo_loss = torch.zeros((), dtype=real_loss.dtype, device=device)
            if pseudo_enabled:
                router_tensor_pass = None
                if learned_gate_module is not None:
                    batch_stats = learned_gate_stats[idx_t]
                    gate_dyn = learned_gate_module(batch_stats)
                    if t < 1.0:
                        gate_static_t = torch.as_tensor(
                            learned_gate_static[idx_np],
                            dtype=gate_dyn.dtype, device=gate_dyn.device,
                        )
                        gate_eff = (1.0 - t) * gate_static_t + t * gate_dyn
                    else:
                        gate_eff = gate_dyn
                    if args.risk_adaptive_mix:
                        gate_eff = gate_eff * torch.as_tensor(
                            risk_scale_np[idx_np], dtype=gate_eff.dtype, device=gate_eff.device
                        )
                    gate_tensor_pass = gate_eff
                    x_prime, sample_weight, _ = make_pseudo_batch(
                        data_np=X_np, batch_indices=idx_np, batch_x=x,
                        mix_mode=args.mix_mode, graph=graph,
                        edge_weights=edge_weights, node_gate=node_gate,
                        mix_neighbors=args.mix_neighbors, rng=rng,
                        random_neighbors=random_neighbors,
                        far_neighbors=far_neighbors,
                        neighbor_estimator=args.neighbor_estimator,
                        gate_tensor=gate_tensor_pass,
                    )
                elif binary_router_module is not None:
                    # Direction B: BinaryRouter — hard routing between anchor and mixed
                    batch_stats = binary_router_stats[idx_t]
                    # Use Gumbel-Softmax during training; argmax during inference
                    router_tensor_pass = binary_router_module(batch_stats, epoch=epoch, hard=False)
                    x_prime, sample_weight, _ = make_pseudo_batch_binary(
                        data_np=X_np, batch_indices=idx_np, batch_x=x,
                        mix_mode=args.mix_mode, graph=graph,
                        edge_weights=edge_weights,
                        mix_neighbors=args.mix_neighbors, rng=rng,
                        random_neighbors=random_neighbors,
                        far_neighbors=far_neighbors,
                        neighbor_estimator=args.neighbor_estimator,
                        router_tensor=router_tensor_pass,
                    )
                else:
                    # static gate path (gate_mode != learned && != binary)
                    x_prime, sample_weight, _ = make_pseudo_batch(
                        data_np=X_np, batch_indices=idx_np, batch_x=x,
                        mix_mode=args.mix_mode, graph=graph,
                        edge_weights=edge_weights, node_gate=node_gate,
                        mix_neighbors=args.mix_neighbors, rng=rng,
                        random_neighbors=random_neighbors,
                        far_neighbors=far_neighbors,
                        neighbor_estimator=args.neighbor_estimator,
                        gate_tensor=None,
                    )
                xp_corrupt, pseudo_mask = apply_mask_noise(x_prime, mask_ratio_param if mask_ratio_param is not None else args.mask_ratio)
                _, pseudo_loss, _ = model.loss_mask_weighted(
                    xp_corrupt, x, pseudo_mask,
                    sample_weight=sample_weight,
                )
                loss = loss + float(args.pseudo_weight) * pseudo_loss
            # v3: LearnableEdgeReliability L2 regularisation.  Keeps the 4
            # gammas from drifting to extreme values that break neighbour sampling.
            if learnable_edge_module is not None:
                edge_reg = learnable_edge_module.regularization_loss()
                loss = loss + edge_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if mae_frozen:
                # MAE is frozen at this epoch.  Zero out the MAE params'
                # gradients so they don't update.  The gate params keep theirs.
                for pg in optimizer.param_groups:
                    if not pg["params"]:
                        continue
                    for p in pg["params"]:
                        if p.grad is not None and id(p) in mae_param_ids:
                            p.grad = None
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        # ── β history logging ─────────────────────────────────────────────
        if learned_gate_module is not None:
            beta_history.append({
                "epoch": epoch,
                "schedule_t": float(t),
                "mae_frozen": bool(mae_frozen),
                **learned_gate_module.beta_snapshot(),
            })
        elif binary_router_module is not None:
            beta_history.append({
                "epoch": epoch,
                "temperature": float(binary_router_module._temperature(epoch)),
                "mae_frozen": bool(mae_frozen),
                **binary_router_module.beta_snapshot(),
            })

        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            sched = f"  sched_t={t:.2f}" if learned_gate_module is not None else ""
            fz = "  [MAE-frozen]" if mae_frozen else ""
            print(f"  [{dataset_name}] epoch {epoch:03d}/{args.epochs}  loss={epoch_loss/max(1,n_batches):.4f}{sched}{fz}", flush=True)

    train_time = time.time() - t0

    # Extract embedding
    model.eval()
    embeddings, labels_out = [], []
    with torch.no_grad():
        for _, x, y in eval_loader:
            z = model.feature(x.to(device))
            embeddings.append(z.detach().cpu().numpy())
            labels_out.append(y.numpy())
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

    # Save
    np.save(save_dir / "embedding_final.npy", embedding)
    np.save(save_dir / "predictions.npy", pred.astype(np.int64))
    if y_np is not None:
        np.save(save_dir / "labels_true.npy", y_np.astype(np.int64))
    if args.legacy_labels_output:
        # Preserve the historical contract for old callers. The manifest-driven
        # protocol sets this flag false so labels and predictions cannot be
        # confused by downstream analysis.
        np.save(save_dir / "labels.npy", y_np.astype(np.int64) if y_np is not None else pred.astype(np.int64))
    if not args.lightweight_outputs:
        np.save(save_dir / "embeddings_base.npy", embedding)
    save_json(metrics, str(save_dir / "metrics.json"))
    # Metrics are saved to metrics.json separately.  Copy them into the
    # top-level json so callers get everything in one file.
    metrics_path = save_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            _metrics = json.load(f)
    else:
        _metrics = {}
    save_json({
        "dataset": dataset_name,
        "method": args.method_name,
        "variant": args.variant_name,
        "seed": int(args.seed),
        "n_samples": int(X_np.shape[0]),
        "n_features": int(X_np.shape[1]),
        "n_clusters": int(n_clusters),
        "labels_used_during_fit": False,
        "k_source": "explicit_n_clusters",
        "legacy_labels_output": bool(args.legacy_labels_output),
        "mix_mode": args.mix_mode,
        "pseudo_weight": float(args.pseudo_weight),
        "gate_mode": args.gate_mode,
        "train_seconds": float(train_time),
        "freeze_mae_after_epoch": int(args.freeze_mae_after_epoch),
        "metrics": _metrics,
        "edge_reliability_summary": edge_summary,
        "gate_summary": gate_summary,
        "learned_gate_final_beta": (
            learned_gate_module.beta_snapshot() if learned_gate_module is not None else None
        ),
        "binary_router_final_beta": (
            binary_router_module.beta_snapshot() if binary_router_module is not None else None
        ),
        "learned_edge_final_gamma": (
            learnable_edge_module.gamma_snapshot() if learnable_edge_module is not None else None
        ),
        "learned_mask_ratio": (
            float(mask_ratio_param.detach().cpu()) if mask_ratio_param is not None else None
        ),
        "learned_gate_beta_history": beta_history if learned_gate_module is not None else None,
        "binary_router_beta_history": beta_history if binary_router_module is not None else None,
        "risk_adaptive_mix": bool(args.risk_adaptive_mix),
        "risk_adaptive_temperature": float(args.risk_adaptive_temperature),
        "risk_summary": risk_summary,
    }, str(save_dir / "summary.json"))

    _f1 = metrics.get('f1_macro')
    f1_str = f"{_f1:.4f}" if _f1 is not None else "N/A"
    print(
        f"[{dataset_name}] done  "
        f"{metrics_str}  "
        f"F1={f1_str}  time={train_time:.1f}s",
        flush=True,
    )


def _load_variant_config(variant_name: str, config_dir: str | Path | None = None) -> dict:
    """Load <variant>.yaml from a configs directory and return flat dict of overrides.

    Args:
        variant_name: e.g. 'learnable_gate_sched' or 'static_gate_full'.  The `.yaml` suffix is appended.
        config_dir:   directory containing the YAML.  Defaults to
                      `methods/TopoGate/learnable_gate/configs/` (the learnable_gate default location).

    The YAML only contains a handful of keys (mix_mode / gate_mode / …).
    Anything not in the YAML falls back to argparse defaults defined in parse_args().
    """
    import yaml as _yaml
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent / "configs"
    cfg_path = Path(config_dir) / f"{variant_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"variant config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f) or {}
    return cfg


def run_topogate(X, n_clusters, y=None, gpu=4, variant="topogate_learnable_gate", save_dir=None,
                 seed=42, return_metrics=False, config_dir=None, **overrides):
    """Public entry point that wraps run_npz.main() with dataset-name + save_dir handling.

    Args:
        X: (n_samples, n_features) feature matrix (np.ndarray / torch.Tensor).
        n_clusters: int, number of clusters (K).
        gpu: physical GPU id (forbidden: 0, 7).
        variant: e.g. 'learnable_gate_sched' (main) or 'static_gate_full'.
                 The '.yaml' is appended; set config_dir to point at the right location.
        save_dir: optional output dir (defaults to a tmp dir under /tmp/topogate_<ts>).
        seed: random seed.
        return_metrics: if True, returns (labels, runtime, metrics_dict).
        config_dir: directory containing the variant YAML.  Defaults to
                    `methods/TopoGate/learnable_gate/configs/` (learnable_gate default).  Pass
                    `methods/TopoGate/static_gate/configs/` for v1 ablation variants.
        **overrides: any CLI arg of run_npz.parse_args() can be overridden by camelCase or snake_case key.

    Returns:
        (labels, runtime) by default; (labels, runtime, metrics_dict) if return_metrics=True.

    Strategy: drive main() via injected sys.argv. We build a complete argv from
    YAML + overrides, inject it, then call main() (which calls parse_args() again
    on the injected argv — argparse is idempotent). Algorithm code in main() is
    NEVER modified; this function is purely a wrapper.
    """
    import argparse
    import tempfile
    import shutil
    import io
    import contextlib

    yaml_cfg = _load_variant_config(variant, config_dir=config_dir)

    # These are wrapper-level metadata controls rather than model parameters.
    # Keeping them explicit prevents the protocol runner from inheriting the
    # legacy ``adhoc`` dataset name and overloaded labels.npy output.
    dataset_name_override = overrides.pop("dataset_name", None)

    # Merge: YAML < explicit overrides (overrides win)
    cli_args = dict(yaml_cfg)
    for k, v in overrides.items():
        snake = ''.join('_' + c.lower() if c.isupper() else c for c in k).lstrip('_')
        cli_args[snake] = v

    # Decide data_path / save_dir (use placeholders for the build phase)
    # Persist X (and optional y) to a temp .npz (main() needs ground-truth for metrics)
    tmp_npz = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp_npz.close()
    save_kwargs = {"X": np.asarray(X, dtype=np.float64)}
    if y is not None:
        save_kwargs["y"] = np.asarray(y).ravel()
    np.savez(tmp_npz.name, **save_kwargs)

    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix=f"topogate_{variant}_")
        _cleanup_tmp = True
    else:
        save_dir = str(save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _cleanup_tmp = False

    # Build the final argv (used by main() → parse_args())
    full_argv = ["run_npz.py"]
    action_flags = {"lightweight_outputs", "no_cuda"}
    for k, v in cli_args.items():
        if k in action_flags:
            if bool(v):
                full_argv.append(f"--{k}")
            continue
        full_argv += [f"--{k}", str(v)]
    full_argv += [
        "--data_path", tmp_npz.name,
        "--save_dir", save_dir,
        "--dataset_name", str(dataset_name_override or "adhoc"),
        "--variant_name", variant,
        "--method_name", "TopoGate",
        "--n_clusters", str(int(n_clusters)),
        "--seed", str(int(seed)),
        "--gpu", str(int(gpu)),
    ]

    # Inject argv, run main(), restore argv
    saved_argv = sys.argv
    sys.argv = full_argv
    buf = io.StringIO()
    try:
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            main()
        elapsed = time.time() - t0
    finally:
        sys.argv = saved_argv

    # Read metrics.json written by main()
    metrics_path = Path(save_dir) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # main() saves embedding_final.npy but not pred_labels — re-run KMeans to recover them
    emb_path = Path(save_dir) / "embedding_final.npy"
    if emb_path.exists():
        embedding = np.load(emb_path)
        km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(seed))
        pred_labels = km.fit_predict(embedding)
    else:
        pred_labels = np.array([], dtype=np.int64)

    # Cleanup tmp artefacts
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
