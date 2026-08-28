#!/usr/bin/env python
"""v5 model — independent runner.

Uses v5 modules (Gumbel-Sigmoid + STE mask, single-γ edge reliability) in
a STANDALONE training loop that mirrors v3's structure exactly:

  - Same MAE encoder/decoder (AutoEncoder from learnable_gate.model)
  - Same loss function (model.loss_mask_weighted)
  - Same pseudo branch (make_pseudo_batch from learnable_gate.mixing)
  - Same optimizer (Adam with lr groups for gate/edge/mask/mae)
  - Same clustering eval (KMeans + ARI)

The ONLY additions vs v3:
  - apply_mask_noise uses Gumbel-Sigmoid + STE (so mask_ratio can learn)
  - LearnableEdgeReliability replaced with single-γ version

v3 code is NOT modified.  This runner is a separate file using v3 components
and overriding only the 2 modules with v5 equivalents.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # ToPoGate/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "methods" / "TopoGate"))

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch
import torch.nn as nn

from learnable_gate.v5_components.mask_noise_v5 import (
    apply_mask_noise_v5_ste,
    mask_ratio_alignment_loss,
)
from learnable_gate.v5_components.learnable_edge_reliability_v5 import (
    LearnableEdgeReliabilityV5,
)
from learnable_gate.v5_components.per_sample_mask_v5 import (
    apply_mask_noise_v5_per_sample,
    compute_sample_salience,
    per_sample_mask_ratio_reg_loss,
)

from learnable_gate.neighbor_graph import build_pca_knn_graph
from learnable_gate.learnable_gate import LearnableGate, build_gate_stats_tensor
from learnable_gate.mixing import make_pseudo_batch
from learnable_gate.learnable_edge_reliability import (
    edge_weights_to_numpy,
    summarize_edge_weights_torch as _v3_summary,
)
from learnable_gate.model import AutoEncoder

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = torch.as_tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx, self.X[idx], 0


def cluster_and_eval(emb, y, K, seed):
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    pred = km.fit_predict(emb)
    if y is not None:
        ari = adjusted_rand_score(y, pred)
        nmi = normalized_mutual_info_score(y, pred)
        cm = np.zeros((K, K), dtype=int)
        for i, j in zip(y, pred):
            cm[i, j] += 1
        ri, ci = linear_sum_assignment(-cm)
        acc = cm[ri, ci].sum() / len(pred)
        return ari, nmi, acc, pred
    return 0.0, 0.0, 0.0, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--method_name", default="TopoGate_v5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0)
    parser.add_argument("--neighbor_k", type=int, default=10)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--mask_alignment_weight", type=float, default=0.1)
    parser.add_argument("--v5_gamma_mode", default="one_param_scalar",
                        choices=["all_params_4f", "one_param_scalar",
                                 "one_fixed_one_learnable"])
    parser.add_argument("--gamma_reg_weight", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--ramp_epochs", type=int, default=10)
    parser.add_argument("--gate_mode", default="learned")
    parser.add_argument("--mask_ratio_learnable", action="store_true")
    parser.add_argument("--mask_ratio_min", type=float, default=0.1)
    parser.add_argument("--mask_ratio_max", type=float, default=0.6)
    parser.add_argument("--mask_ratio_init", type=float, default=0.0,
                        help="Initial logit for mask_ratio_raw (sigmoid(0)=0.5 -> mask=midpoint). "
                             "Use -5 to start near mask_ratio_min (low noise).")
    parser.add_argument("--per_sample_mask", action="store_true",
                        help="Phase 3-A: per-sample adaptive mask ratio (SBAM-style). "
                             "Mask ratio = mask_base + mask_scale * salience_i.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--mae_depth", type=int, default=2)
    parser.add_argument("--pseudo_weight", type=float, default=0.3)
    parser.add_argument("--mix_mode", default="reliability")
    parser.add_argument("--knn_pca_dim", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--mask_loss_weight", type=float, default=0.7)
    parser.add_argument("--masked_data_weight", type=float, default=0.75)
    parser.add_argument("--mask_ratio_eps", type=float, default=1e-5)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device selection
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        device = torch.device("cuda:0")
    elif args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    print(f"[{args.dataset_name}] device={device} (physical gpu={args.gpu})", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data_path)
    X = data["X"] if "X" in data.files else data["x"]
    y = data["y"] if "y" in data.files else None
    if y is not None:
        # Remap y to 0..K-1 (datasets may have non-contiguous labels)
        y_unique = np.unique(y)
        K = int(y_unique.size)
        y = np.searchsorted(y_unique, y).astype(np.int64)
    else:
        K = max(10, int(round(math.sqrt(X.shape[0]))))
    print(f"[{args.dataset_name}] N={X.shape[0]} d={X.shape[1]} K={K}", flush=True)

    # Standardize (matches v3 run_npz.py)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X).astype(np.float32)

    # Build graph
    graph = build_pca_knn_graph(
        X, k=args.neighbor_k, pca_dim=min(args.knn_pca_dim, X.shape[1] - 1),
        tau=args.tau, seed=args.seed,
    )

    # v5 edge module
    edge_module = LearnableEdgeReliabilityV5(
        mode=args.v5_gamma_mode,
        init_gamma=1.0,
        reg_weight=args.gamma_reg_weight,
    ).to(device)
    with torch.no_grad():
        _, weights_t = edge_module(graph)
    edge_weights_np = edge_weights_to_numpy(weights_t)
    gate_static = edge_weights_np.copy()

    # v5 learnable gate
    gate_module = LearnableGate().to(device)
    gate_stats = build_gate_stats_tensor(graph.indices, graph.mutual, graph.snn,
                                         graph.probs, graph.similarity).to(device)
    with torch.no_grad():
        gate_static_t = torch.as_tensor(gate_static, dtype=torch.float32, device=device)
        node_gate = gate_module(gate_stats).detach().cpu().numpy()

    # Mask ratio module (v5: learnable with STE)
    if args.mask_ratio_learnable:
        mask_ratio_raw = nn.Parameter(torch.tensor(float(args.mask_ratio_init), device=device))
    else:
        mask_ratio_raw = None

    # Phase 3-A: per-sample adaptive mask ratio (SBAM-style)
    # mask_base + mask_scale * salience_i, clipped to [mask_ratio_min, mask_ratio_max]
    per_sample_mask_params = None
    salience_tensor = None
    if args.per_sample_mask:
        # Compute per-sample salience once (CPU kNN on graph embedding)
        sal_np = compute_sample_salience(
            torch.as_tensor(graph.embedding, dtype=torch.float32),
            precomputed=None, k=min(10, X.shape[0] - 1),
        ).numpy()
        salience_tensor = torch.as_tensor(sal_np, dtype=torch.float32, device=device)
        # Two learnable scalars: mask_base (initial 0 -> sigmoid(0)=0.5 -> mask=mid)
        # and mask_scale (initial 0 -> scale=0 -> mask=base everywhere)
        # We pre-init mask_base so that mask_base ≈ mask_ratio_min + 0.5*span.
        span_init = max(args.mask_ratio_max - args.mask_ratio_min, 1e-6)
        target_init = (args.mask_ratio_min + 0.5 * span_init)
        p_init = (target_init - args.mask_ratio_min) / span_init
        p_init = min(max(p_init, 1e-4), 1.0 - 1e-4)
        raw_init = float(np.log(p_init / (1.0 - p_init))) if p_init > 0 else 0.0
        mask_base_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32, device=device))
        mask_scale_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
        per_sample_mask_params = [mask_base_raw, mask_scale_raw]

    # MAE
    model = AutoEncoder(
        num_genes=X.shape[1], hidden_size=args.hidden_size,
        dropout=0.0,
        mask_loss_weight=args.mask_loss_weight,
        masked_data_weight=args.masked_data_weight,
    ).to(device)

    # Optimizer groups
    mae_params = list(model.parameters())
    gate_params = list(gate_module.parameters())
    edge_params = list(edge_module.parameters())
    mask_params = [mask_ratio_raw] if mask_ratio_raw is not None else []
    if per_sample_mask_params is not None:
        mask_params.extend(per_sample_mask_params)
    optimizer = torch.optim.Adam([
        {"params": mae_params, "lr": args.lr},
        {"params": gate_params, "lr": args.lr * args.gate_lr_multiplier},
        {"params": edge_params, "lr": args.lr},
        {"params": mask_params, "lr": args.lr * args.gate_lr_multiplier},
    ])

    # Loader
    ds = TabularDataset(X)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False,
    )

    # Training loop
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        t = max(0.0, min(1.0, (epoch - args.warmup_epochs) / max(1, args.ramp_epochs)))
        ep_loss = 0.0
        ep_count = 0
        for idx_t, x_cpu, _ in loader:
            x = x_cpu.to(device)
            idx_np = idx_t.numpy().astype(np.int64)

            # Compute mask_ratio (per-row if per_sample_mask enabled, else scalar)
            if per_sample_mask_params is not None:
                # mask_ratio_per_row_i = mask_base + mask_scale * salience_i
                mask_base_raw, mask_scale_raw = per_sample_mask_params
                span = max(args.mask_ratio_max - args.mask_ratio_min, 1e-6)
                mask_base = args.mask_ratio_min + span * torch.sigmoid(mask_base_raw)
                # scale = ±span*0.5 (so per-row ratio stays in [mask_base - 0.5*span, mask_base + 0.5*span])
                mask_scale = 0.5 * span * torch.tanh(mask_scale_raw)
                # Salience for this batch
                sal_batch = salience_tensor[idx_t]
                mask_ratio_per_sample = (mask_base + mask_scale * sal_batch).clamp(
                    min=args.mask_ratio_min, max=args.mask_ratio_max
                )
                mask_ratio_scalar_for_logging = float(mask_ratio_per_sample.mean().detach().cpu())
            elif mask_ratio_raw is not None:
                span = max(args.mask_ratio_max - args.mask_ratio_min, 1e-6)
                mask_ratio_param = args.mask_ratio_min + span * torch.sigmoid(mask_ratio_raw)
                mask_ratio_scalar_for_logging = float(mask_ratio_param.detach().cpu())
            else:
                mask_ratio_param = torch.tensor(args.mask_ratio, device=device)
                mask_ratio_scalar_for_logging = args.mask_ratio

            # === Real branch ===
            if per_sample_mask_params is not None:
                x_corrupt, real_mask_hard, y_soft_real = apply_mask_noise_v5_per_sample(
                    x, mask_ratio_per_sample, temperature=1.0
                )
            else:
                x_corrupt, real_mask_hard, y_soft_real = apply_mask_noise_v5_ste(
                    x, mask_ratio_param, temperature=1.0
                )
            # STE: forward uses hard mask, backward uses soft mask
            real_mask_ste = real_mask_hard + (y_soft_real - y_soft_real.detach())
            _, real_loss, _ = model.loss_mask_weighted(x_corrupt, x, real_mask_ste)
            loss = real_loss

            # === Pseudo branch ===
            if args.mix_mode != "none" and args.pseudo_weight > 0:
                # Re-compute edge weights to track gradient
                _, weights_t = edge_module(graph)
                edge_weights = weights_t.detach().cpu().numpy()
                # (Use trainable edge_weights_t for downstream gradient)
                with torch.no_grad():
                    weights_t_for_loss = weights_t
                # update gate_dynamic — batch-wise (matches v3)
                gate_stats_batch = gate_stats[idx_t]
                gate_dyn = gate_module(gate_stats_batch)
                # Gate tensor for adaptive mixing
                if t < 1.0:
                    gate_static_t_batch = torch.as_tensor(
                        node_gate[idx_np],
                        dtype=torch.float32, device=device
                    )
                    gate_eff = (1.0 - t) * gate_static_t_batch + t * gate_dyn
                else:
                    gate_eff = gate_dyn
                # Build pseudo batch
                x_prime, sample_weight, _ = make_pseudo_batch(
                    data_np=X, batch_indices=idx_np, batch_x=x,
                    mix_mode=args.mix_mode, graph=graph,
                    edge_weights=edge_weights, node_gate=node_gate,
                    mix_neighbors=args.neighbor_k, rng=rng,
                    random_neighbors=None, far_neighbors=None,
                    neighbor_estimator="current",
                    gate_tensor=gate_eff,
                )
                if per_sample_mask_params is not None:
                    xp_corrupt, pseudo_mask_hard, y_soft_pseudo = apply_mask_noise_v5_per_sample(
                        x_prime, mask_ratio_per_sample, temperature=1.0
                    )
                else:
                    xp_corrupt, pseudo_mask_hard, y_soft_pseudo = apply_mask_noise_v5_ste(
                        x_prime, mask_ratio_param, temperature=1.0
                    )
                pseudo_mask_ste = pseudo_mask_hard + (y_soft_pseudo - y_soft_pseudo.detach())
                _, pseudo_loss, _ = model.loss_mask_weighted(
                    xp_corrupt, x, pseudo_mask_ste,
                    sample_weight=sample_weight,
                )
                loss = loss + args.pseudo_weight * pseudo_loss

            # Alignment loss for mask_ratio (auxiliary)
            if per_sample_mask_params is not None:
                align_loss = per_sample_mask_ratio_reg_loss(
                    y_soft_real, mask_ratio_per_sample, weight=1.0,
                )
                loss = loss + args.mask_alignment_weight * align_loss
            elif mask_ratio_raw is not None:
                # Expected mask ratio should match mask_ratio_param
                align_loss = (y_soft_real.mean() - mask_ratio_param).pow(2)
                loss = loss + args.mask_alignment_weight * align_loss

            # Edge regularization
            loss = loss + edge_module.regularization_loss()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ep_loss += float(loss.item()) * x.shape[0]
            ep_count += x.shape[0]

        avg_loss = ep_loss / max(ep_count, 1)
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            sched_t = t
            if per_sample_mask_params is not None:
                mask_base_raw, mask_scale_raw = per_sample_mask_params
                mask_base_val = float(mask_base_raw.detach().cpu())
                mask_scale_val = float(mask_scale_raw.detach().cpu())
                sal_min = float(salience_tensor.min().detach().cpu())
                sal_max = float(salience_tensor.max().detach().cpu())
                mask_min = mask_ratio_scalar_for_logging
                mask_max = mask_ratio_scalar_for_logging
            else:
                mask_base_val = mask_scale_val = 0.0
                sal_min = sal_max = 0.0
            gs = edge_module.gamma_snapshot()
            g_val = gs.get('gamma_sim', 0)
            print(f"  [{args.dataset_name}] ep {epoch:03d}/{args.epochs} loss={avg_loss:.4f} "
                  f"sched={sched_t:.2f} γ={g_val:.4f} mask={mask_ratio_scalar_for_logging:.4f}", flush=True)

    # Final eval
    model.eval()
    with torch.no_grad():
        X_full = torch.as_tensor(X, dtype=torch.float32, device=device)
        z_full = model.feature(X_full)
    ari, nmi, acc, _ = cluster_and_eval(z_full.cpu().numpy(), y, K, args.seed)
    print(f"[{args.dataset_name}] DONE  ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}", flush=True)

    metrics = {
        "ari": ari, "nmi": nmi, "acc": acc,
        "epochs": args.epochs,
        "model_architecture": "v5 (Gumbel-Sigmoid STE mask + single-γ edge reliability)",
        "variant": args.variant_name,
        "v5_gamma_mode": args.v5_gamma_mode,
        "mask_ratio_learnable": args.mask_ratio_learnable,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "gamma_snapshot": edge_module.gamma_snapshot(),
        "mask_ratio_final": mask_ratio_scalar_for_logging,
        "epochs": args.epochs,
        "v5_gamma_mode": args.v5_gamma_mode,
        "gate_lr_multiplier": args.gate_lr_multiplier,
        "per_sample_mask": args.per_sample_mask,
    }
    if per_sample_mask_params is not None:
        mask_base_raw, mask_scale_raw = per_sample_mask_params
        summary["mask_base_raw_final"] = float(mask_base_raw.detach().cpu())
        summary["mask_scale_raw_final"] = float(mask_scale_raw.detach().cpu())
        summary["salience_min"] = float(salience_tensor.min().detach().cpu())
        summary["salience_max"] = float(salience_tensor.max().detach().cpu())
        summary["salience_mean"] = float(salience_tensor.mean().detach().cpu())
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
