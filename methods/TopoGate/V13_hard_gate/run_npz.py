#!/usr/bin/env python
"""Run the V13 hard-gate TopoGate variant on an NPZ dataset.

V13 replaces the V12 softmax + rank_loss soft gate with a Gumbel-Top-k hard
selection. The autoencoder and graph construction are unchanged from V12.

Core differences from V12:
- The ``LearnableGate`` is replaced by ``GumbelTopKGate``.
- ``rank_alignment_loss`` is removed; top-k ordering is the selection mechanism.
- Neighbour target is normalized by the mask sum (== top_k at inference) rather
  than the constant K.
- ``gumbel_tau`` is annealed from gumbel_tau → gumbel_tau_min over the first
  ``gumbel_tau_anneal_epochs`` epochs, making the relaxation progressively
  sharper and the gate closer to its inference behaviour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = next(
    p for p in [CURRENT_DIR, *CURRENT_DIR.parents]
    if (p / "methods" / "TopoGate").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V13_hard_gate.gumbel_gate import (  # noqa: E402
    GumbelTopKGate,
    build_gate_stats_tensor,
    hard_topk_alignment_loss,
)
from methods.TopoGate.V13_hard_gate.model import AutoEncoder  # noqa: E402
from methods.TopoGate.learnable_gate.neighbor_graph import build_pca_knn_graph  # noqa: E402


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TopoGate V13 hard gate")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--method_name", default="TopoGate")
    parser.add_argument("--variant_name", default="topogate_v13_topk2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--scale_input", type=str2bool, default=True)
    parser.add_argument("--input_mode", choices=["raw", "log1p"], default="raw")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--decoder_mode",
        choices=["legacy_mask_conditioned", "latent_only"],
        default="legacy_mask_conditioned",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--masked_data_weight", type=float, default=0.75)
    parser.add_argument("--mask_loss_weight", type=float, default=0.1)
    parser.add_argument("--mask_loss_mode", choices=["additive", "legacy_weighted"], default="additive")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--topology_enabled", type=str2bool, default=True)
    parser.add_argument("--lambda_topology", type=float, default=0.1)
    parser.add_argument("--topology_warmup_epochs", type=int, default=20)
    parser.add_argument("--topology_ramp_epochs", type=int, default=10)
    parser.add_argument("--top_k_neighbors", type=int, default=2)
    parser.add_argument("--neighbor_k", type=int, default=5)
    parser.add_argument("--knn_pca_dim", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument(
        "--gumbel_tau",
        type=float,
        default=1.0,
        help="Initial Gumbel-Softmax temperature.",
    )
    parser.add_argument(
        "--gumbel_tau_min",
        type=float,
        default=0.1,
        help="Minimum Gumbel-Softmax temperature after annealing.",
    )
    parser.add_argument(
        "--gumbel_tau_anneal_epochs",
        type=int,
        default=50,
        help="Epochs over which to anneal tau from gumbel_tau to gumbel_tau_min.",
    )
    parser.add_argument("--log_interval", type=int, default=10)
    return parser.parse_args()


class _Dataset(Dataset):
    def __init__(self, x: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        return int(index), self.x[index]


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as data:
        x = data.get("x", data.get("X", data.get("data")))
        y = data.get("y", data.get("labels", data.get("label")))
        if x is None:
            raise ValueError(f"{path} must contain x/X/data")
        x_np = np.asarray(x, dtype=np.float32)
        y_np = None if y is None else np.asarray(y).reshape(-1)
    if x_np.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got {x_np.shape}")
    if y_np is not None and y_np.shape[0] != x_np.shape[0]:
        raise ValueError("labels and features have different numbers of rows")
    return x_np, y_np


def _device(gpu: int, no_cuda: bool) -> torch.device:
    if no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    if int(gpu) in {0, 7}:
        raise ValueError("physical GPU 0 and GPU 7 are forbidden")
    return torch.device(f"cuda:{int(gpu)}")


def _apply_mask_noise(x: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    mask = (torch.rand_like(x) < float(ratio)).to(dtype=x.dtype)
    replacement = x if x.shape[0] <= 1 else x[torch.randperm(x.shape[0], device=x.device)]
    return torch.where(mask.bool(), replacement, x), mask


@torch.no_grad()
def _encode_all(
    model: AutoEncoder, x_cpu: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    chunks = []
    for start in range(0, x_cpu.shape[0], max(1, int(batch_size))):
        chunks.append(
            model.feature(x_cpu[start : start + batch_size].to(device)).cpu()
        )
    model.train(was_training)
    return torch.cat(chunks, dim=0)


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, prediction)
    rows, cols = linear_sum_assignment(-cm)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    aligned = np.asarray([mapping.get(int(v), int(v)) for v in prediction])
    return {
        "acc": float(accuracy_score(y_true, aligned)),
        "nmi": float(normalized_mutual_info_score(y_true, prediction)),
        "ari": float(adjusted_rand_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, aligned, average="macro", zero_division=0)),
        "fmi": float(fowlkes_mallows_score(y_true, prediction)),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialise {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    source = Path(args.data_path).resolve()
    x_raw, y_raw = _load_npz(source)
    if args.input_mode == "log1p":
        if np.nanmin(x_raw) < 0:
            raise ValueError("log1p input_mode requires non-negative features")
        x_raw = np.log1p(x_raw)
    x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_np = (
        StandardScaler(with_mean=True, with_std=True)
        .fit_transform(x_raw)
        .astype(np.float32)
        if args.scale_input
        else x_raw.astype(np.float32, copy=False)
    )

    if y_raw is None:
        if args.n_clusters is None:
            raise ValueError("--n_clusters is required when the NPZ has no labels")
        y_encoded = None
        n_clusters = int(args.n_clusters)
        label_mapping = None
        k_source = "explicit_n_clusters"
    else:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_raw).astype(np.int64)
        n_clusters = int(args.n_clusters or np.unique(y_encoded).size)
        label_mapping = {
            str(i): str(value) for i, value in enumerate(encoder.classes_)
        }
        k_source = (
            "benchmark_oracle_from_y"
            if args.n_clusters is None
            else "explicit_n_clusters"
        )

    device = _device(args.gpu, args.no_cuda)
    x_cpu = torch.as_tensor(x_np, dtype=torch.float32)
    train_loader = DataLoader(
        _Dataset(x_np),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )

    edge_indices = None
    edge_stats = None
    gate = None
    if bool(args.topology_enabled):
        graph = build_pca_knn_graph(
            x_np,
            k=int(args.neighbor_k),
            pca_dim=min(int(args.knn_pca_dim), x_np.shape[1]),
            tau=float(args.tau),
            seed=int(args.seed),
        )
        edge_indices = torch.as_tensor(graph.indices, dtype=torch.long)
        edge_stats = build_gate_stats_tensor(
            torch.as_tensor(graph.similarity),
            torch.as_tensor(graph.mutual.astype(np.float32)),
            torch.as_tensor(graph.snn),
            torch.as_tensor(graph.distance),
        ).to(device)
        gate = GumbelTopKGate(
            feature_dim=edge_stats.shape[-1],
            hidden_dim=32,
            top_k=int(args.top_k_neighbors),
            dropout=float(args.dropout),
        ).to(device)

    model = AutoEncoder(
        num_genes=x_np.shape[1],
        hidden_size=int(args.hidden_size),
        dropout=float(args.dropout),
        masked_data_weight=float(args.masked_data_weight),
        mask_loss_weight=float(args.mask_loss_weight),
        mask_loss_mode=str(args.mask_loss_mode),
        decoder_mode=str(args.decoder_mode),
    ).to(device)

    parameters = list(model.parameters()) + ([] if gate is None else list(gate.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: list[dict[str, float | int | bool]] = []
    start_time = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        if gate is not None:
            gate.train()
        z_all_cpu = (
            _encode_all(model, x_cpu, max(512, int(args.batch_size) * 2), device)
            if gate is not None
            else None
        )

        # Gumbel temperature schedule: linear annealing from gumbel_tau to
        # gumbel_tau_min over the first gumbel_tau_anneal_epochs.
        gumbel_tau = max(
            float(args.gumbel_tau_min),
            float(args.gumbel_tau)
            - epoch
            * (float(args.gumbel_tau) - float(args.gumbel_tau_min))
            / max(1, int(args.gumbel_tau_anneal_epochs)),
        )

        epoch_values = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "raw_reconstruction_loss": 0.0,
            "mask_loss": 0.0,
            "raw_mask_loss": 0.0,
            "topology_loss": 0.0,
            "gumbel_tau": float(gumbel_tau),
            "selected_neighbor_count": 0.0,
            "mean_mask_sum": 0.0,
            "gate_grad_norm": 0.0,
        }
        batches = 0
        ramp = 0.0
        if gate is not None:
            warmup = max(0, int(args.topology_warmup_epochs))
            ramp_epochs = max(1, int(args.topology_ramp_epochs))
            ramp = max(0.0, min(1.0, (epoch - warmup) / ramp_epochs))

        for batch_indices, x in train_loader:
            batch_indices = batch_indices.to(dtype=torch.long)
            x = x.to(device)
            x_corrupt, mask = _apply_mask_noise(x, float(args.mask_ratio))
            latent, reconstruction_loss, parts = model.loss_mask_weighted(x_corrupt, x, mask)
            total = reconstruction_loss
            topo_loss = torch.zeros((), dtype=total.dtype, device=device)
            batch_selected_neighbor_count = torch.zeros((), dtype=total.dtype, device=device)
            batch_mean_mask_sum = torch.zeros((), dtype=total.dtype, device=device)

            if gate is not None and edge_indices is not None and edge_stats is not None and z_all_cpu is not None:
                idx_device = batch_indices.to(device)
                gate_inputs = edge_stats[idx_device]

                if ramp > 0.0:
                    # Gate is active: compute the Gumbel-Top-k mask.
                    # Training: soft Gumbel-Softmax with straight-through gradient.
                    # Inference: hard top-k truncation (set by model.train()/eval()).
                    gate_out = gate(gate_inputs, tau=gumbel_tau, hard=False)
                    mask_tensor = gate_out.mask  # (B, K), retains gradients

                    neighbors = edge_indices[batch_indices]
                    z_neighbors = z_all_cpu[neighbors].to(device)

                    topo_loss, _ = hard_topk_alignment_loss(
                        latent,
                        z_neighbors,
                        mask_tensor,
                        detach_neighbors=True,
                    )
                    total = total + float(args.lambda_topology) * ramp * topo_loss

                    # Diagnostics: mask is binary at inference (hard=True), soft in training.
                    mask_sum = mask_tensor.sum(dim=1)  # (B,)
                    batch_mean_mask_sum = mask_sum.mean()
                    # selected_neighbor_count: effective neighbours = sum(mask^1) = mask_sum
                    # at inference; during training it is a soft expectation ≈ top_k.
                    batch_selected_neighbor_count = mask_sum.mean()

            optimizer.zero_grad(set_to_none=True)
            total.backward()

            gate_grad_norm = torch.zeros((), dtype=total.dtype, device=device)
            if gate is not None:
                squared = [
                    p.grad.detach().pow(2).sum()
                    for p in gate.parameters()
                    if p.grad is not None
                ]
                if squared:
                    gate_grad_norm = torch.sqrt(torch.stack(squared).sum())

            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()

            epoch_values["loss"] += float(total.detach().cpu())
            epoch_values["reconstruction_loss"] += float(parts["reconstruction_loss"].cpu())
            epoch_values["raw_reconstruction_loss"] += float(parts["raw_reconstruction_loss"].cpu())
            epoch_values["mask_loss"] += float(parts["mask_loss"].cpu())
            epoch_values["raw_mask_loss"] += float(parts["raw_mask_loss"].cpu())
            epoch_values["topology_loss"] += float(topo_loss.detach().cpu())
            epoch_values["gumbel_tau"] = float(gumbel_tau)
            epoch_values["selected_neighbor_count"] += float(batch_selected_neighbor_count.detach().cpu())
            epoch_values["mean_mask_sum"] += float(batch_mean_mask_sum.detach().cpu())
            epoch_values["gate_grad_norm"] += float(gate_grad_norm.detach().cpu())
            batches += 1

        for key in epoch_values:
            epoch_values[key] /= max(1, batches)
        history.append({"epoch": epoch, "topology_ramp": ramp, **epoch_values})

        if epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.log_interval)) == 0:
            print(
                f"[{args.dataset_name or source.stem}] epoch={epoch:03d}/{args.epochs} "
                f"loss={epoch_values['loss']:.5f} rec={epoch_values['raw_reconstruction_loss']:.5f} "
                f"topo={epoch_values['topology_loss']:.5f} tau={epoch_values['gumbel_tau']:.3f} "
                f"sel_neigh={epoch_values['selected_neighbor_count']:.3f}",
                flush=True,
            )

    embedding = (
        _encode_all(model, x_cpu, max(512, int(args.batch_size) * 2), device)
        .numpy()
        .astype(np.float32)
    )
    predictions = (
        KMeans(n_clusters=n_clusters, n_init=20, random_state=int(args.seed))
        .fit_predict(embedding)
        .astype(np.int64)
    )
    metrics = {} if y_encoded is None else _metrics(y_encoded, predictions)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "embedding_final.npy", embedding)
    np.save(save_dir / "predictions.npy", predictions)
    if y_encoded is not None:
        np.save(save_dir / "labels_true.npy", y_encoded)
    if label_mapping is not None:
        with (save_dir / "label_mapping.json").open("w") as handle:
            json.dump(label_mapping, handle, indent=2, ensure_ascii=True)

    # Final gate evaluation: hard inference to get the binary mask.
    effective_neighbors = 0.0
    final_mask_sum = 0.0
    if gate is not None and edge_stats is not None and edge_indices is not None:
        gate.eval()
        with torch.no_grad():
            final_out = gate(edge_stats, tau=float(args.gumbel_tau_min), hard=True)
            final_mask = final_out.mask.cpu().numpy()  # (N, K) binary
            final_mask_sum = float(final_mask.sum(axis=1).mean())
            effective_neighbors = float(final_mask.sum(axis=1).mean())
        np.savez_compressed(
            save_dir / "final_graph_edges.npz",
            indices=edge_indices.numpy(),
            edge_features=edge_stats.cpu().numpy(),
            mask=final_mask,
        )

    mean_gate_grad_norm = float(
        np.mean([float(row["gate_grad_norm"]) for row in history])
    ) if history else 0.0
    mean_history = {
        key: float(np.mean([float(row[key]) for row in history]))
        for key in (
            "topology_loss",
            "reconstruction_loss",
            "raw_reconstruction_loss",
            "mask_loss",
            "raw_mask_loss",
            "gumbel_tau",
            "selected_neighbor_count",
            "mean_mask_sum",
        )
        if history
    }

    summary: dict[str, Any] = {
        "method_name": str(args.method_name),
        "variant_name": str(args.variant_name),
        "dataset": str(args.dataset_name or source.stem),
        "seed": int(args.seed),
        "input_shape": [int(v) for v in x_np.shape],
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "runner_source_path": str(Path(__file__).resolve()),
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "model_source_path": str(Path(__file__).with_name("model.py").resolve()),
        "model_source_sha256": _sha256(Path(__file__).with_name("model.py").resolve()),
        "gate_source_path": str(Path(__file__).with_name("gumbel_gate.py").resolve()),
        "gate_source_sha256": _sha256(Path(__file__).with_name("gumbel_gate.py").resolve()),
        "n_clusters": int(n_clusters),
        "k_source": k_source,
        "labels_used_during_fit": False,
        "topology_enabled": bool(args.topology_enabled),
        "topology_mode": "topk",
        "lambda_topology": float(args.lambda_topology),
        "topology_warmup_epochs": int(args.topology_warmup_epochs),
        "topology_ramp_epochs": int(args.topology_ramp_epochs),
        "decoder_mode": str(args.decoder_mode),
        "mask_loss_mode": str(args.mask_loss_mode),
        "mask_loss_weight": float(args.mask_loss_weight),
        "top_k_neighbors": int(args.top_k_neighbors),
        "gumbel_tau": float(args.gumbel_tau),
        "gumbel_tau_min": float(args.gumbel_tau_min),
        "gumbel_tau_anneal_epochs": int(args.gumbel_tau_anneal_epochs),
        "selected_neighbor_count": effective_neighbors,
        "mean_mask_sum": final_mask_sum,
        "effective_neighbor_count": effective_neighbors,
        "mean_gate_grad_norm": mean_gate_grad_norm,
        "topology_loss": mean_history.get("topology_loss", 0.0),
        "reconstruction_loss": mean_history.get("raw_reconstruction_loss", 0.0),
        "mask_loss": mean_history.get("raw_mask_loss", 0.0),
        "mean_history": mean_history,
        "train_seconds": float(time.time() - start_time),
        "metrics": metrics,
        "output_contract": {
            "predictions": "predictions.npy",
            "labels_true": "labels_true.npy" if y_encoded is not None else None,
            "embedding": "embedding_final.npy",
        },
    }

    with (save_dir / "history.json").open("w") as handle:
        json.dump(history, handle, indent=2, default=_json_default)
    with (save_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    with (save_dir / "resolved_args.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=_json_default)
    return summary


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
