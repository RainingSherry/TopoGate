"""Label-free training loop for the unified TopoGate V0 model."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from .config import V0Config
from .corruption import apply_scmae_noise, compute_node_gate, make_pseudo_batch
from .diagnostics import evaluate_unsupervised_views
from .graph import (
    NeighborGraph,
    build_pca_knn_graph,
    compute_edge_reliability,
)
from .model import WeightedAutoEncoder


# 仓库级 GPU 防火墙：物理卡 0/7 禁止使用，其余训练卡显式列入白名单。
ALLOWED_PHYSICAL_GPUS = frozenset({1, 2, 3, 4, 5, 6})
FORBIDDEN_PHYSICAL_GPUS = frozenset({0, 7})


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a device while honoring the repository's GPU firewall."""

    requested = torch.device(device)
    if requested.type == "cpu":
        # CPU 不需要做 CUDA_VISIBLE_DEVICES 映射，也不触碰任何 GPU 状态。
        return requested
    if requested.type != "cuda":
        raise ValueError(f"unsupported device type: {requested.type}")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible:
        # 设置 CUDA_VISIBLE_DEVICES 后，torch.cuda 使用的是逻辑序号；这里
        # 先审计其对应的物理卡，再把 requested.index 限制在可见列表范围内。
        try:
            visible_physical = {int(item) for item in visible}
        except ValueError as exc:
            raise ValueError("CUDA_VISIBLE_DEVICES must contain integer GPU ids") from exc
        if visible_physical.intersection(FORBIDDEN_PHYSICAL_GPUS):
            raise ValueError("CUDA_VISIBLE_DEVICES includes forbidden physical GPU 0 or 7")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested.index is None:
            return torch.device("cuda:0")
        # A torch device is logical after CUDA_VISIBLE_DEVICES isolation.  The
        # explicit index is therefore checked against the visible list length.
        if int(requested.index) < 0 or int(requested.index) >= len(visible):
            raise ValueError(
                f"logical CUDA device {requested.index} is outside CUDA_VISIBLE_DEVICES={visible}"
            )
        return torch.device(f"cuda:{int(requested.index)}")
    if requested.index is None:
        raise ValueError("CUDA device must include a physical index")
    # 未设置可见列表时，cuda:N 直接按物理编号解释，因此仍必须经过白名单。
    physical = int(requested.index)
    if physical in FORBIDDEN_PHYSICAL_GPUS or physical not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(
            f"physical GPU {physical} is forbidden or unavailable; allowed GPUs are "
            f"{sorted(ALLOWED_PHYSICAL_GPUS)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(f"cuda:{physical}")


def seed_runtime(seed: int, device: torch.device) -> None:
    """Seed independent Python, NumPy, and torch streams."""

    # 同时固定 Python、NumPy、PyTorch（以及 CUDA）随机流，保证图抽样、
    # DataLoader 顺序、mask 噪声和网络初始化都能由同一个 seed 复现。
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    # 为 real/pseudo 噪声建立独立 generator，避免一个随机调用改变另一条路径。
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _consume_legacy_topology_auxiliary_draws(
    graph: NeighborGraph,
    mix_neighbors: int,
    rng: np.random.Generator,
) -> None:
    """Replay retired T's unused random/far-neighbour RNG draws.

    The legacy reliability path built these two arrays before training even
    though it never consumed them afterwards.  They shift the later pseudo
    neighbour samples, so strict trajectory parity has to reproduce the draws
    without importing or modifying the frozen retired implementation.
    """

    n_samples = int(graph.indices.shape[0])
    if n_samples <= 1:
        return
    width = max(1, min(int(mix_neighbors), n_samples - 1))
    all_indices = np.arange(n_samples, dtype=np.int64)
    for sample in range(n_samples):
        banned = {sample}
        banned.update(np.asarray(graph.indices[sample], dtype=np.int64).tolist())
        candidates = np.setdiff1d(
            all_indices,
            np.fromiter(banned, dtype=np.int64),
            assume_unique=False,
        )
        if candidates.size == 0:
            candidates = all_indices[all_indices != sample]
        rng.choice(candidates, size=width, replace=candidates.size < width)
    for sample in range(n_samples):
        candidates = rng.choice(
            all_indices[all_indices != sample],
            size=min(96, n_samples - 1),
            replace=False,
        )
        similarity = graph.embedding[candidates] @ graph.embedding[sample]
        far = candidates[np.argsort(similarity)[:width]]
        if far.size < width:
            rng.choice(candidates, size=width, replace=True)


def _empty_graph(n_samples: int) -> NeighborGraph:
    # Calling the public graph builder keeps the empty shape/profile contract in
    # one place and avoids a second, subtly different sentinel object.
    return build_pca_knn_graph(
        np.zeros((int(n_samples), 1), dtype=np.float32),
        k=0,
        pca_dim=1,
        tau=1.0,
        seed=0,
    )


@torch.no_grad()
def extract_embedding(
    model: WeightedAutoEncoder,
    data_np: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    # 训练结束后只对干净输入运行 encoder；这里不再生成 mask 或 pseudo view。
    model.eval()
    tensor = torch.as_tensor(data_np, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=max(1, int(batch_size) * 4),
        shuffle=False,
        drop_last=False,
    )
    rows = [model.feature(batch[0].to(device)).detach().cpu().numpy() for batch in loader]
    # 固定顺序拼回所有样本，并把异常浮点值转换为可供 KMeans 使用的有限值。
    embedding = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    return np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)


def _build_operator_state(
    data_np: np.ndarray,
    config: V0Config,
    seed: int,
) -> tuple[NeighborGraph, np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """Build graph, edge weights, node gates, and summaries for either F or T."""

    if not config.graph_enabled:
        # 关闭 pseudo 分支时仍返回形状完整的空图和诊断，保持输出契约一致。
        graph = _empty_graph(data_np.shape[0])
        empty = np.zeros((data_np.shape[0], 0), dtype=np.float32)
        gate, _sample_weight, gate_summary = compute_node_gate(
            graph,
            parameterization=config.parameterization,
            alpha=config.alpha,
            gate_min=config.gate_min,
            gate_max=config.gate_max,
            beta_mutual=config.beta_mutual,
            beta_snn=config.beta_snn,
            beta_perturb=config.beta_perturb,
            beta_uncertainty=config.beta_uncertainty,
        )
        edge_summary = {
            "edge_weight_entropy": 0.0,
            "effective_neighbor_count": 0.0,
            "max_edge_weight_mean": 0.0,
            "max_edge_weight_p95": 0.0,
            "fraction_effective_neighbors_lt_2": 1.0,
        }
        return graph, empty, empty, gate, edge_summary, gate_summary

    # 拓扑算子按固定顺序构建：PCA/kNN 图 -> edge reliability/weights -> node gate。
    # 整个过程只接收表达矩阵，不读取 benchmark 标签。
    graph = build_pca_knn_graph(
        data_np,
        k=min(int(config.neighbor_k), max(1, data_np.shape[0] - 1)),
        pca_dim=int(config.knn_pca_dim),
        tau=float(config.tau),
        seed=int(seed),
    )
    if config.parameterization == "topology":
        # T 用 topology reliability 重加权邻居分布；F 直接沿用基础 kNN 概率。
        reliability, edge_weights, edge_summary = compute_edge_reliability(
            graph,
            mode=config.edge_reliability_mode,
            gamma_sim=config.gamma_sim,
            gamma_mutual=config.gamma_mutual,
            gamma_snn=config.gamma_snn,
            gamma_distance=config.gamma_distance,
        )
    else:
        reliability = np.ones_like(graph.probs, dtype=np.float32)
        edge_weights = graph.probs.copy()
        # Use the same summary function via the reliability helper's ``none``
        # mode; the returned weights are exactly the base probabilities.
        _, _, edge_summary = compute_edge_reliability(
            graph,
            mode="none",
            gamma_sim=0.0,
            gamma_mutual=0.0,
            gamma_snn=0.0,
            gamma_distance=0.0,
        )
    # sample_weight 的数组由 gate 函数同步计算；trainer 只在 T 的 pseudo loss
    # 中使用它，F 的返回值保持单位权重。
    gate, _sample_weight, gate_summary = compute_node_gate(
        graph,
        parameterization=config.parameterization,
        alpha=config.alpha,
        gate_min=config.gate_min,
        gate_max=config.gate_max,
        beta_mutual=config.beta_mutual,
        beta_snn=config.beta_snn,
        beta_perturb=config.beta_perturb,
        beta_uncertainty=config.beta_uncertainty,
    )
    return graph, reliability, edge_weights, gate, edge_summary, gate_summary


def fit_predict(
    X: np.ndarray,
    *,
    n_clusters: int | None,
    config: V0Config,
    seed: int,
    device: str | torch.device,
    precomputed_graph_embedding: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Fit V0 without labels; optional ``n_clusters`` is readout-only.

    The function intentionally has no ``y`` argument.  Graph construction,
    gate calculation, corruption, optimization, and embedding extraction only
    receive ``X``.  A benchmark runner may evaluate the returned predictions
    against labels after this function returns.
    """

    # 这是模型的无标签入口：X 是训练、建图、扰动和 readout 的唯一数据输入。
    # n_clusters 只决定训练后的 KMeans 簇数，不参与参数学习。
    data = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] == 0:
        raise ValueError("X must contain at least two samples and one feature")
    if not np.all(np.isfinite(data)):
        raise ValueError("X contains non-finite values")
    if n_clusters is not None and not 1 <= int(n_clusters) <= data.shape[0]:
        raise ValueError("n_clusters must be in [1, n_samples]")

    runtime_device = resolve_device(device)
    seed_runtime(int(seed), runtime_device)
    legacy_plantnet = config.rng_protocol == "legacy_plantnet"
    if legacy_plantnet and runtime_device.type == "cuda":
        # ``family.set_seed`` in the retired runners called both forms.  Keep
        # their order for the parity protocol while leaving normal V0 unchanged.
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    graph, edge_reliability, edge_weights, node_gate, edge_summary, gate_summary = _build_operator_state(
        data, config, int(seed)
    )

    # The historical F/T both use these offsets.  T also consumed auxiliary
    # random/far-neighbour arrays before its first active pseudo draw.
    mix_rng = np.random.default_rng(
        int(seed) + (2027 if config.parameterization == "fixed" else 3089)
    )
    if legacy_plantnet and config.parameterization == "topology":
        _consume_legacy_topology_auxiliary_draws(graph, int(config.mix_neighbors), mix_rng)

    indices = torch.arange(data.shape[0], dtype=torch.long)
    tensor = torch.as_tensor(data, dtype=torch.float32)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(int(seed))
    # Retired PlantNet F passed ``drop_last=True`` directly to DataLoader, even
    # when the whole dataset was smaller than one batch.  Preserve that edge
    # case only in the explicit parity protocol; ordinary V0 remains friendly
    # to small CPU-smoke datasets.
    drop_last = bool(
        config.drop_last
        and (legacy_plantnet or data.shape[0] >= int(config.batch_size))
    )
    # DataLoader 的索引与表达值保持绑定，pseudo 构造需要用索引回查全量图。
    train_loader = DataLoader(
        TensorDataset(indices, tensor),
        batch_size=int(config.batch_size),
        shuffle=True,
        drop_last=drop_last,
        generator=loader_generator,
        num_workers=int(config.num_workers),
    )
    # F/T 使用同一个 WeightedAutoEncoder；optimizer 的参数集合因此只有共享
    # scMAE backbone，不存在隐藏的 F/T 可学习 gate 或 edge 参数。
    model = WeightedAutoEncoder(
        num_genes=int(data.shape[1]),
        hidden_size=int(config.hidden_size),
        dropout=float(config.dropout),
        masked_data_weight=float(config.masked_data_weight),
        mask_loss_weight=float(config.mask_loss_weight),
    ).to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    # Normal V0 isolates graph sampling and two noise streams.  The explicit
    # legacy protocol instead uses retired F/T's shared global torch stream.
    real_noise_generator = None if legacy_plantnet else _torch_generator(runtime_device, int(seed) + 400_003)
    pseudo_noise_generator = None if legacy_plantnet else _torch_generator(runtime_device, int(seed) + 500_003)
    pseudo_enabled = bool(config.graph_enabled and graph.indices.shape[1] > 0)

    history: dict[str, Any] = {
        "loss": [],
        "real_loss": [],
        "real_reconstruction_loss": [],
        "real_mask_loss": [],
        "pseudo_loss": [],
        "pseudo_reconstruction_loss": [],
        "pseudo_mask_loss": [],
        "mean_node_gate": [],
        "mean_pseudo_perturbation": [],
        "real_mask_rate": [],
        "pseudo_mask_rate": [],
        "parameterization": config.parameterization,
        "mix_mode": config.mix_mode,
        "pseudo_enabled": pseudo_enabled,
    }
    tracked = [key for key, value in history.items() if isinstance(value, list)]

    def _loss_for_protocol(
        corrupted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Use the retired scalar F reduction only in strict parity mode."""

        if legacy_plantnet and config.parameterization == "fixed":
            # Historical NeighborMix F called the inherited ``loss_mask``
            # directly.  It is algebraically equal to V0's unweighted method,
            # but its reduction order is not bit-identical on GPU.
            _, scalar_loss = model.loss_mask(corrupted, target, mask)
            zero = torch.zeros((), dtype=scalar_loss.dtype, device=scalar_loss.device)
            return scalar_loss, {
                "reconstruction_loss": scalar_loss.detach(),
                "mask_loss": zero,
                "mask_positive_rate": mask.mean().detach(),
            }
        _, scalar_loss, parts = model.loss_mask_weighted(
            corrupted,
            target,
            mask,
            sample_weight=sample_weight,
        )
        return scalar_loss, parts

    for _epoch in range(1, int(config.epochs) + 1):
        model.train()
        totals = {key: 0.0 for key in tracked}
        n_batches = 0
        for batch_indices_tensor, batch_cpu in train_loader:
            batch_indices = batch_indices_tensor.numpy().astype(np.int64, copy=False)
            batch = batch_cpu.to(runtime_device)
            # real view 以当前 batch 的 clean expression 为目标，执行历史 row-swap mask。
            corrupted, real_mask = apply_scmae_noise(
                batch,
                float(config.mask_ratio),
                generator=real_noise_generator,
                legacy_plantnet=legacy_plantnet,
            )
            real_loss, real_parts = _loss_for_protocol(corrupted, batch, real_mask)
            loss = real_loss
            pseudo_loss = torch.zeros((), dtype=batch.dtype, device=runtime_device)
            pseudo_parts = {
                "reconstruction_loss": pseudo_loss,
                "mask_loss": pseudo_loss,
                "mask_positive_rate": pseudo_loss,
            }
            mix_info = {"mean_node_gate": 0.0, "mean_perturb_norm": 0.0}
            if pseudo_enabled:
                # pseudo view 的输入由图上的邻居凸组合得到，目标仍是该 batch 的
                # clean anchor；T 另外用 topology gate 给每个样本加权。
                pseudo_batch, sample_weight, mix_info = make_pseudo_batch(
                    data_np=data,
                    batch_indices=batch_indices,
                    batch_x=batch,
                    parameterization=config.parameterization,
                    graph=graph,
                    edge_weights=edge_weights,
                    node_gate=node_gate,
                    mix_neighbors=int(config.mix_neighbors),
                    alpha=float(config.alpha),
                    rng=mix_rng,
                    neighbor_estimator=config.neighbor_estimator,
                    legacy_plantnet=legacy_plantnet,
                )
                pseudo_corrupted, pseudo_mask = apply_scmae_noise(
                    pseudo_batch,
                    float(config.mask_ratio),
                    generator=pseudo_noise_generator,
                    legacy_plantnet=legacy_plantnet,
                )
                # F deliberately has no per-sample gate weighting; T does.
                pseudo_weight = sample_weight if config.parameterization == "topology" else None
                pseudo_loss, pseudo_parts = _loss_for_protocol(
                    pseudo_corrupted,
                    batch,
                    pseudo_mask,
                    sample_weight=pseudo_weight,
                )
                loss = loss + float(config.pseudo_weight) * pseudo_loss

            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite V0 training loss")
            # 反向传播只更新 model.parameters()；图、edge weight、node gate 和
            # pseudo-cell 都是 detached 的无梯度统计量。
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["real_loss"] += float(real_loss.detach().cpu())
            totals["real_reconstruction_loss"] += float(real_parts["reconstruction_loss"].cpu())
            totals["real_mask_loss"] += float(real_parts["mask_loss"].cpu())
            totals["pseudo_loss"] += float(pseudo_loss.detach().cpu())
            totals["pseudo_reconstruction_loss"] += float(pseudo_parts["reconstruction_loss"].cpu())
            totals["pseudo_mask_loss"] += float(pseudo_parts["mask_loss"].cpu())
            totals["mean_node_gate"] += float(mix_info.get("mean_node_gate", 0.0))
            totals["mean_pseudo_perturbation"] += float(mix_info.get("mean_perturb_norm", 0.0))
            totals["real_mask_rate"] += float(real_mask.mean().detach().cpu())
            totals["pseudo_mask_rate"] += (
                float(pseudo_parts["mask_positive_rate"].cpu()) if pseudo_enabled else 0.0
            )
            n_batches += 1
        for key in tracked:
            history[key].append(totals[key] / max(1, n_batches))

    # 训练后的 clean embedding 是唯一的聚类 readout 输入；训练标签从未进入此函数。
    embedding = extract_embedding(model, data, config.batch_size, runtime_device)
    predictions = None
    if n_clusters is not None:
        # KMeans 只在训练完成后执行，n_clusters 是外部协议提供的 readout 参数。
        predictions = KMeans(
            n_clusters=int(n_clusters),
            n_init=int(config.kmeans_n_init),
            random_state=int(seed),
        ).fit_predict(embedding).astype(np.int64)

    unsupervised = (
        evaluate_unsupervised_views(
            model=model,
            data_np=data,
            clean_embedding=embedding,
            graph=graph,
            batch_size=int(config.batch_size),
            mask_ratio=float(config.mask_ratio),
            seed=int(seed),
            device=runtime_device,
        )
        if config.evaluate_unsupervised
        else {}
    )
    graph.profile["parameterization"] = config.parameterization
    graph.profile["mix_mode"] = config.mix_mode
    graph.profile["pseudo_enabled"] = pseudo_enabled
    # 保存图、gate、history 和模型参数用于审计；这些对象不含 ground-truth 标签。
    diagnostics: dict[str, Any] = {
        "neighbor_indices": graph.indices,
        "neighbor_base_probs": graph.probs,
        "neighbor_similarity": graph.similarity,
        "neighbor_distance": graph.distance,
        "edge_reliability": edge_reliability,
        "edge_weights": edge_weights,
        "node_gate": node_gate,
        "pseudo_perturbation": (
            (1.0 - np.sum(graph.probs * graph.similarity, axis=1)).astype(np.float32)
            if graph.probs.size
            else np.zeros(data.shape[0], dtype=np.float32)
        ),
        "graph_profile": graph.profile,
        "edge_summary": edge_summary,
        "gate_summary": gate_summary,
        "training_history": history,
        "unsupervised_diagnostics": unsupervised,
        "model_state": model.state_dict(),
        "core_summary": {
            "protocol_id": config.protocol_id,
            "parameterization": config.parameterization,
            "parameterization_alias": "F" if config.parameterization == "fixed" else "T",
            "seed": int(seed),
            "device": str(runtime_device),
            "n_samples": int(data.shape[0]),
            "n_features": int(data.shape[1]),
            "n_clusters": int(n_clusters) if n_clusters is not None else None,
            "graph_enabled": bool(config.graph_enabled),
            "pseudo_enabled": pseudo_enabled,
            "mix_mode": config.mix_mode,
            "gate_mode": config.gate_mode,
            "edge_reliability_mode": (
                config.edge_reliability_mode
                if config.parameterization == "topology"
                else "base_probability"
            ),
            "readout_enabled": n_clusters is not None,
            "K_used_only_in_readout": n_clusters is not None,
            "labels_used_during_fit": False,
            "labels_used_during_preprocessing": False,
            "labels_used_for_graph": False,
            "labels_used_for_gate": False,
            "labels_used_for_loss": False,
            "labels_used_for_selection": False,
            "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "rng_protocol": config.rng_protocol,
        },
    }
    return predictions, embedding, diagnostics


__all__ = [
    "ALLOWED_PHYSICAL_GPUS",
    "FORBIDDEN_PHYSICAL_GPUS",
    "extract_embedding",
    "fit_predict",
    "resolve_device",
    "seed_runtime",
]
