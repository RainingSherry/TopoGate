#!/usr/bin/env python
"""
NeighborMix-scMAE：用于单细胞 RNA-seq 降维与聚类的方法。

整体流程（main）：
    1. 加载 AnnData，提取基因表达矩阵与细胞标签
    2. 构建 KNN 图（PCA + 余弦距离），作为 pseudo-cell 邻域增强的邻居来源
    3. 训练 scMAE 掩码自编码器，包含两个分支：
         - real_loss：对原始真实细胞进行掩码重建
         - pseudo_loss：对邻域仿细胞增强视图进行掩码重建，但目标仍为原始真实细胞
    4. 提取编码器潜在向量，并使用 KMeans 评估聚类表现
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    f1_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, normalize
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = next(parent for parent in [CURRENT_DIR, *CURRENT_DIR.parents] if (parent / "methods" / "DeepLearning" / "scMAE_family.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .model import AutoEncoder
except ImportError:
    from model import AutoEncoder

from methods.shared_utils import ensure_dir, save_json, sanitize_anndata_for_write
from methods.DeepLearning import scMAE_family as family


# 按优先级排序的候选标签列名，用于自动检测细胞标签列
LABEL_CANDIDATES = [
    "maintype",
    "cell_type",
    "Celltype",
    "celltype",
    "label",
    "labels",
    "cell_label",
    "Cluster",
    "cluster",
    "clusters",
    "Seurat_clusters",
]


@dataclass
class DataBundle:
    """
    打包单个数据集在训练与评估过程中需要用到的全部信息。

    属性：
        adata: 处理后的 AnnData 对象（用于写回 h5ad）
        data: float32 的 numpy 数组，形状为 [n_cells, n_genes]
        labels: 整数编码后的细胞类型标签
        label_names: 原始细胞类型名称
        label_key: obs 中的标签列名
        gene_names: 基因名列表
        profile: 数据集元信息（用于日志记录 / 可复现性）
        preprocess_config: 预处理参数配置
    """
    adata: sc.AnnData
    data: np.ndarray
    labels: np.ndarray
    label_names: np.ndarray
    label_key: str
    gene_names: np.ndarray
    profile: dict
    preprocess_config: dict


class IndexedExpressionDataset(Dataset):
    """
    PyTorch 数据集：返回 `(index, expression, label)` 三元组。

    其中 index 用于在训练时查找预先计算好的 KNN 邻居。
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int):
        return int(idx), self.data[idx], self.labels[idx]


def str2bool(value):
    """为 CLI 的 --flag / --no-flag 参数解析布尔值。接受 true/false/1/0/t/f/yes/no/y/n。"""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_float_list(value: str):
    """将逗号分隔的浮点数字符串解析为列表，用于 --leiden_resolutions 等参数。"""
    if value is None or str(value).strip() == "":
        return []
    return [float(item) for item in str(value).split(",") if item.strip()]


def parse_args():
    """
    解析命令行参数并返回配置对象。

    该函数集中定义 NeighborMix-scMAE 运行所需的全部超参数、
    数据路径、训练设置、评估开关以及输出控制选项。
    返回值为 argparse.Namespace，可直接在主流程中使用。
    """
    parser = argparse.ArgumentParser(description="NeighborMix-scMAE fixed-protocol runner")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--method_name", default="NeighborMix_scMAE")
    parser.add_argument("--variant_name", default="nm_scmae_mid")
    parser.add_argument("--label_key", default="auto")
    parser.add_argument("--input_mode", default="auto", choices=["auto", "raw", "log1p"])
    parser.add_argument("--n_top_genes", type=int, default=1000)
    parser.add_argument("--target_sum", type=float, default=10000.0)
    parser.add_argument("--scale_input", type=str2bool, default=True)
    parser.add_argument("--n_clusters", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--masked_data_weight", type=float, default=0.75)
    parser.add_argument("--mask_loss_weight", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--use_pseudo", type=str2bool, default=True)
    parser.add_argument("--pseudo_weight", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--neighbor_k", type=int, default=5)
    parser.add_argument("--mix_neighbors", type=int, default=4)
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Deprecated in pseudo-anchor-recovery variant; retained for CLI compatibility but unused.",
    )
    parser.add_argument(
        "--mix_weight",
        type=float,
        default=0.0,
        help="Deprecated in pseudo-anchor-recovery variant; retained for CLI compatibility but unused.",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.0,
        help="Deprecated in pseudo-anchor-recovery variant; retained for CLI compatibility but unused.",
    )
    parser.add_argument(
        "--target_mode",
        default="original",
        choices=["original", "mixed"],
        help="Deprecated in pseudo-anchor-recovery variant; retained for CLI compatibility but unused.",
    )
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--knn_pca_dim", type=int, default=50)
    parser.add_argument("--eval_neighbors", type=int, default=15)
    parser.add_argument("--leiden_fixed_resolution", type=float, default=1.0)
    parser.add_argument("--louvain_fixed_resolution", type=float, default=1.0)
    parser.add_argument("--leiden_resolutions", default="0.2,0.4,0.6,0.8,1.0,1.2")
    parser.add_argument("--include_louvain", type=str2bool, default=False)
    parser.add_argument("--run_oracle_sweep", type=str2bool, default=False)
    parser.add_argument("--sweep_max_cells", type=int, default=10000)
    parser.add_argument("--silhouette_sample_size", type=int, default=3000)
    parser.add_argument("--skip_eval", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_save_h5ad", action="store_true")
    parser.add_argument("--lightweight_outputs", action="store_true")
    return parser.parse_args()


# Legacy local helpers below are retained for compatibility/debugging only.
# The active main() path uses methods.DeepLearning.scMAE_family for seeding,
# device selection, dataset loading, embedding extraction, and evaluation.
def set_seed(seed: int) -> None:
    """
    为所有随机数生成器设置种子，以尽量保证结果可复现。

    包括 Python、NumPy、PyTorch 以及 CUDA 相关随机状态。
    若 CUDA 可用，还会启用 cudnn.deterministic 并关闭
    cudnn.benchmark，以减少非确定性行为。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu: int, no_cuda: bool) -> torch.device:
    """
    根据配置与当前环境选择计算设备。

    当 no_cuda 为 True 或当前不可用 CUDA 时，返回 CPU。
    否则会结合 CUDA_VISIBLE_DEVICES 与传入的 gpu 编号，
    解析出实际应使用的 torch.device，同时避开被禁止的物理 GPU。
    """
    if no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
        if set(visible_ids).intersection({"0", "7"}):
            raise ValueError("CUDA_VISIBLE_DEVICES includes forbidden physical GPU 0 or 7.")
        if len(visible_ids) == 1:
            return torch.device("cuda:0")
        if str(gpu) in visible_ids:
            return torch.device(f"cuda:{visible_ids.index(str(gpu))}")
        if 0 <= gpu < len(visible_ids):
            return torch.device(f"cuda:{gpu}")
        raise ValueError(f"--gpu {gpu} is outside isolated CUDA_VISIBLE_DEVICES={visible!r}.")
    if gpu in {0, 7}:
        raise ValueError("Physical GPU 0 and GPU 7 are forbidden. Use 1,2,3,4,5,6 or --no_cuda.")
    return torch.device(f"cuda:{gpu}")


def _ensure_csr(matrix) -> sp.csr_matrix:
    """
    将输入矩阵转换为 float32 的 CSR 稀疏矩阵。

    该函数同时处理稠密矩阵与稀疏矩阵，并将 NaN、正负无穷替换为 0，
    同时把负值截断为 0，以得到适合后续单细胞表达处理流程的规范化矩阵。
    """
    if sp.issparse(matrix):
        out = matrix.tocsr().astype(np.float32)
    else:
        out = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    out.data = np.nan_to_num(out.data, nan=0.0, posinf=0.0, neginf=0.0)
    out.data[out.data < 0.0] = 0.0
    out.eliminate_zeros()
    out.sort_indices()
    return out


def _sample_values(matrix, max_rows: int = 256) -> np.ndarray:
    """
    从矩阵前若干行采样数值，并展平成 float32 数组。

    该函数用于快速观察数据的大致分布，而不必物化整个矩阵。
    对于稀疏矩阵，仅返回采样部分中的非零元素。
    """
    sample = matrix[: min(max_rows, matrix.shape[0])]
    if sp.issparse(sample):
        return sample.data.astype(np.float32, copy=False) if sample.nnz else np.array([], dtype=np.float32)
    return np.asarray(sample, dtype=np.float32).ravel()


def _looks_like_raw_counts(matrix) -> bool:
    """
    启发式判断矩阵是否像原始计数数据。

    若采样值均为非负且近似整数，则视为原始 counts；
    否则更可能是已归一化或已对数变换后的表达矩阵。
    """
    values = _sample_values(matrix)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    values = values[: min(values.size, 100000)]
    return bool(np.all(values >= 0.0) and np.allclose(values, np.round(values), atol=1e-4))


def _var_gene_names(var) -> tuple[np.ndarray, object]:
    """
    从 var 表中提取基因名，并尽量使用更标准的基因名列。

    会依次尝试多个常见列名；若找到长度匹配且元素唯一的候选列，
    则用其作为基因名，并同步更新返回副本的索引。
    """
    var = var.copy()
    gene_names = np.asarray(var.index).astype(str)
    for name_col in ["gene_name", "features", "gene_symbols", "symbol", "_index"]:
        if name_col in var.columns:
            candidate = var[name_col].astype(str).to_numpy()
            if len(candidate) == len(gene_names) and len(np.unique(candidate)) == len(candidate):
                gene_names = candidate
                break
    var.index = gene_names
    return gene_names, var


def _select_count_source(adata: sc.AnnData, input_mode: str):
    """
    为当前 AnnData 选择合适的表达矩阵来源。

    优先使用原始计数来源，如 layers['counts'] 或 adata.raw.X；
    若不可用，则根据 adata.X 的数值特征判断其更像 raw 还是 log1p 数据。
    返回表达矩阵、基因名、var、来源描述以及推断出的输入模式。
    """
    if input_mode in {"auto", "raw"} and "counts" in adata.layers:
        gene_names, var = _var_gene_names(adata.var)
        return adata.layers["counts"], gene_names, var, "layers[counts]", "raw"
    if input_mode in {"auto", "raw"} and adata.raw is not None:
        gene_names, var = _var_gene_names(adata.raw.var)
        return adata.raw.X, gene_names, var, "adata.raw.X", "raw"
    inferred = "raw" if _looks_like_raw_counts(adata.X) else "log1p"
    if input_mode == "raw" and inferred != "raw":
        raise ValueError("--input_mode raw was requested, but no raw-looking X/raw/layers[counts] source is available.")
    if input_mode == "log1p":
        inferred = "log1p"
    gene_names, var = _var_gene_names(adata.var)
    source = "adata.X" if inferred == "raw" else "adata.X_log1p_fallback"
    return adata.X, gene_names, var, source, inferred


def _resolve_labels(adata: sc.AnnData, label_key: str):
    """
    解析并编码细胞标签列。

    若用户显式指定 label_key，则直接使用该列；否则按候选列名顺序自动检测。
    返回整数编码后的标签、原始类别名以及最终使用的标签列名。
    """
    if label_key and label_key != "auto":
        if label_key not in adata.obs.columns:
            raise KeyError(f"Configured label_key={label_key!r} is absent. Available obs columns: {list(adata.obs.columns)}")
        raw = adata.obs[label_key].astype(str).to_numpy()
        encoder = LabelEncoder()
        return encoder.fit_transform(raw).astype(np.int64), encoder.classes_, label_key
    for candidate in LABEL_CANDIDATES:
        if candidate in adata.obs.columns:
            raw = adata.obs[candidate].astype(str).to_numpy()
            encoder = LabelEncoder()
            return encoder.fit_transform(raw).astype(np.int64), encoder.classes_, candidate
    raise KeyError(f"No label column found. Available obs columns: {list(adata.obs.columns)}")


def _dense_float32(matrix) -> np.ndarray:
    """
    将矩阵转换为 float32 的稠密 NumPy 数组。

    无论输入是稀疏还是稠密格式，都会统一转换，
    并将 NaN、正无穷和负无穷替换为 0，以便后续送入模型。
    """
    if sp.issparse(matrix):
        arr = matrix.toarray()
    else:
        arr = np.asarray(matrix)
    arr = np.asarray(arr, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _manual_dense_scale(matrix) -> np.ndarray:
    """
    将表达矩阵转为 float32 稠密数组，并进行稳定的按基因 Z-score 标准化。

    该实现避免调用 scanpy/sklearn 对稀疏矩阵的内部 zero-centering 路径，
    从而降低大数据集上触发底层 BLAS/稀疏实现段错误的风险。
    """
    dense = _dense_float32(matrix)
    mean = dense.mean(axis=0, dtype=np.float64)
    std = dense.std(axis=0, dtype=np.float64)
    std[std < 1e-12] = 1.0
    dense = ((dense - mean) / std).astype(np.float32, copy=False)
    return np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)


def load_dataset(file_path: str, input_mode: str, n_top_genes: int, target_sum: float,
                 scale_input: bool, label_key: str, seed: int) -> DataBundle:
    """
    读取并预处理输入数据集，返回可直接供模型训练的数据包。

    主要步骤包括：
        1. 读取 h5ad 文件
        2. 选择表达矩阵来源（raw / log1p / auto）
        3. 过滤全零基因并保留高变基因
        4. 根据需要执行总量归一化、log1p 和标准化
        5. 自动解析标签列并编码
    """
    adata = sc.read_h5ad(file_path)
    matrix, gene_names, var, source_desc, inferred_mode = _select_count_source(adata, input_mode)
    X = _ensure_csr(matrix)
    keep_genes = np.asarray(X.sum(axis=0)).ravel() > 0
    X = X[:, keep_genes]
    gene_names = gene_names[keep_genes]
    var = var.iloc[keep_genes].copy()

    tmp = sc.AnnData(X=X, obs=adata.obs.copy(), var=var.copy())
    if 0 < n_top_genes < tmp.n_vars:
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat_v3", subset=True)
        gene_names = np.asarray(tmp.var.index).astype(str)

    preprocess_config = {
        "input_mode_requested": input_mode,
        "input_mode_used": inferred_mode,
        "source": source_desc,
        "n_top_genes": int(n_top_genes),
        "target_sum": float(target_sum),
        "scale_input": bool(scale_input),
        "seed": int(seed),
    }

    if inferred_mode == "raw":
        sc.pp.normalize_total(tmp, target_sum=float(target_sum))
        sc.pp.log1p(tmp)
    if scale_input:
        tmp.X = _manual_dense_scale(tmp.X)
    else:
        tmp.X = _dense_float32(tmp.X)

    labels, label_names, resolved_label_key = _resolve_labels(tmp, label_key)
    profile = {
        "file_path": str(Path(file_path).resolve()),
        "n_cells": int(tmp.n_obs),
        "n_genes": int(tmp.n_vars),
        "label_key": resolved_label_key,
        "label_classes": [str(item) for item in label_names.tolist()],
        "input_source": source_desc,
        "input_mode_used": inferred_mode,
    }

    tmp.var_names = pd.Index(np.asarray(gene_names).astype(str))
    return DataBundle(
        adata=tmp,
        data=np.asarray(tmp.X, dtype=np.float32),
        labels=labels,
        label_names=label_names,
        label_key=resolved_label_key,
        gene_names=np.asarray(tmp.var_names).astype(str),
        profile=profile,
        preprocess_config=preprocess_config,
    )


def build_knn_distribution(data_np: np.ndarray, k: int, pca_dim: int, tau: float, seed: int):
    """
    在 PCA + L2 归一化后的空间中构建 KNN 分布。

    返回：
        neighbor_indices: 每个细胞对应的邻居索引，形状 [n_cells, k]
        neighbor_probs: 归一化后的邻居采样概率，形状 [n_cells, k]
        profile: 便于记录的统计信息
    """
    n_cells = int(data_np.shape[0])
    if k <= 0 or n_cells <= 1:
        return None, None, {
            "neighbor_k": 0,
            "tau": float(tau),
            "mean_max_neighbor_prob": 0.0,
            "mean_neighbor_similarity": 0.0,
        }

    max_neighbors = min(int(k), max(1, n_cells - 1))
    pca_dim = max(1, min(int(pca_dim), data_np.shape[1], n_cells - 1 if n_cells > 1 else 1))

    if sp.issparse(data_np):
        data_dense = data_np.toarray().astype(np.float32)
    else:
        data_dense = np.asarray(data_np, dtype=np.float32)

    if min(data_dense.shape) > 1 and pca_dim < min(data_dense.shape):
        embedding = PCA(n_components=pca_dim, random_state=seed).fit_transform(data_dense)
    else:
        embedding = data_dense
    embedding = normalize(embedding, axis=1)

    nn = NearestNeighbors(n_neighbors=max_neighbors + 1, metric="cosine")
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)

    neighbor_indices = indices[:, 1: max_neighbors + 1].astype(np.int64, copy=False)
    cosine_similarity = (1.0 - distances[:, 1: max_neighbors + 1]).astype(np.float32, copy=False)
    scaled = cosine_similarity / max(float(tau), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled, dtype=np.float32)
    probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-12, None)

    profile = {
        "neighbor_k": int(max_neighbors),
        "tau": float(tau),
        "mean_max_neighbor_prob": float(probs.max(axis=1).mean()),
        "mean_neighbor_similarity": float(cosine_similarity.mean()),
    }
    return neighbor_indices, probs.astype(np.float32), profile


def sample_mix(
    data_np: np.ndarray,
    batch_indices: np.ndarray,
    batch_x: torch.Tensor,
    alpha: float,
    mix_neighbors: int,
    rng: np.random.Generator,
    neighbor_indices: np.ndarray | None,
    neighbor_probs: np.ndarray | None,
) -> torch.Tensor:
    """
    为当前 batch 生成 pseudo-cell 邻域增强视图。

    对 batch 中的每个细胞，函数都会从预先计算好的 KNN 图中采样
    `mix_neighbors` 个邻居，按相似度概率加权后求邻居表达的加权平均。
    最终视图采用凸组合：
        x_prime = alpha * x + (1 - alpha) * neighbor_mean
    """
    if int(mix_neighbors) <= 0 or neighbor_indices is None or neighbor_probs is None:
        return batch_x

    bsz = int(batch_indices.shape[0])
    mix_neighbors = max(1, int(mix_neighbors))
    sampled = np.empty((bsz, mix_neighbors), dtype=np.int64)
    weights = np.empty((bsz, mix_neighbors), dtype=np.float32)
    for pos, cell in enumerate(batch_indices):
        probs = neighbor_probs[cell]
        choices = rng.choice(neighbor_indices.shape[1], size=mix_neighbors, replace=True, p=probs)
        sampled[pos] = neighbor_indices[cell, choices]
        picked = probs[choices].astype(np.float32, copy=False)
        weights[pos] = picked / max(float(picked.sum()), 1e-12)

    neighbor_expr = data_np[sampled]
    neighbor_mean = np.sum(neighbor_expr * weights[:, :, None], axis=1).astype(np.float32)
    neighbor_t = torch.as_tensor(neighbor_mean, dtype=batch_x.dtype, device=batch_x.device)
    alpha = float(alpha)
    return alpha * batch_x + (1.0 - alpha) * neighbor_t


def apply_scmae_noise(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    应用 scMAE 风格的掩码扰动：将一部分位置替换为其他细胞对应位置的值。

    对于 x 中的每个元素，都会以 `mask_ratio` 的概率被独立替换为
    随机打乱后的 x 中对应位置的值。返回的 mask 张量用于标记哪些位置被修改：
    1.0 表示发生了替换，0.0 表示保持不变。
    """
    should_swap = torch.bernoulli(float(mask_ratio) * torch.ones_like(x))
    if x.shape[0] <= 1:
        replacement = x
    else:
        replacement = x[torch.randperm(x.shape[0], device=x.device)]
    corrupted = torch.where(should_swap.bool(), replacement, x)
    mask = (corrupted != x).float()
    return corrupted, mask


@torch.no_grad()
def extract_embedding(model: AutoEncoder, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    在推理模式下提取全部样本的潜在表示与标签。

    函数会关闭梯度计算，并调用模型的 feature() 接口获取 embedding。
    所有 batch 的结果最终会在 CPU 上拼接为完整矩阵，
    同时对 NaN/Inf 做安全替换。
    """
    model.eval()
    embeddings = []
    labels = []
    for _, x, y in loader:
        z = model.feature(x.to(device))
        embeddings.append(z.detach().cpu().numpy())
        labels.append(y.numpy())
    emb = np.concatenate(embeddings, axis=0).astype(np.float32)
    labels_np = np.concatenate(labels, axis=0).astype(np.int64)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    return emb, labels_np


def save_embedding_h5(path: Path, embedding: np.ndarray, labels: np.ndarray) -> None:
    """将 embedding 与标签保存为 HDF5 文件。"""
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=embedding.astype(np.float32))
        handle.create_dataset("labels", data=labels.astype(np.int64))


def best_map(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """使用最优一一映射将预测聚类标签对齐到真实标签空间。"""
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    n = max(len(true_values), len(pred_values))
    counts = np.zeros((n, n), dtype=np.int64)
    for i, true_label in enumerate(true_values):
        for j, pred_label in enumerate(pred_values):
            counts[i, j] = int(np.sum((y_true == true_label) & (y_pred == pred_label)))
    rows, cols = linear_sum_assignment(-counts)
    mapped = np.zeros_like(y_pred, dtype=np.int64)
    for row, col in zip(rows, cols):
        if row < len(true_values) and col < len(pred_values):
            mapped[y_pred == pred_values[col]] = true_values[row]
    return mapped


def compute_kmeans_metrics(labels: np.ndarray, pred: np.ndarray) -> tuple[dict, np.ndarray]:
    """计算 K-Means 聚类结果的一组标准评估指标。"""
    mapped = best_map(labels, pred)
    metrics = {
        "acc": float(np.mean(mapped == labels)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
        "f1_macro": float(f1_score(labels, mapped, average="macro", zero_division=0)),
        "fmi": float(fowlkes_mallows_score(labels, pred)),
        "v_measure": float(v_measure_score(labels, pred)),
        "homogeneity": float(homogeneity_score(labels, pred)),
        "completeness": float(completeness_score(labels, pred)),
        "n_pred_clusters": int(len(np.unique(pred))),
        "silhouette": float("nan"),
        "protocol": "fixed",
        "cluster_method": "kmeans_known_k",
        "uses_known_k": True,
    }
    return metrics, mapped.astype(np.int64)


def write_kmeans_known_k_outputs(
    output_dir: Path,
    dataset: str,
    method: str,
    seed: int,
    embedding: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    extra: dict,
) -> dict:
    """在已知聚类数的前提下运行 K-Means，并写出评估结果与预测文件。"""
    pred = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(embedding)
    metrics, mapped = compute_kmeans_metrics(labels, pred.astype(np.int64))
    fixed = {"kmeans_known_k": metrics}
    row = {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        **extra,
        **metrics,
    }
    pd.DataFrame([row]).to_csv(output_dir / "eval_fixed.csv", index=False)
    payload = {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "fixed": fixed,
        "oracle": {},
        "sweep": [],
    }
    save_json(payload, str(output_dir / "eval_metrics.json"))
    np.save(output_dir / "eval_kmeans_known_k.npy", pred.astype(np.int64))
    np.save(output_dir / "eval_kmeans_known_k_mapped.npy", mapped)
    return {"fixed": fixed, "preds": {"kmeans_known_k": pred.astype(np.int64)}}


def main():
    """完整的 NeighborMix-scMAE 流程：加载数据、构建 KNN 图、训练、评估并保存输出。"""
    args = parse_args()
    family.set_seed(args.seed)
    save_dir = Path(ensure_dir(args.save_dir))
    save_json(vars(args), str(save_dir / "args.json"))

    device = family.get_device(args.gpu, args.no_cuda)
    print(f"Using device: {device}")

    bundle = family.load_scmae_dataset(
        file_path=args.data_path,
        input_mode=args.input_mode,
        n_top_genes=args.n_top_genes,
        target_sum=args.target_sum,
        scale_input=args.scale_input,
        label_key=args.label_key,
        seed=args.seed,
    )
    save_json(bundle.profile, str(save_dir / "dataset_profile.json"))
    save_json(bundle.preprocess_config, str(save_dir / "preprocess_config.json"))
    with open(save_dir / "selected_genes.txt", "w", encoding="utf-8") as handle:
        for gene in bundle.gene_names:
            handle.write(f"{gene}\n")

    data_np = bundle.data
    labels = bundle.labels
    n_clusters = int(args.n_clusters) if args.n_clusters and args.n_clusters > 0 else int(len(np.unique(labels)))
    dataset_name = args.dataset_name or Path(args.data_path).stem
    print(f"Cells={data_np.shape[0]} genes={data_np.shape[1]} clusters={n_clusters} variant={args.variant_name}")

    neighbor_indices, neighbor_probs, neighbor_profile = build_knn_distribution(
        data_np,
        k=args.neighbor_k,
        pca_dim=args.knn_pca_dim,
        tau=args.tau,
        seed=args.seed,
    )
    save_json(neighbor_profile, str(save_dir / "neighbor_graph_profile.json"))
    pseudo_branch_enabled = (
        bool(args.use_pseudo)
        and int(args.neighbor_k) > 0
        and int(args.mix_neighbors) > 0
        and float(args.pseudo_weight) > 0.0
    )

    dataset = family.IndexedExpressionDataset(data_np, labels)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    eval_loader = DataLoader(dataset, batch_size=max(args.batch_size * 4, 512), shuffle=False, drop_last=False)

    model = AutoEncoder(
        num_genes=data_np.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        masked_data_weight=args.masked_data_weight,
        mask_loss_weight=args.mask_loss_weight,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(args.seed + 2027)

    history = {
        "loss": [],
        "real_loss": [],
        "pseudo_loss": [],
        "pseudo_weight": [],
        "real_mask_rate": [],
        "pseudo_mask_rate": [],
        "configured_mask_ratio": float(args.mask_ratio),
        "mask_rate_note": "real_mask_rate and pseudo_mask_rate are effective rates where corrupted values differ from original values.",
    }
    epoch_metric_keys = [
        "loss",
        "real_loss",
        "pseudo_loss",
        "real_mask_rate",
        "pseudo_mask_rate",
    ]

    for epoch in range(1, max(1, args.epochs) + 1):
        model.train()
        totals = {key: 0.0 for key in epoch_metric_keys}
        n_batches = 0

        for idx_t, x_cpu, _ in train_loader:
            idx_np = idx_t.numpy().astype(np.int64, copy=False)
            x = x_cpu.to(device)

            x_corrupt, real_mask = family.apply_scmae_noise(x, args.mask_ratio)
            _, loss_real = model.loss_mask(x_corrupt, x, real_mask)

            loss = loss_real
            loss_pseudo = torch.zeros((), device=device, dtype=loss_real.dtype)
            pseudo_mask_rate = torch.zeros((), device=device, dtype=loss_real.dtype)

            if pseudo_branch_enabled:
                x_prime = sample_mix(
                    data_np=data_np,
                    batch_indices=idx_np,
                    batch_x=x,
                    alpha=args.alpha,
                    mix_neighbors=args.mix_neighbors,
                    rng=rng,
                    neighbor_indices=neighbor_indices,
                    neighbor_probs=neighbor_probs,
                )
                x_prime = x_prime.detach()

                xp_corrupt, pseudo_mask = family.apply_scmae_noise(x_prime, args.mask_ratio)
                _, loss_pseudo = model.loss_mask(xp_corrupt, x, pseudo_mask)
                pseudo_mask_rate = pseudo_mask.mean()
                loss = loss + float(args.pseudo_weight) * loss_pseudo

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["real_loss"] += float(loss_real.detach().cpu())
            totals["pseudo_loss"] += float(loss_pseudo.detach().cpu())
            totals["real_mask_rate"] += float(real_mask.mean().detach().cpu())
            totals["pseudo_mask_rate"] += float(pseudo_mask_rate.detach().cpu())
            n_batches += 1

        for key in epoch_metric_keys:
            history[key].append(totals[key] / max(1, n_batches))
        history["pseudo_weight"].append(float(args.pseudo_weight) if pseudo_branch_enabled else 0.0)

        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"loss={history['loss'][-1]:.4f} "
                f"real={history['real_loss'][-1]:.4f} "
                f"pseudo={history['pseudo_loss'][-1]:.4f} "
                f"pseudo_w={history['pseudo_weight'][-1]:.4f}"
            )

    embedding, labels_out = family.extract_embedding(model, eval_loader, device)
    save_json(history, str(save_dir / "training_history.json"))
    if not args.lightweight_outputs:
        np.save(save_dir / "embedding_final.npy", embedding.astype(np.float32))
        np.save(save_dir / "embeddings_base.npy", embedding.astype(np.float32))
        np.save(save_dir / "labels.npy", labels_out.astype(np.int64))
        np.save(save_dir / "gene_names.npy", bundle.gene_names.astype(str))
        family.save_embedding_h5(save_dir / "embedding.h5", embedding, labels_out)
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "gene_names": bundle.gene_names.astype(str),
                "neighbor_profile": neighbor_profile,
            },
            save_dir / "model.pt",
        )

    result = None
    eval_extra = {
        "variant": args.variant_name,
        "use_pseudo": bool(args.use_pseudo),
        "pseudo_weight": float(args.pseudo_weight),
        "alpha": float(args.alpha),
        "neighbor_k": int(args.neighbor_k),
        "mix_neighbors": int(args.mix_neighbors),
        "mask_ratio": float(args.mask_ratio),
    }
    if not args.skip_eval:
        result = family.write_kmeans_known_k_outputs(
            output_dir=save_dir,
            dataset=dataset_name,
            method=args.method_name,
            seed=args.seed,
            embedding=embedding,
            labels=labels_out,
            n_clusters=n_clusters,
            extra=eval_extra,
        )
        save_json(result["fixed"], str(save_dir / "metrics.json"))

    if not args.no_save_h5ad:
        bundle.adata.obsm["X_neighbormix_scmae"] = embedding
        bundle.adata.uns["neighbormix_scmae"] = {
            "method": args.method_name,
            "variant": args.variant_name,
            "use_pseudo": bool(args.use_pseudo),
            "pseudo_weight": float(args.pseudo_weight),
            "alpha": float(args.alpha),
            "neighbor_k": int(args.neighbor_k),
            "mix_neighbors": int(args.mix_neighbors),
            "mask_ratio": float(args.mask_ratio),
        }
        sanitize_anndata_for_write(bundle.adata)
        bundle.adata.write_h5ad(save_dir / "adata_neighbormix_scmae.h5ad", compression="gzip")

    summary = {
        "dataset": dataset_name,
        "method": args.method_name,
        "variant": args.variant_name,
        "seed": int(args.seed),
        "n_cells": int(data_np.shape[0]),
        "n_genes": int(data_np.shape[1]),
        "n_clusters": int(n_clusters),
        "embedding_path": "" if args.lightweight_outputs else str((save_dir / "embedding_final.npy").resolve()),
        "lightweight_outputs": bool(args.lightweight_outputs),
        "fixed_metrics": result["fixed"] if result is not None else {},
        "note": "Pseudo-cell branch is used only as an anchor-recovery training view; final embeddings are extracted from clean real cells.",
    }
    save_json(summary, str(save_dir / "summary.json"))
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
