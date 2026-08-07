from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    f1_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from methods.shared_utils import save_json


# AnnData.obs 中常见的细胞类型标签列名列表。
# 自动检测时会按出现顺序依次搜索这些键。
LABEL_CANDIDATES = [
    "resolved_label",   # from prepare_dataset.py output
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
    "true_label",
]


@dataclass
class DataBundle:
    """用于存放已预处理、可直接用于模型训练/推理的 scRNA-seq 数据集。"""

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
    PyTorch 数据集，返回 `(index, expression_vector, label)` 元组。

    每个样本对应一个细胞的基因表达谱及其标签。
    同时返回整数索引，便于下游代码追踪每个 batch 元素对应的原始细胞。
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int):
        return int(idx), self.data[idx], self.labels[idx]


def str2bool(value):
    """
    将布尔值的字符串表示转换为 Python 的 bool。

    对于 True，接受 'true'、't'、'yes'、'y'、'1'；
    对于 False，接受 'false'、'f'、'no'、'n'、'0'。
    若输入无法识别，则抛出 ArgumentTypeError，
    以便 argparse 生成更友好的错误提示。
    """
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def set_seed(seed: int) -> None:
    """
    为所有随机数生成器设置种子（Python、NumPy、PyTorch、CUDA）。

    这样可以尽量保证多次运行结果可复现。若 CUDA 可用，函数会同时调用
    torch.cuda.manual_seed 和 torch.cuda.manual_seed_all，并开启
    cudnn.deterministic 以避免使用非确定性的卷积算法。
    同时关闭 cudnn.benchmark，因为即使开启了 deterministic，
    benchmark 仍可能引入非确定性。
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
    根据 GPU 可用性与隔离约束，解析合适的 torch.device。

    若设置了 CUDA_VISIBLE_DEVICES，本函数会将 --gpu 指定的逻辑 GPU 索引
    映射到当前可见的物理 GPU。根据使用策略，物理 GPU 0 和 7 被禁止使用，
    以避免与其他任务发生冲突。若 CUDA 不可用或 no_cuda 为 True，
    则返回 CPU 设备。
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


def ensure_csr(matrix) -> sp.csr_matrix:
    """
    将矩阵转换为 float32 的 CSR 格式，并将无效值替换为 0。

    同时兼容稀疏与稠密输入。负值会被截断为 0，NaN/Inf 会被替换为 0，
    并按列索引排序，以确保得到规范化的 CSR 表示。
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


def sample_values(matrix, max_rows: int = 256) -> np.ndarray:
    """
    从矩阵前几行采样并展平成 float32 数组。

    该函数用于在不物化整个矩阵的前提下，快速检查数值范围与分布。
    对于稀疏矩阵，仅采样行中的非零值会被纳入结果。
    返回值统一转换为 float32，便于后续启发式判断。
    """
    sample = matrix[: min(max_rows, matrix.shape[0])]
    if sp.issparse(sample):
        return sample.data.astype(np.float32, copy=False) if sample.nnz else np.array([], dtype=np.float32)
    return np.asarray(sample, dtype=np.float32).ravel()


def looks_like_raw_counts(matrix) -> bool:
    """
    启发式判断矩阵是否像原始（未归一化）计数数据。

    若采样到的所有值都非负，且都与其四舍五入后的整数值近似相等
    （误差不超过 1e-4），则返回 True，这通常是整数计数数据的特征。
    空矩阵，或仅包含 NaN/Inf 的矩阵，会返回 False。
    """
    values = sample_values(matrix)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    values = values[: min(values.size, 100000)]
    return bool(np.all(values >= 0.0) and np.allclose(values, np.round(values), atol=1e-4))


def var_gene_names(var) -> tuple[np.ndarray, object]:
    """
    从 AnnData 的 var DataFrame 中提取基因名，并尝试多个常见列名。

    会按顺序检查 'gene_name'、'features'、'gene_symbols'、'symbol' 和 '_index'。
    返回提取到的基因名数组，以及一个将索引设置为这些基因名的 var 副本。
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


def select_count_source(adata: sc.AnnData, input_mode: str):
    """
    从 AnnData 对象中选择合适的表达矩阵来源。

    原始计数数据的优先级顺序为：
        1. adata.layers["counts"]
        2. adata.raw.X
        3. adata.X（若看起来像原始计数）

    对于 log1p 数据，函数会回退到 adata.X。
    当 input_mode 为 "auto" 时，会根据数据本身自动推断来源。
    若 input_mode 为 "raw" 但不存在合适的原始计数来源，则抛出 ValueError。
    返回一个五元组：
    (matrix, gene_names, var, source_description, inferred_mode)。
    """
    if input_mode in {"auto", "raw"} and "counts" in adata.layers:
        gene_names, var = var_gene_names(adata.var)
        return adata.layers["counts"], gene_names, var, "layers[counts]", "raw"
    if input_mode in {"auto", "raw"} and adata.raw is not None:
        gene_names, var = var_gene_names(adata.raw.var)
        return adata.raw.X, gene_names, var, "adata.raw.X", "raw"
    inferred = "raw" if looks_like_raw_counts(adata.X) else "log1p"
    if input_mode == "raw" and inferred != "raw":
        raise ValueError("--input_mode raw was requested, but no raw-looking X/raw/layers[counts] source is available.")
    if input_mode == "log1p":
        inferred = "log1p"
    gene_names, var = var_gene_names(adata.var)
    source = "adata.X" if inferred == "raw" else "adata.X_log1p_fallback"
    return adata.X, gene_names, var, source, inferred


def resolve_labels(adata: sc.AnnData, label_key: str):
    """
    从 AnnData 对象中解析细胞类型标签列。

    若显式提供了 label_key，则该列必须存在于 adata.obs.columns 中；
    否则函数会按顺序搜索 LABEL_CANDIDATES。
    所有标签都会通过 LabelEncoder 编码为连续整数。
    返回一个三元组：
    (encoded_labels, label_class_names, resolved_key)。
    若找不到匹配列，则抛出 KeyError。
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


def dense_float32(matrix) -> np.ndarray:
    """
    将矩阵转换为连续存储的 float32 NumPy 数组，并将 NaN/Inf 替换为 0。

    该函数统一处理稀疏与稠密输入。
    这是将数据送入 PyTorch 模型之前的最后一步，
    以确保数据类型和内存布局一致。
    """
    if sp.issparse(matrix):
        arr = matrix.toarray()
    else:
        arr = np.asarray(matrix)
    arr = np.asarray(arr, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_total_log1p(matrix, target_sum: float) -> sp.csr_matrix:
    """
    将每个细胞的表达归一化到固定总计数后，再进行 log1p 变换。

    每个细胞的表达向量都会被缩放，使其总和等于 target_sum，
    随后逐元素应用 log1p(x) = log(1 + x)。
    这模拟了 scanpy 中常见的标准化 + 对数变换流程，
    但这里直接在稀疏矩阵上操作，以提高效率。
    """
    x = ensure_csr(matrix).astype(np.float32)
    row_sum = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    scale = np.divide(
        float(target_sum),
        row_sum,
        out=np.zeros_like(row_sum, dtype=np.float32),
        where=row_sum > 0.0,
    )
    x = x.multiply(scale[:, None]).tocsr()
    x.data = np.log1p(x.data).astype(np.float32, copy=False)
    x.eliminate_zeros()
    return x


def strict_hvg_subset(work: sc.AnnData, n_top_genes: int) -> tuple[sc.AnnData, dict]:
    """Return exactly ``min(n_top_genes, n_vars)`` genes deterministically.

    Scanpy's Seurat implementation can mark every gene as highly variable when
    fewer finite normalized dispersions are available than requested.  That
    behavior is dangerous for scMAE because the decoder width is quadratic in
    the selected gene count.  We keep the standard selection when it returns
    the requested size and otherwise fall back to finite empirical variance,
    with original gene order as a deterministic tie break.
    """
    requested = max(0, int(n_top_genes))
    target = min(requested, int(work.n_vars))
    if target <= 0 or work.n_vars <= target:
        return work, {
            "requested": requested,
            "selected": int(work.n_vars),
            "strategy": "all_genes_within_limit",
        }

    strategy = "seurat"
    try:
        sc.pp.highly_variable_genes(work, flavor="seurat", n_top_genes=target, subset=False)
        mask = np.asarray(work.var["highly_variable"], dtype=bool)
    except Exception:
        mask = np.zeros(work.n_vars, dtype=bool)
    if int(mask.sum()) == target:
        selected = np.flatnonzero(mask)
    else:
        strategy = "variance_fallback"
        matrix = work.X
        if sp.issparse(matrix):
            mean = np.asarray(matrix.mean(axis=0)).ravel().astype(np.float64)
            mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel().astype(np.float64)
        else:
            values = np.asarray(matrix, dtype=np.float64)
            mean = np.mean(values, axis=0)
            mean_sq = np.mean(values * values, axis=0)
        variance = np.nan_to_num(mean_sq - mean * mean, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        order = np.lexsort((np.arange(work.n_vars, dtype=np.int64), -variance))
        selected = np.sort(order[:target])
    subset = work[:, selected].copy()
    return subset, {
        "requested": requested,
        "selected": int(subset.n_vars),
        "strategy": strategy,
        "scanpy_selected_before_fallback": int(mask.sum()),
    }


def load_scmae_dataset(
    file_path: str,
    input_mode: str,
    n_top_genes: int,
    target_sum: float,
    scale_input: bool,
    label_key: str,
    seed: int,
) -> DataBundle:
    """
    从 .h5ad 文件加载并预处理单细胞数据集。

    处理流程如下：
        1. 加载 AnnData，并解析表达矩阵来源（原始计数 / log1p / 等）
        2. 若数据看起来像原始计数，则进行总计数归一化并做 log1p 变换
        3. 若 n_top_genes > 0，则选择高变基因（HVG）
        4. 可选地将基因缩放为零均值、单位方差
        5. 提取并编码细胞类型标签
        6. 构建并返回包含 profile 与预处理元信息的 DataBundle
    """
    adata = sc.read_h5ad(file_path)
    source_x, gene_names, var, counts_source, inferred_mode = select_count_source(adata, input_mode)
    counts = ensure_csr(source_x)
    work = sc.AnnData(X=counts.copy(), obs=adata.obs.copy(), var=var.copy())
    work.obs_names = adata.obs_names.copy()
    work.var_names = gene_names.copy()

    if inferred_mode == "raw":
        work.X = normalize_total_log1p(work.X, target_sum=target_sum)
    elif sample_values(work.X).size and float(np.nanmax(sample_values(work.X))) > 30.0:
        work.X = normalize_total_log1p(work.X, target_sum=target_sum)

    hvg_summary = {
        "requested": int(n_top_genes),
        "selected": int(work.n_vars),
        "strategy": "disabled",
    }
    if n_top_genes and n_top_genes > 0:
        work, hvg_summary = strict_hvg_subset(work, n_top_genes)
    if int(work.n_vars) > int(n_top_genes) > 0:
        raise AssertionError(f"HVG contract failed: selected {work.n_vars} > requested {n_top_genes}")

    if scale_input:
        sc.pp.scale(work)

    data = dense_float32(work.X)
    labels, label_names, resolved_label_key = resolve_labels(work, label_key)
    gene_names = np.asarray(work.var_names).astype(str)

    label_counts = {
        str(key): int(value)
        for key, value in work.obs[resolved_label_key].astype(str).value_counts(dropna=False).sort_index().items()
    }
    profile = {
        "dataset_name": Path(file_path).stem,
        "n_cells": int(work.n_obs),
        "n_genes_original": int(adata.n_vars),
        "n_genes": int(work.n_vars),
        "label_key": resolved_label_key,
        "n_cell_types": int(len(label_names)),
        "cell_type_counts": label_counts,
        "counts_source": counts_source,
        "input_mode": inferred_mode,
        "has_raw": bool(adata.raw is not None),
        "has_layers_counts": bool("counts" in adata.layers),
        "scale_input": bool(scale_input),
    }
    preprocess_config = {
        "file_path": str(file_path),
        "counts_source": counts_source,
        "input_mode": inferred_mode,
        "normalization": f"normalize_total(target_sum={target_sum}) + log1p when raw",
        "hvg": {"n_top_genes": int(n_top_genes), "flavor": "seurat", **hvg_summary},
        "scale_input": bool(scale_input),
        "selected_n_genes": int(work.n_vars),
        "seed": int(seed),
        "label_key": resolved_label_key,
    }
    return DataBundle(
        adata=work,
        data=data,
        labels=labels,
        label_names=label_names,
        label_key=resolved_label_key,
        gene_names=gene_names,
        profile=profile,
        preprocess_config=preprocess_config,
    )


def apply_scmae_noise(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    应用 scMAE 风格的掩码扰动：将一部分位置替换为其他细胞对应位置的值。

    对于 x 中的每个元素，都会以 `mask_ratio` 的概率被独立抽中，并替换为
    随机打乱后的 x 中对应位置的值。返回的 mask 张量标记实际发生数值变化的位置：
    1.0 表示替换后数值不同，0.0 表示保持不变或替换后数值相同。
    因此在零值较多的稀疏 scRNA 数据上，实际有效 mask rate 可能低于配置的
    `mask_ratio`。训练脚本应记录 mask.mean() 作为 effective mask rate。
    这模拟了 scMAE 预训练阶段所使用的扰动策略。
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
def extract_embedding(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    以推理模式运行模型，并收集所有细胞的潜在 embedding。

    函数会将模型切换到 eval 模式，并通过 torch.no_grad() 禁用梯度跟踪。
    每个 batch 都会通过 model.feature() 提取 embedding，随后移动到 CPU，
    追加到列表中，最后再统一拼接。
    结果 embedding 矩阵中的 NaN/Inf 会被替换为 0。
    返回一个二元组：
    (embeddings [N x D], labels [N])。
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
    """
    将 embedding 与标签写入 HDF5 文件。

    根组下会创建两个数据集：
        - 'X'：embedding 矩阵（float32）
        - 'labels'：整数标签数组
    """
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=embedding.astype(np.float32))
        handle.create_dataset("labels", data=labels.astype(np.int64))


def best_map(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    寻找从预测聚类 ID 到真实标签的一一最优映射。

    该函数在混淆矩阵（行表示真实标签，列表示预测聚类）上使用匈牙利算法
    （scipy.linear_sum_assignment），以最大化正确匹配数量。
    返回的数组是重新映射后的预测结果：
    每个预测聚类 ID 都会被替换为与其匹配的真实标签。
    这样即使聚类标签只是任意整数编号，也能公平地评估聚类准确率。
    """
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
    """
    在最优标签重映射后，计算完整的聚类评估指标。

    指标包括 Accuracy（ACC）、Normalized Mutual Information（NMI）、
    Adjusted Rand Index（ARI）、F1-macro、Fowlkes-Mallows Index（FMI）、
    V-measure、Homogeneity 和 Completeness。
    Silhouette 被设为 NaN，因为其通常需要完整的两两距离矩阵，
    对高维 embedding 来说计算代价较高。
    返回的 mapped predictions 会通过 best_map 与真实标签对齐，
    因而 ACC 表示最佳可能的标签分配结果。
    """
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
    """
    在已知细胞类型数量的前提下运行 K-Means 聚类，并保存全部结果。

    为提高稳定性，K-Means 使用 n_init=20。
    指标通过 compute_kmeans_metrics 计算，并写入两个文件：
        - eval_fixed.csv：一行汇总所有指标
        - eval_metrics.json：包含 method/dataset/seed 等字段的结构化 JSON
    此外，原始聚类分配结果以及最优重映射后的标签也会保存为 .npy 文件。
    函数返回一个字典，其中包含 fixed-protocol 指标和预测结果数组。
    """
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
