# -*- coding: utf-8 -*-
"""
methods/shared_utils.py — 最小公共工具集

仅包含被多个方法 runner 直接依赖的基础工具函数。
不放模型预处理逻辑、不放复杂抽象。

本文件解除了 methods/ 下所有方法对 PlantSPADE_LGCL 运行时的依赖。
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str) -> str:
    """创建目录（递归），目录已存在时不报错。"""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(payload: dict, path: str) -> None:
    """将字典序列化为 JSON 文件（2-space indent，default=str）。"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def set_seed(seed: int = 42) -> None:
    """
    为所有随机数生成器设置种子（Python、NumPy、PyTorch、CUDA）。
    开启 cudnn.deterministic 并关闭 benchmark，保证可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_anndata_for_write(adata) -> None:
    """
    清理 AnnData 对象中与 HDF5 写入冲突的保留列名/索引名。

    - 若 adata.obs 或 adata.var 中存在 "_index" 或 "reserved_index" 列，
      将其重命名为带后缀的新名字。
    - 若行索引名称为 "_index"，将其改为 "cell_name"（obs）或 "gene_name"（var）。
    """
    for frame in (adata.obs, adata.var):
        for reserved in ("_index", "reserved_index"):
            if reserved in frame.columns:
                replacement = reserved + "_renamed"
                suffix = 1
                while replacement in frame.columns:
                    replacement = f"{reserved}_renamed_{suffix}"
                    suffix += 1
                frame.rename(columns={reserved: replacement}, inplace=True)
        if frame.index.name == "_index":
            frame.index.name = "cell_name" if frame is adata.obs else "gene_name"
