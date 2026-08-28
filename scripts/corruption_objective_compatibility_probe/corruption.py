"""Read-only adapter to the audited C2 P0/P2 corruption implementation."""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts.sparse_corruption_principle_probe.corruption_library import corrupt_matrix, support_mask


def clean_matrix(clean: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(clean, dtype=np.float32)
    changed = np.zeros_like(matrix, dtype=bool)
    return matrix.copy(), {
        "principle": "Clean",
        "changed_mask": changed,
        "exact_budget": True,
        "effective_changed_coordinate_rate": 0.0,
        "support_change_rate": 0.0,
        "value_change_rate": 0.0,
        "total_absolute_change": 0.0,
        "labels_used": False,
    }


def make_corruption(clean: np.ndarray, arm: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    if arm == "Clean":
        return clean_matrix(clean)
    if arm == "P0_Random":
        return corrupt_matrix(clean, "P0_Random", rng)
    if arm == "P2_SupportTarget":
        return corrupt_matrix(clean, "P2_SupportTarget", rng)
    raise ValueError(f"unsupported corruption arm: {arm}")
