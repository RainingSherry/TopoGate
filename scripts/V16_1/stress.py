from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


COMPOUND_STRESS: dict[str, float] = {
    "feature_dropout_fraction": 0.20,
    "count_noise_rate": 0.20,
    "row_contamination_fraction": 0.10,
}


def apply_compound_stress(
    X: sp.spmatrix,
    seed: int,
    *,
    settings: dict[str, float] | None = None,
) -> tuple[sp.csr_matrix, dict[str, Any], np.ndarray]:
    """Apply the fixed V16 compound count-preserving stress condition."""
    params = dict(COMPOUND_STRESS if settings is None else settings)
    rng = np.random.default_rng(int(seed))
    output = sp.csr_matrix(X, dtype=np.int64, copy=True)
    dropout = float(params["feature_dropout_fraction"])
    noise_rate = float(params["count_noise_rate"])
    row_fraction = float(params["row_contamination_fraction"])
    if not 0.0 <= dropout < 1.0 or noise_rate < 0.0 or not 0.0 <= row_fraction <= 1.0:
        raise ValueError("invalid fixed compound stress settings")
    if output.data.size:
        keep = rng.random(output.data.size) >= dropout
        output.data[~keep] = 0
        output.eliminate_zeros()
        if output.data.size and noise_rate > 0.0:
            output.data += rng.poisson(np.maximum(output.data, 1) * noise_rate).astype(np.int64)
    output.sort_indices()
    n = int(output.shape[0])
    count = min(n, int(round(n * row_fraction))) if n else 0
    contaminated = np.zeros(n, dtype=np.uint8)
    if count:
        rows = rng.choice(n, size=count, replace=False)
        donors = rng.choice(n, size=count, replace=True)
        mutable = output.tolil(copy=True)
        for target, donor in zip(rows.tolist(), donors.tolist()):
            mutable[int(target)] = output.getrow(int(donor))
        output = mutable.tocsr()
        contaminated[rows] = 1
    metadata = {
        "mode": "compound",
        "changed": True,
        **params,
        "contaminated_rows": int(contaminated.sum()),
        "mask_kind": "row",
    }
    return output, metadata, contaminated
