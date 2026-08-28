from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V24_conditional_response.config import V24Q1Config


def tiny_config() -> V24Q1Config:
    return V24Q1Config(
        n_samples=120,
        n_features=40,
        n_clusters=3,
        zero_fraction=0.50,
        block_size=4,
        active_blocks_per_sample=5,
        fingerprint_masks=3,
        fingerprint_mask_ratio=0.25,
        dependency_separation_min=0.05,
        outer_folds=3,
        inner_folds=2,
        pair_count_per_fold=20,
        bootstrap_replicates=3,
        calibration_replicates=2,
    )
