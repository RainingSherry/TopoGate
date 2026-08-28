from __future__ import annotations

import json

import numpy as np
import torch

from methods.TopoGate.V19_rg_adapter.config import V19Config
from methods.TopoGate.V19_rg_adapter.model import apply_scmae_noise
from methods.TopoGate.V19_rg_adapter.trainer import fit_predict
from scripts.V19.tune_unsupervised_v2 import (
    BACKBONE_PROFILES,
    MECHANISM_CANDIDATES,
    candidate_catalog_for_stage,
    split_rows,
)


def test_v2_catalog_has_fixed_mechanism_and_joint_profiles(tmp_path) -> None:
    assert len(MECHANISM_CANDIDATES) == 48
    assert len(BACKBONE_PROFILES) == 8
    joint = candidate_catalog_for_stage(
        "backbone_screen",
        candidate_ids=["default__bb_base", "k5__bb_mask03"],
    )
    assert [row["candidate_id"] for row in joint] == [
        "default__bb_base",
        "k5__bb_mask03",
    ]
    selected = tmp_path / "selected_config.json"
    selected.write_text(json.dumps({"top_candidate_ids": ["default", "k5"]}))
    expanded = candidate_catalog_for_stage(
        "backbone_screen",
        selected_config=selected,
        mechanism_count=2,
    )
    assert len(expanded) == 16


def test_v2_row_split_is_reproducible_and_label_free() -> None:
    train_a, valid_a, split_a = split_rows(25, "toy__shared_text", 42)
    train_b, valid_b, split_b = split_rows(25, "toy__shared_text", 42)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(valid_a, valid_b)
    assert split_a == split_b
    assert set(train_a).isdisjoint(set(valid_a))
    assert len(train_a) + len(valid_a) == 25


def test_scmae_noise_accepts_independent_generator() -> None:
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    first, first_mask = apply_scmae_noise(x, 0.5, generator=torch.Generator().manual_seed(9))
    second, second_mask = apply_scmae_noise(x, 0.5, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_mask, second_mask)


def test_v2_fit_uses_held_out_rows_for_diagnostics() -> None:
    rng = np.random.default_rng(17)
    X = np.maximum(rng.normal(size=(28, 7)), 0.0).astype(np.float32)
    config = V19Config(
        protocol_id="v19_rg_unsup_tuning_v2",
        variant="rg_full",
        hidden_size=8,
        epochs=1,
        batch_size=7,
        neighbor_k=3,
        mix_neighbors=2,
        knn_pca_dim=4,
        n_top_features=7,
    )
    train = np.array([19, 2, 14, 5, 21, 0, 17, 8, 11, 23, 4, 26, 1, 16, 7, 24, 10, 3, 18, 12])
    valid = np.array([6, 9, 13, 15, 20, 22, 25, 27])
    predictions, _embedding, diagnostics = fit_predict(
        X,
        n_clusters=None,
        config=config,
        seed=42,
        device="cpu",
        evaluate_unsupervised=True,
        fit_X=X[train],
        evaluation_X=X[valid],
    )
    assert predictions is None
    assert diagnostics["core_summary"]["fit_n_samples"] == 20
    assert diagnostics["core_summary"]["evaluation_n_samples"] == 8
    assert diagnostics["unsupervised_diagnostics"]["validation_protocol"] == "held_out_rows"
    assert diagnostics["unsupervised_diagnostics"]["evaluation_n_samples"] == 8


def test_v2_candidate_uses_reference_evaluation_profile() -> None:
    rng = np.random.default_rng(31)
    X = np.maximum(rng.normal(size=(24, 7)), 0.0).astype(np.float32)
    candidate = V19Config(
        protocol_id="v19_rg_unsup_tuning_v2",
        variant="rg_full",
        hidden_size=8,
        epochs=1,
        batch_size=8,
        neighbor_k=5,
        mix_neighbors=2,
        knn_pca_dim=4,
        n_top_features=7,
        mask_ratio=0.5,
    )
    reference = V19Config(
        protocol_id="v19_rg_unsup_tuning_v2",
        variant="scmae_only",
        hidden_size=8,
        epochs=1,
        batch_size=8,
        neighbor_k=3,
        mix_neighbors=2,
        knn_pca_dim=4,
        n_top_features=7,
        mask_ratio=0.2,
    )
    _, _, diagnostics = fit_predict(
        X,
        n_clusters=None,
        config=candidate,
        seed=42,
        device="cpu",
        evaluate_unsupervised=True,
        fit_X=X[:16],
        evaluation_X=X[16:],
        evaluation_mask_ratio=reference.mask_ratio,
        evaluation_graph_config=reference,
    )
    proxy = diagnostics["unsupervised_diagnostics"]
    assert proxy["evaluation_mask_ratio"] == reference.mask_ratio
    assert proxy["evaluation_graph_profile"]["neighbor_k"] == reference.neighbor_k
