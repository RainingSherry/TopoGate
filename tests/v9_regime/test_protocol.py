from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.v9_regime.discover_external import _entry_metadata
from scripts.v9_regime.lock_panel import lock_panel
from scripts.v9_regime.lock_split import lock_split
from scripts.v9_regime.protocol import VARIANT_OVERRIDES, build_x_only_features, standardize_x


def test_x_only_features_have_expected_graph_fields() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(48, 12)).astype(np.float32)
    x[:4, 0] = 0.0
    features = build_x_only_features(x, seed=20260806, max_analysis_samples=48, max_analysis_features=12)
    for key in (
        "n", "d", "zero_fraction", "analysis_pca_dim", "mean_mutual_ratio",
        "mean_snn", "reliability_entropy", "effective_neighbor_count",
    ):
        assert key in features
    assert features["n"] == 48
    assert features["d"] == 12


def test_standardization_is_independent_of_labels() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 6)).astype(np.float32)
    z1, meta1 = standardize_x(x)
    z2, meta2 = standardize_x(x.copy())
    np.testing.assert_array_equal(z1, z2)
    assert meta1 == meta2


def test_scmae_variant_disables_topology_without_changing_primary_variants() -> None:
    assert VARIANT_OVERRIDES["scmae"] == {
        "gate_mode": "none",
        "mix_mode": "none",
        "pseudo_weight": 0.0,
    }
    assert VARIANT_OVERRIDES["nomix"]["gate_mode"] == "learned"
    assert VARIANT_OVERRIDES["full"]["mix_mode"] == "reliability"


def test_split_contains_no_outcome_or_label_fields(tmp_path: Path) -> None:
    manifest = {
        "protocol_id": "v9_regime_protocol_v1",
        "datasets": [
            {"dataset_id": f"local__d{i}", "status": "eligible", "status_reason": None}
            for i in range(8)
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    feature_path = tmp_path / "features.csv"
    feature_path.write_text(
        "dataset_id,family,n,d,zero_fraction,cv_knn_distance,graph_largest_component_fraction,feature_error\n"
        + "\n".join(f"local__d{i},tabular,100,{i + 2},0.1,0.2,1.0," for i in range(8)),
        encoding="utf-8",
    )
    output = tmp_path / "split.json"
    lock_split(manifest_path, feature_path, output, seed=20260806)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["split_uses_labels_or_outcomes"] is False
    assert all("ari" not in json.dumps(row).lower() for row in payload["assignments"])


def test_openml_quality_array_is_parsed() -> None:
    entry = {
        "did": 2,
        "quality": [
            {"name": "NumberOfInstances", "value": "898.0"},
            {"name": "NumberOfFeatures", "value": "39.0"},
            {"name": "NumberOfClasses", "value": "5.0"},
            {"name": "NumberOfSymbolicFeatures", "value": "33.0"},
        ],
    }
    assert _entry_metadata(entry) == (898, 39, 5, 33)


def test_panel_lock_is_x_only(tmp_path: Path) -> None:
    ids = [f"local__p{i}" for i in range(12)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"datasets": [{"dataset_id": dataset_id, "status": "eligible"} for dataset_id in ids]}),
        encoding="utf-8",
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"assignments": [{"dataset_id": dataset_id, "split": "discovery"} for dataset_id in ids]}),
        encoding="utf-8",
    )
    feature_path = tmp_path / "features.csv"
    feature_path.write_text(
        "dataset_id,family,n,d,zero_fraction,cv_knn_distance,mean_mutual_ratio,mean_snn,graph_largest_component_fraction,graph_components,feature_error\n"
        + "\n".join(
            f"{dataset_id},tabular,{100+i},10,{i/12:.4f},{i/12:.4f},{1-i/12:.4f},{1-i/12:.4f},{i/12:.4f},{12-i},"
            for i, dataset_id in enumerate(ids)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "panel.json"
    payload = lock_panel(manifest_path, feature_path, split_path, output, per_role=1)
    assert payload["selection_uses_labels_or_outcomes"] is False
    assert payload["panel_ids"]
    assert all("ari" not in json.dumps(row).lower() for row in payload["assignments"])
