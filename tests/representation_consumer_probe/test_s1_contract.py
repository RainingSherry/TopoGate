from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.representation_consumer_probe.s1_opportunity import (  # noqa: E402
    MATERIALITY_DELTA,
    _artifact_hash_manifest,
    _verify_artifact_hashes,
    _accuracy_by_optimal_mapping,
    _effect_summary,
    feature_only_embedding,
    graph_diagnostics,
)


def test_effect_summary_requires_material_mean_and_two_positive_seeds() -> None:
    positive = _effect_summary([0.04, 0.05, 0.01])
    assert positive["material_positive"] is True
    assert positive["positive_seed_count"] == 3
    small = _effect_summary([0.04, -0.04, 0.01])
    assert small["material_positive"] is False
    assert small["classification"] == "observed_small"
    assert MATERIALITY_DELTA == 0.03


def test_accuracy_mapping_and_graph_diagnostics_are_posthoc_only() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([1, 1, 0, 0])
    assert _accuracy_by_optimal_mapping(labels, predictions) == 1.0
    graph = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.8],
                [0.0, 0.0, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
    )
    diagnostics = graph_diagnostics(graph, labels)
    assert diagnostics["labels_used_for_diagnostic"] is True
    assert diagnostics["connected_components"] == 2
    assert diagnostics["isolated_nodes"] == 0


def test_feature_only_arm_uses_raw_h0() -> None:
    h0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(feature_only_embedding(h0), h0)


def test_artifact_hash_manifest_rejects_unlisted_extra_file(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "summary.json").write_text("{}\n", encoding="utf-8")
    (root / "artifact_hashes.json").write_text(
        json.dumps(_artifact_hash_manifest(root), indent=2) + "\n", encoding="utf-8"
    )
    assert _verify_artifact_hashes(root)
    (root / "unlisted.txt").write_text("extra\n", encoding="utf-8")
    assert not _verify_artifact_hashes(root)
