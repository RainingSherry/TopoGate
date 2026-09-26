"""Small distribution and leakage checks for the frozen mechanism audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "steps234_information.py"
CORE_PARENT = HERE.parent / "多跳Top5扩充_无测试集调参_20260923"
SPEC = importlib.util.spec_from_file_location("steps234_information", SOURCE)
assert SPEC and SPEC.loader
steps = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(steps)


def synthetic_graph():
    core = steps.load_core(CORE_PARENT)
    graph_type = __import__("v0rg_tunable_multihop.graph", fromlist=["NeighborGraph"]).NeighborGraph
    n = 24
    indices = np.asarray([[(i + 1) % n, (i + 3) % n] for i in range(n)], dtype=np.int64)
    p = np.tile(np.asarray([.7, .3], dtype=np.float32), (n, 1))
    graph = graph_type(indices, p, np.zeros_like(p), np.zeros_like(p),
                       np.zeros((n, 2), dtype=np.float32),
                       np.zeros_like(p, dtype=bool), np.zeros_like(p), {})
    return core, graph, p


def test_partial_exact_shell_matches_original_full_plan():
    core, graph, weights = synthetic_graph()
    original = __import__("v0rg_tunable_multihop.hop_sampling",
                          fromlist=["build_exact_hop_plan"]).build_exact_hop_plan
    full = original(graph, weights, "H8_B2", 2, 1.5)
    transition = core["_transition_matrix"](graph, weights)
    for source in (0, 7, 23):
        row = steps.frozen_row(graph, weights, source, "H8_B2", 1.5,
                               core, transition)
        np.testing.assert_array_equal(row["indices"], full.indices[source])
        np.testing.assert_array_equal(row["hops"], full.hop_labels[source])
        np.testing.assert_allclose(row["probs"], full.probs[source], rtol=1e-10)


def test_current_estimator_double_weights_and_vector_sum():
    X = np.asarray([[0., 0.], [1., 0.], [0., 2.]], dtype=np.float32)
    row = {"indices": np.asarray([1, 2]), "probs": np.asarray([.8, .2]),
           "hops": np.asarray([1, 2])}
    result = steps.mc_decompose(X, 0, row, .2, 2, 20000,
                                np.random.default_rng(4), "current")
    assert result["sampling_mass"][0] > .8
    assert np.isclose(result["endpoint_coefficients"].sum(), 1.0)
    np.testing.assert_allclose(result["vectors"].sum(axis=0),
                               result["mean_total_vector"], atol=1e-6)
    np.testing.assert_allclose(result["mass_repeats"].sum(axis=1), 1.0)


def test_probe_excludes_targets_from_graph_and_matches_arm_dimensions(monkeypatch):
    core = steps.load_core(CORE_PARENT)
    cfg = core["V0_RGConfig"](hop_variant="H2_B1", neighbor_k=2,
                               mix_neighbors=2, knn_pca_dim=3)
    rng = np.random.default_rng(6)
    X = rng.normal(size=(32, 12)).astype(np.float32)
    anchors = np.arange(16)
    seen = []
    actual = steps.graph_and_gate
    def spy(Z, config, seed, package):
        seen.append(Z.copy())
        return actual(Z, config, seed, package)
    monkeypatch.setattr(steps, "graph_and_gate", spy)
    rows, audit = steps.feature_block_probe(
        X, cfg, anchors, 42, core, blocks=2, targets_per_block=2,
    )
    assert len(seen) == 2
    for Z, block in zip(seen, audit["blocks"]):
        targets = block["target_feature_indices_processed"]
        np.testing.assert_array_equal(Z, np.delete(X, targets, axis=1))
    for block in (0, 1):
        for fold in (0, 1):
            pair = [r for r in rows if r["block"] == block and r["fold"] == fold]
            assert len(pair) == 3
            assert len({r["input_dimension"] for r in pair}) == 1
            assert len({r["test_anchor_indices"] for r in pair}) == 1
            assert np.isfinite([r["mse"] for r in pair]).all()


def test_raw_covariates_are_not_standardized_nonzero_counts():
    raw = sp.csr_matrix(np.asarray([[0., 2., 0.], [1., 0., 3.]], dtype=np.float32))
    processed = np.asarray([[-1., 1., -1.], [1., -1., 1.]], dtype=np.float32)
    rows = [{"anchor_index": 0,
             "anchor_processed_nonzero_fraction": float(np.count_nonzero(processed[0]) / 3)},
            {"anchor_index": 1,
             "anchor_processed_nonzero_fraction": float(np.count_nonzero(processed[1]) / 3)}]
    steps.attach_raw_covariates(rows, raw, "raw_count")
    assert rows[0]["anchor_processed_nonzero_fraction"] == 1.0
    assert rows[0]["raw_nonzero_count"] == 1
    assert rows[0]["raw_nonzero_fraction"] == 1 / 3
    assert rows[0]["raw_row_sum"] == rows[0]["raw_library_size"] == 2.0
    assert rows[1]["raw_nonzero_count"] == 2
    assert rows[1]["raw_library_size"] == 4.0


def test_saturated_graph_has_zero_padded_high_hops():
    core = steps.load_core(CORE_PARENT)
    graph_type = __import__("v0rg_tunable_multihop.graph", fromlist=["NeighborGraph"]).NeighborGraph
    n = 4
    indices = np.asarray([[j for j in range(n) if j != i] for i in range(n)])
    p = np.full((n, n - 1), 1 / (n - 1), dtype=np.float32)
    graph = graph_type(indices, p, p, p, np.zeros((n, 2), dtype=np.float32),
                       np.zeros_like(p, dtype=bool), p, {})
    cfg = core["V0_RGConfig"](hop_variant="H8_B1", neighbor_k=3, mix_neighbors=2)
    X = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    anchors, hops, endpoints, frozen, vectors, mass = steps.audit_flow(
        X, graph, p, np.full(n, .1, dtype=np.float32), cfg,
        np.arange(n), 4, 42, core,
    )
    assert len(anchors) == n and len(hops) == 8 * n
    assert vectors.shape == (n, 8, X.shape[1])
    assert mass.shape == (n, 4, 8)
    assert np.all(vectors[:, 1:] == 0)
    assert np.all(mass[:, :, 1:] == 0)
    assert all(row["candidate_count"] == 0 for row in hops if row["hop"] >= 2)
