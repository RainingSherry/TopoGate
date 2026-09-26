"""Small unit checks for the frozen source plans; no model training or GPU use."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

_PAPERS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPERS / "多跳Top5扩充_无测试集调参_20260923"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_LOCAL_METHODS = Path(__file__).resolve().parents[2] / "Temp" / "topogate_repo_audit_20260915"
if _LOCAL_METHODS.exists():
    sys.path.insert(0, str(_LOCAL_METHODS))

from v0rg_tunable_multihop.graph import NeighborGraph
from v0rg_tunable_multihop.config import V0_RGConfig
from v0rg_tunable_multihop.hop_sampling import build_exact_hop_plan
from v0rg_tunable_multihop.hop_sampling import estimate_neighbors_from_plan
from v0rg_tunable_multihop import hop_sampling
from intervention_plan import transform_plan


def _fixture():
    n = 22
    indices = np.asarray([[(row + 1) % n, (row + 2) % n] for row in range(n)], dtype=np.int64)
    weights = np.full((n, 2), 0.5, dtype=np.float32)
    rng = np.random.default_rng(9)
    embedding = rng.normal(size=(n, 5)).astype(np.float32)
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    graph = NeighborGraph(indices=indices, probs=weights, similarity=weights,
                          distance=weights, embedding=embedding,
                          mutual=np.zeros((n, 2), dtype=bool), snn=weights,
                          profile={})
    config = V0_RGConfig(hop_variant="H8_B2", neighbor_k=2, mix_neighbors=2,
                         knn_pca_dim=3)
    reference = build_exact_hop_plan(graph, weights, "H8_B2", 2)
    return graph, weights, config, reference


def test_zero_intervention_is_exact_reference():
    graph, weights, config, reference = _fixture()
    for arm in ("reference", "rescue100", "mixed_target"):
        plan, audit = transform_plan(reference, graph, weights, config, arm, 42)
        assert plan is reference
        assert audit["original_hops"] is reference.hop_labels


def test_replace_and_rescue_preserve_slots_mass_and_self_exclusion():
    graph, weights, config, reference = _fixture()
    replaced, _ = transform_plan(reference, graph, weights, config,
                                 "replace_far_le4", 42)
    rescue, _ = transform_plan(reference, graph, weights, config, "rescue50", 42)
    for anchor, original in enumerate(reference.indices):
        far = reference.hop_labels[anchor] >= 5
        assert np.array_equal(reference.probs[anchor], replaced.probs[anchor])
        assert np.array_equal(reference.probs[anchor], rescue.probs[anchor])
        assert len(original) == len(replaced.indices[anchor]) == len(rescue.indices[anchor])
        assert anchor not in replaced.indices[anchor]
        assert anchor not in rescue.indices[anchor]
        assert np.all(replaced.hop_labels[anchor][far] <= 4)
        restored = rescue.indices[anchor][far] == original[far]
        assert int(restored.sum()) == int(np.floor(far.sum() / 2))
        assert not np.any(replaced.indices[anchor][far] == original[far])
        assert np.all(rescue.indices[anchor][~far] == original[~far])


def test_kwide_and_shuffle_match_count_without_self_or_source_mutation():
    graph, weights, config, reference = _fixture()
    source_path = Path(hop_sampling.__file__).resolve()
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    for arm in ("kwide_countmatched", "far_shuffle"):
        plan, _ = transform_plan(reference, graph, weights, config, arm, 42,
                                 feature_nonzero=np.arange(len(reference.indices)))
        assert plan.draw_count == reference.draw_count
        for anchor, nodes in enumerate(plan.indices):
            assert len(nodes) == len(reference.indices[anchor])
            assert anchor not in nodes
            assert np.isclose(plan.probs[anchor].sum(), 1.0)
            if arm == "far_shuffle":
                assert np.array_equal(plan.probs[anchor], reference.probs[anchor])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before


def test_traced_reference_sampler_is_exact_and_rng_preserving():
    from step5_interventions import TraceCollector, _trace_plan_estimator
    graph, weights, config, plan = _fixture()
    data = np.random.default_rng(3).normal(size=(len(plan.indices), 7)).astype(np.float32)
    batch_indices = np.asarray([0, 4, 7, 9, 12], dtype=np.int64)
    for estimator in ("current", "uniform_sample", "full"):
        original_rng = np.random.default_rng(123)
        traced_rng = np.random.default_rng(123)
        original_x, original_usage = estimate_neighbors_from_plan(
            data, batch_indices, plan, original_rng, estimator)
        with TemporaryDirectory() as directory:
            collector = TraceCollector(Path(directory) / "trace.bin", len(plan.indices))
            collector.original_hops = plan.hop_labels
            traced = _trace_plan_estimator(collector, estimate_neighbors_from_plan)
            traced_x, traced_usage = traced(data, batch_indices, plan, traced_rng, estimator)
            collector.finish()
            assert np.array_equal(traced_x, original_x)
            assert traced_usage.keys() == original_usage.keys()
            assert traced_usage["actual_sampling_mass"] == original_usage["actual_sampling_mass"]
            assert traced_usage["draw_fraction"] == original_usage["draw_fraction"]
            for hop, values in original_usage["_per_sample_actual_sampling_mass"].items():
                assert np.array_equal(traced_usage["_per_sample_actual_sampling_mass"][hop], values)
            assert int(traced_rng.integers(0, 2**31)) == int(original_rng.integers(0, 2**31))


def test_norm_match_preserves_training_rng_and_target_norm():
    from step5_interventions import _clone_numpy_generator, _match_displacement_norm
    graph, weights, config, plan = _fixture()
    data = np.random.default_rng(3).normal(size=(len(plan.indices), 7)).astype(np.float32)
    batch_indices = np.asarray([0, 4, 7, 9, 12], dtype=np.int64)
    rng = np.random.default_rng(72)
    control = np.random.default_rng(72)
    copied = _clone_numpy_generator(rng)
    intervention, _ = estimate_neighbors_from_plan(data, batch_indices, plan, rng, "current")
    expected, _ = estimate_neighbors_from_plan(data, batch_indices, plan, control, "current")
    reference, _ = estimate_neighbors_from_plan(data, batch_indices, plan, copied, "current")
    assert np.array_equal(intervention, expected)
    assert np.array_equal(reference, expected)
    anchor = data[batch_indices]
    mixed = anchor + 0.4 * (intervention - anchor)
    reference_mixed = anchor + 0.2 * (reference - anchor)
    adjusted, audit = _match_displacement_norm(anchor, mixed, reference_mixed)
    assert np.allclose(np.linalg.norm(adjusted - anchor, axis=1),
                       np.linalg.norm(reference_mixed - anchor, axis=1), rtol=1e-5, atol=1e-6)
    assert audit["unmatched"] == 0
    assert int(rng.integers(0, 2**31)) == int(control.integers(0, 2**31))


def test_tiny_fit_smoke_mixed_target_and_h1():
    from step5_interventions import TraceCollector, _patched_training, trainer
    X = np.random.default_rng(11).normal(size=(24, 12)).astype(np.float32)
    for arm in ("mixed_target", "h1_matched_amplitude"):
        config = V0_RGConfig(hop_variant="H8_B2" if arm == "mixed_target" else "H1",
                             hidden_size=16, epochs=1, batch_size=8,
                             neighbor_k=2, mix_neighbors=2, knn_pca_dim=3,
                             kmeans_n_init=1)
        with TemporaryDirectory() as directory:
            collector = TraceCollector(Path(directory) / "trace.bin", len(X))
            collector.config = config
            with _patched_training(arm, 42, np.count_nonzero(X, axis=1), collector):
                predictions, embedding, diagnostics = trainer.fit_predict(
                    X, fit_X=X, n_clusters=3, config=config, seed=42, device="cpu")
            audit = collector.finish()
            assert predictions.shape == (len(X),)
            assert embedding.shape[0] == len(X)
            assert audit["draws"] > 0
            assert np.isfinite(diagnostics["training_history"]["loss"]).all()
