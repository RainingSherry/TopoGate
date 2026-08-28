from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from methods.TopoGate.V23_cycle_response.config import V23Config
from methods.TopoGate.V23_cycle_response.data import fit_semantic_preprocessor, load_matrix_only
from methods.TopoGate.V23_cycle_response.evaluation import evaluate_fingerprints
from methods.TopoGate.V23_cycle_response.masks import build_mask_dictionary, corrupt_semantic
from methods.TopoGate.V23_cycle_response.profiling import _repair, profile_fingerprints, robust_standardize
from methods.TopoGate.V23_cycle_response.synthetic import generate_worlds
from methods.TopoGate.V23_cycle_response.training import fit_backbone
from scripts.V23.run_m0_synthetic import (
    DigestCache,
    _ensure_generated_panel,
    _record_incomplete_attempt,
    _retire_incomplete_marker,
    build_jobs,
    build_stage_commands,
)


def _tiny_config() -> V23Config:
    return V23Config(
        feature_cap=12,
        hidden_size=8,
        epochs=2,
        batch_size=8,
        fingerprint_masks=4,
        fingerprint_mask_ratio=0.25,
        latent_linear_epochs=1,
        lowrank_rank=3,
        profile_batch_size=16,
    )


def test_semantic_zero_is_not_centered_zero() -> None:
    matrix = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    prepared = fit_semantic_preprocessor(matrix, input_protocol="shared_text", feature_cap=3)
    zero_positions = prepared.semantic == 0.0
    assert np.any(prepared.model[zero_positions] != 0.0)
    corrupted, effective = corrupt_semantic(
        prepared.semantic,
        np.asarray([True, False, False]),
        donor_offset=1,
        corruption_mode="zero",
    )
    assert np.all(corrupted[:, 0] == 0.0)
    assert np.array_equal(effective[:, 0], prepared.semantic[:, 0] != 0.0)


def test_mask_dictionary_is_balanced_and_has_fixed_donors() -> None:
    first = build_mask_dictionary(
        n_samples=31,
        n_features=20,
        n_masks=8,
        mask_ratio=0.25,
        mask_seed=11,
        donor_seed=13,
    )
    second = build_mask_dictionary(
        n_samples=31,
        n_features=20,
        n_masks=8,
        mask_ratio=0.25,
        mask_seed=11,
        donor_seed=13,
    )
    assert np.array_equal(first.masks, second.masks)
    assert np.array_equal(first.donor_offsets, second.donor_offsets)
    assert np.all(first.masks.sum(axis=1) == 5)
    usage = first.masks.sum(axis=0)
    assert int(usage.max() - usage.min()) <= 1
    assert np.all((first.donor_offsets > 0) & (first.donor_offsets < 31))


def test_repair_only_changes_effective_coordinates() -> None:
    matrix = np.arange(24, dtype=np.float32).reshape(6, 4)
    prepared = fit_semantic_preprocessor(matrix, input_protocol="clubench_bridge", feature_cap=4)
    reconstruction = prepared.model + 3.0
    effective = np.zeros_like(matrix, dtype=np.bool_)
    effective[::2, 1] = True
    repaired_model = _repair(prepared.semantic, reconstruction, effective, prepared)
    repaired_semantic = prepared.preprocessor.inverse_transform(repaired_model)
    assert np.allclose(repaired_semantic[~effective], prepared.semantic[~effective], atol=1e-5)
    assert np.all(np.abs(repaired_semantic[effective] - prepared.semantic[effective]) > 1e-4)


def test_fit_profile_are_label_free_and_finite() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(32, 12)).astype(np.float32)
    matrix[rng.random(matrix.shape) < 0.65] = 0.0
    prepared = fit_semantic_preprocessor(matrix, input_protocol="shared_text", feature_cap=12)
    config = _tiny_config()
    result = fit_backbone(prepared, config=config, seed=7, device=torch.device("cpu"))
    dictionary = build_mask_dictionary(
        n_samples=matrix.shape[0],
        n_features=matrix.shape[1],
        n_masks=config.fingerprint_masks,
        mask_ratio=config.fingerprint_mask_ratio,
        mask_seed=17,
        donor_seed=19,
    )
    bundle = profile_fingerprints(
        prepared,
        model=result.model,
        linear_decoder=result.linear_decoder,
        mask_dictionary=dictionary,
        config=config,
        seed=7,
        corruption_mode="donor_swap",
        device=torch.device("cpu"),
    )
    assert "cycle_repair_raw" in bundle.arrays
    assert "recovery_gain_raw" in bundle.arrays
    assert "cycle_repair_standardized" in bundle.arrays
    assert bundle.arrays["cycle_repair_raw"].shape == (32, 4)
    for value in bundle.arrays.values():
        if np.issubdtype(value.dtype, np.number):
            assert np.isfinite(value).all()
    assert bundle.diagnostics["primary_scientific_object"] == "cycle_repair_standardized"
    assert bundle.diagnostics["secondary_recoverability_object"] == "recovery_gain_standardized"


def test_fit_and_profile_function_signatures_have_no_labels_or_k() -> None:
    from methods.TopoGate.V23_cycle_response.profiling import profile_fingerprints
    from methods.TopoGate.V23_cycle_response.training import fit_backbone

    for function in (fit_backbone, profile_fingerprints):
        names = set(inspect.signature(function).parameters)
        assert "labels" not in names
        assert "y" not in names
        assert "n_clusters" not in names
        assert "K" not in names


def test_unlabelled_evaluation_without_k_skips_readout() -> None:
    arrays = {
        "clean_embedding": np.eye(8, dtype=np.float32),
        "cycle_repair_standardized": np.eye(8, dtype=np.float32),
    }
    result = evaluate_fingerprints(arrays, labels=None, external_k=None, seed=3)
    assert result.predictions == {}
    assert result.metrics["K_source"] == "unavailable"
    assert result.benchmark_validity["status"] == "unavailable_without_outer_labels"


def test_robust_standardization_is_clipped_and_raw_is_untouched() -> None:
    raw = np.asarray([[0.0], [0.0], [1e-5], [100.0]], dtype=np.float32)
    original = raw.copy()
    standardized, _, _, valid, clipped_fraction = robust_standardize(raw, epsilon=1e-8, clip=10.0)
    assert np.array_equal(raw, original)
    assert bool(valid[0])
    assert float(np.abs(standardized).max()) <= 10.0
    assert clipped_fraction > 0.0


def test_degenerate_fingerprint_is_not_clustered() -> None:
    labels = np.repeat(np.arange(2), 6)
    arrays = {
        "clean_embedding": np.eye(12, dtype=np.float32),
        "support_standardized": np.zeros((12, 4), dtype=np.float32),
    }
    result = evaluate_fingerprints(arrays, labels=labels, external_k=None, seed=3)
    assert result.metrics["representations"]["support_standardized"]["status"] == "degenerate_representation"
    assert "support_standardized" not in result.predictions


def test_outer_evaluation_reports_primary_and_clm_proxy() -> None:
    rng = np.random.default_rng(23)
    labels = np.repeat(np.arange(2), 12)
    signal = labels[:, None] * 3.0 + rng.normal(0.0, 0.1, size=(24, 4))
    arrays = {
        "clean_embedding": signal.astype(np.float32),
        "cycle_repair_standardized": signal.astype(np.float32),
        "recovery_gain_standardized": signal.astype(np.float32),
    }
    result = evaluate_fingerprints(arrays, labels=labels, external_k=None, seed=5)
    assert result.metrics["primary_fingerprint"] == "cycle_repair_standardized"
    assert "cycle_repair_standardized" in result.predictions
    assert result.benchmark_validity["adjusted_ivma_implemented"] is False
    assert result.benchmark_validity["enters_training"] is False


def test_conditional_null_preserves_cluster_feature_marginals() -> None:
    worlds = generate_worlds(
        n_samples=60,
        n_features=20,
        n_clusters=3,
        latent_rank=4,
        zero_fraction=0.7,
        seed=29,
    )
    positive, labels = worlds["cluster_specific_dependency"]
    conditional_null, null_labels = worlds["conditional_dependency_destroyed"]
    assert np.array_equal(labels, null_labels)
    for cluster in np.unique(labels):
        rows = labels == cluster
        assert np.allclose(np.sort(positive[rows], axis=0), np.sort(conditional_null[rows], axis=0))


def test_matrix_only_loader_does_not_require_labels(tmp_path) -> None:
    path = tmp_path / "with_outer_fields.npz"
    matrix = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.savez_compressed(path, X=matrix, y=np.asarray([0, 0, 1, 1]))
    loaded = load_matrix_only(path)
    assert np.array_equal(loaded, matrix)


def test_m0_commands_keep_labels_out_of_fit_and_profile(tmp_path) -> None:
    class Args:
        device = "cpu"
        gpus = []
        config = tmp_path / "protocol.yaml"
        epochs = 2
        batch_size = 8
        feature_cap = 12
        mask_seed = 1701
        donor_seed = 2903
        corruption_mode = "donor_swap"
        fingerprint_masks = 4
        fingerprint_mask_ratio = 0.25

    jobs = build_jobs(tmp_path, (42, 123, 7), ("cluster_specific_dependency",))
    assert len(jobs) == 3
    commands = build_stage_commands(jobs[0], Args())
    assert "--labels" not in commands["fit"]
    assert "--labels" not in commands["profile"]
    assert "--external-k" not in commands["fit"]
    assert "--external-k" not in commands["profile"]
    assert commands["evaluate"][commands["evaluate"].index("--labels") + 1] == str(jobs[0].labels_path)


def test_m0_generated_panel_is_reused_without_rewrite(tmp_path) -> None:
    kwargs = {
        "seed": 7,
        "n_samples": 24,
        "n_features": 12,
        "n_clusters": 3,
        "latent_rank": 3,
        "zero_fraction": 0.5,
    }
    digest = DigestCache()
    assert _ensure_generated_panel(tmp_path, digest=digest, **kwargs) == "generated"
    manifest = json.loads((tmp_path / "manifest_seed7.json").read_text(encoding="utf-8"))
    matrix_path = Path(manifest["records"][0]["matrix_path"])
    modified_time = matrix_path.stat().st_mtime_ns
    assert _ensure_generated_panel(tmp_path, digest=digest, **kwargs) == "reused"
    assert matrix_path.stat().st_mtime_ns == modified_time


def test_m0_cuda_commands_use_the_assigned_physical_gpu(tmp_path) -> None:
    class Args:
        device = "cuda"
        config = tmp_path / "protocol.yaml"
        epochs = None
        batch_size = None
        feature_cap = None
        mask_seed = 1701
        donor_seed = 2903
        corruption_mode = "donor_swap"
        fingerprint_masks = None
        fingerprint_mask_ratio = None

    job = build_jobs(tmp_path, (42,), ("cluster_specific_dependency",))[0]
    commands = build_stage_commands(job, Args(), physical_gpu=4)
    for stage in ("fit", "profile"):
        assert commands[stage][commands[stage].index("--gpu") + 1] == "4"
        assert "--labels" not in commands[stage]


def test_successful_retry_retires_current_failure_marker(tmp_path) -> None:
    job = build_jobs(tmp_path, (42,), ("cluster_specific_dependency",))[0]
    job.run_root.mkdir(parents=True)
    (job.run_root / "fit.log").write_text("failed CUDA attempt\n", encoding="utf-8")
    failure = _record_incomplete_attempt(job, stage="fit", returncode=1)

    marker = job.run_root / "incomplete_compute.json"
    archived = Path(failure["attempt_dir"])
    assert marker.is_file()
    assert (archived / "incomplete_compute.json").is_file()
    assert (archived / "fit.log").read_text(encoding="utf-8") == "failed CUDA attempt\n"

    assert _retire_incomplete_marker(job) == archived
    assert not marker.exists()
    assert (archived / "incomplete_compute.json").is_file()


def test_successful_retry_archives_legacy_failure_marker(tmp_path) -> None:
    job = build_jobs(tmp_path, (42,), ("cluster_specific_dependency",))[0]
    job.run_root.mkdir(parents=True)
    marker = job.run_root / "incomplete_compute.json"
    marker.write_text(
        json.dumps({"status": "incomplete_compute", "failed_stage": "fit"}),
        encoding="utf-8",
    )

    archived = _retire_incomplete_marker(job)
    assert archived is not None
    assert archived.parent == job.run_root / "attempts"
    assert not marker.exists()
    assert (archived / marker.name).is_file()
