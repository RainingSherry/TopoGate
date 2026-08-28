from __future__ import annotations

import argparse
import inspect

import numpy as np

from methods.TopoGate.V24_conditional_response.config import WORLD_NAMES
from methods.TopoGate.V24_conditional_response.contracts import (
    _block_dependency_separation,
    _chance_classifier_pass,
    _macro_ovr_auc,
    ContractAudit,
    audit_global_null_panel,
    audit_world,
)
from methods.TopoGate.V24_conditional_response.calibration import calibrate_estimator
from methods.TopoGate.V24_conditional_response.analyze import analyze_response
from methods.TopoGate.V24_conditional_response.controls import build_marginal_controls
from methods.TopoGate.V24_conditional_response.decision import decide_q1
from methods.TopoGate.V24_conditional_response.evaluation import (
    bootstrap_conditional_delta,
    conditional_pair_utility,
    crossfit_residual_response,
)
from methods.TopoGate.V24_conditional_response.postmortem import run_postmortem
from methods.TopoGate.V24_conditional_response.synthetic import generate_worlds
from scripts.V24.run_q1 import (
    DEFAULT_EXPLORATORY_OUTPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    _ensure_exploratory_request,
    _ensure_exploratory_root,
    _stage_reuse_plan,
    build_jobs,
    build_stage_commands,
    resolve_physical_gpus,
)

from .conftest import tiny_config


def test_w4_preserves_exact_support_and_nonzero_feature_marginals() -> None:
    config = tiny_config()
    matrix, labels = generate_worlds(config, seed=42)["W4_dependency_only"]
    audit = audit_world(matrix, labels, world="W4_dependency_only", config=config, seed=42, run_classifiers=False)
    assert audit.valid
    for feature in range(matrix.shape[1]):
        reference = np.sort(matrix[labels == 0, feature])
        for cluster in range(1, config.n_clusters):
            assert np.array_equal(reference, np.sort(matrix[labels == cluster, feature]))


def test_contract_linear_probe_is_compatible_with_current_sklearn() -> None:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(90, 5)).astype(np.float32)
    labels = np.repeat(np.arange(3, dtype=np.int64), 30)
    auc, probabilities = _macro_ovr_auc(features, labels, seed=19, folds=3)
    assert np.isfinite(auc)
    assert probabilities.shape == (90, 3)


def test_w0_uses_iid_support_rather_than_repeating_one_template_per_class() -> None:
    config = tiny_config()
    matrix, labels = generate_worlds(config, seed=42)["W0_global_null"]
    support = matrix > 0.0
    class_templates = [
        tuple(sorted(row.tobytes() for row in np.packbits(support[labels == cluster], axis=1)))
        for cluster in range(config.n_clusters)
    ]
    assert class_templates[0] != class_templates[1]


def test_per_seed_chance_guard_uses_detectability_ceiling_not_ci_zero_coverage() -> None:
    config = tiny_config()
    finite_null_fluctuation = {
        "support_macro_ovr_auc": 0.5157,
        "support_auc_ci_low": 0.5040,
        "support_auc_ci_high": 0.5272,
    }
    assert _chance_classifier_pass(finite_null_fluctuation, "support", config)
    finite_detectable_signal = dict(finite_null_fluctuation, support_macro_ovr_auc=0.5201)
    assert not _chance_classifier_pass(finite_detectable_signal, "support", config)


def test_global_null_panel_requires_all_primary_seeds_and_mean_centering() -> None:
    config = tiny_config()
    audits = {
        seed: ContractAudit(
            world="W0_global_null",
            valid=True,
            metrics={
                "support_macro_ovr_auc": 0.5 + offset,
                "marginal_macro_ovr_auc": 0.5 - offset,
            },
        )
        for seed, offset in zip(config.primary_seeds, (-0.004, 0.003, 0.001, -0.002, 0.002), strict=True)
    }
    panel = audit_global_null_panel(audits, config)
    assert panel["valid"]
    incomplete = dict(audits)
    incomplete.pop(config.primary_seeds[-1])
    assert not audit_global_null_panel(incomplete, config)["valid"]


def test_w4_dependency_is_stronger_than_mean_and_marginal_only_controls() -> None:
    config = tiny_config()
    worlds = generate_worlds(config, seed=123)
    w4_matrix, w4_labels = worlds["W4_dependency_only"]
    w1_matrix, w1_labels = worlds["W1_mean_only"]
    w3_matrix, w3_labels = worlds["W3_marginal_only"]
    w4 = _block_dependency_separation(w4_matrix, w4_labels, config)
    w1 = _block_dependency_separation(w1_matrix, w1_labels, config)
    w3 = _block_dependency_separation(w3_matrix, w3_labels, config)
    assert w4 >= config.dependency_separation_min
    assert w4 > max(w1, w3)


def test_control_world_contracts_have_their_intended_signal() -> None:
    config = tiny_config()
    worlds = generate_worlds(config, seed=42)
    w1 = audit_world(*worlds["W1_mean_only"], world="W1_mean_only", config=config, seed=42)
    w2 = audit_world(*worlds["W2_support_only"], world="W2_support_only", config=config, seed=42)
    w3 = audit_world(*worlds["W3_marginal_only"], world="W3_marginal_only", config=config, seed=42)
    assert w1.valid and w2.valid and w3.valid
    assert w1.metrics["nonzero_mean_max_standardized_difference"] >= config.mean_shift_min
    assert w2.metrics["support_macro_ovr_auc"] >= config.support_signal_auc_floor
    assert w3.metrics["marginal_dispersion_signal"] >= config.marginal_dispersion_min


def test_marginal_controls_have_fixed_shape_and_handle_empty_effective_positions() -> None:
    config = tiny_config()
    matrix, _ = generate_worlds(config, seed=7)["W4_dependency_only"]
    masks = np.zeros((3, config.n_features), dtype=bool)
    masks[1, :4] = True
    masks[2, 4:8] = True
    controls = build_marginal_controls(matrix, masks, np.asarray([1, 2, 3], dtype=np.int64))
    assert controls.support.shape == (config.n_samples, 3)
    assert controls.marginal.shape == (config.n_samples, 3, 9)
    assert np.count_nonzero(controls.support[:, 0]) == 0
    assert np.isfinite(controls.support).all()
    assert np.isfinite(controls.marginal).all()
    assert controls.diagnostics["support_semantics"] == "effective_changed_count"


def test_standardized_marginal_controls_are_bounded_for_nearly_constant_nonzero_features() -> None:
    matrix = np.asarray(
        [
            [0.00, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    controls = build_marginal_controls(
        matrix,
        np.asarray([[True, False, False, False]], dtype=bool),
        np.asarray([1], dtype=np.int64),
        standardized_clip=3.0,
        relative_scale_floor=0.01,
    )
    assert float(np.max(controls.marginal[:, :, 5:7])) <= 3.0 + 1e-6
    assert controls.diagnostics["marginal_standardized_clip_fraction_effective"] > 0.0


def test_residualizer_has_no_label_or_k_parameter() -> None:
    parameters = set(inspect.signature(crossfit_residual_response).parameters)
    assert not {"labels", "y", "K", "n_clusters"}.intersection(parameters)


def test_outer_pairs_stay_within_their_sample_disjoint_fold() -> None:
    rng = np.random.default_rng(31)
    labels = np.repeat(np.arange(3, dtype=np.int64), 30)
    state = rng.normal(size=(90, 6)).astype(np.float32)
    support = rng.poisson(2.0, size=(90, 3)).astype(np.float32)
    marginal = rng.normal(size=(90, 3, 9)).astype(np.float32)
    response = (state[:, :3] + 0.1 * marginal[:, :, 0] + rng.normal(scale=0.2, size=(90, 3))).astype(np.float32)
    result = conditional_pair_utility(
        state,
        support,
        marginal,
        response,
        labels=labels,
        outer_folds=3,
        inner_folds=2,
        seed=31,
        alpha=1.0,
        pair_count_per_fold=20,
    )
    pairs = result.records["pairs"]
    pair_folds = result.records["pair_fold"]
    for fold, test_rows in enumerate(result.fold_indices):
        current = pairs[pair_folds == fold]
        assert np.isin(current, test_rows).all()
        assert np.all(current[:, 0] != current[:, 1])
    bootstrap = bootstrap_conditional_delta(
        state,
        support,
        marginal,
        response,
        labels=labels,
        outer_folds=3,
        inner_folds=2,
        seed=31,
        alpha=1.0,
        pair_count_per_fold=20,
        replicates=3,
    )
    assert bootstrap.size > 0
    assert np.isfinite(bootstrap).all()


def test_bootstrap_worker_parallelism_preserves_replicate_values() -> None:
    config = tiny_config()
    matrix, labels = generate_worlds(config, seed=23)["W4_dependency_only"]
    rng = np.random.default_rng(23)
    state = rng.normal(size=(config.n_samples, 8)).astype(np.float32)
    support = rng.poisson(2.0, size=(config.n_samples, config.fingerprint_masks)).astype(np.float32)
    marginal = rng.normal(size=(config.n_samples, config.fingerprint_masks, 9)).astype(np.float32)
    response = rng.normal(size=(config.n_samples, config.fingerprint_masks)).astype(np.float32)
    serial = bootstrap_conditional_delta(
        state,
        support,
        marginal,
        response,
        labels=labels,
        outer_folds=config.outer_folds,
        inner_folds=config.inner_folds,
        seed=23,
        alpha=1.0,
        pair_count_per_fold=10,
        replicates=2,
        workers=1,
    )
    parallel = bootstrap_conditional_delta(
        state,
        support,
        marginal,
        response,
        labels=labels,
        outer_folds=config.outer_folds,
        inner_folds=config.inner_folds,
        seed=23,
        alpha=1.0,
        pair_count_per_fold=10,
        replicates=2,
        workers=2,
    )
    np.testing.assert_array_equal(serial, parallel)


def test_fit_and_profile_commands_keep_labels_out_of_training_stages(tmp_path) -> None:
    args = argparse.Namespace(
        device="cpu",
        epochs=None,
        batch_size=None,
        mask_seed=1701,
        donor_seed=2903,
    )
    job = build_jobs(
        tmp_path,
        seeds=(42,),
        worlds=("W4_dependency_only",),
        protocol_id=tiny_config().protocol_id,
    )[0]
    commands = build_stage_commands(job, args, bootstrap_replicates=3)
    assert "--labels" not in commands["fit"]
    assert "--labels" not in commands["profile"]
    assert "--n-clusters" not in commands["fit"]
    assert "--n-clusters" not in commands["profile"]
    assert commands["analyze"][commands["analyze"].index("--labels") + 1] == str(job.labels_path)


def test_job_key_is_bound_to_the_current_protocol_id(tmp_path) -> None:
    config = tiny_config()
    job = build_jobs(
        tmp_path,
        seeds=(42,),
        worlds=("W0_global_null",),
        protocol_id=config.protocol_id,
    )[0]
    assert config.protocol_id in job.key
    assert "q1_v1" not in job.key


def test_runner_rejects_forbidden_physical_gpus() -> None:
    assert resolve_physical_gpus("cuda", "1,2,6") == (1, 2, 6)
    for forbidden in ("0", "7", "1,7"):
        try:
            resolve_physical_gpus("cuda", forbidden)
        except ValueError as error:
            assert "0 and 7" in str(error)
        else:
            raise AssertionError(f"GPU setting {forbidden} should be rejected")


def test_exploratory_override_requires_isolated_root_and_exact_gpu_pool(tmp_path) -> None:
    config = tiny_config()
    args = argparse.Namespace(device="cuda", epochs=None, batch_size=None)
    _ensure_exploratory_request(
        args=args,
        config=config,
        seeds=config.primary_seeds,
        worlds=WORLD_NAMES,
        gpus=(1, 2, 3),
        bootstrap_replicates=config.bootstrap_replicates,
    )
    for bad_gpus in ((1, 2), (1, 2, 4)):
        try:
            _ensure_exploratory_request(
                args=args,
                config=config,
                seeds=config.primary_seeds,
                worlds=WORLD_NAMES,
                gpus=bad_gpus,
                bootstrap_replicates=config.bootstrap_replicates,
            )
        except ValueError as error:
            assert "1,2,3" in str(error)
        else:
            raise AssertionError("non-authorized exploratory GPU pool should be rejected")
    _ensure_exploratory_root(DEFAULT_EXPLORATORY_OUTPUT_ROOT)
    try:
        _ensure_exploratory_root(DEFAULT_OUTPUT_ROOT)
    except ValueError as error:
        assert "separate output root" in str(error)
    else:
        raise AssertionError("formal root must never be accepted by exploratory mode")


def test_exploratory_job_uses_formal_panel_as_input_but_isolated_run_root(tmp_path) -> None:
    config = tiny_config()
    jobs = build_jobs(
        tmp_path / "exploratory",
        seeds=(42,),
        worlds=("W0_global_null",),
        protocol_id=config.protocol_id,
        data_root=tmp_path / "formal",
    )
    assert jobs[0].matrix_path == tmp_path / "formal" / "generated_data" / "W0_global_null" / "seed42" / "matrix_only.npz"
    assert jobs[0].run_root == tmp_path / "exploratory" / "runs" / "W0_global_null" / "seed42"
    assert jobs[0].matrix_path != jobs[0].run_root


def test_stage_reuse_respects_fit_profile_analysis_dependencies() -> None:
    assert _stage_reuse_plan({"fit": True, "profile": True, "analyze": True}) == {
        "fit": True,
        "profile": True,
        "analyze": True,
    }
    assert _stage_reuse_plan({"fit": False, "profile": True, "analyze": True}) == {
        "fit": False,
        "profile": False,
        "analyze": False,
    }
    assert _stage_reuse_plan({"fit": True, "profile": False, "analyze": True}) == {
        "fit": True,
        "profile": False,
        "analyze": False,
    }


def test_p0_is_read_only_and_never_imports_a_training_launcher() -> None:
    source = inspect.getsource(run_postmortem)
    assert "subprocess" not in source
    assert "V23_cycle_response.fit" not in source
    assert "V23_cycle_response.profile" not in source


def test_calibration_reports_the_same_null_centering_gate_as_q1() -> None:
    result = calibrate_estimator(tiny_config(), replicates=2, workers=2)
    assert result.null_deltas.size == 2
    assert result.alternative_deltas.size == 2
    assert result.summary["workers"] == 2
    assert "null_centered_for_null_world_gate" in result.summary
    assert "null_std_delta_auc" in result.summary


def test_analysis_does_not_silently_substitute_control_support_for_missing_profile_support() -> None:
    source = inspect.getsource(analyze_response)
    assert 'fingerprints.get("support_raw"' not in source
    assert "unavailable_without_a_V23_support_raw_fingerprint" in source


def test_q1_decision_requires_calibration_and_postmortem_after_p1_passes() -> None:
    config = tiny_config()
    records = []
    for world in WORLD_NAMES:
        delta = 0.0 if world.startswith(("W0", "W1", "W2", "W3")) else (0.03 if world == "W4_dependency_only" else 0.02)
        lower = -0.01 if delta == 0.0 else 0.005
        upper = 0.01 if delta == 0.0 else 0.05
        for seed in config.primary_seeds:
            records.append(
                {
                    "world": world,
                    "seed": seed,
                    "delta_auc": delta,
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "contract_valid": True,
                }
            )
    calibration = {"calibration_passes": True}
    postmortem = {"status": "completed"}
    accepted = decide_q1(records, config, calibration=calibration, postmortem=postmortem)
    assert accepted["promotion_to_q2"] is True
    blocked = decide_q1(records, config, calibration=None, postmortem=postmortem)
    assert blocked["decision"] == "calibration_incomplete_or_failed"
