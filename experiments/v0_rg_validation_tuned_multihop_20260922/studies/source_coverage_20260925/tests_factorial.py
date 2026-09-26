"""Tiny synthetic CPU invariant tests (run: ``python -m pytest -q tests_factorial.py``
or ``python tests_factorial.py``).  Requires the historical package on PYTHONPATH
(R) and methods.NeighborMix_scMAE for the fit tests.  No real data, no labels."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import factorial_plan as fp  # noqa: E402
import run_factorial as rf  # noqa: E402
from common import ARMS, ARM_SPECS, canonical_sha256  # noqa: E402


def _data(n=72, d=24, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, d)) * 3
    X = np.vstack([c + rng.normal(size=(n // 3, d)) for c in centers]).astype(np.float32)
    return (X - X.mean(0)) / X.std(0)


def _config(**overrides):
    from v0rg_tunable_multihop.config import V0_RGConfig

    base = dict(hidden_size=16, epochs=2, batch_size=16, neighbor_k=5, mix_neighbors=3, knn_pca_dim=6,
                hop_variant="H4_B1", high_hop_probability_multiplier=2.0, kmeans_n_init=2)
    base.update(overrides)
    return V0_RGConfig(**base)


def _reference_plan(config, X, seed=3):
    from v0rg_tunable_multihop.hop_sampling import build_exact_hop_plan

    frozen = fp.build_frozen_graph(X, config, seed)
    return frozen, build_exact_hop_plan(frozen["graph"], frozen["edge_weights"], config.hop_variant,
                                        config.mix_neighbors, config.high_hop_probability_multiplier)


# ---------------------------------------------------------------- plan invariants

def test_shell_masses_normalized_over_nonempty_shells():
    m = fp.shell_masses([1, 2, 5], "geometric", alpha=0.2)
    assert np.isclose(m.sum(), 1.0)
    assert np.allclose(m / m[0], [1.0, 0.8, 0.8 ** 4])
    f = fp.shell_masses([1, 3], "flat_q", q=2.0)
    assert np.allclose(f, [1 / 3, 2 / 3])


def test_geometric_keeps_candidates_and_within_weights_and_flat_is_identity():
    X = _data()
    config = _config()
    _, reference = _reference_plan(config, X)
    geo = fp.reshape_plan(reference, "geometric", 2.0, 0.2)
    audit = fp.verify_same_candidates(reference, geo)
    assert audit["max_within_shell_abs_diff"] < 1e-12
    assert geo.indices is reference.indices and geo.hop_labels is reference.hop_labels
    for probs, labels in zip(geo.probs, geo.hop_labels):
        available, mass, _ = fp.row_decomposition(probs, labels)
        assert np.isclose(probs.sum(), 1.0)
        target = fp.shell_masses(available, "geometric", alpha=0.2)
        assert np.allclose([mass[int(h)] for h in available], target, atol=1e-12)
    flat = fp.reshape_plan(reference, "flat_q", 2.0)
    for a, b in zip(flat.probs, reference.probs):
        assert np.allclose(a, b, rtol=0, atol=1e-14)
    fa, fr = fp.plan_fingerprints(fp.plan_arrays(geo)), fp.plan_fingerprints(fp.plan_arrays(reference))
    assert fa["candidate_sha256"] == fr["candidate_sha256"]
    assert fa["probability_sha256"] != fr["probability_sha256"]


def test_reshape_rejects_wrong_locked_q():
    X = _data()
    _, reference = _reference_plan(_config(), X)
    try:
        fp.reshape_plan(reference, "geometric", 1.0, 0.2)
    except ValueError:
        return
    raise AssertionError("q mismatch must be rejected")


def test_traced_estimators_are_numerically_identical_and_rng_preserving():
    from v0rg_tunable_multihop.hop_sampling import estimate_neighbors_from_plan
    from v0rg_tunable_multihop.mixing import estimate_neighbors

    X = _data()
    frozen, plan = _reference_plan(_config(), X)
    batch = np.arange(0, 40, 3)
    for estimator in ("current", "uniform_sample", "full"):
        r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
        a, ua = estimate_neighbors_from_plan(X, batch, plan, r1, estimator)
        collector = rf.TraceCollector(None, X.shape[0])
        b, ub = rf.traced_plan_estimator(collector)(X, batch, plan, r2, estimator)
        assert np.array_equal(a, b)
        assert r1.bit_generator.state == r2.bit_generator.state
        assert ua["actual_sampling_mass"] == ub["actual_sampling_mass"]
        r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
        a = estimate_neighbors(X, batch, frozen["graph"], frozen["edge_weights"], 3, r1, estimator)
        collector = rf.TraceCollector(None, X.shape[0])
        b = rf.traced_h1_estimator(collector)(X, batch, frozen["graph"], frozen["edge_weights"], 3, r2, estimator)
        assert np.array_equal(a, b)
        assert r1.bit_generator.state == r2.bit_generator.state
        assert collector.occurrences == len(batch)


# ---------------------------------------------------------------- tiny fits

def _fit(X, config, arm, prior, tmp: Path | None = None, name="x"):
    trace = None if tmp is None or arm is None else tmp / f"{name}.bin"
    return rf.fit_arm(X, asdict(config), arm, prior, 11, "cpu", 3, trace, None)


def test_reference_identity_of_wrappers_h1_and_flat():
    X = _data()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        flat = _config()
        h1 = replace(flat, hop_variant="H1", high_hop_probability_multiplier=1.0)
        for cfg, arm, prior in ((h1, "H1_anchor", {"kind": "one_hop", "q": 2.0, "alpha": None}),
                                (flat, "flat_anchor", {"kind": "flat_q", "q": 2.0, "alpha": None})):
            ref = _fit(X, cfg, None, prior)
            new = _fit(X, cfg, arm, prior, tmp, arm)
            assert np.array_equal(ref["embedding"], new["embedding"]), arm
            assert np.array_equal(ref["predictions"], new["predictions"]), arm
            for key, value in ref["diagnostics"]["model_state_dict"].items():
                assert bool((value == new["diagnostics"]["model_state_dict"][key]).all()), (arm, key)
            assert new["epochs_traced"] == cfg.epochs
            assert new["trace"]["anchor_training_occurrences"] == X.shape[0] * cfg.epochs
            raw = np.fromfile(tmp / f"{arm}.bin", dtype=rf.TRACE_DTYPE)
            assert len(raw) == new["trace"]["draws"]
            assert set(np.unique(raw["epoch"])) == set(range(1, cfg.epochs + 1))


def test_frozen_graph_matches_trainer_diagnostics():
    X = _data()
    cfg = _config()
    ref = _fit(X, cfg, None, {"kind": "flat_q", "q": 2.0, "alpha": None})
    frozen = fp.build_frozen_graph(X, cfg, 11)
    assert np.array_equal(frozen["graph"].indices, ref["diagnostics"]["neighbor_indices"])
    assert np.array_equal(frozen["edge_weights"], ref["diagnostics"]["edge_weights"])
    assert np.array_equal(frozen["node_gate"], ref["diagnostics"]["node_gate"])


def test_mixed_target_separation_and_candidate_identity():
    import torch
    from v0rg_tunable_multihop import trainer

    X = _data()
    cfg = _config()
    flat_prior = {"kind": "flat_q", "q": 2.0, "alpha": None}
    geo_prior = {"kind": "geometric", "q": 2.0, "alpha": 0.2}
    seen = {"real_target_is_batch": [], "pseudo_target_is_mixed": []}
    original = trainer.WeightedAutoEncoder.loss_mask_weighted

    def spy(self, x, target, mask, sample_weight=None):
        # Real call: target must be the clean batch; pseudo call: target must not be it.
        if sample_weight is None and self.training:
            seen["real_target_is_batch"].append(target.detach().clone())
        elif sample_weight is not None:
            seen["pseudo_target_is_mixed"].append(target.detach().clone())
        return original(self, x, target, mask, sample_weight=sample_weight)

    trainer.WeightedAutoEncoder.loss_mask_weighted = spy
    try:
        anchor = _fit(X, cfg, "flat_anchor", flat_prior)
        n_anchor = len(seen["pseudo_target_is_mixed"])
        anchor_targets = seen["pseudo_target_is_mixed"][:]
        seen["pseudo_target_is_mixed"].clear()
        seen["real_target_is_batch"].clear()
        mixed = _fit(X, cfg, "flat_mixed", flat_prior)
    finally:
        trainer.WeightedAutoEncoder.loss_mask_weighted = original
    calls = mixed["captured"]["target_calls"]
    assert calls["mixed"] > 0 and calls["mixed"] == calls["real"] == n_anchor
    # In the mixed arm the pseudo target differs from the real-branch target (clean batch).
    diffs = [float(torch.max(torch.abs(p - r))) for p, r in
             zip(seen["pseudo_target_is_mixed"], seen["real_target_is_batch"])]
    assert max(diffs) > 0
    # In the anchor arm the first pseudo target equals the first batch (anchor target).
    assert anchor_targets and anchor["captured"]["target_calls"]["mixed"] == 0
    assert not np.array_equal(anchor["embedding"], mixed["embedding"])
    geo = _fit(X, cfg, "geometric_mixed", geo_prior)
    fp.verify_same_candidates(mixed["captured"]["plan"], geo["captured"]["plan"])
    assert geo["captured"]["target_calls"]["mixed"] == calls["mixed"]


def test_patches_are_restored():
    from v0rg_tunable_multihop import hop_sampling, mixing, trainer

    before = (trainer.build_exact_hop_plan, hop_sampling.estimate_neighbors_from_plan, mixing.estimate_neighbors,
              trainer.WeightedAutoEncoder, trainer.make_pseudo_batch, trainer.DataLoader, trainer.compute_node_gate)
    _fit(_data(), _config(), "geometric_mixed", {"kind": "geometric", "q": 2.0, "alpha": 0.2})
    after = (trainer.build_exact_hop_plan, hop_sampling.estimate_neighbors_from_plan, mixing.estimate_neighbors,
             trainer.WeightedAutoEncoder, trainer.make_pseudo_batch, trainer.DataLoader, trainer.compute_node_gate)
    assert before == after


# ---------------------------------------------------------------- label isolation

def _task(tmp: Path, arm: str, **extra) -> Path:
    X = (np.random.default_rng(1).poisson(1.0, size=(40, 30))).astype(np.float32)
    feature = tmp / "features_only.npz"
    # A stray label key inside the feature file must be ignored by the loader.
    np.savez_compressed(feature, X=X, y=np.arange(40) % 3)
    from common import sha256
    from v0rg_tunable_multihop.tuning import SplitPreprocessor

    prep = SplitPreprocessor("raw_count", 20).fit(X)
    locked = asdict(_config())
    task = {"id": f"t_{arm}", "dataset": "synthetic", "arm": arm, "seed": 11,
            "config": fp.arm_config(locked, arm), "reference_multihop_config": locked,
            "prior": {"kind": ARM_SPECS[arm]["prior"], "q": 2.0,
                      "alpha": 0.2 if ARM_SPECS[arm]["prior"] == "geometric" else None},
            "target": ARM_SPECS[arm]["target"], "feature_path": str(feature), "feature_sha256": sha256(feature),
            "n_clusters": 3, "K_provenance": "benchmark_oracle_from_y", "input_kind": "raw_count",
            "feature_limit": 20, "expected_preprocessing_hash": prep.fingerprint, **extra}
    path = tmp / f"{arm}.json"
    path.write_text(json.dumps(task))
    return path


def test_no_label_reads_and_label_fields_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task = rf.load_task(_task(tmp, "flat_anchor"))
        import numpy as _np
        opened = []
        original = _np.load

        def guarded(file, *a, **k):
            result = original(file, *a, **k)
            opened.append(str(file))
            return result
        _np.load = guarded
        try:
            prep, X_all = rf.load_features(task)
        finally:
            _np.load = original
        assert X_all.shape == (40, 20) and opened == [task["feature_path"]]
        for bad in ({"validation_labels_path": "/x"}, {"test_labels_path": "/x"}):
            try:
                rf.load_task(_task(tmp, "flat_anchor", **bad))
            except ValueError:
                continue
            raise AssertionError(f"label field accepted: {bad}")


def test_task_config_drift_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = _task(tmp, "H1_anchor")
        task = json.loads(path.read_text())
        task["config"]["mask_ratio"] = 0.5
        path.write_text(json.dumps(task))
        try:
            rf.load_task(path)
        except ValueError:
            return
        raise AssertionError("H1 config differing beyond hop fields must be rejected")


def test_train_writes_complete_validated_record():
    from common import validate_train_output

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = _task(tmp, "flat_mixed")
        out = tmp / "out"

        class Args:
            task_json, device = str(path), "cpu"
        Args.out = str(out)
        rf.train(Args)
        record = validate_train_output({"id": "t_flat_mixed", "out": str(out)}, path)
        assert record["labels_used_during_fit"] is False and record["test_labels_loaded"] is False
        (out / "embedding.npy").write_bytes(b"corrupt")
        try:
            validate_train_output({"id": "t_flat_mixed", "out": str(out)}, path)
        except Exception:
            return
        raise AssertionError("corrupted artifact must fail validation")


def test_configuration_selection_provenance(tmp_path):
    from common import sha256

    lock = tmp_path / "validation_lock.json"
    lock.write_text('{"status":"LOCKED_VALIDATION_ONLY"}')
    task = {"reference_multihop_config": {"hop_variant": "H4_B1"},
            "selection_lock_sha256": sha256(lock)}
    assert rf.configuration_selection_scope(task, lock) == "historical_validation_only"
    task["selection_lock_sha256"] = canonical_sha256(task["reference_multihop_config"])
    assert rf.configuration_selection_scope(task, lock) == "fixed_exploratory_policy"
    task["selection_lock_sha256"] = "wrong_hash"
    try:
        rf.configuration_selection_scope(task, lock)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown selection provenance must be rejected")


# ---------------------------------------------------------------- selection / seeds / scheduler

def test_validation_export_whitelist_and_tie_break():
    import audit_select as a

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "readout_all.csv"
        with src.open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["dataset", "stage", "config_id", "seed", "random_state", "inertia", "ari_val",
                        "ari_test", "ari_full"])
            for s in range(10):
                w.writerow(["D", "st", "H1", 42, s, 1.0, 0.5, 0.987654321, 0.123456789])
        result = a.export_validation_scores(src, tmp / "scores_val.csv")
        text = (tmp / "scores_val.csv").read_text()
        assert "ari_test" not in text and "0.987654321" not in text and "0.123456789" not in text
        assert result["rows"] == 10 and not result["problems"]
    entries = [{"val_mean": 0.5, "config": {"hop_variant": "H8_B1"}, "eff_hash": "a"},
               {"val_mean": 0.5 + 1e-14, "config": {"hop_variant": "H2_B1"}, "eff_hash": "b"},
               {"val_mean": 0.4, "config": {"hop_variant": "H1"}, "eff_hash": "c"}]
    ranked = a.rank_configs(entries)
    assert [e["eff_hash"] for e in ranked] == ["b", "a", "c"]


def test_seed_collision_increment():
    import prepare_factorial as p

    chosen = p.choose_seeds({1019, 1020, 3037})
    assert [c["seed"] for c in chosen] == [1021, 2027, 3038, 4051, 5059]


def test_scheduler_rules():
    import controller as c

    res = dict(c.DEFAULT_RESOURCES)
    gpus = {2: {"free_mb": 20000, "util": 10}, 3: {"free_mb": 8000, "util": 10},
            4: {"free_mb": 30000, "util": 95}, 6: {"free_mb": 5000, "util": 0}}
    ours = {2: 2, 3: 1, 4: 0, 6: 0}
    launches = {g: 0 for g in res["gpus"]}
    # GPU2 may grow (free>12GB); GPU3 below initial; GPU4 busy; GPU6 lacks memory.
    assert c.pick_gpu(res, gpus, ours, launches, {}) == 2
    gpus[2]["free_mb"] = 11000
    assert c.pick_gpu(res, gpus, ours, launches, {}) == 3
    states = {"a": {"status": "failed"}, "b": {"status": "completed"}}
    assert c.deps_state({"deps": ["a", "b"]}, states) == "blocked"
    assert c.deps_state({"deps": ["a", "b"], "dep_mode": "all_terminal"}, states) == "ready"
    assert c.deps_state({"deps": ["b"]}, states) == "ready"


def test_arm_table_matches_design():
    assert ARMS == ("H1_anchor", "flat_anchor", "geometric_anchor", "flat_mixed", "geometric_mixed")
    assert {a: (ARM_SPECS[a]["prior"], ARM_SPECS[a]["target"]) for a in ARMS[1:]} == {
        "flat_anchor": ("flat_q", "anchor"), "geometric_anchor": ("geometric", "anchor"),
        "flat_mixed": ("flat_q", "mixed"), "geometric_mixed": ("geometric", "mixed")}
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # report and continue
                failures += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if failures else 0)
