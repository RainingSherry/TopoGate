"""Independent synthetic CPU smoke for the bounded E2/E6 worker.

Run in the historical remote environment. No real data or GPU is used. The
full worker main executes with injected synthetic features so the output and
instrumentation path are exercised without claiming a formal experiment cell.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

import fit_worker as worker


def pinned_task(case_name: str, strategy: str, hop: str) -> dict:
    manifest = json.loads(Path(worker.__file__).with_name("REUSE_MANIFEST.json").read_text())
    case = manifest["cases"][case_name]
    reference = dict(case["expected_config_template"])
    config = dict(reference)
    config["hop_variant"] = hop
    return {"dataset": case_name, "seed": 7, "strategy": strategy,
            "config": config, "reference_config": reference,
            "feature_path": case["feature_path"],
            "feature_sha256": case["feature_sha256"],
            "input_kind": case["input_kind"],
            "feature_limit": case["feature_limit"],
            "preprocessing_hash": case["preprocessing_hash"],
            "n_clusters": case["n_clusters"],
            "K_provenance": case["K_provenance"],
            "core_hashes": manifest["core_source_sha256"]}


def test_pinned_validation(tmp: Path) -> None:
    def check(task: dict, expected: bool) -> None:
        p = tmp / "task-validate.json"
        p.write_text(json.dumps(task))
        try:
            worker.load_and_validate_task(p)
        except ValueError:
            if expected:
                raise
        else:
            if not expected:
                raise AssertionError("invalid frozen task accepted")

    t = pinned_task("student_grade", "path", "H2_B1")
    check(t, True)
    t_bad = json.loads(json.dumps(t)); t_bad["config"]["mix_amplitude_multiplier"] = 1.0
    check(t_bad, False)
    t_bad = json.loads(json.dumps(t)); t_bad["config"]["hop_variant"] = "H2_B2"
    check(t_bad, False)
    t_kwide = pinned_task("student_grade", "kwide", "H4_B1")
    check(t_kwide, True)
    print("PASS pinned task validation and drift rejection")


def test_full_main_synthetic(tmp: Path) -> None:
    from v0rg_tunable_multihop.config import V0_RGConfig
    from v0rg_tunable_multihop.tuning import SplitPreprocessor
    from run_factorial import TRACE_DTYPE

    rng = np.random.default_rng(7781)
    x_raw = np.vstack([rng.normal(loc=float(j), scale=.25, size=(16, 12))
                       for j in range(3)]).astype(np.float32)
    prep = SplitPreprocessor("general", 12).fit(x_raw)
    x_fit = prep.transform(x_raw)
    manifest = json.loads(Path(worker.__file__).with_name("REUSE_MANIFEST.json").read_text())
    original_task_loader = worker.load_and_validate_task
    original_feature_loader = worker.load_features_with_validation
    original_argv = sys.argv
    paths = {}
    try:
        for variant in ("H4_B1", "H8_B1"):
            for strategy in ("path", "kwide"):
                cfg = V0_RGConfig(hop_variant=variant, variant="rg_full", hidden_size=16,
                                  epochs=1, batch_size=12, neighbor_k=4,
                                  mix_neighbors=4, knn_pca_dim=3, kmeans_n_init=1,
                                  num_workers=0)
                d = asdict(cfg)
                task = {"dataset": "synthetic", "seed": 7, "strategy": strategy,
                        "config": d, "reference_config": d,
                        "feature_sha256": "synthetic-only", "preprocessing_hash": prep.fingerprint,
                        "n_clusters": 3, "K_provenance": "benchmark_oracle_from_y",
                        "core_hashes": manifest["core_source_sha256"]}
                worker.load_and_validate_task = lambda _path, task=task: task
                worker.load_features_with_validation = lambda _task: (prep, x_fit, x_raw)
                task_path = tmp / f"{variant}-{strategy}.task.json"
                task_path.write_text('{"synthetic_only":true}')
                out = tmp / f"{variant}-{strategy}"
                sys.argv = ["fit_worker.py", "--task", str(task_path), "--out", str(out),
                            "--device", "cpu"]
                worker.main()
                rec = json.loads((out / "record.json").read_text())
                assert rec["epochs_trained"] == 1
                assert set(rec["artifact_sha256"]) >= {
                    "model.pt", "embedding.npy", "predictions.npy", "history.json",
                    "plan.npz", "graph.npz", "preprocessor.npz", "source_trace.bin",
                    "source_trace.json", "displacement_by_anchor.npz"}
                trace = json.loads((out / "source_trace.json").read_text())
                assert trace["anchor_training_occurrences"] == len(x_fit)
                assert trace["draws"] == len(x_fit) * 4
                assert (out / "source_trace.bin").stat().st_size == trace["draws"] * TRACE_DTYPE.itemsize
                with np.load(out / "displacement_by_anchor.npz") as displacement:
                    assert np.all(displacement["count"] == 1)
                if strategy == "kwide":
                    assert "9" in trace["actual_hop_weight_fraction"]
                    assert "reference_plan.npz" in rec["artifact_sha256"]
                paths[(variant, strategy)] = out
                print("PASS synthetic full worker", variant, strategy)

            with np.load(paths[(variant, "path")] / "graph.npz") as a, \
                 np.load(paths[(variant, "kwide")] / "graph.npz") as b:
                for key in ("indices", "edge_weights", "node_gate"):
                    assert np.array_equal(a[key], b[key]), (variant, key)
            with np.load(paths[(variant, "kwide")] / "reference_plan.npz") as a, \
                 np.load(paths[(variant, "kwide")] / "plan.npz") as b:
                assert np.array_equal(a["offsets"], b["offsets"])
                assert np.array_equal(a["draw_count"], b["draw_count"])
            print("PASS synthetic original graph/gate and per-anchor count", variant)
    finally:
        worker.load_and_validate_task = original_task_loader
        worker.load_features_with_validation = original_feature_loader
        sys.argv = original_argv


def test_kwide_hop9_trace(tmp: Path) -> None:
    """Disconnected directed components force cosine-near, graph-unreachable donors."""
    from types import SimpleNamespace
    from v0rg_tunable_multihop.config import V0_RGConfig
    from v0rg_tunable_multihop.hop_sampling import (
        build_exact_hop_plan, estimate_neighbors_from_plan)
    from intervention_plan import _kwide
    from run_factorial import traced_plan_estimator

    n = 24
    indices = np.empty((n, 2), dtype=np.int64)
    for i in range(n):
        base = 0 if i < 12 else 12
        j = i - base
        indices[i] = [base + (j + 1) % 12, base + (j + 2) % 12]
    angles = 2*np.pi*np.arange(12)/12
    embedding = np.vstack([np.column_stack([np.cos(angles), np.sin(angles)]),
                           np.column_stack([np.cos(angles + .002), np.sin(angles + .002)])]).astype(np.float32)
    graph = SimpleNamespace(indices=indices, embedding=embedding)
    weights = np.full((n, 2), .5)
    config = V0_RGConfig(hop_variant="H4_B1", neighbor_k=2, mix_neighbors=2)
    reference = build_exact_hop_plan(graph, weights, "H4_B1", 2, 1.0)
    kwide, _ = _kwide(reference, graph, weights, config)
    assert any(9 in row for row in kwide.hop_labels)
    assert all(len(a) == len(b) for a, b in zip(reference.indices, kwide.indices))

    data = np.random.default_rng(82).normal(size=(n, 7)).astype(np.float32)
    batch = np.arange(n)
    orig_rng, trace_rng = np.random.default_rng(330), np.random.default_rng(330)
    original, original_usage = estimate_neighbors_from_plan(data, batch, kwide, orig_rng, "full")
    collector = worker.TraceCollector9(tmp / "hop9-trace.bin", n)
    collector.epoch = 1
    traced, traced_usage = traced_plan_estimator(collector)(data, batch, kwide, trace_rng, "full")
    info = collector.finish(tmp)
    assert np.array_equal(original, traced)
    assert original_usage["actual_sampling_mass"] == traced_usage["actual_sampling_mass"]
    assert orig_rng.bit_generator.state == trace_rng.bit_generator.state
    assert info["actual_hop_weight_fraction"]["9"] > 0
    assert np.isclose(sum(info["actual_hop_weight_fraction"].values()), 1.0)
    print("PASS H4 Kwide >8/unreachable code9 and traced-sampler identity")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="topogate-worker-review-") as name:
        work = Path(name)
        test_pinned_validation(work)
        test_full_main_synthetic(work)
        test_kwide_hop9_trace(work)
    print("ALL SYNTHETIC WORKER REVIEW CHECKS PASSED")
