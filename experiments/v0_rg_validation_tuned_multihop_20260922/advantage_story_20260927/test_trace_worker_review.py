"""Tiny hand-computable actual-trace fixture; no model fit or real dataset."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import trace_worker as tw


def fixture(base: Path):
    factorial = base / "source_coverage_20260925/outputs/factorial/Baron"
    factorial.mkdir(parents=True)
    labels = np.array([0, 0, 0, 1], dtype=np.int32)
    comm = np.array([0, 0, 1, 2], dtype=np.int32)
    np.save(factorial / "labels_true.npy", labels)
    np.save(factorial / "community_ids.npy", comm)
    graph = np.array([[1], [0], [0], [0]], dtype=np.int32)
    # a0: 1/4 same-class same-community familiar, 1/4 same-class
    # cross-community new, 1/2 different-class new.
    # a1: 1/2 familiar same/same, 1/2 same-class cross/new.
    # a2: 1/2 cross/familiar (community 0 already in own graph), 1/2 diff/new.
    # a3: all different-class familiar.
    events = [(0, 1, 1, 1), (0, 2, 9, 1), (0, 3, 2, 2),
              (1, 0, 1, 2), (1, 2, 2, 2), (2, 0, 2, 1),
              (2, 3, 2, 1), (3, 0, 1, 1)]
    block = np.zeros(len(events), dtype=tw.TRACE_DTYPE)
    for i, (a, s, hop, weight) in enumerate(events):
        block[i] = (1, a, s, hop, weight, 0.25)
    train = base / "source_coverage_20260925/outputs/train/Baron/flat_anchor"
    for seed in (11, 13, 17, 19, 23):
        folder = train / f"seed_{seed}"
        folder.mkdir(parents=True)
        np.savez(folder / "graph.npz", indices=graph)
        block.tofile(folder / "source_trace.bin")
        record = {"status": "completed", "dataset": "Baron", "arm": "flat_anchor",
                  "seed": seed, "n_samples": 4,
                  "labels_used_during_fit": False, "test_labels_loaded": False,
                  "K_provenance": "benchmark_oracle_from_y",
                  "training_trace": {"trace_complete": True, "draws": len(block),
                                     "trace_dtype": [list(x) for x in tw.TRACE_DTYPE.descr]},
                  "artifact_sha256": {"graph.npz": tw.sha256(folder / "graph.npz"),
                                      "source_trace.bin": tw.sha256(folder / "source_trace.bin")}}
        (folder / "record.json").write_text(json.dumps(record))
    return train, labels, comm, block


def test_hand_computable(tmp: Path):
    train, labels, comm, block = fixture(tmp)
    one = tw.process_seed(11, labels, comm, train, chunk_size=2)
    expected = np.array([
        [0.25, 0, 0, 0.25, 0, 0.5],
        [0.5, 0, 0, 0.5, 0, 0],
        [0, 0, 0.5, 0, 0, 0.5],
        [0, 0, 0, 0, 1, 0],
    ])
    assert np.allclose(one["normalized"], expected)
    all_results = [tw.process_seed(s, labels, comm, train, chunk_size=2)
                   for s in (11, 13, 17, 19, 23)]
    agg = tw.aggregate(all_results, labels)
    assert np.allclose(agg["seed_mean"], expected.mean(axis=0))
    assert not np.allclose(agg["macro_seed"].mean(axis=0), agg["seed_mean"])
    out = tmp / "out"
    record = tw.run(out, tmp, seeds=(11, 13, 17, 19, 23), chunk_size=2)
    assert record["status"] == "completed" and record["six_category_mass_sum"] == 1.0
    assert np.isclose(record["total_new_community_mass"], 0.4375)
    assert np.isclose(record["same_class_new_community_mass"], 0.1875)
    assert set(record["artifact_sha256"]) == {p.name for p in out.iterdir()} - {"record.json"}
    assert all(tw.sha256(out / name) == value for name, value in record["artifact_sha256"].items())
    bad = block[:1].copy(); bad["source"] = 4
    try:
        tw.accumulate_block(bad, labels, comm,
                            tw.near_communities(np.array([[1], [0], [0], [0]]), comm),
                            np.zeros((4, 6)), np.zeros(4))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid source index silently accepted")
    print("PASS chunked exact six-way mass, hop9, seed-specific graph, class macro, artifacts, invalid-index hard error")


def test_archived_regression():
    out = Path(__file__).resolve().parent / "outputs/E5_trace"
    ref = tw.SOURCE / "outputs/factorial/Baron/source_coverage.csv"
    if not (out / "record.json").exists():
        return
    record = json.loads((out / "record.json").read_text())
    actual = pd.read_csv(out / "per_seed.csv")
    reference = pd.read_csv(ref)
    maximum = 0.0
    for row in reference.itertuples():
        one = actual[actual.seed == row.seed].set_index("category")["anchor_mean_mass"]
        same = sum(one[f"{role}_{novelty}"] for role in tw.ROLE_NAMES[:2]
                   for novelty in tw.NOVELTY_NAMES)
        new = sum(one[f"{role}_new"] for role in tw.ROLE_NAMES)
        same_new = sum(one[f"{role}_new"] for role in tw.ROLE_NAMES[:2])
        maximum = max(maximum, abs(same-row.same_mass),
                      abs(new-row.new_community_mass),
                      abs(same_new-row.same_new_community_mass))
    assert maximum < 2e-12, maximum
    assert all(tw.sha256(out / name) == digest
               for name, digest in record["artifact_sha256"].items())
    print(f"PASS archived five-seed source-coverage regression, max absolute deviation {maximum:.3g}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="topogate-trace-review-") as dirname:
        test_hand_computable(Path(dirname))
    test_archived_regression()
