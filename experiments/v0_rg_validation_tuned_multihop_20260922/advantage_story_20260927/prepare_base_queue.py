"""Freeze the 21 new fits and 59 historical posthoc cohorts; never approve/run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LOCK = HERE.parent / "source_coverage_20260925" / "outputs/p0/selection_lock_v2.json"
PYTHON = "/data/luolie/conda/envs/scssl_bench_py310/bin/python"
CASES = ("Quake_Smart-seq2_Lung", "student_grade", "Baron")
FIT_ARTIFACTS = [
    "model.pt", "embedding.npy", "predictions.npy", "history.json",
    "plan.npz", "graph.npz", "preprocessor.npz", "source_trace.bin",
    "source_trace.json", "displacement_by_anchor.npz",
]
COHORT_ARTIFACTS = [
    "frozen_readouts_10.npz", "fixed_input_layout.npz",
    "e3_per_readout.csv", "e3_per_seed.csv", "e3_aggregate.csv",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_config(info: dict) -> bool:
    """Mirror cohort_worker.matched_config for queue eligibility."""
    mh = info.get("selected_multihop")
    h1 = info.get("h1_counterpart_of_selected_multihop")
    if not isinstance(mh, dict) or not isinstance(h1, dict):
        return False
    if mh.get("preprocessing_hash") != h1.get("preprocessing_hash"):
        return False
    diffs = {k for k in set(mh["config"]) | set(h1["config"])
             if mh["config"].get(k) != h1["config"].get(k)}
    if diffs - {"hop_variant", "high_hop_probability_multiplier"}:
        return False
    if h1["config"].get("hop_variant") != "H1":
        return False
    return ({int(s) for s in mh["per_seed"]} == {7, 42, 123}
            and {int(s) for s in h1["per_seed"]} == {7, 42, 123})


def build():
    draft = read_json(HERE / "DRAFT_TRAINING_QUEUE.json")
    manifest = read_json(HERE / "REUSE_MANIFEST.json")
    if sha256(LOCK) != (HERE / "COHORT_LOCK_SHA256.txt").read_text(encoding="utf-8").strip():
        raise RuntimeError("Historical selection lock differs from frozen hash")
    lock = read_json(LOCK)
    names = sorted(n for n, info in lock["datasets"].items() if matched_config(info))
    if len(names) != 59:
        raise RuntimeError(f"Expected 59 matched cohorts, found {len(names)}")
    cells = draft["new_training_cells"]
    if len(cells) != 21 or manifest["summary"] != {
        "depth_cells": 36, "depth_reuse": 24, "depth_missing": 12,
        "kwide_cells": 9, "kwide_reuse": 0, "kwide_missing_exact": 9,
    }:
        raise RuntimeError("New fit/reuse count differs from frozen protocol")
    expected_depth = {(c["dataset"], c["depth"], int(c["seed"]))
                      for c in manifest["depth_cells"] if c["status"] != "reuse"}
    expected_kwide = {(c["dataset"], int(c["seed"]))
                      for c in manifest["kwide_cells"] if c["status"] != "reuse"}
    actual_depth = set()
    actual_kwide = set()
    by_case = {case: [] for case in CASES}
    for cell in cells:
        task_path = HERE / cell["task_file"]
        task = read_json(task_path)
        if (task["id"], task["dataset"], int(task["seed"]), task["strategy"]) != (
                cell["id"], cell["dataset"], int(cell["seed"]), cell["strategy"]):
            raise RuntimeError(f"Draft/task identity mismatch: {cell['id']}")
        if task["core_hashes"] != manifest["core_source_sha256"]:
            raise RuntimeError(f"Core hash pin differs: {cell['id']}")
        if task["strategy"] == "path":
            actual_depth.add((task["dataset"], task["config"]["hop_variant"].split("_")[0], int(task["seed"])))
        else:
            actual_kwide.add((task["dataset"], int(task["seed"])))
        required = FIT_ARTIFACTS + (["reference_plan.npz"] if task["strategy"] == "kwide" else [])
        by_case[task["dataset"]].append({
            "id": cell["id"], "kind": "gpu",
            "command": ["fit_worker.py", "--task", cell["task_file"],
                        "--out", f"outputs/train/{cell['id']}", "--device", "cuda:0"],
            "out": f"outputs/train/{cell['id']}",
            "required_artifacts": required,
        })
    if actual_depth != expected_depth or actual_kwide != expected_kwide:
        raise RuntimeError("Fit cells differ from missing manifest cells")
    for case in CASES:
        by_case[case].sort(key=lambda t: (
            {"H2": 0, "H4": 1, "H8": 1, "Kwide": 2}[t["id"].split("__")[1]],
            int(t["id"].split("seed")[-1])))
    fits = []
    while any(by_case.values()):
        for case in CASES:
            if by_case[case]:
                fits.append(by_case[case].pop(0))
    cohorts = [{
        "id": f"cohort__{name}", "kind": "cpu",
        "command": ["cohort_worker.py", "--dataset", name,
                    "--out", f"outputs/cohort/{name}"],
        "out": f"outputs/cohort/{name}",
        "required_artifacts": COHORT_ARTIFACTS,
    } for name in names]
    queue = {"schema": "advantage_story_base_queue_v1", "tasks": fits + cohorts}
    resources = {
        "gpus": [2, 4], "initial_workers_per_gpu": 4,
        "max_workers_per_gpu": 16, "start_free_mb": 6000,
        "reserve_mb_per_task": 4000, "cpu_workers": 4,
        "cpu_threads_per_task": 2, "min_host_available_mb": 64000,
        "reserve_host_mb_per_task": 1024, "poll_seconds": 15,
        "python": PYTHON,
    }
    return queue, resources


if __name__ == "__main__":
    queue, resources = build()
    if sys.argv[1:] == ["--check"]:
        if read_json(HERE / "queue.json") != queue or read_json(HERE / "resources.json") != resources:
            raise RuntimeError("Frozen base queue or resources differ from reproducible plan")
        import controller
        controller.validate_queue(queue)
        controller.load_resources()
    elif len(sys.argv) == 1:
        (HERE / "queue.json").write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (HERE / "resources.json").write_text(json.dumps(resources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        raise SystemExit("usage: prepare_base_queue.py [--check]")
    print(f"Frozen base queue: {sum(t['kind']=='gpu' for t in queue['tasks'])} fits, "
          f"{sum(t['kind']=='cpu' for t in queue['tasks'])} cohorts; no approval or run")
