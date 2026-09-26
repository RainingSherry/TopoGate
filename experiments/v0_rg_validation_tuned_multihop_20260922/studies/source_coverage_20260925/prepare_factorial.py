"""P1 preparation: choose cases from the validation-only lock, freeze five new
seeds, write preregistration, per-task JSONs, config locks, source hashes and
the controller queue.  Nothing here trains, infers or reads labels.

Case rule (preregistered, validation only):
  * Baron is fixed (primary mechanism case).
  * The second case is Campbell or worm_neuron_cell, chosen by
      1. mean paired validation delta (selected multihop - its same-config H1
         counterpart) over the three historical seeds; if the counterpart is
         incomplete, the dataset's validation-best H1 is used and flagged;
      2. tie (|diff| <= 1e-12): more seeds with a positive delta;
      3. tie: smaller seed-SD of the delta;  4. tie: alphabetical.
Seed rule: candidates 1019, 2027, 3037, 4051, 5059; any collision with the
historical seed inventory (records, seed_* dirs, task/queue/manifest JSON seeds)
or an earlier choice increments that candidate by 1 until free.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from common import (ARM_SPECS, ARMS, CODE_FILES, HISTORICAL_TRAIN_SEEDS, R, atomic_json,  # noqa: E402
                    canonical_sha256, code_hashes, core_source_hashes, json_safe, now, read_json,
                    resolve_path_variant, sha256, verify_sidecar_hash, write_once_json)
import factorial_plan as fp  # noqa: E402

PRIMARY = "Baron"
SECOND_CANDIDATES = ("Campbell", "worm_neuron_cell")
CASES_P0 = ("Baron", "Campbell", "worm_neuron_cell", "student_grade", "Tosches", "fabert")
SEED_CANDIDATES = (1019, 2027, 3037, 4051, 5059)
LOCK = EXP / "outputs" / "p0" / "selection_lock_v2.json"
TIE = 1e-12
INVENTORY_EXCLUDE = ("source_coverage_20260925",)


# ------------------------------------------------------------------ seeds

def seed_inventory(root: Path) -> dict:
    seeds: dict[int, list[str]] = {}

    def add(value, where):
        try:
            seeds.setdefault(int(value), [])
            if len(seeds[int(value)]) < 5:
                seeds[int(value)].append(where)
        except (TypeError, ValueError):
            pass

    def walk_json(obj, where, depth=0):
        if depth > 6:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("seed", "train_seed", "training_seed") and isinstance(v, (int, str)):
                    add(v, where)
                elif k in ("seeds", "train_seeds") and isinstance(v, list):
                    for s in v:
                        add(s, where)
                else:
                    walk_json(v, where, depth + 1)
        elif isinstance(obj, list):
            for v in obj[:200000]:
                walk_json(v, where, depth + 1)

    scanned = {"records": 0, "seed_dirs": 0, "json": 0, "json_errors": 0}
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        if any(part in INVENTORY_EXCLUDE for part in Path(rel).parts):
            dirnames[:] = []
            continue
        for d in dirnames:
            m = re.fullmatch(r"seed_(\d+)", d)
            if m:
                add(m.group(1), os.path.join(rel, d))
                scanned["seed_dirs"] += 1
        for name in filenames:
            m = re.fullmatch(r"(?:record|embedding|predictions|model)_seed_(\d+)\..+", name)
            if m:
                add(m.group(1), os.path.join(rel, name))
                scanned["records"] += 1
            if name.endswith(".json") and (name in ("queue.json",) or "task" in name or "manifest" in name
                                           or "lock" in name or rel.endswith("tasks")):
                path = Path(dirpath) / name
                try:
                    if path.stat().st_size < 64 * 1024 * 1024:
                        walk_json(read_json(path), os.path.join(rel, name))
                        scanned["json"] += 1
                except Exception:
                    scanned["json_errors"] += 1
    return {"seeds": {str(k): v for k, v in sorted(seeds.items())}, "scanned": scanned}


def choose_seeds(used: set[int]) -> list[dict]:
    chosen, taken = [], set(used)
    for candidate in SEED_CANDIDATES:
        seed = candidate
        while seed in taken:
            seed += 1
        taken.add(seed)
        chosen.append({"candidate": candidate, "seed": seed, "collision_shift": seed - candidate})
    return chosen


# ------------------------------------------------------------------ cases

def case_score(info: dict) -> dict:
    mh = info.get("selected_multihop")
    ref = info.get("h1_counterpart_of_selected_multihop")
    ref_kind = "same_config_h1_counterpart"
    if ref is None:
        ref, ref_kind = info.get("selected_h1"), "validation_best_h1_fallback"
    if mh is None or ref is None:
        return {"eligible": False, "reason": "missing multihop or H1 reference"}
    deltas = [mh["per_seed"][str(s)]["val_mean"] - ref["per_seed"][str(s)]["val_mean"]
              for s in HISTORICAL_TRAIN_SEEDS]
    return {"eligible": True, "reference_kind": ref_kind, "deltas": deltas,
            "mean_delta": float(np.mean(deltas)), "positive_seeds": int(sum(d > 0 for d in deltas)),
            "delta_sd": float(np.std(deltas, ddof=1))}


def choose_second(lock: dict) -> dict:
    scored = {}
    for name in SECOND_CANDIDATES:
        if name not in lock["datasets"]:
            raise SystemExit(f"second-case candidate {name} missing from lock; datasets: "
                             f"{sorted(lock['datasets'])}")
        scored[name] = case_score(lock["datasets"][name])
    eligible = [n for n in SECOND_CANDIDATES if scored[n]["eligible"]]
    if not eligible:
        raise SystemExit("no eligible second case")

    def key(n):
        s = scored[n]
        return (-round(s["mean_delta"] / TIE) * TIE, -s["positive_seeds"], s["delta_sd"], n)
    ranked = sorted(eligible, key=key)
    return {"chosen": ranked[0], "scores": scored, "ranking": ranked}


def resolve_preprocessing(info: dict, verify: bool) -> dict:
    """Find the (input_kind, feature_limit) whose fingerprint equals the locked hash (label-free)."""
    target = info["selected_multihop"]["preprocessing_hash"]
    kinds = info.get("input_kind_candidates") or ["raw_count", "log1p_expression", "general"]
    limits = info.get("feature_limit_candidates") or [2000]
    if not verify:
        if len(kinds) == 1 and len(limits) == 1:
            return {"input_kind": kinds[0], "feature_limit": int(limits[0]), "verified": False}
        raise SystemExit("ambiguous preprocessing; rerun with --verify-preprocessing")
    from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
    from v0rg_tunable_multihop.tuning import SplitPreprocessor

    manifest = read_json(info["tuning_manifest_path"])
    X = load_npz_matrix_only(manifest["feature_path"]).X
    tried = []
    for kind in list(kinds) + [k for k in ("raw_count", "log1p_expression", "general") if k not in kinds]:
        for limit in limits:
            try:
                fingerprint = SplitPreprocessor(kind, int(limit)).fit(X).fingerprint
            except ValueError as exc:
                tried.append([kind, limit, f"error:{exc}"])
                continue
            tried.append([kind, limit, fingerprint])
            if fingerprint == target:
                return {"input_kind": kind, "feature_limit": int(limit), "verified": True, "tried": tried}
    raise SystemExit(f"no preprocessing reproduces the locked fingerprint: {tried}")


# ------------------------------------------------------------------ tasks

def train_task(dataset, info, manifest, prep, arm, seed, lock_hash) -> dict:
    locked = info["selected_multihop"]["config"]
    config = fp.arm_config(locked, arm)
    spec = ARM_SPECS[arm]
    q = float(locked.get("high_hop_probability_multiplier", 1.0))
    prior = {"kind": spec["prior"], "q": q,
             "alpha": fp.GEOMETRIC_ALPHA if spec["prior"] == "geometric" else None}
    task_id = f"train_{dataset}_{arm}_s{seed}"
    return {"id": task_id, "dataset": dataset, "arm": arm, "arm_spec": spec, "seed": int(seed),
            "config": config, "config_sha256": canonical_sha256(config),
            "reference_multihop_config": locked,
            "reference_multihop_eff_hash": info["selected_multihop"]["eff_hash"],
            "prior": prior, "target": spec["target"],
            "feature_path": manifest["feature_path"], "feature_sha256": manifest["feature_sha256"],
            "n_clusters": int(manifest["n_clusters"]), "K_provenance": manifest["K_provenance"],
            "input_kind": prep["input_kind"], "feature_limit": prep["feature_limit"],
            "expected_preprocessing_hash": info["selected_multihop"]["preprocessing_hash"],
            "selection_lock_sha256": lock_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lock", default=str(LOCK))
    parser.add_argument("--verify-preprocessing", action="store_true")
    parser.add_argument("--inventory-root", default=str(R))
    args = parser.parse_args()
    lock_path = Path(args.lock)
    lock_hash = verify_sidecar_hash(lock_path)
    lock = read_json(lock_path)
    if lock.get("status") != "LOCKED_VALIDATION_ONLY" or lock.get("sealed_columns_parsed") is not False:
        raise SystemExit("prepare requires the validation-only lock")
    queue_path = EXP / "queue.json"
    if queue_path.exists() and (EXP / "state").exists():
        raise SystemExit("queue already exists and the controller has started; refusing to rewrite")

    second = choose_second(lock)
    datasets = [PRIMARY, second["chosen"]]
    inventory = seed_inventory(Path(args.inventory_root))
    used = {int(s) for s in inventory["seeds"]} | set(HISTORICAL_TRAIN_SEEDS)
    seeds = choose_seeds(used)
    seed_values = [s["seed"] for s in seeds]
    atomic_json(EXP / "prereg" / "seed_inventory.json", inventory)

    tasks_dir = EXP / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    queue_tasks, config_locks = [], {}
    for dataset in datasets:
        info = lock["datasets"][dataset]
        mh = info.get("selected_multihop")
        if mh is None or resolve_path_variant(mh["config"]["hop_variant"]) is None:
            raise SystemExit(f"{dataset}: locked multihop config is not an exact-shell H<=8 variant")
        manifest = read_json(info["tuning_manifest_path"])
        if manifest.get("test_label_path_exposed") is not False:
            raise SystemExit(f"{dataset}: manifest exposes sealed labels")
        prep = resolve_preprocessing(info, args.verify_preprocessing)
        h1_config = fp.arm_config(mh["config"], "H1_anchor")
        diff = fp.config_difference(h1_config, fp.arm_config(mh["config"], "flat_anchor"))
        config_locks[dataset] = {"selected_multihop": mh, "h1_reference_config": h1_config,
                                 "h1_vs_multihop_config_difference": diff,
                                 "historical_h1_counterpart": info.get("h1_counterpart_of_selected_multihop"),
                                 "historical_h1_independent_best_secondary": info.get("selected_h1"),
                                 "preprocessing": prep, "feature_sha256": manifest["feature_sha256"],
                                 "n_clusters": int(manifest["n_clusters"]),
                                 "K_provenance": manifest["K_provenance"]}
        write_once_json(EXP / "configlocks" / f"{dataset}.json", json_safe(config_locks[dataset]))
        refcheck_ids, gpu_ids, source_task_files = [], [], []
        for seed in seed_values:
            files = []
            for arm in ARMS:
                task = train_task(dataset, info, manifest, prep, arm, seed, lock_hash)
                path = tasks_dir / f"{task['id']}.json"
                write_once_json(path, json_safe(task))
                files.append(str(path.relative_to(EXP)))
                out = f"outputs/train/{dataset}/{arm}/seed_{seed}"
                queue_tasks.append({"id": task["id"], "kind": "gpu", "stage": "train", "dataset": dataset,
                                    "arm": arm, "seed": seed, "task_file": str(path.relative_to(EXP)),
                                    "out": out, "deps": [f"refcheck_{dataset}"], "dep_mode": "all_completed",
                                    "command": ["run_factorial.py", "train", "--task-json",
                                                str(path.relative_to(EXP)), "--out", out,
                                                "--device", "cuda:0"]})
                gpu_ids.append(task["id"])
                if arm == "flat_anchor":
                    source_task_files.append(str(path.relative_to(EXP)))
            if seed == seed_values[0]:
                out = f"outputs/refcheck/{dataset}"
                queue_tasks.append({"id": f"refcheck_{dataset}", "kind": "refcheck", "stage": "refcheck",
                                    "dataset": dataset, "out": out, "summary": "summary.json", "deps": [],
                                    "command": ["run_factorial.py", "refcheck", "--task-json", *files,
                                                "--out", out, "--epochs", "2"]})
                refcheck_ids.append(f"refcheck_{dataset}")
        out = f"outputs/sources/{dataset}"
        queue_tasks.append({"id": f"sources_{dataset}", "kind": "cpu", "stage": "cpu_analysis",
                            "dataset": dataset, "out": out, "summary": "summary.json", "deps": [],
                            "command": ["run_factorial.py", "sources", "--task-json", *source_task_files,
                                        "--out", out]})
        out = f"outputs/factorial/{dataset}"
        queue_tasks.append({"id": f"factorial_{dataset}", "kind": "cpu", "stage": "posthoc",
                            "dataset": dataset, "out": out, "summary": "summary.json",
                            "deps": gpu_ids, "dep_mode": "all_terminal", "allow_partial": True,
                            "command": ["analyze_cases.py", "factorial", "--dataset", dataset,
                                        "--lock", str(lock_path), "--out", out]})
    for dataset in CASES_P0:
        if dataset not in lock["datasets"]:
            continue
        out = f"outputs/p0_cases/{dataset}"
        queue_tasks.append({"id": f"p0case_{dataset}", "kind": "cpu", "stage": "cpu_analysis",
                            "dataset": dataset, "out": out, "summary": "summary.json", "deps": [],
                            "allow_partial": True,
                            "command": ["analyze_cases.py", "historical", "--dataset", dataset,
                                        "--lock", str(lock_path), "--out", out]})
    queue_tasks.append({"id": "p0_report", "kind": "cpu", "stage": "cpu_analysis", "out": "outputs/p0/report",
                        "summary": "summary.json", "deps": [],
                        "command": ["audit_select.py", "report", "--out", "outputs/p0"]})
    extension_path = EXP / "extension_tasks.json"
    extension_hash = None
    if extension_path.exists():
        extension_hash = sha256(extension_path)
        for task in read_json(extension_path)["tasks"]:
            if not {"id", "kind", "stage", "out", "command"} <= set(task) or task["kind"] == "gpu":
                raise SystemExit(f"invalid extension task {task.get('id')}")
            queue_tasks.append(task)
    missing_cases = [d for d in CASES_P0 if d not in lock["datasets"]]
    all_ids = [t["id"] for t in queue_tasks]
    queue_tasks.append({"id": "summarize", "kind": "cpu", "stage": "summarize", "out": "outputs/summary",
                        "summary": "summary.json", "deps": all_ids, "dep_mode": "all_terminal",
                        "allow_partial": True,
                        "command": ["analyze_cases.py", "summarize", "--out", "outputs/summary"]})
    ids = [t["id"] for t in queue_tasks]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate task ids")
    n_gpu = sum(t["kind"] == "gpu" for t in queue_tasks)
    if n_gpu != 50:
        raise SystemExit(f"expected 50 main training tasks, got {n_gpu}")
    prereg = {
        "protocol": "source_coverage_20260925_factorial_v1", "prepared_unix": now(),
        "selection_lock_sha256": lock_hash, "cases": datasets, "second_case_rule": second,
        "missing_p0_cases": missing_cases,
        "seeds": seeds, "seed_rule": __doc__.split("Seed rule:", 1)[1].strip(),
        "arms": {a: ARM_SPECS[a] for a in ARMS},
        "geometric_prior": {"alpha": fp.GEOMETRIC_ALPHA, "form": "w_h ∝ (1-alpha)^(h-1), nonempty shells",
                            "not": "PPR/APPNP implementation"},
        "sensitivity_alphas_sources_only": list(fp.SENSITIVITY_ALPHAS),
        "contrasts": {"main": "flat_anchor - H1_anchor",
                      "prior": "flat_anchor - geometric_anchor",
                      "target": "flat_anchor - flat_mixed",
                      "interaction": "(flat_anchor - geometric_anchor) - (flat_mixed - geometric_mixed)"},
        "primary_endpoint": "per training seed: mean full-X ARI over 20 KMeans readouts "
                            "(n_init=20, random_state 10..29); paired over 5 training seeds",
        "secondary_endpoints": ["min-inertia readout ARI", "NMI", "pairwise class split / cross-class merge",
                                "normalized geometry", "community coverage"],
        "readout_repeats_not_independent": True,
        "equal_amplitude_sensitivity_arm": "not in this queue; only if displacement distributions differ markedly, "
                                           "separately preregistered",
        "H1_reference": "same locked config with hop_variant=H1, q=1 (matched causal baseline); "
                        "historical independent-best H1 is secondary/descriptive only",
        "labels": {"fit": "never", "selection": "validation-only lock (historical)",
                   "posthoc": "full labels in analyze_cases only"},
        "K": "benchmark_oracle_from_y (explicit)",
        "failure_policy": "max 3 attempts (2 retries); failed attempts preserved; no seed replacement",
        "code_sha256": code_hashes([f for f in CODE_FILES if (EXP / f).exists()]),
        "extension_tasks_sha256": extension_hash,
    }
    prereg_path = EXP / "prereg" / "prereg.json"
    if prereg_path.exists():
        # Pre-launch revision during review: keep the previous version, never delete it.
        prereg_path.replace(EXP / "prereg" / f"prereg_prelaunch_revision_{int(now())}.json")
    atomic_json(prereg_path, json_safe(prereg))
    source = {"core_package_sha256": core_source_hashes(R), "captured": now()}
    atomic_json(EXP / "prereg" / "source_hashes.json", source)
    tasks_manifest = {t["task_file"]: sha256(EXP / t["task_file"]) for t in queue_tasks if t.get("task_file")}
    queue = {"protocol": prereg["protocol"], "prereg_sha256": sha256(EXP / "prereg" / "prereg.json"),
             "selection_lock_sha256": lock_hash, "core_package_sha256": source["core_package_sha256"],
             "task_file_sha256": tasks_manifest, "tasks": queue_tasks}
    encoded = (json.dumps(queue, ensure_ascii=False, indent=2) + "\n").encode()
    if queue_path.exists() and queue_path.read_bytes() != encoded:
        (EXP / "prereg" / f"queue_prelaunch_revision_{int(now())}.json").write_bytes(queue_path.read_bytes())
    queue_path.write_bytes(encoded)
    (EXP / "queue.sha256").write_text(sha256(queue_path) + "\n")
    print(json.dumps({"cases": datasets, "seeds": seed_values, "tasks": len(queue_tasks), "gpu_tasks": n_gpu,
                      "queue_sha256": sha256(queue_path)}, indent=2))


if __name__ == "__main__":
    main()
