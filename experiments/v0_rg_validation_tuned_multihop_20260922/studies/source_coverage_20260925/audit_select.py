"""P0: audit the historical 10-readout rescoring, export validation-only scores,
lock ONE configuration per dataset, and (separately, afterwards) report.

Subcommands
-----------
``select``  reads only whitelisted validation columns of readout_all.csv plus the
            original run records; writes scores_val.csv, audits and the immutable
            selection_lock_v2.json (+ .sha256) BEFORE anything else.
``report``  requires a verified lock, then reads sealed ari_test/ari_full columns
            for descriptive retrospective comparisons (never feeds back to select).

Selection rule (validation only): per effective config, mean over the 10 readouts
(states 0..9) of validation ARI per training seed, then mean over the three
training seeds {42,123,7}.  Only configs complete on all three seeds compete.
Ties (|diff| <= 1e-12) prefer lower max hop, lower shell width, then q, then a,
then the effective-config hash.  Scores are never averaged across duplicates;
duplicate embeddings of one effective config are resolved by fixed stage order.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from common import (HISTORICAL_TRAIN_SEEDS, R, atomic_json, atomic_text, canonical_sha256,  # noqa: E402
                    code_hashes, hop_family, json_safe, now, read_json, resolve_path_variant,
                    sha256, verify_sidecar_hash, write_once_json)

TOP = R / "top5_h1_h8_20260923"
STAGES = ("pilot_queue", "top5_h1_h8_20260923/depth", "top5_h1_h8_20260923/width",
          "top5_h1_h8_20260923/strength", "tuning19/stage_a", "tuning19/stage_b", "tuning19/stage_c")
READOUT_CSV = Path("/tmp/claude_readout_all/readout_all.csv")
VALIDATION_COLUMNS = ("dataset", "stage", "config_id", "seed", "random_state", "inertia", "ari_val")
SEALED_COLUMNS = ("ari_test", "ari_full")
EXPECTED_STATES = tuple(range(10))
BOOKKEEPING_FIELDS = frozenset({"protocol_id"})
TIE_TOL = 1e-12
DEFAULT_OUT = EXP / "outputs" / "p0"


# ------------------------------------------------------------------ helpers

def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def export_validation_scores(csv_path: Path, out_csv: Path) -> dict:
    """Stream the mixed CSV, keeping ONLY whitelisted columns (sealed ones are never parsed)."""
    header = _read_header(csv_path)
    expected = set(VALIDATION_COLUMNS) | set(SEALED_COLUMNS)
    if set(header) != expected or len(header) != len(expected):
        raise ValueError(f"unexpected readout CSV columns: {header}")
    positions = [header.index(name) for name in VALIDATION_COLUMNS]
    groups: dict[tuple, dict[int, tuple[float, float]]] = defaultdict(dict)
    problems = []
    rows = 0
    with csv_path.open(newline="", encoding="utf-8") as handle, \
            out_csv.open("w", newline="", encoding="utf-8") as sink:
        reader = csv.reader(handle)
        next(reader)
        writer = csv.writer(sink)
        writer.writerow(VALIDATION_COLUMNS)
        for line_no, fields in enumerate(reader, start=2):
            if len(fields) != len(header):
                problems.append({"line": line_no, "problem": "field_count"})
                continue
            kept = [fields[i] for i in positions]
            dataset, stage, config_id = kept[0], kept[1], kept[2]
            try:
                seed, state = int(kept[3]), int(kept[4])
                inertia, ari_val = float(kept[5]), float(kept[6])
            except ValueError:
                problems.append({"line": line_no, "problem": "unparseable_validation_field"})
                continue
            if not (math.isfinite(inertia) and math.isfinite(ari_val)):
                problems.append({"line": line_no, "problem": "nonfinite_validation_value"})
                continue
            key = (dataset, stage, config_id, seed)
            if state in groups[key]:
                problems.append({"line": line_no, "problem": "duplicate_state", "key": list(key)})
                continue
            groups[key][state] = (inertia, ari_val)
            writer.writerow(kept)
            rows += 1
    return {"groups": groups, "rows": rows, "problems": problems}


def latest_attempt_records(dataset: str) -> dict[tuple, dict]:
    """Mirror readout_all_configs.latest_attempts and keep full attempt inventory."""
    inventory: dict[tuple, dict] = {}
    for stage in STAGES:
        base = R / stage / "runs" / dataset
        if not base.is_dir():
            continue
        for seed_dir in base.glob("*/seed_*"):
            config_id = seed_dir.parent.name
            try:
                seed = int(seed_dir.name[5:])
            except ValueError:
                continue
            attempts = []
            for attempt in sorted(seed_dir.glob("attempt_*")):
                digits = "".join(ch for ch in attempt.name if ch.isdigit())
                if not digits:
                    continue
                record = attempt / f"record_seed_{seed}.json"
                attempts.append({"attempt": int(digits), "dir": attempt, "record": record,
                                 "has_record": record.is_file(),
                                 "has_failed": (attempt / "failed.json").is_file()})
            with_record = [a for a in attempts if a["has_record"]]
            if not with_record:
                continue
            used = max(with_record, key=lambda a: a["attempt"])
            inventory[(stage, config_id, seed)] = {
                "used": used,
                "attempt_count": len(attempts),
                "newer_attempts_without_record": [a["attempt"] for a in attempts
                                                  if a["attempt"] > used["attempt"]],
                "older_attempts_with_record": [a["attempt"] for a in with_record
                                               if a["attempt"] < used["attempt"]],
            }
    return inventory


def effective_config(record: dict) -> dict:
    # Old records omit fields introduced later. Compare effective defaults rather
    # than treating absent q=1/a=1 as a different scientific configuration.
    from dataclasses import asdict
    from v0rg_tunable_multihop.config import V0_RGConfig
    resolved = asdict(V0_RGConfig(**record["config"]))
    config = {k: v for k, v in resolved.items() if k not in BOOKKEEPING_FIELDS}
    return {"config": config, "preprocessing_hash": record.get("preprocessing_hash"),
            "feature_sha256": record.get("feature_sha256")}


def config_sort_key(config: dict) -> tuple:
    resolved = resolve_path_variant(config.get("hop_variant", "H1"))
    depth, width = (1, 1) if resolved is None else resolved
    return (depth, width, float(config.get("high_hop_probability_multiplier", 1.0)),
            float(config.get("mix_amplitude_multiplier", 1.0)))


def rank_configs(entries: list[dict], score_key: str = "val_mean") -> list[dict]:
    """Deterministic ranking: score desc with tolerance, then low H/B/q/a, then hash."""
    ordered = sorted(entries, key=lambda e: (-e[score_key], config_sort_key(e["config"]), e["eff_hash"]))
    # Tolerance-aware pass: within tied score blocks re-sort by the cost key only.
    result, block = [], []
    for entry in ordered:
        if block and abs(block[0][score_key] - entry[score_key]) > TIE_TOL:
            result.extend(sorted(block, key=lambda e: (config_sort_key(e["config"]), e["eff_hash"])))
            block = []
        block.append(entry)
    result.extend(sorted(block, key=lambda e: (config_sort_key(e["config"]), e["eff_hash"])))
    return result


def h1_counterpart_hash(entry: dict) -> str:
    config = dict(entry["config"])
    config["hop_variant"] = "H1"
    config["high_hop_probability_multiplier"] = 1.0
    return canonical_sha256({"config": config, "preprocessing_hash": entry["preprocessing_hash"],
                             "feature_sha256": entry["feature_sha256"]})


def _audit_one(args) -> dict:
    dataset, key, info, manifest, csv_mtime = args
    stage, config_id, seed = key
    used = info["used"]
    record_path = used["record"]
    embedding_path = record_path.with_name(f"embedding_seed_{seed}.npz")
    result = {"dataset": dataset, "stage": stage, "config_id": config_id, "seed": seed,
              "attempt": used["attempt"], "record_path": str(record_path),
              "embedding_path": str(embedding_path),
              "newer_attempts_without_record": info["newer_attempts_without_record"],
              "older_attempts_with_record": info["older_attempts_with_record"],
              "problems": []}
    try:
        record = read_json(record_path)
    except Exception as exc:  # corrupt record is an audit finding, not a crash
        result["problems"].append(f"record_unreadable:{exc!r}")
        result["eligible"] = False
        return result
    result["record_sha256"] = sha256(record_path)
    checks = {
        "status_completed": record.get("status") == "completed",
        "seed_matches": int(record.get("seed", -1)) == seed,
        "test_labels_loaded_false": record.get("test_labels_loaded") is False,
        "labels_used_during_fit_false": record.get("labels_used_during_fit") is False,
        "selection_scope_validation_only": record.get("selection_label_scope") == "validation_only",
        "feature_hash_matches_manifest": record.get("feature_sha256") == manifest.get("feature_sha256"),
        "validation_labels_hash_matches": (record.get("validation_labels_sha256")
                                           == manifest.get("validation_labels_sha256")),
        "k_provenance_oracle": record.get("K_provenance", manifest.get("K_provenance"))
                               == "benchmark_oracle_from_y",
        "embedding_exists": embedding_path.is_file(),
        "has_config": isinstance(record.get("config"), dict),
    }
    for name, ok in checks.items():
        if not ok:
            result["problems"].append(name)
    if embedding_path.is_file():
        result["embedding_sha256"] = sha256(embedding_path)
        result["embedding_mtime_after_readout_csv"] = embedding_path.stat().st_mtime > csv_mtime
        if result["embedding_mtime_after_readout_csv"]:
            result["problems"].append("embedding_newer_than_readout_csv")
    if checks["has_config"]:
        try:
            from v0rg_tunable_multihop.config import V0_RGConfig
            V0_RGConfig(**record["config"])
        except Exception as exc:
            result["problems"].append(f"config_invalid:{exc!r}")
        eff = effective_config(record)
        result["eff_hash"] = canonical_sha256(eff)
        result["effective"] = eff
    result["checks"] = checks
    result["eligible"] = not result["problems"]
    return result


# ------------------------------------------------------------------ select

def select(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "selection_lock_v2.json"
    if lock_path.exists():
        verify_sidecar_hash(lock_path)
        print(f"selection lock already exists and verifies: {lock_path}")
        return
    csv_path = Path(args.readout_csv)
    csv_mtime = csv_path.stat().st_mtime
    exported = export_validation_scores(csv_path, out / "scores_val.csv")
    groups = exported["groups"]

    historical_lock_path = TOP / "heldout_eval" / "selection_lock.json"
    historical = {r["dataset"]: r for r in read_json(historical_lock_path)["rows"]}
    old_tasks_path = TOP / "inputs" / "old_top5_tasks.json"
    sources: dict[str, dict] = defaultdict(lambda: {"paths": set(), "input_kind": set(), "feature_limit": set()})
    for task in read_json(old_tasks_path)["tasks"]:
        slot = sources[task["dataset"]]
        slot["paths"].add(task["data_path"])
        if task.get("input_kind") is not None:
            slot["input_kind"].add(task["input_kind"])
        if task.get("feature_limit") is not None:
            slot["feature_limit"].add(int(task["feature_limit"]))

    manifests = {d: read_json(row["tuning_manifest_path"]) for d, row in historical.items()}
    for d, manifest in manifests.items():
        forbidden = {"test_labels_path", "test_label_path", "full_labels_path"} & set(manifest)
        if forbidden or manifest.get("test_label_path_exposed") is not False:
            raise RuntimeError(f"tuning manifest for {d} exposes sealed labels")

    # Readout groups: exact 10 states 0..9.
    state_problems = []
    for key, states in groups.items():
        if tuple(sorted(states)) != EXPECTED_STATES:
            state_problems.append({"key": list(key), "states": sorted(states)})
    datasets = sorted({k[0] for k in groups})
    counts = {"rows": exported["rows"], "embeddings": len(groups), "datasets": len(datasets)}
    expected_counts = {"rows": args.expected_rows, "embeddings": args.expected_embeddings,
                       "datasets": args.expected_datasets}
    count_mismatch = {k: [counts[k], v] for k, v in expected_counts.items() if counts[k] != v}
    if (count_mismatch or exported["problems"] or state_problems) and not args.allow_audit_findings:
        atomic_json(out / "audit_blocking_findings.json", json_safe({
            "count_mismatch": count_mismatch, "csv_problems": exported["problems"][:1000],
            "state_problems": state_problems[:1000]}))
        raise SystemExit("readout CSV audit failed; see audit_blocking_findings.json")
    unknown = sorted(set(datasets) - set(historical))
    if unknown:
        raise SystemExit(f"datasets without historical lock rows: {unknown}")

    # Source-record audit (mapping reconstructed exactly as the readout script did).
    jobs = []
    unmapped = []
    for dataset in datasets:
        inventory = latest_attempt_records(dataset)
        for key in [k for k in groups if k[0] == dataset]:
            inner = key[1:]
            if inner not in inventory:
                unmapped.append(list(key))
                continue
            jobs.append((dataset, inner, inventory[inner], manifests[dataset], csv_mtime))
    with ThreadPoolExecutor(max(1, int(args.hash_workers))) as pool:
        audits = list(pool.map(_audit_one, jobs))
    audit_by_key = {(a["dataset"], a["stage"], a["config_id"], a["seed"]): a for a in audits}

    with (out / "record_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "stage", "config_id", "seed", "attempt", "eligible", "problems",
                         "eff_hash", "record_sha256", "embedding_sha256", "record_path"])
        for a in audits:
            writer.writerow([a["dataset"], a["stage"], a["config_id"], a["seed"], a["attempt"],
                             a["eligible"], ";".join(a["problems"]), a.get("eff_hash", ""),
                             a.get("record_sha256", ""), a.get("embedding_sha256", ""),
                             a["record_path"]])

    # Effective-config aggregation.
    stage_rank = {s: i for i, s in enumerate(STAGES)}
    per_dataset: dict[str, dict] = {}
    missing_rows, duplicate_rows = [], []
    all_config_rows = []
    for dataset in datasets:
        eff: dict[str, dict] = {}
        for key, states in groups.items():
            if key[0] != dataset:
                continue
            audit = audit_by_key.get(key)
            if audit is None or not audit["eligible"] or tuple(sorted(states)) != EXPECTED_STATES:
                continue
            slot = eff.setdefault(audit["eff_hash"], {"effective": audit["effective"], "seeds": defaultdict(list)})
            ari = np.asarray([states[s][1] for s in EXPECTED_STATES], dtype=np.float64)
            inertia = np.asarray([states[s][0] for s in EXPECTED_STATES], dtype=np.float64)
            slot["seeds"][key[3]].append({
                "stage": key[1], "config_id": key[2], "seed": key[3],
                "val_mean": float(ari.mean()), "val_min_inertia": float(ari[int(np.argmin(inertia))]),
                "record_path": audit["record_path"], "record_sha256": audit["record_sha256"],
                "embedding_path": audit["embedding_path"], "embedding_sha256": audit["embedding_sha256"],
            })
        entries = []
        for eff_hash, slot in eff.items():
            reps = {}
            for seed, candidates in slot["seeds"].items():
                candidates = sorted(candidates, key=lambda c: (stage_rank.get(c["stage"], 99), c["config_id"]))
                reps[seed] = candidates[0]
                if len(candidates) > 1:
                    duplicate_rows.append({
                        "dataset": dataset, "eff_hash": eff_hash, "seed": seed,
                        "chosen": [candidates[0]["stage"], candidates[0]["config_id"]],
                        "others": [[c["stage"], c["config_id"]] for c in candidates[1:]],
                        "identical_embeddings": len({c["embedding_sha256"] for c in candidates}) == 1})
            config = slot["effective"]["config"]
            present = sorted(reps)
            missing = sorted(set(HISTORICAL_TRAIN_SEEDS) - set(present))
            entry = {"dataset": dataset, "eff_hash": eff_hash, "config": config,
                     "preprocessing_hash": slot["effective"]["preprocessing_hash"],
                     "feature_sha256": slot["effective"]["feature_sha256"],
                     "family": hop_family(config.get("hop_variant", "H1")),
                     "config_ids": sorted({reps[s]["config_id"] for s in present}),
                     "seeds_present": present, "seeds_missing": missing,
                     "complete": not missing,
                     "per_seed": {str(s): reps[s] for s in present}}
            if entry["complete"]:
                vals = [reps[s]["val_mean"] for s in HISTORICAL_TRAIN_SEEDS]
                entry["val_mean"] = float(np.mean(vals))
                entry["val_seed_sd"] = float(np.std(vals, ddof=1))
                entry["val_min_inertia_mean"] = float(np.mean(
                    [reps[s]["val_min_inertia"] for s in HISTORICAL_TRAIN_SEEDS]))
            else:
                missing_rows.append({"dataset": dataset, "eff_hash": eff_hash,
                                     "config_ids": entry["config_ids"],
                                     "hop_variant": config.get("hop_variant"),
                                     "seeds_present": present, "seeds_missing": missing})
            entries.append(entry)
            all_config_rows.append(entry)
        complete = [e for e in entries if e["complete"]]
        by_hash = {e["eff_hash"]: e for e in complete}
        families = {f: rank_configs([e for e in complete if e["family"] == f]) for f in ("H1", "multihop")}
        path_space = rank_configs([e for e in complete if e["family"] in ("H1", "multihop")])
        for f, ranked in list(families.items()) + [("path_space", path_space)]:
            for rank, e in enumerate(ranked, 1):
                e.setdefault("ranks", {})[f] = rank
        selected_mh = families["multihop"][0] if families["multihop"] else None
        counterpart = None
        if selected_mh is not None:
            counterpart = by_hash.get(h1_counterpart_hash(selected_mh))
        per_dataset[dataset] = {
            "selected_overall": _slim(path_space[0]) if path_space else None,
            "selected_h1": _slim(families["H1"][0]) if families["H1"] else None,
            "selected_multihop": _slim(selected_mh) if selected_mh else None,
            "h1_counterpart_of_selected_multihop": _slim(counterpart) if counterpart else None,
            "h1_counterpart_status": ("complete" if counterpart else
                                      ("absent_or_incomplete" if selected_mh else "no_multihop")),
            "n_complete_configs": {"H1": len(families["H1"]), "multihop": len(families["multihop"]),
                                   "other": sum(e["family"] == "other" for e in complete)},
            "n_incomplete_configs": sum(not e["complete"] for e in entries),
            "tuning_manifest_path": historical[dataset]["tuning_manifest_path"],
            "feature_sha256": manifests[dataset]["feature_sha256"],
            "n_samples": int(manifests[dataset]["n_samples"]),
            "n_clusters": int(manifests[dataset]["n_clusters"]),
            "K_provenance": manifests[dataset]["K_provenance"],
            "validation_labels_sha256": manifests[dataset]["validation_labels_sha256"],
            "input_kind_candidates": sorted(sources[dataset]["input_kind"]),
            "feature_limit_candidates": sorted(sources[dataset]["feature_limit"]),
            "posthoc_label_source_paths": sorted(sources[dataset]["paths"]),
        }

    with (out / "effective_configs_val.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "eff_hash", "family", "hop_variant", "q", "a", "config_ids",
                         "complete", "seeds_missing", "val_mean", "val_seed_sd", "val_min_inertia_mean",
                         "rank_H1", "rank_multihop", "rank_path_space"])
        for e in all_config_rows:
            c = e["config"]
            ranks = e.get("ranks", {})
            writer.writerow([e["dataset"], e["eff_hash"], e["family"], c.get("hop_variant"),
                             c.get("high_hop_probability_multiplier"), c.get("mix_amplitude_multiplier"),
                             "|".join(e["config_ids"]), e["complete"], "|".join(map(str, e["seeds_missing"])),
                             e.get("val_mean", ""), e.get("val_seed_sd", ""), e.get("val_min_inertia_mean", ""),
                             ranks.get("H1", ""), ranks.get("multihop", ""), ranks.get("path_space", "")])
    atomic_json(out / "missing_configs.json", json_safe({"rows": missing_rows,
                                                         "unmapped_readout_groups": unmapped}))
    atomic_json(out / "duplicate_effective_configs.json", json_safe(duplicate_rows))
    ineligible = [a for a in audits if not a["eligible"]]
    atomic_json(out / "ineligible_records.json", json_safe(
        [{k: a[k] for k in ("dataset", "stage", "config_id", "seed", "problems", "record_path")}
         for a in ineligible]))
    files = ["scores_val.csv", "record_audit.csv", "effective_configs_val.csv", "missing_configs.json",
             "duplicate_effective_configs.json", "ineligible_records.json"]
    lock = {
        "status": "LOCKED_VALIDATION_ONLY",
        "created": now(),
        "selection_rule": __doc__.split("Selection rule", 1)[1].strip(),
        "sealed_columns_parsed": False,
        "sealed_columns": list(SEALED_COLUMNS),
        "validation_columns": list(VALIDATION_COLUMNS),
        "readout_states": list(EXPECTED_STATES),
        "readout_n_init": 20,
        "train_seeds": list(HISTORICAL_TRAIN_SEEDS),
        "counts": counts, "expected_counts": expected_counts, "count_mismatch": count_mismatch,
        "csv_problems": len(exported["problems"]), "state_problems": len(state_problems),
        "ineligible_records": len(ineligible), "unmapped_readout_groups": len(unmapped),
        "incomplete_effective_configs": len(missing_rows),
        "input_sha256": {"readout_csv": sha256(csv_path),
                         "historical_selection_lock": sha256(historical_lock_path),
                         "old_top5_tasks": sha256(old_tasks_path)},
        "output_sha256": {name: sha256(out / name) for name in files},
        "code_sha256": code_hashes(["audit_select.py", "common.py"]),
        "K_note": "n_clusters is benchmark_oracle_from_y; validation-ARI selection is label-assisted model selection",
        "datasets": per_dataset,
    }
    write_once_json(lock_path, json_safe(lock))
    atomic_json(out / "select_summary.json", {"status": "completed", "lock_sha256": sha256(lock_path),
                                              "counts": counts, "count_mismatch": count_mismatch})
    print(f"locked {len(per_dataset)} datasets -> {lock_path}")


def _slim(entry: dict) -> dict:
    return {k: entry[k] for k in ("eff_hash", "config", "family", "config_ids", "val_mean", "val_seed_sd",
                                  "val_min_inertia_mean", "per_seed", "preprocessing_hash")}


# ------------------------------------------------------------------ report (sealed, after lock)

def load_sealed(csv_path: Path) -> dict[tuple, dict[str, np.ndarray]]:
    header = _read_header(csv_path)
    idx = {name: header.index(name) for name in VALIDATION_COLUMNS + SEALED_COLUMNS}
    table: dict[tuple, dict[int, tuple]] = defaultdict(dict)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for f in reader:
            key = (f[idx["dataset"]], f[idx["stage"]], f[idx["config_id"]], int(f[idx["seed"]]))
            table[key][int(f[idx["random_state"]])] = (float(f[idx["ari_val"]]), float(f[idx["ari_test"]]),
                                                        float(f[idx["ari_full"]]))
    result = {}
    for key, states in table.items():
        ordered = [states[s] for s in sorted(states)]
        arr = np.asarray(ordered, dtype=np.float64)
        result[key] = {"val": arr[:, 0], "test": arr[:, 1], "full": arr[:, 2]}
    return result


def _entry_scores(entry: dict, sealed: dict, dataset: str) -> dict:
    out = {"val": [], "test": [], "full": []}
    for seed in HISTORICAL_TRAIN_SEEDS:
        rep = entry["per_seed"][str(seed)]
        values = sealed[(dataset, rep["stage"], rep["config_id"], int(seed))]
        for name in out:
            if not np.isfinite(values[name]).all():
                raise ValueError(f"nonfinite sealed value {dataset} {rep}")
            out[name].append(float(values[name].mean()))
    return out


def report(args) -> None:
    out = Path(args.out)
    lock_path = out / "selection_lock_v2.json"
    lock_hash = verify_sidecar_hash(lock_path)
    lock = read_json(lock_path)
    if lock.get("status") != "LOCKED_VALIDATION_ONLY":
        raise SystemExit("lock is not a validation-only lock")
    if sha256(Path(args.readout_csv)) != lock["input_sha256"]["readout_csv"]:
        raise SystemExit("readout CSV changed after lock")
    rep_dir = out / "report"
    rep_dir.mkdir(exist_ok=True)
    sealed = load_sealed(Path(args.readout_csv))
    # Reload every complete effective config from the lock-time audit to keep one mapping.
    configs = _reload_complete_configs(out)

    selected_rows, paired_rows, all_rows, budget_rows, curve_rows = [], [], [], [], []
    for d_index, (dataset, info) in enumerate(sorted(lock["datasets"].items())):
        roles = {r: info.get(k) for r, k in (("overall", "selected_overall"), ("h1_best", "selected_h1"),
                                             ("multihop_best", "selected_multihop"),
                                             ("h1_counterpart", "h1_counterpart_of_selected_multihop"))}
        scores = {}
        for role, entry in roles.items():
            if entry is None:
                continue
            scores[role] = _entry_scores(entry, sealed, dataset)
            for i, seed in enumerate(HISTORICAL_TRAIN_SEEDS):
                selected_rows.append([dataset, role, entry["config"]["hop_variant"], entry["eff_hash"], seed,
                                      scores[role]["val"][i], scores[role]["test"][i], scores[role]["full"][i]])
        for a, b in (("multihop_best", "h1_best"), ("multihop_best", "h1_counterpart"), ("overall", "h1_best")):
            if a in scores and b in scores:
                for i, seed in enumerate(HISTORICAL_TRAIN_SEEDS):
                    paired_rows.append([dataset, f"{a}-{b}", seed,
                                        *(scores[a][m][i] - scores[b][m][i] for m in ("val", "test", "full"))])
        entries = configs.get(dataset, [])
        table = []
        for e in entries:
            s = _entry_scores(e, sealed, dataset)
            table.append((e, {m: float(np.mean(v)) for m, v in s.items()}))
            all_rows.append([dataset, e["eff_hash"], e["family"], e["config"]["hop_variant"],
                             e["config"].get("high_hop_probability_multiplier"),
                             e["config"].get("mix_amplitude_multiplier"),
                             e["val_mean"], float(np.mean(s["test"])), float(np.mean(s["full"]))])
        budget_rows.append(_budget_sensitivity(dataset, table, curve_rows,
                                               int(args.budget_draws), 20260925 + d_index))

    _write_csv(rep_dir / "retrospective_selected.csv",
               ["dataset", "role", "hop_variant", "eff_hash", "seed", "val_ari", "test_ari", "full_ari"], selected_rows)
    _write_csv(rep_dir / "retrospective_paired.csv",
               ["dataset", "contrast", "seed", "delta_val", "delta_test", "delta_full"], paired_rows)
    _write_csv(rep_dir / "retrospective_all_configs.csv",
               ["dataset", "eff_hash", "family", "hop_variant", "q", "a", "val_mean", "test_mean", "full_mean"],
               all_rows)
    _write_csv(rep_dir / "search_budget_sensitivity.csv",
               ["dataset", "status", "b_h1", "n_multihop", "draws", "mean_delta_full", "p05_delta_full",
                "p50_delta_full", "p95_delta_full", "frac_delta_full_gt0", "mean_delta_test", "frac_delta_test_gt0"],
               budget_rows)
    _write_csv(rep_dir / "search_budget_curve.csv",
               ["dataset", "family", "budget", "draws", "mean_selected_full", "mean_selected_test",
                "mean_selected_val"], curve_rows)
    lodo = _shared_policy_lodo(lock, configs, sealed, rep_dir)
    files = ["retrospective_selected.csv", "retrospective_paired.csv", "retrospective_all_configs.csv",
             "search_budget_sensitivity.csv", "search_budget_curve.csv", "shared_policy_coverage.csv",
             "shared_policy_lodo.json"]
    atomic_json(rep_dir / "summary.json", json_safe({
        "status": "completed", "lock_sha256": lock_hash,
        "nature": "retrospective descriptive; full/test values were not used for selection and "
                  "historical full/test results were already exposed to prior research decisions",
        "budget_note": "subsampling of historically adaptive configs; exploratory, not a randomized equal-budget experiment",
        "shared_policy_lodo_status": lodo["status"],
        "output_sha256": {name: sha256(rep_dir / name) for name in files}}))


def _reload_complete_configs(out: Path) -> dict[str, list[dict]]:
    """Recompute complete configs from scores_val.csv + record_audit.csv (lock-time artifacts)."""
    lock = read_json(out / "selection_lock_v2.json")
    for name, expected in lock["output_sha256"].items():
        if sha256(out / name) != expected:
            raise SystemExit(f"lock-time artifact changed: {name}")
    audit = {}
    with (out / "record_audit.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["eligible"] == "True":
                audit[(row["dataset"], row["stage"], row["config_id"], int(row["seed"]))] = row
    val = defaultdict(list)
    with (out / "scores_val.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            val[(row["dataset"], row["stage"], row["config_id"], int(row["seed"]))].append(float(row["ari_val"]))
    configs = defaultdict(dict)
    stage_rank = {s: i for i, s in enumerate(STAGES)}
    for key, row in audit.items():
        if len(val.get(key, [])) != len(EXPECTED_STATES):
            continue
        slot = configs[key[0]].setdefault(row["eff_hash"], {})
        prev = slot.get(key[3])
        if prev is None or (stage_rank.get(key[1], 99), key[2]) < (stage_rank.get(prev[0], 99), prev[1]):
            slot[key[3]] = (key[1], key[2], row["record_path"])
    result = {}
    for dataset, by_hash in configs.items():
        entries = []
        for eff_hash, seeds in by_hash.items():
            if set(seeds) != set(HISTORICAL_TRAIN_SEEDS):
                continue
            record = read_json(seeds[HISTORICAL_TRAIN_SEEDS[0]][2])
            eff = effective_config(record)
            per_seed = {str(s): {"stage": seeds[s][0], "config_id": seeds[s][1]} for s in seeds}
            vals = [float(np.mean(val[(dataset, seeds[s][0], seeds[s][1], s)])) for s in HISTORICAL_TRAIN_SEEDS]
            entries.append({"eff_hash": eff_hash, "config": eff["config"], "family": hop_family(
                eff["config"].get("hop_variant", "H1")), "per_seed": per_seed, "val_mean": float(np.mean(vals)),
                "preprocessing_hash": eff["preprocessing_hash"], "feature_sha256": eff["feature_sha256"]})
        result[dataset] = entries
    return result


def _budget_sensitivity(dataset, table, curve_rows, draws, seed) -> list:
    h1 = [(e, s) for e, s in table if e["family"] == "H1"]
    mh = [(e, s) for e, s in table if e["family"] == "multihop"]
    if not h1 or not mh:
        return [dataset, "unavailable_missing_family", len(h1), len(mh), 0] + [""] * 7
    rng = np.random.default_rng(seed)

    def ordered(items):
        ranked = rank_configs([e for e, _ in items])
        pos = {e["eff_hash"]: i for i, e in enumerate(ranked)}
        return np.asarray([pos[e["eff_hash"]] for e, _ in items]), \
            {m: np.asarray([s[m] for _, s in items]) for m in ("val", "test", "full")}

    h1_rank, h1_s = ordered(h1)
    mh_rank, mh_s = ordered(mh)
    h1_best = int(np.argmin(h1_rank))
    b = len(h1)
    m = min(b, len(mh))

    def pick(rank, budget, n):
        keys = rng.random((draws, n))
        subsets = np.argsort(keys, axis=1)[:, :budget]
        return subsets[np.arange(draws), np.argmin(rank[subsets], axis=1)]

    chosen = pick(mh_rank, m, len(mh))
    d_full = mh_s["full"][chosen] - h1_s["full"][h1_best]
    d_test = mh_s["test"][chosen] - h1_s["test"][h1_best]
    for family, rank, s in (("H1", h1_rank, h1_s), ("multihop", mh_rank, mh_s)):
        for budget in range(1, len(rank) + 1):
            sel = pick(rank, budget, len(rank))
            curve_rows.append([dataset, family, budget, draws, float(s["full"][sel].mean()),
                               float(s["test"][sel].mean()), float(s["val"][sel].mean())])
    return [dataset, "available" if m == b else "multihop_fewer_than_b", b, len(mh), draws,
            float(d_full.mean()), float(np.quantile(d_full, .05)), float(np.quantile(d_full, .5)),
            float(np.quantile(d_full, .95)), float((d_full > 0).mean()), float(d_test.mean()),
            float((d_test > 0).mean())]


def _shared_policy_lodo(lock, configs, sealed, rep_dir) -> dict:
    """Shared path policy = (hop_variant, q) applied to each dataset's H1-best backbone."""
    coverage: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    eligible = []
    for dataset, info in lock["datasets"].items():
        h1 = info.get("selected_h1")
        if h1 is None:
            continue
        eligible.append(dataset)
        by_hash = {e["eff_hash"]: e for e in configs.get(dataset, [])}
        for e in configs.get(dataset, []):
            if e["family"] != "multihop":
                continue
            base = dict(e["config"])
            policy = (base["hop_variant"], float(base.get("high_hop_probability_multiplier", 1.0)))
            base["hop_variant"], base["high_hop_probability_multiplier"] = "H1", 1.0
            if base == h1["config"] and h1["eff_hash"] in by_hash:
                coverage[policy][dataset] = (e, by_hash[h1["eff_hash"]])
    policies = sorted(coverage)
    rows = [["policy_hop_variant", "policy_q", "n_datasets_covered", "n_eligible"] +
            sorted(eligible)]
    for p in policies:
        rows.append([p[0], p[1], len(coverage[p]), len(eligible)] +
                    [int(d in coverage[p]) for d in sorted(eligible)])
    _write_csv(rep_dir / "shared_policy_coverage.csv", rows[0], rows[1:])
    full_cover = [p for p in policies if len(coverage[p]) == len(eligible)]
    if not full_cover or len(eligible) < 3:
        result = {"status": "unavailable",
                  "reason": "no shared (hop_variant,q) policy on the H1-best backbone is complete for every "
                            "eligible dataset; no interpolation performed",
                  "n_eligible": len(eligible), "n_policies": len(policies),
                  "max_coverage": max((len(coverage[p]) for p in policies), default=0),
                  "grouping": "dataset-level (independent study groups are not recorded)"}
        atomic_json(rep_dir / "shared_policy_lodo.json", result)
        return result
    delta = {}
    for p in full_cover:
        for d, (e, h1) in coverage[p].items():
            s_e, s_h = _entry_scores(e, sealed, d), _entry_scores(h1, sealed, d)
            delta[(p, d)] = {m: float(np.mean(s_e[m]) - np.mean(s_h[m])) for m in ("val", "test", "full")}
    heldout = []
    for d in sorted(eligible):
        others = [x for x in eligible if x != d]
        chosen = sorted(full_cover, key=lambda p: (-np.mean([delta[(p, o)]["val"] for o in others]),
                                                    resolve_path_variant(p[0]), p[1]))[0]
        heldout.append({"heldout": d, "policy": list(chosen), **{f"delta_{m}": delta[(chosen, d)][m]
                                                                for m in ("val", "test", "full")}})
    result = {"status": "completed_exploratory", "policies_with_full_coverage": [list(p) for p in full_cover],
              "grouping": "dataset-level (independent study groups are not recorded)",
              "note": "policy chosen on other datasets' validation deltas only; exploratory transfer estimate",
              "rows": heldout}
    atomic_json(rep_dir / "shared_policy_lodo.json", json_safe(result))
    return result


def _write_csv(path: Path, header, rows) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--readout-csv", default=str(READOUT_CSV))
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.add_argument("--expected-rows", type=int, default=54030)
    s.add_argument("--expected-embeddings", type=int, default=5403)
    s.add_argument("--expected-datasets", type=int, default=60)
    s.add_argument("--allow-audit-findings", action="store_true",
                   help="lock despite findings (they are recorded in the lock)")
    s.add_argument("--hash-workers", type=int, default=8)
    r = sub.add_parser("report")
    r.add_argument("--readout-csv", default=str(READOUT_CSV))
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--budget-draws", type=int, default=1000)
    args = parser.parse_args()
    (select if args.command == "select" else report)(args)


if __name__ == "__main__":
    main()
