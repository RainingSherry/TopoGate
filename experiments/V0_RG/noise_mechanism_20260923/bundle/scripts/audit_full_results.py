"""File-level coverage and provenance audit for the registered full four-case run."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import condition_slug, sha256, atomic_json


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RESULTS = Path(CONFIG["output_root"])
INVENTORY = json.loads((RESULTS / "inventories/checkpoints.json").read_text(encoding="utf-8"))


def run():
    if INVENTORY.get("config_sha256") != sha256(CONFIG_PATH):
        raise ValueError("Inventory and registered config have different SHA256 values")
    rows = [r for r in INVENTORY["rows"] if r["status"] == "ready"]
    if len(rows) != 42:
        raise ValueError("Expected the 42 audited effective/reference states")
    variants = Counter(r["variant"] for r in rows)
    runtime = {name: sha256(ROOT / "scripts" / name) for name in ("common.py", "metrics.py", "probe.py")}
    repo = Path(CONFIG["remote_repo"])
    model_sources = {str(p.relative_to(repo)): sha256(p)
                     for p in sorted((repo / "methods/TopoGate/V0_RG").rglob("*.py"))}
    report = {"schema": "four_step_file_audit_v1", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "expected": {"observe": 42 * len(CONFIG["conditions"]) * len(CONFIG["noise_seeds"]),
                           "directions": 42 * 3 * len(CONFIG["direction_multipliers"])},
              "variant_coverage": dict(variants), "actual": {"observe": Counter(), "directions": Counter()},
              "issues": [], "sample_issue_limit": 1000,
              "pairing": {"observe_passed": 0, "observe_failed": 0,
                          "directions_passed": 0, "directions_failed": 0}}
    records = {}
    for row in rows:
        ds, variant, seed = row["dataset"], row["variant"], int(row["model_seed"])
        phases = []
        phases += [("observe", c["corruption"], c["severity"], noise)
                   for c in CONFIG["conditions"] for noise in CONFIG["noise_seeds"]]
        phases += [("directions", "direction_" + direction, multiplier, CONFIG["direction_seed"])
                   for multiplier in CONFIG["direction_multipliers"]
                   for direction in ("native", "random", "shuffled")]
        for stage, kind, severity, noise in phases:
            key = (ds, variant, seed, stage, kind, float(severity), int(noise))
            path = (RESULTS / ds / stage / variant / ("model_seed_" + str(seed)) /
                    condition_slug(kind, severity, noise) / "result.json")
            if not path.is_file():
                report["actual"][stage]["missing"] += 1
                if len(report["issues"]) < report["sample_issue_limit"]:
                    report["issues"].append({"path": str(path), "reason": "missing_result"})
                continue
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("status") != "completed":
                    raise ValueError("result_status_" + str(item.get("status")))
                if item.get("training_performed") is not False:
                    raise ValueError("training_performed_not_false")
                if item.get("clean_replay", {}).get("status") != "passed":
                    raise ValueError("clean_replay_not_passed")
                if not path.with_name("samples.npz").is_file() or path.with_name("samples.npz").stat().st_size == 0:
                    raise ValueError("samples_npz_missing_or_empty")
                p = item["provenance"]
                if item.get("protocol") != CONFIG["protocol"] or p.get("config_sha256") != sha256(CONFIG_PATH):
                    raise ValueError("protocol_or_config_mismatch")
                expected_files = row.get("files", {})
                for fname, field in (("model.pt", "model_sha256"),
                                     ("preprocessor.npz", "preprocessor_sha256"),
                                     ("cluster_centers.npy", "readout_sha256"),
                                     ("test_indices.npy", "test_indices_sha256")):
                    expected = expected_files.get(fname, {}).get("sha256")
                    if not expected or p.get(field) != expected:
                        raise ValueError(field + "_mismatch")
                if p.get("code_sha256") != runtime or p.get("v0rg_source_sha256") != model_sources:
                    raise ValueError("inference_or_V0RG_source_changed")
                if (item.get("dataset"), item.get("variant"), int(item.get("model_seed")),
                    item.get("stage"), item.get("corruption"), float(item.get("severity")),
                    int(item.get("noise_seed"))) != key:
                    raise ValueError("result_identity_mismatch")
                records[key] = item
                report["actual"][stage]["completed_validated"] += 1
            except Exception as exc:
                report["actual"][stage]["failed_validation"] += 1
                if len(report["issues"]) < report["sample_issue_limit"]:
                    report["issues"].append({"path": str(path), "reason": type(exc).__name__ + ": " + str(exc)})
    for key, ablated in records.items():
        ds, variant, seed, stage, kind, severity, noise = key
        if variant == "full":
            continue
        full = records.get((ds, "full", seed, stage, kind, severity, noise))
        test = ["preprocessor_sha256", "test_indices_sha256", "source_data_sha256"]
        if stage == "observe":
            test.append("raw_input_sha256")
        else:
            test += ["model_input_sha256", "direction_sha256", "common_input_sha256",
                     "common_valid_mask_sha256"]
        # The direction hash and common-mask hash live in the direction metadata;
        # model-input and clean artifact hashes live in provenance.
        def field(obj, name):
            return obj.get("provenance", {}).get(name, obj.get("direction", {}).get(name))
        ok = full is not None and all(field(full, f) is not None and field(full, f) == field(ablated, f) for f in test)
        if stage == "directions":
            ok = ok and full.get("prediction", {}).get("common_valid", {}).get("noisy_ari") is not None
            ok = ok and ablated.get("prediction", {}).get("common_valid", {}).get("noisy_ari") is not None
        report["pairing"][stage + ("_passed" if ok else "_failed")] += 1
        if not ok and len(report["issues"]) < report["sample_issue_limit"]:
            report["issues"].append({"key": key, "reason": "full_ablation_pairing_mismatch"})
    report["actual"] = {name: dict(counter) for name, counter in report["actual"].items()}
    report["status"] = "complete" if (
        all(report["actual"][name].get("completed_validated", 0) == count for name, count in report["expected"].items())
        and report["pairing"]["observe_failed"] == 0 and report["pairing"]["directions_failed"] == 0
    ) else "incomplete_or_exclusions"
    atomic_json(RESULTS / "FILE_AUDIT.json", report)
    print(json.dumps({k: report[k] for k in ("expected", "actual", "pairing", "status")}, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(run())
