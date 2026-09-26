"""Aggregate frozen probes, preserving model/noise replication and audited pairing."""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten(value, prefix=""):
    out = {}
    for key, val in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            out.update(flatten(val, name))
        else:
            out[name] = val
    return out


def sd(values):
    return st.stdev(values) if len(values) > 1 else None


KEYS = ("protocol", "stage", "dataset", "variant", "corruption", "severity")
DATASETS = ("Mouse_retina", "dermatology", "har", "micro-mass")


def probe_paths(root):
    """Read canonical case paths only; never merge a nested validation run.

    Passing a smoke run itself as root still works because its immediate
    children are the same four case directories.
    """
    root = Path(root)
    for dataset in DATASETS:
        for stage in ("observe", "directions"):
            folder = root / dataset / stage
            if folder.is_dir():
                yield from sorted(folder.glob("*/model_seed_*/*/result.json"))


def aggregate(rows, value_key="value"):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in KEYS) + (row["metric"], row["model_seed"])].append(row)
    seeds = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        values = [float(r[value_key]) for r in items]
        seeds.append(dict(zip(KEYS, key[:len(KEYS)]), metric=key[-2], model_seed=key[-1],
                          mean=st.mean(values), noise_sd=sd(values), n_noise=len(values),
                          noise_seeds=";".join(str(r["noise_seed"]) for r in sorted(items, key=lambda r: r["noise_seed"]))))
    groups = defaultdict(list)
    for row in seeds:
        groups[tuple(row[k] for k in KEYS) + (row["metric"],)].append(row)
    summary = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        values = [r["mean"] for r in items]
        noise_sds = [r["noise_sd"] for r in items if r["noise_sd"] is not None]
        summary.append(dict(zip(KEYS, key[:len(KEYS)]), metric=key[-1], mean=st.mean(values),
                            model_seed_sd=sd(values), mean_within_model_noise_sd=st.mean(noise_sds) if noise_sds else None,
                            n_model_seeds=len(items), n_noise_min=min(r["n_noise"] for r in items),
                            n_noise_max=max(r["n_noise"] for r in items),
                            model_seeds=";".join(str(r["model_seed"]) for r in items),
                            equal_noise_seed_coverage=len({r["noise_seeds"] for r in items}) == 1))
    return seeds, summary


def find_field(flat, leaf):
    found = [v for k, v in flat.items() if k == leaf or k.endswith("." + leaf)]
    return found[0] if found and all(v == found[0] for v in found) else None


def ari_delta(record):
    pred = record.get("prediction", {})
    if record.get("stage") == "directions":
        common = pred.get("common_valid", {})
        clean, noisy = common.get("clean_ari"), common.get("noisy_ari")
        if clean is None or noisy is None:
            return None
        return float(noisy) - float(clean)
    for key in ("delta_ari", "ari_delta", "delta_noisy_clean", "ari_delta_noisy_clean"):
        if key in pred:
            return float(pred[key])
    clean = next((pred[k] for k in ("clean_ari", "ari_clean") if k in pred), None)
    noisy = next((pred[k] for k in ("noisy_ari", "ari_noisy") if k in pred), None)
    return float(noisy) - float(clean) if clean is not None and noisy is not None else None


def run(results, out, inventory=None):
    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inventory = Path(inventory) if inventory else results / "inventories" / "checkpoints.json"
    inv = read_json(inventory) if inventory.exists() else {"rows": []}
    inv_rows = inv.get("rows", []) if isinstance(inv, dict) else inv
    audits = {(r.get("dataset"), str(r.get("model_seed")), r.get("variant")): r for r in inv_rows}
    records, metrics, rejected, candidates = {}, [], [], []
    for path in probe_paths(results):
        try:
            r = read_json(path)
            if r.get("status") not in ("ok", "success", "complete", "completed"):
                rejected.append({"path": str(path), "reason": "unsuccessful_status", "status": r.get("status")})
                continue
            meta = {k: r.get(k, "") for k in KEYS}
            meta["stage"] = r.get("stage", "observe")
            meta.update(model_seed=int(r["model_seed"]), noise_seed=int(r["noise_seed"]))
            location = path.relative_to(results).parts
            expected = (meta["dataset"], meta["stage"], meta["variant"], f"model_seed_{meta['model_seed']}")
            if tuple(location[:4]) != expected:
                raise ValueError("Probe metadata disagrees with canonical dataset/stage/variant/model-seed path")
            prov = r.get("provenance", {})
            audit = audits.get((meta["dataset"], str(meta["model_seed"]), meta["variant"]))
            if not audit or audit.get("status") not in ("ready", "no_op_control"):
                raise ValueError("inventory_record_missing_or_not_ready")
            inventory_config = inv.get("config_sha256") if isinstance(inv, dict) else None
            if not inventory_config or prov.get("config_sha256") != inventory_config:
                raise ValueError("inventory_config_mismatch_or_missing")
            inventory_protocol = inv.get("protocol") if isinstance(inv, dict) else None
            if not meta["protocol"] or meta["protocol"] != inventory_protocol or prov.get("protocol") != meta["protocol"]:
                raise ValueError("inventory_or_provenance_protocol_mismatch")
            expected_model = audit.get("files", {}).get("model.pt", {}).get("sha256")
            if not expected_model or expected_model != prov.get("model_sha256"):
                raise ValueError("inventory_checkpoint_mismatch_or_missing")
            if r.get("clean_replay", {}).get("status") != "passed":
                raise ValueError("clean_replay_not_passed")
            code = prov.get("code_sha256")
            if not isinstance(code, dict) or not code or not all(isinstance(v, str) and v for v in code.values()):
                raise ValueError("recorded_code_sha256_missing_or_invalid")
            source_code = prov.get("v0rg_source_sha256")
            if source_code is not None and (not isinstance(source_code, dict) or not source_code
                                           or not all(isinstance(v, str) and v for v in source_code.values())):
                raise ValueError("recorded_v0rg_source_sha256_invalid")
            identity = tuple(meta[k] for k in KEYS) + (meta["model_seed"], meta["noise_seed"])
            group = (meta["protocol"], inventory_config, meta["stage"])
            candidates.append((identity, r, meta, path, group, json.dumps({"scripts": code, "v0rg_source": source_code}, sort_keys=True)))
        except (ValueError, KeyError, TypeError) as exc:
            # Duplicate identities must never be silently counted twice.
            if "Duplicate probe identity" in str(exc):
                raise
            rejected.append({"path": str(path), "reason": str(exc)})
    code_groups = defaultdict(set)
    for _, _, _, _, group, signature in candidates:
        code_groups[group].add(signature)
    for identity, r, meta, path, group, _ in candidates:
        if len(code_groups[group]) != 1:
            rejected.append({"path": str(path), "reason": "mixed_recorded_code_or_v0rg_source_sha256_in_protocol_config_stage_group",
                             "group": list(group), "distinct_code_snapshots": len(code_groups[group])})
            continue
        if identity in records:
            raise ValueError(f"Duplicate probe identity: {identity}")
        records[identity] = (r, meta, path)
        flat_metrics = flatten({s: r.get(s, {}) for s in ("input", "representation", "prediction")})
        clean_macro = flat_metrics.get("representation.graph.clean_purity_macro")
        noisy_macro = flat_metrics.get("representation.graph.noisy_purity_macro")
        if clean_macro is not None and noisy_macro is not None:
            flat_metrics.setdefault("representation.graph.purity_macro_change", noisy_macro - clean_macro)
        for key, val in flat_metrics.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                metrics.append(dict(meta, metric=key, value=val, source=str(path)))
    pairs, pair_metrics, pair_rejected = [], [], []
    for identity, (r, meta, path) in records.items():
        if meta["variant"] == "full":
            continue
        full_id = list(identity)
        full_id[KEYS.index("variant")] = "full"
        base = records.get(tuple(full_id))
        audit = audits.get((meta["dataset"], str(meta["model_seed"]), meta["variant"]))
        reason = None
        if not base:
            reason = "missing_full_with_identical_pair_keys"
        elif audit is None or audit.get("effective_intervention") is not True:
            reason = "no_effect_or_unknown_inventory_verdict"
        elif audit.get("status") != "ready":
            reason = "inventory_control_not_ready"
        else:
            full, _, full_path = base
            fflat, rflat = flatten(full), flatten(r)
            full_audit = audits.get((meta["dataset"], str(meta["model_seed"]), "full"))
            if full_audit is None or full_audit.get("status") != "ready":
                reason = "inventory_full_not_ready"
            for probe, audit_row in ((r, audit), (full, full_audit or {})):
                prov = probe.get("provenance", {})
                inventory_config = inv.get("config_sha256") if isinstance(inv, dict) else None
                expected_model = audit_row.get("files", {}).get("model.pt", {}).get("sha256")
                if not inventory_config or inventory_config != prov.get("config_sha256"):
                    reason = "inventory_config_mismatch_or_missing"
                if not expected_model or expected_model != prov.get("model_sha256"):
                    reason = "inventory_checkpoint_mismatch_or_missing"
                if probe.get("clean_replay", {}).get("status") != "passed":
                    reason = "clean_replay_not_passed"
            fcode, rcode = full.get("provenance", {}).get("code_sha256"), r.get("provenance", {}).get("code_sha256")
            if not fcode or fcode != rcode:
                reason = "missing_or_mismatched_code_sha256"
            required = ["protocol", "config_sha256", "preprocessor_sha256", "test_indices_sha256", "source_data_sha256", "model_input_sha256"]
            required += ["direction_sha256", "common_input_sha256", "common_valid_mask_sha256"] if meta["stage"] == "directions" else ["raw_input_sha256"]
            for field in required:
                if reason:
                    break
                a, b = find_field(fflat, field), find_field(rflat, field)
                if a is None or b is None or a != b:
                    reason = "missing_or_mismatched_" + field
                    break
            if reason is None and (ari_delta(full) is None or ari_delta(r) is None):
                reason = "missing_ari_delta"
        if reason:
            pair_rejected.append(dict(meta, reason=reason, source=str(path), inventory_interpretation=(audit or {}).get("interpretation", "")))
            continue
        delta_full, delta_ablation = ari_delta(full), ari_delta(r)
        contrast = delta_full - delta_ablation
        full_pred = full["prediction"].get("common_valid", {}) if meta["stage"] == "directions" else full["prediction"]
        ablation_pred = r["prediction"].get("common_valid", {}) if meta["stage"] == "directions" else r["prediction"]
        pairs.append(dict(meta, delta_full=delta_full, delta_ablation=delta_ablation,
                          clean_ari_full=full_pred.get("clean_ari"), noisy_ari_full=full_pred.get("noisy_ari"),
                          clean_ari_ablation=ablation_pred.get("clean_ari"), noisy_ari_ablation=ablation_pred.get("noisy_ari"),
                          delta_full_minus_ablation=contrast, effective_intervention=True,
                          inventory_interpretation=audit.get("interpretation", ""),
                          full_source=str(full_path), ablation_source=str(path),
                          model_input_sha256=find_field(fflat, "model_input_sha256"),
                          common_valid_mask_sha256=find_field(fflat, "common_valid_mask_sha256") if meta["stage"] == "directions" else "",
                          test_indices_sha256=find_field(fflat, "test_indices_sha256")))
        pair_metrics.append(dict(meta, metric="delta_full_minus_ablation", value=contrast))
    seed_rows, summary = aggregate(metrics)
    pair_seed_rows, pair_summary = aggregate(pair_metrics)
    for name, rows in (("metrics.csv", metrics), ("seed_summary.csv", seed_rows), ("summary.csv", summary),
                       ("paired_ablations.csv", pairs), ("paired_seed_summary.csv", pair_seed_rows),
                       ("paired_summary.csv", pair_summary), ("pairing_rejections.csv", pair_rejected)):
        write_csv(out / name, rows, fields=None if rows else ["status"])
    manifest = {"results_root": str(results.resolve()), "inventory": str(inventory.resolve()),
                "scan_scope": "Only ROOT/{Mouse_retina,dermatology,har,micro-mass}/{observe,directions}/VARIANT/model_seed_SEED/CONDITION/result.json; nested validation and unrelated outputs are not scanned.",
                "successful_probes": len(records), "paired_comparisons": len(pairs), "rejected_results": rejected,
                "accepted_results": [str(path.resolve()) for _, _, path in records.values()],
                "legacy_unverified_v0rg_source_results": [str(path.resolve()) for record, _, path in records.values() if record.get("provenance", {}).get("v0rg_source_sha256") is None],
                "validation_policy": "Every record must match inventory protocol/config/model SHA and pass clean replay. Recorded runtime-code and V0RG-source dictionaries must be identical within each protocol/config/stage group; every member of a mixed group is rejected. All-missing V0RG-source legacy groups remain accepted but explicitly listed as unverified. Current plotting or analysis file hashes are not consulted.",
                "pairing_rejections": pair_rejected,
                "replication": "Equal-weight model-seed means after averaging noise seeds; model-seed SD and within-model noise SD are separate. No independent n=15 claim.",
                "causal_scope": "Audited configuration contrasts, not mediation or two-factor synergy; no-op controls excluded."}
    (out / "summary_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--inventory")
    args = p.parse_args()
    print(json.dumps(run(args.results, args.out, args.inventory), ensure_ascii=False, indent=2))
