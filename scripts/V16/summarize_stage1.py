from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _runs(root: Path) -> list[dict]:
    rows = []
    for path in root.glob("**/summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_path"] = str(path.parent)
        rows.append(payload)
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


NAMES = ["self_only", "fixed_predictive_graph", "V16_predictive_gate", "shuffled_support", "output_disabled"]


def _condition_metrics(variants: dict[str, list[dict]]) -> tuple[dict[str, list[float]], set[int]]:
    by_seed = {
        name: {
            int(item["seed"]): float(item["metrics"]["ari"])
            for item in variants.get(name, [])
            if item.get("metrics", {}).get("ari") is not None
        }
        for name in NAMES
    }
    common = set.intersection(*(set(values) for values in by_seed.values())) if by_seed else set()
    metrics = {name: [by_seed[name][seed] for seed in sorted(common)] for name in NAMES}
    return metrics, common


def classify(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        condition = str(row.get("run_metadata", {}).get("condition", row.get("condition", "clean")))
        grouped[str(row["dataset"])][condition][str(row["variant"])].append(row)
    output = []
    for dataset, conditions in sorted(grouped.items()):
        first_row = next(iter(next(iter(conditions.values())).values()))[0]
        certificate = first_row.get("theory_certificate", {})
        if first_row.get("status") == "theory_domain_not_supported" or certificate.get("theory_domain") != "candidate":
            output.append({"dataset": dataset, "status": "theory_domain_not_supported", "reason": ";".join(certificate.get("domain_reasons", []))})
            continue
        clean_metrics, clean_seeds = _condition_metrics(conditions.get("clean", {}))
        stress_metrics, stress_seeds = _condition_metrics(conditions.get("compound", {}))
        if any(not clean_metrics.get(name) for name in NAMES) or len(clean_seeds) < 3 or any(not stress_metrics.get(name) for name in NAMES) or len(stress_seeds) < 3:
            status = "incomplete"
            reason = "missing_clean_or_compound_five_way_pair_or_three_seeds"
            clean_delta_mean = stress_delta_mean = stress_retention = None
        else:
            clean_delta = [a - b for a, b in zip(clean_metrics["V16_predictive_gate"], clean_metrics["self_only"])]
            clean_fixed = [a - b for a, b in zip(clean_metrics["V16_predictive_gate"], clean_metrics["fixed_predictive_graph"])]
            clean_shuffle = [a - b for a, b in zip(clean_metrics["V16_predictive_gate"], clean_metrics["shuffled_support"])]
            clean_output = [a - b for a, b in zip(clean_metrics["V16_predictive_gate"], clean_metrics["output_disabled"])]
            stress_delta = [a - b for a, b in zip(stress_metrics["V16_predictive_gate"], stress_metrics["self_only"])]
            clean_delta_mean = _mean(clean_delta)
            stress_delta_mean = _mean(stress_delta)
            stress_retention = (
                stress_delta_mean / clean_delta_mean
                if clean_delta_mean is not None and clean_delta_mean > 0.0 and stress_delta_mean is not None
                else None
            )
            passed = (
                (clean_delta_mean or -float("inf")) >= 0.03
                and (_mean(clean_fixed) or -float("inf")) > 0.0
                and (_mean(clean_shuffle) or -float("inf")) > 0.0
                and (_mean(clean_output) or -float("inf")) > 0.0
                and sum(value > 0.0 for value in clean_delta) >= 2
                and (stress_retention is not None and stress_retention >= 0.50)
            )
            status = "candidate_positive" if passed else "empirical_not_supported"
            reason = "fixed_preregistered_clean_and_compound_promotion_rule"
        output.append(
            {
                "dataset": dataset,
                "status": status,
                "reason": reason,
                "clean_self_ari_mean": _mean(clean_metrics["self_only"]),
                "clean_fixed_ari_mean": _mean(clean_metrics["fixed_predictive_graph"]),
                "clean_v16_ari_mean": _mean(clean_metrics["V16_predictive_gate"]),
                "clean_shuffled_ari_mean": _mean(clean_metrics["shuffled_support"]),
                "clean_output_disabled_ari_mean": _mean(clean_metrics["output_disabled"]),
                "clean_delta_self_mean": clean_delta_mean,
                "compound_delta_self_mean": stress_delta_mean,
                "compound_retention": stress_retention,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="V16 fixed Stage-1 promotion summary")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = classify(_runs(Path(args.root)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    if rows:
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
