"""Summarize the completed rank-1 validation-only multihop expansion."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


SEEDS = (42, 123, 7)


def rank_key(variant: str) -> tuple[int, int]:
    if variant == "H1":
        return (1, 1)
    head, breadth = variant.split("_B")
    return (int(head[1:]), int(breadth))


def load_records(root: Path) -> dict[tuple[str, str, int], dict]:
    candidates: dict[tuple[str, str, int], tuple[float, dict]] = {}
    patterns = [
        (root / "pilot_queue" / "runs", "*/*/seed_*/attempt_*/record_seed_*.json", 3, 2),
        (root / "precompleted", "*/*/seed_*/record_seed_*.json", 2, 1),
    ]
    for base, pattern, dataset_parent, variant_parent in patterns:
        for path in base.glob(pattern):
            record = json.loads(path.read_text(encoding="utf-8"))
            dataset = path.parents[dataset_parent].name
            variant = path.parents[variant_parent].name
            seed = int(record["seed"])
            key = (dataset, variant, seed)
            item = (path.stat().st_mtime, record)
            if key not in candidates or item[0] > candidates[key][0]:
                candidates[key] = item
    return {key: item[1] for key, item in candidates.items()}


def summarize(root: Path) -> dict:
    manifest = json.loads((root / "rank1_tasks.json").read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    planned = {(task["dataset"], task["hop_variant"], int(task["seed"])) for task in tasks}
    metadata = {}
    for task in tasks:
        metadata[task["dataset"]] = {
            "entry_id": task["entry_id"],
            "panel": task["panel"],
            "archived_rank": task["archived_rank"],
            "archived_stratum": task["archived_stratum"],
        }
    records = load_records(root)
    available = planned.intersection(records)
    missing = sorted(planned - available)
    extra = sorted(set(records) - planned)

    grouped: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for dataset, variant, seed in available:
        record = records[(dataset, variant, seed)]
        if record.get("test_labels_loaded") is not False:
            raise ValueError(f"test labels accessed: {(dataset, variant, seed)}")
        if record.get("selection_label_scope") != "validation_only":
            raise ValueError(f"not validation-only: {(dataset, variant, seed)}")
        grouped[dataset][variant][seed] = float(record["validation_metrics"]["ari"])

    rows = []
    for dataset in sorted(metadata):
        variants = grouped.get(dataset, {})
        complete = all(set(variants.get(variant, {})) == set(SEEDS) for variant in manifest["hop_variants"])
        if not complete:
            rows.append({**metadata[dataset], "dataset": dataset, "complete": False})
            continue
        means = {variant: mean(variants[variant][seed] for seed in SEEDS) for variant in manifest["hop_variants"]}
        candidates = {variant: value for variant, value in means.items() if variant != "H1"}
        best_score = max(candidates.values())
        tied = [variant for variant, value in candidates.items() if best_score - value <= 0.001]
        chosen = min(tied, key=rank_key)
        deltas = [variants[chosen][seed] - variants["H1"][seed] for seed in SEEDS]
        delta = means[chosen] - means["H1"]
        positives = sum(value > 0 for value in deltas)
        equals = sum(abs(value) <= 1e-12 for value in deltas)
        pass_gate = (delta >= 0.01 and positives >= 2) or (delta >= 0.003 and positives == 3)
        rows.append({
            **metadata[dataset],
            "dataset": dataset,
            "complete": True,
            "h1_mean": means["H1"],
            "chosen_variant": chosen,
            "chosen_mean": means[chosen],
            "delta_ari": delta,
            "positive_seeds": positives,
            "equal_seeds": equals,
            "seed_deltas": {str(seed): deltas[index] for index, seed in enumerate(SEEDS)},
            "passes_followup_gate": pass_gate,
            "all_means": means,
        })

    complete_rows = [row for row in rows if row["complete"]]
    variant_counts = Counter(row["chosen_variant"] for row in complete_rows)
    summary = {
        "protocol": "validation_only; no final-test labels loaded",
        "planned_units": len(planned),
        "completed_units": len(available),
        "missing_units": len(missing),
        "extra_non_rank1_units": len(extra),
        "datasets_planned": len(metadata),
        "datasets_complete": len(complete_rows),
        "datasets_positive_delta": sum(row["delta_ari"] > 0 for row in complete_rows),
        "datasets_three_of_three_positive": sum(row["positive_seeds"] == 3 for row in complete_rows),
        "datasets_passing_followup_gate": sum(row["passes_followup_gate"] for row in complete_rows),
        "mean_selected_delta_ari": mean(row["delta_ari"] for row in complete_rows) if complete_rows else None,
        "chosen_variant_counts": dict(sorted(variant_counts.items(), key=lambda item: rank_key(item[0]))),
        "missing": missing,
        "rows": sorted(complete_rows, key=lambda row: row["delta_ari"], reverse=True),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()
    result = summarize(Path(args.root))
    json_path = Path(args.json_output)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "dataset", "panel", "archived_rank", "archived_stratum", "h1_mean", "chosen_variant",
        "chosen_mean", "delta_ari", "positive_seeds", "equal_seeds", "passes_followup_gate",
    ]
    with Path(args.csv_output).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({field: row[field] for field in fields})
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "missing"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
