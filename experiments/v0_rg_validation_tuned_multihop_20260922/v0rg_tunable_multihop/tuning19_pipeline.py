"""Build and summarize the staged 19-dataset validation-only tuning run."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


# 本轮沿用冻结的 V0_RG 基础配置，只搜索多跳范围、采样倾向和扰动幅度。
# 所有阶段均用这三个 seed 的 validation ARI 进行选型。
SEEDS = (42, 123, 7)
# H 为最大跳数，B 为搜索宽度；q 控制高阶采样倾向，a 控制邻域混合幅度。
HOPS = (2, 3, 4, 5, 6, 8)
WIDTHS = (1, 2, 4)
Q_VALUES = (0.25, 0.5, 1.0, 2.0)
A_VALUES = (0.5, 0.75, 1.0, 1.25, 1.5)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _token(value: float) -> str:
    return (f"{value:g}").replace(".", "p")


def config_id(hop_variant: str, q: float, amplitude: float) -> str:
    return f"{hop_variant}_q{_token(q)}_a{_token(amplitude)}"


def _variant_rank(hop_variant: str) -> tuple[int, int]:
    if hop_variant == "H1":
        return (1, 1)
    head, width = hop_variant.split("_B")
    return int(head[1:]), int(width)


# 从上一轮验证结果中选出平均多跳增益为正的数据集，属于验证驱动的案例筛选。
def _positive_datasets(old_root: Path) -> list[str]:
    summary = json.loads((old_root / "rank1_validation_summary.json").read_text(encoding="utf-8"))
    return sorted(row["dataset"] for row in summary["rows"] if float(row["delta_ari"]) > 0.0)


# 从旧任务读取 H1 基础配置；此处不重新搜索学习率、gate 等基础参数。
def _base_tasks(old_root: Path) -> dict[tuple[str, int], dict]:
    manifest = json.loads((old_root / "rank1_tasks.json").read_text(encoding="utf-8"))
    return {
        (task["dataset"], int(task["seed"])): task
        for task in manifest["tasks"]
        if task["hop_variant"] == "H1"
    }


# 复制基础任务，只替换多跳参数，并保持数据集与 seed 配对。
def _make_task(template: dict, hop_variant: str, q: float, amplitude: float) -> dict:
    task = copy.deepcopy(template)
    task["hop_variant"] = hop_variant
    task["config_id"] = config_id(hop_variant, q, amplitude)
    task["config"]["protocol_id"] = "v0_rg_multihop_tuning19_validation_v1"
    task["config"]["hop_variant"] = hop_variant
    task["config"]["high_hop_probability_multiplier"] = float(q)
    task["config"]["mix_amplitude_multiplier"] = float(amplitude)
    return task


# Stage A：补测第 6、8 跳以及 B=1/2/4；q=a=1，用于探查更远邻域。
def build_stage_a(old_root: Path, output: Path) -> dict:
    datasets = _positive_datasets(old_root)
    templates = _base_tasks(old_root)
    tasks = []
    for dataset in datasets:
        for hop in (6, 8):
            for width in WIDTHS:
                variant = f"H{hop}_B{width}"
                for seed in SEEDS:
                    tasks.append(_make_task(templates[(dataset, seed)], variant, 1.0, 1.0))
    manifest = {
        "protocol": "validation_only_stage_a_extended_range",
        "datasets": datasets,
        "seeds": list(SEEDS),
        "search": {"max_hop": [6, 8], "breadth": list(WIDTHS), "q": [1.0], "amplitude": [1.0]},
        "tasks": tasks,
    }
    _write_json(output, manifest)
    return manifest


def _record_paths(root: Path) -> list[Path]:
    paths = list((root / "runs").glob("*/*/seed_*/attempt_*/record_seed_*.json"))
    paths.extend((root / "pilot_queue" / "runs").glob("*/*/seed_*/attempt_*/record_seed_*.json"))
    paths.extend((root / "precompleted").glob("*/*/seed_*/record_seed_*.json"))
    return paths


# 汇集已有记录；同一配置和 seed 若有重试结果，则保留最新记录。
def load_records(roots: list[Path]) -> dict[tuple[str, str, int], dict]:
    records: dict[tuple[str, str, int], tuple[float, dict]] = {}
    for root in roots:
        for path in _record_paths(root):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("test_labels_loaded") is not False:
                raise ValueError(f"record accessed test labels: {path}")
            config = record["config"]
            hop = config["hop_variant"]
            q = float(config.get("high_hop_probability_multiplier", 1.0))
            amplitude = float(config.get("mix_amplitude_multiplier", 1.0))
            identity = config_id(hop, q, amplitude)
            dataset = path.parts[-5] if "attempt_" in path.parts[-2] else path.parts[-4]
            seed = int(record["seed"])
            key = (dataset, identity, seed)
            item = (path.stat().st_mtime, record)
            if key not in records or item[0] > records[key][0]:
                records[key] = item
    return {key: item[1] for key, item in records.items()}


# 审计任务是否完成，并列出缺失与失败项以便队列恢复。
def audit_manifest(manifest_path: Path, output_root: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = []
    completed = 0
    exhausted = 0
    for task in manifest["tasks"]:
        dataset = task["dataset"]
        identity = task.get("config_id", task["hop_variant"])
        seed = int(task["seed"])
        seed_root = output_root / "runs" / dataset / identity / f"seed_{seed}"
        if list(seed_root.glob(f"attempt_*/record_seed_{seed}.json")):
            completed += 1
            continue
        failures = list(seed_root.glob("attempt_*/failed.json"))
        exhausted += int(bool(failures))
        missing.append({"dataset": dataset, "config_id": identity, "seed": seed,
                        "failed_attempts": len(failures)})
    payload = {
        "manifest": str(manifest_path),
        "planned": len(manifest["tasks"]),
        "completed": completed,
        "missing": len(missing),
        "tasks_with_failures": exhausted,
        "missing_tasks": missing,
    }
    _write_json(output, payload)
    return payload


# 仅对三个 seed 全部完成的配置计算平均 validation ARI。
def _complete_means(records: dict[tuple[str, str, int], dict], dataset: str) -> dict[str, float]:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for (name, identity, seed), record in records.items():
        if name == dataset:
            grouped[identity][seed] = float(record["validation_metrics"]["ari"])
    return {
        identity: mean(values[seed] for seed in SEEDS)
        for identity, values in grouped.items()
        if set(values) == set(SEEDS)
    }


def _checkpoint_path(old_root: Path, new_root: Path, dataset: str, identity: str, seed: int) -> str | None:
    variant = identity.split("_q", 1)[0]
    candidates = [
        new_root / "stage_c" / "runs" / dataset / identity / f"seed_{seed}",
        new_root / "stage_b" / "runs" / dataset / identity / f"seed_{seed}",
        new_root / "stage_a" / "runs" / dataset / identity / f"seed_{seed}",
        old_root / "pilot_queue" / "runs" / dataset / variant / f"seed_{seed}",
        old_root / "precompleted" / dataset / variant / f"seed_{seed}",
    ]
    for seed_root in candidates:
        direct = seed_root / f"model_seed_{seed}.pt"
        if direct.exists():
            return str(direct.resolve())
        models = sorted(seed_root.glob(f"attempt_*/model_seed_{seed}.pt"))
        if models:
            return str(models[-1].resolve())
    return None


# 每个跳数先保留最佳宽度，再按验证集表现选三个范围进入下一阶段。
def _select_three_ranges(means: dict[str, float]) -> list[str]:
    best_per_hop: dict[int, tuple[str, float]] = {}
    for identity, score in means.items():
        hop_variant = identity.split("_q", 1)[0]
        if hop_variant == "H1" or "_q1_a1" not in identity:
            continue
        hop, width = _variant_rank(hop_variant)
        current = best_per_hop.get(hop)
        if current is None or score > current[1] + 1e-12 or (
            abs(score - current[1]) <= 0.001 and width < _variant_rank(current[0].split("_q", 1)[0])[1]
        ):
            best_per_hop[hop] = (identity, score)
    ranked = sorted(best_per_hop.values(), key=lambda item: (-item[1], _variant_rank(item[0].split("_q", 1)[0])))
    return [identity.split("_q", 1)[0] for identity, _ in ranked[:3]]


# Stage B：分别扫描 q 与 a，不穷举全部组合；H1 也扫描 a 作为调参对照。
def build_stage_b(old_root: Path, new_root: Path, output: Path) -> dict:
    datasets = _positive_datasets(old_root)
    templates = _base_tasks(old_root)
    records = load_records([old_root, new_root / "stage_a"])
    selected_ranges = {}
    tasks = []
    for dataset in datasets:
        ranges = _select_three_ranges(_complete_means(records, dataset))
        selected_ranges[dataset] = ranges
        for variant in ranges:
            for q in (0.25, 0.5, 2.0):
                for seed in SEEDS:
                    tasks.append(_make_task(templates[(dataset, seed)], variant, q, 1.0))
            for amplitude in (0.5, 0.75, 1.25, 1.5):
                for seed in SEEDS:
                    tasks.append(_make_task(templates[(dataset, seed)], variant, 1.0, amplitude))
        for amplitude in (0.5, 0.75, 1.25, 1.5):
            for seed in SEEDS:
                tasks.append(_make_task(templates[(dataset, seed)], "H1", 1.0, amplitude))
    manifest = {
        "protocol": "validation_only_stage_b_separate_strength",
        "datasets": datasets,
        "seeds": list(SEEDS),
        "selected_ranges": selected_ranges,
        "search": {"q": list(Q_VALUES), "amplitude": list(A_VALUES)},
        "tasks": tasks,
    }
    _write_json(output, manifest)
    return manifest


# Stage C：组合前阶段分别选出的 q、a；已有结果的组合不重复训练。
def build_stage_c(old_root: Path, new_root: Path, stage_b_manifest: Path, output: Path) -> dict:
    stage_b = json.loads(stage_b_manifest.read_text(encoding="utf-8"))
    templates = _base_tasks(old_root)
    records = load_records([old_root, new_root / "stage_a", new_root / "stage_b"])
    tasks = []
    combinations = {}
    for dataset, ranges in stage_b["selected_ranges"].items():
        means = _complete_means(records, dataset)
        combinations[dataset] = []
        for variant in ranges:
            q_scores = [(q, means.get(config_id(variant, q, 1.0))) for q in Q_VALUES]
            a_scores = [(a, means.get(config_id(variant, 1.0, a))) for a in A_VALUES]
            q_scores = [(q, score) for q, score in q_scores if score is not None]
            a_scores = [(a, score) for a, score in a_scores if score is not None]
            if not q_scores or not a_scores:
                continue
            best_q = max(q_scores, key=lambda item: (item[1], -abs(item[0] - 1.0)))[0]
            best_a = max(a_scores, key=lambda item: (item[1], -abs(item[0] - 1.0)))[0]
            identity = config_id(variant, best_q, best_a)
            combinations[dataset].append(identity)
            if best_q == 1.0 or best_a == 1.0:
                continue
            for seed in SEEDS:
                tasks.append(_make_task(templates[(dataset, seed)], variant, best_q, best_a))
    manifest = {
        "protocol": "validation_only_stage_c_joint_strength",
        "datasets": stage_b["datasets"],
        "seeds": list(SEEDS),
        "selected_combinations": combinations,
        "tasks": tasks,
    }
    _write_json(output, manifest)
    return manifest


# 依据平均 validation ARI 选多跳配置，并与最佳调幅 H1 做同 seed 配对。
# 汇总分数仍用于选型，不能视作独立测试结果。
def summarize(old_root: Path, new_root: Path, output_json: Path, output_csv: Path) -> dict:
    datasets = _positive_datasets(old_root)
    records = load_records([old_root, new_root / "stage_a", new_root / "stage_b", new_root / "stage_c"])
    rows = []
    for dataset in datasets:
        means = _complete_means(records, dataset)
        if not means:
            continue
        best_score = max(means.values())
        tied = [identity for identity, score in means.items() if best_score - score <= 0.001]
        def tie_key(identity: str) -> tuple:
            variant, tail = identity.split("_q", 1)
            q_token, a_token = tail.split("_a", 1)
            q = float(q_token.replace("p", "."))
            amplitude = float(a_token.replace("p", "."))
            hop, width = _variant_rank(variant)
            return hop, width, abs(q - 1.0), abs(amplitude - 1.0)
        chosen = min(tied, key=tie_key)
        chosen_records = [records[(dataset, chosen, seed)] for seed in SEEDS]
        scores = [float(record["validation_metrics"]["ari"]) for record in chosen_records]
        h1_candidates = {key: value for key, value in means.items() if key.startswith("H1_q")}
        best_h1_score = max(h1_candidates.values())
        best_h1 = min([key for key, value in h1_candidates.items() if best_h1_score - value <= 0.001], key=tie_key)
        h1_scores = [float(records[(dataset, best_h1, seed)]["validation_metrics"]["ari"]) for seed in SEEDS]
        deltas = [score - baseline for score, baseline in zip(scores, h1_scores)]
        record_config = chosen_records[0]["config"]
        rows.append({
            "dataset": dataset,
            "best_config_id": chosen,
            "hop_variant": record_config["hop_variant"],
            "high_hop_probability_multiplier": float(record_config.get("high_hop_probability_multiplier", 1.0)),
            "mix_amplitude_multiplier": float(record_config.get("mix_amplitude_multiplier", 1.0)),
            "validation_ari_mean": mean(scores),
            "validation_ari_std": pstdev(scores),
            "best_tuned_h1_config_id": best_h1,
            "tuned_h1_validation_ari_mean": mean(h1_scores),
            "paired_delta_mean": mean(deltas),
            "positive_seed_count": sum(delta > 0 for delta in deltas),
            "seed_deltas": {str(seed): delta for seed, delta in zip(SEEDS, deltas)},
            "complete_config_count": len(means),
            "checkpoint_paths": {
                str(seed): _checkpoint_path(old_root, new_root, dataset, chosen, seed)
                for seed in SEEDS
            },
        })
    payload = {
        "protocol": "validation_only_19_dataset_multihop_tuning_v1",
        "test_labels_loaded": False,
        "datasets_planned": len(datasets),
        "datasets_summarized": len(rows),
        "rows": sorted(rows, key=lambda row: row["paired_delta_mean"], reverse=True),
    }
    _write_json(output_json, payload)
    fields = [key for key in rows[0] if key not in {"seed_deltas", "checkpoint_paths"}] if rows else []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({field: row[field] for field in fields})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build-a", "build-b", "build-c"):
        command = sub.add_parser(name)
        command.add_argument("--old-root", required=True)
        command.add_argument("--new-root", required=True)
        command.add_argument("--output", required=True)
        if name == "build-c":
            command.add_argument("--stage-b-manifest", required=True)
    report = sub.add_parser("summarize")
    report.add_argument("--old-root", required=True)
    report.add_argument("--new-root", required=True)
    report.add_argument("--json-output", required=True)
    report.add_argument("--csv-output", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--output-root", required=True)
    audit.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_manifest(Path(args.manifest), Path(args.output_root), Path(args.output))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    old_root, new_root = Path(args.old_root), Path(args.new_root)
    if args.command == "build-a":
        result = build_stage_a(old_root, Path(args.output))
    elif args.command == "build-b":
        result = build_stage_b(old_root, new_root, Path(args.output))
    elif args.command == "build-c":
        result = build_stage_c(old_root, new_root, Path(args.stage_b_manifest), Path(args.output))
    else:
        result = summarize(old_root, new_root, Path(args.json_output), Path(args.csv_output))
    print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
