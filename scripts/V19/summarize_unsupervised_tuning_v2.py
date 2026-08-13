#!/usr/bin/env python
"""Select V19 RG candidates from paired, held-out X-only diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.V19.tune_unsupervised_v2 import (  # noqa: E402
    PROTOCOL_ID,
    _load_manifest,
    _write_json,
    candidate_catalog_for_stage,
    underlying_dataset_id,
)


LOSS_RELATIVE_THRESHOLD = 0.01
STABILITY_THRESHOLD = 0.005
NEIGHBOR_THRESHOLD = 0.01
SEVERE_LOSS_REGRESSION = -0.05
SEVERE_STABILITY_REGRESSION = -0.05
SEVERE_NEIGHBOR_REGRESSION = -0.10
COLLAPSE_STD_THRESHOLD = 1e-4
COMPARABLE_SCOPE = "archived_sota_bridge_eligible"
NATIVE_SCOPE = "internal_rg_native_only"
MIN_PROMOTION_GROUPS = 2


def _normalised_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    denominator = max(1, len(values) - 1)
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = 1.0 - float(rank) / float(denominator)
    return result


def _read_completed(root: Path, *, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("protocol_id") != PROTOCOL_ID or payload.get("status") != "completed":
            continue
        if payload.get("stage") != stage:
            continue
        if payload.get("labels_accessed") is not False or payload.get("y_key_read") is not False:
            raise ValueError(f"label audit failed: {path}")
        if payload.get("n_clusters_used") is not None or payload.get("readout_enabled") is not False:
            raise ValueError(f"readout audit failed: {path}")
        if any(key in payload for key in ("metrics", "labels_true", "predictions")):
            raise ValueError(f"label-derived output found: {path}")
        for required in ("status.json", "run_record.json", "resolved_config.json", "input_profile.json", "unsupervised_diagnostics.json"):
            required_path = path.parent / required
            if not required_path.is_file():
                raise ValueError(f"incomplete artifact contract: {required_path}")
        status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
        run_record = json.loads((path.parent / "run_record.json").read_text(encoding="utf-8"))
        if status.get("status") != "completed" or run_record.get("status") != "completed":
            raise ValueError(f"status/run_record mismatch: {path}")
        if not isinstance(payload.get("unsupervised_diagnostics"), dict):
            raise ValueError(f"missing X-only diagnostics: {path}")
        rows.append(payload)
    return rows


def _read_reference(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(root.glob("**/summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("protocol_id") != PROTOCOL_ID or payload.get("stage") != "reference":
            continue
        if payload.get("variant") != "scmae_only" or payload.get("status") != "completed":
            continue
        if payload.get("labels_accessed") is not False or payload.get("y_key_read") is not False:
            raise ValueError(f"reference label audit failed: {path}")
        if payload.get("n_clusters_used") is not None or payload.get("readout_enabled") is not False:
            raise ValueError(f"reference readout audit failed: {path}")
        for required in ("status.json", "run_record.json", "resolved_config.json", "input_profile.json", "unsupervised_diagnostics.json"):
            required_path = path.parent / required
            if not required_path.is_file():
                raise ValueError(f"incomplete reference artifact contract: {required_path}")
        status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
        run_record = json.loads((path.parent / "run_record.json").read_text(encoding="utf-8"))
        if status.get("status") != "completed" or run_record.get("status") != "completed":
            raise ValueError(f"reference status/run_record mismatch: {path}")
        key = (str(payload["dataset_id"]), int(payload["seed"]))
        result[key] = payload
    return result


def _load_stage_spec(root: Path) -> dict[str, Any]:
    path = root / "stage_spec.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing stage_spec.json: {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("protocol_id") != PROTOCOL_ID or spec.get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError(f"invalid label-free stage spec: {path}")
    return spec


def _diag(payload: dict[str, Any]) -> dict[str, float]:
    diag = payload["unsupervised_diagnostics"]
    values = {
        "eval_mask_loss": float(diag["eval_mask_loss"]),
        "latent_view_cosine_mean": float(diag["latent_view_cosine_mean"]),
        "input_neighbor_overlap": float(diag["input_neighbor_overlap"]),
        "latent_mean_feature_std": float(diag["latent_mean_feature_std"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite X-only diagnostic in {payload.get('run_key')}")
    return values


def _paired_unit(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = {str(row["dataset_id"]): _diag(row) for row in candidate_rows}
    reference = {str(row["dataset_id"]): _diag(row) for row in reference_rows}
    if set(candidate) != set(reference):
        raise ValueError("candidate/reference protocol rows do not align")
    candidate_mean = {
        key: sum(value[key] for value in candidate.values()) / len(candidate)
        for key in next(iter(candidate.values()))
    }
    reference_mean = {
        key: sum(value[key] for value in reference.values()) / len(reference)
        for key in next(iter(reference.values()))
    }
    loss_relative = (reference_mean["eval_mask_loss"] - candidate_mean["eval_mask_loss"]) / max(
        abs(reference_mean["eval_mask_loss"]), 1e-8
    )
    stability_delta = candidate_mean["latent_view_cosine_mean"] - reference_mean["latent_view_cosine_mean"]
    neighbor_delta = candidate_mean["input_neighbor_overlap"] - reference_mean["input_neighbor_overlap"]
    deltas = {
        "loss_relative_improvement": float(loss_relative),
        "stability_delta": float(stability_delta),
        "neighbor_delta": float(neighbor_delta),
    }
    positive = {
        "loss": loss_relative > LOSS_RELATIVE_THRESHOLD,
        "stability": stability_delta > STABILITY_THRESHOLD,
        "neighbor": neighbor_delta > NEIGHBOR_THRESHOLD,
    }
    severe = {
        "loss": loss_relative < SEVERE_LOSS_REGRESSION,
        "stability": stability_delta < SEVERE_STABILITY_REGRESSION,
        "neighbor": neighbor_delta < SEVERE_NEIGHBOR_REGRESSION,
    }
    std_values = [value["latent_mean_feature_std"] for value in candidate.values()]
    collapse = bool(any(value < COLLAPSE_STD_THRESHOLD for value in std_values))
    return {
        **deltas,
        "positive_metrics": int(sum(positive.values())),
        "severe_regressions": int(sum(severe.values())),
        "collapse": collapse,
        "proxy_win": bool(sum(positive.values()) >= 2 and sum(severe.values()) == 0 and not collapse),
        "candidate_eval_mask_loss": candidate_mean["eval_mask_loss"],
        "reference_eval_mask_loss": reference_mean["eval_mask_loss"],
        "candidate_latent_view_cosine": candidate_mean["latent_view_cosine_mean"],
        "reference_latent_view_cosine": reference_mean["latent_view_cosine_mean"],
        "candidate_input_neighbor_overlap": candidate_mean["input_neighbor_overlap"],
        "reference_input_neighbor_overlap": reference_mean["input_neighbor_overlap"],
        "candidate_latent_mean_feature_std": candidate_mean["latent_mean_feature_std"],
        "reference_latent_mean_feature_std": reference_mean["latent_mean_feature_std"],
    }


def summarize(
    root: Path,
    manifest_path: Path,
    reference_root: Path,
    *,
    top_k: int,
) -> dict[str, Any]:
    spec = _load_stage_spec(root)
    manifest = _load_manifest(manifest_path)
    stage = str(spec["stage"])
    rows = _read_completed(root, stage=stage)
    reference = _read_reference(reference_root)
    expected_records = {str(value) for value in spec["dataset_ids"]}
    seeds = tuple(int(value) for value in spec["seeds"])
    candidate_ids = tuple(str(value) for value in spec["candidate_ids"])
    expected_keys = {
        (dataset_id, candidate_id, seed)
        for dataset_id in expected_records
        for candidate_id in candidate_ids
        for seed in seeds
    }
    indexed = {
        (str(row["dataset_id"]), str(row["candidate_id"]), int(row["seed"])): row
        for row in rows
    }
    completed_keys = set(indexed)
    if completed_keys != expected_keys:
        missing = sorted(expected_keys - completed_keys)
        raise RuntimeError(
            f"refusing to select from incomplete v2 stage: {len(completed_keys)}/{len(expected_keys)}, "
            f"missing={missing[:8]}"
        )
    for dataset_id in expected_records:
        for seed in seeds:
            if (dataset_id, seed) not in reference:
                raise RuntimeError(f"missing fixed scMAE reference for {dataset_id} seed={seed}")
            reference_row = reference[(dataset_id, seed)]
            for candidate_id in candidate_ids:
                candidate_row = indexed[(dataset_id, candidate_id, seed)]
                for field in ("split_seed", "fit_n_samples", "evaluation_n_samples", "input_protocol"):
                    if candidate_row.get(field) != reference_row.get(field):
                        raise RuntimeError(
                            f"candidate/reference split or input mismatch for {dataset_id} seed={seed}: {field}"
                        )

    records_by_id: dict[str, dict[str, Any]] = {}
    comparable_records_by_group: dict[str, list[str]] = defaultdict(list)
    native_records_by_group: dict[str, list[str]] = defaultdict(list)
    for record in manifest["datasets"]:
        dataset_id = str(record["dataset_id"])
        if dataset_id in expected_records:
            records_by_id[dataset_id] = record
            group = underlying_dataset_id(dataset_id)
            scope = str(record.get("comparison_scope", ""))
            if scope == COMPARABLE_SCOPE:
                comparable_records_by_group[group].append(dataset_id)
            elif scope == NATIVE_SCOPE:
                native_records_by_group[group].append(dataset_id)
            else:
                raise ValueError(f"unsupported comparison scope for {dataset_id}: {scope!r}")
    if set(records_by_id) != expected_records or not comparable_records_by_group:
        raise ValueError("stage manifest rows do not cover the expected comparable groups")

    unit_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        for group, group_records in sorted(comparable_records_by_group.items()):
            for seed in seeds:
                candidate_group_rows = [indexed[(dataset_id, candidate_id, seed)] for dataset_id in group_records]
                reference_group_rows = [reference[(dataset_id, seed)] for dataset_id in group_records]
                paired = _paired_unit(candidate_group_rows, reference_group_rows)
                unit_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "underlying_dataset_id": group,
                        "seed": int(seed),
                        "comparison_scope": COMPARABLE_SCOPE,
                        **paired,
                    }
                )

    native_unit_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        for group, group_records in sorted(native_records_by_group.items()):
            for seed in seeds:
                candidate_group_rows = [indexed[(dataset_id, candidate_id, seed)] for dataset_id in group_records]
                reference_group_rows = [reference[(dataset_id, seed)] for dataset_id in group_records]
                paired = _paired_unit(candidate_group_rows, reference_group_rows)
                native_unit_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "underlying_dataset_id": group,
                        "seed": int(seed),
                        "comparison_scope": NATIVE_SCOPE,
                        **paired,
                    }
                )

    rank_scores: dict[tuple[str, str, int], float] = {}
    for group in sorted(comparable_records_by_group):
        for seed in seeds:
            units = [
                row
                for row in unit_rows
                if row["underlying_dataset_id"] == group and int(row["seed"]) == int(seed)
            ]
            for metric in ("loss_relative_improvement", "stability_delta", "neighbor_delta"):
                ranks = _normalised_ranks([float(row[metric]) for row in units])
                for row, score in zip(units, ranks, strict=True):
                    key = (str(row["candidate_id"]), group, int(seed))
                    rank_scores[key] = rank_scores.get(key, 0.0) + float(score) / 3.0
    candidate_scores: list[dict[str, Any]] = []
    n_seed_majority = max(1, math.ceil(len(seeds) * 2 / 3))
    for candidate_id in candidate_ids:
        selected = [row for row in unit_rows if row["candidate_id"] == candidate_id]
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_group[str(row["underlying_dataset_id"])].append(row)
        group_wins = {
            group: (
                sum(bool(row["proxy_win"]) for row in group_rows) >= n_seed_majority
                and all(int(row["severe_regressions"]) == 0 and not bool(row["collapse"]) for row in group_rows)
            )
            for group, group_rows in by_group.items()
        }
        native_selected = [row for row in native_unit_rows if row["candidate_id"] == candidate_id]
        native_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in native_selected:
            native_by_group[str(row["underlying_dataset_id"])].append(row)
        native_guardrail = {
            group: all(
                int(row["severe_regressions"]) == 0 and not bool(row["collapse"])
                for row in group_rows
            )
            for group, group_rows in native_by_group.items()
        }
        scores = [rank_scores[(candidate_id, str(row["underlying_dataset_id"]), int(row["seed"]))] for row in selected]
        margins = [
            (float(row["loss_relative_improvement"]) + float(row["stability_delta"]) + float(row["neighbor_delta"])) / 3.0
            for row in selected
        ]
        candidate_scores.append(
            {
                "candidate_id": candidate_id,
                "proxy_win_groups": int(sum(group_wins.values())),
                "n_groups": len(group_wins),
                "proxy_win_rate": float(sum(group_wins.values()) / max(1, len(group_wins))),
                "mean_rank_score": float(sum(scores) / max(1, len(scores))),
                "p25_rank_score": float(sorted(scores)[max(0, int(len(scores) * 0.25) - 1)]),
                "mean_proxy_margin": float(sum(margins) / max(1, len(margins))),
                "collapse_units": int(sum(bool(row["collapse"]) for row in selected)),
                "severe_units": int(sum(int(row["severe_regressions"]) > 0 for row in selected)),
                "group_proxy_wins": group_wins,
                "n_units": len(selected),
                "native_guardrail": native_guardrail,
                "native_severe_units": int(
                    sum(int(row["severe_regressions"]) > 0 for row in native_selected)
                ),
                "native_collapse_units": int(
                    sum(bool(row["collapse"]) for row in native_selected)
                ),
            }
        )
    candidate_scores.sort(
        key=lambda row: (
            -row["proxy_win_groups"],
            row["severe_units"],
            row["collapse_units"],
            -row["mean_rank_score"],
            -row["p25_rank_score"],
            -row["mean_proxy_margin"],
            row["candidate_id"],
        )
    )
    catalog = candidate_catalog_for_stage(stage, candidate_ids=candidate_ids)
    catalog_by_id = {str(row["candidate_id"]): row for row in catalog}
    best = candidate_scores[0]
    selected_ids = [row["candidate_id"] for row in candidate_scores[: max(1, int(top_k))]]
    anchor_injected = False
    if stage == "mechanism_screen" and "default" in candidate_ids and "default" not in selected_ids:
        selected_ids[-1] = "default"
        anchor_injected = True
    proxy_supported = int(best["proxy_win_groups"]) >= MIN_PROMOTION_GROUPS
    selected_config = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "selection_method": (
            "paired held-out X-only proxy: primary objective is the number of underlying datasets "
            "where at least 2/3 metrics beat fixed scMAE without severe regression or collapse"
        ),
        "selection_target": "comparable_proxy_win_groups; native layers are guardrails only; post-hoc ARI/NMI/SOTA comparison is outside tuning",
        "selection_status": "proxy_supported" if proxy_supported else "no_go",
        "no_go": not proxy_supported,
        "minimum_promotion_groups": MIN_PROMOTION_GROUPS,
        "top_k_anchor_injected": anchor_injected,
        "candidate_id": best["candidate_id"],
        "top_candidate_ids": selected_ids,
        "overrides": catalog_by_id[best["candidate_id"]]["overrides"],
        "candidate_family": catalog_by_id[best["candidate_id"]]["family"],
        "base_config": "methods/TopoGate/V19_rg_adapter/configs/v19_rg.yaml",
        "reference_root": str(reference_root),
        "labels_accessed": False,
        "y_key_read": False,
        "n_clusters_used": None,
        "readout_enabled": False,
        "manifest_id": manifest.get("manifest_id"),
        "stage_spec": spec,
        "thresholds": {
            "loss_relative_improvement": LOSS_RELATIVE_THRESHOLD,
            "stability_delta": STABILITY_THRESHOLD,
            "neighbor_delta": NEIGHBOR_THRESHOLD,
            "severe_loss_regression": SEVERE_LOSS_REGRESSION,
            "severe_stability_regression": SEVERE_STABILITY_REGRESSION,
            "severe_neighbor_regression": SEVERE_NEIGHBOR_REGRESSION,
            "collapse_std": COLLAPSE_STD_THRESHOLD,
        },
        "best_summary": best,
    }
    summary = {
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "manifest_id": manifest.get("manifest_id"),
        "selection_uses_labels_or_outcomes": False,
        "labels_accessed": False,
        "y_key_read": False,
        "n_clusters_used": None,
        "readout_enabled": False,
        "expected_runs": len(expected_keys),
        "completed_runs": len(completed_keys),
        "candidate_scores": candidate_scores,
        "selected_config": selected_config,
        "group_seed_scores": unit_rows,
        "native_group_seed_scores": native_unit_rows,
    }
    _write_json(root / "unsupervised_selection.json", summary)
    _write_json(root / "selected_config.json", selected_config)
    with (root / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "candidate_id",
            "proxy_win_groups",
            "n_groups",
            "proxy_win_rate",
            "mean_rank_score",
            "p25_rank_score",
            "mean_proxy_margin",
            "collapse_units",
            "severe_units",
            "n_units",
            "group_proxy_wins",
            "native_severe_units",
            "native_collapse_units",
            "native_guardrail",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_scores)
    with (root / "group_seed_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(unit_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(unit_rows)
    if native_unit_rows:
        with (root / "native_group_seed_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(native_unit_rows[0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(native_unit_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize V19 RG v2 X-only tuning")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    stage = str(_load_stage_spec(args.output_dir)["stage"])
    expected_top_k = {"mechanism_screen": 12, "mechanism_refine": 1}
    if stage not in expected_top_k:
        raise ValueError(f"unsupported formal stage for summary: {stage}")
    if args.top_k != expected_top_k[stage]:
        raise ValueError(f"formal stage {stage} requires --top-k {expected_top_k[stage]}")
    summary = summarize(args.output_dir, args.manifest, args.reference_dir, top_k=int(args.top_k))
    print(json.dumps(summary["selected_config"], ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
