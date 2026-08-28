#!/usr/bin/env python3
"""Replay the frozen E1 topology policy for the V25 E2-A feature audit.

This is an offline diagnostic.  It does not train a model or choose a policy.
It reconstructs the already-run E1 schedule, loads the final topology Gate
checkpoint, and aggregates selected versus eligible-but-not-selected feature
coordinates.  The inferential unit remains one dataset/seed panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from methods.TopoGate.V21_assignment_adversarial_gate.graph import (  # noqa: E402
    build_svd_knn_graph,
    compute_topology_statistics,
)
from methods.TopoGate.V21_assignment_adversarial_gate.input_adapter import (  # noqa: E402
    load_npz,
    prepare_dual_input,
)
from methods.TopoGate.V21_assignment_adversarial_gate.model import FeatureGate  # noqa: E402
from methods.TopoGate.V25_systematic_mechanism_study.e1_protocol import (  # noqa: E402
    E1Config,
    _make_schedule,
    _materialize_schedule,
    _selection_from_logits,
)
from methods.TopoGate.V25_systematic_mechanism_study.e2_metrics import (  # noqa: E402
    CoordinateMetricAccumulator,
)


PROTOCOL_ID = "v25_e2a_feature_semantics_v1"


def _require_pilot_gate(root: Path, pilot_audit: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Require the frozen pilot materiality gate before E2-A can run.

    E2-A is an explanation of an already-authorized E1 phase.  Keeping this
    check at the entry point prevents a direct invocation from silently
    turning a weak or incomplete pilot into a prospective analysis.  The
    explicit override is useful for copied artifact trees; it still points to
    a phase-summary artifact and is validated identically.
    """
    if pilot_audit is None:
        pilot_audit = (
            root / "Audit" / "phase_summary.json"
            if root.name == "pilot"
            else root.parent / "pilot" / "Audit" / "phase_summary.json"
        )
    pilot_audit = Path(pilot_audit)
    if not pilot_audit.is_file():
        raise ValueError(f"E2-A requires a pilot phase audit: {pilot_audit}")
    try:
        payload = _json(pilot_audit)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"invalid pilot phase audit: {pilot_audit}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"pilot phase audit must be a JSON object: {pilot_audit}")
    gate = payload.get("phase_gate")
    if not isinstance(gate, dict):
        raise ValueError(f"pilot phase audit has no phase_gate: {pilot_audit}")
    try:
        material = int(gate.get("material_dataset_count", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pilot phase audit has invalid material count: {pilot_audit}") from exc
    if gate.get("passes") is not True or material < 2:
        raise ValueError(
            "E2-A requires a passing pilot phase_gate with at least two "
            f"material datasets; got passes={gate.get('passes')!r}, material={material}"
        )
    return pilot_audit, payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _support_matrix(values: Any) -> np.ndarray:
    if sp.issparse(values):
        return values.tocsr().astype(bool)
    return np.asarray(values) != 0


def _feature_fisher(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    encoded, _ = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    _, inverse = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    n_samples = X.shape[0]
    overall = X.mean(axis=0, dtype=np.float64)
    between = np.zeros(X.shape[1], dtype=np.float64)
    within = np.zeros(X.shape[1], dtype=np.float64)
    for class_id in range(encoded.size):
        rows = X[inverse == class_id]
        if rows.shape[0] == 0:
            continue
        mean = rows.mean(axis=0, dtype=np.float64)
        between += float(rows.shape[0]) * (mean - overall) ** 2
        centered = rows.astype(np.float64, copy=False) - mean
        within += np.square(centered).sum(axis=0)
    return (between / float(max(n_samples, 1))) / (within / float(max(n_samples, 1)) + 1e-8)


def _binary_support_mi(support: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Exact MI between a feature's support indicator and the class label."""
    support = np.asarray(support, dtype=bool)
    encoded, inverse = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    n = float(support.shape[0])
    p_x1 = support.mean(axis=0, dtype=np.float64)
    p_x = (1.0 - p_x1, p_x1)
    result = np.zeros(support.shape[1], dtype=np.float64)
    for class_id in range(encoded.size):
        mask = inverse == class_id
        p_y = float(mask.sum()) / n
        if p_y <= 0:
            continue
        p_xy1 = support[mask].mean(axis=0, dtype=np.float64) * p_y
        p_xy = (p_y - p_xy1, p_xy1)
        for state in (0, 1):
            joint = p_xy[state]
            expected = p_x[state] * p_y
            valid = (joint > 0) & (expected > 0)
            result[valid] += joint[valid] * np.log(joint[valid] / expected[valid])
    return result


def _class_support_enrichment(support: np.ndarray, labels: np.ndarray) -> np.ndarray:
    support = np.asarray(support, dtype=bool)
    encoded, inverse = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    overall = support.mean(axis=0, dtype=np.float64)
    maximum = np.zeros(support.shape[1], dtype=np.float64)
    for class_id in range(encoded.size):
        mask = inverse == class_id
        if mask.any():
            maximum = np.maximum(maximum, np.abs(support[mask].mean(axis=0, dtype=np.float64) - overall))
    return maximum


def _feature_metrics(X_model: np.ndarray, raw_X: Any, labels: np.ndarray | None) -> dict[str, np.ndarray]:
    raw_support = _support_matrix(raw_X)
    metrics: dict[str, np.ndarray] = {
        "model_variance": np.var(X_model.astype(np.float64), axis=0),
        "model_zero_fraction": np.mean(X_model == 0, axis=0, dtype=np.float64),
        "model_support_frequency": np.mean(X_model != 0, axis=0, dtype=np.float64),
        "raw_support_frequency": np.asarray(raw_support.mean(axis=0)).reshape(-1).astype(np.float64),
    }
    if labels is not None:
        metrics.update(
            {
                "fisher_separation_posthoc": _feature_fisher(X_model, labels),
                "support_mutual_information_posthoc": _binary_support_mi(raw_support, labels),
                "class_support_enrichment_posthoc": _class_support_enrichment(raw_support, labels),
            }
        )
    return metrics


def _panel_dirs(root: Path, datasets: list[str] | None) -> list[Path]:
    selected = set(datasets or [])
    result: list[Path] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in {"logs", "mplconfig", "E2"}:
            continue
        if selected and dataset_dir.name not in selected:
            continue
        for seed_dir in sorted(dataset_dir.glob("seed*")):
            if seed_dir.is_dir() and (seed_dir / "summary.json").is_file():
                result.append(seed_dir)
    return result


def _audit_from_saved_counts(
    panel_dir: Path,
    summary: dict[str, Any],
    profile: dict[str, Any],
    feature_values: dict[str, np.ndarray],
    labels_available: bool,
) -> dict[str, Any]:
    meta = _json(panel_dir / "T" / "feature_audit_label_free.json")
    with np.load(panel_dir / "T" / "feature_selection_counts.npz") as counts:
        selected_counts = np.asarray(counts["selected_feature_counts"], dtype=np.float64)
        other_counts = np.asarray(counts["eligible_not_selected_feature_counts"], dtype=np.float64)
    if selected_counts.shape != other_counts.shape:
        raise ValueError(f"saved E2-A count shapes differ: {panel_dir}")
    selected_total = int(meta["selected_coordinate_count"])
    other_total = int(meta["eligible_not_selected_coordinate_count"])
    if selected_total != int(selected_counts.sum()) or other_total != int(other_counts.sum()):
        raise ValueError(f"saved E2-A count totals do not reconcile: {panel_dir}")
    metrics: dict[str, Any] = {}
    for name, values in sorted(feature_values.items()):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != selected_counts.shape:
            raise ValueError(f"feature metric {name!r} shape differs from saved counts: {panel_dir}")
        selected_mean = float(np.dot(selected_counts, values) / selected_total) if selected_total else None
        other_mean = float(np.dot(other_counts, values) / other_total) if other_total else None
        metrics[name] = {
            "selected_mean": selected_mean,
            "eligible_not_selected_mean": other_mean,
            "difference": selected_mean - other_mean if selected_mean is not None and other_mean is not None else None,
            "selected_n_coordinates": selected_total,
            "eligible_not_selected_n_coordinates": other_total,
        }
    for name, sums in sorted(meta.get("coordinate_metric_sums", {}).items()):
        selected_mean = float(sums["selected"] / selected_total) if selected_total else None
        other_mean = float(sums["eligible_not_selected"] / other_total) if other_total else None
        metrics[name] = {
            "selected_mean": selected_mean,
            "eligible_not_selected_mean": other_mean,
            "difference": selected_mean - other_mean if selected_mean is not None and other_mean is not None else None,
            "selected_n_coordinates": selected_total,
            "eligible_not_selected_n_coordinates": other_total,
        }
    total = int(meta["coordinate_count"])
    eligible = int(meta["eligible_coordinate_count"])
    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "dataset_id": str(profile["dataset"]),
        "seed": int(profile["seed"]),
        "statistical_unit": "dataset_seed_summary",
        "coordinate_count": total,
        "selected_coordinate_count": selected_total,
        "eligible_not_selected_coordinate_count": other_total,
        "coordinate_distribution_is_descriptive_only": True,
        "metrics": metrics,
        "source_path": str(Path(profile["data_path"]).resolve()),
        "source_sha256": _sha256(Path(profile["data_path"])),
        "e1_protocol_id": summary.get("protocol_id"),
        "selection_policy_snapshot": meta.get("selection_snapshot"),
        "selection_policy_is_not_a_new_fit": True,
        "measurement_timing": "post_intervention_policy_audit",
        "causal_status": "observational",
        "labels_used_for_fit": False,
        "posthoc_labels_used": labels_available,
        "posthoc_label_metrics_are_not_fit_inputs": True,
        "schedule_entry_count": None,
        "schedule_coordinate_count": total,
        "eligible_rate": float(eligible / total) if total else None,
        "selected_rate": float(selected_total / total) if total else None,
        "selected_within_eligible_rate": float(selected_total / eligible) if eligible else None,
        "topology_statistics_hash": _json(panel_dir / "audit.json").get("topology_statistics_hash"),
        "topology_statistics_profile": None,
        "audit_ok": bool(selected_total > 0 and other_total > 0 and total >= eligible),
        "audit_source": "E1_training_time_saved_counts",
    }
    output_path = panel_dir / "e2_feature_audit.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return result


def audit_panel(panel_dir: Path) -> dict[str, Any]:
    summary = _json(panel_dir / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError(f"panel is not completed: {panel_dir}")
    profile = _json(panel_dir / "runner_profile.json")
    config = E1Config(**_json(panel_dir / "resolved_config.json"))
    source = Path(profile["data_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    loaded = load_npz(source)
    prepared = prepare_dual_input(loaded.X, dataset_name=profile["dataset"], input_protocol=profile["input_protocol"])
    X_model = np.ascontiguousarray(np.asarray(prepared.X_model, dtype=np.float32))
    feature_values = _feature_metrics(X_model, loaded.X, loaded.labels)
    saved_counts = panel_dir / "T" / "feature_selection_counts.npz"
    saved_meta = panel_dir / "T" / "feature_audit_label_free.json"
    if saved_counts.is_file() and saved_meta.is_file():
        return _audit_from_saved_counts(panel_dir, summary, profile, feature_values, loaded.labels is not None)
    device = torch.device("cpu")
    X = torch.as_tensor(X_model, dtype=torch.float32, device=device)
    graph = build_svd_knn_graph(
        prepared.X_graph,
        neighbor_k=config.neighbor_k,
        svd_target=config.graph_svd_target,
        svd_min_dim=min(config.graph_svd_min_dim, max(1, X_model.shape[0] - 1)),
        svd_max_dim=min(config.graph_svd_max_dim, max(1, X_model.shape[0] - 1)),
        seed=int(profile["seed"]),
    )
    stats_np, stats_profile = compute_topology_statistics(
        X_model,
        graph,
        block_size=config.stats_block_size,
        cache_dir=None,
        cache_dtype=config.stats_cache_dtype,
        clip=config.stats_clip,
    )
    gate = FeatureGate(config.gate_hidden).to(device)
    checkpoint = torch.load(panel_dir / "T" / "checkpoint.pt", map_location=device, weights_only=False)
    gate.load_state_dict(checkpoint["gate"])
    gate.eval()
    schedule = _make_schedule(X_model.shape[0], config, int(profile["seed"]))
    accumulator = CoordinateMetricAccumulator(str(profile["dataset"]), int(profile["seed"]))
    eligible_total = 0
    selected_total = 0
    schedule_rows = 0
    for entry in schedule.post_branch:
        tensors = _materialize_schedule(X, entry, config, device)
        batch_ids = torch.as_tensor(entry.batch_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = gate(torch.as_tensor(stats_np[batch_ids.cpu().numpy()], dtype=torch.float32, device=device))
            _st, hard, _budgets = _selection_from_logits(logits, tensors["eligible"], tensors["gumbel"], config)
        selected = hard.detach().cpu().numpy().astype(bool)
        eligible = tensors["eligible"].detach().cpu().numpy().astype(bool)
        batch = tensors["batch"].detach().cpu().numpy()
        donor_change = np.abs(tensors["assignment_donor"].detach().cpu().numpy() - batch)
        stats_batch = np.asarray(stats_np[batch_ids.cpu().numpy()], dtype=np.float64)
        batch_metrics: dict[str, np.ndarray] = {
            "topology_deviation": stats_batch[:, :, 0],
            "topology_dispersion": stats_batch[:, :, 1],
            "donor_change_magnitude": donor_change,
        }
        for name, values in feature_values.items():
            batch_metrics[name] = np.broadcast_to(values[None, :], selected.shape)
        accumulator.update(selected, eligible, batch_metrics)
        eligible_total += int(eligible.sum())
        selected_total += int(selected.sum())
        schedule_rows += int(selected.size)
    result = accumulator.finalize()
    result.update(
        {
            "protocol_id": PROTOCOL_ID,
            "source_path": str(source.resolve()),
            "source_sha256": _sha256(source),
            "e1_protocol_id": summary.get("protocol_id"),
            "selection_policy_snapshot": "final_topology_gate_checkpoint_replayed_over_post_branch_schedule",
            "selection_policy_is_not_a_new_fit": True,
            "measurement_timing": "post_intervention_policy_audit",
            "causal_status": "observational",
            "labels_used_for_fit": False,
            "posthoc_labels_used": loaded.labels is not None,
            "posthoc_label_metrics_are_not_fit_inputs": True,
            "schedule_entry_count": len(schedule.post_branch),
            "schedule_coordinate_count": schedule_rows,
            "eligible_rate": float(eligible_total / schedule_rows) if schedule_rows else None,
            "selected_rate": float(selected_total / schedule_rows) if schedule_rows else None,
            "selected_within_eligible_rate": float(selected_total / eligible_total) if eligible_total else None,
            "topology_statistics_hash": hashlib.sha256(np.ascontiguousarray(stats_np).tobytes()).hexdigest(),
            "topology_statistics_profile": stats_profile,
            "audit_ok": bool(result["selected_coordinate_count"] > 0 and result["eligible_not_selected_coordinate_count"] > 0),
        }
    )
    output_path = panel_dir / "e2_feature_audit.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="E1 phase root, e.g. .../E1/pilot")
    parser.add_argument("--out", type=Path, required=True, help="aggregate JSON output")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--pilot-audit",
        type=Path,
        default=None,
        help="optional frozen pilot phase_summary.json; otherwise infer it from --root",
    )
    args = parser.parse_args()
    pilot_audit_path, pilot_summary = _require_pilot_gate(args.root, args.pilot_audit)
    panels = _panel_dirs(args.root, args.datasets)
    if not panels:
        raise ValueError(f"no completed E1 panels found under {args.root}")
    audits = [audit_panel(panel) for panel in panels]
    aggregate = {
        "protocol_id": PROTOCOL_ID,
        "root": str(args.root.resolve()),
        "statistical_unit": "dataset_seed_summary",
        "coordinate_distributions_descriptive_only": True,
        "panel_count": len(audits),
        "audit_ok_count": sum(bool(item.get("audit_ok")) for item in audits),
        "activation_gate": {
            "pilot_phase_summary": str(pilot_audit_path.resolve()),
            "pilot_phase_summary_sha256": _sha256(pilot_audit_path),
            "phase_gate": pilot_summary["phase_gate"],
            "required_material_dataset_count": 2,
        },
        "audits": audits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: aggregate[k] for k in ("protocol_id", "panel_count", "audit_ok_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
