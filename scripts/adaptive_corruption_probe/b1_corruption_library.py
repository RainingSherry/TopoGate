"""B1 fixed corruption-library experiment for ``adaptive_corruption_probe``.

The probe deliberately consumes only the audited S0 ``H0`` stem during fit.
Ground-truth labels are loaded after the encoder is fitted, solely for the
benchmark-known-K readout and post-fit metrics.  The six arms share one small
autoencoder, one standardization, one corruption rate, one optimizer and one
training budget.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import protocol


PROJECT_ROOT = protocol.PROJECT_ROOT
H0_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S0_freeze/datasets"
LABEL_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S1_oracle_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "result/adaptive_corruption_probe/B1_corruption_library"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a runtime dependency for B1
        pass


def _cuda_visible_is_legal() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return False
    try:
        ids = {int(item.strip()) for item in visible.split(",") if item.strip()}
    except ValueError:
        return False
    return bool(ids) and ids.isdisjoint(set(protocol.FORBIDDEN_GPU_IDS))


def _load_h0(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    h0_path = H0_ROOT / dataset / "H0.npy"
    manifest_path = H0_ROOT / dataset / "budget_manifest.json"
    if not h0_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"missing audited S0 H0 for {dataset}")
    h0 = np.asarray(np.load(h0_path), dtype=np.float32)
    if h0.ndim != 2 or not 1 <= h0.shape[1] <= 128:
        raise ValueError(f"unexpected H0 shape for {dataset}: {h0.shape}")
    profile = json.loads(manifest_path.read_text(encoding="utf-8"))
    return h0, {
        "H0_path": str(h0_path.resolve()),
        "H0_sha256": sha256_file(h0_path),
        "budget_manifest_sha256": sha256_file(manifest_path),
        "shape": list(h0.shape),
        "source": "representation_consumer_probe/S0_freeze",
        "labels_used": False,
        "budget_profile": profile,
    }


def _load_labels(dataset: str) -> np.ndarray:
    path = LABEL_ROOT / dataset / "seed42" / "R" / "labels_true.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing post-fit benchmark labels for {dataset}")
    return np.asarray(np.load(path), dtype=np.int64)


def support_mask(h0: np.ndarray) -> np.ndarray:
    row_max = np.max(np.abs(h0), axis=1, keepdims=True)
    threshold = np.maximum(1e-6, protocol.H0_SUPPORT_THRESHOLD_RATIO * row_max)
    return np.abs(h0) >= threshold


def _force_changed(value: float, old: float, scale: float, *, active: bool, rng: np.random.Generator) -> float:
    candidate = float(value)
    if active:
        minimum = max(1e-6, 1.05 * scale)
        if abs(candidate) < minimum:
            candidate = minimum if candidate >= 0.0 else -minimum
        if abs(candidate - old) < 1e-7:
            candidate = -candidate if abs(candidate) >= minimum else minimum
    elif abs(candidate - old) < 1e-7:
        candidate = float(old + (rng.uniform(0.5, 1.5) * max(scale, 1e-3)))
    return candidate


def _random_positions(active: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(rng.choice(active.size, size=min(count, active.size), replace=False), dtype=np.int64)


def corrupt_h0(
    h0: np.ndarray,
    arm: str,
    rng: np.random.Generator,
    *,
    static_residual: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply one frozen support/value corruption and return audit statistics."""
    clean = np.asarray(h0, dtype=np.float32)
    corrupted = clean.copy()
    active = support_mask(clean)
    n, d = clean.shape
    scale = float(np.median(np.abs(clean[active]))) if np.any(active) else 1.0
    scale = max(scale, 1e-4)
    requested = 0
    for row in range(n):
        active_idx = np.flatnonzero(active[row])
        inactive_idx = np.flatnonzero(~active[row])
        m = int(np.ceil(protocol.CORRUPTION_RATE * max(active_idx.size, 1)))
        # A common feasible pair budget makes C0/C1/C2/C3/C4 comparable even
        # when a row has too little inactive support for a nominal 2*m move.
        pair_count = min(m, active_idx.size // 2, inactive_idx.size)
        count = 2 * pair_count
        if arm == "C_clean_no_corruption" or not active_idx.size:
            continue
        requested += count
        donor_row = int(rng.integers(0, n))
        if donor_row == row and n > 1:
            donor_row = (donor_row + 1) % n
        if arm == "C0_MatchedRandom":
            positions = rng.choice(d, size=min(count, d), replace=False)
            for col in positions:
                candidate = float(clean[donor_row, col])
                corrupted[row, col] = _force_changed(candidate, float(clean[row, col]), scale, active=bool(active[row, col]), rng=rng)
        elif arm == "C1_ValueOnly":
            positions = rng.choice(active_idx, size=min(count, active_idx.size), replace=False)
            for col in positions:
                donor_active = np.flatnonzero(active[:, col])
                candidate = float(clean[int(rng.choice(donor_active)) , col]) if donor_active.size else float(clean[donor_row, col])
                corrupted[row, col] = _force_changed(candidate, float(clean[row, col]), scale, active=True, rng=rng)
        elif arm in {"C2_SupportOnly", "C3_MixedMatched"}:
            if pair_count <= 0:
                continue
            sources = rng.choice(active_idx, size=pair_count, replace=False)
            destinations = rng.choice(inactive_idx, size=pair_count, replace=False)
            for src, dst in zip(sources, destinations, strict=True):
                value = float(clean[row, src])
                if arm == "C3_MixedMatched":
                    value *= float(rng.uniform(0.5, 1.5))
                    value = _force_changed(value, 0.0, scale, active=True, rng=rng)
                corrupted[row, src] = 0.0
                corrupted[row, dst] = value
        elif arm == "C4_StaticHard":
            if static_residual is None:
                raise ValueError("C4 requires frozen warm-up residuals")
            residual = np.asarray(static_residual[row], dtype=np.float32)
            positions = np.argsort(-residual, kind="stable")[: min(count, d)]
            for col in positions:
                candidate = float(clean[donor_row, col])
                corrupted[row, col] = _force_changed(candidate, float(clean[row, col]), scale, active=bool(active[row, col]), rng=rng)
        else:
            raise ValueError(f"unknown corruption arm: {arm}")
    changed = np.abs(corrupted - clean) > 1e-7
    support_changed = support_mask(corrupted) != active
    both_active = support_mask(corrupted) & active
    value_changed = changed & both_active
    return corrupted.astype(np.float32, copy=False), {
        "requested_change_ratio": float(requested / max(n * d, 1)),
        "effective_changed_coordinate_rate": float(np.mean(changed)),
        "support_change_ratio": float(np.mean(support_changed)),
        "value_change_ratio": float(np.mean(value_changed)),
        "total_absolute_change": float(np.sum(np.abs(corrupted - clean), dtype=np.float64)),
    }


class _SmallMAE:
    def __init__(self, device: Any, input_dim: int) -> None:
        import torch
        from torch import nn

        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim),
        ).to(device)
        self.encoder = nn.Sequential(*list(self.model.children())[:4]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(protocol.BACKBONE_CONFIG["learning_rate"]))

    def fit_epoch(self, x: np.ndarray, target: np.ndarray, batch_size: int, rng: np.random.Generator) -> float:
        import torch
        self.model.train()
        order = rng.permutation(x.shape[0])
        loss_sum = 0.0
        count = 0
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            xb = torch.as_tensor(x[idx], dtype=torch.float32, device=self.device)
            yb = torch.as_tensor(target[idx], dtype=torch.float32, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            self.optimizer.step()
            loss_sum += float(loss.detach().cpu()) * idx.size
            count += int(idx.size)
        return loss_sum / max(count, 1)

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        import torch
        self.model.eval()
        embeddings: list[np.ndarray] = []
        recon: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device)
                embeddings.append(self.encoder(xb).detach().cpu().numpy())
                recon.append(self.model(xb).detach().cpu().numpy())
        return np.concatenate(embeddings, axis=0), np.concatenate(recon, axis=0)


def _standardize(h0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(h0, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(h0, axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((h0 - mean) / std).astype(np.float32), mean, std


def _warmup_residual(h0_scaled: np.ndarray, seed: int, device: Any) -> np.ndarray:
    _seed_everything(seed + 991)
    model = _SmallMAE(device, h0_scaled.shape[1])
    rng = np.random.default_rng(seed + 991)
    for _ in range(5):
        model.fit_epoch(h0_scaled, h0_scaled, int(protocol.BACKBONE_CONFIG["batch_size"]), rng)
    _, recon = model.predict(h0_scaled)
    residual = np.abs(recon - h0_scaled).astype(np.float32)
    del model
    try:
        import torch

        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
    return residual


def positive_control() -> dict[str, Any]:
    rng = np.random.default_rng(20260818)
    base = np.zeros((24, 16), dtype=np.float32)
    base[:, :4] = rng.uniform(1.0, 2.0, size=(24, 4))
    base[:, 4:8] = rng.uniform(-1.0, 1.0, size=(24, 4))
    checks: dict[str, bool] = {}
    audit: dict[str, Any] = {}
    clean_support = support_mask(base)
    for arm in protocol.CORRUPTION_ARMS:
        residual = np.abs(rng.normal(size=base.shape)).astype(np.float32) if arm == "C4_StaticHard" else None
        corrupted, stats = corrupt_h0(base, arm, rng, static_residual=residual)
        checks[f"{arm}_changes"] = bool(np.any(np.abs(corrupted - base) > 1e-7)) if arm != "C_clean_no_corruption" else bool(np.allclose(corrupted, base))
        audit[arm] = {
            "stats": stats,
            "support_changed": int(np.sum(support_mask(corrupted) != clean_support)),
            "value_changed": int(np.sum(np.abs(corrupted - base) > 1e-7)),
        }
    checks["C1_support_preserved"] = audit["C1_ValueOnly"]["support_changed"] == 0
    checks["C2_support_changed"] = audit["C2_SupportOnly"]["support_changed"] > 0
    checks["C3_support_and_values_changed"] = audit["C3_MixedMatched"]["support_changed"] > 0 and audit["C3_MixedMatched"]["value_changed"] > 0
    passed = bool(all(checks.values()))
    return {"status": "completed_valid" if passed else "protocol_insensitive", "labels_used": False, "checks": checks, "arms": audit}


def run_job(dataset: str, arm: str, seed: int, output_dir: Path) -> dict[str, Any]:
    protocol.validate_contract()
    _seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() and _cuda_visible_is_legal() else "cpu")
    except ImportError as exc:
        raise RuntimeError("B1 requires torch") from exc
    h0_raw, source = _load_h0(dataset)
    h0_scaled, mean, std = _standardize(h0_raw)
    static_residual = _warmup_residual(h0_scaled, seed, device) if arm == "C4_StaticHard" else None
    model = _SmallMAE(device, h0_scaled.shape[1])
    rng = np.random.default_rng(seed)
    metric_accum: list[dict[str, float]] = []
    epoch_loss = []
    for epoch in range(int(protocol.BACKBONE_CONFIG["epochs"])):
        corrupted_raw, stats = corrupt_h0(h0_raw, arm, rng, static_residual=static_residual)
        corrupted_scaled = ((corrupted_raw - mean) / std).astype(np.float32)
        loss = model.fit_epoch(corrupted_scaled, h0_scaled, int(protocol.BACKBONE_CONFIG["batch_size"]), rng)
        epoch_loss.append(float(loss))
        metric_accum.append(stats)
    embedding, reconstruction = model.predict(h0_scaled)
    labels = _load_labels(dataset)
    if labels.size != h0_raw.shape[0]:
        raise ValueError(f"label/H0 mismatch for {dataset}")
    k = int(np.unique(labels).size)
    predictions = KMeans(n_clusters=k, n_init=20, random_state=int(seed)).fit_predict(embedding)
    clean_loss = float(np.mean((reconstruction - h0_scaled) ** 2))
    summary = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "B1",
        "dataset": dataset,
        "role": protocol.ROLE_BY_DATASET[dataset],
        "arm": arm,
        "seed": int(seed),
        "status": "completed_valid",
        "device": str(device),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "K": k,
        "K_source": "benchmark_oracle_from_y",
        "metrics": {
            "ARI": float(adjusted_rand_score(labels, predictions)),
            "NMI": float(normalized_mutual_info_score(labels, predictions)),
            "L_rec": float(epoch_loss[-1]),
            "clean_reconstruction_loss": clean_loss,
        },
        "corruption_audit": {
            "requested_change_ratio_mean": float(np.mean([row["requested_change_ratio"] for row in metric_accum])),
            "effective_changed_coordinate_rate_mean": float(np.mean([row["effective_changed_coordinate_rate"] for row in metric_accum])),
            "support_change_ratio_mean": float(np.mean([row["support_change_ratio"] for row in metric_accum])),
            "value_change_ratio_mean": float(np.mean([row["value_change_ratio"] for row in metric_accum])),
            "total_absolute_change_mean": float(np.mean([row["total_absolute_change"] for row in metric_accum])),
            "epochs_audited": len(metric_accum),
        },
        "source": source,
        "backbone": dict(protocol.BACKBONE_CONFIG),
        "input_dim": int(h0_raw.shape[1]),
        "support_definition": protocol.BACKBONE_CONFIG["support_definition"],
        "positive_control_required": True,
        "raw_arrays_persisted": False,
    }
    audit = {
        "audit_ok": True,
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "cuda_visible_is_legal": bool(device.type != "cuda" or _cuda_visible_is_legal()),
        "embedding_finite": bool(np.isfinite(embedding).all()),
        "prediction_count": int(np.unique(predictions).size),
        "support_value_budget_fields_present": True,
        "raw_artifacts_published": False,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "audit.json", audit)
    write_json(output_dir / "resolved_config.json", {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "B1",
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "backbone": dict(protocol.BACKBONE_CONFIG),
        "input_dim": int(h0_raw.shape[1]),
        "support_definition": protocol.BACKBONE_CONFIG["support_definition"],
        "labels_used_during_fit": False,
        "K_source": "benchmark_oracle_from_y_outer_readout_only",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    return summary


def _run_one_subprocess(dataset: str, arm: str, seed: int, root: Path, gpu_id: int) -> dict[str, Any]:
    run_dir = root / dataset / arm / f"seed{seed}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "4"
    command = [sys.executable, "-m", "scripts.adaptive_corruption_probe.b1_corruption_library", "--dataset", dataset, "--arm", arm, "--seed", str(seed), "--output-dir", str(run_dir)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        error = {"dataset": dataset, "arm": arm, "seed": seed, "status": "incomplete_compute", "returncode": completed.returncode, "stderr_tail": completed.stderr[-4000:]}
        write_json(run_dir / "failure.json", error)
        return error
    summary_path = run_dir / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"dataset": dataset, "arm": arm, "seed": seed, "status": "incomplete_compute", "reason": "missing_summary"}


def _coarse_role(dataset: str) -> str:
    role = protocol.ROLE_BY_DATASET[dataset]
    if role.startswith("sparse_text"):
        return "sparse_text"
    if role.startswith("registered_scrna"):
        return "registered_scrna_count"
    return "generic_sparse_high_dimensional"


def aggregate(root: Path, positive: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in protocol.DEVELOPMENT_PANEL:
        for arm in protocol.CORRUPTION_ARMS:
            for seed in protocol.PRIMARY_SEEDS:
                path = root / dataset / arm / f"seed{seed}" / "summary.json"
                if path.exists():
                    summary = json.loads(path.read_text(encoding="utf-8"))
                    if summary.get("status") == "completed_valid":
                        metrics = summary["metrics"]
                        audit = summary["corruption_audit"]
                        rows.append({"dataset": dataset, "role_class": _coarse_role(dataset), "arm": arm, "seed": seed, "status": "completed_valid", "ARI": metrics["ARI"], "NMI": metrics["NMI"], "L_rec": metrics["L_rec"], **audit})
                        continue
                rows.append({"dataset": dataset, "role_class": _coarse_role(dataset), "arm": arm, "seed": seed, "status": "incomplete_compute"})
    valid = [row for row in rows if row["status"] == "completed_valid"]
    complete = len(valid) == len(rows) == len(protocol.DEVELOPMENT_PANEL) * len(protocol.CORRUPTION_ARMS) * len(protocol.PRIMARY_SEEDS)
    if not positive.get("status") == "completed_valid":
        decision = {"stage": "B1", "status": "protocol_insensitive", "primary_gate_pass": False, "next_stage_authorized": False, "authorized_next_stage": None, "terminal_reason": "positive-control sensitivity fixture failed"}
    elif not complete:
        decision = {"stage": "B1", "status": "incomplete_compute", "primary_gate_pass": None, "next_stage_authorized": False, "authorized_next_stage": None, "terminal_reason": "one or more formal B1 jobs incomplete"}
    else:
        by_da: dict[tuple[str, str], float] = {}
        for dataset in protocol.DEVELOPMENT_PANEL:
            clean = float(np.mean([r["ARI"] for r in valid if r["dataset"] == dataset and r["arm"] == "C_clean_no_corruption"]))
            for arm in protocol.CORRUPTION_ARMS[1:]:
                ari = float(np.mean([r["ARI"] for r in valid if r["dataset"] == dataset and r["arm"] == arm]))
                by_da[(dataset, arm)] = ari - clean
        delta_clean_c0 = {dataset: by_da[(dataset, "C0_MatchedRandom")] for dataset in protocol.DEVELOPMENT_PANEL}
        h_corr = {dataset: max(by_da[(dataset, arm)] for arm in protocol.CORRUPTION_ARMS[1:]) for dataset in protocol.DEVELOPMENT_PANEL}
        delta_random: dict[tuple[str, str], float] = {}
        for dataset in protocol.DEVELOPMENT_PANEL:
            c0 = float(np.mean([r["ARI"] for r in valid if r["dataset"] == dataset and r["arm"] == "C0_MatchedRandom"]))
            for arm in protocol.STRUCTURED_ARMS:
                ari = float(np.mean([r["ARI"] for r in valid if r["dataset"] == dataset and r["arm"] == arm]))
                delta_random[(dataset, arm)] = ari - c0
        level1_material = any(abs(v) >= protocol.MATERIAL_DELTA_ARI for v in delta_clean_c0.values()) or any(abs(v) >= protocol.MATERIAL_DELTA_ARI for v in h_corr.values())
        role_winners: dict[str, dict[str, Any]] = {}
        for role_class in sorted({_coarse_role(dataset) for dataset in protocol.DEVELOPMENT_PANEL}):
            candidates = [(value, dataset, arm) for (dataset, arm), value in delta_random.items() if _coarse_role(dataset) == role_class]
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            if candidates:
                value, dataset, arm = candidates[0]
                role_winners[role_class] = {"dataset": dataset, "arm": arm, "delta_random": value, "material": bool(value >= protocol.MATERIAL_DELTA_ARI)}
        material_winners = [item for item in role_winners.values() if item["material"]]
        distinct_arms = sorted({item["arm"] for item in material_winners})
        arm_material_dataset_counts = {
            arm: sum(
                delta_random[(dataset, arm)] >= protocol.MATERIAL_DELTA_ARI
                for dataset in protocol.DEVELOPMENT_PANEL
            )
            for arm in protocol.STRUCTURED_ARMS
        }
        simple_arms = [arm for arm, count in arm_material_dataset_counts.items() if count >= protocol.SIMPLE_MIN_DATASET_COUNT]
        if not level1_material:
            status = "corruption_not_current_bottleneck"
        elif len(material_winners) >= 2 and len(distinct_arms) >= 2:
            status = "adaptive_corruption_opportunity_present"
        elif simple_arms:
            status = "simple_corruption_principle_sufficient"
        else:
            status = "random_corruption_sufficient"
        decision = {
            "stage": "B1",
            "status": status,
            "primary_gate_pass": status in {"simple_corruption_principle_sufficient", "adaptive_corruption_opportunity_present", "random_corruption_sufficient"},
            "next_stage_authorized": status == "adaptive_corruption_opportunity_present",
            "authorized_next_stage": "B2" if status == "adaptive_corruption_opportunity_present" else None,
            "terminal_reason": None if status == "adaptive_corruption_opportunity_present" else "B1 hierarchy does not justify adaptive location/generator work",
            "level_1": {"delta_clean_C0": delta_clean_c0, "H_corr": h_corr, "material": level1_material},
            "level_2": {"delta_random": {f"{dataset}::{arm}": value for (dataset, arm), value in delta_random.items()}, "role_winners": role_winners},
            "level_3": {"material_role_winner_count": len(material_winners), "distinct_structured_winner_arms": distinct_arms, "material_dataset_count_by_arm": arm_material_dataset_counts, "simple_principle_arms": simple_arms},
        }
    dataset_summary: list[dict[str, Any]] = []
    if complete:
        for dataset in protocol.DEVELOPMENT_PANEL:
            clean = float(np.mean([r["ARI"] for r in valid if r["dataset"] == dataset and r["arm"] == "C_clean_no_corruption"]))
            c0 = float(np.mean([r["ARI"] for r in valid if r["dataset"] == dataset and r["arm"] == "C0_MatchedRandom"]))
            best_structured = max(protocol.STRUCTURED_ARMS, key=lambda arm: float(np.mean([r["ARI"] for r in valid if r["dataset"] == dataset and r["arm"] == arm])))
            best = float(np.mean([r["ARI"] for r in valid if r["dataset"] == dataset and r["arm"] == best_structured]))
            dataset_summary.append({"dataset": dataset, "role_class": _coarse_role(dataset), "ARI_clean": clean, "ARI_C0": c0, "Delta_clean_C0": c0 - clean, "best_structured_arm": best_structured, "best_structured_ARI": best, "Delta_random_best_structured": best - c0})
    _write_csv(root / "b1_run_summary.csv", rows)
    _write_csv(root / "b1_dataset_summary.csv", dataset_summary)
    write_json(root / "positive_control.json", positive)
    write_json(root / "decision.json", decision)
    write_json(root / "audit.json", {
        "project_id": protocol.PROJECT_ID,
        "stage": "B1",
        "status": decision["status"],
        "positive_control_passed": positive.get("status") == "completed_valid",
        "expected_run_count": len(protocol.DEVELOPMENT_PANEL) * len(protocol.CORRUPTION_ARMS) * len(protocol.PRIMARY_SEEDS),
        "completed_valid_run_count": len(valid),
        "all_jobs_completed_valid": complete,
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "raw_artifacts_published": False,
        "gpu_pool": list(protocol.LEGAL_GPU_POOL),
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
    })
    write_json(root / "resolved_config.json", {**protocol.resolved_config(), "stage": "B1", "positive_control_status": positive.get("status"), "raw_artifacts_published": False})
    write_json(root / "run_manifest.json", {"project_id": protocol.PROJECT_ID, "stage": "B1", "expected_jobs": len(rows), "completed_valid": len(valid), "status": decision["status"], "publication_scope": "compact summaries only"})
    write_json(root / "artifact_hashes.json", {"stage": "B1", "files": {path.name: sha256_file(path) for path in sorted(root.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}, "raw_artifacts_included": False})
    return {"decision": decision, "audit": json.loads((root / "audit.json").read_text(encoding="utf-8")), "dataset_summary": dataset_summary}


def run_matrix(root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    protocol.validate_contract()
    root.mkdir(parents=True, exist_ok=True)
    positive = positive_control()
    write_json(root / "positive_control.json", positive)
    if positive.get("status") != "completed_valid":
        return aggregate(root, positive)
    jobs = [(dataset, arm, seed) for dataset in protocol.DEVELOPMENT_PANEL for arm in protocol.CORRUPTION_ARMS for seed in protocol.PRIMARY_SEEDS]
    gpu_pool = tuple(protocol.LEGAL_GPU_POOL)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_pool)) as executor:
        futures = [executor.submit(_run_one_subprocess, dataset, arm, seed, root, gpu_pool[index % len(gpu_pool)]) for index, (dataset, arm, seed) in enumerate(jobs)]
        for future in as_completed(futures):
            results.append(future.result())
    return aggregate(root, positive)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=protocol.DEVELOPMENT_PANEL)
    parser.add_argument("--arm", choices=protocol.CORRUPTION_ARMS)
    parser.add_argument("--seed", type=int, choices=protocol.PRIMARY_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--positive-control", action="store_true")
    parser.add_argument("--run-matrix", action="store_true")
    args = parser.parse_args()
    if args.positive_control:
        print(json.dumps(positive_control(), indent=2, sort_keys=True, default=_json_default))
    elif args.run_matrix:
        print(json.dumps(run_matrix(args.output_dir), indent=2, sort_keys=True, default=_json_default))
    elif args.dataset and args.arm and args.seed is not None:
        print(json.dumps(run_job(args.dataset, args.arm, args.seed, args.output_dir), indent=2, sort_keys=True, default=_json_default))
    else:
        parser.error("choose --positive-control, --run-matrix, or --dataset/--arm/--seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
