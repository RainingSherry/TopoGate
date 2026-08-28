from __future__ import annotations

import json
import inspect
from pathlib import Path

from scripts.V19.tune_ari_dev import (
    DEFAULT_CONFIG,
    FORMAL_SEEDS,
    PROTOCOL_ID,
    TARGET_DATASET_IDS,
    build_stage_spec,
    catalog,
    load_manifest,
)
from methods.TopoGate.V19_rg_adapter.trainer import fit_predict
from scripts.V19.run_ari_final import FINAL_VARIANTS, _variant_configs


MANIFEST = "/home/luolie/ToPoGate/result/V19/v19_rg_dataset_manifest_20260808.json"


def test_ari_catalog_and_locked_target_manifest() -> None:
    manifest = load_manifest(Path(MANIFEST))
    assert len(catalog()) == 48
    assert tuple(row["dataset_id"] for row in manifest["datasets"] if row.get("status") == "eligible" and row.get("input_protocol") in {"clubench_bridge", "shared_text"}) == TARGET_DATASET_IDS


def test_ari_stage_run_counts_and_seed_contract() -> None:
    manifest = load_manifest(Path(MANIFEST))
    records = [row for row in manifest["datasets"] if row["dataset_id"] in TARGET_DATASET_IDS]
    screen = build_stage_spec(manifest, records, catalog(), "screen", (42,), DEFAULT_CONFIG)
    assert screen["expected_runs"] == 384
    assert len(screen["expected_run_keys"]) == 384
    top12 = catalog([row["candidate_id"] for row in catalog()[:12]])
    refine = build_stage_spec(manifest, records, top12, "refine", FORMAL_SEEDS, DEFAULT_CONFIG)
    assert refine["expected_runs"] == 288
    assert len(refine["expected_run_keys"]) == 288
    assert screen["labels_used_during_fit"] is False
    assert screen["labels_used_for_selection"] is True
    assert screen["selection_evidence_type"] == "ARI-selected development evidence"


def test_ari_stage_spec_is_json_serialisable() -> None:
    manifest = load_manifest(Path(MANIFEST))
    records = [row for row in manifest["datasets"] if row["dataset_id"] in TARGET_DATASET_IDS]
    spec = build_stage_spec(manifest, records, catalog(), "screen", (42,), DEFAULT_CONFIG)
    json.dumps(spec)
    assert spec["protocol_id"] == PROTOCOL_ID


def test_ari_final_variants_keep_backbone_fixed_and_fit_has_no_y_argument() -> None:
    selected = {
        "protocol_id": PROTOCOL_ID,
        "stage": "refine",
        "candidate_id": "default",
        "overrides": {},
    }
    configs = _variant_configs(DEFAULT_CONFIG, selected)
    assert tuple(configs) == FINAL_VARIANTS
    assert all(config.hidden_size == 128 for config in configs.values())
    assert all(config.epochs == 80 for config in configs.values())
    assert "y" not in inspect.signature(fit_predict).parameters
