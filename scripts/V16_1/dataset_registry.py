from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from methods.TopoGate.V16_1_predictive_graph_gate.sparse import registered_count_semantics


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("count_candidate_registry.json")


def load_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    registry_path = DEFAULT_REGISTRY_PATH if path is None else Path(path)
    if not registry_path.exists():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset registry must be a mapping")
    return {str(name): dict(metadata) for name, metadata in payload.items()}


def resolve_metadata(dataset_name: str, registry: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    name = str(dataset_name)
    stem = Path(name).stem
    metadata = dict((registry or {}).get(name, (registry or {}).get(stem, {})))
    semantics, source = registered_count_semantics(name)
    if semantics is None:
        semantics, source = registered_count_semantics(stem)
    metadata.setdefault("count_semantics", semantics)
    metadata.setdefault("semantics_source", source)
    metadata.setdefault("source_version", "local_snapshot_unversioned")
    return metadata
