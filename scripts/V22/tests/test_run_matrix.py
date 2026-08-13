from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.V22.run_matrix import _load_manifest, _parse_n_clusters, build_jobs


def test_unlabelled_dataset_requires_explicit_k_mapping(tmp_path: Path) -> None:
    source = tmp_path / "placeholder.npz"
    source.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_id": "test_v22_manifest",
                "datasets": [
                    {
                        "dataset_id": "labeled",
                        "name": "labeled",
                        "source_path": str(source),
                        "input_protocol": "shared_text",
                        "status": "eligible",
                    },
                    {
                        "dataset_id": "pbmc3k__10x_unlabelled_count",
                        "name": "PBMC3k",
                        "source_path": str(source),
                        "input_protocol": "scRNA_count",
                        "status": "eligible_unlabelled",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = _load_manifest(manifest_path)
    pbmc_id = "pbmc3k__10x_unlabelled_count"
    jobs = build_jobs(
        manifest,
        ("v22_topology_discriminator_hard_gate",),
        (42,),
        tmp_path,
    )
    pbmc_job = next(job for job in jobs if job["dataset_id"] == pbmc_id)
    assert pbmc_job["n_clusters"] is None

    mapped = build_jobs(
        manifest,
        ("v22_topology_discriminator_hard_gate",),
        (42,),
        tmp_path,
        n_clusters_by_dataset={pbmc_id: 8},
    )
    mapped_pbmc_job = next(job for job in mapped if job["dataset_id"] == pbmc_id)
    assert mapped_pbmc_job["n_clusters"] == 8


def test_n_clusters_parser_rejects_ambiguous_specs() -> None:
    assert _parse_n_clusters(["pbmc3k__10x_unlabelled_count=8"]) == {
        "pbmc3k__10x_unlabelled_count": 8
    }
    with pytest.raises(ValueError, match="DATASET_ID=K"):
        _parse_n_clusters(["pbmc3k__10x_unlabelled_count"])
    with pytest.raises(ValueError, match="duplicate"):
        _parse_n_clusters(["pbmc3k__10x_unlabelled_count=8", "pbmc3k__10x_unlabelled_count=9"])
