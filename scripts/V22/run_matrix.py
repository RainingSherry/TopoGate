#!/usr/bin/env python3
"""Build or run the preregistered V22 dataset/variant/seed matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "datasets" / "external" / "v22_dataset_extension_20260812" / "manifest.json"
CONFIG_DIR = ROOT / "methods" / "TopoGate" / "V22_topology_discriminator_hard_mask" / "configs"
GPU_POOL = (1, 2, 3, 4, 5, 6)
VARIANT_CONFIGS = {
    "scmae_only": CONFIG_DIR / "v22_scmae_only.yaml",
    "scmae_always_visible": CONFIG_DIR / "v22_scmae_always_visible.yaml",
    "scmae_plus_discriminator_random_mask": CONFIG_DIR / "v22_discriminator_random_mask.yaml",
    "scmae_plus_discriminator_reconstruction_hard_gate": CONFIG_DIR / "v22_reconstruction_hard_gate.yaml",
    "scmae_plus_discriminator_learned_non_topology_gate": CONFIG_DIR / "v22_learned_non_topology_gate.yaml",
    "v22_topology_discriminator_hard_gate": CONFIG_DIR / "v22_topology_discriminator_hard_gate.yaml",
    "v22_topology_discriminator_cooperative_keep_gate": CONFIG_DIR
    / "v22_topology_discriminator_cooperative_keep_gate.yaml",
}


def _load_manifest(path: Path = MANIFEST, *, require_sources: bool = True) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [row for row in payload.get("datasets", []) if row.get("status", "").startswith("eligible")]
    if not records:
        raise ValueError("V22 dataset manifest has no eligible records; finish dataset preparation first")
    for row in records:
        for key in ("dataset_id", "name", "source_path", "input_protocol"):
            if key not in row:
                raise ValueError(f"manifest record missing {key}: {row}")
        if require_sources and not Path(row["source_path"]).is_file():
            raise FileNotFoundError(row["source_path"])
    return {**payload, "datasets": records}


def build_jobs(
    manifest: dict[str, Any],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    output_root: Path,
    selected: set[str] | None = None,
    n_clusters_by_dataset: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    records = manifest["datasets"]
    if selected:
        records = [row for row in records if str(row["dataset_id"]) in selected]
        found = {str(row["dataset_id"]) for row in records}
        if found != selected:
            raise ValueError(f"unknown V22 dataset ids: {sorted(selected - found)}")
    n_clusters_by_dataset = n_clusters_by_dataset or {}
    unknown_k = set(n_clusters_by_dataset) - {str(row["dataset_id"]) for row in records}
    if unknown_k:
        raise ValueError(f"n_clusters supplied for unknown/unselected V22 datasets: {sorted(unknown_k)}")
    for dataset_id, value in n_clusters_by_dataset.items():
        if int(value) <= 1:
            raise ValueError(f"explicit n_clusters for {dataset_id!r} must be greater than one")
    jobs: list[dict[str, Any]] = []
    for record in records:
        dataset_id = str(record["dataset_id"])
        explicit_k = n_clusters_by_dataset.get(dataset_id)
        for variant in variants:
            if variant not in VARIANT_CONFIGS:
                raise ValueError(f"unsupported V22 variant: {variant}")
            for seed in seeds:
                run_key = f"{manifest['manifest_id']}::{dataset_id}::{variant}::seed{seed}"
                jobs.append(
                    {
                        "run_key": run_key,
                        "dataset_id": dataset_id,
                        "record": record,
                        "variant": variant,
                        "seed": int(seed),
                        "n_clusters": None if explicit_k is None else int(explicit_k),
                        "config": str(VARIANT_CONFIGS[variant]),
                        "output_dir": output_root / dataset_id / variant / f"seed{seed}",
                    }
                )
    return jobs


def _requires_explicit_k(record: dict[str, Any]) -> bool:
    """Return whether the manifest intentionally has no benchmark labels."""
    return str(record.get("status", "")).endswith("unlabelled") or record.get("evaluation_status") == (
        "no_ari_without_external_labels"
    )


def _parse_n_clusters(specs: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--n-clusters expects DATASET_ID=K, got {spec!r}")
        dataset_id, raw_k = spec.split("=", 1)
        dataset_id = dataset_id.strip()
        if not dataset_id:
            raise ValueError(f"--n-clusters has an empty dataset id: {spec!r}")
        try:
            value = int(raw_k)
        except ValueError as exc:
            raise ValueError(f"--n-clusters K must be an integer: {spec!r}") from exc
        if value <= 1:
            raise ValueError(f"--n-clusters K must be greater than one: {spec!r}")
        if dataset_id in mapping:
            raise ValueError(f"duplicate --n-clusters mapping for {dataset_id!r}")
        mapping[dataset_id] = value
    return mapping


def _completed(job: dict[str, Any]) -> bool:
    summary_path = Path(job["output_dir"]) / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "completed"
        and summary.get("variant") == job["variant"]
        and int(summary.get("seed", -1)) == int(job["seed"])
        and summary.get("dataset") == job["record"]["name"]
    )


def _run_job(job: dict[str, Any], device: str, gpu: int | None, epochs: int | None) -> None:
    command = [
        sys.executable,
        "-m",
        "methods.TopoGate.V22_topology_discriminator_hard_mask.run",
        "--data",
        str(job["record"]["source_path"]),
        "--dataset-name",
        str(job["record"]["name"]),
        "--input-protocol",
        str(job["record"]["input_protocol"]),
        "--config",
        str(job["config"]),
        "--output-dir",
        str(job["output_dir"]),
        "--seed",
        str(job["seed"]),
        "--device",
        device,
    ]
    if device == "cuda":
        if gpu not in GPU_POOL:
            raise ValueError(f"V22 CUDA jobs require --gpu in {GPU_POOL}")
        command.extend(["--gpu", str(gpu)])
    if epochs is not None:
        command.extend(["--epochs", str(int(epochs))])
    if job.get("n_clusters") is not None:
        command.extend(["--n-clusters", str(int(job["n_clusters"]))])
    Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result" / "V22" / "formal_matrix_20260812")
    parser.add_argument("--variants", nargs="*", default=list(VARIANT_CONFIGS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 7])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--n-clusters",
        action="append",
        default=[],
        metavar="DATASET_ID=K",
        help="explicit K for an unlabelled dataset; repeat once per dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-sources",
        action="store_true",
        help="allow provenance-only manifests whose dataset files are not present; execution still requires real files",
    )
    args = parser.parse_args()
    manifest = _load_manifest(
        args.manifest,
        require_sources=not args.allow_missing_sources,
    )
    if args.allow_missing_sources and not args.dry_run:
        raise ValueError("--allow-missing-sources is valid only with --dry-run")
    n_clusters_by_dataset = _parse_n_clusters(args.n_clusters)
    jobs = build_jobs(
        manifest,
        tuple(args.variants),
        tuple(args.seeds),
        args.output_dir,
        set(args.datasets) if args.datasets else None,
        n_clusters_by_dataset,
    )
    missing_k = sorted(
        {
            str(job["dataset_id"])
            for job in jobs
            if _requires_explicit_k(job["record"]) and job.get("n_clusters") is None
        }
    )
    if missing_k and not args.dry_run:
        raise ValueError(
            "unlabelled V22 datasets require explicit K before execution: "
            + ", ".join(missing_k)
            + "; pass --n-clusters DATASET_ID=K"
        )
    states = []
    for job in jobs:
        state = "reused" if _completed(job) else "queued"
        states.append(
            {
                "run_key": job["run_key"],
                "state": state,
                "output_dir": str(job["output_dir"]),
                "n_clusters": job.get("n_clusters"),
                "requires_explicit_n_clusters": _requires_explicit_k(job["record"]),
            }
        )
        if not args.dry_run and state == "queued":
            _run_job(job, args.device, args.gpu, args.epochs)
            states[-1]["state"] = "completed"
    print(json.dumps({"manifest_id": manifest["manifest_id"], "jobs": states}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
