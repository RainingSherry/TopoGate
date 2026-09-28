#!/usr/bin/env python3
"""Bounded worker for advantage_story_20260927: fit one task, save artifacts.

Usage:
    python fit_worker.py --task task.json --out /path/to/output --device cuda:0

Remote base: /data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922
Imports unmodified v0rg_tunable_multihop + existing helpers via sys.path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import asdict

for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'2')

import numpy as np
import torch

# Add remote base and paper helpers to path
REMOTE_BASE = Path("/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922")
PAPER_DIR = Path(__file__).resolve().parent.parent

for path in [Path('/home/luolie/ToPoGate'), REMOTE_BASE, PAPER_DIR / "source_coverage_20260925", PAPER_DIR / "multihop_mechanism_20260924"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v0rg_tunable_multihop.config import V0_RGConfig
from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
from v0rg_tunable_multihop.tuning import SplitPreprocessor

# Import from existing papers
import sys
sys.path.insert(0, str(PAPER_DIR / "source_coverage_20260925"))
from run_factorial import fit_arm, patched_training, TraceCollector
from factorial_plan import plan_arrays
from intervention_plan import _kwide
from common import json_safe
torch.set_num_threads(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_and_validate_task(task_path: Path) -> dict:
    with open(task_path) as f:
        task = json.load(f)
    manifest=json.loads(Path(__file__).with_name('REUSE_MANIFEST.json').read_text())
    case=manifest['cases'][task['dataset']]
    for key in ('feature_path','feature_sha256','input_kind','feature_limit','preprocessing_hash','n_clusters','K_provenance'):
        if task[key]!=case[key]:raise ValueError(f'Frozen manifest mismatch: {key}')
    expected=asdict(V0_RGConfig(**case['expected_config_template']))
    supplied=asdict(V0_RGConfig(**task['reference_config']))
    if {k:v for k,v in expected.items() if k!='protocol_id'}!={k:v for k,v in supplied.items() if k!='protocol_id'}:
        raise ValueError('Reference config differs from frozen case manifest')
    if task['core_hashes']!=manifest['core_source_sha256']:
        raise ValueError('Expected source hashes differ from frozen manifest')
    
    # Verify no label fields
    forbidden = ["labels_path", "label_path", "test_labels", "validation_labels"]
    for key in task:
        if any(bad in key.lower() for bad in forbidden):
            raise ValueError(f"Task contains forbidden label field: {key}")
    
    # Validate strategy
    if task["strategy"] not in ["path", "kwide"]:
        raise ValueError(f"Invalid strategy: {task['strategy']}")
    
    # Validate K provenance
    if task.get("K_provenance") != "benchmark_oracle_from_y":
        raise ValueError("K_provenance must be benchmark_oracle_from_y")
    
    # Validate config match
    ref_config = V0_RGConfig(**task["reference_config"])
    actual_config = V0_RGConfig(**task["config"])
    
    # Reference must equal manifest template (excluding protocol_id)
    ref_dict = {k: v for k, v in task["reference_config"].items() if k != "protocol_id"}
    
    # For path strategy, config differs only in hop_variant
    if task["strategy"] == "path":
        allowed_diffs = {"hop_variant"}
        diffs = {k for k in ref_dict if ref_dict.get(k) != task["config"].get(k)}
        if not diffs <= allowed_diffs:
            raise ValueError(f"Path config differs beyond hop_variant: {diffs}")
        
        # Validate hop_variant format (H1, H2_B*, H4_B*, H8_B*)
        hop = actual_config.hop_variant
        width=ref_config.hop_variant.split('_')[1]
        if hop not in ['H1']+[f'H{h}_{width}' for h in (2,4,8)]:
            raise ValueError(f"Invalid hop_variant for path: {hop}")
    
    # For kwide, config must exactly match reference
    elif task["strategy"] == "kwide":
        diffs = {k for k in ref_dict if ref_dict.get(k) != task["config"].get(k)}
        if diffs:
            raise ValueError(f"Kwide config must exactly match reference, differs in: {diffs}")
    
    # Validate seed
    if task["seed"] not in [7, 42, 123]:
        raise ValueError(f"Invalid seed: {task['seed']}")
    
    return task


def load_features_with_validation(task: dict) -> tuple:
    feature_path = Path(task["feature_path"])
    
    # Verify SHA256
    actual_sha = sha256_file(feature_path)
    if actual_sha != task["feature_sha256"]:
        raise ValueError(f"Feature SHA256 mismatch: {actual_sha} != {task['feature_sha256']}")
    
    # Load matrix
    loaded = load_npz_matrix_only(feature_path)
    if loaded.labels is not None or loaded.profile.get('labels_accessed') is not False:
        raise RuntimeError("Feature loader accessed labels!")
    
    # Fit preprocessor and verify fingerprint
    prep = SplitPreprocessor(task["input_kind"], task["feature_limit"]).fit(loaded.X)
    if prep.fingerprint != task["preprocessing_hash"]:
        raise ValueError(f"Preprocessing hash mismatch: {prep.fingerprint} != {task['preprocessing_hash']}")
    
    X_full = prep.transform(loaded.X)
    return prep, X_full, loaded.X


class TraceCollector9(TraceCollector):
    """TraceCollector with max_hop=9 for kwide's unreachable nodes."""
    def __init__(self, path: Path | None, n_samples: int):
        super().__init__(path, n_samples, max_hop=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task JSON file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()
    
    task_path = Path(args.task).resolve()
    out_dir = Path(args.out).resolve()
    
    # Check for existing record
    if (out_dir / "record.json").exists():
        raise RuntimeError("Output directory already contains record.json")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and validate
    print("Loading task...")
    task = load_and_validate_task(task_path)
    for rel, expected in task['core_hashes'].items():
        if sha256_file(REMOTE_BASE/rel)!=expected:raise ValueError(f'Imported source lock mismatch: {rel}')
    
    print("Loading features...")
    prep, X_full, X_raw = load_features_with_validation(task)
    
    # Record core hashes before
    from run_factorial import imported_core_hashes
    core_before = imported_core_hashes()
    
    config = V0_RGConfig(**task["config"])
    ref_config = V0_RGConfig(**task["reference_config"])
    seed = task["seed"]
    n_clusters = task["n_clusters"]
    
    # Determine prior and anchor type
    is_h1 = config.hop_variant == "H1"
    prior = {"kind": "flat_q", "q": ref_config.high_hop_probability_multiplier}
    
    start_time = time.time()
    
    if task["strategy"] == "path":
        # Standard path training
        print(f"Training path strategy: {config.hop_variant}")
        
        trace_path = out_dir / "source_trace.bin"
        
        result = fit_arm(X_full, task["config"], "flat_anchor" if not is_h1 else "H1_anchor",
                        prior, seed, args.device, n_clusters, trace_path, out_dir)
    
    else:  # kwide
        print("Training kwide strategy...")
        
        # Use TraceCollector9 for hop code 9
        trace_path = out_dir / "source_trace.bin"
        collector = TraceCollector9(trace_path, X_full.shape[0])
        
        # Wrapper to capture reference plan then transform with _kwide
        from v0rg_tunable_multihop import trainer
        original_builder = trainer.build_exact_hop_plan
        
        captured = {}
        
        def kwide_wrapper(graph, edge_weights, hop_variant, mix_neighbors, q=1.0, *a, **k):
            # Build reference flat plan
            reference = original_builder(graph, edge_weights, hop_variant, mix_neighbors, q, *a, **k)
            captured["reference"] = reference
            captured["graph"] = graph
            captured["edge_weights"] = edge_weights
            
            # Transform to kwide
            before_indices=graph.indices.copy(); before_weights=edge_weights.copy()
            plan, _ = _kwide(reference, graph, edge_weights, config)
            if not np.array_equal(before_indices,graph.indices) or not np.array_equal(before_weights,edge_weights):
                raise RuntimeError('Kwide changed original graph or edge weights')
            
            # Verify counts preserved
            ref_counts = [len(r) for r in reference.indices]
            plan_counts = [len(r) for r in plan.indices]
            if ref_counts != plan_counts:
                raise RuntimeError("Kwide changed candidate counts")
            if plan.draw_count != reference.draw_count:
                raise RuntimeError("Kwide changed draw_count")
            
            captured["plan"] = plan
            return plan
        
        # Temporarily replace builder
        try:
            trainer.build_exact_hop_plan = kwide_wrapper
            
            # Patch with traced estimator
            collector.amplitude = float(config.mix_amplitude_multiplier)
            captured_inner = {}
            
            with patched_training("flat_anchor", prior, collector, captured_inner):
                from torch.utils.data import RandomSampler
                from v0rg_tunable_multihop import trainer as trainer_inner
                
                loader_original = trainer_inner.DataLoader
                epoch_state = {"n": 0}
                
                class EpochLoader(loader_original):
                    def __iter__(self):
                        if isinstance(self.sampler, RandomSampler):
                            epoch_state["n"] += 1
                            collector.epoch = epoch_state["n"]
                        return super().__iter__()
                
                trainer_inner.DataLoader = EpochLoader
                
                try:
                    predictions, embedding, diagnostics = trainer_inner.fit_predict(
                        X_full, fit_X=X_full, n_clusters=n_clusters, config=config,
                        seed=seed, device=args.device)
                finally:
                    trainer_inner.DataLoader = loader_original
        
        except BaseException:
            collector.abort()
            raise
        finally:
            trainer.build_exact_hop_plan = original_builder
        
        trace = collector.finish(out_dir)
        
        result = {
            "predictions": predictions,
            "embedding": embedding,
            "diagnostics": diagnostics,
            "captured": captured,
            "trace": trace,
            "epochs_traced": epoch_state["n"]
        }
    
    elapsed = time.time() - start_time
    
    # Verify core unchanged
    core_after = imported_core_hashes()
    if core_after != core_before:
        raise RuntimeError("Core modules changed during training")
    for rel, expected in task['core_hashes'].items():
        if sha256_file(REMOTE_BASE/rel)!=expected:raise RuntimeError(f'Source changed: {rel}')
    
    # Extract results
    predictions = result["predictions"]
    embedding = result["embedding"]
    diagnostics = result["diagnostics"]
    
    # Validate outputs
    if predictions is None or len(predictions) != X_full.shape[0]:
        raise RuntimeError("Invalid predictions")
    if not np.isfinite(embedding).all():
        raise RuntimeError("Non-finite embedding values")
    if result['epochs_traced']!=config.epochs:raise RuntimeError('Incomplete epoch count')
    expected_occ=len(X_full)*config.epochs
    trace=result['trace']
    from run_factorial import TRACE_DTYPE
    if trace['anchor_training_occurrences']!=expected_occ or trace['draws']!=expected_occ*config.mix_neighbors:
        raise RuntimeError('Trace count does not match frozen epochs/draws')
    if trace_path.stat().st_size!=trace['draws']*TRACE_DTYPE.itemsize:raise RuntimeError('Trace file truncated')
    
    # Save artifacts
    print("Saving artifacts...")
    
    # Model
    torch.save(diagnostics["model_state_dict"], out_dir / "model.pt")
    
    # Embedding and predictions
    np.save(out_dir / "embedding.npy", embedding.astype(np.float32))
    np.save(out_dir / "predictions.npy", predictions.astype(np.int64))
    
    # Training history
    with open(out_dir / "history.json", "w") as f:
        json.dump(diagnostics["training_history"], f)
    
    # Plan arrays
    if task["strategy"] == "kwide":
        plan_data = plan_arrays(result["captured"]["plan"])
        np.savez(out_dir/'reference_plan.npz',**plan_arrays(result['captured']['reference']))
    else:
        if is_h1:
            from factorial_plan import h1_plan_arrays
            graph_indices = diagnostics["neighbor_indices"]
            edge_weights = diagnostics["edge_weights"]
            plan_data = h1_plan_arrays(graph_indices, edge_weights, config.mix_neighbors)
        else:
            plan_data = plan_arrays(result["captured"]["plan"])
    
    np.savez(out_dir / "plan.npz", **plan_data)
    
    # Graph data
    np.savez(out_dir / "graph.npz",
             indices=diagnostics["neighbor_indices"],
             edge_weights=diagnostics["edge_weights"],
             node_gate=diagnostics["node_gate"],
             similarity=diagnostics.get("neighbor_similarity"))
    
    # Preprocessor
    np.savez(out_dir / "preprocessor.npz",
             features=prep.features.astype(np.int64),
             mean=prep.scaler.mean_,
             scale=prep.scaler.scale_)
    
    # Trace metadata
    if result["trace"]:
        with open(out_dir / "source_trace.json", "w") as f:
            json.dump(result["trace"], f)
    
    # Compute artifact hashes
    artifact_names = ["model.pt", "embedding.npy", "predictions.npy", "history.json",
                     "plan.npz", "graph.npz", "preprocessor.npz", "source_trace.bin",
                     "source_trace.json", "displacement_by_anchor.npz"]
    if task['strategy']=='kwide':artifact_names.append('reference_plan.npz')
    
    artifact_hashes = {}
    for name in artifact_names:
        path = out_dir / name
        if not path.exists():raise RuntimeError(f'Missing required artifact {name}')
        artifact_hashes[name] = sha256_file(path)
    
    # Final record
    record = {
        "protocol": "advantage_story_20260927_v1",
        "status": "completed",
        "task_dataset": task["dataset"],
        "task_seed": seed,
        "task_strategy": task["strategy"],
        "config": task["config"],
        "reference_config": task["reference_config"],
        "feature_sha256": task["feature_sha256"],
        "preprocessing_hash": prep.fingerprint,
        "n_clusters": n_clusters,
        "K_provenance": task["K_provenance"],
        "labels_used_for_model_fit": False,
        "test_labels_loaded": False,
        "selection_label_scope": "frozen_historical_validation_config_no_new_selection",
        "new_training": True,
        "input_fullX": True,
        "fixed_epoch_final_checkpoint": True,
        "core_hashes_before": core_before,
        "core_hashes_after": core_after,
        "elapsed_seconds": elapsed,
        "artifact_sha256": artifact_hashes,
        "epochs_trained": result["epochs_traced"],
        "device": args.device
    }
    record.update(task_sha256=sha256_file(task_path),core_source_unchanged=True,
                  actual_hop_code_9='more_than_8_or_unreachable' if task['strategy']=='kwide' else None,
                  manifest_sha256=sha256_file(Path(__file__).with_name('REUSE_MANIFEST.json')))
    
    # Atomic write of record.json (completion marker)
    record_json = json.dumps(json_safe(record), indent=2,allow_nan=False).encode("utf-8")
    atomic_write(out_dir / "record.json", record_json)
    
    print(f"✓ Complete in {elapsed:.1f}s: {out_dir}")


if __name__ == "__main__":
    main()
