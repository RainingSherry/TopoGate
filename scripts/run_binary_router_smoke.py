#!/usr/bin/env python
"""
Direction B (BinaryRouter) smoke test — 3 datasets × 3 variants × 1 seed.

Purpose: Verify the BinaryRouter mechanism works and compare:
  1. BinaryRouter  (gate_mode=binary)     — hard routing
  2. LearnableGate (gate_mode=learned)     — v2, continuous gate
  3. nomix         (mix_mode=none)        — no neighbor mixing

Key question:
  - Can BinaryRouter match or beat nomix on datasets where nomix >> full?
  - Can BinaryRouter preserve full's advantage on datasets where full > nomix?

Datasets (seed=42):
  - enron:    nomix=0.875 >> full=0.768 >> BinaryRouter=?
  - har:      nomix=0.458 << full=0.558 >> BinaryRouter=?
  - Mouse_retina: nomix≈full >> BinaryRouter=?

All use the same hparams: epochs=80, mask_ratio=0.4, neighbor_k=5.

Outputs:
  result/binary_router_smoke/<dataset>/<dataset>__<variant>__seed42.json
  result/binary_router_smoke/merged_summary.csv
  result/binary_router_smoke/routing_analysis.json   (per-epoch routing decisions)
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_DATASET_ROOTS = [
    Path(__file__).resolve().parent.parent / "datasets",
    Path("/data/luolie/ToPoGate/datasets"),
]
DATASET_ROOT = _DATASET_ROOTS[0]
RESULT_ROOT = Path(__file__).resolve().parent.parent / "result" / "binary_router_smoke"
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── 3 datasets ────────────────────────────────────────────────────────────────
DATASETS = [
    "enron",
    "har",
    "Mouse_retina",
]

# ── 3 variants ──────────────────────────────────────────────────────────────
VARIANTS = [
    "binary_router",
    "learnable_gate_sched",
    "nomix",
]

VARIANT_CONFIG = {
    "binary_router": dict(
        gate_mode="binary",
        mix_mode="reliability",
        variant="binary_router",
    ),
    "learnable_gate_sched": dict(
        gate_mode="learned",
        mix_mode="reliability",
        variant="learnable_gate_sched",
    ),
    "nomix": dict(
        gate_mode="learned",
        mix_mode="none",      # no neighbor mixing
        variant="learnable_gate_sched",
    ),
}

# ── Shared hparams (matches the learnable_gate defaults) ────────────────────
FIXED_HPARAMS = dict(
    epochs=80,
    mask_ratio=0.4,
    neighbor_k=5,
    hidden_size=128,
    batch_size=256,
    lr=0.001,
    pseudo_weight=0.3,
    warmup_epochs=20,
    ramp_epochs=10,
    learned_gate_init_mode="zero",
    enhanced_stats=4,
    # BinaryRouter specific
    router_init_temp=5.0,
    router_temp_min=0.01,
    router_warmup_epochs=20,
    router_ramp_epochs=10,
)

# ── Runner ──────────────────────────────────────────────────────────────────


def get_dataset_path(name: str) -> str:
    # Search all configured dataset roots
    for root in _DATASET_ROOTS:
        compressed = root / name
        if compressed.is_dir():
            return str(compressed)
        npz = root / f"{name}.npz"
        if npz.exists():
            return str(npz)
    raise FileNotFoundError(f"Dataset not found: {name}")


def run_single(dataset: str, variant: str, seed: int = 42) -> dict | None:
    cfg = VARIANT_CONFIG[variant]
    save_dir = RESULT_ROOT / dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{dataset}__{variant}__seed{seed}"
    out_json = save_dir / f"{out_name}.json"

    if out_json.exists():
        print(f"  [SKIP] {out_name} already exists, skipping.")
        with open(out_json) as f:
            return json.load(f)

    data_path = get_dataset_path(dataset)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "methods" / "TopoGate" / "learnable_gate" / "run_npz.py"),
        "--data_path", data_path,
        "--save_dir", str(save_dir / out_name),
        "--dataset_name", dataset,
        "--seed", str(seed),
        "--gpu", "3",
        "--epochs", str(FIXED_HPARAMS["epochs"]),
        "--mask_ratio", str(FIXED_HPARAMS["mask_ratio"]),
        "--neighbor_k", str(FIXED_HPARAMS["neighbor_k"]),
        "--hidden_size", str(FIXED_HPARAMS["hidden_size"]),
        "--batch_size", str(FIXED_HPARAMS["batch_size"]),
        "--lr", str(FIXED_HPARAMS["lr"]),
        "--pseudo_weight", str(FIXED_HPARAMS["pseudo_weight"]),
        "--warmup_epochs", str(FIXED_HPARAMS["warmup_epochs"]),
        "--ramp_epochs", str(FIXED_HPARAMS["ramp_epochs"]),
        "--learned_gate_init_mode", str(FIXED_HPARAMS["learned_gate_init_mode"]),
        "--enhanced_stats", str(FIXED_HPARAMS["enhanced_stats"]),
        "--router_init_temp", str(FIXED_HPARAMS["router_init_temp"]),
        "--router_temp_min", str(FIXED_HPARAMS["router_temp_min"]),
        "--router_warmup_epochs", str(FIXED_HPARAMS["router_warmup_epochs"]),
        "--router_ramp_epochs", str(FIXED_HPARAMS["router_ramp_epochs"]),
        "--gate_mode", str(cfg["gate_mode"]),
        "--mix_mode", str(cfg["mix_mode"]),
        "--variant_name", f"{cfg['variant']}_{variant}",
        "--n_top_genes", "0",
    ]

    print(f"  [RUN ] {out_name}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT)
        )
        if result.returncode != 0:
            print(f"  [FAIL] {out_name}: {result.stderr[-500:]}")
            return None

        with open(out_json) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {out_name} (>15min)")
        return None
    except Exception:
        print(f"  [ERROR] {out_name}: {traceback.format_exc()[-300:]}")
        return None


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    routing_histories = {}

    for dataset in DATASETS:
        print(f"\n=== {dataset} ===")
        for variant in VARIANTS:
            result = run_single(dataset, variant, seed=42)
            if result:
                metrics = result.get("metrics", {})
                ari = metrics.get("ari", None)
                acc = metrics.get("acc", None)
                nmi = metrics.get("nmi", None)
                print(f"  {variant}: ARI={ari:.4f} ACC={acc:.4f} NMI={nmi:.4f}")
                results.append({
                    "dataset": dataset,
                    "variant": variant,
                    "ari": ari,
                    "acc": acc,
                    "nmi": nmi,
                })

                # Load beta history if binary_router
                if variant == "binary_router":
                    summary_path = Path(result.get("_source_dir", "")) / "summary.json"
                    # Try to find it
                    save_dir = RESULT_ROOT / dataset
                    candidates = list(save_dir.glob(f"{dataset}__binary_router__seed42*"))
                    if candidates:
                        summary_file = candidates[0] / "summary.json"
                        if summary_file.exists():
                            with open(summary_file) as f:
                                summary = json.load(f)
                            routing_histories[dataset] = summary.get("beta_history", [])

    # Save merged summary
    summary_path = RESULT_ROOT / "merged_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "ari", "acc", "nmi"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {summary_path}")

    # Print comparison table
    print("\n=== Comparison Table ===")
    print(f"{'Dataset':<30} {'binary_router':>14} {'learnable_gate':>14} {'nomix':>14} {'v2 vs nomix':>12} {'Binary vs nomix':>14}")
    print("-" * 90)
    prev = {}
    for r in results:
        prev[r["dataset"]] = prev.get(r["dataset"], {})
        prev[r["dataset"]][r["variant"]] = r["ari"]

    for ds in DATASETS:
        br = prev.get(ds, {}).get("binary_router", None)
        lg = prev.get(ds, {}).get("learnable_gate_sched", None)
        nm = prev.get(ds, {}).get("nomix", None)
        v2_vs_nm = (lg - nm) if (lg is not None and nm is not None) else None
        br_vs_nm = (br - nm) if (br is not None and nm is not None) else None
        br_s = f"{br:.4f}" if br is not None else "N/A"
        lg_s = f"{lg:.4f}" if lg is not None else "N/A"
        nm_s = f"{nm:.4f}" if nm is not None else "N/A"
        v2_s = f"{v2_vs_nm:+.4f}" if v2_vs_nm is not None else "N/A"
        br_s2 = f"{br_vs_nm:+.4f}" if br_vs_nm is not None else "N/A"
        print(f"{ds:<30} {br_s:>14} {lg_s:>14} {nm_s:>14} {v2_s:>12} {br_s2:>14}")

    # Save routing analysis
    if routing_histories:
        print("\n=== Routing Analysis (BinaryRouter β snapshots) ===")
        for ds, hist in routing_histories.items():
            if not hist:
                continue
            print(f"\n{ds}:")
            # Show last 5 epochs
            for snap in hist[-5:]:
                temp = snap.get("temperature", "N/A")
                bmut = snap.get("beta_mutual", 0)
                bsnn = snap.get("beta_snn", 0)
                bpert = snap.get("beta_perturb", 0)
                print(f"  epoch {snap.get('epoch','?')}: temp={temp:.3f} β=({bmut:.2f},{bsnn:.2f},{bpert:.2f})")


if __name__ == "__main__":
    main()
