#!/usr/bin/env python3
"""
LearnableGate 134-dataset sweep runner.

Three-stage stratified search + final comparison:

  Stage 1 – Coarse search:  134 datasets × 4 configs × 1 seed  = 536 runs
             grid: epochs=80, mask_ratio∈{0.3,0.4}, neighbor_k∈{5,10}, gate_max=0.15

  Stage 2 – Fine search:     top-50 difficult datasets (ARI < 0.5 in Stage 1)
                            × 8 configs × 3 seeds                     = 1200 runs
             grid: epochs∈{80,150}, mask_ratio∈{0.3,0.4},
                   neighbor_k∈{5,10}, gate_max∈{0.15,0.30}

  Stage 3 – Final comparison: 134 datasets × 2 variants × 3 seeds = 804 runs
             variants: learnable_gate_sched (best config from Stage 1+2)
                       static_gate_full (baseline)

Usage (one worker per GPU, worker_id = 0..N-1):

  # Stage 1 — 3 workers
  for wid in 0 1 2; do
    python scripts/learnable_gate/run_learnable_gate_134_sweep.py \
      --stage 1 --gpu_ids 4 5 7 --worker_id $wid &
  done
  wait

  # Stage 2 — 6 workers
  for wid in 0 1 2 3 4 5; do
    python scripts/learnable_gate/run_learnable_gate_134_sweep.py \
      --stage 2 --gpu_ids 4 5 7 0 1 2 --worker_id $wid &
  done
  wait

  # Stage 3 — 3 workers
  for wid in 0 1 2; do
    python scripts/learnable_gate/run_learnable_gate_134_sweep.py \
      --stage 3 --gpu_ids 4 5 7 --worker_id $wid &
  done
  wait
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CLUBENCH_ROOT = REPO_ROOT / "baseline" / "CLUBench"
DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
RESULT_ROOT = REPO_ROOT / "result" / "learnable_gate_134_sweep"

for p in (str(REPO_ROOT), str(CLUBENCH_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 134 dataset list (from dataset_npz_info.csv — auto-generated, authoritative) ──
_DS_CSV = os.environ.get("DATASET_CSV") or str(REPO_ROOT / "result" / "dataset_npz_info.csv")
_DS_CSV = Path(_DS_CSV)
if _DS_CSV.exists():
    import csv as _csv

    with open(_DS_CSV) as _f:
        _reader = _csv.DictReader(_f)
        DATASETS = [r["dataset_name"] for r in _reader]
else:
    # Fallback: hardcoded list (134 entries from dataset directory)
    DATASETS = [
        "Baron Human", "Campbell", "hrvatin_filtered", "Mouse_retina",
        "Quake_Smart-seq2_Lung",
        "cifar10", "CIFAR10_CLIP", "coil20", "COIL20_CLIP", "extyaleb",
        "fashion_mnist", "FashionMNIST_CLIP", "flickr_material_database",
        "gina_prior2", "image_segmentation", "Indian_pines",
        "labeled_faces_in_the_wild", "mnist64", "MNIST_CLIP",
        "olivetti_faces", "optical_recognition_of_handwritten_digits",
        "PCam", "satellite_image", "statlog_image_segmentation",
        "street_view_house_numbers",
        "20newsgroups", "banknote_authentication",
        "birds_bones_and_living_habits", "blood_transfusion_service_center",
        "boston", "breast_cancer_coimbra", "breast_cancer_wisconsin_original",
        "breast_cancer_wisconsin_prognostic", "breast_tissue",
        "cardiovascular_study", "classification_in_asteroseismology", "cnae9",
        "credit_risk_classification", "crowdsourced_mapping",
        "customer_classification", "date_fruit", "dermatology",
        "diabetic_retinopathy_debrecen", "dilbert", "Drug Consumption",
        "dry_bean", "durum_wheat_features", "echocardiogram", "ecoli",
        "enron", "epileptic_seizure_recognition", "fabert", "fbis.wc",
        "fetal_health_classification", "first-order-theorem-proving",
        "fraud_detection_bank", "gas-drift", "glass_identification", "har",
        "harbermans_survival", "hate_speech",
        "heart_attack_analysis_prediction_dataset", "heart_disease",
        "hepatitis", "htru2", "human_stress_detection", "imdb",
        "insurance_company_benchmark", "ionosphere", "iris", "ISOLET",
        "JapaneseVowels", "letter_recognition", "magic_gamma_telescope",
        "mammographic_mass", "mfeat-factors", "mfeat-fourier",
        "mfeat-karhunen", "mfeat-morphological", "micro-mass", "microbes",
        "mobile_price_classification", "music_genre_classification",
        "orbit_classification_for_prediction_nasa", "paris_housing_classification",
        "parkinsons", "patient_treatment_classification",
        "pen_based_recognition_of_handwritten_digits", "ph_recognition",
        "pima_indians_diabetes_database", "pistachio", "planning_relax",
        "poker-hand", "predicting_pulsar_star", "pumpkin_seeds", "raisin",
        "reuters", "rice_dataset_cammeo_and_osmancik",
        "rice_seed_gonen_jasmine", "rmftsa_sleepdata", "secom", "seeds",
        "seismic_bumps", "sentiment_labeld_sentences", "shuttle",
        "siberian_weather_stats", "skillcraft1_master_table_dataset",
        "smoker_condition", "sms_spam_collection", "spambase",
        "spectf_heart", "statlog_german_credit", "steel-plates-fault",
        "student_grade", "synthetic_control", "tamilnadu-electricity",
        "tr45.wc", "turkish_music_emotion", "user_knowledge_modeling",
        "vehicle", "wall-robot-navigation", "water_quality", "Waveform",
        "weather", "website_phishing", "wine", "wine_customer",
        "wine_quality", "wireless_indoor_localization", "world12d", "wos",
        "yeast", "zoo",
    ]

# ── Large-dataset subsample config ───────────────────────────────────────────
LARGE_DATASET_SUBSAMPLE: Dict[str, int] = {
    "hrvatin": 10000,
    "hrvatin_filtered": 10000,
    "Campbell": 8000,
    "crowdsourced_mapping": 8000,
    "wos": 8000,
    "dilbert": 8000,
    "20newsgroups": 8000,
    "fraud_detection_bank": 8000,
    "gas-drift": 8000,
    "magic_gamma_telescope": 8000,
    "microbes": 8000,
    "poker-hand": 8000,
    "rice_seed_gonen_jasmine": 8000,
    "shuttle": 8000,
    "tamilnadu-electricity": 8000,
}

# ── Timeout per sample-size bucket ───────────────────────────────────────────
def get_timeout(n_samples: int) -> int:
    if n_samples < 5000:
        return 300
    elif n_samples < 20000:
        return 600
    else:
        return 1200


# ── Stage grids ───────────────────────────────────────────────────────────────
# Stage 1: coarse grid, seed=42
STAGE1_GRID: List[dict] = [
    dict(epochs=80, mask_ratio=0.3, neighbor_k=5,  gate_max=0.15),
    dict(epochs=80, mask_ratio=0.3, neighbor_k=10, gate_max=0.15),
    dict(epochs=80, mask_ratio=0.4, neighbor_k=5,  gate_max=0.15),
    dict(epochs=80, mask_ratio=0.4, neighbor_k=10, gate_max=0.15),
]

# Stage 2: fine grid, 3 seeds
STAGE2_GRID: List[dict] = [
    dict(epochs=80,  mask_ratio=0.3, neighbor_k=5,  gate_max=0.15),
    dict(epochs=80,  mask_ratio=0.3, neighbor_k=10, gate_max=0.15),
    dict(epochs=80,  mask_ratio=0.3, neighbor_k=5,  gate_max=0.30),
    dict(epochs=80,  mask_ratio=0.3, neighbor_k=10, gate_max=0.30),
    dict(epochs=150, mask_ratio=0.3, neighbor_k=5,  gate_max=0.15),
    dict(epochs=150, mask_ratio=0.3, neighbor_k=5,  gate_max=0.30),
    dict(epochs=150, mask_ratio=0.4, neighbor_k=5,  gate_max=0.15),
    dict(epochs=150, mask_ratio=0.4, neighbor_k=5,  gate_max=0.30),
]
STAGE2_SEEDS = [42, 123, 7]

# Stage 3: final comparison variants
STAGE3_VARIANTS = [
    "learnable_gate_sched",
    "static_gate_full",
]
STAGE3_SEEDS = [42, 123, 7]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_dataset_info() -> Dict[str, dict]:
    """Load n_samples from each .npz file."""
    info = {}
    for name in DATASETS:
        npz = DATA_DIR / f"{name}.npz"
        if npz.exists():
            try:
                data = dict(**np.load(npz))
                info[name] = {
                    "n_samples": int(data["x"].shape[0]),
                    "n_features": int(data["x"].shape[1]),
                    "n_clusters": int(len(set(data["y"].flatten()))),
                }
            except Exception:
                info[name] = {"n_samples": 0, "n_features": 0, "n_clusters": 0}
        else:
            info[name] = {"n_samples": 0, "n_features": 0, "n_clusters": 0}
    return info


def _run_one(
    name: str,
    variant: str,
    cfg: dict,
    seed: int,
    gpu_id: int,
    timeout: int,
    subsample_size: int,
) -> Optional[dict]:
    """Run one TopoGate (variant) × dataset × config × seed combination."""
    from CLUBench import TopoGate
    from CLUBench.tools import clustering_evaluation

    npz = DATA_DIR / f"{name}.npz"
    if not npz.exists():
        return None

    data = np.load(npz)
    X = data["x"]
    Y = data["y"]
    X = X.astype("float32")
    Y = Y.astype("int64")
    K = len(set(Y))

    n_samples, n_features = X.shape

    # Build TopoGate kwargs — pass all cfg keys as extra kwargs
    kwargs = dict(
        n_clusters=K,
        epochs=cfg.get("epochs", 80),
        batch_size=256,
        lr=1e-3,
        hidden_size=128,
        variant_name=variant,
        gpu=gpu_id,
        device="cuda",
        seed=seed,
        neighbor_k=cfg.get("neighbor_k", 5),
        mask_ratio=cfg.get("mask_ratio", 0.3),
        warmup_epochs=cfg.get("warmup_epochs", 20),
        ramp_epochs=cfg.get("ramp_epochs", 10),
        gate_max=cfg.get("gate_max", 0.15),
        subsample_size=subsample_size,
    )

    model = TopoGate(**kwargs)

    t0 = time.time()
    try:
        labels = model.fit_predict(X)
    except Exception as exc:
        raise RuntimeError(f"fit_predict failed: {exc}") from exc
    elapsed = time.time() - t0

    metrics = clustering_evaluation(Y, labels)

    return {
        "dataset": name,
        "variant": variant,
        "seed": seed,
        "gpu": gpu_id,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": K,
        "subsample_size": subsample_size,
        "runtime_seconds": float(elapsed),
        "acc": float(metrics["acc"]),
        "nmi": float(metrics["nmi"]),
        "ari": float(metrics["ari"]),
        **{k: v for k, v in cfg.items()},
    }


def _cfg_tag(cfg: dict) -> str:
    """Short string tag for a config dict."""
    parts = []
    for k in ("epochs", "mask_ratio", "neighbor_k", "gate_max"):
        if k in cfg:
            if k == "mask_ratio":
                parts.append(f"mr{cfg[k]}")
            elif k == "neighbor_k":
                parts.append(f"k{cfg[k]}")
            elif k == "gate_max":
                parts.append(f"gmax{cfg[k]}")
            else:
                parts.append(f"ep{cfg[k]}")
    return "_".join(parts)


def build_stage1_jobs() -> List[Tuple[str, dict, int]]:
    jobs = []
    for ds in DATASETS:
        for cfg in STAGE1_GRID:
            jobs.append((ds, cfg, 42))  # seed=42
    return jobs


def build_stage2_jobs() -> List[Tuple[str, dict, int]]:
    """Stage 2: need to know which datasets are difficult (ARI < 0.5 in Stage 1)."""
    stage1_dir = RESULT_ROOT / "stage1"
    ari_by_ds = {}
    for ds in DATASETS:
        best_ari = -1.0
        for cfg in STAGE1_GRID:
            tag = _cfg_tag(cfg)
            json_path = stage1_dir / f"{ds}__{tag}__seed42.json"
            if json_path.exists():
                try:
                    r = json.load(open(json_path))
                    ari = r.get("ari", -1.0)
                    if ari > best_ari:
                        best_ari = ari
                except Exception:
                    pass
        ari_by_ds[ds] = best_ari

    difficult = [ds for ds, ari in ari_by_ds.items() if ari < 0.5]

    # Also include scRNA datasets (always worth fine-tuning)
    scrna = {"Campbell", "hrvatin_filtered", "Mouse_retina",
             "Quake_Smart-seq2_Lung", "Baron Human"}
    difficult = sorted(set(difficult) | (scrna & set(DATASETS)))

    jobs = []
    for ds in difficult:
        for cfg in STAGE2_GRID:
            for seed in STAGE2_SEEDS:
                jobs.append((ds, cfg, seed))
    return jobs


def build_stage3_jobs() -> List[Tuple[str, dict, int]]:
    """Stage 3: best LearnableGate config per dataset + StaticGate baseline."""
    stage1_dir = RESULT_ROOT / "stage1"
    stage2_dir = RESULT_ROOT / "stage2"

    best_cfg_by_ds: Dict[str, dict] = {}

    # Start from Stage 1 best
    for ds in DATASETS:
        best_ari, best_cfg = -1.0, None
        for cfg in STAGE1_GRID:
            tag = _cfg_tag(cfg)
            p = stage1_dir / f"{ds}__{tag}__seed42.json"
            if p.exists():
                try:
                    r = json.load(open(p))
                    ari = r.get("ari", -1.0)
                    if ari > best_ari:
                        best_ari, best_cfg = ari, cfg.copy()
                except Exception:
                    pass
        if best_cfg is not None:
            best_cfg_by_ds[ds] = best_cfg

    # Upgrade with Stage 2 results if available
    for ds in DATASETS:
        if ds not in best_cfg_by_ds:
            continue
        best_ari, best_cfg = -1.0, best_cfg_by_ds[ds]
        for cfg in STAGE2_GRID:
            for seed in STAGE2_SEEDS:
                tag = _cfg_tag(cfg)
                p = stage2_dir / f"{ds}__{tag}__seed{seed}.json"
                if p.exists():
                    try:
                        r = json.load(open(p))
                        ari = r.get("ari", -1.0)
                        if ari > best_ari:
                            best_ari, best_cfg = ari, cfg.copy()
                    except Exception:
                        pass
        best_cfg_by_ds[ds] = best_cfg

    jobs = []
    for ds in DATASETS:
        # LearnableGate with best config
        if ds in best_cfg_by_ds:
            cfg = best_cfg_by_ds[ds]
            for seed in STAGE3_SEEDS:
                jobs.append((ds, dict(cfg, variant="learnable_gate_sched"), seed))
        # StaticGate baseline
        for seed in STAGE3_SEEDS:
            jobs.append((ds, dict(variant="static_gate_full"), seed))

    return jobs


# ── Main worker loop ───────────────────────────────────────────────────────────

def run_worker(
    stage: int,
    gpu_id: int,
    worker_id: int,
    num_workers: int,
    force: bool = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if stage == 1:
        all_jobs = build_stage1_jobs()
    elif stage == 2:
        all_jobs = build_stage2_jobs()
    elif stage == 3:
        all_jobs = build_stage3_jobs()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Round-robin sharding by worker_id
    my_jobs = [j for i, j in enumerate(all_jobs) if i % num_workers == worker_id]

    stage_dir = RESULT_ROOT / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    total = len(my_jobs)
    n_ok, n_skip, n_fail = 0, 0, 0
    rows = []
    for i, job in enumerate(my_jobs, 1):
        if stage in (1, 2):
            ds, cfg, seed = job
            variant = "learnable_gate_sched"
        else:
            ds, cfg, seed = job
            variant = cfg.get("variant", "learnable_gate_sched")
            cfg = {k: v for k, v in cfg.items() if k != "variant"}

        tag = _cfg_tag(cfg)
        out_path = stage_dir / f"{ds}__{tag}__seed{seed}.json"

        if out_path.exists() and not force:
            n_skip += 1
            try:
                rows.append(json.load(open(out_path)))
            except Exception:
                pass
            continue

        subsample = LARGE_DATASET_SUBSAMPLE.get(ds, 0)
        npz_path = DATA_DIR / f"{ds}.npz"
        if npz_path.exists():
            try:
                data = dict(**np.load(npz_path))
                n_samples = int(data["x"].shape[0])
            except Exception:
                n_samples = 0
        else:
            n_samples = 0

        timeout = get_timeout(n_samples)

        print(f"[{i:4d}/{total}] {ds} {tag} seed={seed} "
              f"{'(sub=%d)' % subsample if subsample else ''} …",
              end=" ", flush=True)

        try:
            result = _run_one(ds, variant, cfg, seed, gpu_id, timeout, subsample)
            if result is None:
                print("SKIP (npz not found)")
                n_skip += 1
                continue
            with open(out_path, "w") as f:
                json.dump(result, f)
            rows.append(result)
            print(f"ACC={result['acc']:.4f} NMI={result['nmi']:.4f} "
                  f"ARI={result['ari']:.4f} ({result['runtime_seconds']:.1f}s)")
            n_ok += 1
        except Exception as exc:
            err_path = str(out_path).replace(".json", ".error.json")
            with open(err_path, "w") as f:
                f.write(f"{ds} {tag} seed={seed}\n{exc}\n{traceback.format_exc()}")
            print(f"FAIL: {exc}")
            n_fail += 1

    # Append rows to merged CSV
    csv_path = RESULT_ROOT / f"stage{stage}.csv"
    fieldnames = [
        "dataset", "variant", "seed", "gpu",
        "n_samples", "n_features", "n_clusters", "subsample_size",
        "epochs", "mask_ratio", "neighbor_k", "gate_max",
        "runtime_seconds", "acc", "nmi", "ari",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[worker {worker_id}] stage={stage} done  "
          f"ok={n_ok}  skip={n_skip}  fail={n_fail}")
    if rows:
        aris = [r["ari"] for r in rows]
        print(f"[worker {worker_id}] mean ARI={sum(aris)/len(aris):.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="LearnableGate 134-dataset sweep")
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--gpu_ids", type=int, nargs="+", default=[4, 5, 7])
    p.add_argument("--worker_id", type=int, required=True)
    p.add_argument("--num_workers", type=int, default=None,
                   help="Total number of parallel workers (default: len(--gpu_ids))")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_workers = args.num_workers or len(args.gpu_ids)
    if args.worker_id < 0 or args.worker_id >= n_workers:
        raise ValueError(f"worker_id {args.worker_id} out of range for {n_workers} workers")
    gpu = args.gpu_ids[args.worker_id % len(args.gpu_ids)]
    run_worker(args.stage, gpu, args.worker_id, n_workers, args.force)
