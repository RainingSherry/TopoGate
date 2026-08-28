#!/usr/bin/env python3
"""
消融实验 v3 — 验证「拓扑门控」的核心贡献

实验设计（5 数据集 × 3 seeds × 4 条件 = 60 tasks on 3 GPUs）：

  v2           learnable + learned gate + reliability mix + k-NN
  A_no_gating  learnable + constant gate + reliability mix + k-NN   ← 有拓扑图，无门控
  B_no_topo    learnable + learned gate + random    mix               ← 有门控，无拓扑信息
  C_neither    learnable + constant gate + none     mix               ← 纯 MAE encoder

5 代表性数据集（最优 hyper-params，来自 stage1 sweep）：
  Baron Human  (scRNA-seq,  K=9)   ep=80, mr=0.4, k=5
  har          (activity,    K=6)   ep=80, mr=0.3, k=5
  enron        (text,        K=2)   ep=80, mr=0.4, k=10
  MNIST_CLIP   (image,      K=10)   ep=80, mr=0.3, k=5
  iris         (tabular,     K=3)   ep=80, mr=0.3, k=10

3 seeds: 42, 123, 7
"""
import csv, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

REPO = Path("/home/luolie/ToPoGate")
PY = sys.executable
TIMEOUT = 600

DATASET_CFG = {
    "Baron Human": {"epochs": 80, "mask_ratio": 0.4, "neighbor_k": 5, "gate_max": 0.15, "n_clusters": 9},
    "har":        {"epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5, "gate_max": 0.15, "n_clusters": 6},
    "enron":      {"epochs": 80, "mask_ratio": 0.4, "neighbor_k": 10, "gate_max": 0.15, "n_clusters": 2},
    "MNIST_CLIP": {"epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5, "gate_max": 0.15, "n_clusters": 10},
    "iris":       {"epochs": 80, "mask_ratio": 0.3, "neighbor_k": 10, "gate_max": 0.15, "n_clusters": 3},
}

GPU_POOL = [1, 4, 5]  # wid → physical GPU id (0 and 7 FORBIDDEN)

OUTPUT_DIR = REPO / "result/ablation_topol_gate"
SEEDS = [42, 123, 7]

CONDITIONS = {
    "v2": {
        "mix_mode": "reliability", "gate_mode": "learned",
        "warmup_epochs": 20, "ramp_epochs": 10,
    },
    "A_no_gating": {
        "mix_mode": "reliability", "gate_mode": "constant",
    },
    "B_no_topo": {
        "mix_mode": "random", "gate_mode": "learned",
        "warmup_epochs": 20, "ramp_epochs": 10,
    },
    "C_neither": {
        "mix_mode": "none", "gate_mode": "none",
    },
}


def _run_one(ds_name, cfg, cond_name, cond, seed, gpu_id, task_dir):
    npz_src = REPO / "datasets" / f"{ds_name}.npz"
    assert npz_src.exists(), f"NPZ not found: {npz_src}"

    npz_local = task_dir / "data.npz"
    npz_local.write_bytes(npz_src.read_bytes())

    args = [
        str(REPO / "methods/TopoGate/learnable_gate/run_npz.py"),
        "--data_path", str(npz_local),
        "--save_dir", str(task_dir),
        "--epochs", str(cfg["epochs"]),
        "--mask_ratio", str(cfg["mask_ratio"]),
        "--neighbor_k", str(cfg["neighbor_k"]),
        "--gate_max", str(cfg["gate_max"]),
        "--n_clusters", str(cfg["n_clusters"]),
        "--seed", str(seed),
        "--gpu", str(gpu_id),
        "--mix_mode", cond["mix_mode"],
        "--gate_mode", cond["gate_mode"],
    ]
    if cond.get("warmup_epochs") is not None:
        args += ["--warmup_epochs", str(cond["warmup_epochs"]),
                 "--ramp_epochs", str(cond["ramp_epochs"])]

    log_file = task_dir / "run.log"
    env = os.environ.copy()
    env["TMPDIR"] = "/data/luolie/ToPoGate/tmp"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with open(log_file, "wb") as lf:
        proc = subprocess.Popen(
            [PY] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env,
        )
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            lf.write(chunk)
        proc.stdout.close()
        rc = proc.wait(timeout=TIMEOUT)

    if rc != 0:
        return False, f"exit {rc}"

    metrics_json = task_dir / "metrics.json"
    summary_json = task_dir / "summary.json"
    if metrics_json.exists():
        try:
            return True, json.loads(metrics_json.read_text())
        except Exception as e:
            pass
    if summary_json.exists():
        try:
            data = json.loads(summary_json.read_text())
            return True, data.get("metrics", data)
        except Exception as e:
            return False, f"json parse: {e}"
    return False, "no metrics.json or summary.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = list(DATASET_CFG.keys())
    all_tasks = [
        (ds, cond, seed)
        for ds in datasets
        for cond in CONDITIONS
        for seed in SEEDS
    ]
    total = len(all_tasks)
    print(f"[ablation] {total} tasks = {len(datasets)} ds × {len(CONDITIONS)} cond × {len(SEEDS)} seeds")
    print(f"  Conditions: {list(CONDITIONS.keys())}")
    print(f"  Seeds: {SEEDS}, GPUs: {GPU_POOL}")
    print()

    results = [None] * total
    errors  = [None] * total

    for i, (ds_name, cond_name, seed) in enumerate(all_tasks):
        cfg  = DATASET_CFG[ds_name]
        cond = CONDITIONS[cond_name]
        wid  = i % 3
        gpu  = GPU_POOL[wid]
        task_dir = OUTPUT_DIR / f"task_{i:03d}_{ds_name}__{cond_name}__s{seed}"
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1:02d}/{total}] {ds_name:20s} | {cond_name:15s} | seed={seed} | GPU={gpu} ... ",
              end="", flush=True)

        ok, ret = _run_one(ds_name, cfg, cond_name, cond, seed, gpu, task_dir)
        if ok:
            results[i] = ret
            print(f"ARI={ret.get('ari', -1):.4f}  NMI={ret.get('nmi', -1):.4f}  ACC={ret.get('acc', -1):.4f}")
        else:
            errors[i] = ret
            print(f"ERROR: {ret}")

    # Write CSV
    csv_path = OUTPUT_DIR / "ablation_results.csv"
    fieldnames = ["idx", "dataset", "condition", "seed", "n_clusters",
                  "epochs", "mask_ratio", "neighbor_k", "gate_max",
                  "mix_mode", "gate_mode", "warmup_epochs", "ramp_epochs",
                  "ari", "nmi", "acc", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (ds_name, cond_name, seed) in enumerate(all_tasks):
            cfg  = DATASET_CFG[ds_name]
            cond = CONDITIONS[cond_name]
            r    = results[i]
            w.writerow({
                "idx": i, "dataset": ds_name, "condition": cond_name, "seed": seed,
                "n_clusters": cfg["n_clusters"], "epochs": cfg["epochs"],
                "mask_ratio": cfg["mask_ratio"], "neighbor_k": cfg["neighbor_k"],
                "gate_max": cfg["gate_max"],
                "mix_mode": cond["mix_mode"], "gate_mode": cond["gate_mode"],
                "warmup_epochs": cond.get("warmup_epochs", ""),
                "ramp_epochs": cond.get("ramp_epochs", ""),
                "ari": r.get("ari", "") if r else "",
                "nmi": r.get("nmi", "") if r else "",
                "acc": r.get("acc", "") if r else "",
                "error": errors[i] or "",
            })

    ok_count = sum(1 for r in results if r is not None)
    print(f"\n[ablation] Done. OK: {ok_count}/{total}  Errors: {total - ok_count}")
    print(f"  CSV: {csv_path}")

    # Print summary
    import statistics
    print("\n" + "=" * 95)
    print("SUMMARY: mean ± std ARI over 3 seeds")
    print("=" * 95)
    header = f"{'Dataset':<22}" + "".join(f"{c:>14}" for c in CONDITIONS)
    print(header)
    print("-" * 95)
    for ds in datasets:
        vals = {}
        for cn in CONDITIONS:
            ari_list = [
                results[i].get("ari", 0)
                for i, (d, c, s) in enumerate(all_tasks)
                if d == ds and c == cn and results[i] is not None
            ]
            if len(ari_list) == 3:
                m = statistics.mean(ari_list)
                st = statistics.stdev(ari_list)
                vals[cn] = f"{m:.4f}±{st:.3f}"
            elif len(ari_list) > 0:
                vals[cn] = f"{statistics.mean(ari_list):.4f}(n={len(ari_list)})"
            else:
                vals[cn] = "FAIL"
        row = f"{ds:<22}" + "".join(f"{vals.get(c, 'N/A'):>14}" for c in CONDITIONS)
        print(row)

    # Print delta vs v2
    print("\n" + "=" * 95)
    print("DELTA vs v2 (拓扑门控的贡献)")
    print("=" * 95)
    alt = ["A_no_gating", "B_no_topo", "C_neither"]
    print(f"{'Dataset':<22}" + "".join(f"{'Δ'+c:>14}" for c in alt))
    print("-" * 95)
    for ds in datasets:
        v2_vals = [
            results[i].get("ari", 0)
            for i, (d, c, s) in enumerate(all_tasks)
            if d == ds and c == "v2" and results[i] is not None
        ]
        if not v2_vals:
            print(f"{ds:<22}" + "".join("v2 FAIL".center(14) for _ in alt))
            continue
        v2_mean = statistics.mean(v2_vals)
        deltas = {}
        for cn in alt:
            cn_vals = [
                results[i].get("ari", 0)
                for i, (d, c, s) in enumerate(all_tasks)
                if d == ds and c == cn and results[i] is not None
            ]
            deltas[cn] = statistics.mean(cn_vals) - v2_mean if cn_vals else None
        row = f"{ds:<22}" + "".join(
            f"{deltas[c]:+.4f}".center(14) if deltas[c] is not None else "FAIL".center(14)
            for c in alt
        )
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
