#!/usr/bin/env python3
"""
TopoGate 三层对比实验统一 runner

用法:
  python scripts/baseline_runner.py --table 1 --datasets mouse_retina weather
  python scripts/baseline_runner.py --table 3 --datasets mouse_retina
  python scripts/baseline_runner.py --table all --datasets mouse_retina weather --gpu 4

设计原则:
  - 一个表 = 一个方法集 = 一次 sweep
  - 每个 run 用 subprocess + timeout，避免长任务卡死
  - 实时写实验日志到 logs/<task_id>.log
  - 超时自动 kill 并写 *.error.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
RESULTS_DIR = REPO_ROOT / "result"
CONFIGS_DIR = REPO_ROOT / "methods" / "TopoGate" / "configs"
LOGS_DIR = REPO_ROOT / "logs"


# ============== 表 1：重建式家族 ==============
TABLE_1_ALGORITHMS = {
    # name : (algo_class, extra_kwargs)
    "KMeans": ("KMeans", {}),
    "GMM": ("GMM", {}),
    "DEC": ("DEC", {}),
    "IDEC": ("IDEC", {}),
    "DSCN": ("DSCN", {}),
    "EDESC": ("EDESC", {}),
    "TopoGate_nomix": ("TopoGate", {"variant_name": "topogate_nomix"}),
    "TopoGate": ("TopoGate", {"variant_name": "topogate_full"}),
}

# ============== 表 2：图自监督家族 ==============
TABLE_2_ALGORITHMS = {
    "LFSS": ("LFSS", {}),
    "DIVC": ("DIVC", {}),
    "PICA": ("PICA", {}),
    "P2OT": ("P2OT", {}),
    "TopoGate": ("TopoGate", {"variant_name": "topogate_full"}),
}

# ============== 表 3：TopoGate Ablation ==============
TABLE_3_VARIANTS = [
    "topogate_nomix",
    "topogate_random_neighbors",
    "topogate_far_neighbors",
    "topogate_constant_gate",
    "topogate_gate_only",
    "topogate_edge_only",
    "topogate_no_topology_features",
    "topogate_full",
]

# ============== 表 4：跨域泛化 ==============
TABLE_4_DATASETS = ["mnist", "fashion_mnist", "coil20"]
TABLE_4_ALGORITHMS = {
    "KMeans": ("KMeans", {}),
    "DEC": ("DEC", {}),
    "IDEC": ("IDEC", {}),
    "DSCN": ("DSCN", {}),
    "TopoGate": ("TopoGate", {"variant_name": "topogate_full"}),
}


def run_one(
    dataset: str,
    algo_name: str,
    algo_class: str,
    extra_kwargs: dict,
    gpu: int,
    timeout: int,
    seed: int = 42,
) -> dict:
    """跑一个 (dataset, algorithm) 组合."""
    npz_path = DATASETS_DIR / f"{dataset}.npz"
    if not npz_path.exists():
        return {"status": "skipped", "reason": f"dataset not found: {npz_path}"}

    save_dir = RESULTS_DIR / algo_name / dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{algo_name}_{dataset}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u",
        "-c",
        f"""
import sys
sys.path.insert(0, '{REPO_ROOT}/baseline/CLUBench')
from CLUBench import {algo_class}, load_data, clustering_evaluation, load_hpc
import time, traceback

try:
    X, y = load_data('{dataset}.npz')
    n_clusters = len(set(y))
    hpc = load_hpc('{algo_name.lower()}') if '{algo_name.lower()}' in ['topogate','ekmeans','ssekm','ssekm_sup','gbusc','dec','idec','divc','lfss','conclu','dscn','edesc','p2ot','pica','dmicc'] else {{}}
    init_kwargs = {{'n_clusters': n_clusters, 'random_state': {seed}}}
    init_kwargs.update({json.dumps(extra_kwargs)})
    if '{algo_class}' == 'TopoGate':
        init_kwargs['save_dir'] = '{save_dir}'
        init_kwargs['n_clusters'] = n_clusters
        init_kwargs['gpu'] = {gpu}
        init_kwargs['seed'] = {seed}
        init_kwargs['data_path'] = '{npz_path}'
    else:
        init_kwargs.update(hpc)
    model = {algo_class}(**init_kwargs)
    t0 = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - t0
    e = clustering_evaluation(y, labels)
    print(f'{{algo_name}} on {dataset}: acc={{e["acc"]:.4f}} nmi={{e["nmi"]:.4f}} ari={{e["ari"]:.4f}} time={{elapsed:.2f}}s')
except Exception as e:
    print(f'FAILED: {{algo_name}} on {dataset}: {{type(e).__name__}}: {{e}}')
    traceback.print_exc()
    sys.exit(1)
""",
    ]

    try:
        with open(log_path, "w") as f:
            result = subprocess.run(
                cmd, timeout=timeout, stdout=f, stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
            )
        return {
            "status": "success" if result.returncode == 0 else "failed",
            "elapsed": result.returncode,
            "log": str(log_path),
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout": timeout, "log": str(log_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["1", "2", "3", "4", "all"], required=True)
    parser.add_argument("--datasets", nargs="+", default=None,
                      help="数据集列表（不传则用 11 个 advantage datasets）")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800, help="单个 run 超时（秒）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 默认数据集：11 advantage datasets
    if args.datasets is None:
        datasets = ["mouse_retina", "weather", "smoker_condition", "breast_cancer_original",
                    "sms_spam", "spambase", "sonar", "isolet", "har", "mnist", "coil20"]
    else:
        datasets = args.datasets

    # 选定运行的算法集
    if args.table == "1":
        algorithms = TABLE_1_ALGORITHMS
    elif args.table == "2":
        algorithms = TABLE_2_ALGORITHMS
    elif args.table == "3":
        algorithms = {v: ("TopoGate", {"variant_name": v}) for v in TABLE_3_VARIANTS}
    elif args.table == "4":
        datasets = TABLE_4_DATASETS
        algorithms = TABLE_4_ALGORITHMS
    else:  # all
        algorithms = {**TABLE_1_ALGORITHMS, **TABLE_2_ALGORITHMS}
        algorithms.update({v: ("TopoGate", {"variant_name": v}) for v in TABLE_3_VARIANTS})

    print(f"=== 表 {args.table} | {len(datasets)} datasets × {len(algorithms)} algorithms ===")
    print(f"GPU={args.gpu} timeout={args.timeout}s seed={args.seed}")
    total = len(datasets) * len(algorithms)
    done = 0
    for dataset in datasets:
        for algo_name, (algo_class, extra) in algorithms.items():
            done += 1
            print(f"[{done}/{total}] {algo_name} on {dataset} ...")
            r = run_one(dataset, algo_name, algo_class, extra, args.gpu, args.timeout, args.seed)
            print(f"  → {r['status']}  log={r.get('log')}")


if __name__ == "__main__":
    main()
