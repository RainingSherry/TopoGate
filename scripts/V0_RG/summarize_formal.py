#!/usr/bin/env python3
"""Build reproducible CSV summaries from completed V0_RG final artifacts."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

SEEDS = (42, 123, 7, 2025, 3407)

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows or [{"status":"no_records"}])

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def collect(path: Path, panel: str):
    selected = json.loads((path / "selected.json").read_text())
    summary = json.loads((path / "final_summary.json").read_text())
    rows, failures = [], []
    for record_path in sorted((path / "final").glob("*/seed_*.json")):
        record = json.loads(record_path.read_text())
        row = {"panel": panel, "dataset_id": path.name, "seed": int(record["seed"]),
               "status": record["status"], "wall_seconds": record.get("wall_seconds")}
        row.update(record.get("test_metrics", {})); rows.append(row)
        if record["status"] != "completed": failures.append({"panel":panel,"dataset_id":path.name,"seed":record["seed"],"error":record.get("error","unknown")})
    observed = sorted(r["seed"] for r in rows if r["status"] == "completed")
    if observed != sorted(SEEDS): failures.append({"panel":panel,"dataset_id":path.name,"seed":"","error":f"final seeds mismatch: {observed}"})
    flat = {"panel":panel,"dataset_id":path.name,"selection_hash":summary["selection_hash"],"n_seeds":summary["n_seeds"],"config_hash":selected["winner"]["config_hash"],"validation_ari_mean":selected["winner"]["validation_ari_mean"],"validation_ari_std":selected["winner"]["validation_ari_std"]}
    for metric, values in summary["metrics"].items(): flat[f"{metric}_mean"], flat[f"{metric}_std"] = values["mean"], values["std"]
    return rows, flat, failures

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--clubench-root", type=Path, required=True); p.add_argument("--biology-root", type=Path, required=True); p.add_argument("--biology-retry-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    datasets = [(x,"clubench") for x in sorted(a.clubench_root.iterdir()) if x.is_dir() and (x/"final_summary.json").exists()]
    for root in (a.biology_root, a.biology_retry_root): datasets += [(x,"biology") for x in sorted(root.iterdir()) if x.is_dir() and (x/"final_summary.json").exists()]
    seed_rows, summaries, failures = [], [], []
    for path, panel in datasets:
        rows, summary, errors = collect(path, panel); seed_rows += rows; summaries.append(summary); failures += errors
    write_csv(a.output/"final_metrics_per_seed.csv", seed_rows); write_csv(a.output/"final_summary_per_dataset.csv", summaries); write_csv(a.output/"failures.csv", failures)
    costs = [{"panel":p,"dataset_id":d,"wall_seconds":sum(float(r.get("wall_seconds") or 0) for r in seed_rows if r["panel"]==p and r["dataset_id"]==d)} for p,d in sorted({(r["panel"],r["dataset_id"]) for r in seed_rows})]
    write_csv(a.output/"compute_costs.csv", costs)
    artifacts = [{"panel":panel,"dataset_id":path.name,"path":str(f),"bytes":f.stat().st_size,"sha256":sha256(f)} for path,panel in datasets for f in sorted(path.rglob("*")) if f.is_file()]
    write_csv(a.output/"artifact_manifest.csv", artifacts)
    status = {"clubench_completed":sum(s["panel"]=="clubench" for s in summaries),"biology_completed":sum(s["panel"]=="biology" for s in summaries),"failure_count":len(failures)}
    (a.output/"completion_audit.json").write_text(json.dumps(status, indent=2)); print(json.dumps(status))

if __name__ == "__main__": main()
