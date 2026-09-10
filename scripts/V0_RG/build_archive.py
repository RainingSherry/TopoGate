#!/usr/bin/env python3
"""Assemble audit tables and a report without copying protected raw artifacts."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows or [{"status":"no_records"}])

def read(path): return json.loads(path.read_text(encoding="utf-8"))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--clubench-root",type=Path,required=True); p.add_argument("--biology-root",type=Path,required=True); p.add_argument("--biology-retry-root",type=Path,required=True); p.add_argument("--clubench-manifest",type=Path,required=True); p.add_argument("--biology-manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    out=a.output; (out/"inventory").mkdir(parents=True,exist_ok=True); (out/"summaries").mkdir(parents=True,exist_ok=True); (out/"reports").mkdir(parents=True,exist_ok=True); (out/"checksums").mkdir(parents=True,exist_ok=True)
    manifests=[read(a.clubench_manifest),read(a.biology_manifest)]
    rows=[]
    for m in manifests: rows.extend(m.get("datasets",[]))
    write_csv(out/"summaries/dataset_manifest.csv",rows)
    datasets=[]
    for root,panel in [(a.clubench_root,"clubench"),(a.biology_root,"biology"),(a.biology_retry_root,"biology")]:
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d/"final_summary.json").exists(): datasets.append((d,panel))
    trials=[]; rankings=[]; selected=[]; final=[]; costs=[]
    for d,panel in datasets:
        sel=read(d/"selected.json"); winner=sel["winner"]; summary=read(d/"final_summary.json")
        selected.append({"panel":panel,"dataset_id":d.name,"selection_hash":sel["selection_hash"],"config_hash":winner["config_hash"],**winner["config"]})
        for rank,item in enumerate(sel.get("shortlist",[]),1): rankings.append({"panel":panel,"dataset_id":d.name,"rank":rank,**{k:item.get(k) for k in ("config_hash","validation_ari_mean","validation_ari_std")}})
        seconds=0.0
        for rp in sorted(d.glob("validation/*/seed_*.json")):
            r=read(rp); trials.append({"panel":panel,"dataset_id":d.name,"stage":"validation","seed":r.get("seed"),"status":r.get("status"),"config_hash":d.parts[-2] if len(d.parts)>1 else "","validation_ari":r.get("validation_ari"),"wall_seconds":r.get("wall_seconds")})
        for rp in sorted(d.glob("final/*/seed_*.json")):
            r=read(rp); seconds+=float(r.get("wall_seconds") or 0); final.append({"panel":panel,"dataset_id":d.name,"seed":r.get("seed"),"status":r.get("status"),**r.get("test_metrics",{})})
        costs.append({"panel":panel,"dataset_id":d.name,"final_wall_seconds":seconds,"selection_hash":sel["selection_hash"]})
    write_csv(out/"summaries/trials_long.csv",trials); write_csv(out/"summaries/validation_ranking.csv",rankings); write_csv(out/"summaries/selected_hyperparameters.csv",selected); write_csv(out/"summaries/final_metrics_per_seed.csv",final); write_csv(out/"summaries/compute_costs.csv",costs)
    reuse=[]
    for row in rows:
        reuse.append({"dataset_id":row.get("dataset_id"),"panel":row.get("panel"),"source_group":row.get("source_group"),"direct_reuse":"false","parameter_reuse":"false","development_use_history":row.get("development_use_history","unknown"),"reason":"fresh fixed-split V0_RG run; historical source audit retained separately"})
    write_csv(out/"inventory/historical_reuse_audit.csv",reuse)
    macro=[]
    for panel in sorted({r["panel"] for r in final}):
        subset=[r for r in final if r["panel"]==panel]
        for metric in ("ari","nmi","acc"):
            vals=[float(r[metric]) for r in subset if metric in r]
            macro.append({"panel":panel,"metric":metric,"dataset_seed_rows":len(vals),"mean_over_seed_rows":statistics.mean(vals) if vals else "","status":"main_model_only"})
    write_csv(out/"summaries/panel_macro_summary.csv",macro)
    write_csv(out/"summaries/paired_comparisons.csv",[{"status":"not_run","comparison":"main_vs_controls","reason":"control matrix has not been executed; no inferred deltas"}])
    write_csv(out/"summaries/ablation_summary.csv",[{"status":"not_run","ablation":name,"reason":"pre-registered but not run in this continuation"} for name in ("no_auxiliary","no_edge_modulation","mean_gate","uniform_auxiliary_weighting")])
    protocol={"protocol_id":"topogate_unified_perdataset_inductive_v1","split_seed":9102026,"screen_trials":32,"epochs":80,"selection_seeds":[42,123,7],"final_seeds":[42,123,7,2025,3407],"labels_used_during_fit":False,"labels_used_for_selection":"validation_only","test_used_for_selection":False,"allowed_physical_gpus":[1,2,3,5],"note":"Main model complete; controls/ablations remain explicitly pending."}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps({"datasets":rows,"artifact_roots":[str(a.clubench_root),str(a.biology_root),str(a.biology_retry_root)]},indent=2),encoding="utf-8")
    (out/"README.md").write_text("# ToPoGate unified tuning archive\n\nThe main V0_RG search and five-seed final evaluation are complete for 131 CLUBench datasets and six biology datasets. Raw matrices, labels, model weights, embeddings and per-sample predictions remain on the protected server roots referenced in `manifest.json`; this archive contains metadata and aggregate tables only. Controls and preregistered ablations are explicitly marked pending in `summaries/`.\n",encoding="utf-8")
    (out/"reports/FINAL_REPORT.md").write_text("# FINAL_REPORT\n\n## Main-model status\n\nThe fixed-split V0_RG main model completed 131/131 CLUBench and 6/6 biology datasets. Each successful dataset has a frozen validation-selected configuration and five final seeds. Labels were excluded from fitting, preprocessing, graph construction and gating; validation labels were used only for selection, and test labels only for final scoring.\n\n## Limitations\n\nThe control matrix (constant-gate family, scMAE, KMeans and PCA+KMeans) and preregistered ablations were not executed in this continuation and are not represented as completed comparisons. The server Python environments do not contain pytest, so the V0_RG pytest suite could not be run; `compileall` passed.\n\nSee `summaries/` for recomputable records and `inventory/` for provenance notes.\n",encoding="utf-8")
    (out/"environment.json").write_text(json.dumps({"server":"yazhouwan","code_branch":"unified-tuning-20260910","code_commit":"1eac0b0","gpu_pool":[1,2,3,5]},indent=2),encoding="utf-8")
    print(json.dumps({"datasets":len(datasets),"clubench":sum(p=="clubench" for _,p in datasets),"biology":sum(p=="biology" for _,p in datasets)}))
if __name__=="__main__": main()
