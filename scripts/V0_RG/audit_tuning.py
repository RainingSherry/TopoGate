#!/usr/bin/env python3
"""Audit the completed per-dataset tuning protocol and emit tuning summaries."""
from __future__ import annotations
import argparse, csv, json, hashlib, sqlite3
from collections import Counter, defaultdict
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) or ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def datasets(root, panel, subdir):
    base=Path(root)/subdir
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d/"experiment.json").exists() and (d/"selected.json").exists() and (d/"final_summary.json").exists(): yield d, panel
def sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--clubench-root",required=True); ap.add_argument("--biology-root",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    out=Path(a.output); ds=list(datasets(a.clubench_root,"clubench","datasets"))+list(datasets(a.biology_root,"biology","formal"))+list(datasets(a.biology_root,"biology","formal_retry2"))
    audit=[]; freq=Counter(); regimes=Counter(); gaps=[]; budgets=[]; trials=[]
    for d,panel in ds:
        e=load(d/"experiment.json"); s=load(d/"selected.json"); fs=load(d/"final_summary.json")
        vfiles=list(d.glob("validation/*/seed_*.json")); ffiles=list(d.glob("final/*/seed_*.json"))
        seed42=[p for p in vfiles if load(p).get("seed")==42]; seed_extra=[p for p in vfiles if load(p).get("seed") in (123,7)]
        shortlist=s.get("shortlist",[]); win=s.get("winner",{}); cfg=win.get("config",{})
        anchors=set()
        for p in seed42:
            r=load(p); c=r.get("config",{}); mode=c.get("edge_reliability_mode"); adaptive=(c.get("gate_adaptivity") not in (None,0,0.0))
            if (mode in (None,"none")) and not adaptive: anchors.add("base_constant")
            if (mode not in (None,"none")) and adaptive: anchors.add("topology_adaptive")
            if (mode not in (None,"none")) and not adaptive: anchors.add("topology_constant")
            if (mode in (None,"none")) and adaptive: anchors.add("base_adaptive")
        status=[]
        status += ["trials32" if len(seed42)==32 else f"trials42={len(seed42)}"]
        status += ["shortlist4" if len(shortlist)==4 else f"shortlist={len(shortlist)}"]
        status += ["extra_seeds" if {load(p).get("seed") for p in seed_extra}=={123,7} else "missing_extra_seeds"]
        status += ["final5" if len(ffiles)==5 and fs.get("n_seeds")==5 else f"final={len(ffiles)}"]
        status += ["test_excluded" if s.get("test_used_for_selection") is False else "test_selection_flag"]
        status += ["anchors4" if len(anchors)==4 else "anchors="+str(sorted(anchors))]
        ok=all(x in status for x in ("trials32","shortlist4","extra_seeds","final5","test_excluded","anchors4"))
        audit.append({"panel":panel,"dataset_id":d.name,"ok":ok,"n_screen_seed42":len(seed42),"n_validation_records":len(vfiles),"n_shortlist":len(shortlist),"n_final":len(ffiles),"anchors":",".join(sorted(anchors)),"selection_hash":s.get("selection_hash"),"identity":s.get("identity"),"split_hash":e.get("split_hash"),"code_hash":e.get("code_hash"),"issues":";".join(x for x in status if not (x in ("trials32","shortlist4","extra_seeds","final5","test_excluded","anchors4")))})
        for k in ("variant","neighbor_k","mix_neighbors","gate_adaptivity","edge_reliability_mode","gamma_mutual","gamma_snn","beta_mutual","beta_snn","beta_perturb"):
            freq[(k,str(cfg.get(k)))] += 1
        adaptive=cfg.get("gate_adaptivity") not in (None,0,0.0); edge=cfg.get("edge_reliability_mode") not in (None,"none")
        regimes[("adaptive" if adaptive else "constant", "edge_on" if edge else "edge_off")] += 1
        fm=fs.get("metrics",{}); vmean=win.get("validation_ari_mean"); testmean=fm.get("ari",{}).get("mean")
        if vmean is not None and testmean is not None: gaps.append({"panel":panel,"dataset_id":d.name,"validation_ari_mean":vmean,"test_ari_mean":testmean,"test_minus_validation":testmean-vmean,"test_ari_std":fm.get("ari",{}).get("std")})
        for p in vfiles:
            r=load(p); trials.append({"panel":panel,"dataset_id":d.name,"stage":"validation","seed":r.get("seed"),"status":r.get("status"),"config_hash":sha(r.get("config",{})),"config_dir":p.parent.name,"validation_ari":r.get("validation_ari"),"wall_seconds":r.get("wall_seconds")})
        for p in ffiles:
            r=load(p); trials.append({"panel":panel,"dataset_id":d.name,"stage":"final","seed":r.get("seed"),"status":r.get("status"),"config_hash":sha(r.get("config",{})),"config_dir":p.parent.name,"ari":r.get("test_metrics",{}).get("ari"),"nmi":r.get("test_metrics",{}).get("nmi"),"acc":r.get("test_metrics",{}).get("acc"),"wall_seconds":r.get("wall_seconds")})
        try:
            con=sqlite3.connect(d/"search.db"); ndb=con.execute("select count(*) from trials where state='COMPLETE'").fetchone()[0]; con.close()
        except Exception: ndb=""
        budgets.append({"panel":panel,"dataset_id":d.name,"screen_records":len(seed42),"top4_seed_records":len(seed_extra),"final_records":len(ffiles),"total_records":len(seed42)+len(seed_extra)+len(ffiles),"search_db_complete_trials":ndb,"anchor_records":max(0,len(seed42)-int(ndb or 0)),"budget_ok":len(seed42)==32 and int(ndb or 0)+max(0,len(seed42)-int(ndb or 0))==32,"budget_expected":"28 Optuna trials + 4 independent anchor records = 32 screen candidates; 40 validation records + 5 final records"})
    write(out/"tuning_completion_audit.csv",audit); write(out/"validation_test_gap.csv",gaps); write(out/"search_budget_accounting.csv",budgets); write(out/"trials_long_corrected.csv",trials)
    write(out/"hyperparameter_selection_frequency.csv",[{"parameter":k,"value":v,"count":n,"fraction":n/len(ds)} for (k,v),n in sorted(freq.items())]); write(out/"gate_regime_summary.csv",[{"gate_regime":k[0],"edge_regime":k[1],"count":n,"fraction":n/len(ds)} for k,n in sorted(regimes.items())])
    print(json.dumps({"datasets":len(ds),"audit_ok":sum(x["ok"] for x in audit),"audit_failed":sum(not x["ok"] for x in audit),"validation_test_gaps":len(gaps)}))
if __name__=="__main__": main()
