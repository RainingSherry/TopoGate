#!/usr/bin/env python3
"""Run the unified backbone's graph-disabled scMAE control."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from methods.TopoGate.V0_RG.config import V0_RGConfig
from methods.TopoGate.V0_RG.input_adapter import load_matrix
from methods.TopoGate.V0_RG.tuning import SplitPreprocessor, split_rows
from methods.TopoGate.V0_RG.trainer import fit_predict

SEEDS = (42, 123, 7, 2025, 3407)

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--dataset-id",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda"); a=p.parse_args()
    row=next(r for r in json.loads(a.manifest.read_text())["datasets"] if r["dataset_id"]==a.dataset_id)
    loaded=load_matrix(row["path"],row.get("labels_path")); X,y=loaded.X,np.asarray(loaded.labels).reshape(-1); tr,va,te=split_rows(len(y),9102026); fit_rows=np.r_[tr,va]; prep=SplitPreprocessor(row["input_kind"],2000).fit(X[fit_rows]); fit_x,test_x=prep.transform(X[fit_rows]),prep.transform(X[te])
    cfg=V0_RGConfig(protocol_id="topogate_scmae_control_v1",variant="scmae_only",epochs=80,pseudo_weight=0.0)
    a.output.mkdir(parents=True,exist_ok=True); (a.output/"protocol.json").write_text(json.dumps({"dataset_id":a.dataset_id,"control":"scmae_only","seeds":list(SEEDS),"split_seed":9102026,"fit_scope":"train_validation","score_scope":"test","labels_used_during_fit":False},indent=2))
    rows=[]
    for seed in SEEDS:
        started=time.time(); pred,emb,diag=fit_predict(test_x,fit_X=fit_x,n_clusters=int(row["n_clusters"]),config=cfg,seed=seed,device=a.device); rows.append({"dataset_id":a.dataset_id,"method":"scmae_only","seed":seed,"ari":float(adjusted_rand_score(y[te],pred)),"nmi":float(normalized_mutual_info_score(y[te],pred)),"wall_seconds":time.time()-started}); np.save(a.output/f"predictions_seed_{seed}.npy",pred)
    with (a.output/"metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=sorted(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({"dataset_id":a.dataset_id,"rows":len(rows),"status":"completed"}))
if __name__=="__main__": main()
