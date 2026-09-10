#!/usr/bin/env python3
"""Run preregistered same-input PCA/KMeans controls on a fixed representative panel."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from methods.TopoGate.V0_RG.tuning import SplitPreprocessor, split_rows
from methods.TopoGate.V0_RG.input_adapter import load_matrix

DATASETS = (
    "banknote_authentication", "20newsgroups", "cifar10", "Campbell",
    "Blood_BoneMarrow", "Human_Pancreas_1", "Mouse_Pancreas_1", "PRJNA895163",
)
SEEDS = (42, 123, 7, 2025, 3407)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--clubench-manifest",type=Path,required=True); p.add_argument("--biology-manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    manifests=[]
    for path in (a.clubench_manifest,a.biology_manifest): manifests += json.loads(path.read_text())["datasets"]
    lookup={r["dataset_id"]:r for r in manifests}; rows=[]; protocol={"selection":"fixed type/scale strata before score inspection","datasets":list(DATASETS),"seeds":list(SEEDS),"split_seed":9102026,"fit_scope":"train_validation","score_scope":"test","labels_used_for_fit":False}
    (a.output/"controls_protocol.json").parent.mkdir(parents=True,exist_ok=True); (a.output/"controls_protocol.json").write_text(json.dumps(protocol,indent=2))
    for dataset_id in DATASETS:
        row=lookup[dataset_id]; loaded=load_matrix(row["path"],row.get("labels_path")); X,y=loaded.X,np.asarray(loaded.labels).reshape(-1); train,val,test=split_rows(len(y),9102026); fit=np.concatenate([train,val]); prep=SplitPreprocessor(row["input_kind"],2000).fit(X[fit]); fit_x,test_x=prep.transform(X[fit]),prep.transform(X[test]); k=int(row["n_clusters"])
        for method in ("kmeans","pca_kmeans"):
            for seed in SEEDS:
                started=time.time(); fit_space=fit_x; score_space=test_x; pca_dim=None
                if method=="pca_kmeans":
                    pca_dim=min(50,fit_x.shape[0]-1,fit_x.shape[1]); reducer=PCA(n_components=pca_dim,svd_solver="randomized",random_state=seed); fit_space=reducer.fit_transform(fit_x); score_space=reducer.transform(test_x)
                model=KMeans(n_clusters=k,n_init=20,random_state=seed); model.fit(fit_space); pred=model.predict(score_space); truth=y[test]
                rows.append({"dataset_id":dataset_id,"method":method,"seed":seed,"ari":float(adjusted_rand_score(truth,pred)),"nmi":float(normalized_mutual_info_score(truth,pred)),"wall_seconds":time.time()-started,"pca_dim":pca_dim or ""})
        print(dataset_id,flush=True)
    fields=sorted({k for r in rows for k in r});
    with (a.output/"controls_metrics_per_seed.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps({"datasets":len(DATASETS),"rows":len(rows)}))
if __name__=="__main__": main()
