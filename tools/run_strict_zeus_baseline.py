"""ZEUS protocol gate.

This deliberately refuses to produce formal metrics until the native wrapper
exposes an embedding-only API.  Calling its current ``fit_predict`` on the
whole matrix would leak test samples through PCA/scaling and the KMeans
readout, so that path is rejected rather than silently mislabelled.
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

SEEDS=(42,123,7)

def acc(y,p):
    a,b=np.unique(y),np.unique(p); m=np.zeros((len(a),len(b)),dtype=int)
    for i,x in enumerate(a):
        for j,z in enumerate(b): m[i,j]=np.count_nonzero(p[y==x]==z)
    r,c=linear_sum_assignment(m.max()-m); return float(m[r,c].sum()/len(y))
def met(y,p): return {'ari':float(adjusted_rand_score(y,p)),'nmi':float(normalized_mutual_info_score(y,p)),'acc':acc(y,p)}
def atomic(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(v,indent=2)); os.replace(q,p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--gpu',type=int,default=1); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
    out=a.output_dir; final=out/'zeus_final.json'
    if final.exists() and not a.force:return
    exp=json.loads((a.dataset_dir/'experiment.json').read_text()); split=json.loads((a.dataset_dir/'split.json').read_text())
    d=np.load(exp['dataset']['path'],allow_pickle=False); x=np.asarray(d['x'],dtype=np.float64); y=np.asarray(d['y']).reshape(-1)
    tr=np.asarray(split['train'],dtype=int); va=np.asarray(split['validation'],dtype=int); te=np.asarray(split['test'],dtype=int); fit=np.r_[tr,va]; k=int(exp['dataset']['n_clusters'])
    # The pretrained ZEUS encoder is applied without fitting. PCA/scaling must be fit on fit only.
    import sys; sys.path.insert(0,'/home/luolie/ToPoGate/baseline/CLUBench'); from CLUBench import ZEUS
    zmodel=ZEUS(n_clusters=k,seed=42,device='cuda',gpu=a.gpu,n_init=100)
    raise RuntimeError('ZEUS wrapper exposes fit_predict only; formal strict run is blocked until embedding-only API is added')

if __name__=='__main__': main()
