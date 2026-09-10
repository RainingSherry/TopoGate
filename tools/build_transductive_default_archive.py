#!/usr/bin/env python3
"""Build a resumable transductive-default archive from runs and low-cost baselines."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

SEEDS=(42,123,7)
MODELS=('ToPoGate_strict8','KMeans','PCA+KMeans','scMAE','IDEC','EDESC','TableDC','ZEUS','scNAME','scDeepCluster','scCDCG')

def acc(y,p):
 a,b=np.unique(y),np.unique(p); m=np.zeros((len(a),len(b)),int)
 for i,x in enumerate(a):
  for j,z in enumerate(b):m[i,j]=np.count_nonzero(p[y==x]==z)
 r,c=linear_sum_assignment(m.max()-m); return float(m[r,c].sum()/len(y))
def met(y,p):return {'ARI':float(adjusted_rand_score(y,p)),'NMI':float(normalized_mutual_info_score(y,p)),'ACC':acc(y,p)}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_manifest(p):
 with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--clubench-manifest',type=Path,required=True);ap.add_argument('--historical-root',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True);a=ap.parse_args(); a.archive.mkdir(parents=True,exist_ok=True)
 rows=load_manifest(a.clubench_manifest); datasets=[r['dataset_id'] for r in rows]; meta={r['dataset_id']:r for r in rows}
 bio=sorted(p.stem for p in a.data_root.glob('*.h5')); allsets=datasets+bio
 ledger=[]; metrics=[]; failures=[]; started=time.time()
 # low-cost transductive classical runs for all CLUBench datasets
 for ds in datasets:
  path=Path(meta[ds]['path']);
  try:
   d=np.load(path,allow_pickle=False); x=np.asarray(d['x'],dtype=np.float32); y=np.asarray(d['y']).reshape(-1); k=int(meta[ds]['n_clusters'])
   for model in ('KMeans','PCA+KMeans'):
    for seed in SEEDS:
     out=a.archive/'runs'/model/ds/f'seed{seed}'; out.mkdir(parents=True,exist_ok=True); jf=out/'run.json'
     if jf.exists():
      try:
       old=json.loads(jf.read_text());
       if old.get('status')=='completed': ledger.append(old); metrics.append(old); continue
      except Exception: pass
     t=time.time()
     if sparse.issparse(x):
      z=StandardScaler(with_mean=False).fit_transform(x)
     else:
      z=StandardScaler().fit_transform(x)
     if model=='KMeans':
      obj=KMeans(n_clusters=k,n_init=20,random_state=seed); pred=obj.fit_predict(z)
     else:
      dim=max(1,min(50,z.shape[0]-1,z.shape[1])); proj=TruncatedSVD(n_components=dim,random_state=seed) if sparse.issparse(z) else PCA(n_components=dim,random_state=seed); z=proj.fit_transform(z); obj=KMeans(n_clusters=k,n_init=20,random_state=seed); pred=obj.fit_predict(z)
     mm=met(y,pred); rec={'model':model,'dataset':ds,'seed':seed,'protocol':'transductive','fit_scope':'all_samples','prediction_scope':'same_samples','test_isolation':'not_applicable','default_config':True,'source':'new_run','status':'completed','labels_used_during_fit':False,'ARI':mm['ARI'],'NMI':mm['NMI'],'ACC':mm['ACC'],'elapsed_seconds':time.time()-t,'fit_seconds':time.time()-t,'device':'cpu','physical_gpu':None,'n_samples':len(y),'n_features':x.shape[1],'n_clusters':k,'data_hash':meta[ds].get('data_sha256',''),'output_path':str(out)}
     jf.write_text(json.dumps(rec,indent=2)); np.save(out/'predictions.npy',pred); ledger.append(rec); metrics.append(rec)
  except Exception as e: failures.append({'model':'KMeans/PCA+KMeans','dataset':ds,'reason':repr(e)})
 # reuse only explicit 3-seed historical transductive run.json records
 for model in ('EDESC','TableDC','ZEUS'):
  for p in a.historical_root.glob(f'*/{model}/seed*/run.json'):
   try:
    r=json.loads(p.read_text());
    if r.get('seed') not in SEEDS or r.get('status')!='completed' or not np.isfinite(float(r.get('ARI',np.nan))):continue
    r.update({'protocol':'transductive','fit_scope':'all_samples','prediction_scope':'same_samples','test_isolation':'not_applicable','source':'historical_reuse','historical_source_path':str(p),'default_config':'wrapper_default_unverified','labels_used_during_fit':r.get('labels_used_during_fit',False)})
    ledger.append(r);metrics.append(r)
   except Exception:continue
 # write long tables and placeholders
 def write(name,fields,data):
  with open(a.archive/name,'w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
 write('run_ledger.csv',sorted({k for r in ledger for k in r}),ledger);write('metrics_per_seed.csv',sorted({k for r in metrics for k in r}),metrics);write('failures.csv',['model','dataset','reason'],failures)
 # dataset manifest
 dm=[dict(r,panel='clubench') for r in rows]+[{'dataset_id':x,'panel':'biology','path':str(a.data_root/(x+'.h5'))} for x in bio];write('dataset_manifest.csv',sorted({k for r in dm for k in r}),dm)
 # wide mean/std table as CSV source for workbook
 wide=[]
 for model in MODELS:
  rec={'model':model}
  for ds in allsets:
   rs=[r for r in metrics if r.get('model')==model and r.get('dataset')==ds and int(r.get('seed',-1)) in SEEDS]
   for q in ('ARI','NMI','ACC'):
    vals=[float(r[q]) for r in rs if q in r and np.isfinite(float(r[q]))]
    rec[f'{ds}__{q}_mean']=round(float(np.mean(vals)),6) if len(vals)==3 else 'NA'; rec[f'{ds}__{q}_std']=round(float(np.std(vals,ddof=1)),6) if len(vals)==3 else 'NA'
   rec[f'{ds}__validation_mean']='NA';rec[f'{ds}__validation_std']='NA'
  wide.append(rec)
 write('default_three_seed_wide.csv',sorted({k for r in wide for k in r}),wide)
 cfg={'protocol':'transductive','seeds':list(SEEDS),'models':MODELS,'clubench_count':len(datasets),'biology_count':len(bio),'default_parameters_only':True,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S')};(a.archive/'protocol.json').write_text(json.dumps(cfg,indent=2));(a.archive/'manifest.json').write_text(json.dumps({'dataset_count':len(allsets),'archive':str(a.archive),'elapsed_seconds':time.time()-started},indent=2))
 print(json.dumps({'clubench':len(datasets),'biology':len(bio),'runs':len(ledger),'failures':len(failures),'archive':str(a.archive)}))
if __name__=='__main__':main()

