from __future__ import annotations
import csv,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/luolie/ToPoGate/result/v0_rg_ablation_0911winners_first26_20260917_tabular_fixed')
PY='/data/luolie/conda/envs/scssl_bench_py310/bin/python'; GPUS=[1,2,3,5]
def main():
 tasks=list(csv.DictReader((ROOT/'scheduler/tasks.tsv').open(),delimiter='\t')); pending=[]
 for t in tasks:
  p=ROOT/'datasets'/t['dataset']/t['variant']/t['estimator']/f"seed_{t['seed']}"/'final_summary.json'
  if not p.exists(): pending.append(t)
 completed=len(tasks)-len(pending); failed=0; running={}; logs=ROOT/'scheduler/logs'; logs.mkdir(parents=True,exist_ok=True)
 while pending or running:
  for gpu in GPUS:
   if gpu in running or not pending: continue
   t=pending.pop(0); tag=f"{t['dataset']}__{t['variant']}__{t['estimator']}__seed{t['seed']}"; log=(logs/f'{tag}.log').open('w')
   env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
   cmd=[PY,'scripts/V0_RG/run_transductive_tabular_fixed.py','--dataset',t['dataset'],'--variant',t['variant'],'--estimator',t['estimator'],'--seed',t['seed'],'--gpu',str(gpu)]
   running[gpu]=(subprocess.Popen(cmd,cwd='/home/luolie/ToPoGate',env=env,stdout=log,stderr=subprocess.STDOUT),log,tag)
  for gpu,(proc,log,tag) in list(running.items()):
   rc=proc.poll()
   if rc is None: continue
   log.close(); del running[gpu]; completed += rc==0; failed += rc!=0
   with (ROOT/'scheduler/progress.log').open('a') as f: f.write(f'{time.strftime("%F %T")} gpu={gpu} tag={tag} rc={rc} completed={completed} failed={failed} pending={len(pending)}\n')
  time.sleep(2)
 print({'completed':completed,'failed':failed,'total':len(tasks)})
if __name__=='__main__': main()
