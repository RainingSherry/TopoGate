from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path('/data/luolie/ToPoGate/result/v0_rg_biology_ablation_transductive_old_winner_20260915')
OLD=Path('/data/luolie/ToPoGate/result/topogate_transductive_clubench131_scmae16_64x3_20260911/datasets')
DATASETS=['Quake_Smart-seq2_Lung','Quake_10x_Spleen','Macosko','Young']
VARIANTS=['full','base_scmae','anchor_equal_mask','neighbor_equal','no_edge_modulation','fixed_mix_original_aux','mixed_observation_reconstruction']
SEEDS=[42,123,7]
def write(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False,sort_keys=True))
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 write(ROOT/'protocol.json',{'protocol_id':'v0_rg_transductive_old_winner_ablation_20260915_v1','winner_source_protocol':'topogate_transductive_perdataset_64x3_v1','datasets':DATASETS,'variants':VARIANTS,'seeds':SEEDS,'split_seed':9102026,'fit_scope':'all_samples_unlabeled','evaluation_scope':'fixed_test_only','labels_used_during_fit':False,'selection_labels':'validation_only','test_used_for_selection':False,'raw_sha256':'not_computed_per_user_policy','gpu_pool':[1,2,3,5],'forbidden_gpus':[0,7],'full_acceptance':'ARI/NMI/ACC must reproduce old final per seed within 1e-10 before interpreting other variants'})
 manifest=[]
 for n in DATASETS:
  s=json.loads((OLD/n/'selected.json').read_text()); manifest.append({'dataset':n,'data_path':f'/data/luolie/biopipeline/scCluBench/data/scMAE/{n}.h5','selected_path':str(OLD/n/'selected.json'),'selection_hash':s.get('selection_hash'),'winner_config':s['winner']['config']})
 write(ROOT/'manifest.json',{'datasets':manifest})
 tasks=[]
 for n in DATASETS:
  tasks += [{'dataset':n,'variant':v,'estimator':'current','seed':s} for v in VARIANTS for s in SEEDS]
  if n in {'Quake_Smart-seq2_Lung','Macosko'}: tasks += [{'dataset':n,'variant':'full','estimator':e,'seed':s} for e in ['uniform_sample','full'] for s in SEEDS]
 (ROOT/'scheduler').mkdir(exist_ok=True)
 with (ROOT/'scheduler/tasks.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['dataset','variant','estimator','seed'],delimiter='\t'); w.writeheader(); w.writerows(tasks)
 write(ROOT/'scheduler/status.json',{'total':len(tasks),'completed':0,'failed':0,'pending':len(tasks)})
 (ROOT/'README.md').write_text('Paired transductive ablation using the frozen historical per-dataset winners. Original results are read-only.\n')
 print({'root':str(ROOT),'tasks':len(tasks)})
if __name__=='__main__': main()
