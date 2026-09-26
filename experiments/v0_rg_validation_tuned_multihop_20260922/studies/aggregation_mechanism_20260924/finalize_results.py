"""Combine completed fixed analyses, retaining partial outputs and provenance."""
import json
from pathlib import Path
import traceback
import pandas as pd
import numpy as np
from bridge_analysis import analyze
from readout_analysis import select_exploratory
from case_figures import draw

ROOT=Path(__file__).resolve().parent
DATASETS=('Tosches','Baron','fabert')
SEEDS=(42,123,7)

def main():
    result=ROOT/'results';result.mkdir(exist_ok=True)
    failures=[];bridges=[];readouts=[];classes=[];pairs=[];source_stats=[];audits=[]
    for dataset in DATASETS:
        for seed in SEEDS:
            for step in ('readout','source','geometry'):
                out=ROOT/'outputs'/step/dataset/f'seed_{seed}'
                try:
                    s=json.loads((out/'summary.json').read_text(encoding='utf-8-sig'))
                    if s.get('status')!='completed' or s.get('smoke') or s.get('labels_used_for_model_fit') is not False:
                        raise ValueError('Incomplete/smoke/model-fit audit')
                    audits.append({'dataset':dataset,'seed':seed,'step':step,
                                   'posthoc_full_x_labels_loaded':s['posthoc_full_x_labels_loaded'],
                                   'labels_used_for_model_fit':s['labels_used_for_model_fit']})
                    names={'readout':[('readout_repeats.csv',readouts)],
                           'source':[('source_pair_summary.csv',source_stats)],
                           'geometry':[('per_class.csv',classes),('per_class_pair.csv',pairs)]}[step]
                    for name,dest in names:
                        t=pd.read_csv(out/name);t['dataset']=dataset;t['seed']=seed;dest.append(t)
                except Exception as e:failures.append({'dataset':dataset,'seed':seed,'step':step,'error':repr(e)})
            try:bridges.append(analyze(ROOT,dataset,seed))
            except Exception as e:failures.append({'dataset':dataset,'seed':seed,'step':'bridge','error':repr(e)});traceback.print_exc()
        try:
            select_exploratory(dataset,[ROOT/'outputs/readout'/dataset/f'seed_{s}' for s in SEEDS],result/'examples'/dataset)
            draw(ROOT,dataset)
        except Exception as e:failures.append({'dataset':dataset,'step':'selection','error':repr(e)})
    for filename,frames in [('readout_repeats_all.csv',readouts),('geometry_classes_all.csv',classes),
                            ('geometry_pairs_all.csv',pairs),('source_statistics_all.csv',source_stats),
                            ('bridge_classes_all.csv',bridges)]:
        if frames:pd.concat(frames,ignore_index=True).to_csv(result/filename,index=False)
    if readouts:
        t=pd.concat(readouts)
        byseed=t.groupby(['dataset','seed','arm']).ari.mean().unstack('arm')
        byseed['delta']=byseed.h8-byseed.h1
        byseed.to_csv(result/'readout_seed_means.csv')
        plot_readout(t,result)
    pd.DataFrame(audits).to_csv(result/'audit.csv',index=False)
    payload={'status':'completed' if not failures else 'partial','failures':failures,
             'posthoc_full_x_labels_loaded':True,'labels_used_for_model_fit':False,
             'independent_training_seeds':3,'readout_repeats_per_seed':20,
             'scope':'selected_case_descriptive_analysis_not_new_model_selection'}
    (result/'analysis_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    (ROOT/'RESULTS_INDEX.md').write_text('# Aggregation analysis outputs\n\n'
        'See results/readout_seed_means.csv, results/examples/, results/bridge_classes_all.csv, '
        'results/geometry_classes_all.csv, results/geometry_pairs_all.csv, results/source_statistics_all.csv.\n\n'
        'All source/label links are posthoc; no encoder training or model selection occurred. '
        'Repeated readouts are not independent training replicates.\n',encoding='utf-8')
    return 1 if failures else 0

def plot_readout(t,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for ax,d in zip(axs,DATASETS):
        for x,a,c in [(0,'h1','#797A8E'),(1,'h8','#B5465D')]:
            sub=t[(t.dataset==d)&(t.arm==a)]
            for j,s in enumerate(SEEDS):
                vals=sub[sub.seed==s].ari.to_numpy()
                ax.scatter(x+(j-1)*.1+np.linspace(-.025,.025,len(vals)),vals,s=9,color=c,alpha=.45)
        ax.set_xticks([0,1],['H1','H8']);ax.set_title(d);ax.set_ylabel('Full-X ARI')
    fig.tight_layout();fig.savefig(out/'readout_distribution.pdf');fig.savefig(out/'readout_distribution.png',dpi=180);plt.close(fig)

if __name__=='__main__':raise SystemExit(main())
