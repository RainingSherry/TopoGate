"""Join frozen feature-only source evidence to labels AFTER model tasks finish."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from step1_outcomes import load_labels

ROOT=Path(__file__).resolve().parent

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save(path, data):
    if not data:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    output=ROOT/'results';output.mkdir(exist_ok=True)
    background=json.loads((ROOT/'inputs/validation_best_multihop_vs_h1_full_x_60.json').read_text())
    allrows,summary,errors=[],[],[]
    for d in ['Tosches','Baron','fabert']:
        bg=next(x for x in background['rows'] if x['dataset']==d)
        y,names=load_labels(Path(bg['source_path']))
        freq=np.bincount(y,minlength=len(names))/len(y)
        for seed in [42,123,7]:
            base=ROOT/'outputs/steps234'/d/f'seed_{seed}'
            ep=base/'step2_endpoints.csv'
            changes=ROOT/'outputs/step1'/d/'items'/d/f'sample_pair_changes_seed_{seed}.csv'
            if not ep.exists() or not changes.exists():
                errors.append({'dataset':d,'seed':seed,'reason':'source or pair-change artifact unavailable'});continue
            change={int(r['row_index']):r for r in rows(changes)}
            acc={}
            for r in rows(ep):
                i,j,h=int(r['anchor_index']),int(r['source_index']),int(r['hop'])
                group='H1' if h==1 else ('H2_4' if h<=4 else 'H5_8')
                key=(i,group)
                if key not in acc:acc[key]=dict(mass=0.,same=0.,candidate_mass=0.,candidate_same=0.,sources=set())
                a=acc[key];w=float(r['mc_actual_weight']);p=float(r['candidate_probability'])
                a['mass']+=w;a['same']+=w*int(y[i]==y[j]);a['candidate_mass']+=p
                a['candidate_same']+=p*int(y[i]==y[j]);a['sources'].add(j)
            current=[]
            for (i,group),a in acc.items():
                c=change[i];purity=a['same']/a['mass'] if a['mass'] else None
                row=dict(dataset=d,seed=seed,anchor_index=i,class_name=names[y[i]],hop_group=group,
                    actual_mass=a['mass'],sampled_same_class_fraction=purity,
                    same_class_excess_over_prevalence=None if purity is None else purity-freq[y[i]],
                    candidate_same_class_fraction=a['candidate_same']/a['candidate_mass'] if a['candidate_mass'] else None,
                    candidate_unique_sources=len(a['sources']),
                    normalized_pair_error_reduction=int(c['net_error_reduction'])/(len(y)-1),
                    h1_incident_error=int(c['h1_split'])+int(c['h1_merge']),
                    h8_incident_error=int(c['h8_split'])+int(c['h8_merge']))
                current.append(row)
            allrows.extend(current)
            for group in ['H1','H2_4','H5_8']:
                subset=[r for r in current if r['hop_group']==group]
                if not subset:continue
                valid=[r for r in subset if r['sampled_same_class_fraction'] is not None]
                x=[r['actual_mass'] for r in subset];z=[r['normalized_pair_error_reduction'] for r in subset]
                rho=float(spearmanr(x,z).statistic) if len(set(x))>1 and len(set(z))>1 else None
                if rho is not None and not np.isfinite(rho):rho=None
                summary.append(dict(dataset=d,seed=seed,hop_group=group,n_anchors=len(subset),
                    mean_actual_mass=float(np.mean(x)),
                    mean_sampled_same_class_fraction=float(np.mean([r['sampled_same_class_fraction'] for r in valid])) if valid else None,
                    mean_excess_over_prevalence=float(np.mean([r['same_class_excess_over_prevalence'] for r in valid])) if valid else None,
                    spearman_mass_vs_pair_error_reduction=rho,
                    interpretation='posthoc_association_not_causation'))
    save(output/'source_semantics_per_anchor.csv',allrows)
    save(output/'source_semantics_summary.csv',summary)
    (output/'source_semantics_audit.json').write_text(json.dumps(dict(
        status='partial' if errors else 'completed',posthoc_full_x_labels_loaded=True,
        labels_used_for_graph_or_sources=False,source_selection='precommitted_feature_only_anchors',
        gene_and_donor_identity='unavailable_unverified',errors=errors),ensure_ascii=False,indent=2))
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':9})
        fig,axes=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
        for ax,d in zip(axes,['Tosches','Baron','fabert']):
            for j,group in enumerate(['H1','H2_4','H5_8']):
                vals=[r['mean_excess_over_prevalence'] for r in summary if r['dataset']==d and r['hop_group']==group and r['mean_excess_over_prevalence'] is not None]
                if vals:
                    color='#797A8E' if j==0 else '#B5465D'
                    ax.scatter(np.full(len(vals),j)+np.linspace(-.1,.1,len(vals)),vals,c=color,s=20)
                    ax.plot([j-.2,j+.2],[np.mean(vals)]*2,c=color)
            ax.axhline(0,color='#ACBAC1',lw=.8);ax.set_xticks(range(3),['H1','H2–4','H5–8'])
            ax.set_title(d);ax.set_ylabel('Same-class support minus prevalence')
            ax.spines[['top','right']].set_visible(False)
        fig.savefig(output/'source_semantics.png',dpi=220);fig.savefig(output/'source_semantics.pdf');plt.close(fig)

if __name__=='__main__':main()
