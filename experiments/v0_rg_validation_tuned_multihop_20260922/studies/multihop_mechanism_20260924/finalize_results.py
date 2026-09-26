"""Posthoc analysis only: no fit, scheduling, or parameter selection from labels."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from v0rg_tunable_multihop.input_adapter import load_matrix, encode_labels

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def metrics(y,p):
    _,a=np.unique(y,return_inverse=True); _,b=np.unique(p,return_inverse=True)
    t=np.zeros((a.max()+1,b.max()+1),dtype=np.int64); np.add.at(t,(a,b),1)
    rr,cc=linear_sum_assignment(-t)
    return dict(ari=float(adjusted_rand_score(y,p)),nmi=float(normalized_mutual_info_score(y,p)),
                acc=float(t[rr,cc].sum()/len(y)))

def main():
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    background=json.loads((ROOT/'inputs/validation_best_multihop_vs_h1_full_x_60.json').read_text())
    rows,errors=[],[]
    for item in background['rows']:
        d=item['dataset']
        if d not in ('Tosches','fabert','Baron'): continue
        loaded=load_matrix(item['source_path']); y,names=encode_labels(loaded.labels)
        for recpath in sorted((ROOT/'outputs/step5'/d).glob('*/seed_*/record_seed_*.json')):
            try:
                rec=json.loads(recpath.read_text()); seed=int(rec['seed']); arm=recpath.parent.parent.name
                if rec.get('test_labels_loaded') is not False or rec.get('labels_used_during_fit') is not False:
                    raise RuntimeError('Training audit invalid')
                pp=recpath.parent/f'predictions_seed_{seed}.npz'
                with np.load(pp,allow_pickle=False) as z: pred=z['predictions']
                if pred.shape!=(len(y),): raise RuntimeError('Prediction row mismatch')
                rows.append(dict(dataset=d,arm=arm,seed=seed,**metrics(y,pred),
                    record=str(recpath),prediction_sha256=hashlib.sha256(pp.read_bytes()).hexdigest()))
            except Exception as exc: errors.append(dict(record=str(recpath),error=repr(exc)))
    fields=['dataset','arm','seed','ari','nmi','acc','record','prediction_sha256']
    with (out/'interventions_per_seed.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=[]
    for d,arm in sorted({(r['dataset'],r['arm']) for r in rows}):
        group=[r for r in rows if (r['dataset'],r['arm'])==(d,arm)]
        refs={r['seed']:r for r in rows if r['dataset']==d and r['arm']=='reference'}
        hs={r['seed']:r for r in rows if r['dataset']==d and r['arm']=='h1_matched_amplitude'}
        x=dict(dataset=d,arm=arm,n_seeds=len(group),seeds=[r['seed'] for r in group])
        for metric in ['ari','nmi','acc']:
            x[metric+'_mean']=float(np.mean([r[metric] for r in group]))
            x[metric+'_std']=float(np.std([r[metric] for r in group],ddof=0))
        for name,reference in [('reference',refs),('h1_matched',hs)]:
            delta=[r['ari']-reference[r['seed']]['ari'] for r in group if r['seed'] in reference]
            x['delta_vs_'+name]=float(np.mean(delta)) if delta else None
            x['positive_vs_'+name]=sum(v>0 for v in delta)
            x['paired_n_vs_'+name]=len(delta)
        summary.append(x)
    audit=dict(evaluation_scope='posthoc_full_X_descriptive',posthoc_full_x_labels_loaded=True,
        test_labels_used_for_training=False,test_labels_used_for_selection=False,
        selection_scope='frozen_controls_no_new_selection',errors=errors)
    write(out/'intervention_summary.json',dict(audit=audit,rows=summary))
    if summary:
        with (out/'intervention_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    manifests=[]
    for p in sorted((ROOT/'outputs').rglob('*.json')):
        if p.name.startswith('record_seed_') or 'summary' in p.name or 'metadata' in p.name:
            manifests.append(str(p.relative_to(ROOT)))
    write(out/'artifact_index.json',dict(artifacts=manifests))
    text=['# ToPoGate 多跳机制研究：逐步骤结果索引','',
          '本报告由冻结队列生成；所有标签仅用于事后描述性分析，不能视为独立测试泛化结果。',
          '训练任务失败或缺失以队列状态和下表样本数为准，不丢弃失败来制造完整结论。','',
          '## 步骤1：改善发生在哪里',
          '逐seed错误分解、逐类指标及重复KMeans见 outputs/step1 各数据集 step1_summary.json 和 CSV。','',
          '## 步骤2–4：实际采样、信息内容和增量探针',
          '见 outputs/steps234 各数据集/seed 的独立结果。重建的采样分布不等同历史轨迹。',
          '生物gene/cell ID映射未核验时，仅允许匿名特征解释；不得据此命名marker/pathway。','',
          '## 步骤5：预先指定的训练干预','',
          '| Dataset | Arm | seeds | ARI mean ± population SD | Δ vs H8 | Δ vs matched H1 |',
          '|---|---|---:|---:|---:|---:|']
    fmt=lambda v:'NA' if v is None else f'{v:+.6f}'
    for s in summary:
        text.append(f'| {s["dataset"]} | {s["arm"]} | {s["n_seeds"]} | {s["ari_mean"]:.6f} ± {s["ari_std"]:.6f} | {fmt(s["delta_vs_reference"])} | {fmt(s["delta_vs_h1_matched"])} |')
    text.extend(['','## 证据判定原则',
        '- H8优于同幅度H1支持深度贡献；两者相同时不能把历史整配置增益全部归因于多跳。',
        '- replace_far_le4退化及rescue50恢复提供来源干预证据；结果不满足时保留相反解释。',
        '- kwide_countmatched和far_shuffle检验路径关联；必须结合来源/位移匹配审计解释。',
        '- mixed_target仅检验重建目标的作用，不是MAGIC或其他论文的正式复现。',
        '- 统计增量探针不自动证明生物功能；后续由主代理结合实际数字整合结论。','',
        '## 产物索引',*['- '+x for x in manifests]])
    (out/'RESULTS_INDEX.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':8})
        arms=['reference','h1_matched_amplitude','replace_far_le4','kwide_countmatched','far_shuffle','rescue50','mixed_target']
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,d in zip(axes,['Tosches','Baron','fabert']):
            for j,arm in enumerate(arms):
                values=[r['ari'] for r in rows if r['dataset']==d and r['arm']==arm]
                if values:
                    color='#B5465D' if arm=='reference' else '#797A8E'
                    ax.scatter(np.full(len(values),j)+np.linspace(-.1,.1,len(values)),values,color=color,s=18)
                    ax.plot([j-.22,j+.22],[np.mean(values)]*2,color=color,lw=2)
            ax.set_title(d);ax.set_xticks(range(len(arms)),arms,rotation=60,ha='right');ax.set_ylabel('Full-X ARI (posthoc)')
            ax.spines[['top','right']].set_visible(False)
        fig.savefig(out/'interventions.png',dpi=220)
        fig.savefig(out/'interventions.pdf')
        plt.close(fig)
    from posthoc_links import main as link_sources
    link_sources()
    if errors: sys.exit(2)

if __name__=='__main__':main()
