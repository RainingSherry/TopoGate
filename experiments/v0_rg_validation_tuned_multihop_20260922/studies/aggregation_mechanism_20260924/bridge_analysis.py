"""Posthoc links between fixed graph sources and repeated clustering outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def analyze(root: Path, dataset: str, seed: int):
    source=root/'outputs/source'/dataset/f'seed_{seed}'
    readout=root/'outputs/readout'/dataset/f'seed_{seed}'
    out=root/'results/bridges'/dataset/f'seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    fragments=np.load(source/'fragment_ids.npy',allow_pickle=False)
    ys=np.load(source/'labels_true.npy',allow_pickle=False)
    yr=np.load(readout/'labels_true.npy',allow_pickle=False)
    if not np.array_equal(ys,yr):raise ValueError('Label/order mismatch across analyses')
    table=pd.read_csv(source/'anchor_source_links.csv')
    table=table.copy()
    anchor=table.anchor_index.to_numpy(dtype=int)
    donor=table.source_index.to_numpy(dtype=int)
    with np.load(readout/'readout_predictions.npz',allow_pickle=False) as f:
        preds={a:f[a] for a in ('h1','h8')}
    if any(p.ndim!=2 or p.shape[1]!=len(ys) for p in preds.values()):
        raise ValueError('Invalid prediction shape')
    for a,p in preds.items():
        # Bound temporary allocation for large endpoint tables.
        table[a+'_coassignment']=np.concatenate([
            (p[:,anchor[i:i+10000]]==p[:,donor[i:i+10000]]).mean(axis=0)
            for i in range(0,len(anchor),10000)]) if len(anchor) else np.empty(0)
    table['coassignment_delta']=table.h8_coassignment-table.h1_coassignment
    table['same_class']=ys[anchor]==ys[donor]
    table['cross_fragment']=fragments[anchor]!=fragments[donor]
    table['class_id']=ys[anchor]
    table['source_class_id']=ys[donor]
    table['hop_group']=np.where(table.hop==1,'H1',np.where(table.hop<=4,'H2_4','H5_8'))
    # Correct direction: same-class should coassign; different-class should separate.
    table['correct_assignment_delta']=np.where(table.same_class,table.coassignment_delta,-table.coassignment_delta)
    table.to_csv(out/'source_pair_readout.csv',index=False)
    rows=[]
    for keys,t in table.groupby(['class_id','source_class_id','hop_group','same_class','cross_fragment']):
        w=t.mc_actual_weight.to_numpy(); total=float(w.sum())
        if total<=0:continue
        candidate=t.candidate_probability.to_numpy() if 'candidate_probability' in t else w
        rows.append({**dict(zip(['class_id','source_class_id','hop_group','same_class','cross_fragment'],keys)), **
            {'dataset':dataset,'seed':seed,'n_source_pairs':len(t),'n_anchors':t.anchor_index.nunique(),
             'total_mc_mass':total,'h1_coassignment':float(np.average(t.h1_coassignment,weights=w)),
             'h8_coassignment':float(np.average(t.h8_coassignment,weights=w)),
             'correct_assignment_delta':float(np.average(t.correct_assignment_delta,weights=w)),
             'candidate_probability_weighted_correct_delta':float(np.average(t.correct_assignment_delta,weights=candidate))}})
    result=pd.DataFrame(rows)
    result.to_csv(out/'class_source_summary.csv',index=False)
    (out/'provenance.json').write_text(json.dumps({
        'sampling_distribution_reconstructed':True,'mc_repeats':16,
        'weight_interpretation':'mean sampled/reweighted mixture coefficient; not historical training trace',
        'zero_mc_candidates':'retained for candidate-probability sensitivity; zero-total-MC groups omitted',
        'posthoc_full_x_labels_loaded':True,'labels_used_for_model_fit':False,
        'interpretation':'descriptive source-weighted coassignment association, not causal effect',
        'distance_scales':'source full-X RMS pair distance; geometry sampled RMS centroid radius; not numerically interchangeable'
    },indent=2))
    return result

def test_bridge():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        r=Path(tmp);s=r/'outputs/source/toy/seed_42';p=r/'outputs/readout/toy/seed_42'
        s.mkdir(parents=True);p.mkdir(parents=True)
        np.save(s/'fragment_ids.npy',[0,1,2]);np.save(s/'labels_true.npy',[0,0,1]);np.save(p/'labels_true.npy',[0,0,1])
        pd.DataFrame({'anchor_index':[0,0],'source_index':[1,2],'hop':[5,5],'mc_actual_weight':[.5,.5]}).to_csv(s/'anchor_source_links.csv',index=False)
        np.savez(p/'readout_predictions.npz',h1=[[0,1,0],[0,1,0]],h8=[[0,0,1],[0,0,1]])
        out=analyze(r,'toy',42)
        assert np.allclose(out.correct_assignment_delta,1)

if __name__=='__main__':test_bridge()
