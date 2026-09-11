"""Run frozen-winner final test evaluation for one dataset."""
from __future__ import annotations
import argparse, hashlib, json
from dataclasses import fields
from pathlib import Path
import numpy as np
from methods.TopoGate.V0_RG.config import V0_RGConfig
from methods.TopoGate.V0_RG.transductive_runner import run_transductive, SEEDS, PROTOCOL_ID
from methods.TopoGate.V0_RG.tuning import split_rows


def selection_hash(winner: dict) -> str:
    encoded = json.dumps(winner, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--data-path',required=True); p.add_argument('--out',required=True)
    p.add_argument('--n-clusters',type=int,required=True); p.add_argument('--device',default='cpu'); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    selected_path = out / 'selected.json'
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
    else:
        search_path = out / 'search_summary.json'
        search = json.loads(search_path.read_text())
        winner = search['winner']
        selected = {
            'dataset_id': out.name,
            'protocol_id': search.get('protocol_id', PROTOCOL_ID),
            'screen_candidates': search.get('screen_candidates'),
            'selection_seeds': search.get('selection_seeds'),
            'selection_metric': 'validation_ari_mean',
            'selection_tiebreakers': ['validation_ari_std', 'config_hash'],
            'winner': winner,
            'selection_hash': selection_hash(winner),
            'provenance': {
                'source': 'derived_at_final_start_from_existing_search_summary',
                'search_summary_sha256': hashlib.sha256(search_path.read_bytes()).hexdigest(),
                'note': 'Derived before final execution; no candidate ranking was recomputed.',
            },
        }
        atomic_json(selected_path, selected)
    winner=selected['winner']; raw=winner['config']
    frozen_selection_hash=selected.get('selection_hash',selection_hash(winner))
    allowed={f.name for f in fields(V0_RGConfig)}; cfg=V0_RGConfig(**{k:v for k,v in raw.items() if k in allowed})
    from methods.TopoGate.V0_RG.input_adapter import load_matrix
    labels=np.asarray(load_matrix(a.data_path).labels).reshape(-1); _,_,test=split_rows(len(labels))
    records=[]
    for seed in SEEDS:
        rec=run_transductive(a.data_path,out/'final'/f'seed_{seed}',config=cfg,n_clusters=a.n_clusters,seed=seed,device=a.device,score_indices=test,score_name='test')
        records.append(rec)
    metrics={str(r['seed']):r['test_metrics'] for r in records}
    keys=['ari','nmi','acc','ami','f1_macro','fmi']
    mean={k:float(np.mean([m[k] for m in metrics.values()])) for k in keys if k in next(iter(metrics.values()))}
    std={k:float(np.std([m[k] for m in metrics.values()],ddof=1)) for k in mean}
    summary={'dataset_id':out.name,'status':'completed_runtime_audit_pending','protocol_id':PROTOCOL_ID,'evaluation_mode':'transductive','screen_candidates':64,'selection_seeds':list(SEEDS),'final_seeds':list(SEEDS),'selection_hash':frozen_selection_hash,'winner_config_hash':winner['config_hash'],'final_records':len(records),'final_test_metrics':metrics,'final_test_mean':mean,'final_test_std':std}
    atomic_json(out/'final_summary.json',summary)
    print(json.dumps({'dataset_id':out.name,'final_records':len(records),'mean':mean},ensure_ascii=False))
if __name__=='__main__': main()
