"""Per-dataset 64-candidate/3-seed transductive search driver."""
from __future__ import annotations
import argparse, hashlib, json
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
from .config import V0_RGConfig
from .input_adapter import load_matrix
from .tuning import split_rows, initial_candidates, suggest_config, digest
from .transductive_runner import run_transductive, PROTOCOL_ID, SEEDS


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def selection_hash(winner: dict) -> str:
    encoded = json.dumps(winner, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def run_search(data_path, out, *, n_clusters, input_kind="general", device="cpu", trials=64, search_seed=9102026):
    if trials != 64: raise ValueError("this protocol requires exactly 64 candidates")
    out=Path(out); out.mkdir(parents=True,exist_ok=True); loaded=load_matrix(data_path); y=np.asarray(loaded.labels).reshape(-1)
    _, val, test = split_rows(len(y)); base=V0_RGConfig(protocol_id=PROTOCOL_ID,epochs=80,n_top_features=2000)
    candidates=initial_candidates(base)
    import optuna
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=search_seed))
    records=[]
    def eval_cfg(cfg, idx):
        rows=[]
        for seed in SEEDS:
            r=run_transductive(data_path,out/f"candidate_{idx:03d}"/f"seed_{seed}",config=cfg,n_clusters=n_clusters,seed=seed,input_kind=input_kind,device=device,score_indices=val,score_name="validation")
            rows.append(r)
        scores=[r["validation_metrics"]["ari"] for r in rows]; rec={"candidate":idx,"config_hash":digest(asdict(cfg)),"config":asdict(cfg),"seeds":list(SEEDS),"validation_ari_mean":float(np.mean(scores)),"validation_ari_std":float(np.std(scores,ddof=1))}; records.append(rec); return rec["validation_ari_mean"]
    for i,cfg in enumerate(candidates): eval_cfg(cfg,i)
    def objective(trial): return eval_cfg(suggest_config(trial,base),len(records))
    study.optimize(objective,n_trials=trials-len(candidates))
    winner=sorted(records,key=lambda r:(-r["validation_ari_mean"],r["validation_ari_std"],r["config_hash"]))[0]
    selected={"dataset_id":out.name,"protocol_id":PROTOCOL_ID,"screen_candidates":len(records),"selection_seeds":list(SEEDS),"selection_metric":"validation_ari_mean","selection_tiebreakers":["validation_ari_std","config_hash"],"winner":winner,"selection_hash":selection_hash(winner),"provenance":{"source":"written_at_search_completion"}}
    # The scheduler only observes search_summary.json, so publish the frozen
    # selection first and make both files visible atomically.
    atomic_json(out/"selected.json",selected)
    atomic_json(out/"search_summary.json",{"protocol_id":PROTOCOL_ID,"screen_candidates":len(records),"selection_seeds":list(SEEDS),"validation_only":True,"winner":winner,"candidates":records})
    return winner

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("data_path"); p.add_argument("out"); p.add_argument("--n-clusters",type=int,required=True); p.add_argument("--input-kind",default="general"); p.add_argument("--device",default="cpu"); a=p.parse_args(); run_search(a.data_path,a.out,n_clusters=a.n_clusters,input_kind=a.input_kind,device=a.device)
