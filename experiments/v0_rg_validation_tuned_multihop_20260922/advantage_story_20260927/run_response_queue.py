"""Parent orchestration of nine fixed, already-reviewed response probes."""
import concurrent.futures, hashlib, json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE/'outputs/E5_response'
CASES=['Baron','Quake_Smart-seq2_Lung','student_grade']
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def run(pair):
    ds,seed=pair;out=ROOT/f'{ds}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    rec=out/'record.json'
    if rec.exists():
        data=json.loads(rec.read_text())
        assert data['status']=='completed' and data['labels_used_for_model_fit'] is False
        assert data['test_labels_loaded'] is False
        assert all(digest(out/k)==v for k,v in data['artifact_sha256'].items())
        return dict(dataset=ds,seed=seed,status='completed',reused=True)
    with (out/'execution.log').open('w') as log:
        p=subprocess.run([sys.executable,str(HERE/'response_worker.py'),'--dataset',ds,
                          '--seed',str(seed),'--out',str(out),'--device','cpu'],stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:return dict(dataset=ds,seed=seed,status='failed',returncode=p.returncode)
    return run(pair)
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    code_hash=digest(HERE/'response_worker.py')
    pairs=[(d,s) for d in CASES for s in [7,42,123]]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future={pool.submit(run,p):p for p in pairs}
        for f in concurrent.futures.as_completed(future):
            try:r=f.result()
            except Exception as exc:r=dict(dataset=future[f][0],seed=future[f][1],status='failed',reason=str(exc))
            rows.append(r);print(json.dumps(r),flush=True)
    assert digest(HERE/'response_worker.py')==code_hash
    result=dict(status='completed' if all(r['status']=='completed' for r in rows) else 'partial',
                tasks=rows,worker_sha256=code_hash,new_training=False)
    tmp=ROOT/'QUEUE_RESULT.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(ROOT/'QUEUE_RESULT.json')
