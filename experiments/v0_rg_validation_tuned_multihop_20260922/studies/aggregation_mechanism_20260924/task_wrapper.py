"""Durable completion for fixed posthoc analysis; no model training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def validate(out):
    s=json.loads((out/'summary.json').read_text(encoding='utf-8-sig'))
    if s.get('status')!='completed' or s.get('labels_used_for_model_fit') is not False:
        raise RuntimeError('Incomplete analysis or incorrect model-fitting audit')
    if s.get('posthoc_full_x_labels_loaded') is not True:
        raise RuntimeError('Posthoc label provenance missing')
    if s.get('smoke') is True:
        raise RuntimeError('Smoke output cannot satisfy a research task')

def archive(out,key):
    dest=ROOT/'failed_attempts'/key/str(time.time_ns())
    dest.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():out.rename(dest)

def main():
    key=sys.argv[1]
    task=next(t for t in json.loads((ROOT/'queue.json').read_text())['tasks'] if t['id']==key)
    out=Path(task['out']);marker=out/'TASK_COMPLETED.json'
    if marker.exists():
        try:
            m=json.loads(marker.read_text());validate(out)
            assert m['task_id']==key
            assert all(digest(out/n)==v for n,v in m['files'].items())
            return
        except Exception:archive(out,key)
    elif out.exists() and any(out.iterdir()):archive(out,key)
    out.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,*task['command']],cwd=ROOT)
    if result.returncode:sys.exit(result.returncode)
    validate(out)
    files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    tmp=marker.with_suffix('.tmp')
    tmp.write_text(json.dumps({'task_id':key,'status':'completed','files':files},indent=2))
    tmp.replace(marker)

if __name__=='__main__':main()
