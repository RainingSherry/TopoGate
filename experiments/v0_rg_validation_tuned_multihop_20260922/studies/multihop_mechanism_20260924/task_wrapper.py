"""Validate durable artifacts before acknowledging a queue task."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def validate(task,out):
    if task['kind']=='gpu':
        rec=json.loads((out/f'record_seed_{task["seed"]}.json').read_text())
        if rec.get('status')!='completed' or rec.get('test_labels_loaded') is not False or rec.get('labels_used_during_fit') is not False:
            raise RuntimeError('Training label isolation/status audit failed')
        task_path=ROOT/'tasks'/f'{task["id"]}.json'
        if rec.get('task_sha256')!=sha256(task_path):
            raise RuntimeError('Task hash mismatch')
        for name,expected in rec.get('artifact_sha256',{}).items():
            if sha256(out/name)!=expected:raise RuntimeError('Artifact hash mismatch: '+name)
        if not all((out/f'{kind}_seed_{task["seed"]}.{ext}').exists()
                   for kind,ext in [('model','pt'),('predictions','npz'),('embedding','npz')]):
            raise RuntimeError('Missing checkpoint/prediction/embedding')
    else:
        name='steps234_summary.json' if task['kind']=='analysis' else 'step1_summary.json'
        summary=json.loads((out/name).read_text())
        if summary.get('status')!='completed':raise RuntimeError('Analysis incomplete: '+name)

def main():
    key=sys.argv[1]
    task=next(t for t in json.loads((ROOT/'queue.json').read_text())['tasks'] if t['id']==key)
    out=Path(task['out']); out.mkdir(parents=True,exist_ok=True)
    marker=out/'TASK_COMPLETED.json'
    if marker.exists():
        try:
            saved=json.loads(marker.read_text())
            if saved['task_id']!=key:raise RuntimeError('Completion marker task mismatch')
            for name,meta in saved['files'].items():
                if (out/name).stat().st_size!=meta['bytes']:raise RuntimeError('Artifact size changed: '+name)
            validate(task,out)
            return
        except Exception as exc:
            print('Invalid completed artifacts; preserving before rerun:',repr(exc),flush=True)
            archive=ROOT/'failed_attempts'/key/('invalid_'+str(time.time_ns()))
            archive.parent.mkdir(parents=True,exist_ok=True)
            out.rename(archive);out.mkdir(parents=True)
    result=subprocess.run([sys.executable,*task['command']],cwd=ROOT)
    if result.returncode:
        # Preserve the failed attempt's artifacts; the next attempt gets a clean directory.
        if out.exists():
            archive=ROOT/'failed_attempts'/key/str(time.time_ns())
            archive.parent.mkdir(parents=True,exist_ok=True)
            out.rename(archive)
        sys.exit(result.returncode)
    files=[p for p in out.rglob('*') if p.is_file() and p.stat().st_size>0]
    if not files:
        raise RuntimeError('Task exited without any artifacts')
    validate(task,out)
    manifest={str(p.relative_to(out)):{'bytes':p.stat().st_size} for p in files}
    tmp=marker.with_suffix('.tmp')
    tmp.write_text(json.dumps({'task_id':key,'files':manifest,'status':'completed'},indent=2))
    tmp.replace(marker)

if __name__=='__main__':
    main()
