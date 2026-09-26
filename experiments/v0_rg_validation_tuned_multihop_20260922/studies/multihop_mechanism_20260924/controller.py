"""Restartable bounded queue; labels never determine scheduling or recovery."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
GPUS = (1, 2, 4)
MAX_PER_GPU = 12
MAX_ATTEMPTS = 3

def write(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)

def core_hashes():
    return {str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT.parent/'v0rg_tunable_multihop').glob('*.py'))}

def alive(pid):
    try:
        os.kill(int(pid), 0)
        text = Path(f'/proc/{pid}/stat').read_text()
        return text.split(') ', 1)[1][0] != 'Z'
    except (ProcessLookupError, FileNotFoundError, PermissionError, ValueError):
        return False

def pid_matches(pid, task):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        return ((task['out'] in args and any(task['command'][0] in x for x in args))
                or (task['id'] in args and any('task_wrapper.py' in x for x in args)))
    except (OSError, ValueError):
        return False

def gpu_free():
    result = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
                            capture_output=True,text=True,timeout=20)
    if result.returncode:
        return {}
    return {int(a):int(b) for a,b in (line.split(',') for line in result.stdout.splitlines())}

def main():
    if not (ROOT/'REVIEW_APPROVED.json').exists():
        raise RuntimeError('Parent approval missing')
    queue_path = ROOT/'queue.json'
    queue = json.loads(queue_path.read_text())
    expected = (ROOT/'queue.sha256').read_text().strip()
    if hashlib.sha256(queue_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen queue hash changed')
    baseline_path = ROOT/'core_source_before.json'
    if not baseline_path.exists():
        write(baseline_path, core_hashes())
    elif json.loads(baseline_path.read_text()) != core_hashes():
        raise RuntimeError('Original core source changed before resume')
    state_path = ROOT/'controller_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'tasks':{}, 'protocol':queue['protocol']}
    tasks = {t['id']:t for t in queue['tasks']}
    for key in tasks:
        state['tasks'].setdefault(key, {'status':'pending','attempts':0})
    active = {}
    for key, rec in state['tasks'].items():
        if rec['status'] == 'running':
            if alive(rec['pid']) and pid_matches(rec['pid'], tasks[key]):
                active[key] = None
            else:
                rec['status'] = 'pending' if rec['attempts'] < MAX_ATTEMPTS else 'failed'
    # Recover a worker launched just before a controller-state write.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        pid = int(process.name)
        for key, task in tasks.items():
            rec = state['tasks'][key]
            if rec['status'] == 'pending' and alive(pid) and pid_matches(pid, task):
                rec.update(status='running', pid=pid, started=time.time(), gpu=None)
                active[key] = None
    (ROOT/'logs').mkdir(exist_ok=True)
    while True:
        for key in list(active):
            rec, task = state['tasks'][key], tasks[key]
            proc = active[key]
            code = proc.poll() if proc else (None if alive(rec['pid']) else -1)
            if code is None:
                continue
            marker = Path(task['out'])/'TASK_COMPLETED.json'
            if marker.exists():
                rec.update(status='completed', finished=time.time(), returncode=code)
            else:
                rec.update(status='pending' if rec['attempts'] < MAX_ATTEMPTS else 'failed',
                           finished=time.time(), returncode=code)
                try:
                    tail=Path(rec['log']).read_text(errors='replace')[-4000:]
                except OSError:
                    tail='No readable log'
                rec['last_error']=tail
                if 'out of memory' in tail.lower():
                    state['reduced_concurrency']=True
            del active[key]
        pending=[k for k,r in state['tasks'].items() if r['status']=='pending']
        if not pending and not active:
            break
        free=gpu_free()
        occupancy={g:sum(state['tasks'][k].get('gpu')==g for k in active) for g in GPUS}
        cpus=sum(tasks[k]['kind']=='cpu' for k in active)
        analyses=sum(tasks[k]['kind']=='analysis' for k in active)
        cap=6 if state.get('reduced_concurrency') else MAX_PER_GPU
        for key in pending:
            task,rec=tasks[key],state['tasks'][key]
            gpu=None
            if task['kind']=='gpu':
                options=[g for g in GPUS if occupancy[g]<cap and free.get(g,0)>6500]
                if not options:
                    continue
                gpu=max(options,key=lambda g:free[g]-occupancy[g]*500)
                occupancy[gpu]+=1
                free[gpu]-=3000
            elif task['kind']=='analysis':
                if analyses>=3:
                    continue
                analyses+=1
            else:
                if cpus>=3:
                    continue
                cpus+=1
            rec['attempts']+=1
            log=ROOT/'logs'/f'{key}_attempt{rec["attempts"]}.log'
            env=os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONUNBUFFERED='1',
                       PYTHONPATH=f'{ROOT.parent}:{ROOT}:/home/luolie/ToPoGate',
                       CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '')
            with log.open('ab') as stream:
                proc=subprocess.Popen([sys.executable, str(ROOT/'task_wrapper.py'),key],
                                      cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            active[key]=proc
            rec.update(status='running',pid=proc.pid,started=time.time(),gpu=gpu,log=str(log))
            write(state_path,state)
        state['updated']=time.time()
        state['counts']={s:sum(r['status']==s for r in state['tasks'].values())
                         for s in ('pending','running','completed','failed')}
        write(state_path,state)
        time.sleep(10)
    state['core_source_unchanged']=json.loads(baseline_path.read_text())==core_hashes()
    write(ROOT/'core_source_after.json',core_hashes())
    with (ROOT/'logs/finalize.log').open('ab') as stream:
        result=subprocess.run([sys.executable,str(ROOT/'finalize_results.py')],cwd=ROOT,
                              stdout=stream,stderr=subprocess.STDOUT)
    state['finalization_returncode']=result.returncode
    state['finished']=time.time()
    state['counts']={s:sum(r['status']==s for r in state['tasks'].values())
                    for s in ('pending','running','completed','failed')}
    write(state_path,state)
    ok=not state['counts']['failed'] and result.returncode==0 and state['core_source_unchanged']
    write(ROOT/('PIPELINE_COMPLETED' if ok else 'PIPELINE_PARTIAL'),state['counts'])

if __name__=='__main__':
    main()
