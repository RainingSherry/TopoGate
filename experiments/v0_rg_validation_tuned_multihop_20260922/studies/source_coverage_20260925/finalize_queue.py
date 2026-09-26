from pathlib import Path
from common import *

def main():
    queue=read_json(EXP/'queue.json')
    assert not (EXP/'state/tasks').exists(),'do not revise active queue'
    more=read_json(EXP/'extension_tasks_transfer.json')['tasks']
    for d in read_json(EXP/'prereg/prereg.json')['cases']:
        deps=[t['id'] for t in queue['tasks'] if t['id'].startswith(f'schntl_{d}_s')]
        more.append({'id':f'baseline_eval_{d}','kind':'cpu','stage':'posthoc','out':f'outputs/baseline/{d}','deps':deps,'dep_mode':'all_terminal','allow_partial':True,'command':['analyze_cases.py','baseline','--dataset',d,'--out',f'outputs/baseline/{d}']})
    tasks=[t for t in queue['tasks'] if t['id']!='summarize']+more
    summ=next(t for t in queue['tasks'] if t['id']=='summarize');summ['deps']=[t['id'] for t in tasks];tasks.append(summ)
    assert len({t['id'] for t in tasks})==len(tasks)
    queue['tasks']=tasks
    queue['task_file_sha256']={t['task_file']:sha256(EXP/t['task_file']) for t in tasks if t.get('task_file')}
    queue['auxiliary_locks_sha256']={'prereg/transfer_lock.json':sha256(EXP/'prereg/transfer_lock.json'),'extension_tasks.json':sha256(EXP/'extension_tasks.json'),'extension_tasks_transfer.json':sha256(EXP/'extension_tasks_transfer.json')}
    atomic_json(EXP/'queue.json',queue);(EXP/'queue.sha256').write_text(sha256(EXP/'queue.json')+'\n')
    print({'tasks':len(tasks),'formal_gpu_fits':sum(t['kind']=='gpu' for t in tasks),'baseline_fits':sum(t.get('validator')=='schntl' for t in tasks)})
if __name__=='__main__':main()
