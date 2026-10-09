#!/usr/bin/env python3
"""Read-only throughput/resource samples for a local concurrency experiment."""
import argparse,json,subprocess,time
from pathlib import Path

def sample(root):
    progress=json.loads((root/'progress.json').read_text());steps=0;finished=0;accepted=0;engineering=[]
    for folder in root.glob('seed_*'):
        if not folder.is_dir():continue
        result=folder/'result.json'
        if result.exists():
            r=json.loads(result.read_text());steps+=r.get('steps',0);finished+=1;accepted+=r['accepted']
            if r['reason'] in ('engineering_exception','worker_exception'):engineering.append(dict(seed=r['seed'],reason=r['reason'],error=r.get('error')))
            continue
        stream=folder/'steps.jsonl'
        if stream.exists():
            with stream.open('rb') as f:
                f.seek(max(0,stream.stat().st_size-131072));lines=f.read().splitlines()
            for line in reversed(lines):
                try:r=json.loads(line)
                except (json.JSONDecodeError,UnicodeDecodeError):continue
                if 'after_index' in r:steps+=r['after_index'];break
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))}
    ps=subprocess.check_output(['ps','-eo','ppid=,pid=,pcpu=,rss='],text=True)
    children=[line.split() for line in ps.splitlines() if line.split()[0]==str(progress['pid'])]
    return dict(time=time.time(),workers=progress['workers'],manager_pid=progress['pid'],in_flight=progress.get('in_flight'),finished=finished,accepted=accepted,total_control_steps=steps,gpu_csv=gpu,memory_kib=mem,worker_cpu_percent=sum(float(r[2]) for r in children),worker_rss_kib=sum(int(r[3]) for r in children),child_processes=len(children),engineering_errors=engineering)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=180);a=p.parse_args();start=time.monotonic()
    with a.output.open('a',buffering=1) as f:
        while True:
            r=sample(a.root);line=json.dumps(r);f.write(line+'\n');print(line,flush=True)
            if time.monotonic()-start>=a.seconds:break
            time.sleep(15)
