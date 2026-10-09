#!/usr/bin/env python3
"""Small local process pool. Per-attempt files, one manifest writer, resumable."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time

ROOT=Path(__file__).resolve().parents[1]


def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def job(seed,output):
    folder=Path(output)/f'seed_{seed}'
    with (Path(output)/f'seed_{seed}.log').open('a',buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        from collect_student_native import run, ExpertConfig, write
        try: return run(seed,folder,ExpertConfig())
        except Exception as exc:
            result=dict(seed=seed,accepted=False,reason='engineering_exception',error=str(exc))
            folder.mkdir(exist_ok=True); write(folder/'result.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort',type=Path);p.add_argument('--target',type=int,default=2000);p.add_argument('--workers',type=int,default=24);p.add_argument('--active-workers',type=int);p.add_argument('--start',type=int,default=8200000);p.add_argument('--gate',type=Path);p.add_argument('--max-attempts',type=int,default=10000);a=p.parse_args()
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.output/'manager.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        print('Stop requested: finishing in-flight episodes before shutdown.',flush=True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    seeds=json.loads(a.cohort.read_text())['seeds'] if a.cohort else list(range(a.start,a.start+a.max_attempts))
    if a.cohort: a.target=len(seeds)
    digest=hashlib.sha256((ROOT/'treesim/student_native_expert.py').read_bytes()).hexdigest()
    if not a.cohort:
        gate=json.loads(a.gate.read_text())
        assert gate['gate_a_pass'] and gate['gate_b_pass'] and gate['gate_c_total']==30 and gate['gate_c_success']>=24
        assert gate['expert_sha256']==digest
    config=dict(seed_sequence_sha256=hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),expert_sha256=digest,workers=a.workers,cohort=str(a.cohort),target=a.target,start=a.start,max_attempts=a.max_attempts)
    if (a.output/'run_config.json').exists():
        old=json.loads((a.output/'run_config.json').read_text());assert {k:v for k,v in old.items() if k!='workers'}=={k:v for k,v in config.items() if k!='workers'},'Resume task configuration changed'
        atomic(a.output/'run_config.json',config)
    else:atomic(a.output/'run_config.json',config)
    (a.output/'manager.pid').write_text(str(os.getpid()))
    results={}
    for seed in seeds:
        folder=a.output/f'seed_{seed}'
        if (folder/'result.json').exists():results[seed]=json.loads((folder/'result.json').read_text())
        elif folder.exists():
            folder.rename(a.output/f'interrupted_seed_{seed}_{time.time_ns()}')
    pending=iter(s for s in seeds if s not in results);started=time.monotonic()
    worker_limit=a.active_workers if a.active_workers is not None else a.workers
    assert 1<=worker_limit<=a.workers
    limit_path=a.output/'worker_limit.json'
    atomic(limit_path,dict(workers=worker_limit))
    with (a.output/'sessions.jsonl').open('a') as session_log:
        session_log.write(json.dumps(dict(pid=os.getpid(),started=time.time(),capacity=a.workers,initial_workers=worker_limit,previous_attempts=len(results),previous_accepted=sum(r['accepted'] for r in results.values())))+'\n')
    active={}
    def summarize(status):
        rows=list(results.values());accepted=sum(bool(r['accepted']) for r in rows)
        atomic(a.output/'progress.json',dict(status=status,pid=os.getpid(),attempts=len(rows),accepted=accepted,target=a.target,workers=worker_limit,worker_capacity=a.workers,in_flight=len(active),updated=time.time(),session_wall_seconds=time.monotonic()-started))
        tmp=a.output/'manifest.tmp';tmp.write_text(''.join(json.dumps(dict(**r,annotation=str((a.output/f"seed_{r['seed']}"/'trajectory.json').resolve())))+'\n' for r in sorted(rows,key=lambda r:r['seed'])));tmp.replace(a.output/'manifest.jsonl')
        return accepted
    summarize('running')
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        active={}
        while True:
            # Lowering the limit drains in-flight work; it never kills an episode.
            requested=json.loads(limit_path.read_text())['workers']
            if not isinstance(requested,int) or not 1<=requested<=a.workers:
                print('Ignoring invalid worker limit: '+repr(requested),flush=True)
            elif requested!=worker_limit:
                worker_limit=requested
                print(json.dumps(dict(event='worker_limit',workers=worker_limit,time=time.time())),flush=True)
            accepted=sum(bool(r['accepted']) for r in results.values())
            while not stopping and len(active)<worker_limit and (a.cohort or accepted+len(active)<a.target):
                seed=next(pending,None)
                if seed is None:break
                active[pool.submit(job,seed,str(a.output))]=seed
            if not active:break
            summarize('stopping' if stopping else 'running')
            done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                seed=active.pop(future)
                try:results[seed]=future.result()
                except Exception as exc:
                    results[seed]=dict(seed=seed,accepted=False,reason='worker_exception',error=str(exc))
                    folder=a.output/f'seed_{seed}';folder.mkdir(exist_ok=True);atomic(folder/'result.json',results[seed])
                summarize('stopping' if stopping else 'running');print(json.dumps(results[seed]),flush=True)
    summarize('stopped' if stopping else 'complete' if a.cohort or sum(r['accepted'] for r in results.values())>=a.target else 'attempt_budget_exhausted')
if __name__=='__main__':main()
