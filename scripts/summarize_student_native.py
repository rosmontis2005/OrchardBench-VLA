#!/usr/bin/env python3
"""Summarize attempts; optionally revalidate artifacts and create fresh V2 stats.

Statistics use only accepted complete H5 windows. Never read old action_stats.
Run after the manager exits for a final batch report (a running report is partial).
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect_student_native import validate, write
from treesim.orchard_command import CONTRACT, encode_window, window_starts, load_policy_sample


def audit_sample(folder):
    """Cross-check the independent step stream and real-video policy loader."""
    t=json.loads((folder/'trajectory.json').read_text());n=t['num_commands'];count=0
    with (folder/'steps.jsonl').open() as stream:
        for i,line in enumerate(stream):
            row=json.loads(line)
            assert row['before_index']==i and row['after_index']==i+1
            assert row['command']==t['commands'][i]
            assert row['observation_before']==t['observations'][i]
            assert row['observation_after']==t['observations'][i+1]
            assert {k:v for k,v in row.items() if k not in ('command','observation_before','observation_after')}==t['diagnostics'][i]
            count+=1
    assert count==n
    starts=list(window_starts(t));frames=[starts[0],starts[len(starts)//2],starts[-1]]
    variances=[]
    for frame in frames:
        sample=load_policy_sample(folder,frame)
        assert set(sample)=={'rgb_static','rgb_wrist','instruction','state','action','action_mask'}
        assert sample['action'].shape==(30,32) and sample['state'].shape==(1,32)
        assert np.isfinite(sample['action']).all() and np.isfinite(sample['state']).all()
        variances.append({key:float(np.asarray(sample[key]).var()) for key in ('rgb_static','rgb_wrist')})
    return dict(seed=t['seed'],stream_commands=count,policy_frames=frames,rgb_variances=variances,passed=True)


def inspect_episode(path, revalidate=False, statistics=False):
    result=json.loads(path.read_text()); folder=path.parent
    assert folder.name==f"seed_{result['seed']}", 'Result/episode seed mismatch'
    sums=np.zeros((30,32));squares=np.zeros((30,32));count=0;checked=False
    if revalidate and (folder/'trajectory.json').exists():
        validate(folder);checked=True
    if result['accepted']:
        check=json.loads((folder/'validation.json').read_text());assert check['passed']
        assert check['independent_strict']['strict_success']
        assert result['steps']==check['commands'] and result['windows']==check['windows']
        assert result['strict']==check['independent_strict']
        assert result['controller_quality']['passed']
        target=json.loads((folder/'initial.json').read_text())['truth']['target_id']
        assert any(e['apple_id']==target for e in check['independent_strict']['strict_success_events'])
        if statistics:
            t=json.loads((folder/'trajectory.json').read_text())
            for start in window_starts(t):
                a=encode_window(t,start).astype(float);sums+=a;squares+=a*a;count+=1
    return result,sums,squares,count,checked


def summarize(root, revalidate=False, statistics=False, workers=1):
    assert workers>=1
    rows=[];sums=np.zeros((30,32));squares=np.zeros((30,32));count=0;checked=0
    paths=sorted(root.glob('seed_*/result.json'))
    inspect=partial(inspect_episode,revalidate=revalidate,statistics=statistics)
    pool=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    try:
        results=pool.map(inspect,paths) if pool else map(inspect,paths)
        for row,episode_sum,episode_squares,windows,validated in results:
            rows.append(row);sums+=episode_sum;squares+=episode_squares;count+=windows;checked+=validated
            if len(rows)%50==0 or len(rows)==len(paths):
                print(json.dumps(dict(finalization_completed=len(rows),total=len(paths),revalidated=checked,windows=count)),flush=True)
    finally:
        if pool:pool.shutdown()
    phase_pass={key:0 for key in ['GRASP','PULL','TRANSPORT','DROP']}
    for row in rows:
        phases={p['phase'] for p in row.get('phases',[])}
        for phase in phase_pass:phase_pass[phase]+=phase in phases
    result=dict(contract=CONTRACT,attempts=len(rows),accepted=sum(r['accepted'] for r in rows),
                strict_success=sum(r.get('strict',{}).get('strict_success',False) for r in rows),
                failures=dict(Counter(r['reason'] for r in rows if not r['accepted'])),entered_phase=phase_pass,
                reset_infeasible=sum(r.get('error','').startswith('No feasible fixed-base stance') for r in rows),
                infrastructure_failures=sum(r['reason'] in ('engineering_exception','worker_exception') and not r.get('error','').startswith('No feasible fixed-base stance') for r in rows),
                held15=sum(any(f['held15_step'] is not None for f in r.get('strict',{}).get('fruit_chains',{}).values()) for r in rows),
                detach=sum(any(f['first_detach_after_grasp_step'] is not None for f in r.get('strict',{}).get('fruit_chains',{}).values()) for r in rows),
                actual_release=sum(bool(r.get('strict',{}).get('release_events',[])) for r in rows),
                usable_windows=sum(r.get('windows',0) for r in rows if r['accepted']),
                failed_attempts_retained=True,full_revalidation_requested=revalidate,
                revalidated_trajectories=checked,finalization_workers=workers,rows=rows)
    for key in ['sim_seconds','wall_seconds','ik_failure_rate','clipping_rate','tracking_position_p95','tracking_rotation_p95']:
        vals=[r[key] for r in rows if key in r]
        result[key]=dict(mean=float(np.mean(vals)),p50=float(np.median(vals)),p95=float(np.percentile(vals,95)),max=float(max(vals))) if vals else None
    progress=root/'progress.json'
    manager=json.loads(progress.read_text()) if progress.exists() else None
    result['manager_status']=manager['status'] if manager else 'not_managed'
    result['final']=result['manager_status'] in ('complete','attempt_budget_exhausted')
    if result['final']:
        assert manager['accepted']==result['accepted'] and manager['attempts']==len(rows)
    if revalidate:
        accepted=[r for r in rows if r['accepted']]
        indices=np.linspace(0,len(accepted)-1,min(20,len(accepted)),dtype=int)
        result['sample_audit']=[audit_sample(root/f"seed_{accepted[i]['seed']}") for i in indices]
        print(json.dumps(dict(sample_audit_passed=len(result['sample_audit']))),flush=True)
    write(root/'summary.json',result)
    if statistics:
        assert count>0 and count==result['usable_windows']
        mean=sums/count;std=np.sqrt(np.maximum(0,squares/count-mean*mean))
        write(root/'action_stats_requested_v2.json',dict(contract=CONTRACT,source='accepted trajectories in this batch only',source_split='unsplit_candidates',windows=count,action_mean=mean.tolist(),action_std=std.tolist(),active_dims=list(range(7)),epsilon=1e-6,complete_batch=result['final']))
    if result['final']:
        manifest=root/'accepted_manifest.jsonl';tmp=manifest.with_suffix('.tmp')
        with tmp.open('w') as stream:
            for row in rows:
                if row['accepted']:
                    stream.write(json.dumps(dict(contract=CONTRACT,seed=row['seed'],
                        annotation=str((root/f"seed_{row['seed']}"/'trajectory.json').resolve()),
                        num_commands=row['steps'],windows=row['windows'],split='unsplit_candidates'))+'\n')
        tmp.replace(manifest)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--revalidate',action='store_true');p.add_argument('--stats',action='store_true');p.add_argument('--workers',type=int,default=1);a=p.parse_args();summarize(a.root,a.revalidate,a.stats,a.workers)
