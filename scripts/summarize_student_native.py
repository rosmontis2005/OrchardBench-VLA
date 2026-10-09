#!/usr/bin/env python3
"""Summarize attempts; optionally revalidate artifacts and create fresh V2 stats.

Statistics use only accepted complete H5 windows. Never read old action_stats.
Run after the manager exits for a final batch report (a running report is partial).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect_student_native import validate, write
from treesim.orchard_command import CONTRACT, encode_window, window_starts


def summarize(root, revalidate=False, statistics=False):
    rows=[];sums=np.zeros((30,32));squares=np.zeros((30,32));count=0
    for path in sorted(root.glob('seed_*/result.json')):
        result=json.loads(path.read_text()); folder=path.parent
        if revalidate and (folder/'trajectory.json').exists(): validate(folder)
        if result['accepted']:
            check=json.loads((folder/'validation.json').read_text());assert check['passed']
            assert check['independent_strict']['strict_success']
            if statistics:
                t=json.loads((folder/'trajectory.json').read_text())
                for start in window_starts(t):
                    a=encode_window(t,start).astype(float);sums+=a;squares+=a*a;count+=1
        rows.append(result)
    phase_pass={key:0 for key in ['GRASP','PULL','TRANSPORT','DROP']}
    for row in rows:
        phases={p['phase'] for p in row.get('phases',[])}
        for phase in phase_pass:phase_pass[phase]+=phase in phases
    result=dict(contract=CONTRACT,attempts=len(rows),accepted=sum(r['accepted'] for r in rows),
                strict_success=sum(r.get('strict',{}).get('strict_success',False) for r in rows),
                failures=dict(Counter(r['reason'] for r in rows if not r['accepted'])),entered_phase=phase_pass,
                held15=sum(any(f['held15_step'] is not None for f in r.get('strict',{}).get('fruit_chains',{}).values()) for r in rows),
                detach=sum(any(f['first_detach_after_grasp_step'] is not None for f in r.get('strict',{}).get('fruit_chains',{}).values()) for r in rows),
                actual_release=sum(bool(r.get('strict',{}).get('release_events',[])) for r in rows),
                usable_windows=sum(r.get('windows',0) for r in rows if r['accepted']),
                failed_attempts_retained=True,rows=rows)
    for key in ['sim_seconds','wall_seconds','ik_failure_rate','clipping_rate','tracking_position_p95','tracking_rotation_p95']:
        vals=[r[key] for r in rows if key in r]
        result[key]=dict(mean=float(np.mean(vals)),p50=float(np.median(vals)),p95=float(np.percentile(vals,95)),max=float(max(vals))) if vals else None
    progress=root/'progress.json'
    result['manager_status']=json.loads(progress.read_text())['status'] if progress.exists() else 'not_managed'
    result['final']=result['manager_status'] in ('complete','attempt_budget_exhausted')
    write(root/'summary.json',result)
    if statistics:
        assert count>0
        mean=sums/count;std=np.sqrt(np.maximum(0,squares/count-mean*mean))
        write(root/'action_stats_requested_v2.json',dict(contract=CONTRACT,source='accepted trajectories in this batch only',source_split='unsplit_candidates',windows=count,action_mean=mean.tolist(),action_std=std.tolist(),active_dims=list(range(7)),epsilon=1e-6,complete_batch=result['final']))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--revalidate',action='store_true');p.add_argument('--stats',action='store_true');a=p.parse_args();summarize(a.root,a.revalidate,a.stats)
