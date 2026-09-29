#!/usr/bin/env python3
"""Isolated retiming pilot, canonical setup/acceptance, no RGB/video or training.

Stages: prepare -> regression (mandatory gate) -> smoke -> expand -> analysis.
Every subprocess runs one seed/profile and saves failures as well as successes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import csv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from treesim.arm_motion import ArmMotionProfile

DATA=ROOT.parent/'data/orchard_autopicker_v1'
OUT=ROOT/'artifacts/arm_retime_pilot'
PROFILES={
    'legacy':ArmMotionProfile(),
    'v2.0':ArmMotionProfile('velocity',2.0),
    'v1.5':ArmMotionProfile('velocity',1.5),
    'va1.5_a20':ArmMotionProfile('velocity_accel',1.5,20),
    'va1.5_a10':ArmMotionProfile('velocity_accel',1.5,10),
    'va1.2_a10':ArmMotionProfile('velocity_accel',1.2,10),
}

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    def scalar(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=scalar));tmp.replace(path)


def prepare():
    # Derive from authoritative canonical JSONs, not a potentially stale CSV.
    rows=[]
    for path in sorted((DATA/'json').glob('*/*.json')):
        d=json.loads(path.read_text());ts=np.array(d['orchardbench']['timestamps']);dp=np.diff(d['proprios']['ee_pos'],axis=0)
        speed=np.linalg.norm(dp,axis=1)/np.diff(ts);tr=d['orchardbench']['fixed_base_expert']['state_trace']
        b={r['state']:r['frame']/60 for r in tr};overlap=np.maximum(0,np.minimum(ts[1:],b['DROP'])-np.maximum(ts[:-1],b['TRANSPORT']))
        rows.append(dict(episode_id=d['episode_id'],seed=d['seed'],split=d['split'],canonical_path=str(path),
            p95_tcp_speed=float(np.percentile(speed,95)),transport_path_m=float(np.sum(np.linalg.norm(dp,axis=1)*overlap/np.diff(ts)))))
    assert len(rows)==300
    speed=sorted(rows,key=lambda x:(x['p95_tcp_speed'],x['episode_id']));length=sorted(rows,key=lambda x:(x['transport_path_m'],x['episode_id']))
    anchors=[('short_transport_path',length[0]),('lowest_episode_p95_speed',speed[0]),('median_speed',speed[len(speed)//2]),
        ('longest_transport_path',length[-1]),('high_speed_p90',speed[int(.9*(len(speed)-1))]),
        ('known_extreme',next(r for r in rows if r['episode_id']=='episode_000029')),
        ('speed_p25',speed[int(.25*(len(speed)-1))]),('speed_p75',speed[int(.75*(len(speed)-1))])]
    smoke=[]
    for role,row in anchors:
        if row['seed'] not in [r['seed'] for r in smoke]:smoke.append(dict(row,selection_reason=role))
    for row in speed:
        if len(smoke)==8:break
        if row['seed'] not in [r['seed'] for r in smoke]:smoke.append(dict(row,selection_reason='deterministic_duplicate_fill'))
    ranks_speed={r['seed']:i for i,r in enumerate(speed)};ranks_length={r['seed']:i for i,r in enumerate(length)}
    cells={}
    for r in rows:
        cell=(min(2,ranks_length[r['seed']]//100),min(2,ranks_speed[r['seed']]//100))
        r['stratum']=list(cell);cells.setdefault(cell,[]).append(r)
    expanded=list(smoke)
    # Cycle 3x3 equal-count rank strata; choose the central remaining member in each cell.
    while len(expanded)<30:
        for cell in sorted(cells):
            available=sorted([r for r in cells[cell] if r['seed'] not in [x['seed'] for x in expanded]],key=lambda r:r['seed'])
            if available and len(expanded)<30:expanded.append(dict(available[len(available)//2],selection_reason=f'rank_stratum_{cell}'))
    regression=[next(r for role,r in anchors if role==x) for x in ['median_speed','high_speed_p90','known_extreme']]
    doc=dict(method='Canonical 30Hz measured TCP; transport chords fractionally allocated at odd physics-frame boundaries. Eight declared extremes/quantiles then 3x3 rank-stratum cycling to 30; no outcome information used.',
        smoke=smoke,expanded=expanded,regression=regression,all_candidates=rows,
        profiles={k:asdict(v) for k,v in PROFILES.items()},detach_force_scale=1.5,record_rgb=False,
        catastrophic_rule='Predeclared: <=2 accepted of 8 OR <=3 detached of 8; do not expand. All other profiles use identical 30 seeds.',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'treesim/arm_motion.py',ROOT/'treesim/picker.py',ROOT/'treesim/fixed_base_picker.py',ROOT/'scripts/collect_autopicker_dataset.py']})
    write(OUT/'pilot_seed_selection.json',doc)
    print('Prepared 8 smoke / 30 expanded / 3 regression seeds',flush=True)
    return doc


def one(profile_name,seed):
    from collect_autopicker_dataset import collect_episode
    selection=json.loads((OUT/'pilot_seed_selection.json').read_text());row=next(r for r in selection['all_candidates'] if r['seed']==seed)
    directory=OUT/'runs'/profile_name/str(seed);directory.mkdir(parents=True,exist_ok=True)
    commands=[]
    def observe(frame,sim,picker,tcp,qidx):
        body=sim.body_q_np()[tcp].copy();q=sim.joint_q_np()[qidx].copy();lim=picker.arm_motion
        commands.append(dict(frame=frame,sim_time=float(sim.sim_time),phase=picker.state,
            q_goal=picker._q_goal[:7].tolist(),q_cmd=picker._q_cmd[:7].tolist(),qd_cmd=lim.qd_cmd.tolist(),
            landing=lim.last_landed.tolist(),crossing_count=lim.crossing_count,
            measured_q=q[:7].tolist(),tcp_position=body[:3].tolist(),tcp_quat=body[3:].tolist(),
            gripper_q_cmd=picker._q_cmd[-2:].tolist(),phase_elapsed_frames=picker._t_state,stall_frames=picker._stall,
            retract_distance_m=float(getattr(picker,'_retract_dist',0))))
    started=time.monotonic()
    try:
        result,traj,images=collect_episode(seed,directory,detach_force_scale=1.5,record_rgb=False,
            arm_motion_profile=PROFILES[profile_name],frame_observer=observe)
        assert not images[0] and not images[1]
        payload=dict(original_episode_id=row['episode_id'],seed=seed,profile=profile_name,profile_config=asdict(PROFILES[profile_name]),
            result=result,trajectory=traj,command_trace=commands,wall_time_s=time.monotonic()-started)
    except Exception:
        payload=dict(original_episode_id=row['episode_id'],seed=seed,profile=profile_name,profile_config=asdict(PROFILES[profile_name]),
            result=dict(accepted=False,reject_reason='execution_error',detached=False),trajectory=None,
            command_trace=commands,wall_time_s=time.monotonic()-started,error=traceback.format_exc())
    write(directory/'run.json',payload)
    print(json.dumps({k:payload[k] for k in ['original_episode_id','seed','profile','wall_time_s']}),flush=True)
    print(json.dumps(payload['result']),flush=True)
    return 1 if 'error' in payload else 0


def run_set(profiles,rows):
    for profile in profiles:
        for row in rows:
            directory=OUT/'runs'/profile/str(row['seed']);path=directory/'run.json'
            if path.exists():continue
            directory.mkdir(parents=True,exist_ok=True)
            with (directory/'stdout.log').open('w') as log:
                done=subprocess.run([sys.executable,str(Path(__file__).resolve()),'one','--profile',profile,'--seed',str(row['seed'])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if path.exists():
                p=json.loads(path.read_text());r=p['result']
                print(f"{profile:12s} {row['episode_id']} seed={row['seed']} accepted={r['accepted']} reason={r.get('reject_reason')} wall={p['wall_time_s']:.1f}s",flush=True)
            else:
                write(directory/'run.json',dict(original_episode_id=row['episode_id'],seed=row['seed'],profile=profile,
                    result=dict(accepted=False,reject_reason='worker_crash',detached=False),trajectory=None,command_trace=[],error=f'exit code {done.returncode}',wall_time_s=None))
                print(f'Worker crash {profile} {row["seed"]}',flush=True)


def regression(selection):
    run_set(['legacy'],selection['regression'])
    comparisons=[]
    limits=dict(phase_frame_difference=1,tcp_position_max_m=.001,arm_joint_max_rad=.002,
        orientation_max_rad=.002,gripper_max_m=.001,detach_frame_difference=1,pull_retract_difference_m=.00221,
        pull_displacement_difference_m=.001,max_pull_force_abs_N=.1,max_pull_force_relative=.01)
    manifest={r['seed']:r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines()) if r['accepted']}
    for row in selection['regression']:
        pilot=json.loads((OUT/'runs/legacy'/str(row['seed'])/'run.json').read_text());base=json.loads(Path(row['canonical_path']).read_text())
        result=pilot['result'];p=pilot['trajectory'];checks={};metrics={}
        checks['accepted']=bool(result['accepted']);checks['placed']=result.get('placed') is True
        checks['branch_break_count']=result.get('branch_break_count')==base['orchardbench']['branch_break_count']==0
        if p is not None:
            oldtr=base['orchardbench']['fixed_base_expert']['state_trace'];newtr=p['orchardbench']['fixed_base_expert']['state_trace']
            checks['phase_chain']=[x['state'] for x in newtr]==[x['state'] for x in oldtr]
            metrics['phase_frame_differences']=[b['frame']-a['frame'] for a,b in zip(oldtr,newtr)]
            checks['phase_timing']=checks['phase_chain'] and max(map(abs,metrics['phase_frame_differences']))<=limits['phase_frame_difference']
            ta=np.array(base['orchardbench']['timestamps']);tb=np.array(p['orchardbench']['timestamps']);n=min(len(ta),len(tb))
            checks['length']=abs(len(ta)-len(tb))<=1;metrics['frame_count_difference']=len(tb)-len(ta)
            for key,limit in [('ee_pos',.001),('arm_joint',.002),('gripper_pos',.001)]:
                a=np.array(base['proprios'][key]);b=np.array(p['proprios'][key]);err=np.abs(a[:n]-b[:n])
                metrics[key+'_max_abs_error']=float(err.max());metrics[key+'_rms_error']=float(np.sqrt(np.mean(err**2)))
                checks[key]=metrics[key+'_max_abs_error']<=limit
            ra=np.array(base['proprios']['ee_rotm'][:n]).reshape(-1,3,3);rb=np.array(p['proprios']['ee_rotm'][:n]).reshape(-1,3,3)
            metrics['orientation_max_error_rad']=float(np.linalg.norm(Rotation.from_matrix(ra.transpose(0,2,1)@rb).as_rotvec(),axis=1).max());checks['orientation']=metrics['orientation_max_error_rad']<=.002
            da=base['orchardbench']['detach_diagnostics'];db=p['orchardbench']['detach_diagnostics'];metrics['detach_diagnostics']={}
            checks['detach_diagnostics']=True
            for key,a in da.items():
                b=db.get(key);metrics['detach_diagnostics'][key]=dict(canonical=a,pilot=b)
                if isinstance(a,(float,int)) and not isinstance(a,bool) and b is not None:
                    tol=1 if 'frame' in key else .00221 if 'retract_distance' in key else .001 if 'displacement' in key else 1/60+.0001 if 'time' in key else max(.1,abs(a)*.01) if key.endswith('_N') else 1e-8
                    checks['detach_diagnostics'] &= abs(a-b)<=tol
                else:checks['detach_diagnostics'] &= a==b
            canonical_force=manifest[row['seed']].get('max_pull_N');pilot_force=result.get('max_pull_N')
            metrics['max_pull_N']=dict(canonical=canonical_force,pilot=pilot_force)
            checks['max_pull_N']=pilot_force is not None and abs(pilot_force-canonical_force)<=max(.1,abs(canonical_force)*.01)
        else:checks['trajectory']=False
        comparisons.append(dict(episode_id=row['episode_id'],seed=row['seed'],pass_gate=all(checks.values()),checks=checks,metrics=metrics))
    doc=dict(status='PASS' if all(r['pass_gate'] for r in comparisons) else 'FAIL',tolerances=limits,
        explanation='Predeclared numerical tolerances; no time warping. Compare samples at identical 30Hz timestamps, and separately check phase frame and trajectory lengths. Exact agreement reported where obtained.',episodes=comparisons)
    write(OUT/'legacy_regression.json',doc)
    print('LEGACY REGRESSION '+doc['status'],flush=True)
    return doc['status']=='PASS'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','one','regression','smoke','expand','all']);ap.add_argument('--profile',choices=list(PROFILES));ap.add_argument('--seed',type=int)
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='one':raise SystemExit(one(args.profile,args.seed))
    if args.stage=='prepare':prepare();return
    selection=json.loads((OUT/'pilot_seed_selection.json').read_text()) if (OUT/'pilot_seed_selection.json').exists() else prepare()
    if args.stage in ['regression','all']:
        if not regression(selection):raise SystemExit(2)
    if args.stage in ['smoke','expand','all']:
        assert json.loads((OUT/'legacy_regression.json').read_text())['status']=='PASS','Legacy regression gate not passed'
    if args.stage in ['smoke','all']:run_set(list(PROFILES),selection['smoke'])
    if args.stage in ['expand','all']:
        viable=[];decisions={}
        for profile in PROFILES:
            results=[json.loads((OUT/'runs'/profile/str(r['seed'])/'run.json').read_text())['result'] for r in selection['smoke']]
            accepted=sum(bool(r['accepted']) for r in results);detached=sum(bool(r.get('detached')) for r in results)
            expand=accepted>2 and detached>3
            decisions[profile]=dict(smoke_attempts=8,accepted=accepted,detached=detached,expand=expand)
            if expand:viable.append(profile)
        write(OUT/'expansion_decisions.json',decisions);run_set(viable,selection['expanded'])

if __name__=='__main__':main()
