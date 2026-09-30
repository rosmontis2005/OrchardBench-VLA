#!/usr/bin/env python3
"""Fresh-pair statistics, reusing the previous pilot's measured motion analysis."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_arm_retime_pilot import motion, aggregate, transition_aggregate, diagnostics, stats, csvout, METRICS
from run_arm_retime_pilot import write, PROFILES
from run_arm_retime_rounds import OUT, prepare


def load(path):
    r=json.loads(path.read_text());r['motion']=motion(r['trajectory'],r['result']);r['path']=str(path)
    return r


def divide(a,b):
    return float(a/b) if a is not None and b is not None and abs(b)>1e-12 else None


def per_run(r):
    res=r['result'];m=r['motion'];dd=res.get('detach_diagnostics',{})
    out=dict(seed=r['seed'],success=bool(res['accepted']),placed=bool(res.get('placed')),detached=bool(res.get('detached')),
        grasped=bool(res.get('grasped')),branch_break_count=res.get('branch_break_count',0),
        failure_reason=res.get('reject_reason'),timeout='timeout' in (res.get('reject_reason') or ''),
        duration=res.get('sim_duration_s'),phase_chain=' -> '.join(x['state'] for x in res.get('state_trace',[])),
        max_pull_N=res.get('max_pull_N'),
        **{k:dd.get(k) for k in ['premature_detach','detach_frame','pull_frames_before_detach','retract_distance_at_detach_m','tcp_displacement_since_pull_enter_m']})
    if m is None:return out
    prop=r['trajectory']['proprios'];rot=np.array(prop['ee_rotm']).reshape(-1,3,3)
    # Student applies world-relative XYZ Euler increments by left multiplication.
    dp=np.diff(np.array(prop['ee_pos']),axis=0)
    dr=Rotation.from_matrix(rot[1:]@rot[:-1].transpose(0,2,1)).as_euler('xyz')
    translation_exceed=np.any(np.abs(dp)>.02+1e-9,axis=1)
    rotation_exceed=np.any(np.abs(dr)>.05+1e-9,axis=1)
    for ph in ['REACH','GRASP','PULL','TRANSPORT','DROP']:
        info=m['phases'].get(ph,{})
        out[ph+'_duration']=info.get('duration_s')
        out[ph+'_complete']=info.get('complete',False)
        mask=m['labels']==ph
        for name,values in [('translation',translation_exceed),('rotation',rotation_exceed),('either',translation_exceed|rotation_exceed)]:
            out[f'{ph}_student_{name}_exceed_fraction']=float(np.mean(values[mask])) if mask.any() else None
        for metric in METRICS:
            for key,value in stats(m['metrics'][metric][mask]).items():
                if key in ['P50','P95','max']:out[f'{ph}_{metric}_{key}']=value
        for threshold in [1.,2.]:
            out[f'{ph}_high_speed_gt_{threshold:g}_fraction']=float(np.mean(m['metrics']['tcp_speed'][mask]>threshold)) if mask.any() else None
    ev=m['event']
    if ev:
        for metric in METRICS:
            for key in ['pre','post','ratio']:out[f'PT_{metric}_{key}']=ev['metrics'][metric][key]
            aligned=ev['aligned'][metric]
            out[f'PT_{metric}_post_window_max']=max((v for k,v in aligned.items() if 0<=int(k)<5 and v is not None),default=None)
    cmd=r['command_trace'];t=np.array([x['sim_time'] for x in cmd]);q=np.array([x['q_cmd'] for x in cmd]);goal=np.array([x['q_goal'] for x in cmd])
    if len(t)>2:
        v=np.diff(q,axis=0)/np.diff(t)[:,None];a=np.diff(v,axis=0)/np.diff((t[1:]+t[:-1])/2)[:,None]
        out['command_velocity_max']=float(np.max(np.abs(v)))
        out['command_actual_acceleration_max']=float(np.max(np.abs(a)))
        profile=PROFILES[r['profile']];vmax=profile.vmax_rad_s
        for ph in ['REACH','GRASP','PULL','TRANSPORT','DROP']:
            vmax=(profile.phase_vmax_rad_s or {}).get(ph,profile.vmax_rad_s)
            mask=np.array([x['phase']==ph for x in cmd[1:]])
            active=profile.active_phases is None or ph in profile.active_phases
            out[ph+'_limiter_active']=active
            out[ph+'_limiter_saturation_fraction']=float(np.mean(np.any(np.abs(v[mask])>=vmax-1e-6,axis=1))) if mask.any() and active else None
            out[ph+'_command_velocity_max']=float(np.abs(v[mask]).max()) if mask.any() else None
            out[ph+'_max_stall_frames']=max([x['stall_frames'] for x in cmd if x['phase']==ph],default=None)
        if ev:
            mask=(t[1:]>=ev['time_s'])&(t[1:]<=ev['time_s']+5/60+1e-8)
            out['PT_q_goal_jump_max']=float(np.linalg.norm(np.diff(goal,axis=0),axis=1)[mask].max())
            amask=(t[2:]>=ev['time_s'])&(t[2:]<=ev['time_s']+5/30+1e-8)
            out['PT_command_actual_acceleration_max']=float(np.abs(a[amask]).max()) if amask.any() else None
    out['transport_timeout_fallback']=bool(out.get('TRANSPORT_duration',0) and out['TRANSPORT_duration']>7.)
    out['transport_stall_fallback_possible']=bool((out.get('TRANSPORT_max_stall_frames') or 0)>110)
    return out


def identity(b,c):
    checks={}
    for key in ['selected_apple_debug_index','selected_apple_initial_world_pose','selected_apple_radius','selected_base_pose_xy_yaw','standoff_m','azimuth_offset_rad']:
        x=b['result'].get('stance',{}).get(key);y=c['result'].get('stance',{}).get(key)
        checks[key]=bool(x is not None and y is not None and np.allclose(x,y,atol=1e-9,rtol=0))
    for key in ['ee_pos','ee_rotm','arm_joint','gripper_pos']:
        checks['initial_'+key]=bool(b['trajectory'] is not None and c['trajectory'] is not None and np.allclose(b['trajectory']['proprios'][key][0],c['trajectory']['proprios'][key][0],atol=1e-7,rtol=0))
    checks['seed']=b['seed']==c['seed']
    checks['detach_force_multiplier']=b['result'].get('detach_force_multiplier')==c['result'].get('detach_force_multiplier')==1.5
    return dict(valid=all(checks.values()),checks=checks)


def cohort(rr):
    return dict(N=len(rr),success_count=sum(r['result']['accepted'] for r in rr),
        success_rate=sum(r['result']['accepted'] for r in rr)/len(rr) if rr else None,
        placed=sum(bool(r['result'].get('placed')) for r in rr),detached=sum(bool(r['result'].get('detached')) for r in rr),
        failure_types=dict(Counter(r['result']['reject_reason'] for r in rr if not r['result']['accepted'])),
        branch_breaks=sum(r['result'].get('branch_break_count',0) for r in rr),
        premature_detach=sum(r['result'].get('detach_diagnostics',{}).get('premature_detach') is True for r in rr))


def scoped_diagnostics(runs):
    result=diagnostics(runs)
    for diag,r in zip(result['command_diagnostics'],runs):
        p=PROFILES[r['profile']]
        if p.active_phases is None:continue
        trace=r['command_trace'];t=np.array([x['sim_time'] for x in trace])
        qd=np.array([x['qd_cmd'] for x in trace]);acc=np.diff(qd,axis=0)/np.diff(t)[:,None]
        active=np.array([x['phase'] in p.active_phases for x in trace[1:]])
        diag['bound_scope']=list(p.active_phases)
        diag['max_internal_acceleration_in_scope']=float(np.abs(acc[active]).max()) if active.any() else None
        if p.amax_rad_s2:
            landing=np.array([x['landing'] for x in trace[1:]])
            exceed=(np.abs(acc)>p.amax_rad_s2+1e-7)&active[:,None]
            diag['acceleration_bound_exceptions_including_inactive_phases']=diag['acceleration_bound_exceptions']
            diag['acceleration_bound_exceptions']=int(exceed.sum())
            diag['exceptions_without_landing']=int((exceed&~landing).sum())
    return result


def summarize(directory):
    protocol=prepare();spec=json.loads((directory/'spec.json').read_text());types={r['seed']:r['case_type'] for r in protocol['smoke']}
    baseline=[load(directory/'runs/legacy'/str(s)/'run.json') for s in spec['seeds']]
    candidate=baseline if spec['profile']=='legacy' else [load(directory/'runs'/spec['profile']/str(s)/'run.json') for s in spec['seeds']]
    rows=[];ids=[]
    for b,c in zip(baseline,candidate):
        check=identity(b,c);ids.append(dict(seed=b['seed'],**check))
        bs=per_run(b);cs=per_run(c);row=dict(round=spec['round'],profile=spec['profile'],seed=b['seed'],case_type=types[b['seed']],pair_identity_valid=check['valid'],legacy_path=b['path'],candidate_path=c['path'])
        row.update({'legacy_'+k:v for k,v in bs.items() if k!='seed'});row.update({'candidate_'+k:v for k,v in cs.items() if k!='seed'})
        for k in bs:
            if isinstance(bs[k],(float,int)) and not isinstance(bs[k],bool) and k not in ['seed','detach_frame']:
                row[k+'_candidate_vs_legacy_ratio']=divide(cs.get(k),bs[k])
                row[k+'_candidate_minus_legacy']=cs[k]-bs[k] if cs.get(k) is not None else None
        rows.append(row)
    doc=dict(spec=spec,pair_identity=ids,cohorts={},phase_statistics={},transition_statistics={},paired_ratios={},runs=rows,
             command_watchdog_diagnostics={'legacy':scoped_diagnostics(baseline),'candidate':scoped_diagnostics(candidate)})
    for kind in ['normal','stress','all']:
        bb=[r for r in baseline if kind=='all' or types[r['seed']]==kind];cc=[r for r in candidate if kind=='all' or types[r['seed']]==kind]
        preserved=[(b,c) for b,c in zip(bb,cc) if b['result']['accepted']]
        doc['cohorts'][kind]={'legacy':cohort(bb),'candidate':cohort(cc),
            'legacy_success_preservation':dict(N=len(preserved),candidate_success=sum(c['result']['accepted'] for b,c in preserved),
                regressions=[c['seed'] for b,c in preserved if not c['result']['accepted']],
                recoveries=[c['seed'] for b,c in zip(bb,cc) if not b['result']['accepted'] and c['result']['accepted']])}
        doc['phase_statistics'][kind]={'legacy':aggregate(bb),'candidate':aggregate(cc)}
        doc['transition_statistics'][kind]={'legacy':transition_aggregate(bb),'candidate':transition_aggregate(cc)}
        selected=[r for r in rows if kind=='all' or r['case_type']==kind]
        doc['paired_ratios'][kind]={k:stats([r[k] for r in selected if r.get(k) is not None]) for k in rows[0] if k.endswith('_candidate_vs_legacy_ratio')}
    name='round0_legacy_summary.json' if spec['round']==0 else f"round{spec['round']}_summary.json"
    if directory.name!=f"round{spec['round']}":name=directory.name+'_summary.json'
    write(OUT/name,doc)
    allrows=[]
    for path in sorted(OUT.glob('round*_summary.json')):
        # Targeted repeats remain separate from smoke denominators.
        if path.name in ['round0_legacy_summary.json','round1_summary.json','round2_summary.json','round3_summary.json']:
            allrows.extend(json.loads(path.read_text())['runs'])
    csvout(OUT/'paired_seed_results.csv',allrows)
    assert all(x['valid'] for x in ids), 'Pair setup mismatch, inspect saved identity diagnostics'
    return doc

if __name__=='__main__':
    d=summarize(Path(sys.argv[1]));print(json.dumps(d['cohorts'],indent=2))
