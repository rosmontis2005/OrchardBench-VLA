#!/usr/bin/env python3
"""Measured-state pilot analysis; includes failures and matched canonical cohorts."""
from __future__ import annotations
from pathlib import Path
import csv,json,os,sys,hashlib
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
os.environ.setdefault('MPLCONFIGDIR','/tmp/arm_retime_pilot_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_arm_retime_pilot import ROOT,OUT,DATA,PROFILES,write

PHASES=['REACH','GRASP','PULL','TRANSPORT','DROP','TERMINAL']
METRICS=['tcp_speed','angular_speed','joint_speed','linear_acceleration','angular_acceleration','joint_acceleration']

def stats(v):
    x=np.asarray(v,dtype=float).ravel();x=x[np.isfinite(x)]
    if not len(x):return dict(count=0,mean=None,P50=None,P90=None,P95=None,P99=None,max=None,min=None)
    return dict(count=len(x),mean=float(x.mean()),**{f'P{p}':float(np.percentile(x,p)) for p in [50,90,95,99]},max=float(x.max()),min=float(x.min()))

def rho(a,b):
    if len(a)<3 or np.ptp(a)<1e-12 or np.ptp(b)<1e-12:return None
    return float(spearmanr(a,b).statistic)

def motion(traj,result):
    if traj is None or traj['num_frames']<3:return None
    p=traj['proprios'];ts=np.asarray(traj['orchardbench']['timestamps']);dt=np.diff(ts)
    assert np.all(dt>0) and np.allclose(dt,1/30,atol=1e-9)
    R=np.asarray(p['ee_rotm']).reshape(-1,3,3);q=np.asarray(p['arm_joint']);pos=np.asarray(p['ee_pos'])
    assert np.isfinite(q).all() and np.isfinite(pos).all() and np.isfinite(R).all()
    v=np.diff(pos,axis=0)/dt[:,None];rv=Rotation.from_matrix(R[:-1].transpose(0,2,1)@R[1:]).as_rotvec()
    omega=np.einsum('nij,nj->ni',R[:-1],rv/dt[:,None]);qv=np.diff(q,axis=0)/dt[:,None]
    mid=(ts[:-1]+ts[1:])/2;dm=np.diff(mid)
    metrics=dict(tcp_speed=np.linalg.norm(v,axis=1),angular_speed=np.linalg.norm(rv,axis=1)/dt,joint_speed=np.max(np.abs(qv),axis=1),
        linear_acceleration=np.r_[np.nan,np.linalg.norm(np.diff(v,axis=0)/dm[:,None],axis=1)],
        angular_acceleration=np.r_[np.nan,np.linalg.norm(np.diff(omega,axis=0)/dm[:,None],axis=1)],
        joint_acceleration=np.r_[np.nan,np.max(np.abs(np.diff(qv,axis=0)/dm[:,None]),axis=1)])
    trace=traj['orchardbench']['fixed_base_expert']['state_trace'];fr=np.array([s['frame'] for s in trace]);bt=fr/60
    phases=np.array(['TERMINAL' if s['state']=='DONE' else s['state'] for s in trace]);labels=phases[np.searchsorted(fr,np.arange(len(ts)-1)*2,side='right')-1]
    phaseinfo={}
    for i,ph in enumerate(phases):
        end=bt[i+1] if i+1<len(bt) else result.get('sim_duration_s',ts[-1]);duration=max(0.,end-bt[i])
        overlap=np.maximum(0,np.minimum(ts[1:],end)-np.maximum(ts[:-1],bt[i]));length=float(np.sum(np.linalg.norm(np.diff(pos,axis=0),axis=1)*overlap/dt))
        phaseinfo[ph]=dict(duration_s=float(duration),path_length_m=length,mean_speed=length/duration if duration else 0,complete=i+1<len(bt),
            recorded_duration_s=float(overlap.sum()))
    event=None
    if 'TRANSPORT' in phases:
        boundary=float(bt[list(phases).index('TRANSPORT')]);frame=int(fr[list(phases).index('TRANSPORT')]);center=int(np.ceil(frame/2));event=dict(time_s=boundary,physics_frame=frame,first_full_transition=center,metrics={},aligned={})
        pre=np.where(ts[1:]<=boundary+1e-9)[0][-5:];post=np.where(ts[:-1]>=boundary-1e-9)[0][:5]
        for key in METRICS:
            aa=metrics[key][pre];bb=metrics[key][post];a=float(np.nanmean(aa)) if len(aa) else None;b=float(np.nanmean(bb)) if len(bb) else None
            event['metrics'][key]=dict(pre=a,post=b,ratio=b/max(a,1e-12) if a is not None and b is not None else None,pre_n=len(aa),post_n=len(bb))
            event['aligned'][key]={str(off):float(metrics[key][center+off]) if np.isfinite(metrics[key][center+off]) else None for off in range(-10,21) if 0<=center+off<len(mid)}
    return dict(ts=ts,mid=mid,labels=labels,metrics=metrics,phases=phaseinfo,event=event)

def aggregate(runs):
    output={}
    for ph in ['ALL']+PHASES:
        output[ph]=dict(episodes_with_phase=sum(r['motion'] is not None and (ph=='ALL' or np.any(r['motion']['labels']==ph)) for r in runs),metrics={})
        for k in METRICS:
            values=[r['motion']['metrics'][k] if ph=='ALL' else r['motion']['metrics'][k][r['motion']['labels']==ph] for r in runs if r['motion'] is not None]
            output[ph]['metrics'][k]=stats(np.concatenate(values) if values else [])
    return output

def transition_aggregate(runs):
    es=[r['motion']['event'] for r in runs if r['motion'] is not None and r['motion']['event'] is not None]
    out=dict(episodes_with_transition=len(es),metrics={},aligned={})
    for k in METRICS:
        pairs=[e['metrics'][k] for e in es if e['metrics'][k]['pre'] is not None and e['metrics'][k]['post'] is not None]
        out['metrics'][k]=dict(pre_mean=stats([e['pre'] for e in pairs]),post_mean=stats([e['post'] for e in pairs]),
            paired_ratio=stats([e['ratio'] for e in pairs]),paired_difference=stats([e['post']-e['pre'] for e in pairs]),
            fraction_post_gt_pre=float(np.mean([e['post']>e['pre'] for e in pairs])) if pairs else None,
            fraction_post_gt_2x_pre=float(np.mean([e['post']>2*e['pre'] for e in pairs])) if pairs else None,
            incomplete_post_window=sum(e['post_n']<5 for e in pairs))
        out['aligned'][k]={str(off):stats([e['aligned'][k][str(off)] for e in es if e['aligned'][k].get(str(off)) is not None]) for off in range(-10,21)}
    return out

def pull_compare(runs,baselines):
    matched=[baselines[r['seed']] for r in runs];pairrows=[]
    for r in runs:
        b=baselines[r['seed']];row=dict(seed=r['seed'],episode_id=r['original_episode_id'],accepted=r['result']['accepted'])
        for tag,run in [('canonical',b),('pilot',r)]:
            mo=run['motion'];p=None if mo is None else mo['phases'].get('PULL');dd={} if run['trajectory'] is None else run['trajectory']['orchardbench']['detach_diagnostics']
            row[tag]=dict(pull_duration_s=None if p is None else p['duration_s'],pull_phase_completed=False if p is None else p['complete'],
                max_pull_N=run['result'].get('max_pull_N'),detached=bool(run['result'].get('detached')),
                **{key:dd.get(key) for key in ['retract_distance_at_detach_m','tcp_displacement_since_pull_enter_m','detach_frame','pull_frames_before_detach','premature_detach']})
        pairrows.append(row)
    complete=[r for r in runs if r['motion'] is not None and r['motion']['phases'].get('PULL',{}).get('complete')]
    keys=['pull_duration_s','retract_distance_at_detach_m','tcp_displacement_since_pull_enter_m','detach_frame','pull_frames_before_detach','max_pull_N']
    summary={}
    for k in keys:
        pairs=[r for r in pairrows if r['pilot'][k] is not None and r['canonical'][k] is not None]
        summary[k]=dict(paired_count=len(pairs),canonical=stats([r['canonical'][k] for r in pairs]),pilot=stats([r['pilot'][k] for r in pairs]),
            paired_difference=stats([r['pilot'][k]-r['canonical'][k] for r in pairs]),paired_ratio=stats([r['pilot'][k]/max(abs(r['canonical'][k]),1e-12) for r in pairs]))
    return dict(attempted=len(runs),reached_pull=sum(r['pilot']['pull_duration_s'] is not None for r in pairrows),completed_pull=len(complete),
        detached=sum(r['pilot']['detached'] for r in pairrows),premature_detach=sum(r['pilot']['premature_detach'] is True for r in pairrows),
        all_attempts_pilot_motion=aggregate(runs)['PULL'],same_seeds_canonical_motion=aggregate(matched)['PULL'],
        completed_pull_pilot_motion=aggregate(complete)['PULL'],completed_pull_same_seeds_canonical_motion=aggregate([baselines[r['seed']] for r in complete])['PULL'],
        paired_scalars=summary,episode_pairs=pairrows)

def diagnostics(runs):
    timing={};rows=[];traces=[]
    for ph,limit in [('REACH',300),('GRASP',240),('PULL',360),('TRANSPORT',420)]:
        items=[]
        for r in runs:
            m=r['motion'];p=None if m is None else m['phases'].get(ph)
            if p is not None:items.append(dict(seed=r['seed'],accepted=r['result']['accepted'],phase_frames=round(p['duration_s']*60),
                frame_timeout_threshold=limit,margin_frames=limit-round(p['duration_s']*60),
                max_observed_stall=max([x['stall_frames'] for x in r['command_trace'] if x['phase']==ph],default=0)))
        timing[ph]=dict(runs=items,accepted_margin_frames=stats([i['margin_frames'] for i in items if i['accepted']]))
    complete=[r['motion']['phases']['TRANSPORT'] for r in runs if r['motion'] is not None and r['motion']['phases'].get('TRANSPORT',{}).get('complete')]
    timing['transport_correlations']=dict(n=len(complete),path_vs_duration=rho([p['path_length_m'] for p in complete],[p['duration_s'] for p in complete]),
        path_vs_mean_speed=rho([p['path_length_m'] for p in complete],[p['mean_speed'] for p in complete]),
        duration_s=stats([p['duration_s'] for p in complete]),path_length_m=stats([p['path_length_m'] for p in complete]))
    for r in runs:
        trace=r['command_trace']
        if len(trace)<2:continue
        time=np.array([x['sim_time'] for x in trace]);q=np.array([x['q_cmd'] for x in trace]);qd=np.array([x['qd_cmd'] for x in trace]);goal=np.array([x['q_goal'] for x in trace]);landing=np.array([x['landing'] for x in trace]);actual=np.diff(q,axis=0)/np.diff(time)[:,None]
        # Only executed steps: terminal command is held after done, not re-slewed.
        active=np.array([trace[i]['phase'] not in ['DONE','FAILED'] or trace[i-1]['phase'] not in ['DONE','FAILED'] for i in range(1,len(trace))])
        acc=np.diff(qd,axis=0)/np.diff(time)[:,None]
        profile=PROFILES[r['profile']];bound=profile.amax_rad_s2
        rdiag=dict(seed=r['seed'],crossing_count=trace[-1]['crossing_count'],max_actual_cmd_velocity=float(np.abs(actual).max()),
            max_internal_velocity=float(np.abs(qd).max()),max_internal_acceleration=float(np.abs(acc[active]).max()) if active.any() else 0,
            acceleration_bound_exceptions=int(np.sum((np.abs(acc)>bound+1e-7)&active[:,None])) if bound else None,
            exceptions_without_landing=int(np.sum((np.abs(acc)>bound+1e-7)&(~landing[1:])&active[:,None])) if bound else None,
            final_failure_sample_time_s=float(time[-1]),last_recorded_time_s=float(r['motion']['ts'][-1]) if r['motion'] is not None else None)
        event=None if r['motion'] is None else r['motion']['event']
        if event:
            mask=(time>=event['time_s'])&(time<=event['time_s']+5/60+1e-8);jumps=np.r_[0,np.linalg.norm(np.diff(goal,axis=0),axis=1)]
            rdiag.update(pt_qgoal_jump_max_rad=float(jumps[mask].max()),pt_initial_goal_command_gap_rad=float(np.linalg.norm((goal-q)[mask],axis=1).max()),
                pt_first_10_cmd_speed=stats(np.max(np.abs(qd[(time>=event['time_s'])&(time<=event['time_s']+10/60+1e-8)]),axis=1)))
        traces.append(rdiag)
    return dict(watchdogs=timing,command_diagnostics=traces)

def csvout(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def analyze():
    if json.loads((OUT/'legacy_regression.json').read_text())['status']!='FAIL':
        raise RuntimeError('This report generator covers the stopped legacy gate; it must not label a future sweep using this failure-report template.')
    selection=json.loads((OUT/'pilot_seed_selection.json').read_text());manifest={r['seed']:r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines()) if r['accepted']}
    baselines={}
    for row in selection['expanded']:
        d=json.loads(Path(row['canonical_path']).read_text());r=dict(seed=row['seed'],original_episode_id=row['episode_id'],profile='canonical',result=manifest[row['seed']],trajectory=d,command_trace=[])
        r['motion']=motion(d,r['result']);baselines[row['seed']]=r
    runs=[]
    for path in sorted((OUT/'runs').glob('*/*/run.json')):
        r=json.loads(path.read_text());r['motion']=motion(r['trajectory'],r['result']);runs.append(r)
    phases={};transitions={};pull={};failures={};details={};flat=[]
    for profile in PROFILES:
        rr=[r for r in runs if r['profile']==profile];smoke=[r for r in rr if r['seed'] in {x['seed'] for x in selection['smoke']}]
        phases[profile]=dict(attempted=len(rr),all_attempts=aggregate(rr),smoke_all_attempts=aggregate(smoke),
            accepted_only=aggregate([r for r in rr if r['result']['accepted']]),same_seeds_canonical=aggregate([baselines[r['seed']] for r in rr]))
        transitions[profile]=dict(all_attempts=transition_aggregate(rr),smoke=transition_aggregate(smoke),same_seeds_canonical=transition_aggregate([baselines[r['seed']] for r in rr]),
            episode_events=[dict(seed=r['seed'],accepted=r['result']['accepted'],event=r['motion']['event']) for r in rr if r['motion'] is not None])
        pull[profile]=pull_compare(rr,baselines);details[profile]=diagnostics(rr)
        reasons=Counter(r['result'].get('reject_reason') or 'accepted' for r in rr)
        categories={k:0 for k in ['grasp_failures','no_fruit','reach_stalled','grasp_stalled','transport_stalled','timeout','branch_break','premature_detach','missed_bucket','other_rejection','execution_error']}
        for r in rr:
            reason=r['result'].get('reject_reason') or '';res=r['result'];matched=False
            tests=dict(grasp_failures=('grasp' in reason),no_fruit=('no_fruit' in reason),reach_stalled=('reach_stalled' in reason),grasp_stalled=('grasp_stalled' in reason),transport_stalled=('transport_stalled' in reason),timeout=('timeout' in reason),branch_break=bool(res.get('branch_break_count')) or reason=='branch_break',premature_detach=(res.get('detach_diagnostics',{}).get('premature_detach') is True or reason=='premature_detach'),missed_bucket=('missed_bucket' in reason),execution_error=reason in ['worker_crash','execution_error'])
            for k,value in tests.items():categories[k]+=int(value);matched|=value
            if reason and not matched:categories['other_rejection']+=1
        failures[profile]=dict(attempted=len(rr),accepted=sum(r['result']['accepted'] for r in rr),placed=sum(bool(r['result'].get('placed')) for r in rr),
            smoke_accepted=sum(r['result']['accepted'] for r in smoke),smoke_attempted=len(smoke),reasons=dict(reasons),categories=categories,
            failed_episodes=[dict(episode_id=r['original_episode_id'],seed=r['seed'],reason=r['result'].get('reject_reason'),last_state=r['result'].get('expert_final_state')) for r in rr if not r['result']['accepted']])
        for r in rr:
            res=r['result'];m=r['motion'];dd=res.get('detach_diagnostics',{})
            row=dict(profile=profile,episode_id=r['original_episode_id'],seed=r['seed'],accepted=res['accepted'],reject_reason=res.get('reject_reason'),grasped=res.get('grasped'),detached=res.get('detached'),placed=res.get('placed'),branch_break_count=res.get('branch_break_count'),max_pull_N=res.get('max_pull_N'),wall_time_s=r['wall_time_s'],duration_s=res.get('sim_duration_s'),phase_chain=' -> '.join(x['state'] for x in res.get('state_trace',[])),
                **{k:dd.get(k) for k in ['detach_frame','pull_frames_before_detach','retract_distance_at_detach_m','tcp_displacement_since_pull_enter_m','premature_detach']})
            for ph in PHASES:
                p=None if m is None else m['phases'].get(ph)
                for k in ['duration_s','path_length_m','mean_speed']:row[ph+'_'+k]=None if p is None else p[k]
            for k in METRICS:row[k+'_P95']=None if m is None else stats(m['metrics'][k])['P95']
            flat.append(row)
    csvout(OUT/'pilot_runs.csv',flat)
    write(OUT/'profile_phase_statistics.json',phases);write(OUT/'profile_transition_statistics.json',transitions);write(OUT/'pull_preservation.json',pull);write(OUT/'failure_summary.json',failures);write(OUT/'command_watchdog_diagnostics.json',details)
    plot(runs,phases,transitions,pull,failures,selection)
    report(selection,phases,transitions,pull,failures,details)
    return phases,transitions,pull,failures,details


def plot(runs,phases,transitions,pull,failures,selection):
    directory=OUT/'figures';directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'figure.dpi':140})
    available=[p for p in PROFILES if phases[p]['attempted']]
    colors=dict(zip(PROFILES,plt.cm.tab10.colors))
    for filename,metric,ylabel in [('profile_tcp_speed.png','tcp_speed','TCP speed (m/s)'),('profile_joint_speed.png','joint_speed','Max |joint FD velocity| (rad/s)'),('profile_acceleration.png','linear_acceleration','TCP acceleration (m/s²)')]:
        fig,ax=plt.subplots(figsize=(9,4));x=np.arange(5)
        for p in available:
            a=phases[p]['all_attempts'];b=phases[p]['same_seeds_canonical']
            ax.plot(x,[a[ph]['metrics'][metric]['P95'] for ph in PHASES[:5]],'o-',label=f'{p} P95 (n={phases[p]["attempted"]})',color=colors[p])
            ax.plot(x,[b[ph]['metrics'][metric]['P95'] for ph in PHASES[:5]],'x--',label=f'Matched canonical P95 ({p} seeds)',alpha=.7)
        ax.set_xticks(x,PHASES[:5]);ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set_title('Regression gate only — retiming sweep not run' if len(available)==1 else 'All attempted trajectories, including failures')
        fig.tight_layout();fig.savefig(directory/filename);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7));offset=np.arange(-10,21)
    for ax,key in zip(axes.flat,['tcp_speed','angular_speed','joint_speed','linear_acceleration']):
        for p in available:
            data=transitions[p]['all_attempts']['aligned'][key]
            ax.plot(offset,[data[str(i)]['P50'] for i in offset],label=f'{p}, n={transitions[p]["all_attempts"]["episodes_with_transition"]}')
            data=transitions[p]['same_seeds_canonical']['aligned'][key]
            ax.plot(offset,[data[str(i)]['P50'] for i in offset],ls='--',label='Matched canonical')
        ax.axvline(0,color='black',lw=.8);ax.set_title(key);ax.set_xlabel('30 Hz transition offset');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('PULL → TRANSPORT: measured responses, gate seeds only' if len(available)==1 else 'PULL → TRANSPORT');fig.tight_layout();fig.savefig(directory/'pull_transport_alignment.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for p in available:
        row=pull[p];keys=['pull_duration_s','retract_distance_at_detach_m','tcp_displacement_since_pull_enter_m','max_pull_N']
        axes[0].plot(range(4),[row['paired_scalars'][k]['paired_ratio']['P50'] for k in keys],'o-',label=p)
        a=row['all_attempts_pilot_motion']['metrics'];b=row['same_seeds_canonical_motion']['metrics']
        axes[1].plot(range(4),[a[k][q]/b[k][q] if a[k][q] is not None and b[k][q] else np.nan for k,q in [('tcp_speed','P50'),('tcp_speed','P95'),('joint_speed','P50'),('joint_speed','P95')]],'o-',label=p)
    axes[0].set_xticks(range(4),['Duration','Retract','TCP displacement','Max pull force'],rotation=15);axes[1].set_xticks(range(4),['TCP P50','TCP P95','Joint P50','Joint P95'])
    for ax in axes:ax.axhline(1,color='gray',ls='--');ax.set_ylabel('Pilot / matched canonical');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(directory/'pull_preservation.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for p in available:
        speed=phases[p]['all_attempts']['TRANSPORT']['metrics']['tcp_speed']['P95'];f=failures[p]
        if speed is not None:ax.scatter(speed,f['accepted']/f['attempted']);ax.annotate(f"{p}: {f['accepted']}/{f['attempted']}",(speed,f['accepted']/f['attempted']),xytext=(6,-12),textcoords='offset points')
    ax.set_xlabel('TRANSPORT measured TCP P95 (m/s)');ax.set_ylabel('Accepted / attempted');ax.set_ylim(0,1.1);ax.set_title('No retiming Pareto comparison: regression gate failed' if len(available)==1 else 'Success versus measured speed');fig.tight_layout();fig.savefig(directory/'success_vs_speed.png');plt.close(fig)
    seeds=[r['seed'] for r in selection['regression']];fig,axes=plt.subplots(3,4,figsize=(15,9))
    for i,seed in enumerate(seeds):
        for r in [r for r in runs if r['seed']==seed and r['motion'] is not None and r['motion']['event'] is not None]:
            boundary=r['motion']['event']['time_s'];tr=r['command_trace'];time=np.array([x['sim_time'] for x in tr])-boundary;mask=(time>=-.3334)&(time<=.667)
            gap=np.linalg.norm(np.array([x['q_goal'] for x in tr])-np.array([x['q_cmd'] for x in tr]),axis=1);vel=np.max(np.abs([x['qd_cmd'] for x in tr]),axis=1)
            axes[i,0].plot(time[mask],gap[mask],label=r['profile']);axes[i,1].plot(time[mask],vel[mask])
            mt=r['motion']['mid']-boundary;mm=(mt>=-.3334)&(mt<=.667)
            axes[i,2].plot(mt[mm],r['motion']['metrics']['joint_speed'][mm]);axes[i,3].plot(mt[mm],r['motion']['metrics']['tcp_speed'][mm])
        for j,title in enumerate(['||q_goal − q_cmd|| rad','max |qd_cmd| rad/s','Measured joint FD rad/s','Measured TCP m/s']):
            axes[i,j].axvline(0,color='black',ls='--',lw=.7);axes[i,j].set_title(f'seed {seed}: {title}');axes[i,j].set_xlabel('Time from PULL → TRANSPORT (s)');axes[i,j].grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(directory/'qgoal_qcmd_measured_trace.png');plt.close(fig)
    # Explicit location of the failed regression, with no time alignment/warping.
    fig,axes=plt.subplots(3,1,figsize=(10,7))
    for r in runs:
        if r['profile']!='legacy' or r['motion'] is None:continue
        row=next(x for x in selection['all_candidates'] if x['seed']==r['seed']);base=json.loads(Path(row['canonical_path']).read_text());n=min(base['num_frames'],r['trajectory']['num_frames']);ts=np.array(base['orchardbench']['timestamps'][:n]);a=base['proprios'];b=r['trajectory']['proprios']
        for ax,key in zip(axes,['ee_pos','arm_joint','gripper_pos']):ax.plot(ts,np.max(np.abs(np.array(a[key][:n])-np.array(b[key][:n])),axis=1),label=row['episode_id']);ax.set_ylabel('max |Δ '+key+'|')
    axes[0].legend();axes[-1].set_xlabel('Identical timestamp (s)');fig.suptitle('Canonical versus legacy rerun errors');fig.tight_layout();fig.savefig(directory/'legacy_regression_errors.png');plt.close(fig)


def report(selection,phases,transitions,pull,failures,details):
    regression=json.loads((OUT/'legacy_regression.json').read_text())
    def f(v):return 'not evaluated' if v is None else f'{v:.6g}'
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    gate_failed=regression['status']!='PASS';status='FAIL' if gate_failed else 'PARTIAL'
    lines=['# Arm retiming pilot','',f'**ARM RETIME PILOT: {status}**','',
        '**Legacy regression failed; no velocity-only or acceleration-limited simulator experiments were run.** The failure is numerical trajectory agreement on the known extreme seed, not task success. All three legacy reruns were accepted and retained the exact canonical state-transition frames. No candidate limits were evaluated or selected.','',
        '## Scope and preserved inputs','',
        'Local HEAD at task start: `26a4b97 refine data collection` (`26a4b97ddb71aacc0677549b9f4aa0b45399ec18`). Initial status and complete user diff are saved alongside this report. Existing user edits to `vla_env.py`, `xr0_adapter.py` and untracked action files were preserved. Only the opt-in expert integration and standalone limiter/pilot tools were added. No XR-0 files, canonical data, physics, fingers, cameras, acceptance criteria, PULL retract increment, timeouts or STALL_FRAMES were changed. No commit/push.','',
        'Each rerun called `collect_episode(... record_rgb=False, detach_force_scale=1.5, arm_motion_profile=legacy)`. Canonical reset-time stance planning, clean rebuild, visibility gates, detach diagnostics, success and rejection gates all ran unchanged. Visibility sensors are still instantiated for the existing acceptance check; no video frames were recorded or encoded.','',
        '## Verified local architecture','',
        'Original `_set_arm` stores an IK goal; `_slew_arm` computes `d = clip(q_goal - q_cmd, -0.045, 0.045); q_cmd = q_cmd + d` once per 60 Hz frame, nominal 2.7 rad/s per joint. FixedBaseAutoPicker calls it after phase update. IK update periods are REACH 15, GRASP 8, PULL 10, TRANSPORT 15 physics frames. PULL retract remains 0.0022 m/frame after 25 settle frames.','',
        'VLAEnv remains separate and unchanged: 0.02 m per translation axis/control step, 0.05 rad per relative Euler axis/control step, 0.045 rad/joint/physics frame; 60 Hz physics with repeat 2 gives 30 Hz control. It adds Cartesian clipping before IK whereas expert directly generates IK joint goals. This report does not change the student controller.','',
        '## Opt-in implementation and numerical tests','',
        '`treesim/arm_motion.py` provides `ArmMotionProfile` and `JointMotionLimiter`, with no phase input. A missing picker profile executes the previous full nine-joint operation unchanged. Explicit profiles replace only its first seven outputs; `_fingers()` and the two finger outputs retain the old path. Command velocity state survives `_goto` and `_set_arm`; reset occurs only at initialization or explicit limiter reset.','',
        'Velocity-acceleration mode uses a conservative discrete stopping bound `sqrt((a*dt/2)^2 + 2*a*|error|) - a*dt/2`, then limits velocity change by `a*dt`. If a suddenly replaced goal lies inside stopping distance, no-crossing and bounded deceleration can conflict. The explicit landing guard snaps only the crossing joint to goal and zeros its velocity, recording `last_landed` and `crossing_count`; this exception is exposed for analysis, not silently treated as a guaranteed hard acceleration bound. No experimental candidate was run.','',
        'Six numerical tests passed (`unit_tests.log`): bit-exact legacy goal sequences including custom rate; velocity bounds/landing; acceleration ramps/stops/reversal; near-goal crossing exception; validation/copy isolation; actual AutoPicker method integration with identical finger outputs and no velocity reset on phase transition. These establish arithmetic equivalence, but do not replace canonical simulator regression.','',
        '## Legacy regression gate','',
        'The limits were declared before comparison: phase timing ±1 physics frame, recorded length ±1 observation, max coordinate TCP difference 1 mm, arm joint 0.002 rad, SO(3) orientation 0.002 rad, measured gripper width 1 mm. Pull-force tolerance is max(0.1 N, 1% of canonical); detach fields use their explicit field tolerances in the runner. Samples are compared at identical timestamps without time warping.','',
        table(['Episode / seed','Accepted','Phase frame differences','TCP max coord error m','Joint max error rad','Orientation max error rad','max pull N canonical → rerun','Gate'],[[f"{r['episode_id']} / {r['seed']}",r['checks'].get('accepted'),str(r['metrics'].get('phase_frame_differences')),f(r['metrics'].get('ee_pos_max_abs_error')),f(r['metrics'].get('arm_joint_max_abs_error')),f(r['metrics'].get('orientation_max_error_rad')),f"{f(r['metrics'].get('max_pull_N',{}).get('canonical'))} → {f(r['metrics'].get('max_pull_N',{}).get('pilot'))}",'PASS' if r['pass_gate'] else 'FAIL'] for r in regression['episodes']]),'',
        'All three: identical state chain REACH→GRASP→PULL→TRANSPORT→DROP→DONE, identical recorded frame count, placed=True, branch breaks=0, detach diagnostics within the declared tolerances. Median/high seeds show micron-level TCP and tens-of-microradian joint differences. The extreme seed exceeds joint/orientation and TCP tolerances and changes max pull force by about 2.43%; these are not dismissed as harmless rounding.','',
        'For seed 1000074, the largest TCP coordinate discrepancy is in PULL at t=1.3667 s (3.909 mm); the largest arm-joint discrepancy is in DROP at t=3.2 s (0.055475 rad). REACH differs by less than 0.84 µm TCP and 1.55 µrad joint; GRASP and subsequent dynamics amplify divergence. The deterministic numerical legacy tests and unchanged default expression do **not** establish why this rerun differs from the historical canonical data. Simulator/contact/IK numerical sensitivity is a possibility, not a demonstrated cause. No original-source control rerun or sweep was performed after the gate failure.','',
        'The experiment stopped at this gate as required. The threshold was not relaxed and no easier replacement seed was substituted.','',
        '## Deterministic seed selection and unrun sweep','',
        table(['Smoke role','Episode','Seed','Old P95 TCP m/s','TRANSPORT path m'],[[r['selection_reason'],r['episode_id'],r['seed'],f(r['p95_tcp_speed']),f(r['transport_path_m'])] for r in selection['smoke']]),'',
        'Thirty expansion seeds were selected before outcomes using a 3×3 TRANSPORT-path / episode-P95-speed rank grid, preserving all eight smoke anchors. Full IDs and seeds are in `pilot_seed_selection.json`. No 8-seed smoke or 30-seed expansion was started. The predeclared catastrophic-smoke rule is ≤2/8 accepted or ≤3/8 detached; it was never applied because legacy gating failed.','',
        table(['Profile','Mode','vmax rad/s','amax rad/s²','Simulator attempted','Accepted','Interpretation'],[[p,PROFILES[p].mode,PROFILES[p].vmax_rad_s,PROFILES[p].amax_rad_s2,failures[p]['attempted'],failures[p]['accepted'],'legacy gate only' if p=='legacy' else 'not run: regression gate failed'] for p in PROFILES]),'',
        '## Measured motion retained from the three gate runs','',
        'These distributions pool only the three rerun episodes. They are not 8-/30-seed profile estimates or evidence that retiming improves success. Main metrics use actual 30 Hz `proprios`: translation difference/dt; SO(3) Log(RᵀR_next)/dt; joint position difference/dt. Angular acceleration transports rotvec velocity into a common world frame before differentiation. Derivatives are raw; no smoothing. Phase labels use interval starts, odd physics-frame boundaries are mixed old/new intervals assigned to the old phase, exactly as in the audit.','',
        table(['Phase','TCP P50','P90','P95','P99','max m/s','acc P95 m/s²','acc max'],[[ph]+[f(phases['legacy']['all_attempts'][ph]['metrics']['tcp_speed'][q]) for q in ['P50','P90','P95','P99','max']]+[f(phases['legacy']['all_attempts'][ph]['metrics']['linear_acceleration'][q]) for q in ['P95','max']] for ph in PHASES[:5]]),'',
        table(['Phase','angular P50','angular P95 rad/s','alpha P95 rad/s²','joint FD P50','joint FD P95 rad/s','joint acc P95 rad/s²'],[[ph]+[f(phases['legacy']['all_attempts'][ph]['metrics'][k][q]) for k,q in [('angular_speed','P50'),('angular_speed','P95'),('angular_acceleration','P95'),('joint_speed','P50'),('joint_speed','P95'),('joint_acceleration','P95')]] for ph in PHASES[:5]]),'',
        '## PULL→TRANSPORT and command evidence','',
        'Alignment uses ceil(physics boundary/2) as first fully new-phase transition, [-10,+20] recorded steps, and five fully-before/fully-after intervals for paired means. No absolute Euler differencing.','',
        table(['Metric','median pre','median post','median paired ratio','fraction post>pre','fraction post>2×pre'],[[k]+[f(transitions['legacy']['all_attempts']['metrics'][k][key]['P50']) for key in ['pre_mean','post_mean','paired_ratio']]+[f(transitions['legacy']['all_attempts']['metrics'][k][key]) for key in ['fraction_post_gt_pre','fraction_post_gt_2x_pre']] for k in ['tcp_speed','angular_speed','joint_speed','linear_acceleration']]),'',
        table(['Seed','largest q_goal jump first 5 frames (rad norm)','largest goal-command gap (rad norm)','early command speed max rad/s'],[[r['seed'],f(r.get('pt_qgoal_jump_max_rad')),f(r.get('pt_initial_goal_command_gap_rad')),f(r.get('pt_first_10_cmd_speed',{}).get('max'))] for r in details['legacy']['command_diagnostics']]),'',
        'The 60 Hz command traces on all three representative seeds directly show a large q_goal replacement followed by saturated 2.7 rad/s command motion and measured acceleration. This supports the proposed mechanism for legacy behaviour on these runs; no acceleration-limited ramp was tested. The phase-change snapshot precedes the TRANSPORT IK update by one physics frame. Command traces are after expert.update, while measured states are after the preceding physics step, so the new command influences subsequent measurements.','',
        '## PULL preservation and failures','',
        'Only a legacy-versus-canonical comparison is available; retimed PULL preservation is **not evaluated**. Both motion tables and scalar pairs retain all attempted runs; candidate profiles have count=0/null, never zero-valued fabricated motion.','',
        table(['PULL scalar','paired n','canonical P50','legacy P50','paired difference P50'],[[k,v['paired_count'],f(v['canonical']['P50']),f(v['pilot']['P50']),f(v['paired_difference']['P50'])] for k,v in pull['legacy']['paired_scalars'].items()]),'',
        f"Legacy detach={pull['legacy']['detached']}/3; premature_detach={pull['legacy']['premature_detach']}; accepted=3/3. No reach/grasp stalls, no-fruit, branch-break, timeout or missed-bucket task rejections were observed in these three gate runs. The failed gate is a regression discrepancy, not an episode rejection. Any future failed run is retained by the runner, including its final odd physics-frame sample in command_trace; main 30 Hz trajectory may end half a frame before that sample. The inherited collector `trajectory_type` string is not used to determine acceptance: result.accepted is authoritative.",'',
        '## Watchdogs and path/time evidence','',
        'No watchdog was changed. Actual successful phase durations and time remaining to the original >300/>240/>360/>420-frame timeouts are below; STALL_FRAMES stays 55. TRANSPORT stall can advance to DROP rather than produce a `transport_stalled` rejection, so lack of that rejection alone is not proof of settling. Per-frame observed stall counters are retained.','',
        table(['Phase','minimum successful timeout margin (frames)','median margin','max margin'],[[ph]+[f(details['legacy']['watchdogs'][ph]['accepted_margin_frames'][q]) for q in ['min','P50','max']] for ph in PHASES[:4]]),'',
        f"Legacy TRANSPORT path-duration Spearman ρ={f(details['legacy']['watchdogs']['transport_correlations']['path_vs_duration'])}; path-mean-speed ρ={f(details['legacy']['watchdogs']['transport_correlations']['path_vs_mean_speed'])}; n={details['legacy']['watchdogs']['transport_correlations']['n']}. This selected three-seed cohort cannot establish a retiming correlation improvement, and is not comparable to the full-300 ρ without matching cohorts. No new-profile timeout effects can be assessed.",'',
        '## Tradeoffs and human review','',
        'No retiming Pareto comparison is possible: velocity-only and acceleration-limited profiles have not been experimentally evaluated. Therefore REACH/GRASP/TRANSPORT slowdown, acceleration-smoothing benefit, candidate task success and PULL changes are unknown. No profile is recommended as a final controller or as an empirically preferred setting.','',
        '**Recommended profiles for HUMAN REVIEW: none on performance evidence.** Review the legacy mismatch, opt-in implementation, and recorded command mechanism first. The six configurations remain coarse planned probes only. This task ends here; no automatic adjustment, timeout extension, extra sampling, training or sweep follows.','',
        '## Artifacts and reproduction','',
        '- `legacy_regression.json`: per-field checks and canonical/pilot detach diagnostics.\n- `pilot_seed_selection.json`: exact seed lists, selection metrics, planned probes and threshold.\n- `runs/legacy/<seed>/run.json`: full measured trajectory, canonical acceptance result, 60 Hz commands and terminal failure sampling.\n- `pilot_runs.csv`: one row per actually attempted run (three).\n- `profile_phase_statistics.json`, `profile_transition_statistics.json`, `pull_preservation.json`, `failure_summary.json`, `command_watchdog_diagnostics.json`: actual metrics; untested candidates explicitly empty.\n- `unit_tests.log`, `input_hashes_before.json`, `source_integrity.json`, `initial_git_status.txt`, `initial_user_changes.diff`: validation/provenance.','',
        'Numerical tests: `.pixi/envs/default/bin/python scripts/test_arm_motion.py`. Read-only analysis of retained runs: `.pixi/envs/default/bin/python scripts/analyze_arm_retime_pilot.py`. The runner checks `legacy_regression.json` before smoke/expansion and refuses this failed gate. No candidate run is enabled by analysis.','']
    for path in sorted((OUT/'figures').glob('*.png')):lines+=['!['+path.stem+'](figures/'+path.name+')','']
    (OUT/'arm_retime_pilot.md').write_text('\n'.join(lines))
    write(OUT/'pilot_status.json',dict(status=status,legacy_regression=regression['status'],retiming_sweep_started=not gate_failed,
        attempted_by_profile={p:failures[p]['attempted'] for p in PROFILES},recommended_profiles_for_human_review=[],
        reason='Extreme canonical seed 1000074 exceeded predeclared legacy trajectory and force tolerances; stopped before candidate experiments.'))


if __name__=='__main__':
    analyze()
