#!/usr/bin/env python3
"""Independent fixed-clock rollouts and resumable process pool; no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'scripts')]
import numpy as np
from scipy.spatial.transform import Rotation
from controller import ControlledExpert, TargetInformationProvider, StageFeedback, MODES
from scenes import reset_override
from collect_student_native import truth, strict_mod, write, serial
from treesim.student_native_expert import StudentNativeExpert, ExpertConfig, approach_rotation
from treesim.orchard_action import native_command
from treesim.vla_env import OrchardVLAEnv, VLAEnvConfig, PoseIK

OUT=ROOT/'artifacts/initialization_prior_1009'


def run(scene, mode, initialization, calibration, folder, shadow=False):
    folder.mkdir(parents=True,exist_ok=True)
    env=strict_mod.make_continuing_env(OrchardVLAEnv)(VLAEnvConfig(max_control_steps=2100))
    strict=strict_mod.StrictPlacement();start=time.monotonic();rows=[];release_diagnostics=[]
    dy={'G0':0.,'G-':-.08,'G+':.08}[initialization]
    summary=dict(scene_seed=scene['seed'],controller_mode=mode,initialization_mode=initialization,
        initialization_valid=scene['initializations'][initialization]['initialization_valid'],
        initial_geometry=scene['initializations'][initialization])
    try:
        obs,info=reset_override(env,scene,dy);initial=truth(env)
        bp=np.array(initial['base_pose']);br=Rotation.from_quat(bp[3:]);cp=bp[:3]
        xyz=br.inv().apply(np.array(initial['fruit_position'])-cp)
        # Independent evaluator checks all fruit positions, home joints and pose.
        actual_apples=env.sim.body_q_np()[np.asarray(env.tm.apple_bodies),:3]
        assert np.allclose(actual_apples,scene['apples_world'],atol=1e-5,rtol=0)
        from treesim import robot
        assert np.allclose(obs['joint_pos'],list(robot._ARM_HOME.values()),atol=1e-6,rtol=0)
        assert abs(float(obs['sim_time']))<1e-12
        summary.update(actual_initial_target_base_xyz=xyz.tolist(),base_pose=bp.tolist(),
            planned_fruit_id=initial['target_id'],target_tcp_distance=float(np.linalg.norm(initial['fruit_position']-obs['tcp_pos_world'])))
        # Endpoint PoseIK is a passive diagnostic, never an exclusion criterion.
        direction=xyz.copy();direction[2]=0;direction/=np.linalg.norm(direction)
        a=xyz.copy();a[2]*=.15;rot=approach_rotation(a)
        q=obs['joint_pos'].copy();checks={};diagnostic_ik=PoseIK(env.tm.model.device)
        for name,point in [('pregrasp',xyz-direction*.16),('grasp',xyz)]:
            q,pe,re=diagnostic_ik.solve(point,rot.as_quat(),q)
            checks[name]=dict(position_error=pe,rotation_error=re,reachable=pe<=.01 and re<=.08)
        summary['initial_pose_ik']=checks
        del diagnostic_ik  # the production IK instance is never touched
        if mode=='original':
            fsm=StudentNativeExpert();controlled=None;estimate=xyz
        else:
            inputs={'initial_height':float(xyz[2])} if mode=='Cz' else {'initial_xyz':xyz.copy()} if mode in ('Cxyz0','Cxyz') else {}
            provider=TargetInformationProvider(mode,calibration,**inputs)
            controlled=ControlledExpert(provider);fsm=controlled.fsm;estimate=provider.initial
        summary['estimated_initial_target_base_xyz']=estimate.tolist()
        original=StudentNativeExpert() if shadow else None
        reason=None;max_shadow_error=0.;planning_calls=0
        with gzip.open(folder/'steps.jsonl.gz','wt') as stream:
            while env.control_steps<2100:
                t=truth(env);chain=strict.summary()['fruit_chains'].get(str(t['target_id']),{})
                if controlled is None:
                    block=fsm.plan(obs,t,chain);estimate=br.inv().apply(t['fruit_position']-cp)
                else:
                    feedback=StageFeedback(held=t['held']==t['target_id'],detached=t['target_detached'],
                        held15_step=chain.get('held15_step'),success_step=chain.get('strict_success_step'),fruit_speed=t['fruit_speed'])
                    current_br=Rotation.from_quat(t['base_pose'][3:]);current_cp=np.asarray(t['base_pose'][:3])
                    live={'current_xyz':current_br.inv().apply(t['fruit_position']-current_cp)} if mode=='Cxyz' else {}
                    block,estimate=controlled.plan(obs,env.control_steps,t['base_pose'],t['bucket_position'],feedback,**live)
                planning_calls+=1
                if original:
                    expected=original.plan(obs,t,chain)
                    assert len(expected)==len(block) and original.phase==fsm.phase
                    for left,right in zip(block,expected):
                        err=max(float(np.max(np.abs(left[k]-right[k]))) for k in ('position','rotation'))
                        max_shadow_error=max(max_shadow_error,err)
                        assert err<2e-6 and left['width']==right['width'] and left['phase']==right['phase'], json.dumps(dict(step=env.control_steps,error=err,left=left,right=right,base_live=t['base_pose'],base_initial=bp),default=serial)
                if not block:break
                assert len(block)==5
                for target in block:
                    step=env.control_steps
                    command=native_command(target['position'],target['rotation'],target['width'],obs['tcp_pos_world'],Rotation.from_quat(obs['tcp_quat_world']).as_matrix())
                    assert np.isfinite(command['action']).all()
                    assert target['planning_step']==step-step%5
                    obs,_,_,truncated,info=env.step(**command)
                    assert env.control_steps==step+1 and abs(obs['sim_time']-(step+1)/30)<1e-7
                    post=truth(env);before=len(strict.release_events)
                    strict.update(step+1,post['held'],post['detached'],post['in_bucket'],info['success'])
                    for event in strict.release_events[before:]:
                        body=env.tm.apple_bodies[event['apple_id']]
                        pos=env.sim.body_q_np()[body,:3];speed=np.linalg.norm(env.sim.state_0.body_qd.numpy()[body,:3])
                        release_diagnostics.append(dict(**event,fruit_bucket_distance=float(np.linalg.norm(pos-post['bucket_position'])),fruit_speed=float(speed)))
                    pe=float(np.linalg.norm(target['position']-obs['tcp_pos_world']))
                    row=dict(step=step+1,phase=target['phase'],command=target,native=command,
                        estimated_target_base_xyz=estimate,fruit_position_diagnostic=post['fruit_position'],
                        tcp=obs['tcp_pos_world'],sim_time=obs['sim_time'],held=post['held'],detached=post['detached'],in_bucket=post['in_bucket'],
                        ik_failed=info['ik_failed'],clipped=info['action_clipped'],tracking_position=pe)
                    stream.write(json.dumps(row,default=serial,allow_nan=False)+'\n')
                    rows.append({k:row[k] for k in ('step','phase','ik_failed','clipped','tracking_position')})
                    # Physical task failures are retained, not dataset rejection gates.
                    if not post['finite']:reason='nonfinite_physics';break
                    if (step+1)%100==0:
                        write(folder/'progress.json',dict(step=step+1,phase=fsm.phase,elapsed=time.monotonic()-start))
                    if truncated:break
                if reason or truncated:break
                if strict.success_events:reason='strict_success';break
        ss=strict.summary();chains=list(ss['fruit_chains'].values());success=bool(strict.success_events)
        reason='strict_success' if success else reason or fsm.failure or 'episode_budget'
        if fsm.phase not in ('DONE','FAILED'):fsm.transition('DONE' if success else 'FAILED',env.control_steps,reason)
        def first(key):
            return min((v[key] for v in chains if v[key] is not None),default=None)
        summary.update(held15=first('held15_step') is not None,
            detach=first('first_detach_after_grasp_step') is not None,
            release=bool(strict.release_events),valid_release=any(e['valid_chain'] for e in strict.release_events),
            stable_bucket=any(v['max_stable_bucket_steps']>=60 for v in chains),
            strict_success_any_fruit=success,
            strict_success_planned_fruit=any(e['apple_id']==initial['target_id'] for e in strict.success_events),
            first_grasp_step=first('first_grasp_step'),release_step=min((e['step'] for e in strict.release_events),default=None),
            success_step=first('strict_success_step'),termination_reason=reason,
            ik_failure_rate=float(np.mean([r['ik_failed'] for r in rows])) if rows else None,
            clipping_rate=float(np.mean([r['clipped'] for r in rows])) if rows else None,
            tracking_position_p95=float(np.percentile([r['tracking_position'] for r in rows],95)) if rows else None,
            controller_steps=env.control_steps,phases=fsm.events,strict=ss,release_diagnostics=release_diagnostics,
            branch_breaks=post['branch_breaks'] if rows else 0,
            incidental_detached_ids=sorted(set(post['detached'])-{initial['target_id']}) if rows else [],
            planning_calls=planning_calls,live_position_updates=controlled.provider.updates if controlled else planning_calls,
            shadow_max_command_error=max_shadow_error if shadow else None,
            wall_seconds=time.monotonic()-start)
        summary['phase_diagnostics']={phase:dict(steps=len(rr),ik_failures=sum(r['ik_failed'] for r in rr),clipped_steps=sum(r['clipped'] for r in rr))
            for phase in {r['phase'] for r in rows} if (rr:=[r for r in rows if r['phase']==phase])}
        write(folder/'result.json',summary)
        return summary
    except Exception as e:
        write(folder/'error.json',dict(**summary,error=str(e),traceback=traceback.format_exc()))
        raise
    finally:env.close()


def job(scene,mode,init,calibration,folder,shadow):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.log').open('a',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        return run(scene,mode,init,calibration,folder,shadow)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--name',required=True)
    p.add_argument('--modes',nargs='+',default=list(MODES));p.add_argument('--initializations',nargs='+',default=['G0'])
    p.add_argument('--workers',type=int,default=6);p.add_argument('--limit',type=int);p.add_argument('--shadow',action='store_true')
    a=p.parse_args();out=OUT/a.name;out.mkdir(exist_ok=True)
    cohort_path=OUT/(a.cohort+'_scenes.json')
    cohort=json.loads(cohort_path.read_text()) if cohort_path.exists() else json.loads((OUT/'scene_manifest.json').read_text())[a.cohort]
    scenes=cohort['scenes']
    if a.limit:scenes=scenes[:a.limit]
    calibration=json.loads((OUT/'calibration.json').read_text())
    tasks=[(s,m,g,calibration,str(out/f"{s['seed']}_{m}_{g}"),a.shadow and m=='Cxyz') for s in scenes for g in a.initializations for m in a.modes]
    configuration=dict(cohort=a.cohort,modes=a.modes,initializations=a.initializations,seeds=[s['seed'] for s in scenes],
        expert=asdict(ExpertConfig()),env=asdict(VLAEnvConfig(max_control_steps=2100)),calibration=calibration,lateral_amplitude=.08)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==json.loads(json.dumps(configuration))
    else:write(out/'config.json',configuration)
    done=[];pending=[]
    for task in tasks:
        path=Path(task[4])/'result.json'
        if path.exists():done.append(json.loads(path.read_text()))
        else:pending.append(task)
    started=time.monotonic();new=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        remaining=iter(pending);active={}
        def submit():
            task=next(remaining,None)
            if task is not None:active[pool.submit(job,*task)]=task
        for _ in range(a.workers):submit()
        while active:
            completed,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            observed=sum(r['controller_steps'] for r in new)
            for task in active.values():
                progress=Path(task[4])/'progress.json'
                if progress.exists():observed+=json.loads(progress.read_text())['step']
            with (out/'throughput.jsonl').open('a') as telemetry:
                telemetry.write(json.dumps(dict(elapsed=time.monotonic()-started,steps=observed,active=len(active),workers=a.workers))+'\n')
            for future in completed:
                active.pop(future)
                result=future.result()  # stop dispatch on engineering errors; drain in-flight
                done.append(result);new.append(result)
                write(out/'progress.json',dict(completed=len(done),total=len(tasks),new_steps=sum(r['controller_steps'] for r in new),
                    elapsed=time.monotonic()-started,workers=a.workers))
                print(json.dumps({k:result[k] for k in ['scene_seed','controller_mode','initialization_mode','strict_success_any_fruit','termination_reason','controller_steps','wall_seconds']}),flush=True)
                submit()
    write(out/'summary.json',dict(completed=len(done),total=len(tasks),workers=a.workers,session_wall_seconds=time.monotonic()-started,
        new_steps=sum(r['controller_steps'] for r in new),successes=sum(r['strict_success_any_fruit'] for r in done)))
if __name__=='__main__':
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    main()
