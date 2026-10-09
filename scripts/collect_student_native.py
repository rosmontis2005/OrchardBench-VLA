#!/usr/bin/env python3
"""H5 native collection and independent V2 integrity validation, no training."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from treesim.student_native_expert import StudentNativeExpert, ExpertConfig
from treesim.orchard_action import native_command, decode_targets
from treesim.orchard_command import CONTRACT, encode_window, window_starts, policy_sample
from collect_autopicker_dataset import encode

XR = ROOT.parent/'dualsys/Xiaomi-Robotics-0'
spec = importlib.util.spec_from_file_location('strict_metrics', XR/'xr0/test_grasp_1004/evaluation/strict_metrics.py')
strict_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(strict_mod)


def serial(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    raise TypeError(type(v).__name__)


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,default=serial,allow_nan=False)); tmp.replace(path)


def truth(env):
    from treesim import robot
    bq = env.sim.body_q_np(); fruit = int(env._reset_stance['apple_index'])
    body = int(env.tm.apple_bodies[fruit]); cp, cr = env._chassis_pose()
    fs = strict_mod.fruit_snapshot(env)
    physical = np.flatnonzero(env.sim.apples._held_host[:env.sim.apples.n]).tolist()
    assert physical == ([] if fs['held_apple_id'] is None else [fs['held_apple_id']])
    velocity = env.sim.state_0.body_qd.numpy()[body]
    return dict(step=env.control_steps, target_id=fruit, fruit_position=bq[body,:3],
                fruit_velocity=velocity, fruit_speed=float(np.linalg.norm(velocity[:3])),
                base_pose=bq[env.chassis], bucket_position=cp+cr.apply([robot._BUCKET_CENTER_X,0.,robot._CHASSIS_Z+robot._CHASSIS[2]+robot._BUCKET_WALL_H]),
                held=fs['held_apple_id'], target_detached=bool(env.sim.apples.detached[fruit]),
                detached=fs['detached_apple_ids'], in_bucket=fs['in_bucket_apple_ids'],
                palm_local_fruit=Rotation.from_quat(bq[env.wrist,3:]).inv().apply(bq[body,:3]-bq[env.wrist,:3]),
                branch_breaks=int(env.sim.breaker.broken_count),
                finite=bool(np.isfinite(bq).all() and np.isfinite(env.sim.state_0.body_qd.numpy()).all() and np.isfinite(env.sim.joint_q_np()).all()))


def healthy(t, initial):
    if not t['finite']: return 'nonfinite'
    if t['branch_breaks']: return 'branch_break'
    if set(t['detached'])-{t['target_id']}: return 'incidental_detach'
    if t['held'] is not None and t['held'] != t['target_id']: return 'wrong_fruit'
    if np.linalg.norm(np.asarray(t['base_pose'][:3])-initial['base_pose'][:3]) >= .001: return 'base_drift'
    if (Rotation.from_quat(t['base_pose'][3:])*Rotation.from_quat(initial['base_pose'][3:]).inv()).magnitude() >= .001: return 'base_rotation'


def observation(obs):
    return {k:v for k,v in obs.items() if not k.startswith('rgb_')}


def validate(folder):
    """Reads saved artifacts; never uses a live expert to infer event success."""
    from PIL import Image
    t = json.loads((folder/'trajectory.json').read_text()); n = t['num_commands']
    assert t['contract'] == CONTRACT and n >= 30
    obs, cmds, rows = t['observations'], t['commands'], t['diagnostics']
    assert len(obs) == n+1 and len(cmds) == len(rows) == n
    ts = np.array([o['sim_time'] for o in obs]); assert abs(ts[0]) < 1e-10
    assert np.allclose(ts, np.arange(n+1)/30, atol=1e-8,rtol=0)
    strict = strict_mod.StrictPlacement(); errors = []
    for i,(c,row) in enumerate(zip(cmds,rows)):
        assert c['planning_step'] == i-i%t['execution_horizon']
        assert row['before_index'] == i and row['after_index'] == i+1
        native = native_command(c['position'],c['rotation'],c['width'],obs[i]['tcp_pos_world'],Rotation.from_quat(obs[i]['tcp_quat_world']).as_matrix())
        assert np.allclose(native['action'],c['native']['action'],atol=1e-10,rtol=0)
        assert native['gripper_width'] == c['native']['gripper_width']
        for value in obs[i].values(): assert np.isfinite(np.asarray(value)).all()
        info = row['controller']
        assert info['control_steps'] == i+1
        limits = np.array([.02]*3+[.05]*3+[1.])
        physical = np.clip(native['action'], -limits, limits)
        assert np.allclose(physical, info['physical_action'], atol=1e-10, rtol=0)
        # Effective controller target can differ after clipping; requested labels
        # always retain the original teacher request.
        if np.any(physical[:6]):
            cp=np.asarray(row['truth']['base_pose'][:3]); cr=Rotation.from_quat(row['truth']['base_pose'][3:])
            local=cr.inv().apply(np.asarray(obs[i]['tcp_pos_world'])+physical[:3]-cp)
            expected_p=cp+cr.apply(np.clip(local,[-.65,-.85,.35],[1.,.85,1.55]))
            expected_r=Rotation.from_euler('xyz',physical[3:6])*Rotation.from_quat(obs[i]['tcp_quat_world'])
            assert np.allclose(expected_p,info['target_pose']['position_world'],atol=2e-6,rtol=0)
            assert (expected_r*Rotation.from_quat(info['target_pose']['quaternion_world']).inv()).magnitude()<2e-6
        assert row['failure'] == healthy(row['truth'],t['diagnostics'][0]['truth'])
        if i % t['execution_horizon']:
            assert c['phase']==cmds[i-1]['phase'] and c['width']==cmds[i-1]['width']
        state = row['truth']; strict.update(i+1,state['held'],state['detached'],state['in_bucket'],bool(state['in_bucket']))
    for frame in window_starts(t):
        a = encode_window(t,frame); o = obs[frame]
        pos,rot,width = decode_targets(a,o['tcp_pos_world'],Rotation.from_quat(o['tcp_quat_world']).as_matrix())
        expected = cmds[frame:frame+30]
        err = max(np.max(np.abs(pos-np.array([c['position'] for c in expected]))),np.max(np.abs(rot-np.array([c['rotation'] for c in expected]))),np.max(np.abs(width-np.array([c['width'] for c in expected]))))
        assert err < 2e-6; errors.append(err)
        sample = policy_sample(t,frame,Image.new('RGB',(192,144)),Image.new('RGB',(192,144)))
        assert set(sample)=={'rgb_static','rgb_wrist','instruction','state','action','action_mask'}
        assert sample['action'].shape==(30,32) and sample['state'].shape==(1,32)
        assert np.all(sample['action_mask'][:,:7]==1) and not sample['action_mask'][:,7:].any() and not a[:,7:].any()
    videos={}
    for key,path in t['rgb'].items():
        p = subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(folder/path),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],capture_output=True,check=True)
        assert len(p.stdout)==(n+1)*192*144*3,(key,len(p.stdout),n)
        videos[key]=n+1
    for value in obs[-1].values(): assert np.isfinite(np.asarray(value)).all()
    assert strict.summary()==t['strict'], 'Independent event observer mismatch'
    result=dict(passed=True,commands=n,observations=n+1,windows=len(errors),max_roundtrip_error=max(errors),videos=videos,
                independent_strict=strict.summary(),command_measured_width_max_difference=max(abs(c['width']-o['gripper_width']) for c,o in zip(cmds,obs)),
                replay_claim='saved-command alignment and independent observer; not a deterministic physical replay claim')
    write(folder/'validation.json',result); return result


def run(seed, folder, config):
    from treesim.vla_env import OrchardVLAEnv, VLAEnvConfig
    folder.mkdir(parents=True,exist_ok=False)
    env = strict_mod.make_continuing_env(OrchardVLAEnv)(VLAEnvConfig(max_control_steps=2100))
    expert = StudentNativeExpert(config); strict = strict_mod.StrictPlacement()
    observations=[]; commands=[]; rows=[]; images=[[],[]]; started=time.monotonic(); failure=None
    def record(obs):
        observations.append(observation(obs))
        for j,key in enumerate(('rgb_static','rgb_wrist')): images[j].append(obs[key])
    try:
        obs,info=env.reset(seed=seed); initial=truth(env); record(obs)
        write(folder/'initial.json',dict(seed=seed,truth=initial,stance=info['reset_stance_debug'],config=asdict(config),env=asdict(env.config)))
        with (folder/'steps.jsonl').open('w') as stream:
            while len(commands)<env.config.max_control_steps:
                t=truth(env); chain=strict.summary()['fruit_chains'].get(str(t['target_id']),{})
                block=expert.plan(obs,t,chain)
                if not block: break
                for target in block:
                    i=len(commands); command=native_command(target['position'],target['rotation'],target['width'],obs['tcp_pos_world'],Rotation.from_quat(obs['tcp_quat_world']).as_matrix())
                    obs,_,_,truncated,info=env.step(**command)
                    record(obs); post=truth(env)
                    strict.update(i+1,post['held'],post['detached'],post['in_bucket'],info['success'])
                    failure=healthy(post,initial)
                    pe=float(np.linalg.norm(target['position']-obs['tcp_pos_world']))
                    re=float((Rotation.from_matrix(target['rotation'])*Rotation.from_quat(obs['tcp_quat_world']).inv()).magnitude())
                    commands.append(dict(**target,native=command))
                    row=dict(before_index=i,after_index=i+1,phase=target['phase'],truth=post,controller=info,tracking_position=pe,tracking_rotation=re,failure=failure)
                    rows.append(row); stream.write(json.dumps(dict(**row,command=commands[-1],observation_before=observations[-2],observation_after=observations[-1]),default=serial,allow_nan=False)+'\n'); stream.flush()
                    if (i+1)%150==0: print(json.dumps(dict(seed=seed,step=i+1,phase=expert.phase,held=post['held'],detached=post['target_detached'],tcp=obs['tcp_pos_world'],fruit=post['fruit_position'],ik=info['ik_failed']),default=serial),flush=True)
                    if failure or truncated: break
                if failure or truncated: break
        success=any(e['apple_id']==initial['target_id'] for e in strict.success_events) and failure is None
        # Dataset usability is separate from the unchanged strict event predicate.
        from itertools import groupby
        longest={key:max([sum(1 for _ in group) for flag,group in groupby(r['controller'][key] for r in rows) if flag] or [0]) for key in ('ik_failed','action_clipped')}
        quality=dict(ik_rate=float(np.mean([r['controller']['ik_failed'] for r in rows])),clipping_rate=float(np.mean([r['controller']['action_clipped'] for r in rows])),longest_runs=longest)
        quality['passed']=quality['ik_rate']<=.05 and quality['clipping_rate']<=.05 and max(longest.values())<=15
        if success and not quality['passed']:failure='controller_quality';success=False
        reason='strict_success' if success else failure or expert.failure or 'episode_budget'
        if expert.phase not in ('DONE','FAILED'): expert.transition('DONE' if success else 'FAILED',len(commands),reason)
        for j,name in enumerate(('static','wrist')): encode(folder/(name+'.mp4'),images[j])
        trajectory=dict(contract=CONTRACT,seed=seed,num_commands=len(commands),execution_horizon=config.horizon,control_hz=30,prediction_horizon=30,
                        observations=observations,commands=commands,diagnostics=rows,strict=strict.summary(),rgb=dict(rgb_static='static.mp4',rgb_wrist='wrist.mp4'))
        write(folder/'trajectory.json',trajectory)
        integrity=validate(folder)
        result=dict(seed=seed,accepted=bool(success and integrity['passed']),reason=reason,strict=strict.summary(),phases=expert.events,steps=len(commands),sim_seconds=len(commands)/30,wall_seconds=time.monotonic()-started,
                    controller_quality=quality,ik_failure_rate=float(np.mean([r['controller']['ik_failed'] for r in rows])),clipping_rate=float(np.mean([r['controller']['action_clipped'] for r in rows])),
                    tracking_position_p95=float(np.percentile([r['tracking_position'] for r in rows],95)),tracking_rotation_p95=float(np.percentile([r['tracking_rotation'] for r in rows],95)),windows=integrity['windows'])
        result['phase_diagnostics']={phase:dict(steps=len(rr),ik_failures=sum(r['controller']['ik_failed'] for r in rr),clipped_steps=sum(r['controller']['action_clipped'] for r in rr),tracking_position_p95=float(np.percentile([r['tracking_position'] for r in rr],95))) for phase in {r['phase'] for r in rows} if (rr:=[r for r in rows if r['phase']==phase])}
        result['release_diagnostics']=[]
        for event in strict.release_events:
            rt=rows[event['step']-1]['truth'];delta=np.asarray(rt['fruit_position'])-rt['bucket_position']
            result['release_diagnostics'].append(dict(**event,fruit_bucket_distance=float(np.linalg.norm(delta)),offset_world=delta.tolist(),fruit_speed=rt['fruit_speed']))
        write(folder/'result.json',result); print('RESULT '+json.dumps(result),flush=True); return result
    except Exception as exc:
        write(folder/'error.json',dict(error=str(exc),traceback=traceback.format_exc())); raise
    finally: env.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seeds',type=int,nargs='+'); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--horizon',type=int,default=5); parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    if args.validate: print(validate(args.output)); return
    args.output.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds: run(seed,args.output/f'seed_{seed}',ExpertConfig(horizon=args.horizon))
if __name__=='__main__': main()
