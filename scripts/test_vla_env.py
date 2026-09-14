#!/usr/bin/env python
"""Actual GPU closed-loop contract tests. Run with pixi run python scripts/test_vla_env.py."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from treesim.vla_env import OrchardVLAEnv,VLAEnvConfig
from treesim.xr0_adapter import CalvinActionAdapter,calvin_state


def valid(obs):
    allowed={'rgb_static','rgb_wrist','proprio','tcp_pos_world','tcp_quat_world','tcp_euler_world',
             'gripper_width','joint_pos','joint_vel','base_pos','base_yaw','sim_time'}
    assert set(obs)==allowed, 'Unexpected observation key (possible truth leakage)'
    for k,v in obs.items():
        if isinstance(v,np.ndarray):assert np.isfinite(v).all(),k
    for key in ('rgb_static','rgb_wrist'):
        a=obs[key];assert a.shape==(144,192,3) and a.dtype==np.uint8 and a.std()>1,key
    assert obs['proprio'].shape==(30,) and obs['joint_pos'].shape==(9,)
    assert abs(np.linalg.norm(obs['tcp_quat_world'])-1)<1e-5


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='output/vla_env_tests.json');args=ap.parse_args()
    result={};env=OrchardVLAEnv(VLAEnvConfig(max_control_steps=300))
    try:
        first,info=env.reset(seed=42);valid(first)
        tree=env.sim.state_0.body_q.numpy().copy()
        first_sim=env.sim
        for _ in range(3):env.step([0,0,0,0,0,0,-1])
        second,info=env.reset(seed=42);valid(second)
        assert env.sim is not first_sim
        del first_sim
        for k in first:
            if isinstance(first[k],np.ndarray):assert np.allclose(first[k],second[k],rtol=0,atol=1e-6),k
        assert np.allclose(tree,env.sim.state_0.body_q.numpy(),rtol=0,atol=1e-6)
        assert info['apple_detached_count']==info['branch_break_count']==0 and second['sim_time']==0
        assert np.allclose(second['joint_pos'],env.tm.robot_data['arm_home'],atol=1e-6)
        other,_=env.reset(seed=43);valid(other)
        assert not np.array_equal(first['rgb_static'],other['rgb_static'])
        result['reset']='PASS: hard rebuild; same seed identical state/RGB; different seed different tree'
        obs,_=env.reset(seed=42)
        start=obs['tcp_pos_world'].copy()
        # Prove pure contact mode never calls AppleField.hold, even when closing.
        def forbidden_hold(*args):raise AssertionError('contact mode called hold')
        env.sim.apples.hold=forbidden_hold
        for _ in range(20):
            obs,_,done,trunc,i=env.step([0,0,0,0,0,0,1]);valid(obs);assert not i['ik_failed'] and not done and not trunc
        drift=float(np.linalg.norm(obs['tcp_pos_world']-start))
        assert drift<.025,drift
        result['zero_action']={'steps':20,'tcp_drift_m':drift,'sim_time':obs['sim_time']}
        movements=[]
        for label,axis,sign in [('+x',0,1),('-x',0,-1),('+z',2,1),('-z',2,-1)]:
            # Small repeated measured-relative commands, then let accepted target settle.
            before=obs['tcp_pos_world'].copy()
            for _ in range(12):
                a=np.zeros(7);a[axis]=sign*.004;a[6]=1
                obs,_,_,_,i=env.step(a);valid(obs);assert not i['ik_failed'],i
            for _ in range(8):obs,_,_,_,i=env.step([0,0,0,0,0,0,1])
            delta=obs['tcp_pos_world']-before
            print(label,delta,flush=True);assert delta[axis]*sign>.002,(label,delta)
            movements.append({'axis':label,'delta_m':delta.tolist()})
        result['cartesian_motion']=movements
        rotations=[]
        for axis in range(3):
            before=Rotation.from_quat(obs['tcp_quat_world'])
            a=np.zeros(7);a[3+axis]=.04;a[6]=1
            obs,_,_,_,i=env.step(a);assert not i['ik_failed'],i
            for _ in range(12):obs,_,_,_,i=env.step([0,0,0,0,0,0,1]);valid(obs)
            delta=(Rotation.from_quat(obs['tcp_quat_world'])*before.inv()).as_rotvec()
            print('rotation',axis,delta,flush=True);assert delta[axis]>.005,(axis,delta)
            rotations.append({'world_axis':axis,'actual_rotvec':delta.tolist(),'ik_error':i['ik_error']})
        result['orientation']=rotations
        widths=[]
        for grip in [1,-1,1]:
            for _ in range(15):obs,_,_,_,i=env.step([0,0,0,0,0,0,grip]);valid(obs)
            widths.append(obs['gripper_width'])
        assert widths[0]>.06 and widths[1]<.025 and widths[2]>.06,widths
        result['gripper_open_close_open']=widths
        for value in (np.nan,np.inf):
            time=obs['sim_time'];q=obs['joint_pos'].copy()
            try:env.step([value,0,0,0,0,0,1]);raise AssertionError('Nonfinite action accepted')
            except ValueError:pass
            assert env.get_obs()['sim_time']==time and np.array_equal(q,env.get_obs()['joint_pos'])
        obs,_,_,_,i=env.step([100,0,0,100,0,0,1]);valid(obs);assert i['action_clipped']
        result['invalid_action']={'NaN_Inf':'rejected without mutation','large_action':i}
        # Exercise rejection using a real unreachable IK solve, then injected
        # nonfinite solver output. Neither may reach joint commands/physics.
        original_solve=env.ik.solve
        unreachable=original_solve([10,10,10],[0,0,0,1],obs['joint_pos'])
        assert unreachable[1]>env.config.ik_position_tolerance
        rejected=[]
        try:
            for outcome in (unreachable,(np.full(9,np.nan),np.nan,np.nan)):
                env.ik.solve=lambda *args,outcome=outcome: outcome
                before=env._joint_command[:7].copy()
                obs,_,_,_,i=env.step([.001,0,0,0,0,0,1]);valid(obs)
                assert i['ik_failed'] and np.array_equal(before,env._joint_command[:7])
                rejected.append('safely held previous arm target')
        finally:
            env.ik.solve=original_solve
        result['invalid_ik']=dict(unreachable_position_error_m=unreachable[1],
                                  rejection_tests=rejected,nonfinite_output_injected=True)
        for a,expected in [([1,0,0,0,0,0,1],[.02,0,0,0,0,0,1]),
                           ([0,0,0,1,0,0,-1],[0,0,0,.05,0,0,-1]),
                           ([-1,-1,-1,-1,-1,-1,1],[-.02,-.02,-.02,-.05,-.05,-.05,1])]:
            d=CalvinActionAdapter.physical_delta(a);print('CALVIN',a,'=>',d,flush=True);assert np.allclose(d,expected)
        state=calvin_state(obs);assert state.shape==(32,) and np.count_nonzero(state[7:])==0
        adapter=CalvinActionAdapter();adapter.reset(obs)
        raw=np.array([.1,-.1,0,.2,.3,.4,1]);d=adapter.to_native(raw,obs)
        composed=Rotation.from_euler('xyz',d[3:6])*Rotation.from_quat(obs['tcp_quat_world'])
        desired=Rotation.from_euler('xyz',obs['tcp_euler_world']+raw[3:6]/20)
        assert (composed.inv()*desired).magnitude()<1e-6
        result['calvin_adapter']='PASS scaling, signs, nonidentity Euler-to-SO3 conversion, 32D state'
        env.close()
        # Exercise real Newton contact API and reject assist in empty gripper.
        env=OrchardVLAEnv(VLAEnvConfig(grasp_mode='benchmark_assist',max_control_steps=3))
        obs,_=env.reset(seed=42)
        for _ in range(3):
            obs,r,term,trunc,info=env.step([0,0,0,0,0,0,-1]);assert not info['grasp_assist_triggered']
        assert trunc and not term and r==0
        result['episode']='PASS truncation at configured limit'
        result['benchmark_assist']='PASS empty-contact gate; real apple attachment not established by this motion-only test'
        result['pure_contact']='PASS open/close with hold forbidden; stable fruit transport untested'
        result['status']='PASS'
    finally:
        env.close()
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
