"""Two bounded oracle checks; no model and no new controller."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from treesim.vla_env import OrchardVLAEnv,VLAEnvConfig
from treesim.orchard_action import encode_window,decode_targets,native_command,HORIZON
from run_arm_retime_pilot import write


def replay(path,mode,output):
    run=json.loads(path.read_text());traj=run['trajectory'];n=traj['num_frames']
    targets={}
    # Every horizon has its recorded chunk anchor, never sequentially integrated.
    for start in list(range(0,n-HORIZON+1,HORIZON))+[n-HORIZON]:
        a=encode_window(traj,start);p=traj['proprios']
        xyz,rot,width=decode_targets(a,p['ee_pos'][start],np.array(p['ee_rotm'][start]).reshape(3,3))
        for k in range(HORIZON):targets[start+k]=(xyz[k],rot[k],width[k])
    assert len(targets)==n
    env=OrchardVLAEnv(VLAEnvConfig());rows=[];index=0
    try:
        obs,initial=env.reset(seed=run['seed'])
        assert initial['detach_force_scale']==1.5 and env.sim.sim_time==0
        # Also inspect actual fruit thresholds, not only configuration metadata.
        assert env.sim.apples.detach_force_multiplier==1.5
        assert np.allclose(env.sim.apples.detach_force,np.asarray(env.tm.apple_data['detach_force'])*1.5)
        for step in range(env.config.max_control_steps):
            pos,rot,width=targets[index]
            cmd=native_command(pos,rot,width,obs['tcp_pos_world'],Rotation.from_quat(obs['tcp_quat_world']).as_matrix())
            obs,reward,done,truncated,info=env.step(**cmd)
            pe=float(np.linalg.norm(pos-obs['tcp_pos_world']))
            re=float((Rotation.from_matrix(rot)*Rotation.from_quat(obs['tcp_quat_world']).inv()).magnitude())
            rows.append(dict(step=step,target_index=index,sim_time=obs['sim_time'],position_error_m=pe,rotation_error_rad=re,
                requested_action=cmd['action'].tolist(),width=float(width),measured_width=obs['gripper_width'],
                gripper_intent=env._gripper,info=info))
            if done or truncated:break
            if mode=='time' or (pe<=.01 and re<=.08):index=min(index+1,n-1)
        result=dict(seed=run['seed'],profile=run['profile'],mode=mode,success=bool(info['success']),
            detached=bool(info['apple_detached_count']),max_detached=max(r['info']['apple_detached_count'] for r in rows),
            grasped=any(r['info']['held_apple_id_debug'] is not None for r in rows),
            timeout=bool(truncated),duration_s=obs['sim_time'],target_index=index,total_targets=n,
            ik_failed_steps=sum(r['info']['ik_failed'] for r in rows),clipped_steps=sum(r['info']['action_clipped'] for r in rows),
            force_scale=initial['detach_force_scale'],initial_info=initial,steps=rows)
        write(output,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['steps','initial_info']}),flush=True)
    finally:env.close()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--mode',choices=['time','tracking'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();replay(a.run,a.mode,a.output)
