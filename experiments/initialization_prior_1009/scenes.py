"""Original planner first; immutable lateral stance override before world build."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from collect_autopicker_dataset import episode_config
from treesim import fixed_base_picker as fb, robot


def config(seed):
    c = episode_config(seed)
    c.fruit.detach_force_scale = 1.5
    return c


def stance_for(scene, dy):
    s = fb.FixedBaseStance(**scene['stance'])
    delta = Rotation.from_euler('z', s.base_yaw).apply([0., dy, 0.])
    return replace(s, base_xy=tuple(np.asarray(s.base_xy)+delta[:2]))


def reset_override(env, scene, dy):
    stance = stance_for(scene, dy)
    # Process-local, scoped patch of the imported planner symbol. reset still
    # constructs the original world and native controller. No simulator writes.
    plan = fb.StancePlan(scene['seed'], stance, {'selected': asdict(stance)},
                        *[np.empty(0) for _ in range(5)])
    with patch('treesim.vla_env.plan_fixed_base_stance', return_value=plan):
        return env.reset(seed=scene['seed'])


def geometry_check(plan, stance, ik):
    """Same spawn proxies as planner, without rerunning target/stance search.

    Only initial overlap (<0 clearance) defines invalid geometry. Endpoint IK
    is separately diagnosed at reset; local IK failure is not proof of an
    invalid scene and never excludes a rollout.
    """
    xy,yaw = stance.base_xy,stance.base_yaw
    p = fb.StancePlannerParams()
    wood = fb._capsule_samples(plan.wood_a,plan.wood_b,p.wood_samples)
    pts = fb._to_chassis(wood.reshape(-1,3),xy,yaw)
    rr = np.repeat(plan.wood_r,p.wood_samples)
    apples = fb._to_chassis(plan.apples_world,xy,yaw)
    clear = []
    for points,radii in ((pts,rr),(apples,plan.apple_radii)):
        clear.extend([float((fb._box_signed_dist(points,*box)-radii).min())
                      for box in (fb._CHASSIS_BOX,fb._BUCKET_BOX)])
        clear.append(float((np.linalg.norm(points[:,None]-fb._WHEELS,axis=-1).min(axis=1)
                            -fb._WHEEL_R-radii).min()))
    chain = fb._arm_chain_points(ik,np.array(list(robot._ARM_HOME.values())))
    world = fb._to_world(chain,xy,yaw)
    thick = plan.wood_r >= p.thick_wood_radius
    arm=[]; twigs=0
    for i,(a,b) in enumerate(zip(world[:-1],world[1:])):
        if np.linalg.norm(a-b)<1e-6 and i+1<len(world)-1: continue
        radius=p.hand_radius if i==len(world)-2 else p.arm_link_radius
        dw=fb._seg_seg_dist(plan.wood_a,plan.wood_b,a,b)-plan.wood_r-radius
        if thick.any(): arm.append(float(dw[thick].min()))
        twigs+=int(((dw<0)&~thick).sum())
        arm.append(float((fb._seg_seg_dist(plan.apples_world,plan.apples_world,a,b)-plan.apple_radii-radius).min()))
    return dict(initialization_valid=min(clear+arm)>=0,
                chassis_clearance=min(clear),home_arm_clearance=min(arm),home_twig_contacts=twigs)


def select(start, count, amplitude=.08):
    from treesim.picker import ArmIK
    ik=ArmIK(list(robot._ARM_HOME.values()))
    scenes=[];attempts=[]
    for seed in range(start,start+1000):
        plan=fb.plan_fixed_base_stance(config(seed),ik=ik)
        attempts.append(dict(seed=seed,feasible=plan.feasible,rejection_reasons=plan.report['rejection_reasons']))
        if plan.feasible:
            scene=dict(seed=seed,stance=asdict(plan.stance),apples_world=plan.apples_world.tolist(),
                       apple_radii=plan.apple_radii.tolist())
            scene['initializations']={name:geometry_check(plan,stance_for(scene,dy),ik)
                for name,dy in [('G0',0.),('G-',-amplitude),('G+',amplitude)]}
            scenes.append(scene)
        print('SELECT',seed,plan.feasible,len(scenes),flush=True)
        if len(scenes)==count:break
    assert len(scenes)==count
    return dict(scenes=scenes,attempts=attempts,selection='first original-planner-feasible seeds; no outcome-based selection')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['calibration','development','test','round0'],required=True)
    a=p.parse_args();out=ROOT/'artifacts/initialization_prior_1009';out.mkdir(exist_ok=True)
    path=out/(a.group+'_scenes.json');assert not path.exists()
    if a.group=='round0':
        rows=[select(s,1)['scenes'][0] for s in [1000106,1000268,1000576]]
        data=dict(scenes=rows,attempts=[],selection='three existing Gate A scenes including known failure')
    else:
        start,count={'calibration':(8290000,40),'development':(8300000,20),'test':(8400000,50)}[a.group]
        data=select(start,count)
    path.write_text(json.dumps(data,indent=2))
    if a.group=='calibration':
        xyz=np.array([fb._to_chassis(np.array(s['stance']['target_world']),s['stance']['base_xy'],s['stance']['base_yaw']) for s in data['scenes']])
        fit=np.polyfit(xyz[:,2],xyz[:,0],1)
        calibration=dict(nominal_base_xyz=xyz.mean(axis=0).tolist(),height_to_x=fit.tolist(),
            held_offset_tcp=[0.,0.,0.],seeds=[s['seed'] for s in data['scenes']],
            residual_x_rmse=float(np.sqrt(np.mean((np.polyval(fit,xyz[:,2])-xyz[:,0])**2))))
        (out/'calibration.json').write_text(json.dumps(calibration,indent=2))
if __name__=='__main__': main()
