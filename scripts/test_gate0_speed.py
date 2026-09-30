"""Targeted width replay regression using existing successful measured v1 data."""
import sys,json,itertools
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from treesim.vla_env import OrchardVLAEnv,VLAEnvConfig
from run_arm_retime_pilot import write,ROOT


def main():
    selection=json.loads((ROOT/'artifacts/arm_retime_pilot/pilot_seed_selection.json').read_text())
    rows=[]
    for row in selection['all_candidates']:
        d=json.loads(Path(row['canonical_path']).read_text())
        w=np.array(d['proprios']['gripper_pos']).ravel();t=np.array(d['orchardbench']['timestamps'])
        b={r['state']:r['frame']/60 for r in d['orchardbench']['fixed_base_expert']['state_trace']}
        phase_stats={}
        for phase,nxt in [('GRASP','PULL'),('PULL','TRANSPORT'),('TRANSPORT','DROP')]:
            x=w[(t>=b[phase]-1e-9)&(t<b[nxt]-1e-9)]
            phase_stats[phase]=dict(samples=len(x),min_m=float(x.min()),max_m=float(x.max()),max_positive_step_m=float(max(np.diff(x),default=0)),positive_steps_gt_1um=int(sum(np.diff(x)>1e-6)))
        hold=w[(t>=b['PULL']-1e-9)&(t<b['DROP']-1e-9)]
        env=OrchardVLAEnv();released=[]
        env.sim=SimpleNamespace(apples=SimpleNamespace(release=released.append))
        env._held=0;env._gripper=-1.;env._gripper_width_command=.08;env._width_open_steps=0
        for width in hold:
            width=float(np.clip(width,0,.08));env._update_width_intent(width);env._gripper_width_command=width
            env._update_assist()
        assert not released,(row['seed'],'false release')
        for width in np.linspace(env._gripper_width_command,.08,6).tolist()+[.08]*5:
            env._update_width_intent(width);env._gripper_width_command=width
            if env._held is not None:env._update_assist()
        assert released==[0],(row['seed'],'opening failed')
        positive=np.diff(hold)
        rows.append(dict(seed=row['seed'],canonical_path=row['canonical_path'],phase_stats=phase_stats,hold_samples=len(hold),positive_steps_gt_1um=int(sum(positive>1e-6)),
                         max_positive_step_m=float(max(positive,default=0)),
                         max_open_streak=max([len(list(g)) for k,g in itertools.groupby(hold>=.075) if k] or [0]),
                         hold_preserved=True,explicit_open_released=True))
        env.sim=None
    assert VLAEnvConfig().detach_force_scale==1.5
    assert VLAEnvConfig().max_control_steps/VLAEnvConfig().control_hz==20
    try:VLAEnvConfig(detach_force_scale=0)
    except ValueError:pass
    else:raise AssertionError('invalid detach scale accepted')
    write(ROOT/'log/gate0_speed_30seed/gripper_audit.json',dict(status='PASS',episodes=rows,open_width_m=.075,open_steps=5,
          rationale='v1 physical hold rebound reaches full width transiently; longest near-open streak is four 30Hz samples. Five sustained samples release. Discrete OPEN remains immediate.'))
    print(f'PASS: {len(rows)} real hold sequences preserved; gradual explicit opening releases; force/budget defaults checked')
if __name__=='__main__':main()
