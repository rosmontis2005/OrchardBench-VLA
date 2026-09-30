#!/usr/bin/env python
"""Three controller calls validating the optional width API; no policy evaluation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from treesim.vla_env import OrchardVLAEnv


def main(output):
    env=OrchardVLAEnv()
    report={'status':'RUNNING','steps':[]}
    try:
        obs,_=env.reset(seed=42)
        for width, expected in ((.06,.03),(.09,.04)):
            obs,_,_,_,info=env.step(np.zeros(7),gripper_width=width)
            assert np.allclose(info['executed_joint_target'][-2:],expected,atol=1e-7)
            assert np.isfinite(obs['proprio']).all()
            report['steps'].append(dict(request_width=width,finger_targets=info['executed_joint_target'][-2:],clipped=info['action_clipped']))
        obs,_,_,_,info=env.step(np.r_[np.zeros(6),-1.])
        assert np.allclose(info['executed_joint_target'][-2:],0.,atol=1e-7)
        before=env.control_steps
        try:
            env.step(np.zeros(7),gripper_width=float('nan'))
        except ValueError:
            pass
        else:
            raise AssertionError('NaN width accepted')
        assert env.control_steps==before
        report.update(status='PASS',legacy_signed_close=True,nan_rejected_before_control=True,control_steps=before)
    finally:
        env.close()
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    main(p.parse_args().output)
