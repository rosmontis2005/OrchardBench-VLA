#!/usr/bin/env python
"""Native renderer sanity: full occlusion, visible rim with center blocked, FOV.
No modifications to orchard policy cameras or RGB. Debug artifacts only.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import warp as wp
import newton
from treesim.vla_camera import StaticRGBCamera
from treesim.target_visibility import TargetVisibility


def run(output):
    rows = []
    for case in ('clear', 'fully_occluded', 'center_blocked_rim_visible', 'out_of_fov'):
        with wp.ScopedDevice('cuda:0'):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform((0,0,0),wp.quat_identity()))
            builder.add_shape_sphere(body, radius=.20, color=(1.,0.,0.))
            if case in ('fully_occluded','center_blocked_rim_visible'):
                width = .4 if case=='fully_occluded' else .02
                builder.add_shape_box(-1,xform=wp.transform((0,-1,0),wp.quat_identity()),
                    hx=width,hy=.05,hz=.4,color=(0.,1.,0.))
            model = builder.finalize('cuda:0')
            state = model.state()
            target = (5,-2,0) if case=='out_of_fov' else (0,0,0)
            cam = StaticRGBCamera(model,position=(0,-2,0),target=target)
            # Same static sensor in both slots: adapter only supplies the wrist
            # force keyword so the production checker itself is exercised.
            class Adapter:
                def __getattr__(self, key): return getattr(cam,key)
                def update(self,state,force=False,**kw): return cam.update(state,**kw)
            checker = TargetVisibility(model,body,cam,Adapter())
            row = checker.check(state,'initial',sanity=True)
            row['case'] = case
            visible = case in ('clear','center_blocked_rim_visible')
            assert row['any_policy_view_visible'] == visible, row
            if case=='center_blocked_rim_visible':
                ids = checker.buffers['static'].numpy()[0,0]
                assert not np.isin(ids[71:73,95:97],checker.shapes).any()
            checker.save_gallery(output/case)
            rows.append(row)
    assert 0 < rows[2]['static_visible_pixels'] < rows[0]['static_visible_pixels']
    (output/'sanity.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2))
    print(json.dumps(rows,indent=2))

def test_collector_gate(output):
    """Controlled negative injection tests gate wiring, not visibility truth.

    Renderer truth is tested above. Force its initial result to invisible for
    an otherwise successful physical pick; no dataset/video is published.
    """
    from unittest.mock import patch
    from collect_autopicker_dataset import collect_episode
    original = TargetVisibility.check

    def invisible_initial(self, state, moment, **kwargs):
        row = original(self, state, moment, **kwargs)
        if moment == 'initial':
            row.update(static_visible_pixels=0, wrist_visible_pixels=0,
                       static_visible=False, wrist_visible=False,
                       any_policy_view_visible=False)
        return row

    with patch.object(TargetVisibility, 'check', invisible_initial):
        result, _, _ = collect_episode(920000, output, 1.5, record_rgb=False)
    assert all(result[k] for k in ('grasped', 'detached', 'placed'))
    assert not result['accepted']
    assert result['reject_reason'] == 'target_not_visible_in_policy_observation'
    (output/'collector_gate_injection.json').write_text(json.dumps(dict(
        test='CONTROLLED VISIBILITY INJECTION; NOT REAL VISIBILITY AUDIT DATA',
        status='PASS', result=result), indent=2))
    print('COLLECTOR VISIBILITY GATE INTEGRATION PASS')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--collector-gate', action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);run(args.output)
    if args.collector_gate: test_collector_gate(args.output)
