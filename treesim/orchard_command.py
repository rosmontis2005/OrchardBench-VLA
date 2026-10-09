"""V2 command-supervised data: N commands, N+1 measured observations.

Only real complete windows, anchored on teacher planning boundaries. No terminal
padding. Legacy V1 normalization is incompatible. Teacher metadata is never
returned by policy_sample(). State coordinates remain measured world coordinates.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .orchard_action import decode_targets, action_mask, state_vector, prepare_rgb

CONTRACT = 'orchard_cartesian_local_rotvec_width_requested_v2'


def window_starts(traj):
    return range(0, traj['num_commands']-30+1, traj['execution_horizon'])


def encode_window(traj, frame):
    if traj['contract'] != CONTRACT: raise ValueError('Expected requested-command V2')
    if frame not in window_starts(traj): raise IndexError('Expected complete boundary-aligned window')
    p = traj['observations'][frame]; cmds = traj['commands'][frame:frame+30]
    r = Rotation.from_quat(p['tcp_quat_world']).as_matrix()
    out = np.zeros((30,32), np.float32)
    out[:,:3] = (np.asarray([c['position'] for c in cmds])-p['tcp_pos_world'])@r
    out[:,3:6] = Rotation.from_matrix(r.T@np.asarray([c['rotation'] for c in cmds])).as_rotvec()
    out[:,6] = [c['width'] for c in cmds]
    if not np.isfinite(out).all(): raise ValueError('Nonfinite command')
    return out


def policy_sample(traj, frame, rgb_static, rgb_wrist):
    """Explicit whitelist; RGB arguments must be current frame, never truth."""
    o = traj['observations'][frame]
    return dict(rgb_static=prepare_rgb(rgb_static), rgb_wrist=prepare_rgb(rgb_wrist),
                instruction='Pick an apple and place it in the bucket.',
                state=state_vector(o['tcp_pos_world'], Rotation.from_quat(o['tcp_quat_world']).as_matrix(),
                                   o['gripper_width'], o['joint_pos'][:7]),
                action=encode_window(traj,frame), action_mask=action_mask())


def load_policy_sample(folder, frame):
    """Read precisely the current video frames; return only the policy whitelist.

    This CPU reference loader intentionally does not import an XR-0 model or
    consume any existing normalization statistics.
    """
    import json
    from pathlib import Path
    import subprocess
    from PIL import Image
    folder = Path(folder)
    traj = json.loads((folder/'trajectory.json').read_text())
    if frame not in window_starts(traj): raise IndexError('Incomplete or non-boundary window')
    images = []
    for key in ('rgb_static', 'rgb_wrist'):
        output = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(folder/traj['rgb'][key]),
                                 '-vf', f'select=eq(n\\,{frame})', '-frames:v', '1',
                                 '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1'],
                                capture_output=True, check=True)
        if len(output.stdout) != 192*144*3: raise ValueError('Missing current RGB frame')
        images.append(Image.frombytes('RGB', (192,144), output.stdout))
    return policy_sample(traj, frame, *images)
