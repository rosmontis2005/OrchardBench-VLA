"""Orchard Cartesian v1 contract, shared by training and deployment (no Torch).

State: world xyz metres [0:3], extrinsic xyz Euler radians [3:6], total
finger opening metres [6], all seven FR3 joint radians [7:14], zero [14:32].
Action at chunk anchor t, horizon k: R_t.T (p_{t+k+1}-p_t) metres [0:3],
Log(R_t.T R_{t+k+1}) rotation vector radians [3:6], absolute total opening
metres [6]. [7:32] is zero and loss-masked. This is NOT CALVIN scaling,
Euler subtraction, sequential integration, or the collector's joint objective.
The final stored target repeats the terminal measured state (collector convention).
"""
import numpy as np
from scipy.spatial.transform import Rotation

CONTRACT = 'orchard_cartesian_local_rotvec_width_v1'
HORIZON = 30
EPS = 1e-6


def action_mask():
    mask = np.zeros((HORIZON, 32), dtype=np.int32)
    mask[:, :7] = 1
    return mask


def state_vector(position, rotation, width, joints):
    out = np.zeros((1, 32), dtype=np.float32)
    out[0, :3] = np.asarray(position).reshape(3)
    out[0, 3:6] = Rotation.from_matrix(np.asarray(rotation).reshape(3, 3)).as_euler('xyz')
    out[0, 6] = float(np.asarray(width).item())
    out[0, 7:14] = np.asarray(joints).reshape(7)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite Orchard state')
    return out


def encode_window(traj, frame):
    if not 0 <= frame <= traj['num_frames'] - HORIZON:
        raise IndexError('Only complete 30-step windows are supported')
    p, a = traj['proprios'], traj['actions']
    r = np.asarray(p['ee_rotm'][frame], dtype=np.float64).reshape(3, 3)
    sl = slice(frame, frame + HORIZON)
    out = np.zeros((HORIZON, 32), dtype=np.float32)
    out[:, :3] = (np.asarray(a['ee_pos'][sl]) - p['ee_pos'][frame]) @ r
    out[:, 3:6] = Rotation.from_matrix(r.T @ np.asarray(a['ee_rotm'][sl]).reshape(-1, 3, 3)).as_rotvec()
    out[:, 6] = np.asarray(a['gripper_pos'][sl]).reshape(HORIZON)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite Orchard action')
    return out


def decode_targets(action, anchor_position, anchor_rotation):
    """Denormalized chunk -> absolute world targets; anchor stays fixed per chunk."""
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (HORIZON, 32) or not np.isfinite(a).all():
        raise ValueError('Expected finite denormalized [30,32] chunk')
    r = np.asarray(anchor_rotation).reshape(3, 3)
    return (np.asarray(anchor_position) + a[:, :3] @ r.T,
            r @ Rotation.from_rotvec(a[:, 3:6]).as_matrix(), a[:, 6].copy())


def native_command(position, rotation, width, current_position, current_rotation):
    """Return env.step kwargs. Width remains continuous; controller applies bounds.

    Native orientation is extrinsic xyz Euler of R_target R_current.T, not
    a rotation vector and not subtraction of two absolute Euler vectors.
    """
    delta = Rotation.from_matrix(np.asarray(rotation) @ np.asarray(current_rotation).T).as_euler('xyz')
    return dict(action=np.r_[np.asarray(position) - current_position, delta, 0.],
                gripper_width=float(width))


def prepare_rgb(image):
    """Same deterministic 95% center crop + factor-32 size for train/inference."""
    from PIL import Image
    w, h = image.size
    cw, ch = int(w * .95), int(h * .95)
    x, y = (w - cw) // 2, (h - ch) // 2
    return image.crop((x, y, x + cw, y + ch)).resize(
        (max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)), Image.Resampling.BILINEAR)


class OrchardActionAdapter:
    def set_chunk(self, normalized, mean, std, obs):
        action = np.asarray(normalized) * (np.asarray(std) + EPS) + np.asarray(mean)
        self.set_denormalized_chunk(action, obs)

    def set_denormalized_chunk(self, action, obs):
        """Load physical contract values, using the supplied chunk anchor pose."""
        self.targets = decode_targets(action, obs['tcp_pos_world'],
                                      Rotation.from_quat(obs['tcp_quat_world']).as_matrix())

    def to_native(self, index, obs):
        p, r, w = self.targets
        return native_command(p[index], r[index], w[index], obs['tcp_pos_world'],
                              Rotation.from_quat(obs['tcp_quat_world']).as_matrix())

    @staticmethod
    def state(obs):
        return state_vector(obs['tcp_pos_world'], Rotation.from_quat(obs['tcp_quat_world']).as_matrix(),
                            obs['gripper_width'], obs['joint_pos'][:7])
