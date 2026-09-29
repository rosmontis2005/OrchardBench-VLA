#!/usr/bin/env python
"""Three small GPU smoke tests for the fixed-base VLA evaluation contract.

B/C use controlled detached-fruit fixtures, real Newton collisions and physics;
no expert actions, policy inference, or dataset writes.
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from collect_autopicker_dataset import episode_config
from treesim import robot
from treesim.fixed_base_picker import plan_fixed_base_stance
from treesim.vla_env import OrchardVLAEnv, VLAEnvConfig


SEEDS = (1000106, 1000576)
CLOSE = [0, 0, 0, 0, 0, 0, -1]
OPEN = [0, 0, 0, 0, 0, 0, 1]


def reset_smoke():
    env = OrchardVLAEnv()
    try:
        for seed in SEEDS:
            with wp.ScopedDevice('cuda:0'):
                plan = plan_fixed_base_stance(episode_config(seed))
            assert plan.feasible
            obs, info = env.reset(seed=seed)
            assert np.allclose(obs['base_pos'], [*plan.stance.base_xy, 0.], atol=1e-5)
            assert abs(obs['base_yaw'] - plan.stance.base_yaw) < 1e-5
            assert obs['sim_time'] == 0 and env.control_steps == 0
            assert info['apple_detached_count'] == info['branch_break_count'] == 0
            assert np.allclose(obs['joint_pos'], env.tm.robot_data['arm_home'], atol=1e-6)
            assert set(obs) == {'rgb_static', 'rgb_wrist', 'proprio', 'tcp_pos_world',
                               'tcp_quat_world', 'tcp_euler_world', 'gripper_width',
                               'joint_pos', 'joint_vel', 'base_pos', 'base_yaw', 'sim_time'}
            assert obs['proprio'].shape == (30,) and np.isfinite(obs['proprio']).all()
            for key in ('rgb_static', 'rgb_wrist'):
                assert obs[key].shape == (144, 192, 3)
                assert obs[key].dtype == np.uint8 and obs[key].std() > 1
            print(f'A PASS seed={seed} target_debug={plan.stance.apple_index} '
                  f'base={obs["base_pos"].tolist()} yaw={obs["base_yaw"]:.6f} t=0', flush=True)
    finally:
        env.close()


def place_detached_fixture(env, apple, position):
    """Move just one apple via its translational joint and disable its tether."""
    sim, model = env.sim, env.tm.model
    body = int(env.tm.apple_bodies[apple])
    joint = int(np.flatnonzero(model.joint_child.numpy() == body)[0])
    start = int(model.joint_q_start.numpy()[joint])
    target_q = sim.state_0.joint_q.numpy()[start:start + 3].copy()
    target_q += np.asarray(position) - sim.body_q_np()[body, :3]
    dof = int(model.joint_qd_start.numpy()[joint])
    # The captured substep graph reads both ping-pong buffers. Inject the
    # same fruit coordinates into both so graph replay keeps the fixture.
    for state in (sim.state_0, sim.state_1):
        q = state.joint_q.numpy()
        q[start:start + 3] = target_q
        state.joint_q.assign(q)
        qd = state.joint_qd.numpy()
        qd[dof:dof + 3] = 0
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    sim._bq_np_step = sim._jq_np_step = -1
    apples = sim.apples
    apples.detached[apple] = True
    apples.broken_count = int(apples.detached.sum())
    apples._flag_host[apple] = 1
    apples._flag.assign(apples._flag_host)
    model.collide(sim.state_0, sim.contacts)


def assist_smoke():
    env = OrchardVLAEnv(VLAEnvConfig(max_control_steps=10))
    try:
        env.reset(seed=SEEDS[0])
        assert env.config.grasp_mode == 'benchmark_assist'
        _, _, term, trunc, info = env.step(CLOSE)
        assert not info['grasp_assist_triggered'] and env._held is None
        assert not term and not trunc
        print('B PASS no two-finger contact + close: no attachment', flush=True)
        # Use apple 0 independently of the reset planner's selected target.
        hand = env.sim.body_q_np()[env.wrist]
        position = hand[:3] + Rotation.from_quat(hand[3:]).apply([0., 0., .10])
        place_detached_fixture(env, 0, position)
        contacts = env.sim.contacts
        n = int(contacts.rigid_contact_count.numpy()[0])
        sb = env.tm.model.shape_body.numpy()
        apple_body = int(env.tm.apple_bodies[0])
        fingers = set()
        for s0, s1 in zip(contacts.rigid_contact_shape0.numpy()[:n],
                          contacts.rigid_contact_shape1.numpy()[:n]):
            if s0 < 0 or s1 < 0:
                continue
            b0, b1 = int(sb[s0]), int(sb[s1])
            if b0 == apple_body and b1 in env.finger_bodies:
                fingers.add(b1)
            if b1 == apple_body and b0 in env.finger_bodies:
                fingers.add(b0)
        assert fingers == env.finger_bodies and len(fingers) == 2, fingers
        _, _, term, trunc, info = env.step(CLOSE)
        assert info['grasp_assist_triggered'] and info['held_apple_id_debug'] == 0
        assert env.sim.apples._held.numpy()[0] == 1 and not term and not trunc
        for _ in range(3):
            _, _, term, trunc, info = env.step(np.zeros(7), gripper_width=0.)
            assert not info['grasp_assist_triggered'] and info['held_apple_id_debug'] == 0
            assert env.sim.apples._held.numpy()[0] == 1 and not term and not trunc
        poses = env.sim.body_q_np()
        local = Rotation.from_quat(poses[env.wrist, 3:]).inv().apply(
            poses[apple_body, :3] - poses[env.wrist, :3])
        assert np.linalg.norm(local - [0., 0., .1]) < .06, local
        _, _, _, _, info = env.step(np.zeros(7), gripper_width=.08)
        assert info['held_apple_id_debug'] is None and env.sim.apples._held.numpy()[0] == 0
        print(f'B PASS real contacts={len(fingers)} -> close/attach -> hold 6 physics frames '
              f'(hand-local={local.tolist()}) -> open/release', flush=True)
    finally:
        env.close()


def placement_smoke():
    env = OrchardVLAEnv()
    try:
        env.reset(seed=SEEDS[0])
        cp, cr = env._chassis_pose()
        place_detached_fixture(env, 0, cp + cr.apply([1.2, 0., .8]))
        _, reward, term, trunc, info = env.step(OPEN)
        assert info['apple_detached_count'] > 0 and info['apple_in_bucket_count'] == 0
        assert not info['success'] and not term and not trunc and reward == 0
        print('C PASS detached outside bucket: success=False terminated=False reward=0', flush=True)
        floor = robot._CHASSIS_Z + robot._CHASSIS[2]
        place_detached_fixture(env, 0, cp + cr.apply(
            [robot._BUCKET_CENTER_X, 0., floor + robot._BUCKET_WALL_H / 2]))
        _, reward, term, trunc, info = env.step(OPEN)
        assert info['apple_in_bucket_count'] > 0 and info['success'] and term and reward == 1
        assert not trunc
        print('C PASS detached inside bucket: success=True terminated=True reward=1', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('test', choices=['A', 'B', 'C'])
    args = parser.parse_args()
    {'A': reset_smoke, 'B': assist_smoke, 'C': placement_smoke}[args.test]()
