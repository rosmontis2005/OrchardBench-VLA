#!/usr/bin/env python
"""Collect measured first-attempt fixed-base expert trajectories; never train a model.

Canonical single-arm data is NOT directly compatible with XR-0's dual-arm
JsonDataset. Targets are next measured state, not desired IK/CALVIN actions.

Expert = treesim.fixed_base_picker.FixedBaseAutoPicker: a privileged, physics-
free, seed-deterministic RESET-TIME selection of the target fruit and the fixed
chassis (x, y, yaw) (treesim.fixed_base_picker.plan_fixed_base_stance), then the
official AutoPicker manipulation backend REACH -> GRASP -> PULL -> TRANSPORT ->
DROP inside a clean t=0 episode.  The mobile SCAN/ALIGN/ORBIT/RECOVER states are
not executed and no base command is ever issued (asserted every frame); the
chassis is additionally welded to the world and its drift is audited every
physics frame.  Fruit truth / target identity / IK residuals are recorded as
debug metadata only, never as policy input.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import datetime
import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT.parent / 'log/log_fixed_base_expert.md'
DECODER_DIR = ROOT.parent / 'log/dataset_video_tools'
SCHEMA = 'orchard_xr0_single_arm_v0'
BASE_POLICY = 'privileged_reset_stance_fixed_base_expert_v1_world_weld'
EXPERT = 'FixedBaseAutoPicker(official AutoPicker REACH/GRASP/PULL/TRANSPORT/DROP backend)'
DIMS = dict(ee_pos=3, ee_rotm=9, arm_joint=7, arm_joint_vel=7, gripper_pos=1)
ACTION_KEYS = ('ee_pos', 'ee_rotm', 'arm_joint', 'gripper_pos')
PROMPT = ('The following observations are captured from multiple views.\n'
          '# Ego View\n<image>\n# Left-Wrist View\n<image>\n'
          'Generate robot actions for the task:\nPick an apple and place it in the bucket.')


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.writing')
    with tmp.open('w') as f:
        json.dump(data, f, allow_nan=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def update_log(path, section, body):
    path = Path(path)
    text = path.read_text() if path.exists() else '# OrchardBench AutoPicker dataset collection — 2026-09-14\n'
    pattern = r'(^## ' + str(section) + r'\. [^\n]+\n).*?(?=^## |\Z)'
    text, n = re.subn(pattern, lambda m: m[1] + '\n' + body.rstrip() + '\n\n', text, flags=re.M | re.S)
    if not n:
        text += f'\n## {section}. Collection result\n\n{body}\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def journal(path, message):
    with Path(path).open('a') as f:
        f.write(f'\n[{datetime.datetime.now().isoformat()}] {message}\n')
    print(message, flush=True)


def gpu_memory():
    p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return p.stdout.strip()


def decoder():
    if DECODER_DIR.is_dir():
        sys.path.insert(0, str(DECODER_DIR))
    from decord import VideoReader
    return VideoReader


def preview(traj, frame, length=30):
    """Same current/future indexing and local SE(3) semantics as _arm_action.

    Stable double-precision SO(3) log avoids acos(trace) cancellation near zero.
    Raw data stays unnormalized. Tail repeats last valid action, as _pad does;
    statistics enumerate only complete windows, matching JsonDataset._samples.
    """
    p, a = traj['proprios'], traj['actions']
    n = traj['num_frames']; steps = min(length, n-frame)
    if steps <= 0:
        raise ValueError('Invalid preview frame')
    indices = np.minimum(np.arange(frame, frame+length), n-1)
    r = np.asarray(p['ee_rotm'][frame], dtype=float).reshape(3, 3)
    target_r = np.asarray(a['ee_rotm'], dtype=float)[indices].reshape(-1, 3, 3)
    out = np.zeros((length, 32), dtype=np.float64)
    out[:, :3] = (np.asarray(a['ee_pos'])[indices]-p['ee_pos'][frame]) @ r
    out[:, 3:6] = Rotation.from_matrix(r.T @ target_r).as_rotvec()
    out[:, 6:7] = np.asarray(a['gripper_pos'])[indices]-p['gripper_pos'][frame]
    out[:, 7:14] = np.asarray(a['arm_joint'])[indices]-p['arm_joint'][frame]
    return out.astype(np.float32), steps


def round_trip(traj, length=30):
    rng = np.random.default_rng(traj['seed'])
    n = traj['num_frames']; errors = np.zeros(4)
    for t in sorted(set([0, n//2, n-1, *rng.integers(0, n, size=12).tolist()])):
        chunk, steps = preview(traj, t, length)
        indices = np.minimum(np.arange(t, t+length), n-1)
        p, a = traj['proprios'], traj['actions']
        r = np.asarray(p['ee_rotm'][t]).reshape(3, 3)
        pos = np.asarray(p['ee_pos'][t])+chunk[:, :3].astype(float) @ r.T
        rot = r @ Rotation.from_rotvec(chunk[:, 3:6].astype(float)).as_matrix()
        target_r = np.asarray(a['ee_rotm'])[indices].reshape(-1, 3, 3)
        e = [np.linalg.norm(pos-np.asarray(a['ee_pos'])[indices], axis=1).max(),
             Rotation.from_matrix(np.swapaxes(rot, 1, 2) @ target_r).magnitude().max(),
             np.abs(np.asarray(p['arm_joint'][t])+chunk[:, 7:14]-np.asarray(a['arm_joint'])[indices]).max(),
             np.abs(np.asarray(p['gripper_pos'][t])+chunk[:, 6:7]-np.asarray(a['gripper_pos'])[indices]).max()]
        errors = np.maximum(errors, e)
    assert np.all(errors < [1e-5, 1e-5, 1e-6, 1e-6]), ('DATA FORMAT AUDIT FAIL', errors)
    return dict(zip(('position_m', 'rotation_rad', 'joint_rad', 'gripper_m'), errors.tolist()))


def validate_arrays(traj):
    assert set(traj) == {'schema','trajectory_type','seed','num_frames','record_fps','instruction','observations','proprios','actions','orchardbench','episode_id','split'}
    assert traj['schema'] == SCHEMA and traj['trajectory_type'] == 'success'
    n = traj['num_frames']; assert n >= 30
    assert set(traj['proprios']) == set(DIMS)
    assert set(traj['actions']) == set(ACTION_KEYS)
    assert set(traj['observations']) == {'ego', 'wrist_left'}
    prompt = traj['instruction']['general'][0]
    assert prompt['images'] == ['observations.ego', 'observations.wrist_left']
    assert prompt['conversations'] == [{'from': 'human', 'value': PROMPT}, {'from': 'gpt', 'value': '<bot></bot>'}]
    for group in ('proprios', 'actions'):
        for key, value in traj[group].items():
            a = np.asarray(value); assert a.shape == (n, DIMS[key]) and np.isfinite(a).all(), (group, key, a.shape)
        r = np.asarray(traj[group]['ee_rotm']).reshape(-1, 3, 3)
        assert np.max(np.abs(np.swapaxes(r, 1, 2) @ r-np.eye(3))) < 1e-6
        assert np.max(np.abs(np.linalg.det(r)-1)) < 1e-6
    for key in ACTION_KEYS:
        a = np.asarray(traj['actions'][key]); p = np.asarray(traj['proprios'][key])
        assert np.array_equal(a[:-1], p[1:]) and np.array_equal(a[-1], p[-1]), key
    meta = traj['orchardbench']
    assert meta['debug_usage'] == 'NOT POLICY INPUT'
    assert meta['incidental_detach_count'] == 0
    assert meta['target_visibility']['initial']['any_policy_view_visible']
    assert meta['detach_diagnostics']['detach_frame'] is not None
    assert meta['detach_force_multiplier'] == meta['detach_diagnostics']['detach_force_multiplier']
    assert meta['first_attempt_success'] and all(meta[k] for k in ('grasped', 'detached', 'placed'))
    assert meta['attempt_count'] == 1 and meta['branch_break_count'] == 0
    assert meta['autopicker_grasp_semantics'] == 'official_autopicker_assist'
    assert meta['base_policy'] == BASE_POLICY and meta['expert'] == EXPERT
    fb = meta['fixed_base_expert']
    assert fb['debug_usage'] == 'NOT POLICY INPUT' and fb['base_commands_issued'] == 0
    assert fb['clean_start']['sim_time'] == 0 and fb['clean_start']['branch_break_count'] == 0
    assert all(s['target_apple'] == fb['selected_apple_debug_index']
               for s in fb['state_trace'] if s['state'] == 'PULL')
    visited = [s['state'] for s in fb['state_trace']]
    assert visited[0] == 'REACH' and visited[-1] == 'DONE', visited
    assert not (set(visited) & {'SCAN', 'ALIGN', 'ORBIT', 'RECOVER'}), visited
    assert [s for s in visited if s in ('REACH', 'GRASP', 'PULL', 'TRANSPORT', 'DROP')] == ['REACH', 'GRASP', 'PULL', 'TRANSPORT', 'DROP'], visited
    # policy fields never carry the privileged target: keys are the fixed contract
    assert set(traj['proprios']) == set(DIMS) and set(traj['actions']) == set(ACTION_KEYS)
    ts = np.asarray(meta['timestamps']); assert ts.shape == (n,) and ts[0] == 0 and np.allclose(np.diff(ts), 1/30, atol=1e-8)
    base = np.asarray(meta['base_pose']); assert base.shape == (n, 7) and np.isfinite(base).all()
    trans = np.linalg.norm(base[:, :3]-base[0, :3], axis=1).max()
    angles = Rotation.from_quat(base[:, 3:]).as_euler('xyz')[:, 2]
    yaw = np.max(np.abs(np.arctan2(np.sin(angles-angles[0]), np.cos(angles-angles[0]))))
    assert trans < 1e-3 and yaw < 1e-3
    assert meta['max_base_translation_drift_m'] < 1e-3 and meta['max_base_yaw_drift_rad'] < 1e-3
    assert np.allclose(traj['proprios']['arm_joint'][0], meta['home_arm_joint'], atol=1e-6)
    assert meta['timestamps'][-1] <= 20 + 1e-8
    return round_trip(traj)


def validate_videos(traj):
    VideoReader = decoder(); n = traj['num_frames']; samples = []; results = {}
    for view in ('ego', 'wrist_left'):
        info = traj['observations'][view][0]
        assert info['start'] == 0 and info['end'] == n and info['fps'] == 30
        path = Path(info['path']); assert path.is_file() and path.stat().st_size > 0
        vr = VideoReader(str(path), num_threads=2)
        assert len(vr) == n and abs(vr.get_avg_fps()-30) < .01
        count = 0
        for start in range(0, n, 48):
            frames = vr.get_batch(list(range(start, min(start+48, n)))).asnumpy()
            assert frames.dtype == np.uint8 and frames.shape[1:] == (144, 192, 3)
            count += len(frames)
        assert count == n
        sample = vr.get_batch([0, n//2, n-1]).asnumpy()
        variances = [float(a.var()) for a in sample]
        assert min(variances) > 1, (view, variances)
        results[view] = dict(decoded_frames=count, fps=float(vr.get_avg_fps()), sample_variances=variances)
        samples.append(sample)
        del vr
    assert all(not np.array_equal(a, b) for a, b in zip(*samples)), 'Identical views'
    return results


def encode(path, frames):
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '192x144', '-r', '30', '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for a in frames:
            assert a.shape == (144, 192, 3) and a.dtype == np.uint8
            p.stdin.write(a.tobytes())
        p.stdin.close()
        err = p.stderr.read().decode(); rc = p.wait()
        if rc: raise RuntimeError(f'ffmpeg rc={rc}: {err}')
    finally:
        if p.poll() is None: p.terminate(); p.wait()


def episode_config(seed):
    """Episode world configuration (robot pose is filled in by the stance planner)."""
    from treesim.config import TreeConfig
    cfg = TreeConfig.compliant('apple'); cfg.seed = int(seed); cfg.device = 'cuda:0'
    cfg.lsystem.shape_jitter = 0.; cfg.physics.dynamics_jitter = 0.
    cfg.fruit.enabled = True; cfg.fruit.max_count = 40; cfg.foliage.set_density(.6)
    cfg.breaking.enabled = True; cfg.robot.enabled = True; cfg.physics.terrain = False
    return cfg


_ARM_IK = None


def shared_arm_ik():
    """One Franka-only IK side model per process (stateless between solves)."""
    global _ARM_IK
    if _ARM_IK is None:
        import warp as wp
        from treesim import robot
        from treesim.picker import ArmIK
        with wp.ScopedDevice('cuda:0'):
            _ARM_IK = ArmIK(list(robot._ARM_HOME.values()))
    return _ARM_IK


def compact_plan(report):
    """Stance-planner report without the per-candidate table (manifest/metadata size)."""
    keep = ('planner', 'seed', 'apple_count', 'reachable_height_count', 'candidate_count',
            'feasible_count', 'rejection_reasons', 'privileged_information')
    out = {k: report[k] for k in keep}
    out['per_apple'] = [dict(apple_index=a['apple_index'], reason=a['reason'],
                             furthest_stage=a.get('furthest_stage'), candidates=a['candidates'])
                        for a in report['per_apple']]
    return out


def stance_metadata(st):
    return dict(selected_apple_debug_index=st.apple_index, selected_apple_initial_world_pose=list(st.target_world),
                selected_apple_radius=st.apple_radius, selected_base_pose_xy_yaw=[*st.base_xy, st.base_yaw],
                standoff_m=st.standoff, standoff_band_m=list(st.standoff_band), azimuth_offset_rad=st.azimuth_offset,
                pregrasp_ik_error_m=st.ik_pregrasp_err, grasp_ik_error_m=st.ik_grasp_err,
                pull_ik_errors_m=list(st.ik_pull_err), transport_ik_error_m=st.ik_transport_err,
                initial_safety=dict(min_clearance_chassis_m=st.min_clearance_chassis,
                                    min_clearance_arm_home_m=st.min_clearance_arm_home,
                                    pushable_twig_contacts=st.twig_contacts), planner_score=st.score)


def sanitize_reason(reason):
    return re.sub(r'[^a-z0-9]+', '_', str(reason).lower()).strip('_')


def collect_episode(seed, raw_dir, detach_force_scale=1.0, *, record_rgb=True,
                    initial_only=False, gallery_dir=None):
    """Privileged reset-time stance -> clean t=0 world -> fixed-base expert; record
    independent sensors at 30Hz.  No base command is ever issued (asserted)."""
    import warp as wp
    import newton.viewer
    from treesim import robot
    from treesim.fixed_base_picker import (plan_fixed_base_stance, build_fixed_base_world,
                                           verify_clean_start, FixedBaseAutoPicker)
    from treesim.metrics import Metrics
    from treesim.vla_camera import StaticRGBCamera
    from treesim.expert_diagnostics import DetachDiagnostics
    from treesim.target_visibility import TargetVisibility
    started = time.monotonic(); sim = viewer = None; traj = None; images = [[], []]
    result = dict(seed=seed, accepted=False, reject_reason=None, grasped=False, detached=False, placed=False,
                  frames=0, sim_duration_s=0., branch_break_count=0, incidental_detach_count=0)
    states = {k: [] for k in DIMS}; bases = []; times = []; state_trace = []
    max_trans = max_yaw = 0.; terminal_frame = None; first = None; frame = 0
    try:
        cfg = episode_config(seed)
        cfg.fruit.detach_force_scale = detach_force_scale
        result["detach_force_multiplier"] = detach_force_scale
        with wp.ScopedDevice('cuda:0'):
            ik = shared_arm_ik()
            t_plan = time.monotonic()
            plan = plan_fixed_base_stance(cfg, ik=ik)
            result['plan_wall_s'] = time.monotonic()-t_plan
            result['stance_plan'] = compact_plan(plan.report)
            print('stance_plan_candidates ' + json.dumps(plan.report['candidates']), flush=True)
            if not plan.feasible:
                result['reject_reason'] = 'no_feasible_fixed_base_setup'
                return result, None, images
            st = plan.stance
            result['stance'] = stance_metadata(st)
            print('selected_stance ' + json.dumps(result['stance']), flush=True)
            # clean rebuild at the selected pose: robot spawned directly at its
            # final fixed pose (world weld), same seed geometry as the plan
            tm, sim = build_fixed_base_world(cfg, st, fps=60, substeps=3)
            viewer = newton.viewer.ViewerNull(); sim.set_viewer(viewer)
            driver = robot.RobotDriver(sim, tm, cfg.robot)
            recording_wrist = robot.WristCamera(tm.model, viewer, tm, cfg.robot, perceive=False)
            recording_static = StaticRGBCamera(tm.model, width=192, height=144)
            assert recording_wrist.percept is None
            result['clean_start'] = verify_clean_start(sim, tm, plan, driver)
            met = Metrics(None)
            picker = FixedBaseAutoPicker(sim, tm, driver, cfg.robot, st, met, ik=ik)
            assert picker.state == 'REACH' and picker.cam is None
            names = [n.rsplit('/', 1)[-1] for n in tm.model.joint_label]
            starts = tm.model.joint_q_start.numpy()
            qidx = np.array([starts[names.index(n)] for n in robot._ARM_HOME], dtype=int)
            didx = np.asarray(tm.robot_data['arm_dofs'][0], dtype=int)
            tcp = next(i for i, n in enumerate(tm.model.body_label) if n.endswith('fr3_hand_tcp'))
            ch = int(tm.robot_data['chassis'][0]); initial = sim.body_q_np()[ch].copy()
            initial_yaw = Rotation.from_quat(initial[3:]).as_euler('xyz')[2]
            ds = int(driver.planar_dof[0]); planar = [ds, ds+1, ds+3]
            home = np.asarray(tm.robot_data['arm_home'])
            assert np.allclose(sim.joint_q_np()[qidx], home, atol=1e-6)
            assert sim.sim_time == 0 and sim.apples.broken_count == sim.breaker.broken_count == 0
            result['gpu_after_build_mib'] = gpu_memory()
            diagnostics = DetachDiagnostics(sim, picker, tcp)
            visibility = TargetVisibility(tm.model, int(tm.apple_bodies[st.apple_index]),
                                          recording_static, recording_wrist)
            visibility.check(sim.state_0, 'initial', sanity=initial_only)
            result['target_visibility'] = visibility.data
            result['detach_diagnostics'] = diagnostics.data
            if gallery_dir is not None:
                visibility.save_gallery(gallery_dir)
            if initial_only:
                result['episode_wall_s'] = time.monotonic()-started
                return result, None, images
            def snapshot(frame):
                body = sim.body_q_np(); q = sim.joint_q_np()[qidx]; ik_last = picker.last_ik
                return dict(frame=int(frame), sim_time=round(float(sim.sim_time), 4), state=picker.state,
                            tcp_world=[round(float(v), 4) for v in body[tcp, :3]],
                            target_world=(None if picker._target is None else [round(float(v), 4) for v in picker._target]),
                            ik_target_world=(None if ik_last is None else [round(float(v), 4) for v in ik_last[0]]),
                            ik_residual_m=(None if ik_last is None else round(ik_last[1], 5)),
                            gripper_width_m=round(float(q[-2:].sum()), 5), target_apple=int(picker._target_apple),
                            detached=bool(picker._target_apple >= 0 and sim.apples.detached[picker._target_apple]),
                            apples_detached_total=int(sim.apples.broken_count),
                            branch_break_count=int(sim.breaker.broken_count), fail_reason=picker.fail_reason)
            def record():
                body = sim.body_q_np(); pose = body[tcp]; q = sim.joint_q_np()[qidx]
                qd = sim.state_0.joint_qd.numpy()[didx]
                vals = dict(ee_pos=pose[:3], ee_rotm=Rotation.from_quat(pose[3:]).as_matrix().ravel(),
                            arm_joint=q[:7], arm_joint_vel=qd[:7], gripper_pos=[float(q[-2:].sum())])
                for key, value in vals.items():
                    a = np.asarray(value, dtype=float); assert np.isfinite(a).all(); states[key].append(a.tolist())
                times.append(float(sim.sim_time)); bases.append(body[ch].astype(float).tolist())
                if not record_rgb: return
                recording_wrist.update(sim.state_0, force=True)
                images[0].append(recording_static.update(sim.state_0).copy())
                images[1].append(recording_wrist.last_rgb.copy())
                for a in (images[0][-1], images[1][-1]):
                    assert a.dtype == np.uint8 and a.shape == (144, 192, 3) and a.var() > 1
            record(); state_trace.append(snapshot(0)); print('trace ' + json.dumps(state_trace[-1]), flush=True)
            sim_start = time.monotonic(); last_state = picker.state
            for frame in range(1, 1201):
                sim.step(); met.frame()
                diagnostics.after_physics(frame)
                if terminal_frame is None:
                    picker.update()
                    diagnostics.after_command(frame)
                    if picker.state == 'GRASP' and visibility.data['grasp_entry'] is None:
                        visibility.check(sim.state_0, 'grasp_entry')
                # Fixed base: the expert has no locomotion action, so the planar
                # velocity targets must still be exactly zero (never masked).
                assert np.all(driver._target_host[planar] == 0.), 'expert issued a base command'
                assert picker.state not in FixedBaseAutoPicker.MOBILE_STATES
                b = sim.body_q_np()[ch]
                trans = float(np.linalg.norm(b[:3]-initial[:3]))
                y = Rotation.from_quat(b[3:]).as_euler('xyz')[2]-initial_yaw
                yaw = float(abs(np.arctan2(np.sin(y), np.cos(y))))
                max_trans = max(max_trans, trans); max_yaw = max(max_yaw, yaw)
                if picker.state != last_state:
                    state_trace.append(snapshot(frame)); print('trace ' + json.dumps(state_trace[-1]), flush=True)
                    last_state = picker.state
                if frame % 2 == 0: record()
                if max_trans >= 1e-3 or max_yaw >= 1e-3:
                    result['reject_reason'] = 'base_drift_exceeded'; break
                if sim.breaker.broken_count:
                    result['reject_reason'] = 'branch_break'; break
                if met.picks and terminal_frame is None:
                    first = met.picks[0]
                    if len(met.picks) != 1 or not all(first[k] for k in ('grasped', 'detached', 'placed')) or first['fail_reason']:
                        result['reject_reason'] = 'first_attempt_' + sanitize_reason(first['fail_reason'] or 'incomplete'); break
                    terminal_frame = ((frame+1)//2)*2 + 4
                if terminal_frame is None and picker.done:
                    result['reject_reason'] = 'first_attempt_' + sanitize_reason(picker.fail_reason or 'expert_stopped'); break
                if terminal_frame is not None and frame >= terminal_frame:
                    result['accepted'] = True; break
            else:
                result['reject_reason'] = 'first_attempt_timeout_20s'
            result['simulation_fps'] = frame / max(time.monotonic()-sim_start, 1e-9)
            result['sim_duration_s'] = float(sim.sim_time); result['frames'] = len(times)
            first = first or (met.picks[0] if met.picks else met._open_pick)
            if first:
                result.update({k: bool(first[k]) for k in ('grasped', 'detached', 'placed')})
                result['max_pull_N'] = float(first['max_pull_N'])
            result['branch_break_count'] = int(sim.breaker.broken_count)
            result['incidental_detach_count'] = int(sim.apples.broken_count)-int(sim.apples.detached[st.apple_index])
            if result['accepted'] and diagnostics.data['grasped_apple_debug_index'] != st.apple_index:
                result.update(accepted=False, reject_reason='planned_target_not_picked')
            if result['accepted'] and result['incidental_detach_count']:
                result.update(accepted=False, reject_reason='incidental_detach')
            if result['accepted'] and not visibility.data['initial']['any_policy_view_visible']:
                result.update(accepted=False, reject_reason='target_not_visible_in_policy_observation')
            result['max_base_translation_drift_m'] = max_trans; result['max_base_yaw_drift_rad'] = max_yaw
            result['expert_final_state'] = picker.state; result['expert_fail_reason'] = picker.fail_reason
            result['state_trace'] = state_trace
            metadata = dict(detach_force_multiplier=detach_force_scale,
                detach_diagnostics=diagnostics.data, target_visibility=visibility.data, base_pose=bases, timestamps=times, debug_usage='NOT POLICY INPUT',
                autopicker_grasp_semantics='official_autopicker_assist', first_attempt_success=result['accepted'],
                attempt_count=len(met.picks)+(met._open_pick is not None), home_arm_joint=home[:7].tolist(),
                max_base_translation_drift_m=max_trans, max_base_yaw_drift_rad=max_yaw,
                branch_break_count=result['branch_break_count'], incidental_detach_count=result['incidental_detach_count'],
                expert=EXPERT, base_policy=BASE_POLICY,
                fixed_base_expert=dict(debug_usage='NOT POLICY INPUT', planner=plan.report['planner'],
                    expert_perception='none: privileged reset-time target; GRASP servos on simulator fruit pose '
                                      'exactly as the official backend does',
                    **result['stance'], candidate_count=plan.report['candidate_count'],
                    feasible_count=plan.report['feasible_count'], rejection_reasons=plan.report['rejection_reasons'],
                    clean_start=result['clean_start'], state_trace=state_trace, base_commands_issued=0),
                **{k: result[k] for k in ('grasped', 'detached', 'placed')})
            traj = dict(schema=SCHEMA, trajectory_type='success', seed=seed, num_frames=len(times), record_fps=30,
                instruction={'general': [{'images': ['observations.ego', 'observations.wrist_left'],
                  'conversations': [{'from': 'human', 'value': PROMPT}, {'from': 'gpt', 'value': '<bot></bot>'}]}]},
                observations={}, proprios=states,
                actions={k: states[k][1:]+states[k][-1:] for k in ACTION_KEYS}, orchardbench=metadata)
    finally:
        if sim is not None: wp.synchronize_device(sim.model.device)
        if viewer is not None: viewer.close()
        # Include planner-infeasible / reset-only early returns too.
        result['episode_wall_s'] = time.monotonic()-started
    return result, traj, images


def manifest_write(path, rows):
    tmp = path.with_name(path.name+'.writing')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def statistics(output, rows, length):
    total = np.zeros((length, 32)); square = total.copy(); count = 0
    lo = np.full(14, np.inf); hi = np.full(14, -np.inf)
    for row in rows:
        if not row['accepted'] or row['split'] != 'train': continue
        t = json.loads((output/row['annotation']).read_text()); n = t['num_frames']-length+1
        p = {k: np.asarray(v) for k, v in t['proprios'].items()}
        a = {k: np.asarray(v) for k, v in t['actions'].items()}
        r = p['ee_rotm'][:n].reshape(-1, 3, 3); rt = r.transpose(0, 2, 1)
        for k in range(length):
            v = np.zeros((n, 32))
            v[:, :3] = np.einsum('nij,nj->ni', rt, a['ee_pos'][k:k+n]-p['ee_pos'][:n])
            v[:, 3:6] = Rotation.from_matrix(rt @ a['ee_rotm'][k:k+n].reshape(-1, 3, 3)).as_rotvec()
            v[:, 6:7] = a['gripper_pos'][k:k+n]-p['gripper_pos'][:n]
            v[:, 7:14] = a['arm_joint'][k:k+n]-p['arm_joint'][:n]
            total[k] += v.sum(0); square[k] += (v*v).sum(0)
            lo = np.minimum(lo, v[:, :14].min(0)); hi = np.maximum(hi, v[:, :14].max(0))
        count += n
    if not count: return None
    mean = total/count; std = np.sqrt(np.maximum(0, square/count-mean*mean)); std[:, 14:] = 1.
    data = dict(schema='orchard_single_arm_preview_32_v0', action_length=length,
        mean=mean.tolist(), std=std.tolist(), active_dims=[0, 14], unused_dims=[14, 32],
        source_split='train', full_window_samples=count, normalization_epsilon=1e-6,
        active_min=lo.tolist(), active_max=hi.tolist(), min_active_std=float(std[:, :14].min()),
        max_active_std=float(std[:, :14].max()), requires_single_arm_loader_adapter=True)
    write_json(output/'action_stats_30x32.json', data)
    return data


def final_audit(output, rows):
    accepted = [r for r in rows if r['accepted']]
    assert len({r['seed'] for r in rows}) == len(rows)
    assert len({r['episode_id'] for r in accepted}) == len(accepted)
    assert len(list((output/'json').glob('*/*.json'))) == len(accepted)
    assert len(list((output/'videos').glob('*.mp4'))) == 2*len(accepted)
    sampled = set(np.random.default_rng(914).choice(len(accepted), min(20, len(accepted)), replace=False).tolist())
    maxima = np.zeros(4)
    for i, row in enumerate(accepted):
        assert row['episode_id'] == f'episode_{i+1:06d}'
        assert row['split'] == ('val' if (i+1)%10 == 0 else 'train')
        t = json.loads((output/row['annotation']).read_text())
        assert t['seed'] == row['seed'] and t['episode_id'] == row['episode_id']
        for info in t['observations'].values(): assert Path(info[0]['path']).is_file()
        errors = validate_arrays(t); maxima = np.maximum(maxima, list(errors.values()))
        if i in sampled: validate_videos(t)
    return dict(accepted=len(accepted), train=sum(r['split']=='train' for r in accepted),
        val=sum(r['split']=='val' for r in accepted), duplicate_seed=0, duplicate_episode=0,
        all_json_arrays_and_roundtrips='PASS' if accepted else 'NOT RUN: no accepted episodes',
        redecoded_episodes=len(sampled), roundtrip_max_errors=maxima.tolist() if accepted else None)


def summarize(args, rows):
    accepted = [r for r in rows if r['accepted']]
    stats = statistics(args.output, rows, args.action_length)
    audit = final_audit(args.output, rows)
    reasons = dict(Counter(r['reject_reason'] for r in rows if not r['accepted']))
    def distribution(key):
        a = np.array([r[key] for r in accepted])
        return dict(min=float(a.min()), mean=float(a.mean()), median=float(np.median(a)), max=float(a.max()), sum=float(a.sum())) if len(a) else None
    size = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    passed = len(accepted) == args.accepted
    summary = dict(status='DATASET COLLECTION PASS' if passed else 'DATASET COLLECTION PARTIAL',
        requested_accepted=args.accepted, attempted=len(rows), accepted=len(accepted),
        acceptance_rate=len(accepted)/max(1,len(rows)), rejection_reasons=reasons, integrity=audit,
        episode_statistics={k:distribution(k) for k in ('frames','sim_duration_s','episode_wall_s','encoding_wall_s','simulation_fps')},
        incidental_detach_count=sum(r['incidental_detach_count'] for r in accepted), disk_bytes=size,
        gpu_memory_mib=gpu_memory(), pilot=args.pilot, no_training_performed=True,
        raw_format_direct_official_loader_compatible=False)
    write_json(args.output/'collection_summary.json', summary)
    section = 8 if args.pilot else 12
    update_log(args.log, section, f'Output: {args.output}\n```json\n'+json.dumps(summary, indent=2)+'\n```')
    if args.pilot:
        gate = args.raw_dir/'pilot_gate.json'
        if passed:
            write_json(gate, dict(status='PASS',accepted=len(accepted),schema=SCHEMA,
                config=json.loads((args.output/'collection_config.json').read_text()),
                audit=audit,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        update_log(args.log, 9, f'Accepted={len(accepted)}; roundtrip/video integrity: {json.dumps(audit)}. '+
            ('PILOT PASS; formal collection and training were not started.' if passed else 'PILOT INCOMPLETE; formal collection gate remains closed.'))
    else:
        update_log(args.log, 13, json.dumps(reasons, indent=2))
        update_log(args.log, 14, json.dumps(summary['episode_statistics'], indent=2))
        update_log(args.log, 15, 'No accepted train windows.' if stats is None else json.dumps({k:v for k,v in stats.items() if k not in ('mean','std')},indent=2)+'\nmean/std shape=[30,32]; unused mean0/std1; train full windows only.')
        update_log(args.log, 16, f'Each accepted pair fully decoded by decord immediately; final recheck {audit["redecoded_episodes"]} episodes. Counts/FPS/RGB/variance/view distinction enforced.')
        update_log(args.log, 17, f'Per-physics-frame drift limits<1e-3m/<1e-3rad; reject entire episode otherwise. Accepted max translation={max((r["max_base_translation_drift_m"] for r in accepted),default=None)}, yaw={max((r["max_base_yaw_drift_rad"] for r in accepted),default=None)}.')
        update_log(args.log, 19, f'{size} bytes; no temporary PNGs. 300 is an initial proof-of-learning scale, not a theoretically optimal size; no training performed.')
        update_log(args.log, 20, json.dumps(audit, indent=2))
        update_log(args.log, 21, summary['status'])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'data/orchard_autopicker_v0')
    ap.add_argument('--log', type=Path, default=DEFAULT_LOG)
    ap.add_argument('--accepted', type=int, default=300)
    ap.add_argument('--seed-start', type=int, default=100000)
    ap.add_argument('--max-attempts', type=int, default=1200)
    ap.add_argument('--record-fps', type=int, default=30)
    ap.add_argument('--action-length', type=int, default=30)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--detach-force-scale', type=float, default=1.0)
    ap.add_argument('--raw-dir', type=Path, default=None,
                    help='per-seed stdout captures, pilot gate and pilot checkpoint (default: <log dir>/raw_dataset)')
    args = ap.parse_args(); args.output = args.output.resolve(); args.log = args.log.resolve()
    args.raw_dir = (args.raw_dir or args.log.parent/'raw_dataset').resolve()
    if args.record_fps != 30 or args.action_length != 30 or args.accepted < 1 or args.max_attempts < 1:
        ap.error('v0 requires30Hz,length30,positive counts')
    if not np.isfinite(args.detach_force_scale) or args.detach_force_scale <= 0:
        ap.error('detach-force-scale must be finite and positive')
    if not args.pilot:
        gate_path=args.raw_dir/'pilot_gate.json'
        if not gate_path.exists(): raise RuntimeError('Formal collection requires a completed pilot audit (at least 3 episodes)')
        gate=json.loads(gate_path.read_text())
        if gate['status']!='PASS' or gate['schema']!=SCHEMA or gate['accepted']<3:
            raise RuntimeError('DATA FORMAT AUDIT FAIL: invalid pilot gate')
    decoder(); subprocess.run(['ffmpeg','-version'], stdout=subprocess.DEVNULL, check=True)
    backup=args.raw_dir/'pilot_checkpoint'
    if args.pilot and args.resume and not args.output.exists() and backup.exists():
        shutil.copytree(backup,args.output)
    if args.output.exists() and not args.resume: raise FileExistsError('Output exists; use --resume (never overwrite)')
    if args.resume and not (args.output/'manifest.jsonl').exists(): raise FileNotFoundError('No manifest to resume')
    args.output.mkdir(parents=True, exist_ok=True)
    lock=(args.output/'.collection.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sub in ('json/train','json/val','videos'): (args.output/sub).mkdir(parents=True,exist_ok=True)
    raw = args.raw_dir; raw.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'manifest.jsonl'
    checkpoint = raw/'pilot_checkpoint' if args.pilot else None
    def checkpoint_pilot():
        if checkpoint is not None:
            shutil.copytree(args.output, checkpoint, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.collection.lock','*.writing'))
    rows = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
    from treesim.target_visibility import MIN_VISIBLE_PIXELS
    config = dict(detach_force_multiplier=args.detach_force_scale, target_visibility_min_pixels=MIN_VISIBLE_PIXELS,
                  schema=SCHEMA, seed_start=args.seed_start, record_fps=30, action_length=30,
                  base_policy=BASE_POLICY, expert=EXPERT, stance_planner='fixed_base_stance_planner_v2',
                  terminal_control_frames=2,
                  max_sim_seconds=20, split_rule='every10th accepted val', script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    config_path = args.output/'collection_config.json'
    if not args.pilot:
        for key in ('detach_force_multiplier','target_visibility_min_pixels','base_policy','expert','stance_planner','record_fps','action_length','terminal_control_frames','max_sim_seconds'):
            if config[key]!=gate['config'].get(key): raise RuntimeError(f'Pilot/config mismatch: {key}')
    if args.resume:
        old = json.loads(config_path.read_text())
        for key in config:
            if key != 'script_sha256': assert config[key] == old[key], ('Resume config mismatch',key)
        pending_path=args.output/'.pending_episode.json'
        if pending_path.exists():
            pending=json.loads(pending_path.read_text()); r=pending['result']; t=pending['trajectory']
            entry=next(x for x in rows if x['seed']==r['seed'])
            if not entry['accepted']:
                assets=[Path(v[0]['path']) for v in t['observations'].values()]
                if all(path.is_file() for path in assets):
                    validate_arrays(t);validate_videos(t)
                    dest=args.output/r['annotation']
                    if dest.exists(): assert json.loads(dest.read_text())==t
                    else: write_json(dest,t)
                    entry.clear();entry.update(r);manifest_write(manifest,rows)
                else:
                    # These files belong only to our incomplete transaction,
                    # never to a committed/accepted manifest entry.
                    for path in assets:
                        assert path.parent==args.output/'videos'
                        path.unlink(missing_ok=True)
            pending_path.unlink()
        for row in rows:
            if row.get('status') == 'in_progress':
                row.update(status='complete',accepted=False,reject_reason='interrupted_attempt')
        manifest_write(manifest,rows)
        # A interrupted transaction must never silently replace completed assets.
        known = {r.get('annotation') for r in rows if r['accepted']}
        orphans = [str(p) for p in (args.output/'json').glob('*/*.json') if str(p.relative_to(args.output)) not in known]
        if orphans: raise RuntimeError(f'Orphan committed files need integrity review: {orphans}')
    else:
        write_json(config_path,config); manifest_write(manifest,rows)
    command = 'cd '+shlex.quote(str(ROOT))+'\nexport PATH="$HOME/.pixi/bin:$PATH"\npixi run python '+shlex.join(['scripts/collect_autopicker_dataset.py', *sys.argv[1:]])
    journal(args.log, ('PILOT' if args.pilot else 'FORMAL')+' launch:\n```bash\n'+command+'\n```')
    if not args.pilot: update_log(args.log,10,'Actual formal launch command:\n```bash\n'+command+'\n```')
    count = sum(r['accepted'] for r in rows); seed = max([args.seed_start-1,*[r['seed'] for r in rows]])+1
    try:
        while count < args.accepted and len(rows) < args.max_attempts:
            row = dict(seed=seed,status='in_progress',accepted=False,reject_reason='interrupted_attempt')
            rows.append(row); manifest_write(manifest, rows);checkpoint_pilot()
            with (raw/f'{"pilot" if args.pilot else "formal"}_{seed}.txt').open('a') as capture:
                with redirect_stdout(capture), redirect_stderr(capture):
                    result,traj,images = collect_episode(seed,raw,args.detach_force_scale)
            result['status']='complete'; result['encoding_wall_s']=0.
            if result['accepted']:
                eid = f'episode_{count+1:06d}'; split = 'val' if (count+1)%10==0 else 'train'
                traj.update(episode_id=eid,split=split)
                dest = args.output/f'json/{split}/{eid}.json'
                assert not dest.exists()
                encoded = time.monotonic()
                with tempfile.TemporaryDirectory(prefix='orchard_record_') as temp:
                    temp = Path(temp)
                    for view,frames in zip(('ego','wrist_left'),images):
                        path = temp/f'{eid}_{view}.mp4'; encode(path,frames)
                        traj['observations'][view]=[dict(path=str(path),start=0,end=traj['num_frames'],fps=30,crop_bbox=None)]
                    validate_arrays(traj); result['video_audit']=validate_videos(traj)
                    result['roundtrip']=round_trip(traj)
                    sources=[]
                    for view in ('ego','wrist_left'):
                        src=Path(traj['observations'][view][0]['path']);final=args.output/'videos'/src.name
                        assert not final.exists();sources.append((src,final))
                        traj['observations'][view][0]['path']=str(final)
                    result['encoding_wall_s']=time.monotonic()-encoded
                    result.update(episode_id=eid,split=split,annotation=str(dest.relative_to(args.output)))
                    write_json(args.output/'.pending_episode.json',dict(result=result,trajectory=traj))
                    for src,final in sources: shutil.move(str(src),final)
                    write_json(dest,traj)
                count+=1
            row.clear();row.update(result);manifest_write(manifest,rows)
            (args.output/'.pending_episode.json').unlink(missing_ok=True)
            checkpoint_pilot()
            msg=f'seed={seed} accepted={result["accepted"]} total={count}/{args.accepted} attempts={len(rows)} reason={result["reject_reason"]} frames={result["frames"]} sim_s={result["sim_duration_s"]:.2f} states={[s["state"] for s in result.get("state_trace", [])]} breaks={result["branch_break_count"]} drift_m={result.get("max_base_translation_drift_m")} wall={result["episode_wall_s"]:.2f}s'
            journal(args.log,msg)
            if not args.pilot: update_log(args.log,11,msg)
            del traj,images;gc.collect();seed+=1
    except BaseException:
        journal(args.log,'COLLECTION INTERRUPTED/ERROR:\n```text\n'+traceback.format_exc()+'\n```')
        raise
    summary=summarize(args,rows);checkpoint_pilot();journal(args.log,summary['status'])
    return 0 if summary['accepted']==args.accepted else 2


if __name__ == '__main__': sys.exit(main())
