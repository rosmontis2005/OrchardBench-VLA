#!/usr/bin/env python3
"""Isolated dataset transaction or frozen oracle job; never launches training."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import collect_autopicker_dataset as collector
from treesim.arm_motion import ArmMotionProfile

PROFILE = ArmMotionProfile('velocity_accel', .5, 20, active_phases=('TRANSPORT',))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_sources(out):
    protocol = read(out / 'protocol.json')
    for path, digest in protocol['source_hashes'].items():
        assert sha(path) == digest, f'Frozen source changed: {path}'
    return protocol


def collect(out, seed, dest):
    protocol = check_sources(out)
    assert protocol['formal_seed_range'][0] <= seed <= protocol['formal_seed_range'][1]
    gate = read(out / 'pilot_raw/pilot_gate.json')
    config = read(out / 'collection_config.json')
    assert gate['status'] == 'PASS' and gate['accepted'] >= 3
    for key in ('schema', 'arm_motion_profile', 'detach_force_multiplier', 'record_fps',
                'action_length', 'base_policy', 'expert', 'stance_planner',
                'terminal_control_frames', 'max_sim_seconds', 'target_visibility_min_pixels'):
        assert gate['config'][key] == config[key], key
    assert not (dest / 'result.json').exists(), 'Never repeat a completed attempt'
    dest.mkdir(parents=True, exist_ok=True)
    collector.write_json(dest / 'started.json', dict(seed=seed, started_unix=time.time(),
                         protocol_sha256=sha(out / 'protocol.json')))
    result, traj, images = collector.collect_episode(
        seed, dest, detach_force_scale=1.5, record_rgb=True, arm_motion_profile=PROFILE)
    assert result['seed'] == seed
    assert result['arm_motion_profile'] == asdict(PROFILE)
    collector.write_json(dest / 'expert_outcome.json', result)
    result.update(status='complete', encoding_wall_s=0.,
                  protocol_sha256=sha(out / 'protocol.json'),
                  worker_source_sha256=sha(__file__),
                  infrastructure_attempt=int(dest.name.removeprefix('try_')))
    if result['accepted']:
        # Temporary transaction identity only; the parent assigns all final IDs/splits.
        traj.update(episode_id=f'pending_seed_{seed}', split='unassigned')
        started = time.monotonic()
        for view, frames in zip(('ego', 'wrist_left'), images):
            path = dest / f'{view}.mp4'
            collector.encode(path, frames)
            traj['observations'][view] = [dict(path=str(path), start=0,
                end=traj['num_frames'], fps=30, crop_bbox=None)]
        result['roundtrip'] = collector.validate_arrays(traj)
        result['video_audit'] = collector.validate_videos(traj)
        result['integrity_valid'] = True
        result['encoding_wall_s'] = time.monotonic() - started
        collector.write_json(dest / 'trajectory.json', traj)
        result['trajectory'] = str(dest / 'trajectory.json')
        result['trajectory_sha256'] = sha(dest / 'trajectory.json')
        result['video_sha256'] = {view: sha(dest / f'{view}.mp4')
                                 for view in ('ego', 'wrist_left')}
    collector.write_json(dest / 'result.json', result)
    print(json.dumps(dict(seed=seed, accepted=result['accepted'],
                         reason=result['reject_reason'])), flush=True)


def replay(out, trajectory, output):
    check_sources(out)
    assert (out / 'raw_cohort_frozen.json').exists(), 'Finish the entire cohort before replay'
    assert not output.exists(), 'Never retry a completed semantic replay'
    source = out / 'sources/frozen_gate1_oracle.py'
    spec = importlib.util.spec_from_file_location('v1_frozen_oracle', source)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    assert frozen.CONFIG.max_control_steps == 600  # Explicitly override the old default.
    frozen.CONFIG = replace(frozen.CONFIG, max_control_steps=900)
    expected = read(out / 'replay_config.json')
    assert json.loads(json.dumps(asdict(frozen.CONFIG))) == expected['env_config']
    assert frozen.MAX_DWELL == 30

    def compact_write(path, result):
        # Inspect the unmodified baseline's result; only omit large per-step debug output.
        rows = result['steps']
        assert result['control_steps'] == len(rows) == sum(result['waypoint_dwell_steps'])
        assert result['control_steps'] <= 900 and result['sim_duration'] <= 30 + 1e-7
        assert result['mode'] == 'reach-conditioned' and result['max_dwell'] <= 30
        assert result['success'] == result['strict_bucket_success'] == rows[-1]['info']['success']
        assert rows[0]['target_index'] == 0
        for a, b in zip(rows, rows[1:]):
            advance = a['within_tolerance'] or a['dwell_timeout']
            assert b['target_index'] - a['target_index'] == int(advance)
        for r in rows:
            assert r['dwell_timeout'] == (not r['within_tolerance'] and r['dwell_steps'] >= 30)
        compact = {k: v for k, v in result.items() if k not in
                   ('steps', 'initial_info', 'initial_pose', 'waypoint_dwell_steps', 'dwell_timeout_events')}
        compact.update(forced_advancement_count=result['skip_count'],
            frozen_replay_source_sha256=sha(source), worker_source_sha256=sha(__file__),
            env_config=asdict(frozen.CONFIG), semantics_verified=True,
            initial_reset_seed=result['seed'])
        collector.write_json(path, compact)

    frozen.write = compact_write
    frozen.replay(trajectory, 'reach-conditioned', output, sha(out / 'protocol.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'replay'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'collect':
        collect(args.run.resolve(), args.seed, args.output.resolve())
    else:
        replay(args.run.resolve(), args.trajectory.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
