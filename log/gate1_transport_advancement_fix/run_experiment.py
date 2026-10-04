"""Reuse the exact accepted experts; reach-only, four failures before regression."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_gate1_oracle as gate

gate.CONFIG = replace(gate.CONFIG, max_control_steps=900)
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'log/gate1_transport_30s_revalidation'
FAILED = [1000056, 1000294, 1000465, 1000676]
KEYS = ['success', 'grasped', 'detached', 'timeout', 'control_steps', 'dwell_timeout_count',
        'skip_count', 'IK_failed_steps', 'clipped_steps']


def compact(r):
    return {k: r[k] for k in KEYS + ['seed', 'final_target_index', 'total_targets', 'final_phase']} | {
        'passed_waypoint_count': r.get('passed_waypoint_count', 0),
        'phase_steps': dict(Counter(s['phase'] for s in r['steps'])),
        'dwell_by_phase': dict(Counter(s['phase'] for s in r['steps'] if s['dwell_timeout'])),
        'ik_by_phase': dict(Counter(s['phase'] for s in r['steps'] if s['info']['ik_failed']))}


def summarize(episodes, stage):
    pairs = []
    for e in episodes:
        name = e['episode_id'] + '_reach-conditioned.json'
        a = json.loads((BASE / 'episodes' / name).read_text())
        b = json.loads((OUT / 'episodes' / name).read_text())
        assert a['source_sha256'] == b['source_sha256']
        assert a['initial_pose'] == b['initial_pose'] and a['initial_info'] == b['initial_info']
        assert b['control_steps'] <= 900
        indices = [s['target_index'] for s in b['steps']]
        assert all(y - x in (0, 1) for x, y in zip(indices, indices[1:]))
        for s in b['steps']:
            if s['passed_waypoint']:
                d = s['transport_progress']
                assert s['phase'] == 'TRANSPORT' and d['eligible'] and d['passed']
                assert 0 <= d['progress_m'] <= d['next_segment_length_m']
                assert d['lateral_error_m'] <= gate.CONFIG.ik_position_tolerance
                assert s['rotation_error_rad'] <= gate.CONFIG.ik_rotation_tolerance
        same_steps = len(a['steps']) == len(b['steps']) and all(
            all(x[k] == y[k] for k in ('target_index', 'measured_position', 'measured_quaternion', 'info'))
            for x, y in zip(a['steps'], b['steps']))
        pairs.append(dict(episode_id=e['episode_id'], seed=e['seed'], old=compact(a), new=compact(b),
                          exact_baseline_step_match=same_steps))
    sums = {version: {k: sum(p[version][k] for p in pairs) for k in KEYS + ['passed_waypoint_count']}
            for version in ['old', 'new']}
    summary = dict(stage=stage, episodes=len(pairs), aggregate=sums, per_seed=pairs,
                   regression_seeds=[p['seed'] for p in pairs if p['old']['success'] and not p['new']['success']],
                   resolved_seeds=[p['seed'] for p in pairs if not p['old']['success'] and p['new']['success']],
                   bounded_lookahead=False, incorrect_pass_events=0,
                   conclusion='PASS' if sums['new']['success'] == len(pairs) else 'FAIL')
    gate.write(OUT / f'{stage}_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_seed'}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['four', 'regression', 'worker'])
    ap.add_argument('--episode')
    args = ap.parse_args()
    old = json.loads((BASE / 'frozen_protocol.json').read_text())
    protocol_path = OUT / 'frozen_protocol.json'
    if not protocol_path.exists():
        assert args.stage == 'four'
        sources = {p: gate.sha(Path(p)) for p in old['source_hashes']}
        for p, h in old['source_hashes'].items():
            if Path(p).name != 'check_gate1_oracle.py':
                assert sources[p] == h, p
        experts = {}
        for e in old['episodes']:
            d = BASE / 'expert' / e['episode_id']
            c = json.loads((d / 'collection.json').read_text())
            assert c['result']['accepted']
            experts[str(d / 'trajectory.json')] = gate.sha(d / 'trajectory.json')
            assert experts[str(d / 'trajectory.json')] == c['trajectory_sha256']
        gate.write(protocol_path, dict(baseline=str(BASE), episodes=old['episodes'],
            env_config=asdict(gate.CONFIG), maximum_dwell=gate.MAX_DWELL,
            source_hashes=sources, expert_hashes=experts, expert_accepted=30,
            mode='reach-conditioned', bounded_lookahead=False,
            first_last_protected_samples=2, path_tube_m=gate.CONFIG.ik_position_tolerance,
            max_adjacent_segment_m=2 * gate.CONFIG.ik_position_tolerance,
            max_turn_degrees=60, max_advance_per_step=1))
    protocol = json.loads(protocol_path.read_text())
    for p, h in {**protocol['source_hashes'], **protocol['expert_hashes']}.items():
        assert gate.sha(Path(p)) == h, p
    if args.stage == 'worker':
        gate.replay(BASE / 'expert' / args.episode / 'trajectory.json', 'reach-conditioned',
                    OUT / 'episodes' / (args.episode + '_reach-conditioned.json'), gate.sha(protocol_path))
        return
    episodes = [e for e in old['episodes'] if args.stage == 'regression' or e['seed'] in FAILED]
    if args.stage == 'regression':
        assert (OUT / 'four_summary.json').exists()
    def run(e):
        target = OUT / 'episodes' / (e['episode_id'] + '_reach-conditioned.json')
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            print(f"RUN {e['seed']}", flush=True)
            with target.with_suffix('.log').open('w') as f:
                subprocess.run([sys.executable, __file__, 'worker', '--episode', e['episode_id']],
                               cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        r = json.loads(target.read_text())
        assert r['protocol_sha256'] == gate.sha(protocol_path)
        print(json.dumps(compact(r)), flush=True)
    # Independent environments only; reuse the four unchanged-build results.
    with ThreadPoolExecutor(max_workers=1 if args.stage == 'four' else 3) as pool:
        list(pool.map(run, episodes))
    summarize(episodes, args.stage)


if __name__ == '__main__':
    main()
