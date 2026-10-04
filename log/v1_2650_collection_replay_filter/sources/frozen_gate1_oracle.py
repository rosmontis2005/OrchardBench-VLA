#!/usr/bin/env python3
"""Frozen, paired canonical GT replay through the deployment action adapter.

Run --smoke once, then run without it to retain that pair and finish 30 pairs.
No model, expert execution, trajectory modification, or controller overrides.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from treesim.orchard_action import CONTRACT, HORIZON, OrchardActionAdapter, encode_window
from treesim.vla_env import OrchardVLAEnv, VLAEnvConfig

MODES = ('time-indexed', 'reach-conditioned')
CONFIG = VLAEnvConfig()
MAX_DWELL = HORIZON  # One second at 30 Hz, declared before any validation replay.


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(count=0, mean=None, p50=None, p95=None, max=None)
    return dict(count=len(a), mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def chunks(traj):
    """Complete windows only; overlap the final window but execute each index once.

    The oracle uses recorded anchor observations to preserve absolute GT targets.
    Reanchoring an encoded demo at the lagging live pose would move the path.
    Live observations are used by to_native on EVERY control step, as in deployment.
    """
    n = traj['num_frames']
    assert n >= HORIZON
    for first in range(0, n, HORIZON):
        anchor = min(first, n - HORIZON)
        p = traj['proprios']
        obs = dict(tcp_pos_world=np.asarray(p['ee_pos'][anchor]),
                   tcp_quat_world=Rotation.from_matrix(
                       np.asarray(p['ee_rotm'][anchor]).reshape(3, 3)).as_quat())
        adapter = OrchardActionAdapter()
        adapter.set_denormalized_chunk(encode_window(traj, anchor), obs)
        # Minimal GT entry check, not a dataset/schema re-audit.
        end = min(first + HORIZON, n)
        local = slice(first - anchor, end - anchor)
        xyz, rot, width = adapter.targets
        assert np.allclose(xyz[local], traj['actions']['ee_pos'][first:end], atol=2e-6, rtol=0)
        assert np.allclose(rot[local], np.asarray(traj['actions']['ee_rotm'][first:end]).reshape(-1, 3, 3), atol=2e-6, rtol=0)
        assert np.allclose(width[local], np.asarray(traj['actions']['gripper_pos'][first:end]).reshape(-1), atol=2e-6, rtol=0)
        yield first, end, anchor, adapter


def phase_at(traj, index):
    timestamp = traj['orchardbench']['timestamps'][min(index + 1, traj['num_frames'] - 1)]
    trace = traj['orchardbench']['fixed_base_expert']['state_trace']
    return next(r['state'] for r in reversed(trace) if r['frame'] / CONFIG.sim_hz <= timestamp + 1e-8)


def replay(path, mode, output, protocol_hash):
    traj = json.loads(path.read_text())
    n = traj['num_frames']
    all_chunks = list(chunks(traj))
    env = OrchardVLAEnv(CONFIG)
    rows, events = [], []
    dwell = np.zeros(n, dtype=int)
    reached = np.zeros(n, dtype=bool)
    index = chunk_index = consecutive_ik = max_consecutive_ik = 0
    started = time.monotonic()
    try:
        obs, initial = env.reset(seed=traj['seed'])
        assert obs['sim_time'] == 0 and initial['detach_force_scale'] == 1.5
        initial_pose = {k: obs[k] for k in ('tcp_pos_world', 'tcp_quat_world', 'joint_pos', 'gripper_width', 'base_pos', 'base_yaw')}
        while index < n:
            first, end, anchor, adapter = all_chunks[chunk_index]
            if index >= end:
                chunk_index += 1
                first, end, anchor, adapter = all_chunks[chunk_index]
            local = index - anchor
            cmd = adapter.to_native(local, obs)
            obs, reward, done, truncated, info = env.step(**cmd)
            xyz, rot, width = adapter.targets
            pe = float(np.linalg.norm(xyz[local] - obs['tcp_pos_world']))
            re = float((Rotation.from_matrix(rot[local]) * Rotation.from_quat(obs['tcp_quat_world']).inv()).magnitude())
            in_tolerance = pe <= CONFIG.ik_position_tolerance and re <= CONFIG.ik_rotation_tolerance
            dwell[index] += 1
            reached[index] |= in_tolerance
            consecutive_ik = consecutive_ik + 1 if info['ik_failed'] else 0
            max_consecutive_ik = max(max_consecutive_ik, consecutive_ik)
            expired = mode == 'reach-conditioned' and not in_tolerance and dwell[index] >= MAX_DWELL
            if expired:
                recent = rows[-(MAX_DWELL - 1):]
                failures = sum(r['info']['ik_failed'] for r in recent) + int(info['ik_failed'])
                reason = 'persistent_ik_failure' if failures == MAX_DWELL else 'pose_not_reached_within_maximum_dwell'
                events.append(dict(target_index=index, phase=phase_at(traj, index), reason=reason,
                                   position_error_m=pe, rotation_error_rad=re,
                                   ik_failed_steps=failures, next_target_index=index + 1,
                                   action='advance_one_waypoint' if not (done or truncated) else 'episode_ended'))
            advance = mode == 'time-indexed' or in_tolerance or expired
            rows.append(dict(step=len(rows), target_index=index, chunk_anchor_index=anchor,
                             phase=phase_at(traj, index), sim_time=obs['sim_time'],
                             position_error_m=pe, rotation_error_rad=re,
                             within_tolerance=bool(in_tolerance), dwell_steps=int(dwell[index]),
                             dwell_timeout=bool(expired), requested_action=cmd['action'],
                             target_position=xyz[local], target_rotation=rot[local],
                             target_width=width[local], measured_position=obs['tcp_pos_world'],
                             measured_quaternion=obs['tcp_quat_world'], measured_width=obs['gripper_width'],
                             gripper_intent=env._gripper, info=info))
            if done or truncated:
                termination = 'strict_bucket_success' if done else 'episode_budget'
                break
            if advance:
                index += 1
        else:
            termination = 'target_sequence_exhausted'
        success = bool(info['success'])
        grasped = any(r['info']['held_apple_id_debug'] is not None or r['info']['grasp_assist_triggered'] for r in rows)
        detached = any(r['info']['apple_detached_count'] > 0 for r in rows)
        failure = []
        if not success:
            failure.append('grasp_failure' if not grasped else 'detach_failure' if not detached else 'DROP_bucket_failure')
            failure.append(termination)
            if any(r['info']['ik_failed'] for r in rows):
                failure.append('IK_failure_observed')
            if any(r['info']['action_clipped'] for r in rows):
                failure.append('clipping_observed')
            if events:
                failure.append('dwell_timeout')
            if any(not r['within_tolerance'] for r in rows):
                failure.append('tracking_lag_observed')
        result = dict(seed=traj['seed'], episode_id=traj['episode_id'], mode=mode,
                      source_annotation=str(path), source_sha256=sha(path), protocol_sha256=protocol_hash,
                      success=success, strict_bucket_success=success, grasped=grasped, detached=detached,
                      timeout=termination == 'episode_budget', termination_reason=termination,
                      sim_duration=obs['sim_time'], control_steps=len(rows), wall_seconds=time.monotonic() - started,
                      final_target_index=rows[-1]['target_index'], total_targets=n,
                      target_progress_fraction=float(np.count_nonzero(dwell) / n),
                      targets_visited=int(np.count_nonzero(dwell)), targets_reached=int(reached.sum()),
                      target_reached_fraction=float(reached.mean()),
                      IK_failed_steps=sum(r['info']['ik_failed'] for r in rows),
                      max_consecutive_IK_failures=max_consecutive_ik,
                      clipped_steps=sum(r['info']['action_clipped'] for r in rows),
                      branch_break_count=max(r['info']['branch_break_count'] for r in rows),
                      tracking_outside_tolerance_steps=sum(not r['within_tolerance'] for r in rows),
                      position_error=stats([r['position_error_m'] for r in rows]),
                      rotation_error=stats([r['rotation_error_rad'] for r in rows]),
                      waypoint_dwell_steps=dwell, max_dwell=int(dwell.max()), maximum_dwell_limit=MAX_DWELL if mode == MODES[1] else 1,
                      dwell_timeout_count=len(events), replan_count=0,
                      skip_count=sum(e['action'] == 'advance_one_waypoint' for e in events),
                      replan_reason='No replan; dwell timeout abandons only the current unmet target.',
                      dwell_timeout_events=events, failure_reasons=failure, final_phase=rows[-1]['phase'],
                      initial_info=initial, initial_pose=initial_pose, steps=rows)
        write(output, result)
        print(json.dumps({k: result[k] for k in ('episode_id', 'seed', 'mode', 'success', 'grasped', 'detached', 'sim_duration', 'final_target_index', 'total_targets', 'IK_failed_steps', 'clipped_steps', 'dwell_timeout_count', 'termination_reason')}), flush=True)
    finally:
        env.close()


def summarize(out, protocol):
    results = [json.loads((out / 'episodes' / f"{e['episode_id']}_{m}.json").read_text())
               for e in protocol['episodes'] for m in MODES]
    compact = [{k: v for k, v in r.items() if k not in ('steps', 'initial_info', 'initial_pose', 'waypoint_dwell_steps', 'dwell_timeout_events')} for r in results]
    with (out / 'episode_results.csv').open('w', newline='') as f:
        fields = list(compact[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in compact)
    table = {f'time {a} / reach {b}': [] for a in ('PASS', 'FAIL') for b in ('PASS', 'FAIL')}
    pair_rows = []
    for a, b in zip(results[::2], results[1::2]):
        assert a['source_sha256'] == b['source_sha256'] and a['total_targets'] == b['total_targets']
        assert a['initial_pose'] == b['initial_pose'] and a['initial_info'] == b['initial_info'], 'Paired resets differ'
        key = f"time {'PASS' if a['success'] else 'FAIL'} / reach {'PASS' if b['success'] else 'FAIL'}"
        table[key].append(a['episode_id'])
        pair_rows.append(dict(episode_id=a['episode_id'], seed=a['seed'], category=key,
                              time_failure=a['failure_reasons'], reach_failure=b['failure_reasons']))
    aggregates = {}
    for mode in MODES:
        rr = [r for r in results if r['mode'] == mode]
        aggregates[mode] = dict(episodes=len(rr), **{k: sum(r[k] for r in rr) for k in
            ('success', 'strict_bucket_success', 'grasped', 'detached', 'timeout', 'IK_failed_steps', 'clipped_steps', 'branch_break_count', 'dwell_timeout_count', 'skip_count')},
            episodes_with_IK_failure=sum(r['IK_failed_steps'] > 0 for r in rr),
            episodes_with_clipping=sum(r['clipped_steps'] > 0 for r in rr),
            episodes_with_dwell_timeout=sum(r['dwell_timeout_count'] > 0 for r in rr),
            failure_reasons=dict(Counter(x for r in rr for x in r['failure_reasons'])),
            position_error=stats([s['position_error_m'] for r in rr for s in r['steps']]),
            rotation_error=stats([s['rotation_error_rad'] for r in rr for s in r['steps']]),
            sim_duration=stats([r['sim_duration'] for r in rr]))
    # A descriptive conclusion; no post-hoc numeric acceptance threshold.
    reach_success = aggregates[MODES[1]]['success']
    if reach_success == len(protocol['episodes']):
        conclusion = '当前 deployment execution stack 对 canonical GT trajectory 具有足够可执行性，允许进入首次正式模型训练。'
        if table['time FAIL / reach PASS']:
            conclusion += ' 原 time-indexed execution 存在明确 execution-timing mismatch，reach-conditioned execution 消除了主要问题。'
    else:
        conclusion = 'Gate 1 FAIL。当前仍存在 controller / evaluator / execution contract blocker。停止在此处，不进行正式模型训练。'
    summary = dict(protocol=protocol, modes=aggregates, paired_counts={k: len(v) for k, v in table.items()},
                   paired_episode_ids=table, pairs=pair_rows, conclusion=conclusion,
                   integrity_verified=True, paired_reset_equality_verified=True)
    write(out / 'gate1_summary.json', summary)
    lines = ['# Gate 1 — canonical GT oracle replay', '',
             'GT actions → encode_window → OrchardActionAdapter.set_denormalized_chunk → to_native(live obs) → OrchardVLAEnv.step → 当前 strict bucket evaluator。无模型。', '',
             '两模式只改变 waypoint advancement：time 每步推进；reach 在步后测得 TCP position / SO(3) error 达标时推进，或到 maximum dwell 后记录并放弃当前未达标点、推进一个点。无插值、retiming、recovery 或额外末端等待。targets 耗尽、evaluator success 或 episode budget 到达即停止。', '',
             'GT chunk 使用示教中固定的 anchor，避免 live tracking lag 改变绝对 GT 路径；native command 每步使用真实 live observation。尾部使用重叠完整窗口，但每个 GT index 只按顺序消费一次。该 oracle 测试绝对 GT targets 的可执行性，不测试模型或偏离示教后的预测能力。', '',
             f'固定参数：position ≤ {CONFIG.ik_position_tolerance} m；SO(3) rotation ≤ {CONFIG.ik_rotation_tolerance} rad；maximum dwell = {MAX_DWELL} control steps（1 s，按一个 chunk 长度预先设定，未做 pilot 或 val 调参）；episode budget = {CONFIG.max_control_steps} steps（20 s）；30 Hz。其余环境配置全部默认。', '',
             'canonical manifest 自动选取全部 30 accepted val。读取的是原 canonical annotation，未用 Gate 0 retimed run 替换；两者不能混为同一数据版本或 seed cohort。首次一个 val paired smoke 直接计入正式 30 对，无重复挑选。协议、输入与执行源码哈希被冻结并核验；paired reset 完全一致。', '',
             '| mode | strict success | grasped | detached | budget timeout | IK failed steps | clipped steps | dwell timeouts |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m, a in aggregates.items():
        lines.append(f"| {m} | {a['success']}/30 | {a['grasped']}/30 | {a['detached']}/30 | {a['timeout']} | {a['IK_failed_steps']} | {a['clipped_steps']} | {a['dwell_timeout_count']} |")
    lines += ['', '| paired outcome | episodes |', '| --- | ---: |']
    lines += [f'| {k} | {len(v)} |' for k, v in table.items()]
    lines += ['', 'success 与 strict_bucket_success 均直接来自当前 evaluator；没有把 detach、collector accepted 或 target 消费完成算作 PASS。target_progress_fraction 为已尝试的不同 target 数 / 总数，targets_reached 另记。误差为每步执行后的 GT target 与实际 TCP 误差。', '',
              '| mode | position mean / P95 / max (m) | rotation mean / P95 / max (rad) |', '| --- | --- | --- |']
    for m, a in aggregates.items():
        fmt = lambda s: ' / '.join(f'{s[k]:.5f}' for k in ('mean', 'p95', 'max'))
        lines.append(f"| {m} | {fmt(a['position_error'])} | {fmt(a['rotation_error'])} |")
    lines += ['', '失败标签是可重叠的观测证据，IK failure / clipping / tracking lag 不单独证明根因。DROP_bucket_failure 表示已 grasp + detach 但未满足 strict predicate。未抓到、抓到未 detach、已 detach 未入桶分别记录。', '']
    for m, a in aggregates.items():
        lines.append(f"- {m}: {json.dumps(a['failure_reasons'], ensure_ascii=False)}；出现 IK failure / clipping / dwell timeout 的 episode 数 = {a['episodes_with_IK_failure']} / {a['episodes_with_clipping']} / {a['episodes_with_dwell_timeout']}。")
    lines += ['', '| episode / seed | time | reach | reach final phase; index/N | reach failure evidence |', '| --- | --- | --- | --- | --- |']
    for a, b in zip(results[::2], results[1::2]):
        lines.append(f"| {a['episode_id']} / {a['seed']} | {'PASS' if a['success'] else 'FAIL'} | {'PASS' if b['success'] else 'FAIL'} | {b['final_phase']}; {b['final_target_index']}/{b['total_targets']} | {', '.join(b['failure_reasons']) or '—'} |")
    lines += ['', conclusion, '', '全部 60 次 replay 已完成。未开始训练、未修改 controller / evaluator / physics / canonical 数据。逐步误差、IK、clipping、gripper、phase 和 dwell timeout 原因见 episodes/*.json；每 episode 汇总见 episode_results.csv。', '']
    (out / 'gate1_report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(modes=aggregates, paired_counts=summary['paired_counts'], conclusion=conclusion), ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT.parent / 'data/orchard_autopicker_v1')
    ap.add_argument('--output', type=Path, default=ROOT / 'log/gate1_gt_oracle')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--worker', type=Path)
    ap.add_argument('--mode', choices=MODES)
    ap.add_argument('--protocol-hash')
    args = ap.parse_args()
    if args.worker:
        replay(args.worker, args.mode, args.output, args.protocol_hash)
        return
    out, data = args.output.resolve(), args.data.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = data / 'manifest.jsonl'
    episodes = [r for r in map(json.loads, manifest.read_text().splitlines()) if r.get('accepted') and r.get('split') == 'val']
    assert len(episodes) == 30 and len({e['seed'] for e in episodes}) == 30
    selected = []
    for e in episodes:
        p = data / e['annotation']
        d = json.loads(p.read_text())
        assert d['seed'] == e['seed'] and d['episode_id'] == e['episode_id'] and d['split'] == 'val'
        assert d['record_fps'] == CONFIG.control_hz
        selected.append(dict(episode_id=e['episode_id'], seed=e['seed'], path=str(p), sha256=sha(p)))
    sources = [Path(__file__).resolve(), *sorted((ROOT / 'treesim').glob('*.py'))]
    protocol = dict(contract=CONTRACT, env_config=asdict(CONFIG), maximum_dwell=MAX_DWELL,
                    maximum_dwell_rationale='One HORIZON (30) control steps; fixed without outcome-based selection.',
                    terminal_rule='Stop at success, budget, or sequence exhaustion; no extra terminal hold.',
                    data_root=str(data), episodes=selected,
                    input_hashes={str(p): sha(p) for p in (manifest, data / 'action_stats_30x32.json', data / 'collection_config.json')},
                    source_hashes={str(p): sha(p) for p in sources})
    protocol = json.loads(json.dumps(protocol))  # Normalize dataclass tuples for resume equality.
    protocol_path = out / 'frozen_protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Frozen protocol/input/source changed; refuse to mix runs'
    else:
        write(protocol_path, protocol)
    protocol_hash = sha(protocol_path)
    for e in selected[:1] if args.smoke else selected:
        for mode in MODES:
            target = out / 'episodes' / f"{e['episode_id']}_{mode}.json"
            if target.exists():
                old = json.loads(target.read_text())
                assert old['protocol_sha256'] == protocol_hash and old['source_sha256'] == e['sha256']
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', e['path'], '--mode', mode,
                       '--output', str(target), '--protocol-hash', protocol_hash]
            print(f"START {e['episode_id']} seed={e['seed']} {mode}", flush=True)
            with target.with_suffix('.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
            r = json.loads(target.read_text())
            print(f"DONE {e['episode_id']} {mode} success={r['success']} duration={r['sim_duration']:.3f}s dwell_timeouts={r['dwell_timeout_count']}", flush=True)
    for p, digest in {**protocol['input_hashes'], **protocol['source_hashes'], **{e['path']: e['sha256'] for e in selected}}.items():
        assert sha(Path(p)) == digest, f'Input/source changed: {p}'
    if not args.smoke:
        summarize(out, protocol)


if __name__ == '__main__':
    main()
