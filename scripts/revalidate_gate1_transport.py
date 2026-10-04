#!/usr/bin/env python3
"""One fixed Gate 0 TRANSPORT profile, old Gate 1 cohort and unchanged workers."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import traceback

from scipy.stats import binomtest

import check_gate1_oracle as gate1
from treesim.arm_motion import ArmMotionProfile

ROOT = gate1.ROOT
PROFILE = ArmMotionProfile('velocity_accel', .5, 20, active_phases=('TRANSPORT',))
MODES = gate1.MODES
# User-requested amendment: this is the only deployment configuration override.
gate1.CONFIG = replace(gate1.CONFIG, max_control_steps=900)


def normalized(obj):
    return json.loads(json.dumps(obj))


def collect(out, episode):
    from collect_autopicker_dataset import collect_episode
    protocol = json.loads((out / 'frozen_protocol.json').read_text())
    e = next(e for e in protocol['episodes'] if e['episode_id'] == episode)
    dest = out / 'expert' / episode
    dest.mkdir(parents=True, exist_ok=True)
    try:
        result, trajectory, images = collect_episode(
            e['seed'], dest, detach_force_scale=1.5, record_rgb=False,
            arm_motion_profile=PROFILE)
        assert not images[0] and not images[1]
        assert normalized(result['arm_motion_profile']) == normalized(asdict(PROFILE))
        if trajectory is not None:
            trajectory.update(episode_id=e['episode_id'], split='val')
            gate1.write(dest / 'trajectory.json', trajectory)
        payload = dict(seed=e['seed'], episode_id=episode, result=result,
                       experiment_only=True, canonical_dataset_modified=False,
                       profile=asdict(PROFILE), trajectory_available=trajectory is not None,
                       trajectory_sha256=gate1.sha(dest / 'trajectory.json') if trajectory else None,
                       protocol_sha256=gate1.sha(out / 'frozen_protocol.json'))
    except Exception:
        payload = dict(seed=e['seed'], episode_id=episode, result=dict(accepted=False,
                       reject_reason='collection_exception'), error=traceback.format_exc(),
                       trajectory_available=False, profile=asdict(PROFILE))
    gate1.write(dest / 'collection.json', payload)
    print(json.dumps(dict(seed=e['seed'], episode_id=episode, result=payload['result'])), flush=True)


def aggregate(rows):
    counts = {k: sum(r[k] for r in rows) for k in ('success', 'grasped', 'detached', 'timeout',
              'IK_failed_steps', 'clipped_steps', 'dwell_timeout_count', 'skip_count', 'control_steps')}
    counts.update(episodes=len(rows), strict_bucket_success=counts['success'],
        failure_final_phases=dict(Counter(r['final_phase'] for r in rows if not r['success'])),
        budget_failure_phases=dict(Counter(r['final_phase'] for r in rows if r['timeout'] and not r['success'])),
        failure_reasons=dict(Counter(reason for r in rows for reason in r['failure_reasons'])),
        transport_budget_failure=sum(r['timeout'] and not r['success'] and r['final_phase'] == 'TRANSPORT' for r in rows),
        episodes_with_IK_failure=sum(r['IK_failed_steps'] > 0 for r in rows),
        episodes_with_clipping=sum(r['clipped_steps'] > 0 for r in rows),
        episodes_with_dwell_timeout=sum(r['dwell_timeout_count'] > 0 for r in rows),
        success_with_forced_advancement=sum(r['success'] and r['skip_count'] > 0 for r in rows),
        branch_break_episodes=sum(r['branch_break_count'] > 0 for r in rows))
    counts['IK_failure_step_fraction'] = counts['IK_failed_steps'] / max(1, counts['control_steps'])
    counts['clipped_step_fraction'] = counts['clipped_steps'] / max(1, counts['control_steps'])
    counts['position_error'] = gate1.stats([s['position_error_m'] for r in rows for s in r['steps']])
    counts['rotation_error'] = gate1.stats([s['rotation_error_rad'] for r in rows for s in r['steps']])
    return counts


def summarize(out, protocol):
    import csv
    baseline = Path(protocol['baseline'])
    modes, comparisons, collections, details = {}, [], [], []
    for e in protocol['episodes']:
        c = json.loads((out / 'expert' / e['episode_id'] / 'collection.json').read_text())
        collections.append(c)
    accepted_ids = {c['episode_id'] for c in collections if c['result']['accepted']}
    replayed = {}
    paired_checks = 0
    for mode in MODES:
        old, new, diagnostic = [], [], []
        changes = Counter({'old_FAIL_to_new_PASS': 0, 'old_PASS_to_new_FAIL': 0,
                           'unchanged_PASS': 0, 'unchanged_FAIL': 0, 'not_comparable': 0})
        transport_resolved, transport_new = [], []
        for e in protocol['episodes']:
            eid = e['episode_id']
            a = json.loads((baseline / 'episodes' / f'{eid}_{mode}.json').read_text())
            old.append(a)
            p = out / 'episodes' / f'{eid}_{mode}.json'
            b = json.loads(p.read_text()) if p.exists() else None
            valid = eid in accepted_ids and b is not None
            if b:
                replayed[eid, mode] = b
                assert a['initial_pose'] == b['initial_pose'] and a['initial_info'] == b['initial_info']
                assert b['control_steps'] == len(b['steps']) == sum(b['waypoint_dwell_steps'])
                assert b['max_dwell'] <= b['maximum_dwell_limit']
                indices = [s['target_index'] for s in b['steps']]
                assert indices[0] == 0 and all(y - x in (0, 1) for x, y in zip(indices, indices[1:]))
                if mode == 'time-indexed':
                    assert indices == list(range(len(indices)))
                (new if valid else diagnostic).append(b)
                details.append({k: v for k, v in b.items() if k not in
                    ('steps', 'initial_pose', 'initial_info', 'waypoint_dwell_steps', 'dwell_timeout_events')}
                    | dict(expert_accepted=eid in accepted_ids))
            change = ('not_comparable' if not valid else
                      'unchanged_PASS' if a['success'] and b['success'] else
                      'unchanged_FAIL' if not a['success'] and not b['success'] else
                      'old_FAIL_to_new_PASS' if b['success'] else 'old_PASS_to_new_FAIL')
            changes[change] += 1
            old_tb = a['timeout'] and not a['success'] and a['final_phase'] == 'TRANSPORT'
            new_tb = b['timeout'] and not b['success'] and b['final_phase'] == 'TRANSPORT' if valid else None
            if valid and old_tb and not new_tb:
                transport_resolved.append(dict(episode_id=eid, seed=e['seed'], new_success=b['success'],
                                               new_termination=b['termination_reason'], new_final_phase=b['final_phase']))
            if valid and not old_tb and new_tb:
                transport_new.append(eid)
            comparisons.append(dict(seed=e['seed'], episode_id=eid, mode=mode, expert_accepted=eid in accepted_ids,
                old_success=a['success'], new_success=b['success'] if valid else None, change=change,
                old_final_phase=a['final_phase'], new_final_phase=b['final_phase'] if b else None,
                old_transport_budget_failure=old_tb, new_transport_budget_failure=new_tb,
                new_failure_reasons=b['failure_reasons'] if b else ['no_replayable_expert_trajectory']))
        changes['unchanged'] = changes['unchanged_PASS'] + changes['unchanged_FAIL']
        discordant = changes['old_FAIL_to_new_PASS'] + changes['old_PASS_to_new_FAIL']
        modes[mode] = dict(old=aggregate(old), new=aggregate(new), diagnostic_rejected_expert=aggregate(diagnostic),
                          changes=dict(changes), transport_budget_resolved=transport_resolved,
                          new_transport_budget_failure_ids=transport_new,
                          paired_exact_mcnemar_two_sided_p=(float(binomtest(changes['old_FAIL_to_new_PASS'], discordant).pvalue) if discordant else 1.0))
    paired = Counter({f'time {a} / reach {b}': 0 for a, b in [('PASS', 'PASS'), ('FAIL', 'PASS'), ('PASS', 'FAIL'), ('FAIL', 'FAIL')]})
    for e in protocol['episodes']:
        eid = e['episode_id']
        if (eid, MODES[0]) not in replayed or (eid, MODES[1]) not in replayed:
            continue
        a, b = (replayed[eid, mode] for mode in MODES)
        assert a['source_sha256'] == b['source_sha256']
        assert a['initial_pose'] == b['initial_pose'] and a['initial_info'] == b['initial_info']
        aa = {s['target_index']: s for s in a['steps']}
        for s in b['steps']:
            if s['target_index'] in aa:
                assert all(s[k] == aa[s['target_index']][k] for k in
                           ('target_position', 'target_rotation', 'target_width', 'chunk_anchor_index'))
                paired_checks += 1
        if eid in accepted_ids:
            paired[f"time {'PASS' if a['success'] else 'FAIL'} / reach {'PASS' if b['success'] else 'FAIL'}"] += 1
    for filename, rows in [('episode_results.csv', details), ('old_new_comparison.csv', comparisons)]:
        with (out / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)
    rejected = [dict(episode_id=c['episode_id'], seed=c['seed'], reason=c['result'].get('reject_reason'))
                for c in collections if not c['result']['accepted']]
    r = modes['reach-conditioned']
    delta = r['new']['success'] - r['old']['success']
    conclusion = (f"本轮 reach strict success {r['old']['success']}/30 → {r['new']['success']}/30（净变化 {delta:+d}）；"
                  f"paired exact McNemar 双侧 p={r['paired_exact_mcnemar_two_sided_p']:.4g}。"
                  + ('本轮显示 reach 整体成功率显著改善（双侧 p<0.05）。' if delta > 0 and r['paired_exact_mcnemar_two_sided_p'] < .05
                     else '本轮未证明 reach 整体成功率有统计显著改善。') +
                  "新旧比较同时改变了 expert TRANSPORT 配置与 replay budget（20→30 s），不能将全部改善单独归因于 retiming。"
                  "未增加 fresh legacy 重采对照，接触动力学重复性也可能影响逐 seed 差异。")
    blocker = bool(rejected or r['new']['success'] < 30)
    conclusion += (' Gate 1 仍存在 execution blocker，停止，不训练或修 controller。' if blocker
                   else ' 本轮 reach 全部复现 strict success；本轮止于验证，不启动训练。')
    summary = dict(protocol=protocol, expert_attempted=len(collections), expert_accepted=len(accepted_ids),
                   rejected_expert=rejected, replay_count=len(replayed), modes=modes,
                   paired_counts=dict(paired), per_seed_comparison=comparisons, execution_blocker=blocker,
                   conclusion=conclusion, paired_target_step_checks=paired_checks, integrity_verified=True)
    gate1.write(out / 'gate1_summary.json', summary)
    lines = ['# Gate 1 — frozen TRANSPORT retiming revalidation', '',
             '沿用旧 Gate 1 的 30-seed cohort，每个 seed 仅生成一次 expert trajectory，无替换/重试/调参。',
             '`ArmMotionProfile(mode="velocity_accel", vmax_rad_s=0.5, amax_rad_s2=20, active_phases=("TRANSPORT",))`；REACH / GRASP / PULL / DROP legacy。',
             '直接调用原 check_gate1_oracle.replay，仅按用户追加要求将 CONFIG.max_control_steps 覆盖为 900 / 30 s；环境源码、adapter、IK、workspace、clipping、joint step、gripper、evaluator 与 advancement rule 完全不变。position 0.01 m、SO(3) 0.08 rad、maximum dwell 30。无额外 terminal hold。',
             '已中止的 20 s 新轨迹实验保存在 ../gate1_transport_revalidation/，其结果不混入本轮；本轮 30 seeds 从头重新生成、重新 replay。旧 Gate 1 对照仍是 canonical V1 + 20 s。',
             '轨迹仅存于本次 expert/ 目录；record_rgb=False 沿用 Gate 0 collector 调用，未生成视频、训练 dataset、split 或 normalization stats。',
             f'Expert accepted：{len(accepted_ids)}/30；replay：{len(replayed)}/60。Rejected expert：{rejected or "无"}。若 rejected 轨迹仍可 replay，仅计 diagnostic，不冒充合法成功示教。', '',
             '| mode | strict success old → new | grasp old → new | detach old → new | budget timeout old → new | TRANSPORT budget failure old → new |',
             '| --- | --- | --- | --- | --- | --- |']
    for mode, m in modes.items():
        a, b = m['old'], m['new']
        fields = ' | '.join(f'{a[k]} → {b[k]}' for k in ('success', 'grasped', 'detached', 'timeout', 'transport_budget_failure'))
        lines.append(f'| {mode} | {fields} |')
    lines += ['', '| mode | dwell timeout old → new | forced advancement old → new | IK failed steps old → new | clipped steps old → new |',
              '| --- | --- | --- | --- | --- |']
    for mode, m in modes.items():
        a, b = m['old'], m['new']
        fields = ' | '.join(f'{a[k]} → {b[k]}' for k in ('dwell_timeout_count', 'skip_count', 'IK_failed_steps', 'clipped_steps'))
        lines.append(f'| {mode} | {fields} |')
    lines += ['', '步数与暴露时间不同；结构化 summary 同时提供 IK/clipping step fraction 和受影响 episode 数。失败相位指最终正在执行的示教 target 相位，不是重新定义的在线 expert state。', '',
              '| mode | old FAIL → new PASS | old PASS → new FAIL | unchanged | not comparable |', '| --- | ---: | ---: | ---: | ---: |']
    for mode, m in modes.items():
        c = m['changes']
        lines.append(f"| {mode} | {c['old_FAIL_to_new_PASS']} | {c['old_PASS_to_new_FAIL']} | {c['unchanged']} | {c['not_comparable']} |")
    lines += ['', '| new paired outcome | count |', '| --- | ---: |']
    lines += [f'| {k} | {v} |' for k, v in paired.items()]
    for mode, m in modes.items():
        lines += ['', f"- {mode} final failure phase：{m['old']['failure_final_phases']} → {m['new']['failure_final_phases']}。",
                  f"- {mode} budget failure phase：{m['old']['budget_failure_phases']} → {m['new']['budget_failure_phases']}。",
                  f"- {mode} 不再为 TRANSPORT budget failure 的旧例（含是否真正成功）：{m['transport_budget_resolved']}；新增同类失败：{m['new_transport_budget_failure_ids']}。",
                  f"- {mode} 成功但发生 forced advancement：{m['new']['success_with_forced_advancement']}；失败证据：{m['new']['failure_reasons']}。"]
    lines += ['', '| episode / seed | time old → new | reach old → new | reach new failure evidence |', '| --- | --- | --- | --- |']
    for e in protocol['episodes']:
        pair = [next(c for c in comparisons if c['episode_id'] == e['episode_id'] and c['mode'] == m) for m in MODES]
        fmt = lambda c: f"{'PASS' if c['old_success'] else 'FAIL'} → {'N/A' if c['new_success'] is None else 'PASS' if c['new_success'] else 'FAIL'}"
        lines.append(f"| {e['episode_id']} / {e['seed']} | {fmt(pair[0])} | {fmt(pair[1])} | {', '.join(pair[1]['new_failure_reasons']) or '—'} |")
    lines += ['', conclusion, '', '执行源码、旧 artifacts、旧 canonical 输入哈希未变；新旧 reset 与新 paired targets 已核验。原始证据见 expert/、episodes/；逐 seed 对照见 old_new_comparison.csv。本轮已停止。', '']
    (out / 'gate1_report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(expert_accepted=len(accepted_ids), replay_count=len(replayed), paired=dict(paired),
                         comparison={m: v['changes'] for m, v in modes.items()}, conclusion=conclusion), ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'log/gate1_transport_30s_revalidation')
    ap.add_argument('--baseline', type=Path, default=ROOT / 'log/gate1_gt_oracle')
    ap.add_argument('--collect', metavar='EPISODE_ID')
    ap.add_argument('--replay', type=Path)
    ap.add_argument('--mode', choices=MODES)
    ap.add_argument('--protocol-hash')
    args = ap.parse_args()
    out, baseline = args.output.resolve(), args.baseline.resolve()
    if args.replay:
        gate1.replay(args.replay, args.mode, out, args.protocol_hash)
        return
    if args.collect:
        collect(out, args.collect)
        return
    old = json.loads((baseline / 'frozen_protocol.json').read_text())
    assert len(old['episodes']) == 30
    assert normalized(asdict(gate1.CONFIG)) == old['env_config'] | {'max_control_steps': 900}
    assert gate1.MAX_DWELL == old['maximum_dwell'] == 30
    for p, h in old['source_hashes'].items():
        assert gate1.sha(Path(p)) == h, f'Old execution source changed: {p}'
    protected = {str(p): gate1.sha(p) for p in baseline.rglob('*') if p.is_file()}
    protected.update(old['input_hashes'])
    protected.update({e['path']: e['sha256'] for e in old['episodes']})
    for p in (ROOT / 'log/gate0_speed_30seed').rglob('*'):
        if p.is_file():
            protected[str(p)] = gate1.sha(p)
    sources = dict(old['source_hashes'])
    sources.update({str(p): gate1.sha(p) for p in (Path(__file__).resolve(), ROOT / 'scripts/collect_autopicker_dataset.py')})
    protocol = dict(baseline=str(baseline), episodes=old['episodes'], profile=asdict(PROFILE),
                    env_config=asdict(gate1.CONFIG), maximum_dwell=30,
                    budget_amendment='User requested stop of 20 s run and full restart at 900 control steps / 30 s; this is the sole deployment parameter change.',
                    replay_worker=str(ROOT / 'scripts/check_gate1_oracle.py'),
                    expert_collection=dict(detach_force_scale=1.5, record_rgb=False, attempts_per_seed=1),
                    invalid_expert_policy='Never replace or retry. Replay available complete-window trajectories diagnostically; exclude rejected inputs from accepted-expert success counts.',
                    protected_hashes=protected, source_hashes=sources)
    protocol = normalized(protocol)
    out.mkdir(parents=True, exist_ok=True)
    p = out / 'frozen_protocol.json'
    if p.exists():
        assert json.loads(p.read_text()) == protocol, 'Frozen sources or inputs changed'
    else:
        gate1.write(p, protocol)
    protocol_hash = gate1.sha(p)
    for e in protocol['episodes']:
        eid = e['episode_id']
        dest = out / 'expert' / eid
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / 'collection.json').exists():
            print(f"COLLECT {eid} seed={e['seed']}", flush=True)
            with (dest / 'stdout.log').open('w') as log:
                subprocess.run([sys.executable, __file__, '--output', str(out), '--collect', eid],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        c = json.loads((dest / 'collection.json').read_text())
        print(f"EXPERT {eid} accepted={c['result']['accepted']} reason={c['result'].get('reject_reason')}", flush=True)
        trajectory = dest / 'trajectory.json'
        if not c['trajectory_available']:
            continue
        assert gate1.sha(trajectory) == c['trajectory_sha256']
        if json.loads(trajectory.read_text())['num_frames'] < gate1.HORIZON:
            continue
        for mode in MODES:
            target = out / 'episodes' / f'{eid}_{mode}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                print(f'REPLAY {eid} {mode}', flush=True)
                with target.with_suffix('.log').open('w') as log:
                    subprocess.run([sys.executable, __file__, '--replay', str(trajectory),
                                    '--mode', mode, '--output', str(target), '--protocol-hash', protocol_hash],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            r = json.loads(target.read_text())
            assert r['protocol_sha256'] == protocol_hash and r['source_sha256'] == c['trajectory_sha256']
            print(f"DONE {eid} {mode} success={r['success']} phase={r['final_phase']} budget={r['timeout']} dwell={r['dwell_timeout_count']}", flush=True)
    for p, h in {**protected, **sources}.items():
        assert gate1.sha(Path(p)) == h, f'Protected input/source changed: {p}'
    summarize(out, protocol)


if __name__ == '__main__':
    main()
