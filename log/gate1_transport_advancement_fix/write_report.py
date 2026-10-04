"""Render the frozen experiment evidence without changing execution/evaluation."""
from collections import Counter
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_gate1_oracle as gate

s = json.loads((OUT / 'regression_summary.json').read_text())
four = json.loads((OUT / 'four_summary.json').read_text())
protocol = json.loads((OUT / 'frozen_protocol.json').read_text())
base = Path(protocol['baseline'])
phase_totals = {}
for version, directory in [('old', base), ('new', OUT)]:
    counters = {k: Counter() for k in ['steps', 'dwell_timeout', 'forced_advancement', 'ik_failed', 'clipped', 'passed']}
    for e in protocol['episodes']:
        r = json.loads((directory / 'episodes' / (e['episode_id'] + '_reach-conditioned.json')).read_text())
        for row in r['steps']:
            phase = row['phase']
            counters['steps'][phase] += 1
            counters['dwell_timeout'][phase] += int(row['dwell_timeout'])
            counters['ik_failed'][phase] += int(row['info']['ik_failed'])
            counters['clipped'][phase] += int(row['info']['action_clipped'])
            counters['passed'][phase] += int(row.get('passed_waypoint', False))
        for event in r['dwell_timeout_events']:
            counters['forced_advancement'][event['phase']] += int(event['action'] == 'advance_one_waypoint')
    phase_totals[version] = {k: dict(v) for k, v in counters.items()}
for p, h in {**protocol['source_hashes'], **protocol['expert_hashes']}.items():
    assert gate.sha(Path(p)) == h, p
s.update(expert_accepted=30, baseline=str(base), four_seed_results=four['per_seed'],
    phase_totals=phase_totals, source_files=['scripts/check_gate1_oracle.py'],
    test_files=['scripts/test_gate1_transport_advancement.py'],
    bounded_lookahead_reason='All four failures have zero TRANSPORT IK failure; the necessary evidence is absent.',
    diagnosis='Four seeds enter TRANSPORT after 18.9–20.7 s. All old/new dwell timeouts and IK failures of these seeds occur before TRANSPORT. Unmet TRANSPORT targets are ahead of the TCP, not geometrically passed. The proposed passage mechanism does not address these failures.',
    no_parameter_tuning=True, no_time_indexed_runs=True, no_training_dataset_or_training=True,
    fixed_budget_seconds=30, fixed_budget_steps=900,
    regression_reuses_four_unchanged_build_runs=True,
    checks=dict(geometry_unittests=4, source_and_expert_hashes=True, paired_reset=True,
                index_increments_zero_or_one=True, pass_event_guards=True),
    outcome_scope='FAIL: the requested strict-success improvement and dwell reduction were not achieved; conservative passage implemented and regression completed.')
gate.write(OUT / 'summary.json', s)
lines = ['# Gate 1 TRANSPORT advancement 修复验证', '',
    f"最终判断：**{s['conclusion']}**。固定 cohort strict success {s['aggregate']['old']['success']}/30 → {s['aggregate']['new']['success']}/30。未达成本轮消除四条失败、降低 dwell/forced advancement 的目标。", '',
    '## 修改与约束', '',
    '- 执行源码仅修改 `scripts/check_gate1_oracle.py`；新增聚焦测试 `scripts/test_gate1_transport_advancement.py`。实验 runner、report generator 和原始结果均在本目录。',
    '- 原有 position ≤ 0.01 m 且 rotation ≤ 0.08 rad 的 reach 不变。普通 TRANSPORT waypoint 可由几何 passed 推进一步；每个 control step 最多一步。',
    '- passed 要求 TCP 同时越过 incoming/outgoing 目标平面，outgoing 投影落在当前→下一个 target 段内，横向误差 ≤ 0.01 m，当前 target 旋转误差 ≤ 0.08 rad。',
    '- 首尾各两个 TRANSPORT waypoint 不应用 passed；不跨 phase。退化段（<1 µm）、相邻段 >0.02 m、转弯 >60° 不应用 passed。阈值在首次复测前固定，此后未调整。',
    '- 原有 maximum dwell=30 的异常 fallback 保留；边界点仍使用原严格 reach 判定，不增加宽松 pass。`skip_count` 继续只统计旧 timeout 强推，几何推进另列 `passed_waypoint_count`。',
    '- 无 bounded IK look-ahead：四条失败在 TRANSPORT 的 IK failure 均为 0，不满足启用前提。',
    '- 沿用原 30 条已 accepted expert 文件，SHA256 一致，没有重新采集。Gate 0 TRANSPORT profile、30 s/900 steps、环境、adapter、IK、action、evaluator、grasp/PULL/DROP 均不改。未运行 time-indexed，未创建训练 dataset 或启动训练。', '',
    '## 原四条失败', '',
    '| seed | strict old → new | final target old → new / total | dwell old → new | forced old → new | IK old → new | clipped old → new | passed |',
    '| --- | --- | --- | --- | --- | --- | --- | --- |']
for p in four['per_seed']:
    a,b=p['old'],p['new']
    state=lambda r:'PASS' if r['success'] else 'FAIL'
    fields=' | '.join(f'{a[k]} → {b[k]}' for k in ['dwell_timeout_count','skip_count','IK_failed_steps','clipped_steps'])
    lines.append(f"| {p['seed']} | {state(a)} → {state(b)} | {a['final_target_index']} → {b['final_target_index']} / {b['total_targets']} | {fields} | {b['passed_waypoint_count']} |")
lines += ['', '四条均 grasp/detach 成功，均在 TRANSPORT 达到 900 步 budget。target index 是日志中的 0-based index。', '',
    '### 对共同根因假设的核验', '',
    '| seed | 原进入 TRANSPORT 时刻 | 原 timeout 的 phase 分布 | 原 IK failure 的 phase 分布 | TRANSPORT 未 reach 但可 passed |',
    '| --- | --- | --- | --- | --- |']
for d in json.loads((OUT/'baseline_failure_diagnostics.json').read_text()):
    lines.append(f"| {d['seed']} | {d['transport_entry_s']:.3f} s | {d['dwell_by_phase']} | {d['ik_by_phase']} | {d['unmet_pass_candidates']} |")
lines += ['', '原日志中四条 TRANSPORT dwell timeout=0、IK failure=0；普通 waypoint 多在 2–5 步 reach，未达标的 TCP 沿前进方向仍在 target 后方。原总体 dwell/IK 统计不能归因为最终失败 phase。所有四条未达标 TRANSPORT 状态的 outgoing progress 也没有出现正向越过。',
    '因此“TRANSPORT 反复等待 maximum dwell”及“这两条在 TRANSPORT 连续 IK 不可解”的假设均不受现有日志支持。进入 TRANSPORT 时已经消耗约 19–21 s，剩余预算未能容纳逐点跟踪到 DROP。进一步改变前置阶段或使 TCP 尚未越过 target 就前进，不属于本轮授权的几何 passed 修复。', '',
    '## 固定 30-seed 回归', '',
    '先完成四条并检查无不合理跳点，再执行其余 26 条以核验回退；完整结果包含相同源码/协议下的四条首轮复测，合计每 seed 一次。其余 26 条使用三个独立环境进程调度，没有增加样本或重试调参。', '',
    '| metric | old | new |', '| --- | ---: | ---: |']
for k in ['success','grasped','detached','timeout','control_steps','dwell_timeout_count','skip_count','IK_failed_steps','clipped_steps','passed_waypoint_count']:
    lines.append(f"| {k} | {s['aggregate']['old'][k]} | {s['aggregate']['new'][k]} |")
lines += ['', f"原 PASS 回退 seeds：{s['regression_seeds'] or '无'}；原 FAIL 改善 seeds：{s['resolved_seeds'] or '无'}。",
    f"几何 passed 事件共 {s['aggregate']['new']['passed_waypoint_count']}；检查未发现错误 passed。事件若为 0，仅表示本 cohort 未触发新分支，不能声称已从仿真证明该分支改善执行。", '',
    '| phase | dwell old → new | forced old → new | IK old → new | clipped old → new |',
    '| --- | --- | --- | --- | --- |']
for phase in ['REACH','GRASP','PULL','TRANSPORT','DROP','DONE']:
    fields=' | '.join(f"{phase_totals['old'][k].get(phase,0)} → {phase_totals['new'][k].get(phase,0)}" for k in ['dwell_timeout','forced_advancement','ik_failed','clipped'])
    lines.append(f'| {phase} | {fields} |')
lines += ['', '逐 seed 完整对照见 `summary.json` 的 `per_seed`；原始 TCP、target、误差、IK、clipping、progress 和推进原因见 `episodes/*.json`。',
    '个别步数/末端数值与旧结果存在差异，部分差异在进入 TRANSPORT 前已出现，不能归因于几何 passed；未通过重复试验选取有利结果。', '',
    '## 验证及复现', '',
    '- 四项 unittest 覆盖合法越过、后方/偏离/越界/旋转错误、phase 首尾保护、退化/急转/大间隔拒绝。',
    '- 核验相同 expert SHA、reset pose/info、源码 SHA、900 步预算、target index 每步只增 0 或 1，以及实际 passed 的所有几何约束。',
    '- `.pixi/envs/default/bin/python scripts/test_gate1_transport_advancement.py`',
    '- `.pixi/envs/default/bin/python log/gate1_transport_advancement_fix/run_experiment.py four`',
    '- `.pixi/envs/default/bin/python log/gate1_transport_advancement_fix/run_experiment.py regression`',
    '- `.pixi/envs/default/bin/python log/gate1_transport_advancement_fix/write_report.py`', '',
    '到此停止。本轮未改善共同失败，不能以实现完成代替 Gate 1 PASS，也不继续扩大阈值、budget 或修改禁止范围。', '']
(OUT / 'report.md').write_text('\n'.join(lines))
gate.write(OUT / 'artifact_manifest.json', {str(p.relative_to(ROOT)):gate.sha(p) for p in [
    ROOT/'scripts/check_gate1_oracle.py', ROOT/'scripts/test_gate1_transport_advancement.py',
    OUT/'run_experiment.py', OUT/'write_report.py', OUT/'frozen_protocol.json']})
print(json.dumps(dict(conclusion=s['conclusion'], aggregate=s['aggregate'], regression=s['regression_seeds']), ensure_ascii=False))
