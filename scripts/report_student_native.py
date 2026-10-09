#!/usr/bin/env python3
"""Render actual recorded gate/collection results without claiming future work."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'artifacts/student_native_1008'


def read(path):return json.loads(path.read_text())
def rows(directory):return [read(p) for p in sorted(directory.glob('seed_*/result.json'))]
def rate(rs,key):return sum(r.get(key,0)*r.get('steps',0) for r in rs)/max(1,sum(r.get('steps',0) for r in rs))


def main():
    a0=rows(EXP/'dev_v0');a=rows(EXP/'dev_v1');c1=rows(EXP/'gate_c_v1');c=rows(EXP/'gate_c_v2')
    gate=read(EXP/'gate_status.json');collection=ROOT/'data/orchard_requested_v2_2000';progress=read(collection/'progress.json') if (collection/'progress.json').exists() else None
    accepted=[r for r in c if r['accepted']]
    lines=['# Student-native expert / requested-command V2', '',
      '本轮仅重建专家与命令监督数据管线；没有 XR-0 训练、模型结构改动或生产控制器/物理参数修改。', '',
      '## 实现与执行契约', '',
      '- 复用 AutoPicker 的 0.16 m 预抓、沿接近轴最大 0.32 m 回撤、桶上方运输/释放意图；复用现有固定底座规划器、视频编码器和 StrictPlacement。',
      '- 独立小 FSM，无 AutoPicker 继承、私有关节控制、直接 hold/release 或物理状态写入。共享基座系运输 waypoint 为 (0.25,0,0.95)、(-0.26,0,0.85)。',
      '- H=5 固定时钟：每五个真实控制步生成五条缓存 TCP 请求，每条仅调用一次 native_command → OrchardVLAEnv.step → 原 PoseIK/servo。仿真 60 Hz，控制 30 Hz，action_repeat=2；无 reach-conditioned/dwell。',
      '- 最终速度参数 reach/grasp/pull/transport=0.20/0.04/0.08/0.20 m/s，旋转 0.30 rad/s；块内平移/旋转领先量限制 0.018 m/0.040 rad，抓取保留最多 0.010 m 位置领先量。实际速度低于名义值。',
      '- 运输阶段预算从 30 s 有限修正为 45 s，总 episode 70 s；没有改变任何单条动作的执行时长。', '',
      '## 数据契约', '',
      '`orchard_cartesian_local_rotvec_width_requested_v2`：N 条真实 requested command 与 N+1 帧实测 observation 分开保存。每条 command 包含世界系 TCP/旋转/绝对夹爪宽度及实际 native kwargs；有效控制目标、IK、限幅、关节命令和物理响应独立记录。基座位姿和连续时间戳保留。', '',
      '局部动作仍为锚点 TCP 下的平移、SO(3) rotvec、绝对宽度，[30,32] 仅 0:7 有效。仅从真实 5 步边界提取完整 30 步窗口，无尾部复制、删帧或跨 reset 拼接。旧 V1 next-measured-state 标签和 action_stats 未使用。', '',
      '白名单参考 loader 仅输出当前双 RGB、实测 world-frame state、固定任务文本、action/mask。阶段、果实和桶真值只进入教师与诊断。状态坐标系的新训练方案不在本轮范围。', '',
      '## Gate A/B/C 实际结果', '',
      '| 实验 | 完成 | 严格成功 | 数据可接受 | 失败分布 |', '|---|---:|---:|---:|---|']
    for name,rs in [('Gate A 初版 H5',a0),('Gate A 限幅修正 H5',a),('Gate C / 运输30s',c1),('Gate C / 运输45s',c)]:
        lines.append(f"| {name} | {len(rs)} | {sum(r.get('strict',{}).get('strict_success',False) for r in rs)} | {sum(r['accepted'] for r in rs)} | {dict(Counter(r['reason'] for r in rs if not r['accepted']))} |")
    lines += ['', '开发集固定为 1000106、1000268、1000576、1000235、1000356、1000074；初始目标高度约 0.814–1.174 m、站位距离约 0.695–0.760 m、规划器 twig 接触计数 1–7。困难场景 1000268 的初版 H1 对照也发生抓取超时。开发集成功率不作为性能估计。', '',
              'Gate B：全部 6 条开发轨迹通过连续 30 Hz、N/N+1 对齐、native kwargs、controller effective target、SE(3) round-trip、绝对宽度、完整窗口和 mask、全视频解码及独立原事件观察器复核。另有三项 CPU 契约测试，以及真实视频 loader 和流式日志对齐检查。此为数据正确性验证，不声称完整物理轨迹可以确定性重放。', '',
              'Gate C 种子为 8100000–8100029，运行前用原规划器确认 reset 可行；第一批 30 个均可行，无按结果替换场景。第二轮复用同一清单，属于修正后的工程复验，不是未使用测试集，也不证明总体成功率达到 80%。', '']
    if c:
        phases=dict(reach=sum(any(p['phase']=='GRASP' for p in r.get('phases',[])) for r in c),held15=sum(any(f['held15_step'] is not None for f in r.get('strict',{}).get('fruit_chains',{}).values()) for r in c),detach=sum(any(f['first_detach_after_grasp_step'] is not None for f in r.get('strict',{}).get('fruit_chains',{}).values()) for r in c),drop=sum(any(p['phase']=='DROP' for p in r.get('phases',[])) for r in c),release=sum(bool(r.get('strict',{}).get('release_events')) for r in c),stable_bucket=sum(r.get('strict',{}).get('strict_success',False) for r in c))
        lines.append('第二轮阶段通过数：`'+json.dumps(phases)+'`。')
        lines += ['',f"合格轨迹的步数加权 IK 失败率 {rate(accepted,'ik_failure_rate'):.3%}，clipping 比例 {rate(accepted,'clipping_rate'):.3%}。"]
        if accepted:
            times=[r['sim_seconds'] for r in accepted]
            lines.append(f"合格轨迹仿真时间 P50/P95/max = {np.percentile(times,50):.2f}/{np.percentile(times,95):.2f}/{max(times):.2f} s。可用完整 H5 窗口共 {sum(r['windows'] for r in accepted)} 个。")
        lines += ['', '| 第二轮未接受 seed | 原因 |', '|---|---|']
        lines += [f"| {r['seed']} | {r['reason']} |" for r in c if not r['accepted']]
    lines += ['', '每条轨迹 result.json 保存阶段开始/结束步数和真实时间、失败原因、跟踪误差、IK/clipping、释放距离和速度。成功定义仍为原同果 held15 → detach → actual release → non-held → bucket60；释放距离/速度作为独立诊断。额外数据质量要求为 IK/clipping 各不超过5%、连续受影响步数不超过15，不改变 strict 成功定义。', '',
      '## 已知限制', '',
      '较难的枝条接触仍可能阻挡预抓或抓取，亦可能提前拉断果柄；部分机械臂姿态存在运输 IK 受限。打开夹爪时还可能出现横向释放冲量，导致果实落在桶沿。专家无搜索或恢复，失败会被保存并拒收。benchmark_assist 的双指接触/掌心体积抓持抽象继续使用，因此结果不能解释为纯摩擦抓取能力。实际运输较慢，任务通常需要数十秒。', '',
      '## 正式采集和管理', '']
    if progress:
        lines += [f"已启动独立 tmux 会话 `orchard-native-v2`，manager PID `{progress['pid']}`，并发上限 {progress['workers']}，池容量 {progress.get('worker_capacity',progress['workers'])}，当前在途 {progress.get('in_flight','未记录')}。报告生成时实际 attempts={progress['attempts']}，accepted={progress['accepted']}，状态={progress['status']}；目标为2000条 accepted，尚未声称整批完成或全量验收通过。", '', f'输出：`{collection}`。']
    else:lines += ['正式2000条采集尚未启动。启动入口会拒绝不满足 Gate A/B/C 或专家版本不匹配的运行。']
    lines += ['', '以下命令从 `/home/rosmontis/Projects/orchardbench` 执行：', '', '```bash', '# 启动 / 恢复（复用完成结果，保留中断前缀）', 'bash scripts/start_student_native_collection.sh', '# 进度', 'cat data/orchard_requested_v2_2000/progress.json', '# 日志', 'tail -f data/orchard_requested_v2_2000/manager.log', '# 平稳停止：结束已在执行的 episode 后退出', 'kill -TERM "$(cat data/orchard_requested_v2_2000/manager.pid)"', '# 结束后全量复核与新的 V2 统计', '.pixi/envs/default/bin/python scripts/summarize_student_native.py data/orchard_requested_v2_2000 --revalidate --stats', '```', '',
      '归一化文件 `action_stats_requested_v2.json` 明确标记为 unsplit_candidates；后续划分 train/val 后应仅用训练集重新计算，不可把候选集统计冒充训练集统计。', '',
      '源代码入口：`treesim/student_native_expert.py`、`treesim/orchard_command.py`、`scripts/collect_student_native.py`、`scripts/run_student_native_batch.py`；契约说明见 `docs/student_native_contract.md`。本地版本、冻结配置和原始结果位于本实验目录。']
    if (EXP/'worker_scaling.md').exists(): lines += ['', '本轮并发实测与运行设置更新见 [worker_scaling.md](worker_scaling.md)。']
    (EXP/'final_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
