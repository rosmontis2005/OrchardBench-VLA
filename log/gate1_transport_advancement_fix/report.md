# Gate 1 TRANSPORT advancement 修复验证

最终判断：**FAIL**。固定 cohort strict success 26/30 → 26/30。未达成本轮消除四条失败、降低 dwell/forced advancement 的目标。

## 修改与约束

- 执行源码仅修改 `scripts/check_gate1_oracle.py`；新增聚焦测试 `scripts/test_gate1_transport_advancement.py`。实验 runner、report generator 和原始结果均在本目录。
- 原有 position ≤ 0.01 m 且 rotation ≤ 0.08 rad 的 reach 不变。普通 TRANSPORT waypoint 可由几何 passed 推进一步；每个 control step 最多一步。
- passed 要求 TCP 同时越过 incoming/outgoing 目标平面，outgoing 投影落在当前→下一个 target 段内，横向误差 ≤ 0.01 m，当前 target 旋转误差 ≤ 0.08 rad。
- 首尾各两个 TRANSPORT waypoint 不应用 passed；不跨 phase。退化段（<1 µm）、相邻段 >0.02 m、转弯 >60° 不应用 passed。阈值在首次复测前固定，此后未调整。
- 原有 maximum dwell=30 的异常 fallback 保留；边界点仍使用原严格 reach 判定，不增加宽松 pass。`skip_count` 继续只统计旧 timeout 强推，几何推进另列 `passed_waypoint_count`。
- 无 bounded IK look-ahead：四条失败在 TRANSPORT 的 IK failure 均为 0，不满足启用前提。
- 沿用原 30 条已 accepted expert 文件，SHA256 一致，没有重新采集。Gate 0 TRANSPORT profile、30 s/900 steps、环境、adapter、IK、action、evaluator、grasp/PULL/DROP 均不改。未运行 time-indexed，未创建训练 dataset 或启动训练。

## 原四条失败

| seed | strict old → new | final target old → new / total | dwell old → new | forced old → new | IK old → new | clipped old → new | passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000056 | FAIL → FAIL | 168 → 169 / 207 | 14 → 14 | 14 → 14 | 90 → 90 | 621 → 620 | 0 |
| 1000294 | FAIL → FAIL | 182 → 182 / 228 | 10 → 10 | 10 → 10 | 0 → 0 | 390 → 390 | 0 |
| 1000465 | FAIL → FAIL | 159 → 161 / 218 | 12 → 11 | 12 → 11 | 1 → 1 | 653 → 652 | 0 |
| 1000676 | FAIL → FAIL | 162 → 163 / 210 | 15 → 15 | 15 → 15 | 153 → 153 | 675 → 674 | 0 |

四条均 grasp/detach 成功，均在 TRANSPORT 达到 900 步 budget。target index 是日志中的 0-based index。

### 对共同根因假设的核验

| seed | 原进入 TRANSPORT 时刻 | 原 timeout 的 phase 分布 | 原 IK failure 的 phase 分布 | TRANSPORT 未 reach 但可 passed |
| --- | --- | --- | --- | --- |
| 1000056 | 19.833 s | {'REACH': 11, 'GRASP': 3} | {'GRASP': 90} | 0 |
| 1000294 | 18.867 s | {'GRASP': 10} | {} | 0 |
| 1000465 | 20.667 s | {'GRASP': 11, 'PULL': 1} | {'GRASP': 1} | 0 |
| 1000676 | 20.667 s | {'REACH': 10, 'GRASP': 2, 'PULL': 3} | {'GRASP': 63, 'PULL': 90} | 0 |

原日志中四条 TRANSPORT dwell timeout=0、IK failure=0；普通 waypoint 多在 2–5 步 reach，未达标的 TCP 沿前进方向仍在 target 后方。原总体 dwell/IK 统计不能归因为最终失败 phase。所有四条未达标 TRANSPORT 状态的 outgoing progress 也没有出现正向越过。
因此“TRANSPORT 反复等待 maximum dwell”及“这两条在 TRANSPORT 连续 IK 不可解”的假设均不受现有日志支持。进入 TRANSPORT 时已经消耗约 19–21 s，剩余预算未能容纳逐点跟踪到 DROP。进一步改变前置阶段或使 TCP 尚未越过 target 就前进，不属于本轮授权的几何 passed 修复。

## 固定 30-seed 回归

先完成四条并检查无不合理跳点，再执行其余 26 条以核验回退；完整结果包含相同源码/协议下的四条首轮复测，合计每 seed 一次。其余 26 条使用三个独立环境进程调度，没有增加样本或重试调参。

| metric | old | new |
| --- | ---: | ---: |
| success | 26 | 26 |
| grasped | 30 | 30 |
| detached | 30 | 30 |
| timeout | 4 | 4 |
| control_steps | 17966 | 17969 |
| dwell_timeout_count | 67 | 66 |
| skip_count | 67 | 66 |
| IK_failed_steps | 576 | 576 |
| clipped_steps | 8026 | 8016 |
| passed_waypoint_count | 0 | 0 |

原 PASS 回退 seeds：无；原 FAIL 改善 seeds：无。
几何 passed 事件共 0；检查未发现错误 passed。事件若为 0，仅表示本 cohort 未触发新分支，不能声称已从仿真证明该分支改善执行。

| phase | dwell old → new | forced old → new | IK old → new | clipped old → new |
| --- | --- | --- | --- | --- |
| REACH | 21 → 21 | 21 → 21 | 0 → 0 | 1831 → 1829 |
| GRASP | 37 → 37 | 37 → 37 | 336 → 336 | 1597 → 1623 |
| PULL | 9 → 8 | 9 → 8 | 240 → 240 | 761 → 734 |
| TRANSPORT | 0 → 0 | 0 → 0 | 0 → 0 | 3715 → 3706 |
| DROP | 0 → 0 | 0 → 0 | 0 → 0 | 122 → 124 |
| DONE | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |

逐 seed 完整对照见 `summary.json` 的 `per_seed`；原始 TCP、target、误差、IK、clipping、progress 和推进原因见 `episodes/*.json`。
个别步数/末端数值与旧结果存在差异，部分差异在进入 TRANSPORT 前已出现，不能归因于几何 passed；未通过重复试验选取有利结果。

## 验证及复现

- 四项 unittest 覆盖合法越过、后方/偏离/越界/旋转错误、phase 首尾保护、退化/急转/大间隔拒绝。
- 核验相同 expert SHA、reset pose/info、源码 SHA、900 步预算、target index 每步只增 0 或 1，以及实际 passed 的所有几何约束。
- `.pixi/envs/default/bin/python scripts/test_gate1_transport_advancement.py`
- `.pixi/envs/default/bin/python log/gate1_transport_advancement_fix/run_experiment.py four`
- `.pixi/envs/default/bin/python log/gate1_transport_advancement_fix/run_experiment.py regression`
- `.pixi/envs/default/bin/python log/gate1_transport_advancement_fix/write_report.py`

到此停止。本轮未改善共同失败，不能以实现完成代替 Gate 1 PASS，也不继续扩大阈值、budget 或修改禁止范围。
