# TRANSPORT-only expert retiming validation

**Decision: READY_FOR_HUMAN_ACCEPTANCE**

原 30 seeds 已全部完成：accepted **30/30**，grasped / detached / placed / 严格 VLA bucket success = **30 / 30 / 30 / 29**。Tracking oracle **3/4** 完整完成任务；所有 oracle 失败同样保留。

## 1. 最终配置与范围

```python
ArmMotionProfile(
    mode="velocity_accel",
    vmax_rad_s=0.5,
    amax_rad_s2=20,
    active_phases=("TRANSPORT",),
)
```

REACH / GRASP / PULL / DROP 均使用 legacy motion。未使用 `phase_vmax_rad_s`，没有 REACH=1.5 override。现有 limiter 已支持该配置，本轮仅改变实验调用和报告，`treesim/*.py` 与所有既有 `scripts/*.py` 的前后 SHA256 完全相同。没有修改 gripper、detach force、physics、success、watchdog、planner、action contract 或模型训练代码。

原 selection 的 SHA256：`eda4e291f55d467435e2a8a0dd257a536d675cba8c58a78a5a03e1bcbc88175d`。指定 artifacts 文件缺失，已从上一轮 log 中哈希一致的副本原样恢复；没有重新选 seed。重跑结果位于 [transport_only_validation](transport_only_validation/)，前次报告保留在该目录。smoke 的 8 个新结果直接计入 expanded 30，再运行原清单的其余 22 个；没有额外 expert preflight。

## 2. 四个问题

1. **Expert task viability：** 30/30 accepted，失败原因 {}。详见逐 seed 表及失败分析。
2. **TRANSPORT 高速降低：** 同一组 8 seeds 相比旧 TRANSPORT 1.5，either exceed 从 75.87% 降至 14.14%，下降 61.73 个百分点（相对 81.36%）；TCP P95 降低 58.81%。30 seeds 的 TRANSPORT either exceed 为 12.84%。
3. **当前 VLA 接口的路径可执行性：** 3 条 tracking replay 完成抓取、detach 和严格入桶，证明这些 retimed paths 可通过当前接口执行；1 条未完成的结果保留，不宣称所有 seed 均可在 20s 内完成。
4. **REACH mismatch：** 保持 legacy contact-sensitive approach，not retimed，instantaneous envelope mismatch retained；只记录，不用于拒绝 profile，也没有再次通过限速改变接触过程。

## 3. Fresh 8-seed smoke

| Cohort | attempted | accepted | detached | placed | VLA bucket |
| --- | --- | --- | --- | --- | --- |
| normal | 6 | 6 | 6 | 6 | 6 |
| stress | 2 | 2 | 2 | 2 | 2 |
| all | 8 | 8 | 8 | 8 | 8 |

Failure：`{'premature_detach': 0, 'incidental_detach_episodes': 0, 'incidental_detach_count': 0, 'branch_break': 0, 'timeout': 0, 'transport_timeout_fallback': 0, 'transport_stall_fallback_observed': 0}`。没有 normal systematic regression。与历史 TRANSPORT0.5 相比，前 TRANSPORT phase 帧序列一致的 seed 数为 7/8；normal 6/6 和 stress 1000268 的前 TRANSPORT 帧数一致。stress 1000074 的 TRANSPORT 起点由历史167帧变为本轮102帧（发生于 limiter 启用之前），历史 incidental_detach rejection 本轮未复现；这反映 contact/physics 敏感性，不能把其改善归因于尚未启用的 TRANSPORT limiter。细小浮点差异保留在 JSON。

## 4. Tracking oracle

使用未修改的 `treesim/orchard_action.py` encode_window / decode_targets / native_command，以及 `OrchardVLAEnv.step()`；30-target chunk anchor 不变，无模型 inference。position error ≤0.01m 且 SO(3) error ≤0.08rad 后才推进。保留现有 600 step / 20s 预算。success 即当前严格 VLA bucket predicate；任务成功会提前终止，因此不要求消费最后的示教静止 targets。

| Seed | success / strict bucket | grasped / detached | sim s | target index / N | IK failed | clipped | timeout |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000106 | 否 | 是 / 是 | 20.000 | 218 / 251 | 0 | 283 | 是 |
| 1000235 | 是 | 是 / 是 | 16.900 | 195 / 216 | 0 | 130 | 否 |
| 1000356 | 是 | 是 / 是 | 16.367 | 172 / 197 | 0 | 171 | 否 |
| 1000576 | 是 | 是 / 是 | 18.867 | 177 / 203 | 0 | 323 | 否 |

1000106 在20s时已 grasp/detach，0 IK failure，刚推进到示教 DROP 开始的 target 218/251，尚未满足 bucket predicate。该失败说明既定预算仍有边界，不归为成功。1000576 完整成功后，按原清单顺序补测1000235、1000356；不是改变 limiter 参数或接口后重试。原始每一步进度和误差均保存于 [oracle](transport_only_validation/oracle/)。本轮不额外运行 time-indexed oracle；历史 time-indexed 结果不作为本 profile 的通过证据或拒绝条件。

## 5. TRANSPORT dynamics 与旧 profile 对照

30Hz 实测有限差分；TCP 用平移速度模长，angular 用 SO(3)，joint 为每步关节绝对速度最大值。保留 phase 交界峰值，不平滑。下表旧1.5对照仅有原8 seeds，因此降低比例只在这8个 matched seeds 上计算，30-seed结果独立报告。

| Cohort | TCP m/s P50/P95/max | Angular rad/s P50/P95/max | Joint rad/s P50/P95/max | translation | rotation | either |
| --- | --- | --- | --- | --- | --- | --- |
| old TRANSPORT1.5 (8) | 0.943 / 1.896 / 2.441 | 2.667 / 4.733 / 6.319 | 1.507 / 1.810 / 3.429 | 66.76% | 68.63% | 75.87% |
| TRANSPORT0.5 fresh smoke (8) | 0.342 / 0.781 / 0.888 | 1.068 / 1.729 / 2.826 | 0.506 / 0.579 / 2.582 | 4.26% | 13.02% | 14.14% |
| TRANSPORT0.5 expanded (30) | 0.362 / 0.729 / 0.888 | 1.053 / 1.684 / 2.826 | 0.505 / 0.525 / 2.582 | 1.98% | 12.24% | 12.84% |

Envelope exceed 按每个实际30Hz interval：世界 XYZ 位移任一轴 >0.02m，`Euler_xyz(R_next @ R_prev.T)` 任一轴 >0.05rad；either 为并集。统计按 interval 加权，沿用 interval-start phase 标签，不能等同于速度模长阈值，也不等同于 oracle clipping。

| 30-seed TRANSPORT metric | P50 | P95 | max |
| --- | --- | --- | --- |
| linear_acceleration (m/s²) | 0.625 | 2.281 | 5.243 |
| angular_acceleration (rad/s²) | 1.459 | 6.024 | 11.586 |
| joint_acceleration (rad/s²) | 1.061 | 4.430 | 10.096 |
| duration (s) | 3.425 | 4.133 | 4.183 |

## 6. Limiter 与 REACH

TRANSPORT command saturation = 99.17%（6597/6652 steps，至少一个关节达到/超过0.5）；landing crossings = 1056 个 joint events；acceleration exceptions = 380 个 joint-step events；unexplained = 0。所有 acceleration exceptions 均有对应 landing 标记。command internal acceleration max = 162.000 rad/s²，不能把 amax20 解释成 measured motion 或 crossing guard 的绝对保证。进入 TRANSPORT 沿用既有 q/qd state，短暂 inherited velocity 与 landing exception 保留；没有为抹平瞬时峰值修改 limiter。

**REACH：legacy contact-sensitive approach；not retimed；instantaneous envelope mismatch retained。**

| Cohort | TCP m/s P50/P95/max | Angular rad/s P50/P95/max | translation | rotation | either |
| --- | --- | --- | --- | --- | --- |
| smoke (8) | 0.893 / 1.983 / 2.307 | 1.015 / 2.184 / 2.456 | 61.04% | 23.38% | 74.03% |
| expanded (30) | 1.015 / 2.140 / 2.819 | 1.037 / 2.401 / 3.065 | 64.54% | 21.99% | 74.11% |

**TRANSPORT：free-space transport；retimed for student executability。** 两阶段分别解释，没有合并为“全 trajectory 必须严格满足 native envelope”的验收条件。

## 7. 30-seed task 与 failure

Task：`{'attempted': 30, 'accepted': 30, 'grasped': 30, 'detached': 30, 'placed': 30, 'vla_bucket_success': 29, 'success': 30, 'failure_reasons': {}}`。

Failure observations：`{'premature_detach': 0, 'incidental_detach_episodes': 0, 'incidental_detach_count': 0, 'branch_break': 0, 'timeout': 0, 'transport_timeout_fallback': 0, 'transport_stall_fallback_observed': 0}`。stall fallback 按末尾 TRANSPORT sample 达到现有110-frame阈值记录；保留逐帧 stall trace，不改触发逻辑。

| Seed | cohort | accepted | grasped | detached | placed | VLA bucket | failure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000106 | normal | 是 | 是 | 是 | 是 | 是 | — |
| 1000268 | stress | 是 | 是 | 是 | 是 | 是 | — |
| 1000576 | normal | 是 | 是 | 是 | 是 | 是 | — |
| 1000235 | normal | 是 | 是 | 是 | 是 | 是 | — |
| 1000356 | normal | 是 | 是 | 是 | 是 | 是 | — |
| 1000074 | stress | 是 | 是 | 是 | 是 | 是 | — |
| 1000232 | normal | 是 | 是 | 是 | 是 | 是 | — |
| 1000441 | normal | 是 | 是 | 是 | 是 | 是 | — |
| 1000436 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000287 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000402 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000408 | unclassified | 是 | 是 | 是 | 是 | 否 | — |
| 1000512 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000360 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000386 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000430 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000214 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000431 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000445 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000508 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000339 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000345 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000455 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000291 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000319 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000394 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000569 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000385 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000389 | unclassified | 是 | 是 | 是 | 是 | 是 | — |
| 1000429 | unclassified | 是 | 是 | 是 | 是 | 是 | — |

原8-seed中的 normal/stress 标签不变；原 expanded 其余22个没有该标签，保留 unclassified。没有按运行成败重分类。

30/30 没有 expert 拒绝样本，因此本轮未观察到 TRANSPORT limiter 新引入的任务失败。历史 stress incidental detach 的偶发性不作为重新调速的依据。

严格 VLA bucket 未成功：seed 1000408，collector accepted/placed=True，但30Hz与60Hz记录均没有满足严格 predicate。它完整经历 REACH→GRASP→PULL→TRANSPORT→DROP→DONE，未发生 branch break、异常 detach、timeout 或 stall fallback。collector placement 范围与 VLA strict bucket 原本不同；本次记录没有保存逐帧 apple 坐标，无法进一步定位是哪条几何边界未满足，也没有 matched 旧profile strict-bucket证据证明或排除其由 retiming 引起。该例按29/30如实计入，单例未建立系统性 regression；不修改 bucket success，不启动额外实验。

## 8. Phase durations 与停止点

| Seed | REACH s | GRASP s | PULL s | TRANSPORT s | DROP s | premature | incidental | branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000106 | 0.583 | 0.517 | 2.033 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000268 | 0.217 | 1.717 | 1.833 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000576 | 0.650 | 0.333 | 1.400 | 3.267 | 1.000 | 否 | 0 | 0 |
| 1000235 | 0.200 | 0.267 | 1.500 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000356 | 0.200 | 0.267 | 1.617 | 3.383 | 1.000 | 否 | 0 | 0 |
| 1000074 | 0.150 | 1.000 | 0.550 | 3.183 | 1.000 | 否 | 0 | 0 |
| 1000232 | 0.233 | 0.250 | 1.550 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000441 | 0.250 | 0.250 | 1.500 | 3.317 | 1.000 | 否 | 0 | 0 |
| 1000436 | 0.217 | 0.250 | 1.467 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000287 | 0.733 | 0.250 | 1.500 | 3.167 | 1.000 | 否 | 0 | 0 |
| 1000402 | 0.217 | 0.250 | 1.683 | 3.367 | 1.000 | 否 | 0 | 0 |
| 1000408 | 0.650 | 0.333 | 1.683 | 3.233 | 1.000 | 否 | 0 | 0 |
| 1000512 | 0.200 | 0.267 | 1.500 | 3.333 | 1.000 | 否 | 0 | 0 |
| 1000360 | 0.200 | 0.250 | 1.733 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000386 | 0.250 | 0.267 | 1.767 | 3.383 | 1.000 | 否 | 0 | 0 |
| 1000430 | 0.233 | 0.250 | 1.650 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000214 | 0.700 | 0.300 | 1.467 | 3.200 | 1.000 | 否 | 0 | 0 |
| 1000431 | 0.167 | 1.017 | 1.633 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000445 | 0.217 | 0.250 | 1.617 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000508 | 0.283 | 0.267 | 1.517 | 3.333 | 1.000 | 否 | 0 | 0 |
| 1000339 | 0.250 | 0.267 | 1.767 | 3.400 | 1.000 | 否 | 0 | 0 |
| 1000345 | 0.183 | 0.567 | 1.567 | 3.450 | 1.000 | 否 | 0 | 0 |
| 1000455 | 0.250 | 0.267 | 1.683 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000291 | 0.667 | 0.333 | 1.650 | 3.250 | 1.000 | 否 | 0 | 0 |
| 1000319 | 0.200 | 0.267 | 1.817 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000394 | 0.167 | 1.167 | 1.733 | 4.133 | 1.000 | 否 | 0 | 0 |
| 1000569 | 0.217 | 0.250 | 1.500 | 3.350 | 1.000 | 否 | 0 | 0 |
| 1000385 | 0.200 | 0.250 | 1.750 | 4.183 | 1.000 | 否 | 0 | 0 |
| 1000389 | 0.267 | 0.283 | 1.417 | 3.333 | 1.000 | 否 | 0 | 0 |
| 1000429 | 0.200 | 0.250 | 1.500 | 4.133 | 1.000 | 否 | 0 | 0 |

完整 phase dynamics、每 seed saturation / acceleration / landing 及 source hashes 见 [final_summary.json](final_summary.json) 和 [expanded_summary.json](transport_only_validation/expanded_summary.json)。所有成功/失败 run.json、stdout 和 oracle step traces 均保留。本轮已停止；未创建 V2 dataset、未开始训练或下一阶段。

READY_FOR_HUMAN_ACCEPTANCE
