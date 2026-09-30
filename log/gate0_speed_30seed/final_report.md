# Gate 0 / AutoPicker 调速收尾报告

**Decision: BLOCKED_BEFORE_V2。未完成 30-seed 验收，未获得可用于采集的最终 profile。**
Gate 0 最小修补已完成；TRANSPORT 限速收敛到可进一步验证的范围，但加入 REACH 限速后出现系统性普通 seed 抓取回归。一次有限修正后仍为 normal 2/6、stress 0/2。遵守任务的 smoke 扩展前提，未启动剩余 22 seeds；这不是已通过或已完成的 30-seed validation。

当前 HEAD：`36bf3659dcba3820901f21c26ae52a641725f164`。直接使用该 HEAD 的工作树；没有回退、重建框架、改写 v1 数据或开始 V2 / 训练。修改后源码 SHA256 在 [final_summary.json](final_summary.json) 的 `source_hashes`，最后候选运行源快照在 `reach1.5_transport0.5_a20_source_snapshot.json`，已与当前运行源码核对一致。

## 1. Gate 0 实际修改

- `VLAEnvConfig.detach_force_scale=1.5`；reset 显式传入 TreeConfig，info 暴露值。四次真实 oracle reset 都断言 AppleField multiplier=1.5，且实际断裂阈值等于原始阈值×1.5。未修改果梗模型。
- continuous width 增加 1 mm closing deadband；持有时须连续 5 个 control step 达到 75 mm，才转为 opening/release。物理 finger target 仍是原始连续 width；离散 OPEN 仍立即释放。
- VLA 默认从 300 增为 600 control steps（30 Hz 下 20 s），与 collector 20 s 一致。没有修改 REACH、GRASP、PULL、TRANSPORT 的 phase timeout/watchdog。
- collector CLI 接入 mode/vmax/amax/active phases 和 phase vmax override；写入 collection_config、episode metadata 和结果；resume / pilot 配置检查包含 profile。旧 config 缺少 profile 按 legacy 处理，legacy 默认仍可用。

## 2. Gripper width audit / regression

只读审计已有 v1 成功 episode，先检查指定 8 个 smoke，再对 300 条已有成功轨迹校准持续开启阈值。逐 phase 的 min/max、上升量与路径见 [gripper_audit.json](gripper_audit.json)，未重新做大规模动力学 preflight。
8 个 smoke 的 hold 段最大单步上升为 29.47 mm，明显超过旧 1 μm 判定。300 条 hold 序列中，width≥75 mm 最长连续 4 帧，因此采用 5 帧（0.167 s）开启确认。
300/300 原始 PULL→TRANSPORT 持有序列不释放，接入渐进明确开启序列后 300/300 释放。阈值处理的是当前数据中的全开意图：低于 75 mm 的连续宽度仍改变物理指间距，但不会仅因小幅回升就释放 assist。
针对真实碰撞的 Gate 0 B 单项回归通过：双指接触→attach→hold→前四个 near-open command 仍 hold→第五个 release。旧 B 的“单帧 .08 立即 release”断言按新 hysteresis 更新；首次旧断言失败日志保留。A/C 未重跑。limiter 8 项回归通过（含 phase override 不重置速度状态），源码编译及 diff whitespace 检查通过。

## 3. Placement geometry / success

collector：bucket 中心相同，但 XY 半宽 0.17 m、Z ∈ (0.17,0.77) m。VLA：XY 半宽 0.15 m、Z ∈ (0.31,0.46) m，且必须 detached。两者确有范围差异，Z 也不只是几毫米 tolerance。本轮不改 success 或 predicate；新 expert worker 对真实 apple pose 每帧只读记录 VLA predicate，汇总只使用偶数 physics frame（30 Hz）。
TRANSPORT-only 0.5 的 8/8 曾进入严格 VLA bucket，accepted 7/8；不能用严格入桶掩盖 incidental detach 的 collector rejection。旧1.5日志没有 apple bucket pose/flag，VLA-equivalent 标记为未记录，不能推成 false。

## 4. Profile 与有限探索

最后测试（已拒绝，不是采集推荐）：

```python
ArmMotionProfile(mode="velocity_accel", vmax_rad_s=0.5, amax_rad_s2=20,
                 active_phases=("REACH", "TRANSPORT"),
                 phase_vmax_rad_s={"REACH": 1.5})
```

保留的局部结果是 `transport_va0.5_a20`，仅 TRANSPORT 生效；它没有解决 REACH，因此不能冒充最终配置。
顺序：旧 TRANSPORT1.5 作为已有对照→TRANSPORT1.0→依据仍普遍超限减到0.5→固定 TRANSPORT0.5，仅加入 REACH0.5→一次保守修正为 REACH1.5。amax 始终20，没有 global retime、网格扫描或新规划器。phase override 仅给已有 limiter 的 vmax 传值，不重置 q/qd，也不影响 GRASP/PULL。

## 5. 8-seed smoke（所有失败保留）

| Profile | normal | stress | all accepted | grasped/detached/placed | 严格 VLA 入桶 | 主要拒绝 |
| --- | --- | --- | --- | --- | --- | --- |
| transport_va1.0_a20 | 6/6 | 1/2 | 7/8 | 8/8/8 | 8 | {'premature_detach': 1} |
| transport_va0.5_a20 | 6/6 | 1/2 | 7/8 | 8/8/8 | 8 | {'incidental_detach': 1} |
| reach_transport_va0.5_a20 | 1/6 | 0/2 | 1/8 | 2/2/2 | 2 | {'incidental_detach': 1, 'first_attempt_no_fruit_at_detection': 5, 'first_attempt_reach_stalled': 1} |
| reach1.5_transport0.5_a20 | 2/6 | 0/2 | 2/8 | 2/2/2 | 2 | {'branch_break': 1, 'first_attempt_no_fruit_at_detection': 5} |

normal/stress 沿用原 smoke protocol 与 case_overrides：1000074、1000268 为 stress；不按本次结果重新分类。其余22 expanded seeds 原选择文件没有标签，保留 unclassified。
REACH0.5 的主要问题不是单纯预算：5 次 no fruit、1 次 reach stall；REACH1.5 仍有5次 no fruit和1次stress branch break。1000235 的 REACH→GRASP 从 legacy 的第12帧变为第19帧，随后第55帧 no fruit；GRASP/PULL 代码未改。3 个最终 normal 失败还记录了提前 detach，说明实际 fruit interaction 发生回归，不能通过延长 watchdog 修好。

## 6. Oracle execution check

从最后候选的两个成功普通 seed 使用真实专家轨迹。通过 `encode_window()` / `decode_targets()` 固定每个30-target chunk的示教anchor，再调用 `native_command()` 与 `OrchardVLAEnv.step()`。A每步推进示教索引；B在 position error≤0.01m 且 SO(3) error≤0.08rad 后推进。未使用模型、关节直接回放或额外控制器。末尾重复最后target，仍受20s硬预算约束。

| Seed | Oracle | success | grasped/detached | sim s | target index / N | IK失败step | clipped step | timeout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000106 | time | 否 | 否/否 | 20.000 | 255/256 | 449 | 577 | 是 |
| 1000106 | tracking | 是 | 是/是 | 19.933 | 235/256 | 0 | 255 | 否 |
| 1000576 | time | 否 | 否/否 | 20.000 | 213/214 | 473 | 597 | 是 |
| 1000576 | tracking | 是 | 是/是 | 19.200 | 188/214 | 0 | 303 | 否 |

结论：两例均 A失败/B成功，归类为 execution timing mismatch。A在REACH已经累积大幅 tracking lag，后续TRANSPORT出现大量IK失败；B完整获得contact hold、detach、release及严格入桶，0 IK failure、0 branch break。force=1.5、现有bucket detector与本轮hold/release在这两例B中可工作。B分别用19.93s、19.20s，剩余预算很小；没有据此继续延长预算或盲降速度。oracle只覆盖两个成功示教，不代表修复了8-seed专家失败。

## 7. 固定30 expanded seeds 的实际覆盖

原 selection 文件SHA256：`eda4e291f55d467435e2a8a0dd257a536d675cba8c58a78a5a03e1bcbc88175d`。没有重选seed。以下8个是最后候选smoke覆盖；剩余22个未运行。**完整30-seed结果尚不存在，不能进行通过性人工验收。**

| Seed | designation | 状态 | accepted | grasped | detached | placed | VLA-equivalent | failure / 未运行原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000106 | normal | RUN_AS_SMOKE | 是 | 是 | 是 | 是 | 是 | — |
| 1000268 | stress | RUN_AS_SMOKE | 否 | 否 | 否 | 否 | 否 | first_attempt_no_fruit_at_detection |
| 1000576 | normal | RUN_AS_SMOKE | 是 | 是 | 是 | 是 | 是 | — |
| 1000235 | normal | RUN_AS_SMOKE | 否 | 否 | 否 | 否 | 否 | first_attempt_no_fruit_at_detection |
| 1000356 | normal | RUN_AS_SMOKE | 否 | 否 | 否 | 否 | 否 | first_attempt_no_fruit_at_detection |
| 1000074 | stress | RUN_AS_SMOKE | 否 | 否 | 否 | 否 | 否 | branch_break |
| 1000232 | normal | RUN_AS_SMOKE | 否 | 否 | 否 | 否 | 否 | first_attempt_no_fruit_at_detection |
| 1000441 | normal | RUN_AS_SMOKE | 否 | 否 | 否 | 否 | 否 | first_attempt_no_fruit_at_detection |
| 1000436 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000287 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000402 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000408 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000512 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000360 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000386 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000430 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000214 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000431 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000445 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000508 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000339 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000345 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000455 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000291 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000319 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000394 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000569 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000385 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000389 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |
| 1000429 | unclassified | NOT_RUN | — | — | — | — | — | smoke prerequisite failed |

已运行8/30，accepted2、rejected6、未运行22。已标注normal2/6、stress0/2；unclassified0/22执行。不能将未运行22个当作失败或成功。

## 8. Dynamics 与 native envelope 方法

复用现有 motion/aggregate/transition/scoped diagnostics。30Hz实测位置有限差分；angular speed用SO(3)角度/时间，joint speed为每步max|joint FD speed|；linear/angular/joint acceleration按速度有限差分。phase标签沿用interval-start，跨phase瞬时峰值保留，不平滑掩盖。
translation exceed：连续实际目标世界坐标位移任一轴abs>0.02m；rotation exceed：`Euler_xyz(R_next @ R_prev.T)` 任一分量abs>0.05rad。either是二者并集。按phase逐interval汇总，不把速度模长≤0.6m/s当作条件，也不把整段chunk相对anchor累计位移与单步阈值比较。oracle另报实际native命令clipping，包含tracking lag导致的更大需求。

| Profile | Phase | TCP m/s P50/P95/max | Angular rad/s P50/P95/max | translation | rotation | either | intervals |
| --- | --- | --- | --- | --- | --- | --- | --- |
| old transport1.5 | REACH | 0.893 / 1.983 / 2.307 | 1.015 / 2.184 / 2.456 | 61.04% | 23.38% | 74.03% | 77 |
| old transport1.5 | TRANSPORT | 0.943 / 1.896 / 2.441 | 2.667 / 4.733 / 6.319 | 66.76% | 68.63% | 75.87% | 373 |
| transport_va1.0_a20 | REACH | 0.893 / 1.983 / 2.307 | 1.015 / 2.184 / 2.456 | 61.04% | 23.38% | 74.03% | 77 |
| transport_va1.0_a20 | TRANSPORT | 0.661 / 1.367 / 1.788 | 2.062 / 3.601 / 4.370 | 47.94% | 58.94% | 62.67% | 509 |
| transport_va0.5_a20 | REACH | 0.893 / 1.983 / 2.307 | 1.015 / 2.184 / 2.456 | 61.04% | 23.38% | 74.03% | 77 |
| transport_va0.5_a20 | TRANSPORT | 0.341 / 0.741 / 0.823 | 1.068 / 1.715 / 1.860 | 1.68% | 11.96% | 12.74% | 895 |
| reach_transport_va0.5_a20 | REACH | 0.214 / 0.541 / 0.551 | 0.180 / 0.537 / 0.579 | 0.00% | 0.00% | 0.00% | 269 |
| reach_transport_va0.5_a20 | TRANSPORT | 0.504 / 0.781 / 0.808 | 1.097 / 1.759 / 1.774 | 7.35% | 28.43% | 30.88% | 204 |
| reach1.5_transport0.5_a20 | REACH | 0.632 / 1.438 / 1.554 | 0.464 / 1.480 / 1.618 | 44.44% | 2.38% | 44.44% | 126 |
| reach1.5_transport0.5_a20 | TRANSPORT | 0.327 / 0.601 / 0.666 | 1.152 / 1.615 / 1.722 | 0.00% | 12.61% | 12.61% | 222 |

最后候选TRANSPORT只有两个成功episode贡献数据，存在选择偏差；不能把其低超限率当成8/8正常。更可靠的TRANSPORT单独0.5证据覆盖8例完整transport，其either从旧75.87%降到12.74%。最终REACH仍44.44% translation超限，同时任务失败，不能宣称已形成合理全链示教。

### 最后候选各phase完整动力学

| Phase | TCP P50/P95/max m/s | Angular P50/P95/max rad/s | Joint P50/P95/max rad/s | Linear acc P50/P95/max m/s² | Joint acc P50/P95/max rad/s² |
| --- | --- | --- | --- | --- | --- |
| REACH | 0.632 / 1.438 / 1.554 | 0.464 / 1.480 / 1.618 | 1.384 / 1.537 / 1.570 | 2.929 / 8.138 / 8.359 | 4.508 / 8.758 / 8.936 |
| GRASP | 0.488 / 0.778 / 0.916 | 0.724 / 1.831 / 1.980 | 1.854 / 2.373 / 2.475 | 4.273 / 10.496 / 11.498 | 11.352 / 25.568 / 34.091 |
| PULL | 0.122 / 0.281 / 0.645 | 0.057 / 0.680 / 0.909 | 0.386 / 0.631 / 2.023 | 1.487 / 6.488 / 10.133 | 4.591 / 15.408 / 37.163 |
| TRANSPORT | 0.327 / 0.601 / 0.666 | 1.152 / 1.615 / 1.722 | 0.504 / 0.552 / 0.738 | 0.549 / 1.946 / 4.447 | 1.083 / 4.395 / 10.096 |
| DROP | 0.007 / 0.267 / 0.488 | 0.044 / 1.067 / 1.398 | 0.046 / 1.429 / 1.896 | 0.097 / 2.955 / 5.712 | 0.739 / 12.197 / 15.751 |

### 最后候选时长与失败

| Seed | episode s | REACH s | GRASP s | PULL s | TRANSPORT s | DROP s | premature | incidental | branch | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000074 | 1.200 | 0.250 | 0.950 | — | — | — | — | 0 | 1 | branch_break |
| 1000106 | 8.500 | 0.983 | 0.283 | 2.033 | 4.133 | 1.000 | 否 | 0 | 0 | accepted |
| 1000232 | 1.017 | 0.383 | 0.633 | — | — | — | 是 | 0 | 0 | first_attempt_no_fruit_at_detection |
| 1000235 | 0.917 | 0.317 | 0.600 | — | — | — | 是 | 0 | 0 | first_attempt_no_fruit_at_detection |
| 1000268 | 1.100 | 0.333 | 0.767 | — | — | — | — | 0 | 0 | first_attempt_no_fruit_at_detection |
| 1000356 | 0.900 | 0.317 | 0.583 | — | — | — | — | 0 | 0 | first_attempt_no_fruit_at_detection |
| 1000441 | 1.083 | 0.417 | 0.667 | — | — | — | 是 | 0 | 0 | first_attempt_no_fruit_at_detection |
| 1000576 | 7.100 | 1.083 | 0.283 | 1.400 | 3.250 | 1.000 | 否 | 0 | 0 | accepted |

### PULL→TRANSPORT transition

事件窗口沿用现有分析：转场前最后5个30Hz间隔与转场后首5个完整间隔，另保存-10..+20对齐序列、q_goal jump与command acceleration。下表是各episode窗口均值再跨episode汇总；详见JSON的transition_metrics及逐seed CSV。

| Profile | 完成转场N | TCP pre/post m/s | Angular pre/post rad/s | Joint pre/post rad/s | Linear acc pre/post m/s² |
| --- | --- | --- | --- | --- | --- |
| old transport1.5 | 8 | 0.147 / 0.652 | 0.212 / 1.277 | 0.541 / 0.708 | 1.481 / 7.250 |
| transport_va1.0_a20 | 8 | 0.155 / 0.548 | 0.299 / 1.160 | 0.695 / 0.700 | 2.096 / 5.587 |
| transport_va0.5_a20 | 8 | 0.137 / 0.305 | 0.069 / 0.603 | 0.414 / 0.404 | 1.595 / 3.151 |
| reach_transport_va0.5_a20 | 2 | 0.140 / 0.302 | 0.077 / 0.521 | 0.404 / 0.302 | 1.563 / 3.515 |
| reach1.5_transport0.5_a20 | 2 | 0.135 / 0.329 | 0.064 / 0.709 | 0.425 / 0.562 | 1.471 / 3.131 |

## 9. Timeout、detach、branch及limiter exceptions

| Profile | timeout | stall失败 | premature detach | incidental detach | branch break | TRANSPORT timeout fallback | stall fallback possible | landing crossings | acc exceptions | 无landing解释exceptions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transport_va1.0_a20 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 196 | 85 | 0 |
| transport_va0.5_a20 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 291 | 110 | 0 |
| reach_transport_va0.5_a20 | 0 | 1 | 3 | 1 | 0 | 0 | 0 | 249 | 33 | 0 |
| reach1.5_transport0.5_a20 | 0 | 0 | 3 | 0 | 1 | 0 | 0 | 185 | 39 | 0 |

提前detach diagnostics与collector第一条reject_reason是不同维度：即使reject_reason为no fruit，目标仍可能已在GRASP物理碰撞中提前脱落；表中没有漏掉这类事件。`detached`列沿用collector pick事件标记，未被grasp的提前脱落单独在premature列和原始detach_diagnostics中保留。
limiter landing guard仍优先不越过任意新goal，因此允许记录加速度例外；本轮scope内异常全部伴随landing，无无法解释的例外。进入active phase时继承速度，短暂高于新vmax也不隐式清零。phase duration/stall counters、每seed加速度诊断、失败轨迹均在summary和runs目录。

## 10. CLI复现与保存位置

以下仅是参数复现方式，不是批准采集：

```sh
# 最后被拒绝的候选
--detach-force-scale 1.5 \
--arm-motion-mode velocity_accel --arm-vmax 0.5 --arm-amax 20 \
--arm-active-phases REACH TRANSPORT --arm-phase-vmax '{"REACH":1.5}'

# 仅TRANSPORT的局部可行候选
--detach-force-scale 1.5 \
--arm-motion-mode velocity_accel --arm-vmax 0.5 --arm-amax 20 \
--arm-active-phases TRANSPORT
```

传给现有 `scripts/collect_autopicker_dataset.py`；其原有数据格式pilot审核仍有效。默认不传profile参数为legacy。所有本轮实验仅写 `log/gate0_speed_30seed` 的诊断结果，没有调用正式采集创建数据集。
每轮原始成功和失败保存在 `runs/<profile>/<seed>/run.json` 与 `stdout.log`，包含trajectory和60Hz command trace；`<profile>_summary.json` / `_seeds.csv`含全部phase指标、episode duration、failure、watchdog与limiter诊断。oracle逐step证据在 `oracle/<seed>_<time|tracking>.json`。

## 11. 停止判断 / 剩余blocker

**尚未达到“可以完成30-seed人工验收并准备V2 collection”的状态。** Gate 0修补与TRANSPORT局部降速结果可人工审阅，但不足以批准数据采集。
1. REACH限速改变fruit interaction；最后候选normal仍4/6失败，有提前脱落证据，不能用旧watchdog太短解释。
2. 更保守的REACH1.5仍有44.44% translation超限；TRANSPORT-only0.5虽normal6/6，却保留了原REACH高速。
3. A/B oracle表明执行时序lag，tracking成功已接近20s预算。需承认此限制，不能据两例成功声称全链可按示教时间执行。
4. 遵守“明显smoke失败不进入30seeds”前提，剩余22未运行；没有完整30-seed结果。

停止于此，不再追加候选链、修改抓取/物理/成功定义，未创建V2、300-episode批量采集、training stats或训练。`BLOCKED_BEFORE_V2`只是本轮阻塞状态，不是下一阶段许可。
