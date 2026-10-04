# Gate 1 — frozen TRANSPORT retiming revalidation

沿用旧 Gate 1 的 30-seed cohort，每个 seed 仅生成一次 expert trajectory，无替换/重试/调参。
`ArmMotionProfile(mode="velocity_accel", vmax_rad_s=0.5, amax_rad_s2=20, active_phases=("TRANSPORT",))`；REACH / GRASP / PULL / DROP legacy。
直接调用原 check_gate1_oracle.replay，仅按用户追加要求将 CONFIG.max_control_steps 覆盖为 900 / 30 s；环境源码、adapter、IK、workspace、clipping、joint step、gripper、evaluator 与 advancement rule 完全不变。position 0.01 m、SO(3) 0.08 rad、maximum dwell 30。无额外 terminal hold。
已中止的 20 s 新轨迹实验保存在 ../gate1_transport_revalidation/，其结果不混入本轮；本轮 30 seeds 从头重新生成、重新 replay。旧 Gate 1 对照仍是 canonical V1 + 20 s。
轨迹仅存于本次 expert/ 目录；record_rgb=False 沿用 Gate 0 collector 调用，未生成视频、训练 dataset、split 或 normalization stats。
Expert accepted：30/30；replay：60/60。Rejected expert：无。若 rejected 轨迹仍可 replay，仅计 diagnostic，不冒充合法成功示教。

| mode | strict success old → new | grasp old → new | detach old → new | budget timeout old → new | TRANSPORT budget failure old → new |
| --- | --- | --- | --- | --- | --- |
| time-indexed | 0 → 0 | 3 → 3 | 3 → 3 | 0 → 0 | 0 → 0 |
| reach-conditioned | 17 → 26 | 30 → 30 | 29 → 30 | 11 → 4 | 8 → 4 |

| mode | dwell timeout old → new | forced advancement old → new | IK failed steps old → new | clipped steps old → new |
| --- | --- | --- | --- | --- |
| time-indexed | 0 → 0 | 0 → 0 | 779 → 2449 | 3656 → 5913 |
| reach-conditioned | 79 → 67 | 79 → 67 | 1002 → 576 | 11316 → 8026 |

步数与暴露时间不同；结构化 summary 同时提供 IK/clipping step fraction 和受影响 episode 数。失败相位指最终正在执行的示教 target 相位，不是重新定义的在线 expert state。

| mode | old FAIL → new PASS | old PASS → new FAIL | unchanged | not comparable |
| --- | ---: | ---: | ---: | ---: |
| time-indexed | 0 | 0 | 30 | 0 |
| reach-conditioned | 9 | 0 | 21 | 0 |

| new paired outcome | count |
| --- | ---: |
| time PASS / reach PASS | 0 |
| time FAIL / reach PASS | 26 |
| time PASS / reach FAIL | 0 |
| time FAIL / reach FAIL | 4 |

- time-indexed final failure phase：{'DONE': 30} → {'DONE': 30}。
- time-indexed budget failure phase：{} → {}。
- time-indexed 不再为 TRANSPORT budget failure 的旧例（含是否真正成功）：[]；新增同类失败：[]。
- time-indexed 成功但发生 forced advancement：0；失败证据：{'grasp_failure': 27, 'target_sequence_exhausted': 30, 'IK_failure_observed': 30, 'clipping_observed': 30, 'tracking_lag_observed': 30, 'DROP_bucket_failure': 3}。

- reach-conditioned final failure phase：{'TRANSPORT': 8, 'DONE': 2, 'DROP': 1, 'PULL': 2} → {'TRANSPORT': 4}。
- reach-conditioned budget failure phase：{'TRANSPORT': 8, 'DROP': 1, 'PULL': 2} → {'TRANSPORT': 4}。
- reach-conditioned 不再为 TRANSPORT budget failure 的旧例（含是否真正成功）：[{'episode_id': 'episode_000050', 'seed': 1000136, 'new_success': True, 'new_termination': 'strict_bucket_success', 'new_final_phase': 'DROP'}, {'episode_id': 'episode_000080', 'seed': 1000214, 'new_success': True, 'new_termination': 'strict_bucket_success', 'new_final_phase': 'DROP'}, {'episode_id': 'episode_000090', 'seed': 1000235, 'new_success': True, 'new_termination': 'strict_bucket_success', 'new_final_phase': 'DROP'}, {'episode_id': 'episode_000140', 'seed': 1000371, 'new_success': True, 'new_termination': 'strict_bucket_success', 'new_final_phase': 'DROP'}, {'episode_id': 'episode_000150', 'seed': 1000394, 'new_success': True, 'new_termination': 'strict_bucket_success', 'new_final_phase': 'DROP'}, {'episode_id': 'episode_000270', 'seed': 1000702, 'new_success': True, 'new_termination': 'strict_bucket_success', 'new_final_phase': 'DROP'}]；新增同类失败：['episode_000180', 'episode_000260']。
- reach-conditioned 成功但发生 forced advancement：5；失败证据：{'DROP_bucket_failure': 4, 'episode_budget': 4, 'IK_failure_observed': 3, 'clipping_observed': 4, 'dwell_timeout': 4, 'tracking_lag_observed': 4}。

| episode / seed | time old → new | reach old → new | reach new failure evidence |
| --- | --- | --- | --- |
| episode_000010 / 1000025 | FAIL → FAIL | PASS → PASS | — |
| episode_000020 / 1000056 | FAIL → FAIL | FAIL → FAIL | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000030 / 1000079 | FAIL → FAIL | FAIL → PASS | — |
| episode_000040 / 1000106 | FAIL → FAIL | PASS → PASS | — |
| episode_000050 / 1000136 | FAIL → FAIL | FAIL → PASS | — |
| episode_000060 / 1000164 | FAIL → FAIL | PASS → PASS | — |
| episode_000070 / 1000192 | FAIL → FAIL | PASS → PASS | — |
| episode_000080 / 1000214 | FAIL → FAIL | FAIL → PASS | — |
| episode_000090 / 1000235 | FAIL → FAIL | FAIL → PASS | — |
| episode_000100 / 1000250 | FAIL → FAIL | PASS → PASS | — |
| episode_000110 / 1000274 | FAIL → FAIL | FAIL → PASS | — |
| episode_000120 / 1000294 | FAIL → FAIL | FAIL → FAIL | DROP_bucket_failure, episode_budget, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000130 / 1000345 | FAIL → FAIL | FAIL → PASS | — |
| episode_000140 / 1000371 | FAIL → FAIL | FAIL → PASS | — |
| episode_000150 / 1000394 | FAIL → FAIL | FAIL → PASS | — |
| episode_000160 / 1000420 | FAIL → FAIL | PASS → PASS | — |
| episode_000170 / 1000440 | FAIL → FAIL | PASS → PASS | — |
| episode_000180 / 1000465 | FAIL → FAIL | FAIL → FAIL | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000190 / 1000486 | FAIL → FAIL | PASS → PASS | — |
| episode_000200 / 1000510 | FAIL → FAIL | PASS → PASS | — |
| episode_000210 / 1000542 | FAIL → FAIL | PASS → PASS | — |
| episode_000220 / 1000572 | FAIL → FAIL | PASS → PASS | — |
| episode_000230 / 1000593 | FAIL → FAIL | PASS → PASS | — |
| episode_000240 / 1000626 | FAIL → FAIL | PASS → PASS | — |
| episode_000250 / 1000646 | FAIL → FAIL | PASS → PASS | — |
| episode_000260 / 1000676 | FAIL → FAIL | FAIL → FAIL | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000270 / 1000702 | FAIL → FAIL | FAIL → PASS | — |
| episode_000280 / 1000731 | FAIL → FAIL | PASS → PASS | — |
| episode_000290 / 1000750 | FAIL → FAIL | PASS → PASS | — |
| episode_000300 / 1000779 | FAIL → FAIL | PASS → PASS | — |

本轮 reach strict success 17/30 → 26/30（净变化 +9）；paired exact McNemar 双侧 p=0.003906。本轮显示 reach 整体成功率显著改善（双侧 p<0.05）。新旧比较同时改变了 expert TRANSPORT 配置与 replay budget（20→30 s），不能将全部改善单独归因于 retiming。未增加 fresh legacy 重采对照，接触动力学重复性也可能影响逐 seed 差异。 Gate 1 仍存在 execution blocker，停止，不训练或修 controller。

本轮按用户追加要求从串行改为最多3个独立 seed 进程并发；每 seed 仍按 expert → time → reach 顺序，已完成结果未重跑。调度记录见 parallel_scheduling.json。

**预算影响（仅分析既有日志，未追加实验）：** 新轨迹在前600步/20 s内成功 **18/30**；旧 canonical+20 s为 **17/30**。同预算日志对照为3条 FAIL→PASS、2条 PASS→FAIL、25条 unchanged，paired exact McNemar 双侧 p=1。另有 **8 条**成功发生于20–30 s。故26/30相对17/30的改善是 retiming+30 s 的组合结果，不能归为 retiming 单独的显著收益。详见 budget_prefix_diagnostic.json。

**剩余4条失败：** 全部已 grasp+detach，但30 s时仍在 TRANSPORT；其中3条有 IK failure，1条无 IK failure。

| seed | final target index / N | IK failed steps | clipped steps | dwell timeout / forced advancement |
| --- | --- | ---: | ---: | --- |
| 1000056 | 168 / 207 | 90 | 621 | 14 / 14 |
| 1000294 | 182 / 228 | 0 | 390 | 10 / 10 |
| 1000465 | 159 / 218 | 1 | 653 | 12 / 12 |
| 1000676 | 162 / 210 | 153 | 675 | 15 / 15 |

结论：**组合配置使 reach 成功率明显改善，但仍有 execution blocker；仅 retiming 的显著独立收益未被本轮证据证明。** 不训练、不修 controller，止于本次验证。

执行源码、旧 artifacts、旧 canonical 输入哈希未变；新旧 reset 与新 paired targets 已核验。原始证据见 expert/、episodes/；逐 seed 对照见 old_new_comparison.csv。本轮已停止。
