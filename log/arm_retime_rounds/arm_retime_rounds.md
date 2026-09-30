# Arm retime iterative repair test

**Decision: GO_TO_30_SEED_VALIDATION**

## 1. Objective

复用现有 arm limiter 做有限、逐轮的工程测试；Round 1 系统性接触失败后增加可选 TRANSPORT scope：在保持摘果任务成功的前提下，降低 TRANSPORT 动态与 student action envelope 的差距。没有采集完整数据集、训练、修改 action/observation semantics、tree physics、grasp/hold mechanics、成功定义或 fixed-base 工作位策略。

## 2. Existing diagnosis

已读取原 speed_audit.md、summary.json、episode_phase_statistics.csv、episode_statistics.csv，以及上一轮 arm_retime_pilot.md、failure_summary.json、final_summary.txt 和 runner/limiter/analysis 源码。300 accepted episodes 的 TRANSPORT TCP P50/P95/max 为 1.414/2.719/4.249 m/s；高速是系统性问题。旧 pilot 只运行了三个 legacy，因 1000074 的历史逐点比较失败而停止，没有候选结果。

`AutoPicker._set_arm` 保存 IK q_goal；`_slew_arm` 每个 60 Hz physics frame 将 q_cmd 向 goal 移动，legacy 每轴最多 0.045 rad，即 2.7 rad/s。TRANSPORT 每 15 帧从固定底座计算桶上方目标并求 IK，首次新目标在 phase 切换后一帧发出；IK 以当前 q_cmd 为 seed。因此限速也可能改变后续 IK 解与接触轨迹。现有 opt-in limiter 替换七个 arm joint command，未限制 IK goal 本身、未改动力学状态或夹爪。velocity/acceleration 状态跨 phase 保留。

student `VLAConfig` 保留 translation 每轴 0.02 m/control action、rotation 相对 world XYZ Euler 每轴 0.05 rad/control action、joint slew 0.045 rad/physics frame；physics 60 Hz、repeat=2，因此 control 30 Hz。expert 的 joint command 限速并不保证 TCP translation/rotation 完全落入此 envelope，也不保证 measured joint velocity 不超过 command limit。

## 3. Experimental protocol

复用 `run_arm_retime_pilot.one` / `collect_episode`、原 ArmMotionProfile 和原 motion/aggregate/transition/diagnostics 数值函数。Round 0 对全部八个 seed 建立 fresh legacy；每个候选轮次再次按 seed 顺序运行 fresh legacy → candidate（独立 subprocess）。同 seed、same world config、apple selection、reset planner、60 Hz / 3 substeps、detach_force_scale=1.5；record_rgb=False，但原 visibility acceptance 检查保留。

配对检查选中果实索引/初始位置/半径、base pose、standoff/azimuth、初始 TCP/arm/gripper、seed 与 detach force multiplier；每轮底层源码 SHA256 在轮前轮后核验。Round 1 后仅在 arm_motion.py/picker.py 增加可选 scope，变更前快照与 source_revision.json 保留；同一轮的 legacy/candidate 使用同一源码。历史 canonical 只作为背景，不参与通过/停止判断。所有失败 retained。

success 严格使用原 collector accepted：第一次 grasp/detach/placed 完整，无 first-attempt fail、base drift、branch break、wrong target、incidental detach、visibility 或 premature-detach rejection。`DONE` 本身不是 success。记录 20 s overall timeout、PULL timeout，以及 TRANSPORT >420 帧/大 stall 后直接 DROP 的回退迹象。

运动主表来自 actual 30 Hz proprios：TCP vector finite differences、SO(3) log angular velocity、max abs joint finite-difference velocity；加速度为相邻 velocity 的 raw differences。phase 用 interval 起点，奇数 physics frame 的混合边界归旧 phase；transition 取最后五个完全在边界前、最先五个完全在边界后的 interval，另保留 [-10,+20] 对齐序列。command 60 Hz trace 是 physics step 后、expert.update 后，不能与同帧 measured state 混作因果同时量。Pooled-frame quantiles 与每 seed 配对倍率同时报告，不对失败或慢轨迹静默筛选。

## 4. Seeds and stress-case definition

| Seed | 原 selection role | Case | 历史 TCP P95 (m/s) | 历史 transport path (m) |
| --- | --- | --- | --- | --- |
| 1000106 | short_transport_path | normal | 1.853 | 1.304 |
| 1000268 | lowest_episode_p95_speed | stress | 1.705 | 1.623 |
| 1000576 | median_speed | normal | 2.402 | 1.446 |
| 1000235 | longest_transport_path | normal | 2.724 | 2.142 |
| 1000356 | high_speed_p90 | normal | 2.677 | 1.709 |
| 1000074 | known_extreme | stress | 3.557 | 2.025 |
| 1000232 | speed_p25 | normal | 2.071 | 1.392 |
| 1000441 | speed_p75 | normal | 2.579 | 1.706 |

1000074 在实验前按历史数值敏感性及极端动态标为 stress。1000268 最初为 normal，但 Round 0/1 两次 fresh legacy 在完全相同 setup 下，GRASP→PULL 相差 44 physics frames、DROP 相差 86 frames，总时长 5.10→6.533 s；均成功。仅依据此 legacy-only 证据补标 stress（case_overrides.json），最终 normal=6、stress=2。原始分类仍保留在 protocol.json：Round 1 原 normal 1/7、stress 0/1；新分类 normal 1/6、stress 0/2，系统性失败结论不变。无 seed 删除/替换，也不因 candidate 失败本身改类。

## 5. Round 0 — fresh legacy

任务：normal 6/6，stress 2/2；所有八个完整 phase chain，branch break / premature detach 均为 0。

| Phase | TCP P50/P95/max (m/s) | angular P50/P95/max (rad/s) | joint P50/P95/max (rad/s) | TCP acc P95/max (m/s²) |
| --- | --- | --- | --- | --- |
| REACH | 0.8928 / 1.983 / 2.307 | 1.015 / 2.184 / 2.456 | 2.042 / 2.779 / 2.836 | 12.82 / 15.64 |
| GRASP | 0.5572 / 0.8816 / 1.191 | 1.373 / 2.918 / 3.362 | 2.232 / 2.946 / 3.438 | 9.758 / 10.74 |
| PULL | 0.127 / 0.3516 / 0.7326 | 0.08049 / 1.391 / 3.634 | 0.3978 / 2.029 / 3.283 | 6.399 / 12.27 |
| TRANSPORT | 1.529 / 2.899 / 4.251 | 3.962 / 6.988 / 8.429 | 2.668 / 3.42 / 4.005 | 20.4 / 30.81 |
| DROP | 0.01121 / 0.2494 / 0.4399 | 0.2112 / 1.622 / 3.113 | 0.2098 / 1.714 / 2.521 | 2.921 / 6.759 |

## 6. Round 1 — v2.0

Round0 all 8 accepted, including stress 1000074, no branch break/premature detach. Test existing mild velocity-only 2.0 rad/s profile to reduce saturated transport commands, preserving the phase-independent limiter.

| Cohort | Fresh legacy success | Candidate success | Candidate failures |
| --- | --- | --- | --- |
| normal | 6/6 | 1/6 | {"first_attempt_no_fruit_at_detection": 5} |
| stress | 2/2 | 0/2 | {"first_attempt_no_fruit_at_detection": 2} |
| all | 8/8 | 1/8 | {"first_attempt_no_fruit_at_detection": 7} |

注意：本轮仅 1/8 candidate 到达 TRANSPORT，下方 pooled legacy/candidate 样本覆盖不同；不能将 pooled ratio 解释为全 cohort 改善。失败导致的短 duration 也不是效率收益。

| TRANSPORT metric | fresh legacy | candidate | pooled ratio |
| --- | --- | --- | --- |
| tcp_speed P50 | 1.384 | 1.027 | 0.7418 |
| tcp_speed P95 | 2.793 | 1.563 | 0.5597 |
| tcp_speed max | 4.28 | 1.578 | 0.3688 |
| angular_speed P50 | 3.613 | 3.694 | 1.023 |
| angular_speed P95 | 6.977 | 5.152 | 0.7385 |
| angular_speed max | 7.927 | 5.221 | 0.6587 |
| joint_speed P50 | 2.64 | 2.004 | 0.7593 |
| joint_speed P95 | 3.363 | 2.225 | 0.6617 |
| joint_speed max | 3.968 | 2.314 | 0.5831 |
| linear_acceleration P50 | 9.466 | 4.064 | 0.4293 |
| linear_acceleration P95 | 19.94 | 8.634 | 0.433 |
| linear_acceleration max | 32.22 | 13.03 | 0.4043 |
| joint_acceleration P50 | 11.51 | 7.461 | 0.6484 |
| joint_acceleration P95 | 20.27 | 12.74 | 0.6286 |
| joint_acceleration max | 38.21 | 13.2 | 0.3455 |

TRANSPORT duration median 1.017 → 1.3 s；paired duration ratio median 1.3。Episode duration median 4.317 → 1.017 s；paired ratio median 0.2368。

| Seed | Case | Success L→C | Reason | duration L→C (s) | PULL duration L→C | TRANSPORT duration L→C | TCP P95 ratio | acc P95 ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000106 | normal | True → True | none | 5.2 → 5.5 | 2.033 → 2.033 | 1 → 1.3 | 0.7719 | 0.6323 |
| 1000268 | stress | True → False | first_attempt_no_fruit_at_detection | 6.533 → 1.167 | 1.833 → — | 1.7 → — | — | — |
| 1000576 | normal | True → False | first_attempt_no_fruit_at_detection | 4.5 → 1.85 | 1.4 → — | 1.033 → — | — | — |
| 1000235 | normal | True → False | first_attempt_no_fruit_at_detection | 4.533 → 0.9 | 1.5 → — | 1.483 → — | — | — |
| 1000356 | normal | True → False | first_attempt_no_fruit_at_detection | 4.067 → 0.9 | 1.617 → — | 0.9 → — | — | — |
| 1000074 | stress | True → False | first_attempt_no_fruit_at_detection | 3.8 → 1.033 | 0.75 → — | 0.85 → — | — | — |
| 1000232 | normal | True → False | first_attempt_no_fruit_at_detection | 4.1 → 0.95 | 1.55 → — | 1 → — | — | — |
| 1000441 | normal | True → False | first_attempt_no_fruit_at_detection | 4.133 → 1 | 1.5 → — | 1.05 → — | — | — |

确认结果：Both seeds 1000576 and 1000356 repeated legacy success and candidate same no-fruit failure at exactly frames 111 and 54, with premature detach. Global v2.0 regression is reproducible on these two originally normal cases.

## 7. Decision after Round 1

全阶段 v2.0 fresh legacy 8/8 成功，candidate 仅 1/8；七个 candidate 在 GRASP 报 no fruit at detection，均未进入 PULL，六个记录 premature detach。原分类 normal 1/7、stress 0/1；按 legacy-only 证据将 1000268 标为 stress 后为 normal 1/6、stress 0/2，结论不变。唯一完成 transport 的 seed 1000106 TCP P95 比配对 legacy 下降约 22.8%，但不能用这个幸存样本推断全体收益。启动 1000576/1000356 的 fresh 配对确认，检查失败可复现性。

Next action: 先完成两个失败 seed 的重复配对。若相同失败再次出现，仅将现有 2.0 rad/s velocity retime 的作用范围缩为 TRANSPORT，保留 REACH/GRASP/PULL/DROP 的 legacy 命令。这是针对已观察到的 pre-PULL regression 的一次有限 limiter 修改；不继续收紧全阶段限制。

## 8. Round 2 — transport_v2.0

Round1 global v2.0 failed 7/8 before PULL; two fresh confirmation pairs reproduce identical GRASP failure/premature detach while legacy succeeds. Change only limiter scope to TRANSPORT at unchanged 2.0 rad/s, retaining legacy motion in contact-sensitive pre-detach phases; optional scope does not alter state transitions or reset command state.

| Cohort | Fresh legacy success | Candidate success | Candidate failures |
| --- | --- | --- | --- |
| normal | 6/6 | 6/6 | {} |
| stress | 0/2 | 2/2 | {} |
| all | 6/8 | 8/8 | {} |

下表保留所有尝试的 measured intervals，未筛除失败。

| TRANSPORT metric | fresh legacy | candidate | pooled ratio |
| --- | --- | --- | --- |
| tcp_speed P50 | 1.24 | 1.217 | 0.9811 |
| tcp_speed P95 | 2.769 | 2.207 | 0.7971 |
| tcp_speed max | 3.442 | 3.187 | 0.926 |
| angular_speed P50 | 3.37 | 3.362 | 0.9975 |
| angular_speed P95 | 6.925 | 5.984 | 0.8642 |
| angular_speed max | 7.448 | 7.568 | 1.016 |
| joint_speed P50 | 2.595 | 2.005 | 0.7729 |
| joint_speed P95 | 3.286 | 2.514 | 0.7652 |
| joint_speed max | 3.983 | 3.338 | 0.838 |
| linear_acceleration P50 | 9.242 | 6.237 | 0.6749 |
| linear_acceleration P95 | 19.77 | 16.01 | 0.8096 |
| linear_acceleration max | 24.17 | 18.06 | 0.7472 |
| joint_acceleration P50 | 11.03 | 7.992 | 0.7247 |
| joint_acceleration P95 | 20.88 | 15.46 | 0.7404 |
| joint_acceleration max | 37.79 | 32.89 | 0.8702 |

TRANSPORT duration median 1.042 → 1.275 s；paired duration ratio median 1.1。Episode duration median 4.517 → 4.417 s；paired ratio median 1.027。

| Seed | Case | Success L→C | Reason | duration L→C (s) | PULL duration L→C | TRANSPORT duration L→C | TCP P95 ratio | acc P95 ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000106 | normal | True → True | none | 5.2 → 5.5 | 2.033 → 2.033 | 1 → 1.3 | 0.7705 | 0.5774 |
| 1000268 | stress | False → True | none | 6.267 → 6.4 | 1.833 → 1.833 | 1.467 → 1.55 | 0.7972 | 0.7563 |
| 1000576 | normal | True → True | none | 4.5 → 4.5 | 1.4 → 1.4 | 1.033 → 1.033 | 0.8392 | 0.7408 |
| 1000235 | normal | True → True | none | 4.533 → 4.333 | 1.5 → 1.5 | 1.483 → 1.283 | 0.727 | 0.8463 |
| 1000356 | normal | True → True | none | 4.067 → 4.433 | 1.617 → 1.617 | 0.9 → 1.267 | 0.7635 | 0.7034 |
| 1000074 | stress | False → True | none | 5.6 → 3.867 | 1.65 → 0.5333 | 1.767 → 1.1 | 0.9391 | 0.807 |
| 1000232 | normal | True → True | none | 4.1 → 4.4 | 1.55 → 1.55 | 1 → 1.3 | 0.7515 | 0.6784 |
| 1000441 | normal | True → True | none | 4.133 → 4.267 | 1.5 → 1.5 | 1.05 → 1.2 | 0.786 | 0.6906 |

## 8.1. Decision after Round 2

TRANSPORT-only v2.0 候选 normal 6/6、stress 2/2 成功，无 stall/timeout/branch break/premature detach。Fresh legacy normal 6/6、stress 0/2（missed bucket/incidental detach），因此 stress 恢复不能单次归因为限速收益。Normal TRANSPORT TCP P95 2.729→2.123 m/s（-22.2%）、acc P95 18.994→13.833 m/s²（-27.2%）；all seed 配对 TCP P95 倍率中位数 0.778、episode duration +2.68%。但 PULL→TRANSPORT command derivative 中位最大值仍 120 rad/s²，measured boundary acc 1.495→11.550 m/s²（约 7.7x），仅较 legacy post-window 下降约 23%。因此按 Case B 做最后一次针对边界突变的检验，不做参数网格。

Next action: Round 3 使用已有温和组合 1.5 rad/s + 20 rad/s²，继续只作用 TRANSPORT。目标检验残余边界尖峰是否进一步缓和且保留任务；这是最后一轮。速度与加速度同时变化，因此不能把效果完全归因于 acceleration limiter，也不能据此证明它绝对必要。

## 9. Round 3 — transport_va1.5_a20

Round2 transport_v2.0 preserved all 8 candidates and improved normal TCP P95 by 22.2%, but PT command derivative median maximum remains 120 rad/s^2 and measured acceleration rises 1.495 -> 11.550 m/s^2 (~7.7x). Case B: final bounded test of existing mild 1.5 rad/s + 20 rad/s^2 combo, same TRANSPORT-only scope; no further tuning. Combined velocity/acceleration change cannot identify acceleration-only causality.

| Cohort | Fresh legacy success | Candidate success | Candidate failures |
| --- | --- | --- | --- |
| normal | 6/6 | 6/6 | {} |
| stress | 2/2 | 2/2 | {} |
| all | 8/8 | 8/8 | {} |

下表保留所有尝试的 measured intervals，未筛除失败。

| TRANSPORT metric | fresh legacy | candidate | pooled ratio |
| --- | --- | --- | --- |
| tcp_speed P50 | 1.363 | 0.9434 | 0.6921 |
| tcp_speed P95 | 2.793 | 1.896 | 0.6788 |
| tcp_speed max | 4.255 | 2.441 | 0.5738 |
| angular_speed P50 | 3.834 | 2.667 | 0.6956 |
| angular_speed P95 | 6.975 | 4.733 | 0.6786 |
| angular_speed max | 8.42 | 6.319 | 0.7505 |
| joint_speed P50 | 2.643 | 1.507 | 0.5702 |
| joint_speed P95 | 3.379 | 1.81 | 0.5358 |
| joint_speed max | 3.968 | 3.429 | 0.8642 |
| linear_acceleration P50 | 9.509 | 3.652 | 0.3841 |
| linear_acceleration P95 | 19.73 | 10.11 | 0.5122 |
| linear_acceleration max | 30.86 | 12.73 | 0.4126 |
| joint_acceleration P50 | 11.46 | 6.152 | 0.5371 |
| joint_acceleration P95 | 20.52 | 9.537 | 0.4647 |
| joint_acceleration max | 38.22 | 15.5 | 0.4057 |

TRANSPORT duration median 1.008 → 1.642 s；paired duration ratio median 1.462。Episode duration median 4.3 → 4.783 s；paired ratio median 1.094。

| Seed | Case | Success L→C | Reason | duration L→C (s) | PULL duration L→C | TRANSPORT duration L→C | TCP P95 ratio | acc P95 ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000106 | normal | True → True | none | 5.2 → 5.833 | 2.033 → 2.033 | 1 → 1.633 | 0.6044 | 0.4585 |
| 1000268 | stress | True → True | none | 6.533 → 6.5 | 1.833 → 1.833 | 1.7 → 1.65 | 0.6049 | 0.4755 |
| 1000576 | normal | True → True | none | 4.5 → 4.8 | 1.4 → 1.4 | 1.033 → 1.333 | 0.6598 | 0.4972 |
| 1000235 | normal | True → True | none | 4.533 → 4.667 | 1.5 → 1.5 | 1.483 → 1.633 | 0.5494 | 0.5135 |
| 1000356 | normal | True → True | none | 4.067 → 4.833 | 1.617 → 1.617 | 0.9 → 1.683 | 0.6539 | 0.5283 |
| 1000074 | stress | True → True | none | 3.733 → 3.933 | 0.5333 → 0.5333 | 0.9833 → 1.167 | 0.5907 | 0.387 |
| 1000232 | normal | True → True | none | 4.1 → 4.767 | 1.55 → 1.55 | 1 → 1.65 | 0.552 | 0.4011 |
| 1000441 | normal | True → True | none | 4.1 → 4.733 | 1.5 → 1.5 | 1.017 → 1.667 | 0.6432 | 0.4622 |

## 9.1. Decision after Round 3

最终 TRANSPORT-only va1.5_a20：fresh legacy 与 candidate 均 normal 6/6、stress 2/2、all 8/8 成功。All pooled TRANSPORT TCP P95 2.793→1.896 m/s (-32.1%)、acc P95 19.731→10.107 m/s² (-48.8%)；normal 分别 2.728→1.680 (-38.4%)、19.008→8.885 (-53.3%)。Per-seed TCP P95 倍率中位数 0.605、acc P95 0.469。PT post-window measured acc 中位数 14.663→7.425，actual command acceleration window max 的中位数 162→20.871 rad/s²。配对 episode duration +9.42%，TRANSPORT duration +46.18%；最长 episode 6.5 s、最长 TRANSPORT 1.683 s、max transport stall counter 6。两个 scope profiles 都保持成功；组合配置对速度/加速度改善更明显，成本可接受。

Next action: GO_TO_30_SEED_VALIDATION；停止 smoke 调参。

## 10. Normal vs stress cases

| Profile | Normal success | Stress success | All success | All placed/detached |
| --- | --- | --- | --- | --- |
| legacy | 6/6 (100.0%) | 2/2 (100.0%) | 8/8 (100.0%) | 8/8 |
| v2.0 | 1/6 (16.7%) | 0/2 (0.0%) | 1/8 (12.5%) | 1/1 |
| transport_v2.0 | 6/6 (100.0%) | 2/2 (100.0%) | 8/8 (100.0%) | 8/8 |
| transport_va1.5_a20 | 6/6 (100.0%) | 2/2 (100.0%) | 8/8 (100.0%) | 8/8 |

每轮 legacy 均独立运行，最终 normal/stress 的 denominator 在各轮分析中统一为 6/2；原分类与补标证据同时保留。以上为有意挑选的 smoke cohort，不是随机总体 success-rate 估计。

Fresh legacy repeatability（相对 Round 0；同时间比较，phase timing 不同会放大该数值，不是判定 gate）：

| Seed | Repeat | Success | phase frame Δ | TCP max coord Δ (m) | joint max Δ (rad) |
| --- | --- | --- | --- | --- | --- |
| 1000106 | round1 | True | [0, 0, 0, 0, 0, 0] | 3.517e-06 | 2.166e-05 |
| 1000106 | round2 | True | [0, 0, 0, 0, 0, 0] | 2.432e-05 | 0.0001975 |
| 1000106 | round3 | True | [0, 0, 0, 0, 0, 0] | 2.217e-05 | 0.0002219 |
| 1000268 | round1 | True | [0, 0, 44, 45, 86, 86] | 0.8961 | 1.996 |
| 1000268 | round2 | False | [0, 0, 46, 47, 74, 74] | 0.8999 | 2.028 |
| 1000268 | round3 | True | [0, 0, 44, 45, 86, 86] | 0.8961 | 1.996 |
| 1000576 | round1 | True | [0, 0, 0, 0, 0, 0] | 6.676e-06 | 2.897e-05 |
| 1000576 | round1_confirm | True | [0, 0, 0, 0, 0, 0] | 1.633e-05 | 7.057e-05 |
| 1000576 | round2 | True | [0, 0, 0, 0, 0, 0] | 1.851e-05 | 7.963e-05 |
| 1000576 | round3 | True | [0, 0, 0, 0, 0, 0] | 5.007e-06 | 3.105e-05 |
| 1000235 | round1 | True | [0, 0, 0, 0, 0, 0] | 2.742e-05 | 0.0001339 |
| 1000235 | round2 | True | [0, 0, 0, 0, 0, 0] | 0.001266 | 0.00654 |
| 1000235 | round3 | True | [0, 0, 0, 0, 0, 0] | 1.055e-05 | 0.0001105 |
| 1000356 | round1 | True | [0, 0, 0, 0, 0, 0] | 8.285e-06 | 4.059e-05 |
| 1000356 | round1_confirm | True | [0, 0, 0, 0, 0, 0] | 8.523e-06 | 6.306e-05 |
| 1000356 | round2 | True | [0, 0, 0, 0, 0, 0] | 1.132e-05 | 7.51e-05 |
| 1000356 | round3 | True | [0, 0, 0, 0, 0, 0] | 2.748e-05 | 0.0001502 |
| 1000074 | round1 | True | [0, 0, -2, 11, 3, 3] | 0.5727 | 0.5431 |
| 1000074 | round2 | False | [0, 0, -2, 65, 112, 112] | 0.8432 | 2.497 |
| 1000074 | round3 | True | [0, 0, 0, 0, 0, 0] | 0.001551 | 0.006953 |
| 1000232 | round1 | True | [0, 0, 0, 0, 0, 0] | 0.0002843 | 0.003577 |
| 1000232 | round2 | True | [0, 0, 0, 0, 0, 0] | 0.0002549 | 0.004408 |
| 1000232 | round3 | True | [0, 0, 0, 0, 0, 0] | 0.000101 | 0.003494 |
| 1000441 | round1 | True | [0, 0, 0, 0, 2, 2] | 0.004547 | 0.009955 |
| 1000441 | round2 | True | [0, 0, 0, 0, 2, 2] | 0.004547 | 0.009947 |
| 1000441 | round3 | True | [0, 0, 0, 0, 0, 0] | 3.517e-06 | 4.81e-05 |

## 11. Failure analysis

全阶段 v2.0 是本轮唯一稳定的新任务失败模式：7/8 在 GRASP 报 first_attempt_no_fruit_at_detection，6/8 有 premature detach diagnostics，未进入 PULL。1000576 与 1000356 的额外 fresh 确认配对均重复 legacy 成功、candidate 在相同帧失败。不能将此归为单个 stress 波动，因此该全阶段 profile 不合适。

限制作用范围后，Round 2 和 Round 3 的 candidate 各 8/8 success / placed / detached，全部经过 REACH→GRASP→PULL→TRANSPORT→DROP→DONE；没有 candidate branch break、premature detach、timeout 或 watchdog fallback。Round 2 fresh legacy 的两个 stress 失败分别是 1000268 missed bucket、1000074 incidental_detach（placed=True 但 accepted=False）；Round 3 两者 legacy 和 candidate 均成功。它们体现 baseline 自身敏感性，不是候选造成的新失败。Round 0/1/2/3 与确认 runs 均保留，总计 60 次执行（34 legacy、26 candidate）；两个确认 pairs 不混入八 seed smoke 分母。

| Round/profile | branch breaks | premature detach | timeout | transport timeout fallback | possible stall fallback |
| --- | --- | --- | --- | --- | --- |
| legacy | 0 | 0 | 0 | 0 | 0 |
| v2.0 | 0 | 6 | 0 | 0 | 0 |
| transport_v2.0 | 0 | 0 | 0 | 0 | 0 |
| transport_va1.5_a20 | 0 | 0 | 0 | 0 | 0 |

TRANSPORT fallback 检查使用完整 command trace 的 elapsed/stall counter；它是诊断标志，不替换原成功定义。若进入 DROP 的同帧 counter 被 reset，max stall 可少一帧，因此同时保留完整 trace 供复查。

## 12. Dynamic comparison

Round 1 仅一个 candidate 到达 TRANSPORT，不能将其 pooled quantile 与八个 legacy 当成无偏改善率；其逐 seed 配对值仅代表该幸存 case。后续完整 cohort 才用于工程推荐。

下表为每个 seed candidate/fresh-legacy 倍率的中位数。不同于 pooled-frame P95 比值，避免较慢 episode 的更多帧主导唯一结论。

| Profile | TCP P95 ratio | TCP acc P95 ratio | joint P95 ratio | PT post TCP ratio | PT post acc ratio | episode duration ratio |
| --- | --- | --- | --- | --- | --- | --- |
| v2.0 | 0.7719 | 0.6323 | 0.6271 | 0.8285 | 0.7326 | 0.2368 |
| transport_v2.0 | 0.7782 | 0.7221 | 0.7484 | 0.7911 | 0.7734 | 1.027 |
| transport_va1.5_a20 | 0.6046 | 0.4688 | 0.5112 | 0.4599 | 0.5165 | 1.094 |

| Profile / cohort | TRANSPORT TCP P95 L→C | TRANSPORT acc P95 L→C |
| --- | --- | --- |
| legacy / normal | 2.728 → 2.728 | 19.01 → 19.01 |
| legacy / stress | 3.987 → 3.987 | 28.58 → 28.58 |
| v2.0 / normal | 2.728 → 1.563 | 18.99 → 8.634 |
| v2.0 / stress | 3.892 → — | 28.03 → — |
| transport_v2.0 / normal | 2.729 → 2.123 | 18.99 → 13.83 |
| transport_v2.0 / stress | 3.213 → 3.101 | 21.09 → 17.58 |
| transport_va1.5_a20 / normal | 2.728 → 1.68 | 19.01 → 8.885 |
| transport_va1.5_a20 / stress | 3.821 → 2.399 | 27.11 → 10.32 |

| Profile | PT TCP pre→post median (m/s) | PT acc pre→post median (m/s²) | PT command actual acc median max (rad/s²) | PT q_goal jump median (rad norm) | transport >2 m/s frame fraction median |
| --- | --- | --- | --- | --- | --- |
| legacy | 0.1385 → 1.414 | 1.488 → 14.94 | 162 | 2.519 | 0.4032 |
| v2.0 | 0.1349 → 1.016 | 1.632 → 9.148 | 222 | 3.421 | 0 |
| transport_v2.0 | 0.1379 → 1.047 | 1.495 → 11.55 | 120 | 2.649 | 0.1352 |
| transport_va1.5_a20 | 0.1379 → 0.6107 | 1.495 → 7.425 | 20.87 | 2.65 | 0 |

推荐配置 command limiter 诊断（acceleration exceptions 只在 active scope 内计数；crossing_count 为逐关节累计）：

| Seed | crossing count | acc bound exceptions | exceptions without landing | max internal acc in scope | PT actual command acc max |
| --- | --- | --- | --- | --- | --- |
| 1000106 | 23 | 13 | 0 | 162 | 108.2 |
| 1000268 | 22 | 8 | 0 | 33.84 | 20 |
| 1000576 | 24 | 16 | 0 | 37.54 | 20 |
| 1000235 | 23 | 11 | 0 | 36.93 | 22.47 |
| 1000356 | 21 | 13 | 0 | 38.27 | 22.3 |
| 1000074 | 15 | 11 | 0 | 137.4 | 20 |
| 1000232 | 24 | 10 | 0 | 86.24 | 20 |
| 1000441 | 23 | 14 | 0 | 40.61 | 21.74 |

![TRANSPORT comparison](transport_comparison.png)

![PULL to TRANSPORT](pull_transport_transition.png)

推荐 profile 各 phase 的保留情况与动态（全部尝试，包括失败）：

| Phase | N L/C | TCP P95 L→C | angular P95 L→C | joint P95 L→C | duration median L→C |
| --- | --- | --- | --- | --- | --- |
| REACH | 8/8 | 1.983 → 1.983 | 2.184 → 2.184 | 2.779 → 2.779 | 0.225 → 0.225 |
| GRASP | 8/8 | 0.853 → 0.853 | 2.856 → 2.849 | 2.797 → 2.796 | 0.3 → 0.3 |
| PULL | 8/8 | 0.2988 → 0.2969 | 1.189 → 1.19 | 1.463 → 1.463 | 1.525 → 1.525 |
| TRANSPORT | 8/8 | 2.793 → 1.896 | 6.975 → 4.733 | 3.379 → 1.81 | 1.008 → 1.642 |
| DROP | 8/8 | 0.2859 → 0.2028 | 1.447 → 0.961 | 1.516 → 1.078 | 1 → 1 |

Student envelope 残余差距：按当前 VLA world-relative Euler clipping 的每轴阈值检查 measured 30 Hz motion，只作对齐诊断，不作新 acceptance gate。

| Profile | TRANSPORT translation exceed median | rotation exceed median | either exceed median |
| --- | --- | --- | --- |
| legacy | 0.8138 | 0.8667 | 0.9056 |
| v2.0 | 0.8205 | 0.7949 | 0.8205 |
| transport_v2.0 | 0.7589 | 0.7266 | 0.7974 |
| transport_va1.5_a20 | 0.6361 | 0.6161 | 0.78 |

## 13. Recommended profile

**transport_va1.5_a20**

推荐 `ArmMotionProfile(mode="velocity_accel", vmax_rad_s=1.5, amax_rad_s2=20, active_phases=("TRANSPORT",))`。这是 nominal acceleration 配置，landing guard 允许例外。

研究问题回答：

1. **能明显减轻 TRANSPORT 高速。** 推荐 profile 的 all-seed TCP P95 -32.1%、max 4.255→2.441 m/s；normal TCP P95 -38.4%，六个 normal 的 pooled TCP max 降至 1.980 m/s。
2. **Velocity-only 已足以取得第一步改善，前提是只限 TRANSPORT。** transport_v2.0 已 8/8 成功，normal TCP P95 -22.2%；全阶段 v2.0 明显不适用。
3. **没有证明 acceleration limiter 是降速的必要条件。** Round 2 残余 onset derivative / acceleration spike 明确存在，Round 3 组合配置进一步降低实测动态；但它同时把速度从 2.0 降到 1.5，不能把全部增益归因于 acceleration，也不能证明同 scope v1.5 velocity-only 会失败。为避免无目的调参，本轮不额外增加消融。
4. **Scoped retime 未观察到破坏五个操作阶段及 detach。** 全阶段 retime 破坏 GRASP，并阻止进入 PULL；scope 修复后 normal 6/6、stress 2/2 均成功，PULL 动作/判定/timeout 未改。
5. **当前折中选 transport_va1.5_a20。** 相比 scope v2.0，它降低更多速度与加速度、保留任务；episode 配对中位成本 +9.4%，没有 watchdog 压力。scope v2.0 可作为保守备选，但无需本轮再调。
6. **已有足够工程证据进入 30-seed validation。** 不是完整 action-envelope 对齐的证明，也不意味着生产默认立即切换。

## 14. Whether to proceed to 30-seed validation

**GO_TO_30_SEED_VALIDATION**

停止调参，复用上一轮 pilot_seed_selection.json 的 30 个 expanded seeds，以推荐 TRANSPORT-only profile 做 fresh paired validation，并保留 1000074/1000268 的 stress 标记；继续报告 normal/stress/all 成功率、paired dynamics、incidental detach、landing exceptions 与 timeout/stall。此任务只给出 GO 决策，尚未运行 30-seed validation。

## 15. Remaining risks

八个 smoke seeds 是历史成功轨迹的有意选择，不是泛化成功率估计；stress 重复次数仍少，30-seed validation 必须保留失败样本。1000268/1000074 的 legacy 有 timing/trajectory/甚至 accepted outcome variation，不能把一次配对成功差异当成 retime 因果收益。

当前仅压低最突出的 TRANSPORT mismatch；REACH 原偏快问题保留。推荐 profile 的 TRANSPORT measured student translation/rotation 任一轴超 envelope 的 per-seed fraction 中位数仍为 78%（fresh legacy 89.4%），stress 的 measured joint velocity max 仍约 3.429 rad/s，故尚不等于 student 可逐步无裁剪复现。需要 30-seed 验证后再决定下一阶段数据使用。

Acceleration landing guard 在八个推荐 runs 中累计 106 个 scope 内逐关节 internal acceleration bound exceptions，均伴随 landing（无 unexplained exceptions）。PT actual command acceleration 仍有个别峰值超过 20，名义参数不是硬保证；必须看 raw measured dynamics 与完整 command trace。组合限速/限加速度的消融没有单独识别 acceleration 的净贡献。

Scope 修复只改变七关节 command retiming 的启用范围；默认 active_phases=None 和无 opt-in profile 的 legacy 行为保持。推荐 profile 仍是实验配置，没有把它悄悄设为全部采集任务默认值。

Acceleration limiter 的 landing guard 在 goal 落于 stopping distance 内时可能牺牲 acceleration bound；现有实现显式记录 landing/crossing。只有运行过的 profile 才有性能结论。Command derivative 与 measured acceleration 是不同量；30 Hz raw finite differences 不是全部 60 Hz 瞬时峰值。

## Artifacts and reproduction

- `protocol.json`、`case_overrides.json`、`seed_geometry.json`、每轮 `spec.json`：固定 seeds、输入源码 hash、profile 与运行前 reason。
- `round0_legacy_summary.json`、实际运行的 `roundN_summary.json`：all/normal/stress phase statistics、transition statistics、配对结果、command/watchdog 诊断。
- `paired_seed_results.csv`：每 seed、每轮的任务、所有 phase P50/P95/max、加速度、duration、transition、limiter saturation、student envelope exceed 和配对倍率。
- `roundN/runs/<profile>/<seed>/run.json`、`stdout.log`：完整 measured trajectory、command trace、reset/acceptance diagnostics，成功失败全量保留。
- `decisions.json`：实际上一轮结果驱动的中间决策与最终决策；`legacy_repeatability.json`：fresh legacy 重复诊断。
- `source_integrity.json`、`limiter_tests.log`、`validation.json`：每轮输入源码一致、允许的 scope 修改另行记录，原六项及新增 scope 测试共七项通过、配对与统计核验。
- 新增实验源码：`scripts/run_arm_retime_rounds.py`、`scripts/analyze_arm_retime_rounds.py`、`scripts/report_arm_retime_rounds.py`。另有最小 limiter 集成修改 `treesim/arm_motion.py`、`treesim/picker.py` 和作用范围回归测试 `scripts/test_arm_motion.py`。physics/student/collector/fixed-base stance 与 phase transitions 未改。

解释器 `.pixi/envs/default/bin/python`。从仓库根运行 `scripts/run_arm_retime_rounds.py run --round N --profile PROFILE --reason REASON`；每轮 spec 不可覆盖，已完成 runs 可续跑。分析可用 `scripts/analyze_arm_retime_rounds.py log/arm_retime_rounds/roundN`，最终报告用 `scripts/report_arm_retime_rounds.py`。旧 pilot 结果未覆盖。
