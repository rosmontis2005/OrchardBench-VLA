# Gate 1 — canonical GT oracle replay

GT actions → encode_window → OrchardActionAdapter.set_denormalized_chunk → to_native(live obs) → OrchardVLAEnv.step → 当前 strict bucket evaluator。无模型。

两模式只改变 waypoint advancement：time 每步推进；reach 在步后测得 TCP position / SO(3) error 达标时推进，或到 maximum dwell 后记录并放弃当前未达标点、推进一个点。无插值、retiming、recovery 或额外末端等待。targets 耗尽、evaluator success 或 episode budget 到达即停止。

GT chunk 使用示教中固定的 anchor，避免 live tracking lag 改变绝对 GT 路径；native command 每步使用真实 live observation。尾部使用重叠完整窗口，但每个 GT index 只按顺序消费一次。该 oracle 测试绝对 GT targets 的可执行性，不测试模型或偏离示教后的预测能力。

固定参数：position ≤ 0.01 m；SO(3) rotation ≤ 0.08 rad；maximum dwell = 30 control steps（1 s，按一个 chunk 长度预先设定，未做 pilot 或 val 调参）；episode budget = 600 steps（20 s）；30 Hz。其余环境配置全部默认。

canonical manifest 自动选取全部 30 accepted val。读取的是原 canonical annotation，未用 Gate 0 retimed run 替换；两者不能混为同一数据版本或 seed cohort。首次一个 val paired smoke 直接计入正式 30 对，无重复挑选。协议、输入与执行源码哈希被冻结并核验；paired reset 完全一致。

| mode | strict success | grasped | detached | budget timeout | IK failed steps | clipped steps | dwell timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| time-indexed | 0/30 | 3/30 | 3/30 | 0 | 779 | 3656 | 0 |
| reach-conditioned | 17/30 | 30/30 | 29/30 | 11 | 1002 | 11316 | 79 |

| paired outcome | episodes |
| --- | ---: |
| time PASS / reach PASS | 0 |
| time PASS / reach FAIL | 0 |
| time FAIL / reach PASS | 17 |
| time FAIL / reach FAIL | 13 |

success 与 strict_bucket_success 均直接来自当前 evaluator；没有把 detach、collector accepted 或 target 消费完成算作 PASS。target_progress_fraction 为已尝试的不同 target 数 / 总数，targets_reached 另记。误差为每步执行后的 GT target 与实际 TCP 误差。

| mode | position mean / P95 / max (m) | rotation mean / P95 / max (rad) |
| --- | --- | --- |
| time-indexed | 0.41139 / 0.84401 / 0.95588 | 1.14848 / 2.86752 / 3.12773 |
| reach-conditioned | 0.03401 / 0.08940 / 0.31630 | 0.07090 / 0.21412 / 0.92164 |

失败标签是可重叠的观测证据，IK failure / clipping / tracking lag 不单独证明根因。DROP_bucket_failure 表示已 grasp + detach 但未满足 strict predicate。未抓到、抓到未 detach、已 detach 未入桶分别记录。

- time-indexed: {"grasp_failure": 27, "target_sequence_exhausted": 30, "IK_failure_observed": 24, "clipping_observed": 30, "tracking_lag_observed": 30, "DROP_bucket_failure": 3}；出现 IK failure / clipping / dwell timeout 的 episode 数 = 24 / 30 / 0。
- reach-conditioned: {"DROP_bucket_failure": 12, "episode_budget": 11, "IK_failure_observed": 8, "clipping_observed": 13, "dwell_timeout": 11, "tracking_lag_observed": 13, "target_sequence_exhausted": 2, "detach_failure": 1}；出现 IK failure / clipping / dwell timeout 的 episode 数 = 14 / 30 / 15。

| episode / seed | time | reach | reach final phase; index/N | reach failure evidence |
| --- | --- | --- | --- | --- |
| episode_000010 / 1000025 | FAIL | PASS | DROP; 114/141 | — |
| episode_000020 / 1000056 | FAIL | FAIL | TRANSPORT; 76/138 | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000030 / 1000079 | FAIL | FAIL | DONE; 121/122 | DROP_bucket_failure, target_sequence_exhausted, clipping_observed, tracking_lag_observed |
| episode_000040 / 1000106 | FAIL | PASS | DROP; 132/157 | — |
| episode_000050 / 1000136 | FAIL | FAIL | TRANSPORT; 94/135 | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000060 / 1000164 | FAIL | PASS | DROP; 104/129 | — |
| episode_000070 / 1000192 | FAIL | PASS | DROP; 108/129 | — |
| episode_000080 / 1000214 | FAIL | FAIL | TRANSPORT; 87/132 | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000090 / 1000235 | FAIL | FAIL | TRANSPORT; 93/137 | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000100 / 1000250 | FAIL | PASS | DROP; 133/155 | — |
| episode_000110 / 1000274 | FAIL | FAIL | DONE; 128/129 | DROP_bucket_failure, target_sequence_exhausted, clipping_observed, tracking_lag_observed |
| episode_000120 / 1000294 | FAIL | FAIL | TRANSPORT; 91/148 | DROP_bucket_failure, episode_budget, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000130 / 1000345 | FAIL | FAIL | DROP; 120/131 | DROP_bucket_failure, episode_budget, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000140 / 1000371 | FAIL | FAIL | TRANSPORT; 88/133 | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000150 / 1000394 | FAIL | FAIL | TRANSPORT; 108/159 | DROP_bucket_failure, episode_budget, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000160 / 1000420 | FAIL | PASS | DROP; 112/132 | — |
| episode_000170 / 1000440 | FAIL | PASS | DROP; 102/122 | — |
| episode_000180 / 1000465 | FAIL | FAIL | PULL; 80/164 | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000190 / 1000486 | FAIL | PASS | DROP; 112/121 | — |
| episode_000200 / 1000510 | FAIL | PASS | DROP; 123/149 | — |
| episode_000210 / 1000542 | FAIL | PASS | DROP; 109/131 | — |
| episode_000220 / 1000572 | FAIL | PASS | DROP; 106/127 | — |
| episode_000230 / 1000593 | FAIL | PASS | DROP; 108/123 | — |
| episode_000240 / 1000626 | FAIL | PASS | DROP; 118/127 | — |
| episode_000250 / 1000646 | FAIL | PASS | DROP; 108/127 | — |
| episode_000260 / 1000676 | FAIL | FAIL | PULL; 64/141 | detach_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000270 / 1000702 | FAIL | FAIL | TRANSPORT; 114/172 | DROP_bucket_failure, episode_budget, IK_failure_observed, clipping_observed, dwell_timeout, tracking_lag_observed |
| episode_000280 / 1000731 | FAIL | PASS | DROP; 112/124 | — |
| episode_000290 / 1000750 | FAIL | PASS | DROP; 107/121 | — |
| episode_000300 / 1000779 | FAIL | PASS | DROP; 106/119 | — |

17 条 time FAIL / reach PASS 说明，改变 advancement rule 对这些 canonical GT episodes 的任务成功有明确影响；但 reach 仍失败 13/30，不能据此认定主要执行问题已全部消除。17 条 reach 成功中，4 条发生过 bounded forced advancement（episode_000280, episode_000200, episode_000300, episode_000240），其余 13 条未发生。成功表示任务完成，不声称整条路径所有 waypoint 均达到容差。

reach 的 11 条预算失败按最终示教 target 相位分布为 {'PULL': 2, 'DROP': 1, 'TRANSPORT': 8}；另 2 条耗尽 targets 后未入桶。13 条失败中 8 条出现 IK failure，5 条无 IK failure；因此 IK failure 不能单独解释全部失败。30/30 抓取、29/30 脱果，唯一 detach failure 为 episode_000260 / seed 1000676。两模式均无 branch break。clipping 是各模式都存在的观测，reach 步数更多，不宜只比较 clipping 总数。

最后核验 60 份 replay JSON 与 CSV 60 行、全部 paired reset、已执行 targets 逐步完全相同、time 每步严格推进、reach dwell ≤30，以及冻结输入/执行源码哈希均通过。见 result_verification.json。

Gate 1 FAIL。当前仍存在 controller / evaluator / execution contract blocker。停止在此处，不进行正式模型训练。

全部 60 次 replay 已完成。未开始训练、未修改 controller / evaluator / physics / canonical 数据。逐步误差、IK、clipping、gripper、phase 和 dwell timeout 原因见 episodes/*.json；每 episode 汇总见 episode_results.csv。
