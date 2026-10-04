# OrchardBench V1 — 450 expert collection and frozen replay filtering

正式 raw cohort：450 accepted；strict replay PASS 401，FAIL 49；最终 V1 401 条。未加载模型、未下载 checkpoint、未做 gradient update。

## Frozen protocol

Source of truth: `log/gate1_transport_30s_revalidation`，reach-conditioned 26/30、grasp/detach 30/30。已逐项核对源码 SHA。
原始 frozen replay SHA256: `6142e7a9c1b84168f804c0567e99271470fbfd27b90cb81ff05efe42ca54acbb`。
Collector SHA256: `8d1ecbe15023664aa1bd4aa05d190ab664d1ee934f5a074a03f59806e773420d`；git `df73ee4e804e8aa3eb3f97aab471276db30a0838`。
TRANSPORT velocity_accel vmax=0.5 rad/s, amax=20 rad/s²；其余 phases legacy；detach_force_scale=1.5；真实 ego/wrist RGB，30 Hz，measured proprio/next-state actions。
Replay：position≤0.01 m、SO(3)≤0.08 rad、maximum dwell=30；旧 600-step 默认显式覆盖为 900 steps / 30 s。reach OR maximum-dwell fallback，每次推进一个目标。无 passed-waypoint、lookahead、插值、额外 terminal hold、recovery 或 replanning。
GT encode_window → OrchardActionAdapter → to_native(live obs) → OrchardVLAEnv.step；PASS 只取当前 strict evaluator info.success。

## Collection

Total attempts 1144；正式 expert accepted 450；expert rejected 681；并发 overflow accepted 13（单独保留，不进入正式 cohort）。
全部完成 attempt 的 acceptance rate 40.47%（包含 overflow；未因 replay FAIL 补采）。
正式 attempt seed 1020000–1021143；450th accepted seed 1021101。独立 pilot 3 accepted，不计入 450。
Rejection reasons: `{"first_attempt_no_fruit_at_detection": 525, "first_attempt_grasp_stalled": 44, "branch_break": 70, "target_not_visible_in_policy_observation": 10, "planned_target_not_picked": 9, "premature_detach": 8, "incidental_detach": 9, "no_feasible_fixed_base_setup": 4, "first_attempt_missed_bucket": 1, "base_drift_exceeded": 1}`。

## Parallel execution

Collection workers 24；replay workers 24；infrastructure retries 0。
Collection wall-clock 95.6 min；replay wall-clock 134.9 min。
每个 job 独立进程、Newton environment、seed、日志和事务。中央按 seed 排序后确定前 450、episode ID 与 split；正式 cohort 完整冻结后才启动 replay。完整结果（包括 FAIL）从不重跑。

## Oracle replay

Total replayed 450；strict PASS 401；strict FAIL 49；PASS rate 89.11%。
Grasp 414；detach 419；budget timeout 24。
IK failed steps 4743；clipped steps 98173；dwell timeouts 645；forced advancements 645。
FAIL final phase: `{"DONE": 25, "PULL": 3, "TRANSPORT": 16, "DROP": 3, "GRASP": 2}`；termination reason: `{"target_sequence_exhausted": 25, "episode_budget": 24}`。

## Dataset

Raw train/val: 405/45；filtered train/val: 362/39。
Final V1 total 401；excluded replay-fail 49；integrity-invalid 0。
原始 every 10th accepted→val split 在 replay 前确定，过滤后保留 ID 和 split；没有时长、IK、clipping、dwell 或平滑度筛选。
每条 accepted 双视频已完整解码并核对 frame count/FPS/shape；合并核对视频 SHA、JSON schema、有限数值、next-state indexing 和 roundtrip；raw 450 在筛选后保持原样。
Raw root: `/home/rosmontis/Projects/orchardbench/data/orchard_v1_450_raw`。Filtered root: `/home/rosmontis/Projects/orchardbench/data/orchard_v1_replay_verified`（独立 manifest，JSON/video 链接，不复制大量视频）。
只用 filtered train 的所有完整窗口重算 `action_stats_30x32.json`（原 14 active preview dims）；同时用现有未修改 preparation 工具重算 loader 所需 `action_stats.json`（既有 Cartesian contract、7 active dims）。两种表示均保持原定义；val 使用 train stats。
已有 OrchardBenchDataset 对 train/val 各实际读取 3 个样本，验证双视角、30×32 action、1×32 state、mask、finite normalization 和 training fingerprint。

主要验收产物：`report.md`、`dataset_summary.json`、`v1_trainable_manifest.jsonl`、`v1_excluded_manifest.jsonl`。逐条结果、reject diagnostics、overflow、源码哈希、配置、命令与资源记录都保存在本目录。

全部 dataset preparation 完成后停止，没有启动任何训练。
