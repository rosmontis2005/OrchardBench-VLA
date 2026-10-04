# OrchardBench V1 — 2650 expert collection and frozen replay filtering

正式 raw cohort：2650 accepted；strict replay PASS 2374，FAIL 276；最终 V1 2374 条。未加载模型、未下载 checkpoint、未做 gradient update。

## Frozen protocol

Source of truth: `log/gate1_transport_30s_revalidation`，reach-conditioned 26/30、grasp/detach 30/30。已逐项核对源码 SHA。
原始 frozen replay SHA256: `6142e7a9c1b84168f804c0567e99271470fbfd27b90cb81ff05efe42ca54acbb`。
Collector SHA256: `8d1ecbe15023664aa1bd4aa05d190ab664d1ee934f5a074a03f59806e773420d`；git `df73ee4e804e8aa3eb3f97aab471276db30a0838`。
TRANSPORT velocity_accel vmax=0.5 rad/s, amax=20 rad/s²；其余 phases legacy；detach_force_scale=1.5；真实 ego/wrist RGB，30 Hz，measured proprio/next-state actions。
Replay：position≤0.01 m、SO(3)≤0.08 rad、maximum dwell=30；旧 600-step 默认显式覆盖为 900 steps / 30 s。reach OR maximum-dwell fallback，每次推进一个目标。无 passed-waypoint、lookahead、插值、额外 terminal hold、recovery 或 replanning。
GT encode_window → OrchardActionAdapter → to_native(live obs) → OrchardVLAEnv.step；PASS 只取当前 strict evaluator info.success。

## Collection

Total attempts 6593；正式 expert accepted 2650；expert rejected 3933；并发 overflow accepted 10（单独保留，不进入正式 cohort）。
全部完成 attempt 的 acceptance rate 40.35%（包含 overflow；未因 replay FAIL 补采）。
正式 attempt seed 2010000–2016592；2650th accepted seed 2016561。独立 pilot 3 accepted，不计入 2650。
Rejection reasons: `{"first_attempt_no_fruit_at_detection": 2985, "branch_break": 362, "first_attempt_grasp_stalled": 278, "target_not_visible_in_policy_observation": 83, "no_feasible_fixed_base_setup": 59, "planned_target_not_picked": 64, "incidental_detach": 32, "base_drift_exceeded": 11, "premature_detach": 49, "first_attempt_reach_stalled": 1, "first_attempt_missed_bucket": 9}`。

## Parallel execution

Collection workers 32；replay workers 32；infrastructure retries 0。
Collection wall-clock 183.5 min；replay wall-clock 127.6 min。
每个 job 独立进程、Newton environment、seed、日志和事务。中央按 seed 排序后确定前 2650、episode ID 与 split；正式 cohort 完整冻结后才启动 replay。完整结果（包括 FAIL）从不重跑。
本批使用私有 CUDA MPS 服务调度 CUDA clients，启动前已确认设备支持；控制参数和源码不变，没有改变 GPU compute mode。运行环境、服务日志和逐 job 环境变量见 runtime_environment.json、mps/、launch_commands.jsonl。参考：[NVIDIA MPS 接口](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html)。
复用 450 批次已通过的相同配置 3-accepted pilot；本批未新增 pilot attempts，也未将 pilot 纳入 cohort。

## Oracle replay

Total replayed 2650；strict PASS 2374；strict FAIL 276；PASS rate 89.58%。
Grasp 2481；detach 2496；budget timeout 161。
IK failed steps 27886；clipped steps 589494；dwell timeouts 4004；forced advancements 4001。
FAIL final phase: `{"DONE": 115, "TRANSPORT": 108, "PULL": 38, "DROP": 8, "GRASP": 6, "REACH": 1}`；termination reason: `{"target_sequence_exhausted": 115, "episode_budget": 161}`。

## Dataset

Raw train/val: 2385/265；filtered train/val: 2144/230。
Final V1 total 2374；excluded replay-fail 276；integrity-invalid 0。
原始 every 10th accepted→val split 在 replay 前确定，过滤后保留 ID 和 split；没有时长、IK、clipping、dwell 或平滑度筛选。
每条 accepted 双视频已完整解码并核对 frame count/FPS/shape；合并核对视频 SHA、JSON schema、有限数值、next-state indexing 和 roundtrip；raw 2650 在筛选后保持原样。
Raw root: `/home/rosmontis/Projects/orchardbench/data/orchard_v1_2650/raw`。Filtered root: `/home/rosmontis/Projects/orchardbench/data/orchard_v1_2650/filtered`（独立 manifest，JSON/video 链接，不复制大量视频）。
只用 filtered train 的所有完整窗口重算 `action_stats_30x32.json`（原 14 active preview dims）；同时用现有未修改 preparation 工具重算 loader 所需 `action_stats.json`（既有 Cartesian contract、7 active dims）。两种表示均保持原定义；val 使用 train stats。
已有 OrchardBenchDataset 对 train/val 各实际读取 3 个样本，验证双视角、30×32 action、1×32 state、mask、finite normalization 和 training fingerprint。

主要验收产物：`report.md`、`dataset_summary.json`、`v1_trainable_manifest.jsonl`、`v1_excluded_manifest.jsonl`。逐条结果、reject diagnostics、overflow、源码哈希、配置、命令与资源记录都保存在本目录。

全部 dataset preparation 完成后停止，没有启动任何训练。
