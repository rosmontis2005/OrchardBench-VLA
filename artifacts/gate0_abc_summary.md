# Gate 0 A/B/C

Repo HEAD: `26a4b97ddb71aacc0677549b9f4aa0b45399ec18`

本轮修改：

- `treesim/vla_env.py`：reset 复用 deterministic `plan_fixed_base_stance` 和 `build_fixed_base_world`，使用 collector 的初始几何配置、直接 spawn/weld、clean t=0；无可行 stance 时抛出包含 seed/rejection reasons 的 RuntimeError。不构造或运行 expert。planner metadata 仅在 `info.reset_stance_debug`。
- 同文件：默认 `benchmark_assist`；保留可选 `contact`。沿用双指真实 collision contact、hand-local volume 和唯一候选门控，复用 AppleField.hold/release。持有后不重新选果；明确 open 才 release。连续宽度与上一次命令比较，避免恒定命令因手指物理偏差误释放。
- 同文件：沿用 bucket detector，`success = apple_in_bucket_count > 0`；仅 placement 终止并奖励 1，单纯 detach 继续。保留 max-step truncation 和现有安全检查，无新失败终止规则。info 保留 detached/bucket/branch counts、assist trigger，新增 `held_apple_id_debug`，保留旧 debug alias。
- `scripts/test_gate0_abc.py`：仅 A/B/C 三组 GPU smoke tests。
- `scripts/test_vla_env.py`：原 contact regression 显式指定 contact（本轮未运行这个完整旧测试）。
- `artifacts/gate0_abc_summary.md`：本记录。

实际测试命令（仓库根目录）：

```bash
.pixi/envs/default/bin/python scripts/test_gate0_abc.py A
.pixi/envs/default/bin/python -u scripts/test_gate0_abc.py B
.pixi/envs/default/bin/python -u scripts/test_gate0_abc.py C
```

| 测试 | 实际结果 |
| --- | --- |
| A | PASS / exit 0。seed 1000106、1000576 均 feasible；VLA base position/yaw 与独立 collector-config planner 一致，t=0，arm home，零初始 detach/break，RGB/proprio 和原 observation keys 有效。无 expert actions。 |
| B | PASS / exit 0。无双指接触时 close 不 attach。受控 detached apple 0（不同于 planner target 16）由真实 Newton collision 得到双指接触；close 触发 hold，随后持续 6 physics frames、不再次触发；手部局部坐标约 (-0.000842, 0.00000047, 0.100349) m；连续宽度 open 释放并清空持有状态。 |
| C | PASS / exit 0。同一个 episode 中，detached apple 桶外：success=False / terminated=False / truncated=False / reward=0；移入桶内并真实 step 后：success=True / terminated=True / truncated=False / reward=1。 |

B/C 是直接放置一个 detached apple 的受控夹具，不代表端到端 policy 采摘成功。C 初次夹具仅写一个 CUDA graph 状态缓冲区，位置在 replay 后被覆盖；修正为两个缓冲区一致后通过，并重跑共用夹具的 B，通过。未修改 simulator/physics。

Remaining blockers / limits：

- A/B/C smoke 均已通过。
- 仍有已有的物理配置差异：`../data/orchard_autopicker_v1/collection_config.json` 的 `detach_force_multiplier=1.5`，VLA reset 的 TreeConfig 默认 `detach_force_scale=1.0`。本轮明确禁止修改 stem multiplier，因此保留。它不阻止训练程序启动，但在宣称 oracle replay 与 canonical 数据的完整物理契约一致之前，需要单独处理。
- 未执行 oracle replay 或 XR-0 training；本轮不对端到端 replay 可执行性作结论。

原工作树已有改动保留；未修改 canonical 数据、采集器、expert、动作表示、30-step chunk、图像处理、语言、控制器限幅/IK tolerance/arm profile 或果树参数，未进行训练、重采集、retiming、recovery 或 seed sweep。
