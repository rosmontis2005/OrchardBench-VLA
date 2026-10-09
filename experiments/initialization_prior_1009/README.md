# 初始化先验与目标空间信息实验

运行目录为仓库根目录，Python 使用 `.pixi/envs/default/bin/python`。生产专家、环境、控制器、物理和数据集均不修改。XR-0 的原 `StrictPlacement` 由现有采集模块加载，默认相邻仓库路径与生产采集相同。

`controller.py` 直接调用生产 `StudentNativeExpert.plan()`，仅构造白名单输入：机器人位姿、桶的已知安装位置、受限位置估计和 `StageFeedback`。C0 接口没有目标测量参数；Cz 只收到初始高度；Cxyz0 只收到初始三维坐标；Cxyz 每五个真实控制步接收一次三维坐标。低信息组抓持后统一假设果实中心在 TCP，无在线偏移估计。共享特权反馈包括规划果实 held、detached、held15、任务完成标记及果实速率标量。阶段反馈不包含位置、方向或速度向量。

`scenes.py` 先用原规划器选取目标和站位，再在原局部 y 方向平移底座 ±0.08 m；yaw、果实、树、关节、物理参数不变。运行时只在当前 worker 的 reset 作用域覆盖已导入的 planner 返回值，随后执行原环境的 world 构造。不会对物理状态 teleport。出生时 chassis / bucket / wheels / home arm 代理几何与树、果实重叠作为初始化无效依据；端点 PoseIK 使用独立求解器，只作诊断，不根据局部求解失败排除场景。全部已选场景均运行并报告。

校准集为 8290000 起的前 40 个原规划器可行场景；开发集为 8300000 起的前 20 个；正式集为 8400000 起的前 50 个。名单不依据控制结果筛选。C0 的均值模板与 Cz 的线性回归只使用校准集，开发和正式评测不重新拟合。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.pixi/envs/default/bin/python
$PY experiments/initialization_prior_1009/test_information.py
$PY experiments/initialization_prior_1009/scenes.py --group calibration
$PY experiments/initialization_prior_1009/scenes.py --group round0
$PY experiments/initialization_prior_1009/scenes.py --group development
$PY experiments/initialization_prior_1009/run.py --cohort round0 --name round0_reference --modes original Cxyz --workers 6 --shadow
$PY experiments/initialization_prior_1009/run.py --cohort round0 --name round0_smoke --limit 2 --workers 6
$PY experiments/initialization_prior_1009/run.py --cohort development --name round1 --workers 6
$PY experiments/initialization_prior_1009/run.py --cohort development --name round2 --initializations G- G+ --workers 12
# 在审阅开发结果、冻结 config.json 后才进入正式评测
$PY experiments/initialization_prior_1009/scenes.py --group test
$PY experiments/initialization_prior_1009/run.py --cohort test --name round3 --initializations G0 G- G+ --workers 12
$PY experiments/initialization_prior_1009/analyze.py
$PY experiments/initialization_prior_1009/audit.py round0_reference
```

名单生成禁止覆盖已有文件；运行器复用已完成 result.json，配置变化时拒绝恢复。代码／环境异常停止派发新任务，等待在途任务结束；物理任务失败正常保存并继续。逐步日志保存在本地 `artifacts/initialization_prior_1009/<batch>/<seed>_<mode>_<init>/steps.jsonl.gz`，不上传视频或逐步缓存。日志将诊断真值与规划估计分开记录。

MPS 可由外部设置 `CUDA_MPS_PIPE_DIRECTORY` 接入本实验独立服务；不修改 GPU compute mode，不操作正式采集服务。报告记录实际吞吐和选择的并发数。

`analyze.py` 输出全部尝试及共同有效场景的结果，单格成功率使用 Wilson 95% 区间；所有差异按同一基础 seed 配对，以基础场景为单位做 20,000 次 bootstrap。三种初始化平均的信息增益也先在每个 seed 内平均，再按 seed 重采样。差异区间未作多重比较校正；不能以“不显著”宣称等效。

正式批次已完成。四个交付文件位于 `artifacts/initialization_prior_1009/`。已有 `scene_manifest.json` 时，运行器可直接读取其中名单，无需重新生成 cohort 文件。正式日志抽查入口：`audit.py round3 --sample-per-cell 2`；`analyze.py` 后运行 `report.py` 可再生成报告。报告中的研究解释对应本次冻结运行。报告阶段增加了精确 McNemar 敏感性检查，仅用于解释稀疏二元配对；主要差异区间仍是预先固定的场景 bootstrap。
