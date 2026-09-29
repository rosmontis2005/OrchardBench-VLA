# XR-0 OrchardBench post-training V1

完整实现说明、动作契约、全量审计结果、实际 smoke 证据及正式训练命令见：

[XR-0 TRAINING_NOTES.md](../dualsys/Xiaomi-Robotics-0/xr0/TRAINING_NOTES.md)

共享动作实现是 `treesim/orchard_action.py`；连续夹爪入口是
`OrchardVLAEnv.step(action, gripper_width=...)`。请使用新的 Orchard adapter，
不要把局部 rotation-vector 动作交给原 CALVIN adapter。

已验证 300 accepted（270 train / 30 val）、真实前向/反向/optimizer step、
20 步小样本诊断和三次控制接口调用。未运行正式训练或闭环策略评估。
