# OrchardBench VLA interface

The single-world `OrchardVLAEnv` runs the official Newton 1.3 CUDA physics,
MuJoCo CG, robot/tree collisions, foliage, apples and breaking. It constructs
no AutoPicker. Neither PyTorch nor Transformers is required in Pixi.

```python
from treesim.vla_env import OrchardVLAEnv, VLAEnvConfig

env = OrchardVLAEnv(VLAEnvConfig(grasp_mode="contact"))
try:
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(
        [0.002, 0, 0, 0, 0.01, 0, 1])
finally:
    env.close()
```

## Observations and reset

Both `rgb_static` and `rgb_wrist` are actual SensorTiledCamera RGB, uint8 HWC,
default `(144, 192, 3)`. They work with ViewerNull, without a GL framebuffer.
Packed uint32 color is decoded by Newton's RGBA utility and alpha is discarded.
The fixed static camera sees the robot and canopy; the wrist camera moves with
the hand. Arrays returned to callers are copies.

`proprio` is float32[30]: TCP xyz, TCP quaternion xyzw, total finger width,
9 joint positions, 9 joint velocities, base xyz, base yaw. The corresponding
explicit fields are `tcp_pos_world`, `tcp_quat_world`, `tcp_euler_world`
(extrinsic xyz), `gripper_width`, `joint_pos`, `joint_vel`, `base_pos`,
`base_yaw`, and `sim_time`. Distances are metres, angles radians.
Fruit truth, segmentation, targets and AutoPicker state are absent from obs.
Language is supplied separately by the model wrapper.

`reset(seed=...)` destroys owned simulation/controller/sensor references and
rebuilds everything. Initial RGB is rendered immediately, at time zero and
home joints, without hidden physics steps. Same-seed state/RGB determinism and
different-seed geometry are tested. The initial total finger width is 0.04m.

## Actions and episode

Native shape[7] is `[dx,dy,dz,droll,dpitch,dyaw,gripper]` in world coordinates,
metres/radians. For nonzero Cartesian actions, position uses measured TCP plus
delta and orientation uses `R_new = R_xyz(delta) * R_current`. A zero Cartesian
action holds the last accepted setpoint. Grip positive opens, negative closes,
zero retains the command. Open width is 0.08m (two 0.04m fingers).

Default 60Hz simulation / action_repeat=2 gives 30Hz control. Full quaternion
PoseIK uses independent position, rotation and joint-limit objectives; all
three rotation axes are tested. Per-axis deltas clip at 0.02m / 0.05rad, followed
by chassis workspace limits, IK residual checks and joint target slew limits.
NaN/Inf actions raise before mutation. Rejected IK holds the previous arm
target; `info` records errors, clipping, failure and requested/executed targets.
The VLA arm controller adds existing MuJoCo generalized bias feedforward,
bounded by original effort limits, to avoid position-servo gravity sag.
Original physical parameters, servo gains and AutoPicker remain unchanged.

The episode truncates after 300 control steps by default. Evaluation success
is **at least one physically detached apple**; this terminates and rewards 1
on first success, otherwise reward is 0. Detach, branch break and bucket counts
are privileged evaluation fields in `info`, never policy observations.

## Grasp modes

`contact` only controls fingers and never calls AppleField.hold. Stable fruit
transport through pure contact has not been established by the interface test.

`benchmark_assist` requires a close command, actual contacts from **both**
fingers with the same apple, and its center within a strict hand-local volume.
Only one unambiguous candidate may trigger the benchmark spring attachment.
No intended, nearest, visible, language or AutoPicker target is used. The ID
and trigger are debug/info only. Empty-contact rejection is tested; positive
fruit attachment is not yet validated. This is a simulator grasp abstraction.

## CALVIN / XR-0 boundary

`get_calvin_state()` or `calvin_state(obs)` returns separate float32[32]:
world xyz, Euler xyz, total width, then 25 zeros. Native proprio is unchanged.

`CalvinActionAdapter.physical_delta(a)` clips normalized xyz/Euler to [-1,1]
and divides by 50 / 20. Thus `[1,0,0,0,0,0,1]` gives +0.02m x/open, and
`[0,0,0,1,0,0,-1]` gives +0.05rad roll/close. `to_native()` preserves local
CALVIN `use_target_pose=true`: accumulate desired xyz/Euler and convert the
absolute target orientation to a world spatial delta relative to measured TCP.
Call adapter.reset(obs) on each environment reset. Environment safety clipping
still applies, so rejected/clipped commands need not reach CALVIN's target.

The smoke runner uses existing XR-0 Python in a separate local subprocess,
JSON pipes and lossless PNG files. It reproduces local evaluator prompt,
0.95 center crop, task `calvin_abcd_orig`, two PIL RGB views, state and language.
Checkpoint action_length says 30, but the actual CALVIN mask and runtime output
are `[1,10,32]`; decoded first seven dimensions are normalized CALVIN actions.
It executes three actions then observes/replans, for three cycles by default.
No success at harvesting is assumed for the CALVIN checkpoint in this domain.

```bash
pixi run python scripts/test_vla_env.py
pixi run python scripts/xr0_orchard_smoke.py --cycles 3 --actions-per-cycle 3
```

The latter accepts `--model`, `--model-python`, and `--output`. Defaults point
to the existing local Xiaomi checkpoint and xr0-mibot environment. It loads
BF16 offline, without modifying Pixi dependencies or quantizing weights.
The recorded validation covers interface transport and short safe motions;
it is not a policy-quality or long-episode harvest evaluation.
