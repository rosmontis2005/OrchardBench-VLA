# Student-native command contract V2

The geometric teacher is `treesim/student_native_expert.py`. It never owns a
joint controller, calls AutoPicker.update(), or writes simulator state. It uses
the existing fixed-base reset planner without altering its sampling or physics.
The task stages are RESET, REACH, GRASP, PULL, TRANSPORT, DROP, DONE/FAILED.
Pregrasp is 0.16 m behind the live fruit along the horizontal base-to-fruit
axis. Pull is bounded to 0.32 m along the reverse axis. Shared transport
waypoints are expressed in the base frame. Fruit/TCP offset and fruit speed
are consulted only at planning boundaries before releasing over the bucket.

At control indices 0, 5, 10, ... the teacher observes geometry and produces five
absolute world TCP targets and absolute gripper widths. Each is consumed once.
`orchard_action.native_command()` converts the cached target against the current
measured robot pose into the existing relative-world translation / extrinsic-XYZ
rotation delta. This conversion uses robot feedback, not fresh fruit truth.
`OrchardVLAEnv.step()` performs its usual limits, PoseIK, joint slew, bias force
and two 60 Hz physics steps. The control clock advances by 1/30 s, irrespective
of tracking error. No dwell or forced advancement is implemented.

The original two-finger contact + palm-volume benchmark assist remains in the
environment. Closing uses requested width 0.0 m; opening uses 0.08 m for at least
five control steps. Measured finger separation is recorded separately. No
teacher hold/release call is allowed. The existing `StrictPlacement` observer
checks the same-fruit held15, detach after grasp, actual release, non-held,
and 60 consecutive in-bucket boundaries. Its existing continuation wrapper
only suppresses early episode termination at legacy bucket entry, allowing
those 60 steps to be observed. It does not alter physical success or physics.

## Stored sequence

Contract: `orchard_cartesian_local_rotvec_width_requested_v2`.

An episode has N real commands and N+1 observations including the terminal
measurement. `observations[i]`, `commands[i]`, and `observations[i+1]` bracket
exactly one env.step call. `commands[i].native` is the exact kwargs passed to
that call. Requested position, rotation matrix and width are retained even if
the controller clips or rejects IK. Controller effective targets, joint
commands, clipping flags, IK errors and tracking errors live in diagnostics.
Two 30 Hz RGB videos contain all N+1 observations. Base pose and timestamps
permit later coordinate conversion; no coordinate-system training decision
is made here.

A complete window starts at an actual H5 boundary and contains commands
[t:t+30], anchored at measured observation t. Translation is R_t^T(p_requested
- p_t); rotation is Log(R_t^T R_requested); width is absolute, not a delta.
Only dimensions 0:7 in the [30,32] tensor are active. No terminal replication,
window crossing, frame removal or time compression is permitted. Partial tails
remain diagnostic records. V1 next-measured-state labels and statistics must
not be mixed with V2.

`orchard_command.policy_sample()` explicitly returns current two RGB views,
current measured state, fixed task instruction, action and mask. Seed, phase,
fruit identity/position/velocity, bucket truth, and outcomes never become
student inputs. This is a lightweight loader boundary, not an XR-0 training
integration or a network change.

## Validation and operation

`collect_student_native.py` validates each completed attempt from saved files:
30 Hz timing, observation counts, every native command, controller clipped
command and effective target, full boundary-aligned SE(3) windows, mask, finite
values, video decoding and an independent replay of the original strict event
observer. This is data correctness, not a claim of deterministic physical
trajectory replay. Failed physics attempts remain in their own directories.

`run_student_native_batch.py` uses independent workers and one manifest writer.
Seeds do not overlap; each worker writes a distinct directory. Resume reads
completed result files and preserves unfinished attempts under an interrupted
name before retrying. A formal collection requires a passing gate_status.json
and the frozen expert source digest. SIGTERM to the manager finishes in-flight
episodes and stops dispatch. A tmux session can keep it independent of SSH.

`summarize_student_native.py OUTPUT --revalidate --stats` rereads recorded
artifacts and produces a batch report and fresh
`action_stats_requested_v2.json`. The statistics source is explicitly
`unsplit_candidates`, not a training split: a future train/validation split
must recompute statistics from training trajectories only. A running or stopped
partial collection is explicitly marked as incomplete.

CPU contract tests: `.pixi/envs/default/bin/python scripts/test_orchard_command.py`.
Experiment results and frozen configurations live in
`artifacts/student_native_1008/`; consult its final report for actual gate and
background-collection status. This document makes no claim that the 2000-episode
collection has started or finished.

The final shared transport timeout is 45 s (the initial 30 s trial is retained
separately). Episode budget is 2100 control steps / 70 s. This is a stage exit
budget, never a per-target dwell: action indices advance at every env.step.
Dataset acceptance separately requires IK and clipping fractions each <=5%
and at most 15 consecutive affected boundaries. These are data usability
checks and do not change StrictPlacement; reports retain strict success even
when a trajectory is rejected for controller quality.

Full per-step records include command plus before/after observations in
`steps.jsonl`, so interrupted attempts retain their executed prefix. Complete
records are also stored as `trajectory.json`. Release diagnostics are sampled
at the actual release control boundary; the after-step velocity includes
motion during that control interval, not just the decision-boundary velocity.
