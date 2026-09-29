# Arm retiming pilot

**ARM RETIME PILOT: FAIL**

**Legacy regression failed; no velocity-only or acceleration-limited simulator experiments were run.** The failure is numerical trajectory agreement on the known extreme seed, not task success. All three legacy reruns were accepted and retained the exact canonical state-transition frames. No candidate limits were evaluated or selected.

## Scope and preserved inputs

Local HEAD at task start: `26a4b97 refine data collection` (`26a4b97ddb71aacc0677549b9f4aa0b45399ec18`). Initial status and complete user diff are saved alongside this report. Existing user edits to `vla_env.py`, `xr0_adapter.py` and untracked action files were preserved. Only the opt-in expert integration and standalone limiter/pilot tools were added. No XR-0 files, canonical data, physics, fingers, cameras, acceptance criteria, PULL retract increment, timeouts or STALL_FRAMES were changed. No commit/push.

Each rerun called `collect_episode(... record_rgb=False, detach_force_scale=1.5, arm_motion_profile=legacy)`. Canonical reset-time stance planning, clean rebuild, visibility gates, detach diagnostics, success and rejection gates all ran unchanged. Visibility sensors are still instantiated for the existing acceptance check; no video frames were recorded or encoded.

## Verified local architecture

Original `_set_arm` stores an IK goal; `_slew_arm` computes `d = clip(q_goal - q_cmd, -0.045, 0.045); q_cmd = q_cmd + d` once per 60 Hz frame, nominal 2.7 rad/s per joint. FixedBaseAutoPicker calls it after phase update. IK update periods are REACH 15, GRASP 8, PULL 10, TRANSPORT 15 physics frames. PULL retract remains 0.0022 m/frame after 25 settle frames.

VLAEnv remains separate and unchanged: 0.02 m per translation axis/control step, 0.05 rad per relative Euler axis/control step, 0.045 rad/joint/physics frame; 60 Hz physics with repeat 2 gives 30 Hz control. It adds Cartesian clipping before IK whereas expert directly generates IK joint goals. This report does not change the student controller.

## Opt-in implementation and numerical tests

`treesim/arm_motion.py` provides `ArmMotionProfile` and `JointMotionLimiter`, with no phase input. A missing picker profile executes the previous full nine-joint operation unchanged. Explicit profiles replace only its first seven outputs; `_fingers()` and the two finger outputs retain the old path. Command velocity state survives `_goto` and `_set_arm`; reset occurs only at initialization or explicit limiter reset.

Velocity-acceleration mode uses a conservative discrete stopping bound `sqrt((a*dt/2)^2 + 2*a*|error|) - a*dt/2`, then limits velocity change by `a*dt`. If a suddenly replaced goal lies inside stopping distance, no-crossing and bounded deceleration can conflict. The explicit landing guard snaps only the crossing joint to goal and zeros its velocity, recording `last_landed` and `crossing_count`; this exception is exposed for analysis, not silently treated as a guaranteed hard acceleration bound. No experimental candidate was run.

Six numerical tests passed (`unit_tests.log`): bit-exact legacy goal sequences including custom rate; velocity bounds/landing; acceleration ramps/stops/reversal; near-goal crossing exception; validation/copy isolation; actual AutoPicker method integration with identical finger outputs and no velocity reset on phase transition. These establish arithmetic equivalence, but do not replace canonical simulator regression.

## Legacy regression gate

The limits were declared before comparison: phase timing ±1 physics frame, recorded length ±1 observation, max coordinate TCP difference 1 mm, arm joint 0.002 rad, SO(3) orientation 0.002 rad, measured gripper width 1 mm. Pull-force tolerance is max(0.1 N, 1% of canonical); detach fields use their explicit field tolerances in the runner. Samples are compared at identical timestamps without time warping.

| Episode / seed | Accepted | Phase frame differences | TCP max coord error m | Joint max error rad | Orientation max error rad | max pull N canonical → rerun | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| episode_000222 / 1000576 | True | [0, 0, 0, 0, 0, 0] | 2.95043e-06 | 1.57356e-05 | 1.44066e-05 | 24.1313 → 24.1313 | PASS |
| episode_000134 / 1000356 | True | [0, 0, 0, 0, 0, 0] | 6.85453e-06 | 3.82662e-05 | 2.66345e-05 | 25.6937 → 25.6939 | PASS |
| episode_000029 / 1000074 | True | [0, 0, 0, 0, 0, 0] | 0.00390899 | 0.0554748 | 0.0507526 | 37.0894 → 37.9918 | FAIL |

All three: identical state chain REACH→GRASP→PULL→TRANSPORT→DROP→DONE, identical recorded frame count, placed=True, branch breaks=0, detach diagnostics within the declared tolerances. Median/high seeds show micron-level TCP and tens-of-microradian joint differences. The extreme seed exceeds joint/orientation and TCP tolerances and changes max pull force by about 2.43%; these are not dismissed as harmless rounding.

For seed 1000074, the largest TCP coordinate discrepancy is in PULL at t=1.3667 s (3.909 mm); the largest arm-joint discrepancy is in DROP at t=3.2 s (0.055475 rad). REACH differs by less than 0.84 µm TCP and 1.55 µrad joint; GRASP and subsequent dynamics amplify divergence. The deterministic numerical legacy tests and unchanged default expression do **not** establish why this rerun differs from the historical canonical data. Simulator/contact/IK numerical sensitivity is a possibility, not a demonstrated cause. No original-source control rerun or sweep was performed after the gate failure.

The experiment stopped at this gate as required. The threshold was not relaxed and no easier replacement seed was substituted.

## Deterministic seed selection and unrun sweep

| Smoke role | Episode | Seed | Old P95 TCP m/s | TRANSPORT path m |
| --- | --- | --- | --- | --- |
| short_transport_path | episode_000040 | 1000106 | 1.85264 | 1.30422 |
| lowest_episode_p95_speed | episode_000106 | 1000268 | 1.70476 | 1.62322 |
| median_speed | episode_000222 | 1000576 | 2.40238 | 1.44592 |
| longest_transport_path | episode_000090 | 1000235 | 2.72395 | 2.14152 |
| high_speed_p90 | episode_000134 | 1000356 | 2.6773 | 1.7091 |
| known_extreme | episode_000029 | 1000074 | 3.55672 | 2.02462 |
| speed_p25 | episode_000088 | 1000232 | 2.07053 | 1.39188 |
| speed_p75 | episode_000171 | 1000441 | 2.57865 | 1.70609 |

Thirty expansion seeds were selected before outcomes using a 3×3 TRANSPORT-path / episode-P95-speed rank grid, preserving all eight smoke anchors. Full IDs and seeds are in `pilot_seed_selection.json`. No 8-seed smoke or 30-seed expansion was started. The predeclared catastrophic-smoke rule is ≤2/8 accepted or ≤3/8 detached; it was never applied because legacy gating failed.

| Profile | Mode | vmax rad/s | amax rad/s² | Simulator attempted | Accepted | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| legacy | legacy | 2.7 | None | 3 | 3 | legacy gate only |
| v2.0 | velocity | 2.0 | None | 0 | 0 | not run: regression gate failed |
| v1.5 | velocity | 1.5 | None | 0 | 0 | not run: regression gate failed |
| va1.5_a20 | velocity_accel | 1.5 | 20 | 0 | 0 | not run: regression gate failed |
| va1.5_a10 | velocity_accel | 1.5 | 10 | 0 | 0 | not run: regression gate failed |
| va1.2_a10 | velocity_accel | 1.2 | 10 | 0 | 0 | not run: regression gate failed |

## Measured motion retained from the three gate runs

These distributions pool only the three rerun episodes. They are not 8-/30-seed profile estimates or evidence that retiming improves success. Main metrics use actual 30 Hz `proprios`: translation difference/dt; SO(3) Log(RᵀR_next)/dt; joint position difference/dt. Angular acceleration transports rotvec velocity into a common world frame before differentiation. Derivatives are raw; no smoothing. Phase labels use interval starts, odd physics-frame boundaries are mixed old/new intervals assigned to the old phase, exactly as in the audit.

| Phase | TCP P50 | P90 | P95 | P99 | max m/s | acc P95 m/s² | acc max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| REACH | 0.843997 | 1.92982 | 2.15404 | 2.26561 | 2.30698 | 14.0346 | 15.6424 |
| GRASP | 0.585462 | 0.727044 | 0.755989 | 1.01545 | 1.14234 | 9.67878 | 10.7457 |
| PULL | 0.124668 | 0.261069 | 0.35834 | 0.705831 | 0.732104 | 6.79133 | 10.291 |
| TRANSPORT | 1.726 | 3.00497 | 3.74863 | 4.1391 | 4.2503 | 26.7051 | 30.7983 |
| DROP | 0.0111295 | 0.228525 | 0.310125 | 0.398187 | 0.438492 | 4.16993 | 6.75938 |

| Phase | angular P50 | angular P95 rad/s | alpha P95 rad/s² | joint FD P50 | joint FD P95 rad/s | joint acc P95 rad/s² |
| --- | --- | --- | --- | --- | --- | --- |
| REACH | 1.03429 | 2.29702 | 23.4548 | 2.10769 | 2.75654 | 17.1415 |
| GRASP | 1.70075 | 3.18414 | 34.9912 | 2.29868 | 3.2992 | 35.8086 |
| PULL | 0.0879956 | 1.9752 | 14.235 | 0.396688 | 2.78447 | 20.6872 |
| TRANSPORT | 4.55736 | 7.26174 | 35.8886 | 2.64626 | 3.21026 | 20.2383 |
| DROP | 0.206707 | 1.85696 | 17.3009 | 0.206349 | 1.90105 | 11.0376 |

## PULL→TRANSPORT and command evidence

Alignment uses ceil(physics boundary/2) as first fully new-phase transition, [-10,+20] recorded steps, and five fully-before/fully-after intervals for paired means. No absolute Euler differencing.

| Metric | median pre | median post | median paired ratio | fraction post>pre | fraction post>2×pre |
| --- | --- | --- | --- | --- | --- |
| tcp_speed | 0.1384 | 1.66897 | 11.0287 | 1 | 1 |
| angular_speed | 0.0705344 | 3.1664 | 35.6387 | 1 | 1 |
| joint_speed | 0.408889 | 1.45388 | 3.55568 | 1 | 0.666667 |
| linear_acceleration | 1.48654 | 16.0434 | 10.7925 | 1 | 1 |

| Seed | largest q_goal jump first 5 frames (rad norm) | largest goal-command gap (rad norm) | early command speed max rad/s |
| --- | --- | --- | --- |
| 1000074 | 2.73662 | 2.637 | 2.7 |
| 1000356 | 2.46631 | 2.36666 | 2.7 |
| 1000576 | 2.28297 | 2.18815 | 2.7 |

The 60 Hz command traces on all three representative seeds directly show a large q_goal replacement followed by saturated 2.7 rad/s command motion and measured acceleration. This supports the proposed mechanism for legacy behaviour on these runs; no acceleration-limited ramp was tested. The phase-change snapshot precedes the TRANSPORT IK update by one physics frame. Command traces are after expert.update, while measured states are after the preceding physics step, so the new command influences subsequent measurements.

## PULL preservation and failures

Only a legacy-versus-canonical comparison is available; retimed PULL preservation is **not evaluated**. Both motion tables and scalar pairs retain all attempted runs; candidate profiles have count=0/null, never zero-valued fabricated motion.

| PULL scalar | paired n | canonical P50 | legacy P50 | paired difference P50 |
| --- | --- | --- | --- | --- |
| pull_duration_s | 3 | 1.4 | 1.4 | 0 |
| retract_distance_at_detach_m | 3 | 0.1276 | 0.1276 | 0 |
| tcp_displacement_since_pull_enter_m | 3 | 0.120375 | 0.120374 | -3.27826e-07 |
| detach_frame | 3 | 125 | 125 | 0 |
| pull_frames_before_detach | 3 | 84 | 84 | 0 |
| max_pull_N | 3 | 25.6937 | 25.6939 | 0.000200272 |

Legacy detach=3/3; premature_detach=0; accepted=3/3. No reach/grasp stalls, no-fruit, branch-break, timeout or missed-bucket task rejections were observed in these three gate runs. The failed gate is a regression discrepancy, not an episode rejection. Any future failed run is retained by the runner, including its final odd physics-frame sample in command_trace; main 30 Hz trajectory may end half a frame before that sample. The inherited collector `trajectory_type` string is not used to determine acceptance: result.accepted is authoritative.

## Watchdogs and path/time evidence

No watchdog was changed. Actual successful phase durations and time remaining to the original >300/>240/>360/>420-frame timeouts are below; STALL_FRAMES stays 55. TRANSPORT stall can advance to DROP rather than produce a `transport_stalled` rejection, so lack of that rejection alone is not proof of settling. Per-frame observed stall counters are retained.

| Phase | minimum successful timeout margin (frames) | median margin | max margin |
| --- | --- | --- | --- |
| REACH | 261 | 288 | 291 |
| GRASP | 180 | 220 | 224 |
| PULL | 263 | 276 | 328 |
| TRANSPORT | 358 | 361 | 366 |

Legacy TRANSPORT path-duration Spearman ρ=-0.5; path-mean-speed ρ=1; n=3. This selected three-seed cohort cannot establish a retiming correlation improvement, and is not comparable to the full-300 ρ without matching cohorts. No new-profile timeout effects can be assessed.

## Tradeoffs and human review

No retiming Pareto comparison is possible: velocity-only and acceleration-limited profiles have not been experimentally evaluated. Therefore REACH/GRASP/TRANSPORT slowdown, acceleration-smoothing benefit, candidate task success and PULL changes are unknown. No profile is recommended as a final controller or as an empirically preferred setting.

**Recommended profiles for HUMAN REVIEW: none on performance evidence.** Review the legacy mismatch, opt-in implementation, and recorded command mechanism first. The six configurations remain coarse planned probes only. This task ends here; no automatic adjustment, timeout extension, extra sampling, training or sweep follows.

## Artifacts and reproduction

- `legacy_regression.json`: per-field checks and canonical/pilot detach diagnostics.
- `pilot_seed_selection.json`: exact seed lists, selection metrics, planned probes and threshold.
- `runs/legacy/<seed>/run.json`: full measured trajectory, canonical acceptance result, 60 Hz commands and terminal failure sampling.
- `pilot_runs.csv`: one row per actually attempted run (three).
- `profile_phase_statistics.json`, `profile_transition_statistics.json`, `pull_preservation.json`, `failure_summary.json`, `command_watchdog_diagnostics.json`: actual metrics; untested candidates explicitly empty.
- `unit_tests.log`, `input_hashes_before.json`, `source_integrity.json`, `initial_git_status.txt`, `initial_user_changes.diff`: validation/provenance.

Numerical tests: `.pixi/envs/default/bin/python scripts/test_arm_motion.py`. Read-only analysis of retained runs: `.pixi/envs/default/bin/python scripts/analyze_arm_retime_pilot.py`. The runner checks `legacy_regression.json` before smoke/expansion and refuses this failed gate. No candidate run is enabled by analysis.

![legacy_regression_errors](figures/legacy_regression_errors.png)

![profile_acceleration](figures/profile_acceleration.png)

![profile_joint_speed](figures/profile_joint_speed.png)

![profile_tcp_speed](figures/profile_tcp_speed.png)

![pull_preservation](figures/pull_preservation.png)

![pull_transport_alignment](figures/pull_transport_alignment.png)

![qgoal_qcmd_measured_trace](figures/qgoal_qcmd_measured_trace.png)

![success_vs_speed](figures/success_vs_speed.png)
