#!/usr/bin/env python
"""Fixed-base expert validation (Tests 1 and 2 of the fixed-base round).

    # Test 1: privileged stance selection + clean t=0 world, several seeds
    pixi run python scripts/test_fixed_base_expert.py init --seeds 910000 910001 ...

    # Test 2: one full episode with a per-transition state trace
    pixi run python scripts/test_fixed_base_expert.py trace --seed 910001

No training, no dataset writing.  Everything reported here is expert / debug
bookkeeping (fruit truth, target identity, IK residuals) and is never a
policy observation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation

OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "fixed_base_expert"


def episode_config(seed: int):
    """The collector's episode configuration (identical to the dataset run)."""
    from treesim.config import TreeConfig
    cfg = TreeConfig.compliant("apple")
    cfg.seed = int(seed)
    cfg.device = "cuda:0"
    cfg.lsystem.shape_jitter = 0.0
    cfg.physics.dynamics_jitter = 0.0
    cfg.fruit.enabled = True
    cfg.fruit.max_count = 40
    cfg.foliage.set_density(0.6)
    cfg.breaking.enabled = True
    cfg.robot.enabled = True
    cfg.physics.terrain = False
    return cfg


def base_drift(sim, ch, initial):
    b = sim.body_q_np()[ch]
    trans = float(np.linalg.norm(b[:3] - initial[:3]))
    y = Rotation.from_quat(b[3:]).as_euler("xyz")[2] - Rotation.from_quat(initial[3:]).as_euler("xyz")[2]
    return trans, float(abs(np.arctan2(np.sin(y), np.cos(y))))


def snapshot(sim, tm, picker, qidx, tcp, frame, apple0=None):
    bq = sim.body_q_np()
    q = sim.joint_q_np()[qidx]
    ik = picker.last_ik
    ab = int(tm.apple_bodies[picker.planned_apple])
    live = bq[ab, :3]
    # nearest robot body to the planned apple (which link is shoving it, if any)
    ch = int(tm.robot_data["chassis"][0]); wr = int(tm.robot_data["wrist"][0])
    rb = np.arange(ch, wr + 4)
    dd = np.linalg.norm(bq[rb, :3] - live, axis=1)
    return dict(frame=int(frame), sim_time=round(float(sim.sim_time), 4), state=picker.state,
                tcp_world=[round(float(v), 4) for v in bq[tcp, :3]],
                target_world=(None if picker._target is None else [round(float(v), 4) for v in picker._target]),
                ik_target_world=(None if ik is None else [round(float(v), 4) for v in ik[0]]),
                ik_residual_m=(None if ik is None else round(ik[1], 5)),
                gripper_width_m=round(float(q[-2:].sum()), 5),
                target_apple=int(picker._target_apple), planned_apple=int(picker.planned_apple),
                planned_apple_world=[round(float(v), 4) for v in live],
                planned_apple_displacement_m=(None if apple0 is None else round(float(np.linalg.norm(live - apple0)), 4)),
                nearest_robot_body_to_apple=dict(body=int(rb[dd.argmin()]),
                                                 label=tm.model.body_label[int(rb[dd.argmin()])].rsplit("/", 1)[-1],
                                                 dist_m=round(float(dd.min()), 4)),
                detached=bool(picker._target_apple >= 0 and sim.apples.detached[picker._target_apple]),
                planned_apple_detached=bool(sim.apples.detached[picker.planned_apple]),
                apples_detached_total=int(sim.apples.broken_count),
                branch_break_count=int(sim.breaker.broken_count),
                fail_reason=picker.fail_reason)


def run_init(args):
    import warp as wp
    import newton.viewer
    from treesim import robot
    from treesim.picker import ArmIK
    from treesim.fixed_base_picker import (plan_fixed_base_stance, build_fixed_base_world,
                                           verify_clean_start, FixedBaseAutoPicker)
    from treesim.metrics import Metrics
    rows = []
    with wp.ScopedDevice("cuda:0"):
        ik = ArmIK(list(robot._ARM_HOME.values()))
        for seed in args.seeds:
            cfg = episode_config(seed)
            t0 = time.monotonic()
            plan = plan_fixed_base_stance(cfg, ik=ik)
            row = dict(seed=seed, plan_wall_s=round(time.monotonic() - t0, 3),
                       apple_count=plan.report["apple_count"],
                       reachable_height_count=plan.report["reachable_height_count"],
                       candidate_count=plan.report["candidate_count"],
                       feasible_count=plan.report["feasible_count"],
                       rejection_reasons=plan.report["rejection_reasons"], feasible=plan.feasible)
            if not plan.feasible:
                row["result"] = "no_feasible_fixed_base_setup"
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            st = plan.stance
            row["stance"] = dict(apple_index=st.apple_index, target_world=st.target_world,
                                 base_xy=st.base_xy, base_yaw=st.base_yaw, standoff=st.standoff,
                                 standoff_band=st.standoff_band, azimuth_offset=st.azimuth_offset,
                                 ik_pregrasp_err=st.ik_pregrasp_err, ik_grasp_err=st.ik_grasp_err,
                                 ik_pull_err=st.ik_pull_err, ik_transport_err=st.ik_transport_err,
                                 min_clearance_chassis=st.min_clearance_chassis,
                                 min_clearance_arm_home=st.min_clearance_arm_home,
                                 twig_contacts=st.twig_contacts, score=st.score)
            t0 = time.monotonic()
            tm, sim = build_fixed_base_world(cfg, st)
            viewer = newton.viewer.ViewerNull()
            sim.set_viewer(viewer)
            driver = robot.RobotDriver(sim, tm, cfg.robot)
            row["build_wall_s"] = round(time.monotonic() - t0, 3)
            try:
                row["clean_start"] = verify_clean_start(sim, tm, plan, driver)
                met = Metrics(None)
                picker = FixedBaseAutoPicker(sim, tm, driver, cfg.robot, st, met, ik=ik)
                row["expert_initial_state"] = picker.state
                ch = int(tm.robot_data["chassis"][0])
                initial = sim.body_q_np()[ch].copy()
                # spawn safety: hold the arm at home and let the contact solver
                # act for a short window WITHOUT the expert (this is a test, not
                # an episode) -- the first pilot broke branches within 3-27 frames
                max_t = max_y = 0.0
                for frame in range(1, args.settle_frames + 1):
                    sim.step()
                    t, y = base_drift(sim, ch, initial)
                    max_t, max_y = max(max_t, t), max(max_y, y)
                ds = int(driver.planar_dof[0])
                row["settle"] = dict(frames=args.settle_frames, sim_time=float(sim.sim_time),
                                     branch_break_count=int(sim.breaker.broken_count),
                                     apple_detached_count=int(sim.apples.broken_count),
                                     max_base_translation_drift_m=max_t, max_base_yaw_drift_rad=max_y,
                                     base_velocity_targets_zero=bool(np.all(
                                         driver._target_host[[ds, ds + 1, ds + 3]] == 0.0)))
                ok = (row["settle"]["branch_break_count"] == 0 and row["settle"]["apple_detached_count"] == 0
                      and max_t < 1e-3 and max_y < 1e-3)
                row["result"] = "PASS" if ok else "FAIL"
            except AssertionError as e:
                row["result"] = f"FAIL: {e}"
            finally:
                wp.synchronize_device(sim.model.device)
                viewer.close()
                del sim, tm, driver
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = dict(test="init", seeds=args.seeds, feasible=sum(r["feasible"] for r in rows),
                   passed=sum(r.get("result") == "PASS" for r in rows), rows=rows)
    out = Path(args.output or OUT_DIR / "test1_init.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"[test1] feasible={summary['feasible']}/{len(rows)} passed={summary['passed']}/{len(rows)} -> {out}")
    return 0 if summary["passed"] == len(rows) else 2


def run_trace(args):
    import warp as wp
    import newton.viewer
    from treesim import robot
    from treesim.picker import ArmIK
    from treesim.fixed_base_picker import (plan_fixed_base_stance, build_fixed_base_world,
                                           verify_clean_start, FixedBaseAutoPicker)
    from treesim.metrics import Metrics
    result = dict(test="trace", seed=args.seed)
    with wp.ScopedDevice("cuda:0"):
        ik = ArmIK(list(robot._ARM_HOME.values()))
        cfg = episode_config(args.seed)
        plan = plan_fixed_base_stance(cfg, ik=ik)
        result["plan"] = {k: v for k, v in plan.report.items() if k not in ("candidates", "per_apple", "params")}
        if not plan.feasible:
            result["outcome"] = "no_feasible_fixed_base_setup"
            print(json.dumps(result, indent=1))
            return 2
        st = plan.stance
        tm, sim = build_fixed_base_world(cfg, st)
        viewer = newton.viewer.ViewerNull()
        sim.set_viewer(viewer)
        driver = robot.RobotDriver(sim, tm, cfg.robot)
        result["clean_start"] = verify_clean_start(sim, tm, plan, driver)
        met = Metrics(None)
        picker = FixedBaseAutoPicker(sim, tm, driver, cfg.robot, st, met, ik=ik)
        names = [n.rsplit("/", 1)[-1] for n in tm.model.joint_label]
        starts = tm.model.joint_q_start.numpy()
        qidx = np.array([starts[names.index(n)] for n in robot._ARM_HOME], dtype=int)
        tcp = next(i for i, n in enumerate(tm.model.body_label) if n.endswith("fr3_hand_tcp"))
        ch = int(tm.robot_data["chassis"][0])
        ds = int(driver.planar_dof[0])
        initial = sim.body_q_np()[ch].copy()
        apple0 = sim.body_q_np()[int(tm.apple_bodies[picker.planned_apple]), :3].copy()
        trace = [snapshot(sim, tm, picker, qidx, tcp, 0, apple0)]
        print(json.dumps(trace[-1]), flush=True)
        last = picker.state
        max_t = max_y = 0.0
        outcome = None
        t_wall = time.monotonic()
        per_frame = []          # dense diagnostic: planned apple displacement per frame
        for frame in range(1, args.max_frames + 1):
            sim.step()
            met.frame()
            picker.update()
            assert np.all(driver._target_host[[ds, ds + 1, ds + 3]] == 0.0), "expert issued a base command"
            t, y = base_drift(sim, ch, initial)
            max_t, max_y = max(max_t, t), max(max_y, y)
            if args.dense_every and frame % args.dense_every == 0:
                s = snapshot(sim, tm, picker, qidx, tcp, frame, apple0)
                per_frame.append(dict(frame=frame, state=s["state"], apple_disp=s["planned_apple_displacement_m"],
                                      nearest=s["nearest_robot_body_to_apple"], tcp=s["tcp_world"]))
            if picker.state != last:
                trace.append(snapshot(sim, tm, picker, qidx, tcp, frame, apple0))
                print(json.dumps(trace[-1]), flush=True)
                last = picker.state
            if max_t >= 1e-3 or max_y >= 1e-3:
                outcome = "base_drift_exceeded"; break
            if sim.breaker.broken_count:
                outcome = "branch_break"; break
            if picker.done:
                outcome = "expert_done" if picker.state == "DONE" else f"expert_failed: {picker.fail_reason}"
                break
        else:
            outcome = "timeout"
        wall = time.monotonic() - t_wall
        trace.append(snapshot(sim, tm, picker, qidx, tcp, frame, apple0))
        pick = met.picks[0] if met.picks else met._open_pick
        result.update(outcome=outcome, frames=frame, sim_time=float(sim.sim_time),
                      sim_fps=round(frame / max(wall, 1e-9), 1),
                      pick_record=pick, branch_break_count=int(sim.breaker.broken_count),
                      apples_detached_total=int(sim.apples.broken_count),
                      max_base_translation_drift_m=max_t, max_base_yaw_drift_rad=max_y,
                      states_visited=[s["state"] for s in trace],
                      success=bool(pick and pick["grasped"] and pick["detached"] and pick["placed"]
                                   and not pick["fail_reason"] and sim.breaker.broken_count == 0
                                   and picker.state == "DONE"),
                      trace=trace, dense=per_frame)
        wp.synchronize_device(sim.model.device)
        viewer.close()
    out = Path(args.output or OUT_DIR / f"test2_trace_{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    print(f"[test2] seed={args.seed} outcome={outcome} success={result['success']} frames={frame} "
          f"states={result['states_visited']} breaks={result['branch_break_count']} -> {out}")
    return 0 if result["success"] else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("--seeds", type=int, nargs="+", required=True)
    a.add_argument("--settle-frames", type=int, default=30)
    a.add_argument("--output", default=None)
    b = sub.add_parser("trace")
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--max-frames", type=int, default=1200)
    b.add_argument("--dense-every", type=int, default=0,
                   help="also record the planned apple's displacement every N frames (diagnostic)")
    b.add_argument("--output", default=None)
    args = ap.parse_args()
    return run_init(args) if args.cmd == "init" else run_trace(args)


if __name__ == "__main__":
    sys.exit(main())
