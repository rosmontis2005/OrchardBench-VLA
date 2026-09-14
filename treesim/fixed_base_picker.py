"""Fixed-base harvesting expert: a privileged reset-time target + stance, then the
official AutoPicker manipulation backend (REACH -> GRASP -> PULL -> TRANSPORT -> DROP).

The official :class:`treesim.picker.AutoPicker` is a MOBILE manipulation expert.
Its SCAN / ALIGN / ORBIT / RECOVER states discover fruit with the wrist camera
and drive the Ridgeback into a stand-off band facing the fruit, and its stuck
watchdog assumes that a commanded base velocity produces displacement.  A data
collection task whose chassis may never translate or rotate cannot run those
states honestly: with the base welded, every ALIGN request looks like a wedge
and the machine recovers/orbits forever (the 82 % ``recovery at frame ~50``
outcome of the first pilot).

This module moves what the mobile states were responsible for -- target
discovery, chassis placement, stand-off adjustment -- into a **privileged,
physics-free planning pass before the episode**, and runs only the manipulation
states inside the recorded episode:

* :func:`plan_fixed_base_stance` regenerates the seed's skeleton and apple
  placements with the SAME generators the builder uses (no Newton model, no
  dynamics), enumerates candidate (fruit, chassis x/y/yaw) pairs and rejects the
  infeasible ones: fruit height, spawn clearance of chassis / bucket / wheels /
  the home-pose arm against wood and fruit, IK at the pre-grasp, grasp, pull and
  bucket poses with margins tighter than the runtime watchdogs, thick wood in
  the approach corridor and around the arm at the pre-grasp / grasp poses.  It
  returns the best candidate plus a full rejection report.  Everything it uses
  (fruit truth, target identity, reachability) is DEBUG METADATA ONLY and never
  enters a policy observation.
* :func:`build_fixed_base_world` builds the clean episode world at that stance
  (robot spawned directly at the final pose, chassis welded to the world) and
  :func:`verify_clean_start` proves the world matches the plan at ``t = 0`` --
  no hidden warm-up dynamics, no time reset.
* :class:`FixedBaseAutoPicker` is an :class:`AutoPicker` subclass that starts in
  REACH on that target from that stance.  It has NO locomotion: ``_drive`` and
  the mobile states raise, the stuck watchdog and perception tracks are not run,
  and a failed or completed pick terminates the expert instead of re-scanning.
  Every manipulation state, the IK, the arm slew, the fingers, the grasp assist
  and the metrics are the official implementation, untouched.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, asdict

import numpy as np
import warp as wp
import newton

from . import robot as _robot
from . import lsystem as _lsystem
from . import fruit as _fruit
from .picker import AutoPicker, ArmIK, _inv_xform, _rot


# --------------------------------------------------------------------------- #
# planner parameters / results
# --------------------------------------------------------------------------- #
@dataclass
class StancePlannerParams:
    """Feasibility margins of the reset-time stance planner.  Every IK bound is
    TIGHTER than the runtime watchdog it protects (REACH fails at 0.08 m, GRASP
    at 0.10 m), so a stance that passes here is not reachable-by-luck."""
    azimuth_offsets: tuple = (0.0, 0.4, -0.4)   # rad around the outward radial from the trunk axis
    distance_fracs: tuple = (0.75, 0.35)        # position inside the stand-off band [lo, hi]
    clearance_margin: float = 0.04              # m: chassis/bucket/wheels/home arm vs wood & fruit
    thick_wood_radius: float = 0.012            # wood at/above this radius cannot be brushed aside
    arm_link_radius: float = 0.085              # arm proxy capsule radius (TwigBrush._ARM_PR)
    hand_radius: float = 0.07                   # wrist->TCP proxy capsule radius
    corridor_radius: float = 0.075              # hand/forearm corridor behind the fruit
    corridor_span: tuple = (0.05, 0.55)         # m behind the fruit along the approach axis
    ik_pregrasp_max: float = 0.04               # runtime REACH watchdog: 0.08
    ik_grasp_max: float = 0.04                  # runtime GRASP watchdog: 0.10 (+-0.09 droop offset)
    ik_pull_max: float = 0.10                   # PULL never fails on IK; above this the retract stalls
    ik_pull_soft: float = 0.05
    ik_transport_max: float = 0.08              # DROP verifies the fruit inside a +-0.17 m bucket box
    pull_checks: tuple = (0.16, 0.32)           # retract distances tested (AutoPicker.RETRACT = 0.32)
    min_reach_travel: float = 0.10              # pre-grasp must lie outside REACH's 0.05 m exit radius
                                                # around the HOME TCP, so REACH really aligns the hand
                                                # before GRASP advances (v1 pilot seed 920001: pre-grasp
                                                # 3.6 cm from home -> GRASP at frame 1 from home pose)
    twig_penalty: float = 0.02                  # score per pushable-twig contact
    ik_iters: int = 32
    wood_samples: int = 9                       # points sampled along each wood capsule


@dataclass
class FixedBaseStance:
    """One feasible (target fruit, fixed chassis pose) pair.  DEBUG / EXPERT
    BOOKKEEPING ONLY -- never part of a policy observation."""
    seed: int
    apple_index: int                 # env-relative apple index (AppleField index for env 0)
    target_world: tuple              # apple centre at t = 0 (its hang point) [m]
    apple_radius: float
    base_xy: tuple
    base_yaw: float
    standoff: float
    standoff_band: tuple
    azimuth_offset: float
    ik_pregrasp_err: float
    ik_grasp_err: float
    ik_pull_err: tuple
    ik_transport_err: float
    q_pregrasp: list
    min_clearance_chassis: float
    min_clearance_arm_home: float
    twig_contacts: int
    score: float


@dataclass
class StancePlan:
    seed: int
    stance: "FixedBaseStance | None"
    report: dict
    apples_world: np.ndarray = field(repr=False)     # (N, 3) t=0 apple centres (plan truth)
    apple_radii: np.ndarray = field(repr=False)
    wood_a: np.ndarray = field(repr=False)           # (M, 3) capsule proximal ends
    wood_b: np.ndarray = field(repr=False)           # (M, 3) capsule distal ends
    wood_r: np.ndarray = field(repr=False)

    @property
    def feasible(self) -> bool:
        return self.stance is not None


# --------------------------------------------------------------------------- #
# geometry helpers (numpy, chassis frame unless stated)
# --------------------------------------------------------------------------- #
# robot spawn proxies in the CHASSIS frame, from robot.py's build constants
_CHASSIS_BOX = (np.array([0.0, 0.0, _robot._CHASSIS_Z]), np.array(_robot._CHASSIS))
_BUCKET_BOX = (np.array([_robot._BUCKET_CENTER_X, 0.0,
                         _robot._CHASSIS_Z + _robot._CHASSIS[2] + 0.5 * _robot._BUCKET_WALL_H]),
               np.array([_robot._BUCKET_HALF + 0.012, _robot._BUCKET_HALF + 0.012,
                         0.5 * _robot._BUCKET_WALL_H]))
_WHEELS = np.array([[sx * 0.30, sy * 0.36, 0.10] for sx in (-1, 1) for sy in (-1, 1)])
_WHEEL_R = 0.10
# arm proxy chain: consecutive link origins of the FR3 side model (duplicates
# such as link1/link2 collapse to zero-length capsules and are skipped)
_ARM_CHAIN = ("fr3_link1", "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link7",
              "fr3_hand", "fr3_hand_tcp")


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])


def _quat_yaw(q) -> float:
    """Yaw of an xyzw quaternion (rotation of local +x about world z)."""
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _to_chassis(pts, base_xy, yaw):
    """World -> chassis frame for an (..., 3) array (chassis origin at z = 0)."""
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(pts, dtype=float) - np.array([base_xy[0], base_xy[1], 0.0])
    out = np.empty_like(p)
    out[..., 0] = c * p[..., 0] + s * p[..., 1]
    out[..., 1] = -s * p[..., 0] + c * p[..., 1]
    out[..., 2] = p[..., 2]
    return out


def _to_world(pts, base_xy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(pts, dtype=float)
    out = np.empty_like(p)
    out[..., 0] = c * p[..., 0] - s * p[..., 1] + base_xy[0]
    out[..., 1] = s * p[..., 0] + c * p[..., 1] + base_xy[1]
    out[..., 2] = p[..., 2]
    return out


def _box_signed_dist(pts, centre, half):
    """Signed distance from points (..., 3) to an axis-aligned box: positive
    outside, negative (penetration depth) inside."""
    d = np.abs(np.asarray(pts, float) - centre) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.where(np.all(d < 0.0, axis=-1), d.max(axis=-1), 0.0)
    return outside + inside


def _capsule_samples(a, b, k):
    t = np.linspace(0.0, 1.0, k)
    return a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]        # (M, k, 3)


def _seg_seg_dist(P0, P1, Q0, Q1):
    """Closest distance between N segments P0->P1 (N, 3) and ONE segment Q0->Q1
    (3,).  Vectorised Ericson (Real-Time Collision Detection, 5.1.9)."""
    d1 = P1 - P0
    d2 = Q1 - Q0
    r = P0 - Q0
    a = (d1 * d1).sum(-1)
    e = float(d2 @ d2)
    f = (r * d2).sum(-1)
    c = (d1 * r).sum(-1)
    b = d1 @ d2
    eps = 1e-12
    if e < eps:                                 # degenerate probe: point vs segments
        s = np.clip(-c / np.maximum(a, eps), 0.0, 1.0)
        t = np.zeros_like(s)
    else:
        denom = a * e - b * b
        s = np.where(denom > eps, np.clip((b * f - c * e) / np.where(denom > eps, denom, 1.0), 0.0, 1.0), 0.0)
        t = (b * s + f) / e
        s = np.where(t < 0.0, np.clip(-c / np.maximum(a, eps), 0.0, 1.0),
                     np.where(t > 1.0, np.clip((b - c) / np.maximum(a, eps), 0.0, 1.0), s))
        t = np.clip(t, 0.0, 1.0)
    cp = P0 + d1 * s[:, None]
    cq = Q0 + d2 * t[:, None]
    return np.linalg.norm(cp - cq, axis=-1)


def _arm_chain_points(ik: ArmIK, q_arm) -> np.ndarray:
    """FK of the FR3 side model at ``q_arm`` (9) -> (K, 3) proxy chain in the
    chassis frame (same frame the IK solves in)."""
    m = ik.model
    qh = m.joint_q.numpy()
    qh[ik._arm_q] = np.asarray(q_arm, dtype=np.float32)
    m.joint_q.assign(qh)
    newton.eval_fk(m, m.joint_q, m.joint_qd, ik._state)
    bq = ik._state.body_q.numpy()
    names = [l.rsplit("/", 1)[-1] for l in m.body_label]
    return np.array([bq[names.index(n), :3] for n in _ARM_CHAIN], dtype=float)


def _ik_world(ik: ArmIK, base_xy, yaw, world_pt, q_seed, iters=32, approach_world=None):
    """Exactly :meth:`AutoPicker._ik_to` for a chassis at (base_xy, z=0, yaw)."""
    base = np.array([base_xy[0], base_xy[1], 0.0])
    bq = _yaw_quat(yaw)
    world_pt = np.asarray(world_pt, dtype=float)
    tgt = _inv_xform(base, bq, world_pt)
    if approach_world is None:
        a = world_pt - base
        approach_world = np.array([a[0], a[1], 0.15 * a[2]])
    app = _rot(np.array([-bq[0], -bq[1], -bq[2], bq[3]]), approach_world)
    if np.linalg.norm(app) < 1e-6:
        app = np.array([1.0, 0.0, 0.0])
    q, err = ik.solve(tgt, app, np.asarray(q_seed, dtype=float)[:len(ik._arm_q)], iters=iters)
    return q, float(err)


# --------------------------------------------------------------------------- #
# the planner
# --------------------------------------------------------------------------- #
class _Geometry:
    """Seed geometry at t = 0: wood capsules + apple spheres, world frame."""

    def __init__(self, cfg):
        self.skel = _lsystem.generate(cfg.lsystem, seed=cfg.seed)
        if not (cfg.fruit.enabled and cfg.deformable):
            raise ValueError("fixed-base planning needs fruit on a deformable tree")
        self.placements = _fruit.place_apples(self.skel, cfg.fruit, seed=cfg.seed)
        stem = float(cfg.fruit.stem_length)
        # builder.py: the apple body is spawned AT its hang point (attach - drop)
        self.apples = np.array([pl.attach - np.array([0.0, 0.0, stem + pl.radius])
                                for pl in self.placements], dtype=float).reshape(-1, 3)
        self.apple_r = np.array([pl.radius for pl in self.placements], dtype=float)
        self.apple_seg = np.array([pl.parent_seg for pl in self.placements], dtype=int)
        segs = self.skel.segments
        self.wood_a = np.array([s.start for s in segs], dtype=float)
        self.wood_b = np.array([s.end for s in segs], dtype=float)
        self.wood_r = np.array([max(s.mean_radius, 5e-4) for s in segs], dtype=float)
        self.wood_parent = np.array([s.parent for s in segs], dtype=int)


def plan_fixed_base_stance(cfg, ik: "ArmIK | None" = None,
                           params: "StancePlannerParams | None" = None) -> StancePlan:
    """Privileged, physics-free, deterministic (given ``cfg.seed``) selection of
    the target fruit and the fixed chassis pose for one episode.

    The tree skeleton and apple placements are regenerated with the builder's
    own generators (identical RNG streams; the robot pose does not enter them),
    so the plan describes exactly the world :func:`build_fixed_base_world`
    creates -- :func:`verify_clean_start` asserts that.
    """
    p = params or StancePlannerParams()
    ik = ik if ik is not None else ArmIK(list(_robot._ARM_HOME.values()))
    g = _Geometry(cfg)
    home = np.asarray(list(_robot._ARM_HOME.values()), dtype=float)
    reach_z = AutoPicker.REACH_Z
    thick = g.wood_r >= p.thick_wood_radius
    wood_pts = _capsule_samples(g.wood_a, g.wood_b, p.wood_samples)      # (M, k, 3) world
    n = len(g.apples)
    rej = Counter()
    per_apple = []
    candidates = []
    n_reach = 0
    best = None

    # stage order: a candidate's "reason" is the first failing check
    def clearance_robot(base_xy, yaw):
        """Min clearance of the spawn chassis/bucket/wheels to wood & fruit."""
        pts = _to_chassis(wood_pts.reshape(-1, 3), base_xy, yaw)             # (M*k, 3)
        rr = np.repeat(g.wood_r, p.wood_samples)
        d_box = _box_signed_dist(pts, *_CHASSIS_BOX) - rr
        d_buck = _box_signed_dist(pts, *_BUCKET_BOX) - rr
        d_wheel = np.min(np.linalg.norm(pts[:, None, :] - _WHEELS[None], axis=-1), axis=1) - _WHEEL_R - rr
        wood_clear = float(min(d_box.min(), d_buck.min(), d_wheel.min()))
        ap = _to_chassis(g.apples, base_xy, yaw)
        a_box = _box_signed_dist(ap, *_CHASSIS_BOX) - g.apple_r
        a_buck = _box_signed_dist(ap, *_BUCKET_BOX) - g.apple_r
        a_wheel = np.min(np.linalg.norm(ap[:, None, :] - _WHEELS[None], axis=-1), axis=1) - _WHEEL_R - g.apple_r
        fruit_clear = float(min(a_box.min(), a_buck.min(), a_wheel.min()))
        return wood_clear, fruit_clear

    def arm_contacts(chain_c, base_xy, yaw, exclude_segs=(), exclude_apple=-1):
        """Arm proxy chain (chassis frame) vs wood/fruit (world) -> (min clearance
        to THICK wood, number of pushable-twig contacts, min clearance to fruit)."""
        chain_w = _to_world(chain_c, base_xy, yaw)
        thick_clear = np.inf
        twigs = 0
        fruit_clear = np.inf
        keep = np.ones(len(g.wood_r), dtype=bool)
        for s in exclude_segs:
            if s >= 0:
                keep[s] = False
        for i in range(len(chain_w) - 1):
            q0, q1 = chain_w[i], chain_w[i + 1]
            if np.linalg.norm(q1 - q0) < 1e-6 and i + 1 < len(chain_w) - 1:
                continue
            r_probe = p.hand_radius if i == len(chain_w) - 2 else p.arm_link_radius
            d = _seg_seg_dist(g.wood_a, g.wood_b, q0, q1) - g.wood_r - r_probe
            th = thick & keep
            if th.any():
                thick_clear = min(thick_clear, float(d[th].min()))
            twigs += int(((d < 0.0) & ~thick & keep).sum())
            # fruit: apple centre (degenerate segment) to the probe capsule
            dap = _seg_seg_dist(g.apples, g.apples, q0, q1) - g.apple_r - r_probe
            if exclude_apple >= 0:
                dap[exclude_apple] = np.inf
            fruit_clear = min(fruit_clear, float(dap.min()))
        return float(thick_clear), twigs, float(fruit_clear)

    def corridor(target, away, base_xy, yaw):
        c0 = target - np.array([away[0], away[1], 0.0]) * p.corridor_span[1]
        c1 = target - np.array([away[0], away[1], 0.0]) * p.corridor_span[0]
        d = _seg_seg_dist(g.wood_a, g.wood_b, c0, c1) - g.wood_r - p.corridor_radius
        blocked = float(d[thick].min()) if thick.any() else np.inf
        return blocked, int(((d < 0.0) & ~thick).sum())

    for j in range(n):
        f = g.apples[j]
        apple_rep = dict(apple_index=j, target_world=[float(v) for v in f], reason=None,
                         candidates=0)
        if not (reach_z[0] <= f[2] <= reach_z[1]):
            rej["fruit_height"] += 1
            apple_rep["reason"] = "fruit_height"
            per_apple.append(apple_rep)
            continue
        n_reach += 1
        lo, hi = AutoPicker._standoff_band(f[2])
        rad = f[:2] / (np.linalg.norm(f[:2]) + 1e-9)
        if np.linalg.norm(f[:2]) < 0.05:
            rad = np.array([1.0, 0.0])
        spur = int(g.apple_seg[j])
        excl = (spur, int(g.wood_parent[spur]))
        apple_best_stage = -1
        apple_reason = None
        for daz in p.azimuth_offsets:
            ca, sa = math.cos(daz), math.sin(daz)
            direc = np.array([ca * rad[0] - sa * rad[1], sa * rad[0] + ca * rad[1]])   # fruit -> base
            for frac in p.distance_fracs:
                d = lo + frac * (hi - lo)
                base_xy = f[:2] + direc * d
                yaw = math.atan2(-direc[1], -direc[0])                 # face the fruit
                apple_rep["candidates"] += 1
                cand = dict(apple_index=j, base_xy=[float(base_xy[0]), float(base_xy[1])],
                            yaw=float(yaw), standoff=float(d), azimuth_offset=float(daz))
                stage = 0

                def fail(reason):
                    nonlocal apple_best_stage, apple_reason
                    rej[reason] += 1
                    cand["reason"] = reason
                    candidates.append(cand)
                    if stage > apple_best_stage:
                        apple_best_stage, apple_reason = stage, reason

                # 1. spawn clearance of the chassis body
                wood_clear, fruit_clear = clearance_robot(base_xy, yaw)
                cand.update(chassis_wood_clearance=wood_clear, chassis_fruit_clearance=fruit_clear)
                if wood_clear < p.clearance_margin:
                    fail("chassis_tree_contact"); continue
                if fruit_clear < p.clearance_margin:
                    fail("chassis_fruit_contact"); continue
                stage = 1
                # 2. spawn clearance of the arm at HOME
                chain_home = _arm_chain_points(ik, home)
                th_clear, twigs_home, fr_clear = arm_contacts(chain_home, base_xy, yaw)
                cand.update(home_arm_wood_clearance=th_clear, home_arm_fruit_clearance=fr_clear,
                            home_arm_twigs=twigs_home)
                if th_clear < p.clearance_margin:
                    fail("home_arm_wood_contact"); continue
                if fr_clear < p.clearance_margin:
                    fail("home_arm_fruit_contact"); continue
                stage = 2
                # 3. approach corridor behind the fruit (hand/forearm sweep)
                away = (f[:2] - base_xy) / (np.linalg.norm(f[:2] - base_xy) + 1e-9)
                blocked, twigs_corr = corridor(f, away, base_xy, yaw)
                cand.update(corridor_thick_clearance=blocked, corridor_twigs=twigs_corr)
                if blocked < 0.0:
                    fail("approach_corridor_blocked"); continue
                stage = 3
                # 4. IK: pre-grasp (REACH) seeded from home, exactly as _st_reach
                pre = f - np.array([away[0], away[1], 0.0]) * AutoPicker.PREGRASP
                # REACH must do real work: its exit test is |TCP - pre| < 0.05 m, so a
                # pre-grasp point next to the HOME TCP skips straight to GRASP with an
                # un-aligned hand (degenerate episode, no REACH phase)
                reach_travel = float(np.linalg.norm(pre - _to_world(chain_home[-1], base_xy, yaw)))
                cand["reach_travel_m"] = reach_travel
                if reach_travel < p.min_reach_travel:
                    fail("pregrasp_at_home_pose"); continue
                q_pre, e_pre = _ik_world(ik, base_xy, yaw, pre, home, p.ik_iters)
                cand["ik_pregrasp_err"] = e_pre
                if e_pre > p.ik_pregrasp_max:
                    fail("pregrasp_unreachable"); continue
                stage = 4
                th_pre, twigs_pre, fr_pre = arm_contacts(_arm_chain_points(ik, q_pre), base_xy, yaw,
                                                         exclude_segs=excl, exclude_apple=j)
                cand.update(pregrasp_arm_wood_clearance=th_pre, pregrasp_arm_twigs=twigs_pre,
                            pregrasp_arm_fruit_clearance=fr_pre)
                if th_pre < 0.0:
                    fail("pregrasp_arm_wood_contact"); continue
                stage = 5
                # 5. IK: grasp (GRASP servo target = live fruit) seeded from pre-grasp
                q_gr, e_gr = _ik_world(ik, base_xy, yaw, f, q_pre, p.ik_iters)
                cand["ik_grasp_err"] = e_gr
                if e_gr > p.ik_grasp_max:
                    fail("grasp_unreachable"); continue
                stage = 6
                th_gr, twigs_gr, _ = arm_contacts(_arm_chain_points(ik, q_gr), base_xy, yaw,
                                                  exclude_segs=excl, exclude_apple=j)
                cand.update(grasp_arm_wood_clearance=th_gr, grasp_arm_twigs=twigs_gr)
                if th_gr < 0.0:
                    fail("grasp_arm_wood_contact"); continue
                stage = 7
                # 6. IK: pull-back poses (PULL retracts horizontally toward the base)
                pull_err = []
                q_seed = q_gr
                for rd in p.pull_checks:
                    goal = f + np.array([-away[0], -away[1], 0.0]) * rd
                    q_seed, e_pl = _ik_world(ik, base_xy, yaw, goal, q_seed, p.ik_iters)
                    pull_err.append(e_pl)
                cand["ik_pull_err"] = pull_err
                if max(pull_err) > p.ik_pull_max:
                    fail("pull_unreachable"); continue
                stage = 8
                # 7. IK: bucket over-point (TRANSPORT), seeded from the last pull pose
                over_c = np.array([_robot._BUCKET_CENTER_X, 0.0,
                                   _robot._CHASSIS_Z + _robot._CHASSIS[2] + _robot._BUCKET_WALL_H + 0.11])
                over_w = _to_world(over_c, base_xy, yaw)
                _, e_tr = _ik_world(ik, base_xy, yaw, over_w, q_seed, p.ik_iters)
                cand["ik_transport_err"] = e_tr
                if e_tr > p.ik_transport_max:
                    fail("transport_unreachable"); continue
                stage = 9
                twigs = twigs_home + twigs_corr + twigs_pre + twigs_gr
                score = (2.0 * (e_pre + e_gr) + 0.5 * float(sum(pull_err)) + e_tr
                         + p.twig_penalty * twigs + 0.05 * abs(daz) + 0.2 * abs(frac - 0.75))
                cand.update(reason=None, twig_contacts=twigs, score=float(score))
                candidates.append(cand)
                if stage > apple_best_stage:
                    apple_best_stage, apple_reason = stage, None
                st = FixedBaseStance(
                    seed=int(cfg.seed), apple_index=j, target_world=tuple(float(v) for v in f),
                    apple_radius=float(g.apple_r[j]), base_xy=(float(base_xy[0]), float(base_xy[1])),
                    base_yaw=float(yaw), standoff=float(d), standoff_band=(float(lo), float(hi)),
                    azimuth_offset=float(daz), ik_pregrasp_err=e_pre, ik_grasp_err=e_gr,
                    ik_pull_err=tuple(pull_err), ik_transport_err=e_tr,
                    q_pregrasp=[float(v) for v in q_pre],
                    min_clearance_chassis=min(wood_clear, fruit_clear),
                    min_clearance_arm_home=min(th_clear, fr_clear), twig_contacts=twigs,
                    score=float(score))
                # deterministic: strictly better score wins; ties keep the earlier
                # (lower apple index / earlier candidate) one
                if best is None or st.score < best.score - 1e-12:
                    best = st
        apple_rep["reason"] = apple_reason
        apple_rep["furthest_stage"] = apple_best_stage
        per_apple.append(apple_rep)

    report = dict(
        planner="fixed_base_stance_planner_v2", seed=int(cfg.seed), apple_count=n,
        reachable_height_count=n_reach, candidate_count=len(candidates),
        feasible_count=sum(1 for c in candidates if c.get("reason") is None),
        rejection_reasons=dict(rej), params=asdict(p), per_apple=per_apple,
        candidates=candidates,
        selected=(asdict(best) if best is not None else None),
        privileged_information="simulator truth (fruit poses, tree geometry, IK) used at RESET ONLY; "
                               "debug metadata, NOT POLICY INPUT")
    return StancePlan(seed=int(cfg.seed), stance=best, report=report, apples_world=g.apples,
                      apple_radii=g.apple_r, wood_a=g.wood_a, wood_b=g.wood_b, wood_r=g.wood_r)


# --------------------------------------------------------------------------- #
# clean episode world at the planned stance
# --------------------------------------------------------------------------- #
def build_fixed_base_world(cfg, stance: FixedBaseStance, *, weld: bool = True,
                           fps: int = 60, substeps: int = 3):
    """Build the episode world with the robot spawned DIRECTLY at the planned
    pose and (by default) its chassis welded to the world.  Returns
    ``(tm, sim)``; ``sim.sim_time == 0`` and no physics has run."""
    from . import builder as _builder
    from .sim import Sim
    cfg.robot.position = (float(stance.base_xy[0]), float(stance.base_xy[1]))
    cfg.robot.yaw = float(stance.base_yaw)
    original_build = _robot.build_robot

    def welded_build(b, rp):
        rb = original_build(b, rp)
        if weld:
            b.add_equality_constraint_weld(
                body1=-1, body2=rb["chassis"],
                relpose=wp.transform(wp.vec3(float(rp.position[0]), float(rp.position[1]), 0.0),
                                     wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rp.yaw))),
                label="fixed_base_weld")
        return rb

    _robot.build_robot = welded_build
    try:
        tm = _builder.generate_and_build(cfg)
    finally:
        _robot.build_robot = original_build
    sim = Sim(tm, solver="mujoco", fps=fps, substeps=substeps, enable_breaking=True, collisions=True)
    return tm, sim


def verify_clean_start(sim, tm, plan: StancePlan, driver=None, *, atol_geom: float = 1e-4,
                       atol_pose: float = 1e-5) -> dict:
    """Prove the built world is the planned world at ``t = 0``: same tree and
    fruit geometry as the planning pass, robot at the planned pose, arm at
    home, no time elapsed, nothing broken or detached, zero base velocity
    targets.  Raises AssertionError with the offending quantity otherwise."""
    st = plan.stance
    assert st is not None, "no stance to verify"
    out = {}
    assert sim.sim_time == 0.0, f"sim_time={sim.sim_time}"
    assert sim.breaker is not None and sim.breaker.broken_count == 0
    assert sim.apples is not None and sim.apples.broken_count == 0
    bq = sim.body_q_np()
    # tree skeleton identical to the planning skeleton
    segs = tm.skeleton.segments
    a = np.array([s.start for s in segs]); b = np.array([s.end for s in segs])
    assert a.shape == plan.wood_a.shape, "segment count differs from plan"
    out["max_skeleton_dev_m"] = float(max(np.abs(a - plan.wood_a).max(), np.abs(b - plan.wood_b).max()))
    assert out["max_skeleton_dev_m"] < atol_geom, out
    # fruit at their planned hang points, all grown (non-DR build)
    ab = np.asarray(tm.apple_bodies, dtype=int)
    assert len(ab) == len(plan.apples_world), "apple count differs from plan"
    out["max_apple_dev_m"] = float(np.abs(bq[ab, :3] - plan.apples_world).max())
    assert out["max_apple_dev_m"] < atol_geom, out
    first_shape = {}
    for s_, b_ in enumerate(tm.model.shape_body.numpy()):
        first_shape.setdefault(int(b_), int(s_))
    ssc = tm.model.shape_scale.numpy()
    grown = ssc[first_shape[int(ab[st.apple_index])]][0]
    assert grown >= 0.015, f"planned apple shrunk (scale {grown})"
    out["target_apple_radius_built"] = float(grown)
    # robot exactly at the planned fixed pose, arm at home
    ch = int(tm.robot_data["chassis"][0])
    pose = bq[ch]
    out["base_pose_built"] = [float(v) for v in pose]
    assert np.isfinite(pose).all()
    assert np.linalg.norm(pose[:2] - np.asarray(st.base_xy)) < atol_pose, out
    assert abs(pose[2]) < atol_pose, out
    assert abs(_wrap(_quat_yaw(pose[3:]) - st.base_yaw)) < atol_pose, out
    names = [n_.rsplit("/", 1)[-1] for n_ in tm.model.joint_label]
    starts = tm.model.joint_q_start.numpy()
    qidx = np.array([starts[names.index(n_)] for n_ in _robot._ARM_HOME], dtype=int)
    home = np.asarray(tm.robot_data["arm_home"], dtype=float)
    assert np.allclose(sim.joint_q_np()[qidx], home, atol=1e-6), "arm not at home"
    if driver is not None:
        ds = int(driver.planar_dof[0])
        assert np.all(driver._target_host[[ds, ds + 1, ds + 3]] == 0.0), "nonzero base velocity target"
        assert np.all(sim.control.joint_target_qd.numpy()[[ds, ds + 1, ds + 3]] == 0.0)
    out["sim_time"] = float(sim.sim_time)
    out["branch_break_count"] = int(sim.breaker.broken_count)
    out["apple_detached_count"] = int(sim.apples.broken_count)
    return out


# --------------------------------------------------------------------------- #
# the fixed-base expert
# --------------------------------------------------------------------------- #
class FixedBaseAutoPicker(AutoPicker):
    """Single-target, fixed-base manipulation expert built on the official
    :class:`AutoPicker` backend.

    Episode structure: privileged reset-time target/stance (see
    :func:`plan_fixed_base_stance`) -> t = 0 -> REACH -> GRASP -> PULL ->
    TRANSPORT -> DROP -> DONE.  A failed attempt ends in FAILED with
    ``fail_reason`` set (the official watchdog reason, e.g. ``"reach
    stalled"``); the expert never re-scans.

    The base has NO locomotion action: ``_drive`` raises, SCAN / ALIGN / ORBIT
    / RECOVER raise if ever entered, the stuck watchdog (which only exists for
    driving states) and the perception tracks are not run.  Nothing about the
    chassis pose is faked: ``_chassis_pose`` / ``_base_yaw`` are inherited and
    read the true simulated body.
    """

    MOBILE_STATES = ("SCAN", "ALIGN", "ORBIT", "RECOVER")
    TERMINAL_STATES = ("DONE", "FAILED")

    def __init__(self, sim, tm, driver, rp, stance: FixedBaseStance, metrics=None,
                 env: int = 0, ik: "ArmIK | None" = None, pose_tol: float = 1e-3):
        # cam=None: this expert has no perception; the target is privileged
        super().__init__(sim, tm, None, driver, rp, metrics, env, ik)
        self.stance = stance
        gj = int(stance.apple_index) + self.env * self.n_env_apples
        if gj not in set(int(v) for v in self._apple_glob):
            raise ValueError(f"planned apple {gj} is not a grown apple of env {self.env}")
        live = self._bq()[int(tm.apple_bodies[gj]), :3].copy()
        dev = float(np.linalg.norm(live - np.asarray(stance.target_world)))
        if dev > pose_tol:
            raise ValueError(f"planned target deviates {dev:.4f} m from the built apple pose")
        base, bq = self._chassis_pose()
        if (np.linalg.norm(base[:2] - np.asarray(stance.base_xy)) > pose_tol
                or abs(_wrap(_quat_yaw(bq) - stance.base_yaw)) > pose_tol):
            raise ValueError("chassis is not at the planned fixed pose")
        self.planned_apple = gj          # debug bookkeeping only
        self.fail_reason = None
        self.last_ik = None              # (world target, residual) of the latest IK solve
        self._target = live
        if self.met is not None:
            lab, zv, zr = self._zone_of(self._target)
            self.met.pick_start(gj, self._target, zone=lab, zone_v=zv, zone_r=zr)
        self._goto("REACH")

    # ---- no locomotion -------------------------------------------------------
    def _drive(self, v, w):
        raise RuntimeError(f"FixedBaseAutoPicker has no locomotion action (v={v}, w={w} requested)")

    def _mobile_state(self):
        raise RuntimeError(f"mobile state {self.state} is not part of the fixed-base expert")

    _st_scan = _st_align = _st_orbit = _st_recover = _mobile_state

    def _st_done(self):
        pass

    def _st_failed(self):
        pass

    # ---- terminal hand-offs (instead of SCAN) -------------------------------
    def _after_fail(self, reason):
        self.fail_reason = str(reason)
        self.state = "FAILED"
        self._t_state = 0
        self.done = True

    def _after_drop(self):
        self.state = "DONE"
        self._t_state = 0
        self.done = True

    # ---- bookkeeping only ----------------------------------------------------
    def _ik_to(self, world_pt, approach_world=None):
        q, err = super()._ik_to(world_pt, approach_world)
        self.last_ik = (np.asarray(world_pt, dtype=float).copy(), float(err))
        return q, err

    # ---- per-frame -----------------------------------------------------------
    def update(self):
        """Call once per frame after ``sim.step()``.  No perception tracks, no
        stuck watchdog, no viewer overlay -- just the manipulation state."""
        if self.done:
            return
        self._t_state += 1
        self._frame += 1
        self.drv.follow_ground(frame=self._frame)      # terrain z-servo only (no-op on flat ground)
        if self.state in self.MOBILE_STATES:
            self._mobile_state()
        getattr(self, "_st_" + self.state.lower())()
        self._slew_arm()

    @property
    def succeeded(self) -> bool:
        return self.state == "DONE"
