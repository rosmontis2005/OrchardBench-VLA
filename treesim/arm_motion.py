"""Opt-in command time parameterization for seven arm joints.

No physics, gripper, IK, expert-state or simulator dependencies. These are command
limits, not guarantees on measured actuator response. Default picker remains legacy.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ArmMotionProfile:
    mode: str = 'legacy'
    vmax_rad_s: float = 2.7
    amax_rad_s2: float | None = None
    active_phases: tuple[str, ...] | None = None
    phase_vmax_rad_s: dict[str, float] | None = None

    def __post_init__(self):
        if self.active_phases is not None and (not isinstance(self.active_phases, tuple) or not self.active_phases or
                any(p not in ('REACH', 'GRASP', 'PULL', 'TRANSPORT', 'DROP') for p in self.active_phases)):
            raise ValueError('active_phases must be a nonempty tuple of manipulation phases')
        if self.phase_vmax_rad_s is not None:
            for phase, vmax in self.phase_vmax_rad_s.items():
                if phase not in ('REACH', 'GRASP', 'PULL', 'TRANSPORT', 'DROP') or not np.isfinite(vmax) or vmax <= 0:
                    raise ValueError('phase vmax requires manipulation phases and positive finite rates')
                if self.active_phases is not None and phase not in self.active_phases:
                    raise ValueError('phase vmax override must be in active_phases')
        if self.mode not in ('legacy', 'velocity', 'velocity_accel'):
            raise ValueError('unknown arm motion mode')
        if not np.isfinite(self.vmax_rad_s) or self.vmax_rad_s <= 0:
            raise ValueError('vmax_rad_s must be positive and finite')
        if self.amax_rad_s2 is not None and (not np.isfinite(self.amax_rad_s2) or self.amax_rad_s2 <= 0):
            raise ValueError('amax_rad_s2 must be positive and finite')
        if self.mode == 'velocity_accel' and self.amax_rad_s2 is None:
            raise ValueError('velocity_accel requires amax_rad_s2')


class JointMotionLimiter:
    """Seven-joint state; reset only at controller initialization/explicit reset.

    Velocity-acceleration mode uses a conservative discrete stopping bound:
        v_stop = sqrt((a*dt/2)^2 + 2*a*abs(error)) - a*dt/2
    then acceleration-limits velocity toward the new goal. There is no phase input.

    An arbitrary replacement goal can appear inside the existing stopping distance;
    bounded acceleration and never crossing that goal are then incompatible. The
    explicit crossing guard prioritizes no overshoot, lands on the goal and zeros
    that velocity. `last_landed`/`crossing_count` expose these exceptions. No hidden
    velocity reset occurs when an IK goal is replaced or an expert phase changes.
    """
    def __init__(self, q_initial, profile=None, dt=1/60):
        self.profile = profile or ArmMotionProfile()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        self.dt = float(dt)
        self.reset(q_initial)

    @staticmethod
    def _vector(q):
        a = np.asarray(q, dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError('expected seven finite arm joint positions')
        return a

    def reset(self, q_initial):
        self.q_cmd = self._vector(q_initial).copy()
        self.qd_cmd = np.zeros(7)
        self.last_landed = np.zeros(7, dtype=bool)
        self.crossing_count = 0

    def step(self, q_goal, *, legacy_step=0.045, active=True, vmax_rad_s=None):
        vmax = self.profile.vmax_rad_s if vmax_rad_s is None else vmax_rad_s
        if not np.isfinite(vmax) or vmax <= 0:
            raise ValueError('vmax_rad_s must be positive and finite')
        goal = self._vector(q_goal)
        error = goal - self.q_cmd
        self.last_landed[:] = False
        if not active or self.profile.mode in ('legacy', 'velocity'):
            # Inactive scope follows legacy arithmetic while keeping command state
            # synchronized. Entering the retimed scope never resets position/velocity.
            rate = legacy_step if not active or self.profile.mode == 'legacy' else vmax * self.dt
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError('legacy_step must be positive and finite')
            # Preserve original operations exactly, including the addition (not +=).
            d = np.clip(error, -rate, rate)
            self.q_cmd = self.q_cmd + d
            self.qd_cmd = d / self.dt  # diagnostic only; no acceleration state used
        else:
            a = self.profile.amax_rad_s2
            v_stop = np.sqrt((a*self.dt/2)**2 + 2*a*np.abs(error)) - a*self.dt/2
            desired = np.sign(error) * np.minimum(vmax, v_stop)
            self.qd_cmd = self.qd_cmd + np.clip(desired-self.qd_cmd, -a*self.dt, a*self.dt)
            displacement = self.qd_cmd * self.dt
            # A zero error with residual velocity must also hold the target.
            crossing = (error == 0) & (displacement != 0)
            crossing |= (error * displacement > 0) & (np.abs(displacement) >= np.abs(error))
            self.q_cmd = self.q_cmd + displacement
            self.q_cmd[crossing] = goal[crossing]
            self.qd_cmd[crossing] = 0.
            self.last_landed = crossing
            self.crossing_count += int(np.sum(crossing))
        return self.q_cmd.copy()
