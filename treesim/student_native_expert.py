"""Geometry-only teacher. plan() is called at real H-step boundaries.

No simulator mutation or private joint controller. The caller executes each
returned absolute TCP target exactly once through orchard_action.native_command.
Truth belongs exclusively to this teacher and passive diagnostics.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ExpertConfig:
    horizon: int = 5
    reach_speed: float = .20
    grasp_speed: float = .04
    pull_speed: float = .08
    transport_speed: float = .20
    rotation_speed: float = .30
    pregrasp: float = .16
    retract: float = .32
    reach_timeout: float = 12.
    grasp_timeout: float = 12.
    pull_timeout: float = 8.
    transport_timeout: float = 45.
    drop_timeout: float = 8.


def approach_rotation(direction):
    z = np.asarray(direction, dtype=float); z = z / np.linalg.norm(z)
    y = np.cross(z, [0., 0., 1.])
    if np.linalg.norm(y) < 1e-6: y = np.array([0., 1., 0.])
    y /= np.linalg.norm(y)
    return Rotation.from_matrix(np.column_stack((np.cross(y, z), y, z)))


class StudentNativeExpert:
    def __init__(self, config=None):
        self.config = config or ExpertConfig()
        if self.config.horizon not in (1, 5): raise ValueError('H must be 1 or 5')
        self.phase = 'RESET'; self.entered = 0; self.events = []
        self.failure = None; self.closed = False; self.ready_steps = 0
        self.transport_leg = 0
        self.last_target = None

    def transition(self, phase, step, reason=None):
        self.events.append(dict(phase=self.phase, start_step=self.entered,
                                end_step=step, start_time=self.entered/30,
                                end_time=step/30, reason=reason))
        self.phase = phase; self.entered = step
        if phase == 'FAILED': self.failure = reason

    def plan(self, obs, truth, chain):
        c = self.config; step = truth['step']
        assert step % c.horizon == 0
        p = np.asarray(obs['tcp_pos_world'], float)
        r = Rotation.from_quat(obs['tcp_quat_world'])
        fruit = np.asarray(truth['fruit_position']); base = np.asarray(truth['base_pose'][:3])
        br = Rotation.from_quat(truth['base_pose'][3:]); bucket = np.asarray(truth['bucket_position'])
        held = truth['held'] == truth['target_id']; detached = truth['target_detached']
        if self.phase == 'RESET':
            self.direction = fruit-base; self.direction[2] = 0.; self.direction /= np.linalg.norm(self.direction)
            a = fruit-base; a[2] *= .15
            self.grasp_rotation = approach_rotation(a)
            self.transition('REACH', step)
        age = (step-self.entered)/30
        limit = getattr(c, self.phase.lower()+'_timeout', float('inf'))
        if age >= limit: self.transition('FAILED', step, self.phase.lower()+'_timeout')
        if self.phase in ('PULL','TRANSPORT') and not held:
            self.transition('FAILED', step, 'lost_hold')
        if self.phase in ('REACH','GRASP') and detached and not held:
            self.transition('FAILED', step, 'premature_detach')
        goal = p.copy(); rotation = r; width = .08; speed = c.reach_speed
        if self.phase == 'REACH':
            goal = fruit-self.direction*c.pregrasp; rotation = self.grasp_rotation
            if np.linalg.norm(goal-p) < .025 and (rotation*r.inv()).magnitude() < .10:
                self.transition('GRASP', step)
        if self.phase == 'GRASP':
            goal = fruit.copy(); rotation = self.grasp_rotation; speed = c.grasp_speed
            if np.linalg.norm(fruit-p) < .020: self.closed = True
            width = 0. if self.closed else .08
            if held:
                goal = p.copy()
                if chain.get('held15_step') is not None:
                    self.pull_start = p.copy(); self.pull_rotation = r
                    self.transition('PULL', step)
        if self.phase == 'PULL':
            goal = self.pull_start-self.direction*c.retract
            rotation = self.pull_rotation; width = 0.; speed = c.pull_speed
            if detached:
                self.transition('TRANSPORT', step)
        if self.phase == 'TRANSPORT':
            width = 0.; speed = c.transport_speed
            # Two scene-independent base-frame waypoints keep the transfer above
            # the arm mount and turn the hand before descending over the bucket.
            points = ([.25, 0., .95], [-.26, 0., .85])
            rotation = br*Rotation.from_euler('xyz', [np.pi, 0., 0.])
            if self.transport_leg < len(points):
                goal = base+br.apply(points[self.transport_leg])
                if np.linalg.norm(goal-p) < .035 and (rotation*r.inv()).magnitude() < .12:
                    self.transport_leg += 1
            else:
                goal = p + (bucket+br.apply([0.,0.,.075])-fruit)
                local = br.inv().apply(fruit-bucket)
                ready = np.linalg.norm(local[:2]) < .06 and .025 < local[2] < .12 and truth['fruit_speed'] < .20
                self.ready_steps = self.ready_steps+c.horizon if ready else 0
                if self.ready_steps >= 15:
                    self.drop_position = p.copy(); self.drop_rotation = r
                    self.transition('DROP', step)
        if self.phase == 'DROP':
            goal = self.drop_position; rotation = self.drop_rotation; width = .08
            if chain.get('strict_success_step') is not None: self.transition('DONE', step)
        if self.phase in ('DONE','FAILED'): return []
        # Carry a small, bounded positional lead through contact. Resetting every
        # target to measured TCP otherwise removes the force-producing servo error.
        start = p.copy()
        if self.last_target is not None and self.phase == 'GRASP':
            lead = self.last_target-p
            start += lead*min(1., .010/max(np.linalg.norm(lead),1e-12))
        d = goal-start; distance = np.linalg.norm(d)
        rv = (rotation*r.inv()).as_rotvec(); angle = np.linalg.norm(rv)
        # Anchor at measured pose only at decision boundary; commands within the
        # block are immutable, even when contacts move the fruit.
        block = [dict(position=start+d*min(1., speed*(k+1)/30/max(distance,1e-12)),
                     rotation=(Rotation.from_rotvec(rv*min(1., c.rotation_speed*(k+1)/30/max(angle,1e-12)))*r).as_matrix(),
                     width=width, phase=self.phase, planning_step=step)
                for k in range(c.horizon)]

        # Conservative block envelope with margin for servo lag. This is teacher
        # target generation, never extra task replanning between env.step calls.
        for target in block:
            offset = target['position']-p
            target['position'] = p+offset*min(1., .018/max(np.linalg.norm(offset),1e-12))
            delta = (Rotation.from_matrix(target['rotation'])*r.inv()).as_rotvec()
            target['rotation'] = (Rotation.from_rotvec(delta*min(1., .040/max(np.linalg.norm(delta),1e-12)))*r).as_matrix()
        self.last_target = block[-1]['position'].copy()
        return block
