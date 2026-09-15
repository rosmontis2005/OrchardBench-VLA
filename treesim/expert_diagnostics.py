"""DEBUG / DATASET GATE ONLY — NOT POLICY INPUT.

Observe rupture immediately after physics and BEFORE the next expert command:
_st_pull increments its command distance even on the frame it notices rupture.
"""
import numpy as np

PREMATURE_RETRACT_EPSILON_M = 1e-3
MEANINGFUL_PULL_M = 0.03
STRONG_MEANINGFUL_PULL_M = 0.05


class DetachDiagnostics:
    def __init__(self, sim, picker, tcp):
        self.sim, self.picker, self.tcp = sim, picker, tcp
        self.index = picker.planned_apple
        self.pull_tcp = None
        self.data = dict(debug_usage='DEBUG / DATASET GATE ONLY; NOT POLICY INPUT',
            settle_frames=picker.PULL_SETTLE_FRAMES,
            premature_epsilon_m=PREMATURE_RETRACT_EPSILON_M,
            detach_force_N=float(sim.apples.detach_force[self.index]),
            detach_force_multiplier=sim.apples.detach_force_multiplier,
            planned_apple_debug_index=self.index, grasped_apple_debug_index=None,
            pull_enter_frame=None, pull_enter_sim_time=None, detach_frame=None,
            detach_sim_time=None, pull_frames_before_detach=None,
            retract_distance_at_detach_m=None, direct_pull_force_at_detach_N=None,
            stem_tension_at_detach_N=None, detach_trigger=None,
            premature_detach=None, meaningful_pull=None, strong_meaningful_pull=None,
            tcp_displacement_since_pull_enter_m=None)

    def after_physics(self, frame):
        d = self.data
        if self.sim.apples.detached[self.index] and d['detach_frame'] is None:
            retract = float(getattr(self.picker, '_retract_dist', 0.0))
            d.update(detach_frame=frame, detach_sim_time=float(self.sim.sim_time),
                pull_frames_before_detach=(None if d['pull_enter_frame'] is None else frame-d['pull_enter_frame']),
                retract_distance_at_detach_m=retract,
                premature_detach=retract <= PREMATURE_RETRACT_EPSILON_M,
                meaningful_pull=retract >= MEANINGFUL_PULL_M,
                strong_meaningful_pull=retract >= STRONG_MEANINGFUL_PULL_M,
                tcp_displacement_since_pull_enter_m=(None if self.pull_tcp is None else
                    float(np.linalg.norm(self.sim.body_q_np()[self.tcp, :3]-self.pull_tcp))))
            d.update(self.sim.apples.detach_events[self.index] or {'detach_trigger': 'unknown'})

    def after_command(self, frame):
        if self.picker.state == 'PULL' and self.data['pull_enter_frame'] is None:
            self.data.update(pull_enter_frame=frame, pull_enter_sim_time=float(self.sim.sim_time),
                             grasped_apple_debug_index=int(self.picker._target_apple))
            self.pull_tcp = self.sim.body_q_np()[self.tcp, :3].copy()
