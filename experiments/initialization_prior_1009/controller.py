"""Spatial-information ablation; production FSM is reused without modification.

The FSM receives only estimated fruit position and a separate, explicit stage
feedback whitelist. No env, seed, planner metadata, or truth callback is passed
into either the FSM or low-information providers.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from treesim.student_native_expert import StudentNativeExpert, ExpertConfig

MODES = ('C0', 'Cz', 'Cxyz0', 'Cxyz')


@dataclass(frozen=True)
class StageFeedback:
    held: bool
    detached: bool
    held15_step: int | None
    success_step: int | None
    fruit_speed: float  # shared privileged scalar, NOT a velocity vector


class TargetInformationProvider:
    def __init__(self, mode, calibration, *, initial_height=None, initial_xyz=None):
        assert mode in MODES
        self.mode = mode
        if mode == 'C0':
            assert initial_height is None and initial_xyz is None
            self.initial = np.array(calibration['nominal_base_xyz'], float)
        elif mode == 'Cz':
            assert initial_height is not None and initial_xyz is None
            slope, intercept = calibration['height_to_x']
            self.initial = np.array([slope*initial_height+intercept, 0., initial_height])
        else:
            assert initial_xyz is not None and initial_height is None
            self.initial = np.array(initial_xyz, float)
        self.fixed_tcp_offset = np.array(calibration['held_offset_tcp'], float)
        self.updates = 0

    def estimate(self, *, step, tcp_base, tcp_rotation_base, held, current_xyz=None):
        assert step % 5 == 0
        if self.mode == 'Cxyz':
            assert current_xyz is not None
            self.updates += 1
            return np.array(current_xyz, float).copy()
        assert current_xyz is None, 'Unauthorized live target position'
        # Global rigid-grasp assumption; never calibrated using this rollout.
        if held:
            return np.array(tcp_base)+tcp_rotation_base.apply(self.fixed_tcp_offset)
        return self.initial.copy()


class ControlledExpert:
    def __init__(self, provider, config=None):
        self.provider = provider
        self.fsm = StudentNativeExpert(config or ExpertConfig())

    def plan(self, obs, step, base_pose, bucket_position, feedback, *, current_xyz=None):
        base = np.asarray(base_pose[:3]); br = Rotation.from_quat(base_pose[3:])
        estimate = self.provider.estimate(step=step,
            tcp_base=br.inv().apply(np.asarray(obs['tcp_pos_world'])-base),
            tcp_rotation_base=br.inv()*Rotation.from_quat(obs['tcp_quat_world']),
            held=feedback.held, current_xyz=current_xyz)
        # Constant synthetic identity: IDs/seeds/debug never enter the FSM.
        restricted = dict(step=step, fruit_position=base+br.apply(estimate),
            base_pose=np.array(base_pose), bucket_position=np.array(bucket_position),
            held=0 if feedback.held else None, target_id=0,
            target_detached=feedback.detached, fruit_speed=feedback.fruit_speed)
        chain = dict(held15_step=feedback.held15_step,
                     strict_success_step=feedback.success_step)
        return self.fsm.plan(obs, restricted, chain), estimate
