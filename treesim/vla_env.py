"""Independent, single-world VLA interface over the official Newton physics.

No AutoPicker is constructed or called. RGB/state are policy observations;
fruit truth is used only by the evaluator and optional physical grasp adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import newton
import newton.ik as ik
import newton.viewer

from . import builder, robot
from .config import TreeConfig
from .sim import Sim
from .vla_camera import StaticRGBCamera


@dataclass(frozen=True)
class VLAEnvConfig:
    sim_hz: int = 60
    action_repeat: int = 2
    max_control_steps: int = 300
    substeps: int = 3
    width: int = 192
    height: int = 144
    apple_count: int = 40
    foliage_density: float = .6
    grasp_mode: str = "contact"
    max_translation: float = .02       # per-axis metres/control action
    max_rotation: float = .05          # per-axis radians/control action
    max_joint_step: float = .045       # rad / physics frame
    ik_position_tolerance: float = .01
    ik_rotation_tolerance: float = .08
    # Chassis coordinates; controller safety bounds, not altered tree physics.
    workspace_lower: tuple = (-.65, -.85, .35)
    workspace_upper: tuple = (1.0, .85, 1.55)

    @property
    def control_hz(self):
        return self.sim_hz / self.action_repeat

    def __post_init__(self):
        if self.sim_hz != 60 or self.action_repeat < 1 or self.max_control_steps < 1:
            raise ValueError("Use 60Hz physics and positive repeat/episode length")
        if self.grasp_mode not in ("contact", "benchmark_assist"):
            raise ValueError("Unknown grasp_mode")


class PoseIK:
    """Full position AND quaternion objective on a private fixed-base Franka.

    Inputs use chassis-frame metres and xyzw quaternions. Output order is the
    nine named joints in robot._ARM_HOME (7 arm + 2 fingers). Orientation is
    checked independently in radians, not hidden inside a position residual.
    """
    def __init__(self, device):
        with wp.ScopedDevice(device):
            b = newton.ModelBuilder()
            b.add_urdf(robot.franka_urdf_path(),
                       xform=wp.transform(wp.vec3(*robot._ARM_MOUNT), wp.quat_identity()),
                       floating=False, enable_self_collisions=False)
            names = [n.rsplit('/', 1)[-1] for n in b.joint_label]
            for name, value in robot._ARM_HOME.items():
                b.joint_q[b.joint_q_start[names.index(name)]] = value
            self.model = b.finalize()
            names = [n.rsplit('/', 1)[-1] for n in self.model.joint_label]
            starts = self.model.joint_q_start.numpy()
            dofs = self.model.joint_qd_start.numpy()
            self.indices = np.array([starts[names.index(n)] for n in robot._ARM_HOME], dtype=int)
            di = np.array([dofs[names.index(n)] for n in robot._ARM_HOME], dtype=int)
            self.lower = self.model.joint_limit_lower.numpy()[di]
            self.upper = self.model.joint_limit_upper.numpy()[di]
            self.tcp = next(i for i,n in enumerate(self.model.body_label) if n.endswith('fr3_hand_tcp'))
            self.q = wp.array(self.model.joint_q.numpy()[None], dtype=wp.float32)
            self.position = ik.IKObjectivePosition(self.tcp, wp.vec3(), wp.array([wp.vec3()], dtype=wp.vec3))
            self.rotation = ik.IKObjectiveRotation(
                self.tcp, wp.quat_identity(), wp.array([wp.vec4(0.,0.,0.,1.)], dtype=wp.vec4), weight=1.)
            limits = ik.IKObjectiveJointLimit(self.model.joint_limit_lower, self.model.joint_limit_upper, weight=10.)
            self.solver = ik.IKSolver(self.model, n_problems=1,
                objectives=[self.position, self.rotation, limits], lambda_initial=.1,
                jacobian_mode=ik.IKJacobianType.ANALYTIC)
            self.state = self.model.state()

    def solve(self, target_position_base, target_quaternion_base, q_init, iterations=32):
        p = np.asarray(target_position_base, dtype=float)
        qrot = np.asarray(target_quaternion_base, dtype=float)
        q_init = np.asarray(q_init, dtype=float)
        if p.shape != (3,) or qrot.shape != (4,) or q_init.shape != (9,):
            raise ValueError("PoseIK expects position(3), quaternion(4), joint state(9)")
        if not np.isfinite(np.r_[p,qrot,q_init]).all() or np.linalg.norm(qrot) < 1e-8:
            raise ValueError("Nonfinite or invalid IK input")
        qrot /= np.linalg.norm(qrot)
        with wp.ScopedDevice(self.model.device):
            qh = self.q.numpy()
            qh[0,self.indices] = np.clip(q_init,self.lower,self.upper)
            self.q.assign(qh)
            self.position.set_target_position(0,wp.vec3(*p))
            self.rotation.set_target_rotation(0,wp.vec4(*qrot))
            self.solver.step(self.q,self.q,iterations=iterations)
            solution = self.q.numpy()[0]
            if not np.isfinite(solution).all():
                return q_init.copy(), float('inf'), float('inf')
            solution[self.indices] = np.clip(solution[self.indices], self.lower,self.upper)
            self.model.joint_q.assign(solution)
            newton.eval_fk(self.model,self.model.joint_q,self.model.joint_qd,self.state)
            pose = self.state.body_q.numpy()[self.tcp]
        pe = float(np.linalg.norm(pose[:3]-p))
        re = float((Rotation.from_quat(pose[3:]).inv()*Rotation.from_quat(qrot)).magnitude())
        return solution[self.indices].copy(), pe, re


class OrchardVLAEnv:
    """Gym-like interface, without Gym/Torch/Transformers dependencies.

    Native action [dx,dy,dz,droll,dpitch,dyaw,gripper]: measured-world relative
    translation (metres), extrinsic xyz rotation delta (radians), Rnew=Rdelta*R.
    Zero Cartesian action holds the last accepted Cartesian setpoint (avoids
    integrating servo sag). Positive grip opens, negative closes, zero holds.
    One action runs two 60Hz physics frames by default. No base motion action.

    Success/reward: first observed detached fruit; evaluator truth stays in info.
    This is an interface task, not a validated harvest-success benchmark.
    """
    def __init__(self, config=None):
        self.config = config or VLAEnvConfig()
        self.sim = self.tm = self.viewer = self.driver = None
        self.wrist_camera = self.static_camera = self.ik = None
        self._obs = None
        self._done = True

    def reset(self, *, seed=42):
        self.close()
        self.seed = int(seed)
        c = TreeConfig.compliant('apple')
        c.seed = self.seed
        c.lsystem.shape_jitter = 0.0  # same as official CLI default
        c.fruit.enabled = True
        c.fruit.max_count = self.config.apple_count
        c.foliage.set_density(self.config.foliage_density)
        c.breaking.enabled = True
        c.robot.enabled = True
        c.robot.camera_width, c.robot.camera_height = self.config.width, self.config.height
        with wp.ScopedDevice('cuda:0'):
            self.tm = builder.generate_and_build(c)
            self.sim = Sim(self.tm,solver='mujoco',fps=self.config.sim_hz,
                substeps=self.config.substeps,enable_breaking=True,collisions=True)
            self.viewer = newton.viewer.ViewerNull()
            self.sim.set_viewer(self.viewer)
            self.driver = robot.RobotDriver(self.sim,self.tm,c.robot)
            self.driver.update()
            names = [n.rsplit('/',1)[-1] for n in self.tm.model.joint_label]
            qstarts = self.tm.model.joint_q_start.numpy()
            self.joint_indices = np.array([qstarts[names.index(n)] for n in robot._ARM_HOME],dtype=int)
            rb = self.tm.robot_data
            self.dof_indices = np.asarray(rb['arm_dofs'][0],dtype=int)
            self.target_indices = np.asarray(rb['arm_tq'][0],dtype=int)
            mj_to_newton = self.sim.solver.mjc_dof_to_newton_dof.numpy()[0]
            self._bias_indices = np.array([np.flatnonzero(mj_to_newton == d)[0]
                                           for d in self.dof_indices[:7]])
            self._joint_force = self.sim.control.joint_f.numpy()
            self._effort_limits = self.tm.model.joint_effort_limit.numpy()[self.dof_indices[:7]]
            self.chassis = int(rb['chassis'][0])
            self.wrist = int(rb['wrist'][0])
            self.tcp_body = next(i for i,n in enumerate(self.tm.model.body_label) if n.endswith('fr3_hand_tcp'))
            self.finger_bodies = {i for i,n in enumerate(self.tm.model.body_label)
                                  if n.endswith(('fr3_leftfinger','fr3_rightfinger'))}
            self.ik = PoseIK(self.tm.model.device)
            self.wrist_camera = robot.WristCamera(self.tm.model,self.viewer,self.tm,c.robot,perceive=False)
            self.static_camera = StaticRGBCamera(self.tm.model,width=self.config.width,height=self.config.height)
            self._joint_command = np.asarray(rb['arm_home'],dtype=float).copy()
            self._gripper = 0.0  # home width .04m; first explicit grip chooses direction
            self._held = None
            self.control_steps = 0
            self._success = False
            self._done = False
            # Compile IK and render first sensors without integrating physics.
            # No hidden pre-episode motion, break flags, solver-state restore or time reset.
            self._refresh_obs()
            self._target_position = self._obs['tcp_pos_world'].copy()
            self._target_rotation = Rotation.from_quat(self._obs['tcp_quat_world'])
            cp, cr = self._chassis_pose()
            _,pe,re = self.ik.solve(cr.inv().apply(self._target_position-cp),
                                   (cr.inv()*self._target_rotation).as_quat(),self._joint_command)
            if not np.isfinite([pe,re]).all() or pe>self.config.ik_position_tolerance or re>self.config.ik_rotation_tolerance:
                raise RuntimeError(f'Home pose IK failed: {pe}m/{re}rad')
        assert self.sim.sim_time == 0. and self.sim.apples.broken_count == 0 and self.sim.breaker.broken_count == 0
        return self.get_obs(), self._info()

    def _chassis_pose(self):
        pose = self.sim.body_q_np()[self.chassis]
        return pose[:3].astype(float), Rotation.from_quat(pose[3:])

    def _refresh_obs(self):
        self.wrist_camera.update(self.sim.state_0,force=True)
        static = self.static_camera.update(self.sim.state_0)
        bq = self.sim.body_q_np()
        tcp, base = bq[self.tcp_body], bq[self.chassis]
        jp = self.sim.joint_q_np()[self.joint_indices].copy()
        jv = self.sim.state_0.joint_qd.numpy()[self.dof_indices].copy()
        euler = Rotation.from_quat(tcp[3:]).as_euler('xyz')
        width = float(jp[-2:].sum())
        yaw = float(Rotation.from_quat(base[3:]).as_euler('xyz')[2])
        # Native proprio has explicit physical components, no CALVIN padding.
        proprio = np.r_[tcp[:3],tcp[3:],width,jp,jv,base[:3],yaw].astype(np.float32)
        if not np.isfinite(proprio).all():
            self._done = True
            raise RuntimeError('Nonfinite robot state')
        self._obs = dict(rgb_static=static.copy(),rgb_wrist=self.wrist_camera.last_rgb.copy(),
            proprio=proprio,tcp_pos_world=tcp[:3].copy(),tcp_quat_world=tcp[3:].copy(),
            tcp_euler_world=euler,gripper_width=width,joint_pos=jp,joint_vel=jv,
            base_pos=base[:3].copy(),base_yaw=yaw,sim_time=float(self.sim.sim_time))

    def get_obs(self):
        """Snapshot from reset/last completed step; arrays are owned copies."""
        if self._obs is None:
            raise RuntimeError('Call reset before get_obs')
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self._obs.items()}

    def get_calvin_state(self):
        """Compatibility layer: world xyz/Euler xyz/width + 25 zeros."""
        obs = self.get_obs()
        return np.r_[obs['tcp_pos_world'],obs['tcp_euler_world'],obs['gripper_width'],
                     np.zeros(25)].astype(np.float32)

    def _update_assist(self):
        """Two-finger contact gate; no intended/detected/nearest target selection.

        Uses actual last collision contacts AND strict palm volume. If multiple
        apples qualify simultaneously, decline to attach rather than pick one.
        This is the benchmark's spring grasp abstraction, not natural friction.
        """
        if self.config.grasp_mode == 'contact':
            return False
        if self._gripper >= 0:
            if self._held is not None:
                self.sim.apples.release(self._held)
                self._held = None
            return False
        if self._held is not None:
            return False
        contacts = self.sim.contacts
        n = int(contacts.rigid_contact_count.numpy()[0])
        shape_body = self.tm.model.shape_body.numpy()
        s0 = contacts.rigid_contact_shape0.numpy()[:n]
        s1 = contacts.rigid_contact_shape1.numpy()[:n]
        apple_to_index = {int(b):i for i,b in enumerate(self.tm.apple_bodies)}
        touched = {}
        for a,b in zip(s0,s1):
            if a<0 or b<0: continue
            ba,bb = int(shape_body[a]),int(shape_body[b])
            for finger,fruit in ((ba,bb),(bb,ba)):
                if finger in self.finger_bodies and fruit in apple_to_index:
                    touched.setdefault(fruit,set()).add(finger)
        poses = self.sim.body_q_np()
        hand = poses[self.wrist]
        inv = Rotation.from_quat(hand[3:]).inv()
        eligible=[]
        for body,fingers in touched.items():
            local = inv.apply(poses[body,:3]-hand[:3])
            if fingers == self.finger_bodies and len(fingers)==2 and abs(local[0])<.025 and abs(local[1])<.04 and .065<local[2]<.125:
                eligible.append(apple_to_index[body])
        if len(eligible)==1:
            self._held=eligible[0]
            self.sim.apples.hold(self._held,self.wrist)
            return True
        return False

    def _info(self):
        apples = self.sim.apples
        cp,cr = self._chassis_pose()
        positions = self.sim.body_q_np()[np.asarray(self.tm.apple_bodies),:3]
        local = cr.inv().apply(positions-cp)
        floor = robot._CHASSIS_Z+robot._CHASSIS[2]
        bucket = ((np.abs(local[:,0]-robot._BUCKET_CENTER_X)<robot._BUCKET_HALF)
                  & (np.abs(local[:,1])<robot._BUCKET_HALF)
                  & (local[:,2]>floor) & (local[:,2]<floor+robot._BUCKET_WALL_H)
                  & np.asarray(apples.detached,dtype=bool))
        return dict(seed=self.seed,control_steps=self.control_steps,
            apple_detached_count=int(apples.broken_count),branch_break_count=int(self.sim.breaker.broken_count),
            apple_in_bucket_count=int(bucket.sum()),success=bool(apples.broken_count>0),
            success_definition='at least one physically detached apple',grasp_mode=self.config.grasp_mode,
            grasp_assist_triggered=False,grasped_apple_id_debug=self._held,
            sim_hz=self.config.sim_hz,control_hz=self.config.control_hz,action_repeat=self.config.action_repeat)

    def step(self, action):
        if self._done:
            raise RuntimeError('Episode finished or not initialized; call reset')
        a = np.asarray(action,dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError('Action must be finite shape (7,); rejected before physics/control mutation')
        c = self.config
        limits = np.array([c.max_translation]*3+[c.max_rotation]*3+[1.])
        clipped = np.clip(a,-limits,limits)
        action_clipped = not np.array_equal(a,clipped)
        obs = self._obs
        if np.any(clipped[:6]):
            target = obs['tcp_pos_world']+clipped[:3]
            rotation = Rotation.from_euler('xyz',clipped[3:6])*Rotation.from_quat(obs['tcp_quat_world'])
        else:
            target = self._target_position.copy()
            rotation = self._target_rotation
        cp,cr = self._chassis_pose()
        target_base = cr.inv().apply(target-cp)
        bounded = np.clip(target_base,c.workspace_lower,c.workspace_upper)
        action_clipped |= not np.allclose(target_base,bounded,rtol=0,atol=1e-9)
        target = cp+cr.apply(bounded)
        q,pe,re = self.ik.solve(bounded,(cr.inv()*rotation).as_quat(),obs['joint_pos'])
        failed = (not np.isfinite(q).all() or not np.isfinite([pe,re]).all()
                  or pe>c.ik_position_tolerance or re>c.ik_rotation_tolerance)
        if failed:
            q = self._joint_command.copy()
        else:
            self._target_position = target.copy()
            self._target_rotation = rotation
        if clipped[6] != 0:
            self._gripper = float(np.sign(clipped[6]))
        q[-2:] = .04 if self._gripper>0 else (0. if self._gripper<0 else self._joint_command[-2:])
        triggered = False
        for _ in range(c.action_repeat):
            self._joint_command += np.clip(q-self._joint_command,-c.max_joint_step,c.max_joint_step)
            self._joint_command = np.clip(self._joint_command,self.ik.lower,self.ik.upper)
            self.driver._tq_host[self.target_indices] = self._joint_command
            self.sim.control.joint_target_q.assign(self.driver._tq_host)
            self.driver.update()
            # Position servos alone have gravity sag (~6.6mm at home), which
            # accumulates under measured-relative deltas. Feed forward the
            # existing solver's generalized bias force on the seven arm DOFs.
            # This is a controller command; masses/gains/contacts/physics are
            # unchanged. Chassis, fingers, tree and apples get no compensation.
            bias = self.sim.solver.mjw_data.qfrc_bias.numpy()[0,self._bias_indices]
            if not np.isfinite(bias).all():
                self._done = True
                raise RuntimeError('Nonfinite arm bias force')
            self._joint_force[self.dof_indices[:7]] = np.clip(bias,-self._effort_limits,self._effort_limits)
            self.sim.control.joint_f.assign(self._joint_force)
            triggered |= self._update_assist()
            self.sim.step()
        self.control_steps += 1
        self._refresh_obs()
        info = self._info()
        info.update(ik_error=dict(position_m=pe,rotation_rad=re),ik_failed=bool(failed),
            action_clipped=bool(action_clipped),physical_action=clipped.tolist(),
            target_pose=dict(position_world=target.tolist(),quaternion_world=rotation.as_quat().tolist()),
            executed_joint_target=self._joint_command.tolist(),
            arm_bias_feedforward=self._joint_force[self.dof_indices[:7]].tolist(),
            grasp_assist_triggered=bool(triggered))
        terminated = info['success']
        truncated = self.control_steps>=c.max_control_steps
        reward = float(terminated and not self._success)
        self._success |= terminated
        self._done = terminated or truncated
        return self.get_obs(),reward,bool(terminated),bool(truncated),info

    def close(self):
        if self.sim is not None:
            wp.synchronize_device(self.sim.model.device)
        if self.viewer is not None:
            self.viewer.close()
        self._obs = None
        self.wrist_camera = self.static_camera = self.ik = None
        self.driver = self.viewer = self.sim = self.tm = None
        self._done = True
        gc.collect()
