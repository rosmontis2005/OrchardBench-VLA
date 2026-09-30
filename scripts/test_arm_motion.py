"""CPU numerical regression tests; no simulator execution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
import numpy as np
from treesim.arm_motion import ArmMotionProfile, JointMotionLimiter


class ArmMotionTests(unittest.TestCase):
    def test_legacy_exact_random_goals_and_custom_rate(self):
        rng=np.random.default_rng(421)
        for rate in [.045,.02]:
            old=rng.normal(size=7);lim=JointMotionLimiter(old)
            for _ in range(1000):
                goal=rng.normal(size=7)
                d=np.clip(goal-old,-rate,rate);old=old+d
                np.testing.assert_array_equal(lim.step(goal,legacy_step=rate),old)

    def test_velocity_bound_and_landing(self):
        lim=JointMotionLimiter(np.zeros(7),ArmMotionProfile('velocity',1.5))
        for goal in [np.ones(7),-np.ones(7),np.zeros(7)]:
            for _ in range(100):
                old=lim.q_cmd.copy();new=lim.step(goal)
                self.assertLessEqual(np.abs(new-old).max(),1.5/60+1e-14)
                self.assertTrue(np.all((goal-old)*(goal-new)>=-1e-14))
            np.testing.assert_array_equal(lim.q_cmd,goal)

    def test_acceleration_ramp_stop_and_reversal(self):
        for a in [10.,20.]:
            lim=JointMotionLimiter(np.zeros(7),ArmMotionProfile('velocity_accel',1.5,a))
            for goal,steps in [(np.ones(7),100),(-np.ones(7),130),(np.zeros(7),100)]:
                for _ in range(steps):
                    old=lim.q_cmd.copy();vel=lim.qd_cmd.copy();new=lim.step(goal)
                    self.assertLessEqual(np.abs(lim.qd_cmd).max(),1.5+1e-12)
                    self.assertTrue(np.all((goal-old)*(goal-new)>=-1e-12))
                    normal=~lim.last_landed
                    self.assertTrue(np.all(np.abs(lim.qd_cmd[normal]-vel[normal])<=a/60+1e-12))
                np.testing.assert_allclose(lim.q_cmd,goal,atol=1e-10)

    def test_target_inside_stopping_distance_is_explicit(self):
        lim=JointMotionLimiter(np.zeros(7),ArmMotionProfile('velocity_accel',1.5,10.))
        for _ in range(15):lim.step(np.ones(7))
        goal=lim.q_cmd+1e-5;lim.step(goal)
        np.testing.assert_array_equal(lim.q_cmd,goal)
        np.testing.assert_array_equal(lim.qd_cmd,np.zeros(7))
        self.assertEqual(lim.crossing_count,7)
        lim.reset(np.ones(7));self.assertEqual(lim.crossing_count,0)

    def test_phase_velocity_override_keeps_state(self):
        p=ArmMotionProfile('velocity_accel', .5, 20., ('REACH','TRANSPORT'), {'REACH':1.5})
        lim=JointMotionLimiter(np.zeros(7),p)
        for _ in range(20):lim.step(np.ones(7)*3, vmax_rad_s=p.phase_vmax_rad_s['REACH'])
        self.assertAlmostEqual(lim.qd_cmd[0],1.5)
        previous=lim.q_cmd.copy()
        lim.step(np.ones(7)*3, vmax_rad_s=.5)
        # Entry decelerates at the existing acceleration bound, without resetting.
        np.testing.assert_allclose(lim.qd_cmd,1.5-20/60)
        np.testing.assert_allclose(lim.q_cmd-previous,lim.qd_cmd/60)
        for _ in range(4):lim.step(np.ones(7)*3,vmax_rad_s=.5)
        np.testing.assert_allclose(lim.qd_cmd,.5)
        with self.assertRaises(ValueError):
            ArmMotionProfile(phase_vmax_rad_s={'REACH':-1})

    def test_validation_and_copy(self):
        for kwargs in [dict(mode='x'),dict(vmax_rad_s=0),dict(vmax_rad_s=np.inf),dict(mode='velocity_accel'),dict(amax_rad_s2=-1)]:
            with self.assertRaises(ValueError):ArmMotionProfile(**kwargs)
        for q in [np.zeros(9),np.full(7,np.nan)]:
            with self.assertRaises(ValueError):JointMotionLimiter(q)
        q=np.zeros(7);lim=JointMotionLimiter(q);q[:]=5;self.assertEqual(lim.q_cmd.max(),0)
        result=lim.step(np.ones(7));result[:]=100;self.assertLess(lim.q_cmd.max(),1)

    def test_transport_scope_preserves_legacy_until_detach(self):
        from treesim.picker import AutoPicker
        class Sink:
            def assign(self, values):self.values=values.copy()
        def picker(profile):
            p=AutoPicker.__new__(AutoPicker);p._q_cmd=np.zeros(9);p._q_goal=np.zeros(9)
            p.arm_motion=None if profile is None else JointMotionLimiter(np.zeros(7),profile)
            p.arm_tq=np.arange(9);p.drv=SimpleNamespace(_tq_host=np.zeros(9))
            p.sim=SimpleNamespace(control=SimpleNamespace(joint_target_q=Sink()))
            return p
        old=picker(None)
        scoped=picker(ArmMotionProfile('velocity',2.0,active_phases=('TRANSPORT',)))
        rng=np.random.default_rng(321)
        for phase in ['REACH','GRASP','PULL']:
            old.state=scoped.state=phase
            for _ in range(80):
                goal=rng.normal(size=9)
                for p in [old,scoped]:
                    p._set_arm(goal.copy());p._fingers(.016);p._slew_arm()
                np.testing.assert_array_equal(old._q_cmd,scoped._q_cmd)
                np.testing.assert_array_equal(scoped.arm_motion.q_cmd,scoped._q_cmd[:7])
        scoped.state='TRANSPORT'
        for _ in range(80):
            previous=scoped._q_cmd.copy();scoped._set_arm(np.ones(9));scoped._slew_arm()
            self.assertLessEqual(np.max(np.abs(scoped._q_cmd[:7]-previous[:7])),2/60+1e-12)
        scoped.state='DROP';previous=scoped._q_cmd.copy();scoped._set_arm(-np.ones(9))
        scoped._slew_arm()
        np.testing.assert_array_equal(scoped._q_cmd,previous+np.clip(-np.ones(9)-previous,-.045,.045))
        for phases in [(),('BOGUS',),['TRANSPORT']]:
            with self.assertRaises(ValueError):ArmMotionProfile(active_phases=phases)

    def test_picker_finger_and_legacy_integration_exact(self):
        # Use actual methods with a minimal driver, no physics or IK instantiation.
        from treesim.picker import AutoPicker
        rng=np.random.default_rng(98)
        class Sink:
            def assign(self,a):self.value=a.copy()
        def picker(profile):
            p=AutoPicker.__new__(AutoPicker);p._q_cmd=np.zeros(9);p._q_goal=np.zeros(9)
            p.arm_motion=None if profile is None else JointMotionLimiter(np.zeros(7),profile)
            p.arm_tq=np.arange(9);p.drv=SimpleNamespace(_tq_host=np.zeros(9))
            p.sim=SimpleNamespace(control=SimpleNamespace(joint_target_q=Sink()))
            return p
        old=picker(None);new=picker(ArmMotionProfile());slow=picker(ArmMotionProfile('velocity_accel',1.5,10))
        for i in range(300):
            goal=rng.normal(size=9)
            for p in [old,new,slow]:
                p._set_arm(goal.copy())
                if i%3==0:p._fingers(.04 if i%2 else .016)
                p._slew_arm()
            np.testing.assert_array_equal(old._q_cmd,new._q_cmd)
            np.testing.assert_array_equal(old._q_cmd[-2:],slow._q_cmd[-2:])
            v=slow.arm_motion.qd_cmd.copy();slow._goto('TRANSPORT')
            np.testing.assert_array_equal(v,slow.arm_motion.qd_cmd)

if __name__=='__main__':unittest.main(verbosity=2)
