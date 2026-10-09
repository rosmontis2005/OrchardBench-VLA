"""CPU tests for the temporal/coordinate boundary of the new data contract."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import unittest
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from treesim.orchard_command import CONTRACT, encode_window, window_starts, policy_sample
from treesim.orchard_action import decode_targets
from treesim.student_native_expert import StudentNativeExpert


class CommandContractTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(8)
        self.traj=dict(contract=CONTRACT,num_commands=36,execution_horizon=5,
            observations=[dict(tcp_pos_world=rng.normal(size=3),tcp_quat_world=Rotation.random(random_state=rng).as_quat(),gripper_width=.038,joint_pos=np.zeros(9)) for _ in range(37)],
            commands=[dict(position=rng.normal(size=3),rotation=Rotation.random(random_state=rng).as_matrix(),width=0. if i<18 else .08) for i in range(36)])

    def test_real_windows_and_noncommuting_rotations(self):
        self.assertEqual(list(window_starts(self.traj)),[0,5])
        for start in window_starts(self.traj):
            a=encode_window(self.traj,start);o=self.traj['observations'][start]
            p,r,w=decode_targets(a,o['tcp_pos_world'],Rotation.from_quat(o['tcp_quat_world']).as_matrix())
            cs=self.traj['commands'][start:start+30]
            np.testing.assert_allclose(p,[c['position'] for c in cs],atol=1e-6)
            np.testing.assert_allclose(r,[c['rotation'] for c in cs],atol=1e-6)
            np.testing.assert_allclose(w,[c['width'] for c in cs],atol=1e-8)
            self.assertTrue(np.all(a[:,7:]==0))
        for invalid in [-1,1,10,35]:
            with self.assertRaises(IndexError):encode_window(self.traj,invalid)

    def test_truth_exclusion_and_absolute_width(self):
        im=Image.new('RGB',(192,144));a=policy_sample(self.traj,0,im,im)
        modified=copy.deepcopy(self.traj);modified['diagnostics']={'fruit_position':[999]*3,'phase':'secret'}
        modified['observations'][0]['fruit_truth']=[999]*3
        b=policy_sample(modified,0,im,im)
        self.assertEqual(set(a),{'rgb_static','rgb_wrist','instruction','state','action','action_mask'})
        for k in ['state','action','action_mask']:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(float(a['action'][0,6]),0.)
        self.assertAlmostEqual(float(a['action'][29,6]),.08)
        self.assertAlmostEqual(float(a['state'][0,6]),.038)
        modified['contract']='orchard_cartesian_local_rotvec_width_v1'
        with self.assertRaises(ValueError):encode_window(modified,0)

    def test_five_fixed_commands_and_no_state_mutation(self):
        expert=StudentNativeExpert();obs=self.traj['observations'][0]
        obs['tcp_pos_world']=np.array([.4,0.,.7]);obs['tcp_quat_world']=[0.,0.,0.,1.]
        truth=dict(step=0,fruit_position=np.array([.7,0.,.9]),base_pose=[0.,0.,0.,0.,0.,0.,1.],bucket_position=[-.26,0.,.46],held=None,target_id=1,target_detached=False,fruit_speed=0.)
        before=copy.deepcopy(truth);block=expert.plan(obs,truth,{})
        self.assertEqual(len(block),5)
        for k in truth:np.testing.assert_array_equal(truth[k],before[k])
        for target in block:
            self.assertEqual(target['planning_step'],0)
            self.assertLessEqual(np.linalg.norm(target['position']-obs['tcp_pos_world']),.01800001)
        truth['step']=1
        with self.assertRaises(AssertionError):expert.plan(obs,truth,{})

if __name__=='__main__':unittest.main()
