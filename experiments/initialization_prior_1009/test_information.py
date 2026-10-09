"""CPU checks for information permissions, late transport and FSM equivalence."""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[2])]
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from controller import ControlledExpert,TargetInformationProvider,StageFeedback,MODES
from treesim.student_native_expert import StudentNativeExpert

CAL=dict(nominal_base_xyz=[.73,0,1.],height_to_x=[-.2,.93],held_offset_tcp=[0.,0.,0.])
OBS=dict(tcp_pos_world=np.array([.50,0.,1.]),tcp_quat_world=np.array([0.,0.,0.,1.]))
BASE=np.array([0.,0.,0.,0.,0.,0.,1.]);BUCKET=np.array([-.3,0,.5])


def provider(mode):
    kwargs={'initial_height':1.} if mode=='Cz' else {'initial_xyz':[.73,0,1.]} if mode in ('Cxyz0','Cxyz') else {}
    return TargetInformationProvider(mode,CAL,**kwargs)


class Permissions(unittest.TestCase):
    def test_only_authorized_measurements(self):
        with self.assertRaises(AssertionError):TargetInformationProvider('C0',CAL,initial_xyz=[1,2,3])
        with self.assertRaises(AssertionError):TargetInformationProvider('Cz',CAL,initial_height=1.,initial_xyz=[1,2,3])
        for mode in MODES[:-1]:
            p=provider(mode)
            with self.assertRaises(AssertionError):
                p.estimate(step=0,tcp_base=[0,0,0],tcp_rotation_base=Rotation.identity(),held=False,current_xyz=[99,99,99])
            for held in (False,True):
                got=p.estimate(step=5,tcp_base=[.1,.2,.3],tcp_rotation_base=Rotation.identity(),held=held)
                np.testing.assert_allclose(got,[.1,.2,.3] if held else p.initial)
            self.assertEqual(p.updates,0)

    def test_late_transport_has_no_live_channel(self):
        for mode in MODES[:-1]:
            expert=ControlledExpert(provider(mode));expert.fsm.phase='TRANSPORT';expert.fsm.transport_leg=2
            obs=dict(OBS,tcp_pos_world=np.array([-.3,0,.575]))
            feedback=StageFeedback(True,True,15,None,.01)
            for step in [20,25,30]:
                block,estimate=expert.plan(obs,step,BASE,BUCKET,feedback)
                np.testing.assert_allclose(estimate,obs['tcp_pos_world'])
                self.assertEqual(len(block),5)
            self.assertEqual(expert.fsm.phase,'DROP')
            self.assertEqual(expert.provider.updates,0)

    def test_original_equivalence_all_stages(self):
        for phase in ['RESET','REACH','GRASP','PULL','TRANSPORT','DROP']:
            old=StudentNativeExpert();new=ControlledExpert(provider('Cxyz'))
            for fsm in (old,new.fsm):
                fsm.phase=phase;fsm.direction=np.array([1.,0.,0.]);fsm.grasp_rotation=Rotation.identity()
                fsm.pull_start=OBS['tcp_pos_world'].copy();fsm.pull_rotation=Rotation.identity()
                fsm.drop_position=OBS['tcp_pos_world'].copy();fsm.drop_rotation=Rotation.identity();fsm.transport_leg=2
            held=phase in ['PULL','TRANSPORT'];fruit=np.array([.73,0.,1.])
            feedback=StageFeedback(held,held,15 if held else None,None,.01)
            truth=dict(step=20,fruit_position=fruit,base_pose=BASE,bucket_position=BUCKET,
                       held=42 if held else None,target_id=42,target_detached=held,fruit_speed=.01)
            lhs=old.plan(OBS,truth,dict(held15_step=feedback.held15_step,strict_success_step=None))
            rhs,_=new.plan(OBS,20,BASE,BUCKET,feedback,current_xyz=fruit)
            self.assertEqual(old.phase,new.fsm.phase)
            for a,b in zip(lhs,rhs):
                for key in ['position','rotation','width']:np.testing.assert_allclose(a[key],b[key],atol=1e-14)

    def test_c0_coordinate_equivariance(self):
        a=ControlledExpert(provider('C0'));b=ControlledExpert(provider('C0'))
        r=Rotation.from_euler('z',1.2);t=np.array([4.,-2.,0.]);bp=np.r_[t,r.as_quat()]
        obs=dict(tcp_pos_world=t+r.apply(OBS['tcp_pos_world']),tcp_quat_world=r.as_quat())
        feedback=StageFeedback(False,False,None,None,0.)
        aa,_=a.plan(OBS,0,BASE,BUCKET,feedback);bb,_=b.plan(obs,0,bp,t+r.apply(BUCKET),feedback)
        for x,y in zip(aa,bb):np.testing.assert_allclose(y['position'],t+r.apply(x['position']))

if __name__=='__main__':unittest.main()
