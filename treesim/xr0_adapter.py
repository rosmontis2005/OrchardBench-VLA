"""CALVIN/XR-0 compatibility without Torch/Transformers in the simulator.

Contract audited against local XR-0 eval_calvin/main.py, processing_mibot.py,
and CALVIN robot/robot.py + conf/robot/panda.yaml. No model-specific state
normalization is applied to native proprioception.
"""
from __future__ import annotations
import numpy as np
from PIL import Image


def calvin_state(obs):
    state = np.r_[obs['tcp_pos_world'],obs['tcp_euler_world'],obs['gripper_width'],np.zeros(25)].astype(np.float32)
    if state.shape != (32,) or not np.isfinite(state).all():
        raise ValueError('Invalid CALVIN state')
    return state


def xr0_images(obs, crop_ratio=.95):
    images=[]
    for key in ('rgb_static','rgb_wrist'):
        a=obs[key]
        if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[-1]!=3:
            raise ValueError(f'{key} must be uint8 HWC RGB')
        im=Image.fromarray(a)
        # Match official evaluator center crop then resize to original size.
        w,h=im.size;cw,ch=int(w*crop_ratio),int(h*crop_ratio)
        if not (0<cw<=w and 0<ch<=h):raise ValueError('Invalid crop ratio')
        left,top=(w-cw)//2,(h-ch)//2
        images.append(im.crop((left,top,left+cw,top+ch)).resize((w,h),Image.Resampling.BILINEAR))
    return images


def xr0_prompt(language):
    return ('<|im_start|>user\nThe following observations are captured from multiple views.\n'
            '# Base View\n<|vision_start|><|image_pad|><|vision_end|>\n'
            '# Left-Wrist View\n<|vision_start|><|image_pad|><|vision_end|>\n'
            f'Generate robot actions for the task:\n{language} /no_cot<|im_end|>\n'
            '<|im_start|>assistant\n<cot></cot><|im_end|>\n')


class CalvinActionAdapter:
    """Decoded CALVIN normalized actions -> native Orchard world SE(3) deltas.

    Dataset relative action = metres*50 / Euler-radians*20. Processor decoding
    only undoes learned mean/std, so scaling is still required afterwards.
    Local CALVIN use_target_pose=true: accumulate desired xyz AND Euler xyz;
    convert the resulting absolute orientation to Rdelta=Rtarget*Rcurrent^-1.
    This preserves CALVIN additive Euler semantics even away from identity.
    """
    POSITION_SCALE=50.
    ROTATION_SCALE=20.

    def reset(self,obs):
        self.target_position=np.array(obs['tcp_pos_world'],dtype=float,copy=True)
        self.target_euler=np.array(obs['tcp_euler_world'],dtype=float,copy=True)

    @classmethod
    def physical_delta(cls,action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('Expected finite decoded CALVIN action(7)')
        return np.r_[np.clip(a[:3],-1,1)/cls.POSITION_SCALE,
                     np.clip(a[3:6],-1,1)/cls.ROTATION_SCALE,1. if a[6]>0 else -1.]

    def to_native(self,action,obs):
        from scipy.spatial.transform import Rotation
        if not hasattr(self,'target_position'):self.reset(obs)
        d=self.physical_delta(action)
        self.target_position+=d[:3]
        self.target_euler+=d[3:6]
        target=Rotation.from_euler('xyz',self.target_euler)
        current=Rotation.from_quat(obs['tcp_quat_world'])
        delta=(target*current.inv()).as_euler('xyz')
        return np.r_[self.target_position-obs['tcp_pos_world'],delta,d[6]]
