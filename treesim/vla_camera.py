"""RGB sensors for policies, independent of a window or ViewerGL framebuffer."""
from __future__ import annotations

import math
import numpy as np
import warp as wp
from newton.sensors import SensorTiledCamera


def look_at_transform(position, target):
    """Camera-to-world xyzw transform: local +X right, +Y up, -Z forward."""
    p = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - p
    if not np.isfinite(np.r_[p, forward]).all() or np.linalg.norm(forward) < 1e-8:
        raise ValueError("Invalid camera look-at pose")
    forward /= np.linalg.norm(forward)
    up = np.array([0., 0., 1.])
    if abs(forward @ up) > .99:
        up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = np.column_stack([right, np.cross(right, forward), -forward])
    q = wp.quat_from_matrix(wp.mat33(*rotation.ravel()))
    return wp.transform(wp.vec3(*p), q)


class StaticRGBCamera:
    """Fixed external workspace view; model sensor includes robot and canopy.

    Pixel layout follows Newton pinhole rays: row 0 is top, column 0 left.
    No segmentation or other privileged buffers are exposed to the policy.
    """
    def __init__(self, model, *, width=192, height=144, fov=60.,
                 position=(3.8, -3.8, 2.8), target=(1., 0., 1.0)):
        self.model = model
        self.position = np.asarray(position, dtype=float)
        self.target = np.asarray(target, dtype=float)
        with wp.ScopedDevice(model.device):
            self.sensor = SensorTiledCamera(model=model)
            self.sensor.utils.create_default_light(enable_shadows=False)
            self.rays = self.sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(fov))
            self.color = self.sensor.utils.create_color_image_output(width, height, 1)
            self.pose = wp.array([[look_at_transform(position, target)]], dtype=wp.transform)
        self.last_rgb = None

    def update(self, state, *, debug_shape_index_image=None):
        with wp.ScopedDevice(self.model.device):
            self.model.bvh_refit_shapes(state)
            self.sensor.update(state, self.pose, self.rays, color_image=self.color,
                               shape_index_image=debug_shape_index_image)
            rgba = self.sensor.utils.to_rgba_from_color(self.color).numpy()
        self.last_rgb = np.ascontiguousarray(rgba[0, ..., :3])
        return self.last_rgb
