"""Privileged rendered target truth. DEBUG / DATASET GATE ONLY; NOT POLICY INPUT.

Newton's shape-index output shares the RGB nearest-hit scene ray, including
foliage/robot occlusion. No ideal projected sphere or center-ray approximation.
"""
from pathlib import Path
import numpy as np

# One ray-hit pixel is exact (no probabilistic mask); retain tiny visible slivers.
# Audit reports 1..16-pixel cases separately instead of rejecting partial fruit.
MIN_VISIBLE_PIXELS = 1


class TargetVisibility:
    def __init__(self, model, target_body, static, wrist):
        self.cameras = dict(static=static, wrist=wrist)
        self.shapes = np.flatnonzero(model.shape_body.numpy() == target_body)
        if not len(self.shapes):
            raise ValueError('Target body has no renderable shapes')
        self.buffers = {name: cam.sensor.utils.create_shape_index_image_output(
            cam.color.shape[-1], cam.color.shape[-2], 1) for name, cam in self.cameras.items()}
        self.masks = {}
        self.data = dict(method='Newton SensorTiledCamera nearest-hit shape_index_image',
            debug_usage='DEBUG / DATASET GATE ONLY; NOT POLICY INPUT',
            min_visible_pixels=MIN_VISIBLE_PIXELS, target_shape_indices=self.shapes.tolist(),
            initial=None, grasp_entry=None)

    def check(self, state, moment, *, sanity=False):
        row = {}
        for name, cam in self.cameras.items():
            if sanity:
                if name == 'wrist': cam.update(state, force=True)
                else: cam.update(state)
                before = cam.last_rgb.copy()
            kw = dict(debug_shape_index_image=self.buffers[name])
            if name == 'wrist': kw['force'] = True
            cam.update(state, **kw)
            ids = self.buffers[name].numpy()[0, 0]
            mask = np.isin(ids, self.shapes)
            self.masks[name] = mask
            n = int(mask.sum())
            row[name+'_visible_pixels'] = n
            row[name+'_visible'] = n >= MIN_VISIBLE_PIXELS
            if sanity:
                assert np.array_equal(before, cam.last_rgb), 'debug output changed policy RGB'
                row[name+'_rgb_unchanged_with_shape_output'] = True
        row['any_policy_view_visible'] = row['static_visible'] or row['wrist_visible']
        self.data[moment] = row
        return row

    def save_gallery(self, directory):
        from PIL import Image
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, cam in self.cameras.items():
            rgb = cam.last_rgb.copy()
            Image.fromarray(rgb).save(directory/f'{name}_rgb.png')
            mask = self.masks[name]
            Image.fromarray(mask.astype(np.uint8)*255).save(directory/f'{name}_mask.png')
            rgb[mask] = (0, 255, 255)
            Image.fromarray(rgb).save(directory/f'{name}_overlay.png')
