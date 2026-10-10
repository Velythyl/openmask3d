"""An exact, memory-light replacement for upstream's ``PointProjector``.

Upstream (``openmask3d/mask_features_computation/features_extractor.py``)
materialises ``visible_points_in_view_in_mask``: a dense boolean array of shape
``(views, masks, H, W)``. For a Replica scene at OpenLex3D's stride of 10 that is
200 x ~150 x 680 x 1200 = 25 GB; for an HM3D walk (359 views x ~480 masks x
720 x 1080) it is ~134 GB. Upstream only ever reads two things from it:

* per (view, mask), the number of True pixels -> ``get_top_k_indices_per_mask``;
* for a mask's top-k views, ``np.transpose(np.where(array[view][mask]))``, the
  True pixels in row-major order -> SAM's point prompts.

This class computes exactly those, from the same per-view projection and
visibility code, and keeps only each (view, mask) pair's sorted unique pixel
indices. Nothing else changes: the counts matrix has upstream's dtype and goes
through upstream's own ``np.argsort``, so ties break identically, and the pixel
lists come out in the same order, so ``run_sam``'s shuffles see the same input.
``ragmap_adapter.selfcheck`` compares both against upstream's dense class.
"""
from __future__ import annotations

import os

import imageio
import numpy as np
from tqdm import tqdm


class SparsePointProjector:
    def __init__(self, camera, point_cloud, masks, vis_threshold, indices):
        self.vis_threshold = vis_threshold
        self.indices = indices
        self.camera = camera
        self.point_cloud = point_cloud
        self.masks = masks
        self._compute()

    # -- upstream's get_visible_points_view, one view at a time ----------------
    def _project_view(self, i, idx, poses, X, intrinsic, width, height, n_points):
        """Upstream's STEP 1 and STEP 2 for view ``i``, verbatim, minus the
        (views, n_points, 2) buffer: returns that view's slice and visibility."""
        depth_scale = self.camera.depth_scale
        depths_path = self.camera.depths_path
        projected_points = np.zeros((n_points, 2), dtype=int)
        projected_points_not_norm = (intrinsic @ poses[i] @ X.T).T
        mask = (projected_points_not_norm[:, 2] != 0)
        projected_points[mask] = np.column_stack([[projected_points_not_norm[:, 0][mask] / projected_points_not_norm[:, 2][mask],
                                                   projected_points_not_norm[:, 1][mask] / projected_points_not_norm[:, 2][mask]]]).T
        depth_path = os.path.join(depths_path, str(idx) + '.png')
        sensor_depth = imageio.imread(depth_path) / depth_scale
        inside_mask = (projected_points[:, 0] >= 0) * (projected_points[:, 1] >= 0) \
            * (projected_points[:, 0] < width) \
            * (projected_points[:, 1] < height)
        pi = projected_points.T
        point_depth = projected_points_not_norm[:, 2]
        visibility_mask = (np.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
                                  - point_depth[inside_mask]) <= self.vis_threshold).astype(bool)
        inside_mask[inside_mask == True] = visibility_mask  # noqa: E712 (upstream's idiom)
        return projected_points, inside_mask

    def _compute(self):
        indices = self.indices
        poses = self.camera.load_poses(indices)
        X = self.point_cloud.get_homogeneous_coordinates()
        n_points = self.point_cloud.num_points
        resolution = imageio.imread(os.path.join(self.camera.depths_path, '0.png')).shape
        height, width = resolution[0], resolution[1]
        intrinsic = self.camera.get_adapted_intrinsic(resolution)
        masks = self.masks.masks
        masks = masks.numpy() if hasattr(masks, "numpy") else np.asarray(masks)
        num_masks = masks.shape[1]
        # Point indices of each mask (upstream: `masks.masks[:, j] * visible > 0`).
        members = [np.flatnonzero(masks[:, j] > 0) for j in range(num_masks)]
        counts = np.zeros((len(indices), num_masks), dtype=int)
        pixels = {}
        print("[INFO] Computing the visible points in each view (sparse, exact).")
        for i, idx in tqdm(list(enumerate(indices))):
            projected, visible = self._project_view(i, idx, poses, X, intrinsic, width, height, n_points)
            for j in range(num_masks):
                chosen = members[j][visible[members[j]]]
                if len(chosen) == 0:
                    continue
                proj_points = projected[chosen]
                linear = np.unique(proj_points[:, 1].astype(np.int64) * width + proj_points[:, 0])
                counts[i, j] = len(linear)
                pixels[(i, j)] = linear.astype(np.int64)
        self.counts = counts
        self._pixels = pixels
        self.resolution = resolution
        self._width = width

    def get_top_k_indices_per_mask(self, k):
        # Upstream: num = dense.sum(axis=2).sum(axis=2); argsort(-num, axis=0)[:k, :].T
        return np.argsort(-self.counts, axis=0)[:k, :].T

    def mask_pixels(self, view, mask):
        """``np.transpose(np.where(dense[view][mask] == True))``: (row, col) pairs."""
        linear = self._pixels.get((int(view), int(mask)))
        if linear is None:
            return np.zeros((0, 2), dtype=np.int64)
        return np.stack([linear // self._width, linear % self._width], axis=1)
