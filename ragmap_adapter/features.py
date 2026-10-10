"""Upstream's ``FeaturesExtractor`` with two adapter-side changes, nothing else.

1. **CLIP backbone.** Upstream loads OpenAI ``clip.load('ViT-L/14@336px')`` and
   hard-codes 768-d features. OpenLex3D evaluated every method except OpenScene
   with OpenCLIP ViT-H-14 (laion2b_s32b_b79k), and its published OpenMask3D
   embeddings are 1024-d, so the authors swapped the backbone; this does the
   same: ``open_clip.create_model_and_transforms`` (its eval transform), fp32,
   and the width is taken from the model instead of the literal 768.
2. **Projector.** ``ragmap_adapter.projector.SparsePointProjector`` (exact) by
   default, upstream's dense ``PointProjector`` when ``sparse=False``.

``extract_features`` is upstream's loop line for line apart from those two
points; it also records each mask's top-k views and crop count for run.json.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from openmask3d.mask_features_computation.features_extractor import FeaturesExtractor, PointProjector
from openmask3d.mask_features_computation.utils import initialize_sam_model, mask2box_multi_level, run_sam

from ragmap_adapter.projector import SparsePointProjector


class RagmapFeaturesExtractor(FeaturesExtractor):
    def __init__(self, camera, clip_model, clip_pretrained, images, masks, pointcloud,
                 sam_model_type, sam_checkpoint, vis_threshold, device, sparse=True):
        import open_clip

        self.camera = camera
        self.images = images
        self.device = device
        projector = SparsePointProjector if sparse else PointProjector
        self.point_projector = projector(camera, pointcloud, masks, vis_threshold, images.indices)
        self.predictor_sam = initialize_sam_model(device, sam_model_type, sam_checkpoint)
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained, device=device)
        model.eval()
        self.clip_model, self.clip_preprocess = model, preprocess
        self.feature_dim = int(model.visual.output_dim)
        self.top_views = None
        self.crops_per_mask = None

    def _mask_pixels(self, view, mask):
        projector = self.point_projector
        if isinstance(projector, SparsePointProjector):
            return projector.mask_pixels(view, mask)
        return np.transpose(np.where(projector.visible_points_in_view_in_mask[view][mask] == True))  # noqa: E712

    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds,
                         num_selected_points, save_crops, out_folder, optimize_gpu_usage=False):
        if save_crops:
            out_folder = os.path.join(out_folder, "crops")
            os.makedirs(out_folder, exist_ok=True)

        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)

        num_masks = self.point_projector.masks.num_masks
        mask_clip = np.zeros((num_masks, self.feature_dim))  # upstream: np.zeros((num_masks, 768))
        crops_per_mask = np.zeros(num_masks, dtype=int)

        np_images = self.images.get_as_np_list()
        for mask in tqdm(range(num_masks)):
            images_crops = []
            if optimize_gpu_usage:
                self.clip_model.to(torch.device('cpu'))
                self.predictor_sam.model.cuda()
            for view_count, view in enumerate(topk_indices_per_mask[mask]):
                if optimize_gpu_usage:
                    torch.cuda.empty_cache()
                point_coords = self._mask_pixels(view, mask)
                if point_coords.shape[0] > 0:
                    self.predictor_sam.set_image(np_images[view])
                    best_mask = run_sam(image_size=np_images[view],
                                        num_random_rounds=num_random_rounds,
                                        num_selected_points=num_selected_points,
                                        point_coords=point_coords,
                                        predictor_sam=self.predictor_sam,)
                    for level in range(num_levels):
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level, multi_level_expansion_ratio)
                        cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                        if save_crops:
                            cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))
                        cropped_img_processed = self.clip_preprocess(cropped_img)
                        images_crops.append(cropped_img_processed)

            if optimize_gpu_usage:
                self.predictor_sam.model.cpu()
                self.clip_model.to(torch.device('cuda'))
            if len(images_crops) > 0:
                image_input = torch.tensor(np.stack(images_crops))
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input.to(self.device)).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                mask_clip[mask] = image_features.mean(axis=0).cpu().numpy()
            crops_per_mask[mask] = len(images_crops)

        self.top_views = topk_indices_per_mask
        self.crops_per_mask = crops_per_mask
        return mask_clip
