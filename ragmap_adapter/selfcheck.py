"""Identity check: SparsePointProjector against upstream's dense PointProjector.

    python -m ragmap_adapter.selfcheck --input /input --masks <scene>_masks.pt [--frequency 40]

Builds upstream's single-scene layout from the contract input, runs both
projectors over the same views, and compares what ``extract_features`` reads:
the (views, masks) pixel-count matrix, ``get_top_k_indices_per_mask(5)``, and
the row-major pixel list of every (view, mask) pair. Exits 1 on any difference.
Upstream's dense array is (views, masks, H, W) bools, so pick a frequency whose
views fit in memory (Replica at 40: 50 x ~150 x 816 000 = ~6 GB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openmask3d.data.load import Camera, InstanceMasks3D, PointCloud, get_number_of_images
from openmask3d.mask_features_computation.features_extractor import PointProjector

from ragmap_adapter.projector import SparsePointProjector
from ragmap_adapter.run import prepare_scene, read_cloud, write_ply


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--frequency", type=int, default=40)
    parser.add_argument("--vis-threshold", type=float, default=0.2)
    args = parser.parse_args()
    meta = json.loads((args.input / "meta.json").read_text())
    frames = [json.loads(l) for l in (args.input / "frames.jsonl").read_text().splitlines() if l.strip()]
    work = Path("/tmp/selfcheck-scene")
    info = prepare_scene(args.input, work, meta, frames)
    points, colors = read_cloud(args.input / "scene_cloud.ply")
    write_ply(work / "scene.ply", points, colors)
    indices = np.arange(0, get_number_of_images(str(work / "pose")), step=args.frequency)
    camera = Camera(str(work / "intrinsic" / "intrinsic_color.txt"), info["resolution"], str(work / "pose"),
                    str(work / "depth"), ".png", float(meta["depth_scale"]))
    masks = InstanceMasks3D(str(args.masks))
    cloud = PointCloud(str(work / "scene.ply"))
    sparse = SparsePointProjector(camera, cloud, masks, args.vis_threshold, indices)
    dense = PointProjector(camera, cloud, masks, args.vis_threshold, indices)
    counts = dense.visible_points_in_view_in_mask.sum(axis=2).sum(axis=2)
    report = {"views": int(len(indices)), "masks": int(masks.num_masks),
              "counts_equal": bool(np.array_equal(counts, sparse.counts)) and counts.dtype == sparse.counts.dtype,
              "topk_equal": bool(np.array_equal(dense.get_top_k_indices_per_mask(5), sparse.get_top_k_indices_per_mask(5)))}
    pairs = differing = 0
    for view in range(len(indices)):
        for mask in range(masks.num_masks):
            reference = np.transpose(np.where(dense.visible_points_in_view_in_mask[view][mask] == True))  # noqa: E712
            ours = sparse.mask_pixels(view, mask)
            pairs += 1
            if reference.shape != ours.shape or reference.dtype != ours.dtype or not np.array_equal(reference, ours):
                differing += 1
    report.update({"pairs": pairs, "pairs_differing": differing, "nonempty_pairs": int((counts > 0).sum()),
                   "visible_pixels": int(counts.sum())})
    report["identical"] = report["counts_equal"] and report["topk_equal"] and differing == 0
    print(json.dumps(report, indent=2))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
