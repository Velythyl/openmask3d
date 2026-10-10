"""ragmap-run: OpenMask3D behind RAGMAP's object-mapping container contract.

    ragmap-run --input /input --output /output [key=value ...]

INPUT (RAGMAP ``ragmap/objects/base.py`` contract, plus one OpenMask3D file)::

    meta.json, frames.jsonl, rgb/, depth/     the shared contract (OpenCV
                                              camera-to-world poses, metres)
    scene_cloud.ply                           the scene's reconstructed geometry
                                              (x, y, z, rgb) in the same frame
    scene_cloud.json                          {"source", "dataset_from_build", ...}

OpenMask3D's class-agnostic mask module (Mask3D) segments a *given* scene cloud;
RAGMAP's ``openmask3d`` module hands it the dataset's own mesh/cloud, as
OpenLex3D ran it ("access to clean mesh inputs", paper Sec. 4.2).

What runs is upstream's ``run_openmask3d_single_scene.sh``, step for step:

1. ``class_agnostic_mask_computation/get_masks_single_scene.py`` as a
   subprocess, with the script's own overrides (``model.num_queries=120``,
   ``general.use_dbscan=true``, ``general.dbscan_eps=0.95``, ScanNet200 Mask3D
   checkpoint). Upstream code, unmodified. It writes ``<scene>_masks.pt``:
   (n_points, n_masks) float16 0/1, columns sorted by Mask3D score, highest first.
2. ``compute_features_single_scene.py``'s body, in process, with upstream's
   ``openmask3d_inference.yaml`` values (top_k 5, 3 levels, expansion 0.1,
   vis_threshold 0.2, 10 SAM rounds of 5 points, frequency 10) through
   ``ragmap_adapter.features.RagmapFeaturesExtractor``: OpenCLIP ViT-H-14 (the
   OpenLex3D backbone) instead of OpenAI ViT-L/14@336, and an exact sparse
   projector instead of the dense (views, masks, H, W) array. Upstream draws
   SAM's point prompts with an unseeded ``np.random.shuffle``; ``ragmap.seed``
   seeds numpy first (recorded in run.json).

OUTPUT::

    openmask3d/<scene>_masks.pt                upstream's mask file, unchanged
    openmask3d/<scene>_openmask3d_features.npy upstream's (n_masks, D) features
    openmask3d/top_views.npy, crops_per_mask.npy
    objects.jsonl, objects/mask_<j>.ply        one object per non-empty mask
    run.json

Every exception is recorded in run.json (``status: failed``) before exiting 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("OPENMASK3D_ROOT", "/opt/openmask3d"))
WEIGHTS = Path(os.environ.get("OPENMASK3D_WEIGHTS", "/weights"))
SCHEMA = "ragmap.openmask3d/v1"

#: run_openmask3d_single_scene.sh, step 1 (minus paths and visualisation).
MASK_OVERRIDES = {
    "general.train_mode": "false",
    "data.test_mode": "test",
    "model.num_queries": "120",
    "general.use_dbscan": "true",
    "general.dbscan_eps": "0.95",
    "general.save_visualizations": "false",
}

RAGMAP_DEFAULTS = {
    # run_openmask3d_single_scene.sh's MASK_MODULE_CKPT_PATH (README: the
    # checkpoint "for running the model on an arbitrary scene").
    "checkpoint": "scannet200_model.ckpt",
    "sam_checkpoint": "sam_vit_h_4b8939.pth",
    "sam_model_type": "vit_h",
    "clip_model": "ViT-H-14",
    "clip_weights": "open_clip_vit_h14_laion2b_s32b_b79k.bin",
    "seed": "0",
    "sparse_projector": "true",
    # Frame Mask3D sees: "build" (the input's z-up frame) or "dataset" (the
    # cloud's source frame, scene_cloud.json dataset_from_build). The masks
    # are per point, so features and outputs are unaffected by this choice.
    "mask3d_frame": "build",
    "object_clouds": "true",
}


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_overrides(items):
    openmask3d, mask, ragmap = {}, {}, dict(RAGMAP_DEFAULTS)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"override {item!r} is not key=value")
        key, value = item.split("=", 1)
        group, _, name = key.partition(".")
        if group == "openmask3d" and name:
            openmask3d[name] = value
        elif group == "mask" and name:
            mask[name] = value
        elif group == "ragmap" and name in RAGMAP_DEFAULTS:
            ragmap[name] = value
        else:
            raise SystemExit(f"unknown override {key!r} (openmask3d.*, mask.*, ragmap.{sorted(RAGMAP_DEFAULTS)})")
    return openmask3d, mask, ragmap


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY: double xyz + uchar rgb (open3d reads both)."""
    vertex = np.empty(len(points), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertex.tobytes())


def read_cloud(path: Path):
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    colors = (np.asarray(cloud.colors) * 255).round().astype(np.uint8) if cloud.has_colors() \
        else np.full((len(points), 3), 128, np.uint8)
    return points, colors


def prepare_scene(input_dir: Path, scene_dir: Path, meta: dict, frames: list[dict]) -> dict:
    """Upstream's single-scene layout: pose/<i>.txt, color/<i><ext>, depth/<i>.png,
    intrinsic/intrinsic_color.txt, with i the frame's position in frames.jsonl."""
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    for sub in ("pose", "color", "depth", "intrinsic"):
        (scene_dir / sub).mkdir(parents=True)
    extensions = {Path(frame["rgb"]).suffix.lower() for frame in frames}
    if len(extensions) != 1:
        raise RuntimeError(f"frames mix image extensions {sorted(extensions)}")
    ext = extensions.pop()
    for i, frame in enumerate(frames):
        pose = np.asarray(frame["pose"], dtype=np.float64).reshape(4, 4)
        np.savetxt(scene_dir / "pose" / f"{i}.txt", pose, fmt="%.17g")
        os.symlink(input_dir / frame["rgb"], scene_dir / "color" / f"{i}{ext}")
        os.symlink(input_dir / frame["depth"], scene_dir / "depth" / f"{i}.png")
    intrinsic = np.eye(4)
    intrinsic[0, 0], intrinsic[1, 1] = meta["fx"], meta["fy"]
    intrinsic[0, 2], intrinsic[1, 2] = meta["cx"], meta["cy"]
    np.savetxt(scene_dir / "intrinsic" / "intrinsic_color.txt", intrinsic, fmt="%.17g")
    return {"image_ext": ext, "resolution": [int(meta["height"]), int(meta["width"])]}


def run_masks(scene_ply: Path, out_dir: Path, checkpoint: Path, extra: dict, log) -> Path:
    overrides = dict(MASK_OVERRIDES)
    overrides.update(extra)
    command = [
        sys.executable, "class_agnostic_mask_computation/get_masks_single_scene.py",
        "general.experiment_name=ragmap",
        f"general.checkpoint={checkpoint}",
        *[f"{key}={value}" for key, value in overrides.items()],
        f"general.scene_path={scene_ply}",
        f"general.mask_save_dir={out_dir}",
        f"hydra.run.dir={out_dir}/hydra_outputs/class_agnostic_mask_computation",
    ]
    log["mask_command"] = command
    print("[ragmap] " + " ".join(command), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="3")  # as run_openmask3d_single_scene.sh
    subprocess.run(command, cwd=ROOT / "openmask3d", env=env, check=True)
    masks = out_dir / f"{scene_ply.stem}_masks.pt"
    if not masks.is_file():
        raise RuntimeError(f"Mask3D wrote no {masks}")
    return masks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ragmap-run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default="/input", type=Path)
    parser.add_argument("--output", default="/output", type=Path)
    parser.add_argument("overrides", nargs="*", help="openmask3d.<k>=v | mask.<hydra key>=v | ragmap.<k>=v")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    run: dict = {"schema": SCHEMA, "status": "running", "openmask3d_git_sha": os.environ.get("OPENMASK3D_GIT_SHA"),
                 "timings": {}}
    started = time.time()
    try:
        _main(args, run)
        run["status"] = "ok"
        return_code = 0
    except BaseException as error:  # noqa: BLE001 - recorded, then re-signalled by the exit code
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        run["traceback"] = traceback.format_exc()
        print(run["traceback"], file=sys.stderr, flush=True)
        return_code = 1
    run["timings"]["total_s"] = round(time.time() - started, 2)
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return return_code


def _main(args, run: dict) -> None:
    import torch
    from omegaconf import OmegaConf

    input_dir: Path = args.input.resolve()
    output: Path = args.output.resolve()
    openmask3d_cfg, mask_extra, ragmap = parse_overrides(args.overrides)
    inference = OmegaConf.load(ROOT / "openmask3d" / "configs" / "openmask3d_inference.yaml")
    for key, value in openmask3d_cfg.items():
        if key not in inference.openmask3d:
            raise SystemExit(f"openmask3d.{key} is not in openmask3d_inference.yaml")
        inference.openmask3d[key] = type(inference.openmask3d[key])(value)
    params = OmegaConf.to_container(inference.openmask3d)
    seed = int(ragmap["seed"])
    run["config"] = {"openmask3d": params, "mask_overrides": {**MASK_OVERRIDES, **mask_extra}, "ragmap": ragmap}

    meta = json.loads((input_dir / "meta.json").read_text())
    frames = [json.loads(line) for line in (input_dir / "frames.jsonl").read_text().splitlines() if line.strip()]
    cloud_path = input_dir / "scene_cloud.ply"
    if not cloud_path.is_file():
        raise RuntimeError(f"{cloud_path} is missing: OpenMask3D segments a given scene cloud")
    cloud_meta = json.loads((input_dir / "scene_cloud.json").read_text()) if (input_dir / "scene_cloud.json").is_file() else {}
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: Mask3D and SAM need a GPU")
    device = torch.device("cuda", 0)
    checkpoint = WEIGHTS / ragmap["checkpoint"]
    for path in (checkpoint, WEIGHTS / ragmap["sam_checkpoint"], WEIGHTS / ragmap["clip_weights"]):
        if not path.is_file():
            raise RuntimeError(f"weight file {path} is missing (mount the weights directory at {WEIGHTS})")
    run["weights"] = {"checkpoint": ragmap["checkpoint"], "checkpoint_sha256": _sha256(checkpoint)}
    run["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0)}
    try:
        import MinkowskiEngine
        import open_clip
        run["versions"].update({"MinkowskiEngine": MinkowskiEngine.__version__, "open_clip": open_clip.__version__})
    except Exception:  # noqa: BLE001 - informational only
        pass

    work = Path("/tmp/openmask3d-scene")
    native = output / "openmask3d"
    native.mkdir(parents=True, exist_ok=True)
    scene_info = prepare_scene(input_dir, work, meta, frames)
    scene_name = "scene"
    scene_ply = work / f"{scene_name}.ply"
    points, colors = read_cloud(cloud_path)
    write_ply(scene_ply, points, colors)  # features: the cloud in the frames' frame
    mask_ply = scene_ply
    if ragmap["mask3d_frame"] == "dataset":
        transform = np.asarray(cloud_meta.get("dataset_from_build"), dtype=np.float64)
        if transform.shape != (4, 4):
            raise RuntimeError("ragmap.mask3d_frame=dataset needs scene_cloud.json dataset_from_build (4x4)")
        mask_ply = work / "mask3d" / f"{scene_name}.ply"
        mask_ply.parent.mkdir()
        write_ply(mask_ply, points @ transform[:3, :3].T + transform[:3, 3], colors)
    elif ragmap["mask3d_frame"] != "build":
        raise RuntimeError(f"ragmap.mask3d_frame must be build or dataset, not {ragmap['mask3d_frame']!r}")
    run["scene"] = {"points": int(len(points)), "frames": len(frames), **scene_info, "cloud": cloud_meta}

    # 1. Class-agnostic masks (upstream script).
    tick = time.time()
    masks_file = run_masks(mask_ply, native, checkpoint, mask_extra, run)
    run["timings"]["masks_s"] = round(time.time() - tick, 2)

    # 2. Mask features (compute_features_single_scene.py's body).
    from openmask3d.data.load import Camera, Images, InstanceMasks3D, PointCloud, get_number_of_images
    from ragmap_adapter.features import RagmapFeaturesExtractor

    tick = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    masks = InstanceMasks3D(str(masks_file))
    if masks.masks.shape[0] != len(points):
        raise RuntimeError(f"{masks_file} has {masks.masks.shape[0]} rows for {len(points)} points")
    print(f"[INFO] Masks loaded. {masks.num_masks} masks found.", flush=True)
    indices = np.arange(0, get_number_of_images(str(work / "pose")), step=params["frequency"])
    images = Images(images_path=str(work / "color"), extension=scene_info["image_ext"], indices=indices)
    pointcloud = PointCloud(str(scene_ply))
    camera = Camera(intrinsic_path=str(work / "intrinsic" / "intrinsic_color.txt"),
                    intrinsic_resolution=scene_info["resolution"],
                    poses_path=str(work / "pose"), depths_path=str(work / "depth"),
                    extension_depth=".png", depth_scale=float(meta["depth_scale"]))
    extractor = RagmapFeaturesExtractor(
        camera=camera, clip_model=ragmap["clip_model"], clip_pretrained=str(WEIGHTS / ragmap["clip_weights"]),
        images=images, masks=masks, pointcloud=pointcloud, sam_model_type=ragmap["sam_model_type"],
        sam_checkpoint=str(WEIGHTS / ragmap["sam_checkpoint"]), vis_threshold=params["vis_threshold"],
        device=device, sparse=_bool(ragmap["sparse_projector"]))
    run["timings"]["projection_s"] = round(time.time() - tick, 2)
    features = extractor.extract_features(
        topk=params["top_k"], multi_level_expansion_ratio=params["multi_level_expansion_ratio"],
        num_levels=params["num_of_levels"], num_random_rounds=params["num_random_rounds"],
        num_selected_points=params["num_selected_points"], save_crops=False, out_folder=str(native))
    np.save(native / f"{scene_name}_openmask3d_features.npy", features)
    np.save(native / "top_views.npy", extractor.top_views)
    np.save(native / "crops_per_mask.npy", extractor.crops_per_mask)
    run["timings"]["features_s"] = round(time.time() - tick, 2)

    # 3. Objects (contract OUTPUT): one per non-empty mask, in score order.
    mask_matrix = np.asarray(masks.masks) > 0
    counts = getattr(extractor.point_projector, "counts", None)
    objects_dir = output / "objects"
    objects_dir.mkdir(exist_ok=True)
    lines, empty = [], 0
    for j in range(mask_matrix.shape[1]):
        members = np.flatnonzero(mask_matrix[:, j])
        if not len(members):
            empty += 1
            continue
        xyz = points[members]
        views = [int(v) for v in extractor.top_views[j] if counts is None or counts[int(v), j] > 0]
        record = {
            "id": f"mask_{j:04d}", "label": "object", "caption": None, "crop": None,
            "centroid": xyz.mean(0).tolist(), "bbox_min": xyz.min(0).tolist(), "bbox_max": xyz.max(0).tolist(),
            "frame_indices": [int(frames[int(indices[v])]["frame_index"]) for v in views],
            "floor": None, "room": None,
            "extra": {"mask_index": j, "score_rank": j, "points": int(len(members)),
                      "crops": int(extractor.crops_per_mask[j]),
                      "feature_norm": float(np.linalg.norm(features[j]))},
        }
        if _bool(ragmap["object_clouds"]):
            write_ply(objects_dir / f"mask_{j:04d}.ply", xyz, colors[members])
            record["pointcloud"] = f"objects/mask_{j:04d}.ply"
        lines.append(json.dumps(record))
    (output / "objects.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    covered = mask_matrix.any(axis=1)
    run["counts"] = {
        "masks": int(mask_matrix.shape[1]), "empty_masks": empty, "objects": len(lines),
        "views": int(len(indices)), "masks_without_crops": int((extractor.crops_per_mask == 0).sum()),
        "points_in_any_mask": int(covered.sum()), "points_in_several_masks": int((mask_matrix.sum(1) > 1).sum()),
    }
    run["files"] = {"masks": str(masks_file.relative_to(output)),
                    "features": f"openmask3d/{scene_name}_openmask3d_features.npy",
                    "scene_cloud": "input:scene_cloud.ply"}
    run["seed"] = seed


if __name__ == "__main__":
    raise SystemExit(main())
