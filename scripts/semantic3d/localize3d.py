"""Stage-2 localization: text query + retrieved keyframe -> 3D point cloud.

Combines my semantic retrieval hit (image_id), SAM3 text-prompted segmentation
on the RGB keyframe, my verified point-cloud transformation, and multi-view
fusion. We use IR-left depth as the RGB camera's depth and do NOT apply any
RGB<->IR transform. Pose comes from the DFR2 (rectified-left-IR cam_from_world,
kept current by the dense mapper).

Per frame:
  RGB image -> SAM3(query) -> best_mask (RGB res)
    -> scale_mask_to_depth -> IR res
    -> DFR2 -> project_mask_to_3d_cloud -> world (N,3)
Skipped (no 3D) when: no RGB image, no SAM3 mask, or no DFR2 depth frame.

Multi-view (localize3d_multi): run per top-K retrieved frame, concatenate,
voxel-downsample, centroid + bbox.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .depth_reader import load_depth_frame
    from .projection import project_depth_frame, scale_mask_to_depth, voxel_downsample
except ImportError:
    from depth_reader import load_depth_frame  # type: ignore
    from projection import project_depth_frame, scale_mask_to_depth, voxel_downsample  # type: ignore


def _load_rgb(rgb_path: Path) -> Optional[np.ndarray]:
    from PIL import Image

    if not rgb_path.is_file():
        return None
    return np.array(Image.open(rgb_path).convert("RGB"))


def localize3d(
    query: str,
    image_id: int,
    map_dir,
    sam3,
    rgb_path=None,
    retrieval_score: float = 0.0,
    depth_range: tuple = (0.1, 20.0),
    min_conf: float = 0.0,
    min_sam3_score: float = 0.5,
    max_points: Optional[int] = None,
    cache_key: Optional[tuple] = None,
) -> dict:
    """One retrieved keyframe -> world-frame 3D point cloud for `query`."""
    map_dir = Path(map_dir)
    frame_id = int(image_id)

    if rgb_path is None:
        rgb_path = map_dir / "images" / "left_rgb" / f"{frame_id:06d}_0000000000.png"
    rgb_path = Path(rgb_path)

    out = {
        "query": query, "frame_id": frame_id, "retrieval_score": float(retrieval_score),
        "sam3_score": 0.0, "detected": False, "point_cloud": None, "centroid": None,
        "has_rgb": False, "has_depth": False, "mask_px": 0,
    }

    # Find RGB if the exact name wasn't given (state.db uses <id:06d>_<ts>.png).
    if not rgb_path.is_file():
        rgb_dir = map_dir / "images" / "left_rgb"
        cands = sorted(rgb_dir.glob(f"{frame_id:06d}_*.png"))
        if cands:
            rgb_path = cands[0]
    rgb = _load_rgb(rgb_path)
    if rgb is None:
        return out
    out["has_rgb"] = True

    # SAM3 text-prompted segmentation on RGB.
    seg = sam3.segment(rgb, query, cache_key=cache_key or (query, frame_id))
    mask_rgb = seg["best_mask"]
    out["sam3_score"] = float(seg["best_score"])
    out["mask_px"] = 0 if mask_rgb is None else int(mask_rgb.sum())
    if mask_rgb is None or out["sam3_score"] < min_sam3_score:
        # No mask, or too low confidence -> reject (avoid projecting background).
        return out
    out["detected"] = True

    # Point cloud transformation (IR depth as RGB depth; no RGB<->IR transform).
    frame = load_depth_frame(map_dir, frame_id, pixels=True)
    if frame is None:
        out["detected"] = True  # object seen, but no dense depth frame
        out["has_depth"] = False
        return out
    out["has_depth"] = True
    mask_ir = scale_mask_to_depth(mask_rgb, (frame.height, frame.width))
    cloud = project_depth_frame(
        frame, mask=mask_ir, depth_range=depth_range, min_conf=min_conf, max_points=max_points,
    )
    out["point_cloud"] = cloud
    if cloud is not None and len(cloud) > 0:
        out["centroid"] = cloud.mean(axis=0)
    return out


def localize3d_multi(
    query: str,
    image_ids,
    map_dir,
    sam3,
    retrieval_scores=None,
    top_k: Optional[int] = None,
    rgb_names: Optional[dict] = None,
    voxel: float = 0.0,
    depth_range: tuple = (0.1, 20.0),
    min_conf: float = 0.0,
    **kw,
) -> dict:
    """Top-K retrieved keyframes -> fused world-frame 3D point cloud."""
    ids = list(image_ids)
    if top_k is not None:
        ids = ids[:top_k]
    retrieval_scores = retrieval_scores or {}

    map_dir = Path(map_dir)

    per_frame = []
    clouds = []
    for iid in ids:
        rgb_path = None
        if rgb_names is not None and iid in rgb_names:
            rgb_path = map_dir / "images" / rgb_names[iid]
        res = localize3d(
            query, int(iid), map_dir, sam3, rgb_path=rgb_path,
            retrieval_score=retrieval_scores.get(int(iid), 0.0),
            depth_range=depth_range, min_conf=min_conf,
            **kw,
        )
        per_frame.append(res)
        if res["point_cloud"] is not None and len(res["point_cloud"]) > 0:
            clouds.append(res["point_cloud"])

    if not clouds:
        cloud = None
    else:
        cloud = np.vstack(clouds)
        if voxel > 0:
            cloud = voxel_downsample(cloud, voxel)

    return {
        "query": query,
        "point_cloud": cloud,
        "centroid": (cloud.mean(axis=0) if cloud is not None and len(cloud) else None),
        "bbox": (
            {"min": cloud.min(axis=0).tolist(), "max": cloud.max(axis=0).tolist()}
            if cloud is not None and len(cloud) else None
        ),
        "num_points": int(len(cloud)) if cloud is not None else 0,
        "num_frames": len(per_frame),
        "per_frame": per_frame,
    }
