"""2D->3D point cloud transformation (mask/depth -> world-frame point cloud).

Self-contained; aligns with memory-over-maps' project_mask_to_3d_cloud but
uses THIS system's coordinate convention, which needs no manual flip:

  rectified-left-IR camera (OpenCV): +x right, +y down, +z forward
  P_cam   = [x_cam, y_cam, z_cam]            # back-projection below
  cam_from_world (R,t): X_cam = R @ X_world + t
  world_from_cam = inv(cam_from_world)
  P_world = world_from_cam @ P_cam           # 4x4 homogeneous applied

The DFR2 `cam_from_world` already encodes the (rectified) camera->world
relationship in the system's world frame, so we just invert it. We do NOT
apply MoM's [x,-y,-z] (that is Habitat/ARKit-world specific) and we do NOT
apply any RGB<->IR transform. We use IR-left depth as the RGB camera's depth,
per project decision.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from .depth_reader import DepthFrame
except ImportError:
    from depth_reader import DepthFrame  # type: ignore


def _homog(points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.hstack([points.astype(np.float64), ones])


def backproject_pixels(
    us: np.ndarray,
    vs: np.ndarray,
    zs: np.ndarray,
    intrinsics: dict,
) -> np.ndarray:
    """Image pixels (u,v) + depth z (m) -> camera-frame points (N,3)."""
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    x_cam = (us.astype(np.float64) - cx) * zs / fx
    y_cam = (vs.astype(np.float64) - cy) * zs / fy
    z_cam = zs.astype(np.float64)
    return np.stack([x_cam, y_cam, z_cam], axis=1)


def cam_to_world(p_cam: np.ndarray, world_from_cam: np.ndarray) -> np.ndarray:
    """Transform camera-frame points (N,3) to world frame (N,3)."""
    return (world_from_cam @ _homog(p_cam).T).T[:, :3]


def project_world_to_pixel(
    p_world: np.ndarray,
    cam_from_world: np.ndarray,
    intrinsics: dict,
) -> tuple:
    """World point (3,) -> (u, v, depth) in this camera. Returns (nan,nan,nan) if behind."""
    pw = np.array([*p_world, 1.0], dtype=np.float64)
    p_cam = (cam_from_world @ pw)[:3]
    z = p_cam[2]
    if z <= 1e-9:
        return (float("nan"), float("nan"), float("nan"))
    u = intrinsics["fx"] * p_cam[0] / z + intrinsics["cx"]
    v = intrinsics["fy"] * p_cam[1] / z + intrinsics["cy"]
    return (u, v, z)


def project_mask_to_3d_cloud(
    mask: Optional[np.ndarray],
    depth_m: np.ndarray,
    intrinsics: dict,
    world_from_cam: np.ndarray,
    depth_range: tuple = (0.1, 20.0),
    conf: Optional[np.ndarray] = None,
    min_conf: float = 0.0,
    max_points: Optional[int] = None,
    rng_seed: int = 42,
) -> Optional[np.ndarray]:
    """Backproject masked pixels to a world-frame (N,3) point cloud.

    Args:
        mask: (H,W) bool / nonzero. None -> all pixels.
        depth_m: (H,W) float depth in meters. <=0 or out of range -> skipped.
        intrinsics: {'fx','fy','cx','cy'} (rectified).
        world_from_cam: 4x4 world_from_cam (= inv(cam_from_world)).
        conf: optional (H,W) confidence in 0..1; pixels < min_conf skipped.
        max_points: optional random subsample.
    Returns:
        (N,3) float64 world points, or None if no valid points.
    """
    h, w = depth_m.shape
    if mask is None:
        ys, xs = np.mgrid[0:h, 0:w].reshape(2, -1)
    else:
        m = mask
        if m.shape != (h, w):
            raise ValueError(f"mask shape {m.shape} != depth shape {(h, w)}")
        ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None

    zs = depth_m[ys, xs].astype(np.float64)
    valid = (zs > depth_range[0]) & (zs < depth_range[1])
    if conf is not None and min_conf > 0.0:
        valid &= conf[ys, xs] >= min_conf
    if not np.any(valid):
        return None

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    zs = zs[valid]

    p_cam = backproject_pixels(xs, ys, zs, intrinsics)
    p_world = cam_to_world(p_cam, world_from_cam)

    if max_points is not None and len(p_world) > max_points:
        idx = np.random.default_rng(rng_seed).choice(len(p_world), max_points, replace=False)
        p_world = p_world[idx]
    return p_world


def project_depth_frame(
    frame: DepthFrame,
    mask: Optional[np.ndarray] = None,
    depth_range: tuple = (0.1, 20.0),
    min_conf: float = 0.0,
    max_points: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Convenience: unproject a DFR2 DepthFrame -> world (N,3) cloud.

    With no SAM3 yet, pass mask=None to dump the whole frame, or a (H,W) bool
    mask (already scaled to IR resolution)."""
    return project_mask_to_3d_cloud(
        mask=mask,
        depth_m=frame.depth_m,
        intrinsics=frame.intrinsics(),
        world_from_cam=frame.world_from_cam(),
        depth_range=depth_range,
        conf=frame.conf,
        min_conf=min_conf,
        max_points=max_points,
    )


def scale_mask_to_depth(mask: np.ndarray, depth_shape: tuple, threshold: float = 0.5) -> np.ndarray:
    """Resize a (h_rgb,w_rgb) bool mask to (H_ir,W_ir) nearest-neighbour.

    Used when the mask comes from the RGB image but we read IR depth (no
    RGB<->IR transform; resolution difference handled by resize)."""
    import cv2

    h, w = depth_shape
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if m.shape == (h, w):
        return m.astype(bool)
    resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return resized >= (threshold * 255)


def voxel_downsample(cloud: np.ndarray, voxel: float) -> np.ndarray:
    """Simple voxel downsampling (one point per voxel, mean)."""
    if cloud is None or len(cloud) == 0 or voxel <= 0:
        return cloud
    keys = np.floor(cloud / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    k = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=k)
    out = np.zeros((k, 3))
    for ax in range(3):
        out[:, ax] = np.bincount(inv, weights=cloud[:, ax], minlength=k) / cnt
    return out
