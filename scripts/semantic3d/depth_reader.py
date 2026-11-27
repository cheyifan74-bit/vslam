"""Parse the dense mapper's per-keyframe depth file (DFR2).

Layout (see dense_mapper/frame_store.cpp, kMagic="DFR2", kPoseOffset=12):
  "DFR2" (4)
  image_id    uint32
  flags       uint32
  cam_from_world q[4] double   (x,y,z,w on disk; X_cam = R*X_world + t)
  cam_from_world t[3] double
  width,height int32
  fx,fy,cx,cy,baseline float32
  depth_mm    uint16[W*H]
  conf        uint8 [W*H]
  gray        uint8 [W*H]

The pose is cam_from_world of the *rectified* left IR camera, and the depth +
intrinsics are rectified too (R1 applied). It is kept current by the dense
mapper (updatePose -> writePose). No RGB<->IR transform is applied here.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

MAGIC = b"DFR2"
KPOSE_OFFSET = 12  # 4 magic + 4 image_id + 4 flags


def quat_to_rotation_matrix(q_xyzw) -> np.ndarray:
    x, y, z, w = (float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3]))
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_from_quat_t(q_xyzw, t) -> np.ndarray:
    """4x4 homogeneous cam_from_world, X_cam = M @ X_world(homogeneous)."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = quat_to_rotation_matrix(q_xyzw)
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


@dataclass
class DepthFrame:
    image_id: int
    q_xyzw: np.ndarray      # rectified-left-IR cam_from_world rotation (x,y,z,w)
    t: np.ndarray           # rectified-left-IR cam_from_world translation
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    depth_m: np.ndarray     # (H,W) float32, meters; 0 where no depth
    conf: np.ndarray        # (H,W) float32 0..1
    gray: np.ndarray        # (H,W) uint8

    def cam_from_world(self) -> np.ndarray:
        return pose_from_quat_t(self.q_xyzw, self.t)

    def world_from_cam(self) -> np.ndarray:
        return np.linalg.inv(self.cam_from_world())

    def intrinsics(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def read_depth_frame(path, pixels: bool = True) -> Optional[DepthFrame]:
    p = Path(path)
    if not p.is_file():
        return None
    data = p.read_bytes()
    if len(data) < KPOSE_OFFSET or data[:4] != MAGIC:
        return None
    try:
        image_id = struct.unpack_from("<I", data, 4)[0]
        q = struct.unpack_from("<4d", data, KPOSE_OFFSET)
        t = struct.unpack_from("<3d", data, KPOSE_OFFSET + 32)
        off = KPOSE_OFFSET + 32 + 24  # q(32) + t(24)
        width, height = struct.unpack_from("<2i", data, off)
        off += 8
        fx, fy, cx, cy, baseline = struct.unpack_from("<5f", data, off)
        off += 20
        count = width * height
        if pixels:
            need = off + count * 2 + count + count
            if len(data) < need:
                return None
            depth_mm = np.frombuffer(data, dtype="<u2", count=count, offset=off).reshape(height, width)
            off += count * 2
            conf = np.frombuffer(data, dtype=np.uint8, count=count, offset=off).reshape(height, width)
            off += count
            gray = np.frombuffer(data, dtype=np.uint8, count=count, offset=off).reshape(height, width)
            depth_m = depth_mm.astype(np.float32) * 1e-3
            conf_f = conf.astype(np.float32) / 255.0
        else:
            depth_m = np.zeros((height, width), dtype=np.float32)
            conf_f = np.zeros((height, width), dtype=np.float32)
            gray = np.zeros((height, width), dtype=np.uint8)
    except (struct.error, ValueError):
        return None

    q_arr = np.array(q, dtype=np.float64)
    q_arr /= max(np.linalg.norm(q_arr), 1e-12)
    return DepthFrame(
        image_id=int(image_id),
        q_xyzw=q_arr,
        t=np.array(t, dtype=np.float64),
        width=int(width),
        height=int(height),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), baseline=float(baseline),
        depth_m=depth_m,
        conf=conf_f,
        gray=gray,
    )


def load_depth_frame(map_dir, image_id: int, pixels: bool = True) -> Optional[DepthFrame]:
    """Read <map>/dense/frames/<image_id>.bin."""
    p = Path(map_dir) / "dense" / "frames" / f"{image_id}.bin"
    return read_depth_frame(p, pixels=pixels)
