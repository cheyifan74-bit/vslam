#!/usr/bin/env bash
# Build vslam + OpenVINS (ov_core / ov_init / ov_msckf) under ROS2 in a standard colcon workspace.
#
# Usage:  scripts/build_ros2.sh [extra colcon args...]
#         source ~/vslam_ws/install/setup.bash
#         ros2 launch vslam vslam.launch.py config:=euroc
#
# Workspace layout (override location with VSLAM_WS):
#   $VSLAM_WS/src/vslam    (this repo as a real directory, not a symlink)
# colcon does not crawl into a package directory, so the submodule is passed as an extra base path.
# ov_init and colmap must use the same Ceres (>= 2.1, colmap uses ceres::Manifold); override with CERES_DIR.
# Default: thirdparty/ceres-solver-2.2.0/install (same tree as the Ceres sources).
# INSTALL_RPATH_USE_LINK_PATH keeps that non-system libceres resolvable at runtime without LD_LIBRARY_PATH.
# CUDA archs default to "native" (the local GPU, e.g. sm_120 on RTX 50); mixvpr's own fallback (75;86;89)
# misses Blackwell. Override with CUDA_ARCHS, e.g. CUDA_ARCHS="86;89;120".
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VSLAM_WS=${VSLAM_WS:-$HOME/vslam_ws}
CERES_DIR=${CERES_DIR:-$ROOT/thirdparty/ceres-solver-2.2.0/install/lib/cmake/Ceres}

if [ ! -f "$CERES_DIR/CeresConfig.cmake" ]; then
  echo "CeresConfig.cmake not found in $CERES_DIR (set CERES_DIR to a Ceres >= 2.1 install)" >&2
  exit 1
fi

PKG="$VSLAM_WS/src/vslam"
if [ -L "$PKG" ]; then
  echo "$PKG is a symlink. Move the repository there instead of linking." >&2
  exit 1
fi
if [ ! -f "$PKG/package.xml" ]; then
  echo "vslam package not found at $PKG" >&2
  exit 1
fi
if [ "$(readlink -f "$ROOT")" != "$(readlink -f "$PKG")" ]; then
  echo "This checkout is $ROOT; build it from $PKG." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"

# colmap's FetchContent faiss needs CMake >= 3.24 (Ubuntu 22.04 ships 3.22). Prefer the project-local
# 3.31 install if present: python3 -m pip install --target thirdparty/cmake-3.31 "cmake>=3.31,<3.32"
LOCAL_CMAKE_BIN="$ROOT/thirdparty/cmake-3.31/cmake/data/bin"
if [ -x "$LOCAL_CMAKE_BIN/cmake" ]; then
  export PATH="$LOCAL_CMAKE_BIN:$PATH"
fi
echo "Using $(cmake --version | head -1) from $(command -v cmake)"

# CUDA toolkit from NVIDIA's apt repo installs to /usr/local/cuda without touching PATH.
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
command -v nvcc >/dev/null 2>&1 && echo "Using $(nvcc --version | tail -1)"

cd "$VSLAM_WS"
colcon build \
  --base-paths src src/vslam/open_vins \
  --packages-up-to vslam \
  "$@" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCeres_DIR="$CERES_DIR" -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-native}"
