#!/usr/bin/env bash
# Export one S²M² TensorRT engine for the rectified size used by dense mapping.
# Run this on the machine that will run vslam_node. The engine is tied to that
# GPU and to the system TensorRT (this machine has 10.16, not the README's 10.13).
#
# Weights are not in the git checkout. Download the S model once:
#   mkdir -p /home/cyf/s2m2/weights/pretrain_weights
#   curl -L -o /home/cyf/s2m2/weights/pretrain_weights/CH128NTR1.pth \
#     https://huggingface.co/minimok/s2m2/resolve/main/CH128NTR1.pth
#
# Usage: scripts/export_s2m2_engine.sh [S|M] [width] [height]
set -euo pipefail
MODEL="${1:-S}"
WIDTH="${2:-512}"
HEIGHT="${3:-512}"
ROOT="${S2M2_ROOT:-/home/cyf/s2m2}"
DEST="${VSLAM_MODEL_DIR:-/home/cyf/vslam_ws/src/vslam/data/models}"
cd "${ROOT}"
python demo/export_onnx.py --model_type "${MODEL}" --img_width "${WIDTH}" --img_height "${HEIGHT}" --num_refine 3
python demo/export_tensorrt.py --model_type "${MODEL}" --img_width "${WIDTH}" --img_height "${HEIGHT}" --precision fp16
mkdir -p "${DEST}"
install -m 644 "${ROOT}/weights/trt_save/S2M2_${MODEL}_${WIDTH}_${HEIGHT}_fp16.engine" \
  "${DEST}/S2M2_${MODEL}_${WIDTH}_${HEIGHT}_fp16.engine"
echo "engine installed at ${DEST}/S2M2_${MODEL}_${WIDTH}_${HEIGHT}_fp16.engine"
