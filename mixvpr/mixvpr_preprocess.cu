// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "mixvpr/mixvpr_preprocess.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>

namespace colmap {
namespace {

__device__ __forceinline__ float SampleChannel(const uint8_t* src,
                                               int width,
                                               int height,
                                               int pitch,
                                               int channels,
                                               int x,
                                               int y,
                                               int c) {
  x = max(0, min(width - 1, x));
  y = max(0, min(height - 1, y));
  if (channels == 1) {
    return static_cast<float>(src[y * pitch + x]);
  }
  return static_cast<float>(src[y * pitch + x * channels + c]);
}

__global__ void MixVprPreprocessKernel(const uint8_t* src,
                                       int src_w,
                                       int src_h,
                                       int src_pitch,
                                       int src_c,
                                       float* dst,
                                       int dst_size) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_size || y >= dst_size) {
    return;
  }

  const float gx =
      (static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) /
          static_cast<float>(dst_size) -
      0.5f;
  const float gy =
      (static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) /
          static_cast<float>(dst_size) -
      0.5f;
  const int x0 = static_cast<int>(floorf(gx));
  const int y0 = static_cast<int>(floorf(gy));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float ax = gx - static_cast<float>(x0);
  const float ay = gy - static_cast<float>(y0);
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float stdv[3] = {0.229f, 0.224f, 0.225f};
  const int plane = dst_size * dst_size;

  for (int c = 0; c < 3; ++c) {
    const int sc = (src_c == 1) ? 0 : c;
    const float v00 =
        SampleChannel(src, src_w, src_h, src_pitch, src_c, x0, y0, sc);
    const float v10 =
        SampleChannel(src, src_w, src_h, src_pitch, src_c, x1, y0, sc);
    const float v01 =
        SampleChannel(src, src_w, src_h, src_pitch, src_c, x0, y1, sc);
    const float v11 =
        SampleChannel(src, src_w, src_h, src_pitch, src_c, x1, y1, sc);
    const float v =
        ((1.0f - ax) * (1.0f - ay) * v00 + ax * (1.0f - ay) * v10 +
         (1.0f - ax) * ay * v01 + ax * ay * v11) /
        255.0f;
    dst[c * plane + y * dst_size + x] = (v - mean[c]) / stdv[c];
  }
}

}  // namespace

bool MixVprPreprocessCuda(const uint8_t* h_src,
                          int src_width,
                          int src_height,
                          int src_pitch,
                          int src_channels,
                          uint8_t* d_src,
                          size_t d_src_bytes,
                          float* d_nchw,
                          int dst_size,
                          cudaStream_t stream) {
  if (h_src == nullptr || d_src == nullptr || d_nchw == nullptr ||
      src_width <= 0 || src_height <= 0 || src_pitch <= 0 || dst_size <= 0) {
    return false;
  }
  if (src_channels != 1 && src_channels != 3) {
    return false;
  }
  const size_t src_bytes =
      static_cast<size_t>(src_pitch) * static_cast<size_t>(src_height);
  if (src_bytes > d_src_bytes) {
    LOG(ERROR) << "MixVPR preprocess: source image too large (" << src_bytes
               << " > " << d_src_bytes << ")";
    return false;
  }
  cudaError_t err = cudaMemcpyAsync(
      d_src, h_src, src_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    LOG(ERROR) << "MixVPR preprocess H2D failed: " << cudaGetErrorString(err);
    return false;
  }
  const dim3 block(16, 16);
  const dim3 grid((dst_size + block.x - 1) / block.x,
                  (dst_size + block.y - 1) / block.y);
  MixVprPreprocessKernel<<<grid, block, 0, stream>>>(
      d_src, src_width, src_height, src_pitch, src_channels, d_nchw, dst_size);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "MixVPR preprocess kernel failed: "
               << cudaGetErrorString(err);
    return false;
  }
  return true;
}

}  // namespace colmap
