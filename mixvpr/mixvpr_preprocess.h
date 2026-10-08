// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef VSLAM_MIXVPR_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace colmap {

#ifdef VSLAM_MIXVPR_CUDA_ENABLED

// Bilinear resize + ImageNet normalize into NCHW float32 on GPU.
bool MixVprPreprocessCuda(const uint8_t* h_src,
                          int src_width,
                          int src_height,
                          int src_pitch,
                          int src_channels,
                          uint8_t* d_src,
                          size_t d_src_bytes,
                          float* d_nchw,
                          int dst_size,
                          cudaStream_t stream);

#endif  // VSLAM_MIXVPR_CUDA_ENABLED

}  // namespace colmap
