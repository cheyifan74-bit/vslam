// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include "colmap/util/types.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace colmap {

class Bitmap;

inline constexpr int kMixVprInputSize = 320;
inline constexpr int kMixVprDescDim = 512;

// MixVPR (ResNet50, 512-D) via TensorRT FP16 engine.
class MixVprEncoder {
 public:
  MixVprEncoder(const std::string& model_path,
                bool use_gpu,
                const std::string& gpu_index);
  ~MixVprEncoder();

  MixVprEncoder(const MixVprEncoder&) = delete;
  MixVprEncoder& operator=(const MixVprEncoder&) = delete;

  bool Encode(const std::string& image_abs_path, Eigen::VectorXf* desc);
  bool Encode(const Bitmap& bitmap, Eigen::VectorXf* desc);
  bool UsedGpu() const;
  int DescDim() const;
  const std::string& Backend() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Cosine ranking over MixVPR descriptors. Descriptors are assumed
// L2-normalized. Score is cosine, larger better. `knn` <= 0 keeps all.
std::vector<std::pair<image_t, float>> RankMixVprMatches(
    const Eigen::VectorXf& query,
    const std::vector<std::pair<image_t, Eigen::VectorXf>>& entries,
    const std::function<bool(image_t)>& usable,
    int knn = 0);

}  // namespace colmap
