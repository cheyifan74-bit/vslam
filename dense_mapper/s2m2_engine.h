#ifndef VSLAM_DENSE_MAPPER_S2M2_ENGINE_H
#define VSLAM_DENSE_MAPPER_S2M2_ENGINE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace vslam
{

/// TensorRT runtime for a fixed-size S²M² engine. PyTorch is not linked.
/// Inputs are RGB 0–255. Outputs are disparity, occlusion and confidence.
class S2M2Engine
{
public:
  S2M2Engine();
  ~S2M2Engine();

  S2M2Engine(const S2M2Engine &) = delete;
  S2M2Engine &operator=(const S2M2Engine &) = delete;

  bool load(const std::string &engine_path, int width, int height);
  bool ready() const { return ready_; }

  /// `left_rgb` and `right_rgb` are 8-bit, 3-channel, width x height.
  bool infer(const cv::Mat &left_rgb, const cv::Mat &right_rgb, std::vector<float> *disparity,
             std::vector<float> *occlusion, std::vector<float> *confidence);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool ready_ = false;
  int width_ = 0;
  int height_ = 0;
};

} // namespace vslam

#endif
