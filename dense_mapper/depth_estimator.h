#ifndef VSLAM_DENSE_MAPPER_DEPTH_ESTIMATOR_H
#define VSLAM_DENSE_MAPPER_DEPTH_ESTIMATOR_H

#include <cstdint>
#include <deque>
#include <vector>

#include <opencv2/core.hpp>

#include "dense_mapper/dense_mapper.h"
#include "dense_mapper/depth_frame.h"
#include "dense_mapper/s2m2_engine.h"
#include "dense_mapper/stereo_rectifier.h"

namespace vslam
{
namespace dense
{

struct StereoKeyframe
{
  uint32_t image_id = 0;
  cv::Mat left;
  cv::Mat right;
  Rigid cam_from_world;
  /// Map points observed by this keyframe.
  std::vector<Eigen::Vector3f> sparse_xyz;
};

/// Rectification, S²M² and depth filtering of a stereo keyframe. Touches no
/// map state; used by the inference thread, except rectifiedPose().
class DepthEstimator
{
public:
  explicit DepthEstimator(const DenseConfig &config) : config_(config) {}

  /// False when the rectifier cannot be set up. A missing engine only leaves ready() false.
  bool init(const std::vector<CameraParams> &cameras);
  bool ready() const { return engine_.ready(); }

  /// Filtered depth of the keyframe in the rectified left camera; false if none is left.
  bool build(const StereoKeyframe &keyframe, DepthFrame *frame);

  /// True when the pose is within the skip thresholds of one of the last
  /// inferred keyframes; otherwise the pose is remembered as inferred.
  bool similarToRecent(const Rigid &cam_from_world);

  /// cam_from_world of the left camera -> of the rectified camera.
  Rigid rectifiedPose(const Rigid &cam_from_world) const;

private:
  Eigen::Matrix3d rectFromLeft() const;

  const DenseConfig config_;
  StereoRectifier rectifier_;
  S2M2Engine engine_;
  /// Poses (cam_from_world) of the last inferred keyframes.
  std::deque<Rigid> recent_inferred_;
};

} // namespace dense
} // namespace vslam

#endif
