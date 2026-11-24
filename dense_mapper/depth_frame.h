#ifndef VSLAM_DENSE_MAPPER_DEPTH_FRAME_H
#define VSLAM_DENSE_MAPPER_DEPTH_FRAME_H

#include <cstdint>
#include <vector>

#include "dense_mapper/dense_types.h"

namespace vslam
{
namespace dense
{

/// Filtered depth of one keyframe in the rectified left camera, plus the
/// state of its fusion into a submap.
struct DepthFrame
{
  uint32_t image_id = 0;
  /// cam_from_world (rectified camera).
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  int width = 0;
  int height = 0;
  float fx = 0;
  float fy = 0;
  float cx = 0;
  float cy = 0;
  float baseline = 0;
  std::vector<uint16_t> depth_mm;
  std::vector<uint8_t> conf;
  std::vector<uint8_t> gray;
  /// False while the frame waits for integrate_delay_frames newer keyframes.
  bool integrated = false;
  uint32_t submap = 0;
  /// cam_from_submap the frame was last fused with.
  Rigid fused;
  BlockSet blocks;

  bool resident() const { return !depth_mm.empty(); }
  Rigid pose() const
  {
    Rigid pose;
    pose.q = q;
    pose.t = t;
    return pose;
  }
  Eigen::Vector3d center() const { return -(q.conjugate() * t); }
  void releasePixels()
  {
    std::vector<uint16_t>().swap(depth_mm);
    std::vector<uint8_t>().swap(conf);
    std::vector<uint8_t>().swap(gray);
  }
};

} // namespace dense
} // namespace vslam

#endif
