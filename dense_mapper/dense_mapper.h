#ifndef VSLAM_DENSE_MAPPER_DENSE_MAPPER_H
#define VSLAM_DENSE_MAPPER_DENSE_MAPPER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "colmap/online_sfm/online_incremental_mapper.h"
#include "map_manager/map.h"

namespace vslam
{

struct DenseConfig
{
  bool enabled = false;
  std::string engine_path;
  int rect_width = 512;
  int rect_height = 512;
  double voxel_m = 0.05;
  double truncation_m = 0.15;
  double min_range_m = 0.4;
  double max_range_m = 8.0;
  double min_confidence = 0.5;
  /// S²M² occlusion output: values above this are visible (official vis uses 0.5).
  double min_visible = 0.5;
  /// Depth weight scales as (z_ref / z)^2 and is capped at 1. <=0 disables.
  /// Disparity noise grows with z^2 / (f B), so far pixels must not enter at full weight.
  double weight_reference_range_m = 2.0;
  /// Gray values at or above this are saturated and carry no stereo texture. 0 disables.
  int saturate_gray = 250;
  /// Drop dense depth that is farther than a nearby sparse point by both this
  /// ratio and the absolute margin. <=0 disables the sparse ceiling.
  double sparse_depth_ratio = 1.25;
  double sparse_depth_margin_m = 0.4;
  /// How far the sparse ceiling spreads, in rectified pixels. Keeps the halo
  /// small so a near ladder does not erase the wall behind it.
  int sparse_dilate_px = 12;
  /// How many earlier keyframes to test before a depth pixel may enter the
  /// volume. <=0 fuses every pixel.
  int verify_frames = 8;
  /// Ignore reference frames closer than this. The two views must actually move.
  double verify_min_baseline_m = 0.3;
  /// A reference depth within this distance of the pixel confirms it.
  /// A confirmed pixel is not written when another extractable surface already
  /// lies on the same ray, outside the truncation band but inside this distance.
  double verify_agree_m = 0.15;
  /// A reference depth farther than the pixel by this much sees through it.
  double verify_gap_m = 0.4;
};

struct DenseCloud
{
  std::vector<Eigen::Vector3f> xyz;
  std::vector<uint8_t> rgb;
};

/// Consumes registered stereo keyframes and fuses a block TSDF. submit() only
/// enqueues; rectification, S²M² and fusion run on the dense thread.
/// Loop-corrected and removed keyframes rebuild the blocks they touched.
class DenseMapper
{
public:
  using CloudCallback = std::function<void(const DenseCloud &)>;

  DenseMapper(DenseConfig config, std::vector<CameraParams> cameras, std::string map_dir);
  ~DenseMapper();

  DenseMapper(const DenseMapper &) = delete;
  DenseMapper &operator=(const DenseMapper &) = delete;

  void setCloudCallback(CloudCallback callback);
  void start();
  void stop();

  /// Copies the keyframe and returns. Rectification and S²M² run on the dense
  /// thread, so the next keyframe is not blocked on the forward pass.
  void submit(uint32_t image_id, bool registered, const cv::Mat &left, const cv::Mat &right,
              const colmap::MappingPose &cam_from_world,
              const std::vector<uint32_t> &removed_ids,
              const std::vector<std::pair<uint32_t, colmap::MappingPose>> &dirty_poses,
              const std::vector<Eigen::Vector3f> &sparse_xyz);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vslam

#endif
