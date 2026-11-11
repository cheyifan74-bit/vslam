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
  /// A depth frame is fused only after this many newer keyframes, so local BA
  /// has refined its pose and later frames can verify it. 0 fuses at once.
  int integrate_delay_frames = 5;
  /// Poses of the newest N submitted keyframes are refreshed on every keyframe.
  int pose_track_frames = 30;
  /// A fused frame whose refreshed pose moved more than this is re-fused.
  double pose_update_trans_m = 0.01;
  double pose_update_rot_deg = 0.3;
  /// With no keyframe for this long, delayed frames are fused anyway. <=0 never.
  double idle_flush_s = 2.0;
  /// Voxels in front of an observed surface are pushed towards free space.
  bool carve_free_space = true;
  /// Weight scale of a free-space update relative to a surface update.
  double carve_weight = 0.5;
  /// Voxels below this weight are neither extracted nor treated as surface.
  double surface_min_weight = 0.5;
  /// Drop a depth pixel when a 4-neighbour differs by more than this share of
  /// its depth (flying pixels at depth edges). <=0 disables.
  double edge_ratio = 0.08;
  /// Block rebuilds after pose changes / removals are batched over at most this
  /// many keyframes, or run as soon as the queue is empty.
  int replay_max_defer_jobs = 5;
  /// Redundant keyframes (culled / capacity-evicted) leave their contribution
  /// in the volume when false; only scene-change removals rebuild blocks.
  bool rebuild_redundant_removals = false;
  /// Depth pixels of only the newest N fused keyframes stay in memory (they are
  /// the verification references); older ones are re-read from disk when a
  /// rebuild needs them. <=0 keeps every frame in memory.
  int resident_frames = 30;
  /// Minimum interval between two published updates.
  double publish_period_s = 1.0;
  /// A keyframe joins the submap that already holds most of its view, else the
  /// active submap while it holds fewer than submap_max_frames keyframes and
  /// is within submap_max_distance_m of its origin. <=0: one global volume.
  int submap_max_frames = 40;
  double submap_max_distance_m = 4.0;
};

struct DenseCloud
{
  std::vector<Eigen::Vector3f> xyz;
  std::vector<uint8_t> rgb;
};

/// Surface of one submap, in the submap frame. world = R(q) * p + t.
struct DenseSubmapCloud
{
  uint32_t id = 0;
  float q_world_submap[4] = {0, 0, 0, 1}; ///< xyzw
  float t_world_submap[3] = {0, 0, 0};
  /// Shared between updates when only the submap pose changed.
  std::shared_ptr<const DenseCloud> points;
};

struct DenseUpdate
{
  /// Submaps whose surface or pose changed since the previous update.
  std::vector<std::shared_ptr<const DenseSubmapCloud>> changed;
  std::vector<uint32_t> removed;
  /// Every current submap, for full-map consumers and late subscribers.
  std::vector<std::shared_ptr<const DenseSubmapCloud>> submaps;
  float point_size_m = 0.02f;
};

/// Consumes registered stereo keyframes and fuses them into TSDF submaps.
/// submit() only enqueues. Rectification and S²M² run on the inference thread,
/// fusion and rebuilds on the fusion thread, so the GPU and CPU stages overlap.
/// Each submap has its own frame whose pose follows the consensus of its
/// keyframes, so a loop correction moves submaps rigidly; only keyframes that
/// moved relative to the rest of their submap rebuild, and only inside it.
class DenseMapper
{
public:
  /// Called on the fusion thread; the receiver should only keep the pointer.
  using CloudCallback = std::function<void(std::shared_ptr<const DenseUpdate>)>;

  DenseMapper(DenseConfig config, std::vector<CameraParams> cameras, std::string map_dir);
  ~DenseMapper();

  DenseMapper(const DenseMapper &) = delete;
  DenseMapper &operator=(const DenseMapper &) = delete;

  void setCloudCallback(CloudCallback callback);
  void start();
  void stop();

  /// Image ids whose current poses should be passed as pose_updates.
  std::vector<uint32_t> trackedIds() const;

  /// Copies the keyframe and returns. Rectification and S²M² run on the dense
  /// thread, so the next keyframe is not blocked on the forward pass.
  /// redundant_ids: the subset of removed_ids dropped only for redundancy.
  /// pose_updates: current poses of any known keyframes (loop corrections and trackedIds()).
  /// sparse_xyz: map points observed by this keyframe.
  void submit(uint32_t image_id, bool registered, const cv::Mat &left, const cv::Mat &right,
              const colmap::MappingPose &cam_from_world,
              const std::vector<uint32_t> &removed_ids,
              const std::vector<uint32_t> &redundant_ids,
              const std::vector<std::pair<uint32_t, colmap::MappingPose>> &pose_updates,
              const std::vector<Eigen::Vector3f> &sparse_xyz);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vslam

#endif
