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

#include "dense_mapper/rigid.h"
#include "map_manager/map.h"

namespace vslam
{

class ConfigReader;

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
  /// A depth frame is fused only after this many newer keyframes, so local BA
  /// has refined its pose. 0 fuses at once.
  int integrate_delay_frames = 5;
  /// Poses of the newest N submitted keyframes are refreshed on every keyframe.
  int pose_track_frames = 30;
  /// A fused frame whose refreshed pose moved more than this is re-fused.
  double pose_update_trans_m = 0.01;
  double pose_update_rot_deg = 0.3;
  /// With no keyframe for this long, delayed frames are fused anyway. <=0 never.
  double idle_flush_s = 2.0;
  /// Voxels in front of an observed surface, outside the truncation band, lose weight.
  bool carve_free_space = true;
  /// Fraction of the maximum voxel weight removed by one such observation.
  /// 1 clears a saturated surface in a single view.
  double carve_weight = 1.0;
  /// Voxels below this weight are neither extracted nor treated as surface.
  double surface_min_weight = 0.5;
  /// Drop a depth pixel when a 4-neighbour differs by more than this share of
  /// its depth (flying pixels at depth edges). <=0 disables.
  double edge_ratio = 0.08;
  /// Block rebuilds after pose changes / removals are batched over at most this
  /// many keyframes, or run as soon as the queue is empty.
  int replay_max_defer_jobs = 5;
  /// Redundant keyframes (grid-replaced by a newer pass of the same cell)
  /// leave their contribution in the volume when false; changes are cleared
  /// by the newer pass's observations. true rebuilds their blocks without them.
  bool rebuild_redundant_removals = false;
  /// Fusion-thread time per call spent on a submap rebuild; the rest continues
  /// on later calls, so new keyframes are not held up. <=0: a whole submap per call.
  double replay_budget_ms = 20.0;
  /// Near-surface voxel replacement: a newer view whose sdf (in truncation
  /// units) has the opposite sign and differs by more than this replaces the
  /// voxel instead of averaging into it, so a surface that moved by less than
  /// the truncation distance updates in one view. <=0 disables.
  double replace_sdf_diff = 0.5;
  /// Minimum depth confidence (0..1) of the pixel for a replacement.
  double replace_min_conf = 0.5;
  /// Weight a replaced voxel restarts with; only voxels above it are replaced.
  double replace_weight = 2.0;
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
  /// OpenMP threads for TSDF fusion / rebuild. <=0 uses all cores (hurts VIO).
  int num_threads = 2;
  /// TSDF submaps are saved under <map>/dense/tsdf at this period and on stop,
  /// so a resumed map loads them instead of re-fusing every stored depth frame.
  /// <=0 saves only on stop.
  double snapshot_period_s = 30.0;
  /// Volume statistics (empty blocks, occupancy, points per block) are logged
  /// with a publish at most this often. <=0 disables.
  double stats_period_s = 10.0;
  /// Every N-th pixel in x and y seeds the candidate blocks of a fusion. The
  /// voxel update still reads every pixel. Capped per frame so neighbouring
  /// seeds stay within one voxel at the farthest depth.
  int integrate_pixel_stride = 4;
  /// Blocks whose voxels all lost their weight are freed. A tombstone keeps
  /// their newest keyframe id so an older frame cannot write them back.
  bool free_empty_blocks = true;
  /// A keyframe within both thresholds of one of the last inferred keyframes
  /// skips S²M² (its poses and removals still apply). <=0 disables either test.
  double skip_similar_trans_m = 0.10;
  double skip_similar_rot_deg = 5.0;
  /// At most this many keyframe images wait for S²M²; the oldest images are
  /// dropped first. Inference also waits while this many depths wait for
  /// fusion. <=0 unlimited.
  int max_queue_frames = 30;
  /// A submap unused for this long (not the active one) keeps only its surface
  /// in memory; its voxels go to the snapshot file and are read back when a
  /// keyframe, carve or rebuild touches it. <=0 keeps every submap in memory.
  double freeze_idle_s = 30.0;

  /// Reads this struct's keys from vslam.yaml.
  void load(ConfigReader &reader);
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
  /// Poses are cam_from_world of the (unrectified) left camera.
  void submit(uint32_t image_id, bool registered, const cv::Mat &left, const cv::Mat &right,
              const dense::Rigid &cam_from_world,
              const std::vector<uint32_t> &removed_ids,
              const std::vector<uint32_t> &redundant_ids,
              const std::vector<std::pair<uint32_t, dense::Rigid>> &pose_updates,
              const std::vector<Eigen::Vector3f> &sparse_xyz);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vslam

#endif
