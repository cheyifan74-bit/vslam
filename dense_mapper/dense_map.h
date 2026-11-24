#ifndef VSLAM_DENSE_MAPPER_DENSE_MAP_H
#define VSLAM_DENSE_MAPPER_DENSE_MAP_H

#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dense_mapper/dense_mapper.h"
#include "dense_mapper/depth_frame.h"
#include "dense_mapper/frame_store.h"
#include "dense_mapper/snapshot_store.h"
#include "dense_mapper/submap.h"
#include "dense_mapper/surface_extractor.h"
#include "dense_mapper/tsdf_integrator.h"

namespace vslam
{
namespace dense
{

/// Map changes of one keyframe job, with poses already in the rectified camera.
struct MapJob
{
  std::vector<uint32_t> removed_ids;
  /// The subset of removed_ids dropped only for redundancy.
  std::vector<uint32_t> redundant_ids;
  /// cam_from_world of the rectified camera.
  std::vector<std::pair<uint32_t, Rigid>> pose_updates;
  std::unique_ptr<DepthFrame> depth;
};

struct ApplyStats
{
  int fused = 0;
  double fuse_ms = 0.0;
  double rebuild_ms = 0.0;
  double publish_ms = 0.0;
  int retired = 0;
  size_t moved_submaps = 0;
  size_t pending = 0;
};

/// The TSDF submap map of the fusion thread: depth frames and their fusion
/// state, submap assignment and poses, rebuilds after pose changes, surface
/// publishing, freezing idle submaps and the snapshot on disk.
class DenseMap
{
public:
  DenseMap(const DenseConfig &config, FrameStore frame_store, std::filesystem::path tsdf_dir);

  void setCloudCallback(DenseMapper::CloudCallback callback)
  {
    cloud_callback_ = std::move(callback);
  }

  /// Reads the stored frames, restores the snapshot and re-fuses what it
  /// does not hold. stopping() interrupts the re-fusion.
  void load(const std::function<bool()> &stopping);

  /// Applies removals, pose updates and the new depth, fuses delayed frames
  /// and, unless stopping, advances rebuilds, publishes, freezes and saves.
  ApplyStats apply(MapJob job, bool idle, bool stopping);
  /// No job waiting: continues rebuilds and fuses delayed frames after idle_flush_s.
  void onIdle();
  /// A rebuild is running or waiting.
  bool rebuildPending() const;
  /// Writes pending frames with their refined pose and saves the snapshot.
  void shutdown();

private:
  /// A submap rebuild in progress. Frames are replayed into a shadow copy of
  /// the dirty blocks, a time slice at a time; the live blocks are replaced
  /// when all are done.
  struct Rebuild
  {
    bool active = false;
    uint32_t submap = 0;
    BlockSet dirty;
    std::unordered_set<uint32_t> changed;
    bool any_changed = false;
    uint32_t changed_max = 0;
    /// Newest frame id known at the start; live voxels written later win.
    uint32_t start_id = 0;
    std::vector<uint32_t> todo;
    size_t next = 0;
    /// Frames fused into the live submap during the rebuild, replayed last.
    std::set<uint32_t> fused_live;
    Submap shadow;
    size_t num_members = 0;
    size_t replayed = 0;
    size_t moved = 0;
    int slices = 0;
    double work_ms = 0.0;
    Clock::time_point started;
  };

  // dense_map.cpp: fusion and submap assignment.
  void updatePose(uint32_t id, const Rigid &pose, std::unordered_set<uint32_t> *touched);
  int fusePending(size_t keep);
  void fuseFrame(DepthFrame *frame, Submap &submap, bool log);
  void evictOld();
  Submap &createSubmap(const DepthFrame &first);
  double overlap(const DepthFrame &frame, const Submap &submap) const;
  Submap &chooseSubmap(const DepthFrame &frame);
  void refreshSubmap(Submap &submap);
  void dropOrphans();
  bool isRebuilding(uint32_t submap_id) const;

  // dense_map_rebuild.cpp
  bool startRebuild();
  void stepRebuild(Clock::time_point deadline);
  void replayInto(DepthFrame *frame);
  void finishRebuild(Submap &submap);
  void flushReplay();

  // dense_map_publish.cpp
  void extractSurface(Submap &submap);
  bool hasUnpublished() const;
  void publish();
  void logStats(size_t total_points) const;
  void maybePublish();

  // dense_map_storage.cpp: freezing and the snapshot.
  bool freeze(Submap &submap);
  void thaw(Submap &submap);
  void maybeFreeze();
  void saveSnapshot(const char *reason);
  void maybeSnapshot();
  bool loadSnapshot(size_t *restored);

  const DenseConfig config_;
  const VoxelGrid grid_;
  const FrameStore frame_store_;
  const SnapshotStore snapshots_;
  const TsdfIntegrator integrator_;
  const SurfaceExtractor extractor_;
  DenseMapper::CloudCallback cloud_callback_;

  std::unordered_map<uint32_t, DepthFrame> frames_;
  std::deque<uint32_t> pending_;
  std::map<uint32_t, Submap> submaps_;
  uint32_t active_submap_ = 0;
  uint32_t next_submap_ = 1;
  std::vector<uint32_t> removed_submaps_;
  int jobs_since_dirty_ = 0;
  Rebuild rebuild_;
  Clock::time_point last_job_ = Clock::now();
  Clock::time_point last_publish_;
  Clock::time_point last_snapshot_ = Clock::now();
  Clock::time_point last_stats_;
  uint64_t snapshot_gen_ = 0;
  bool index_dirty_ = false;
};

} // namespace dense
} // namespace vslam

#endif
