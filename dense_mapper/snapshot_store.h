#ifndef VSLAM_DENSE_MAPPER_SNAPSHOT_STORE_H
#define VSLAM_DENSE_MAPPER_SNAPSHOT_STORE_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dense_mapper/depth_frame.h"
#include "dense_mapper/submap.h"

namespace vslam
{
namespace dense
{

/// Per-submap entry of the index.
struct IndexSubmap
{
  uint32_t id = 0;
  uint64_t file_gen = 0;
  int32_t num_members = 0;
  Rigid world_from_submap;
  BlockSet replay_dirty;
  std::unordered_set<uint32_t> replay_changed;
};

/// Per-fused-frame entry of the index.
struct IndexFrame
{
  uint32_t image_id = 0;
  uint32_t submap = 0;
  Rigid fused;
  BlockSet blocks;
};

struct IndexData
{
  uint64_t gen = 0;
  uint32_t active = 0;
  uint32_t next = 0;
  std::vector<IndexSubmap> submaps;
  std::vector<IndexFrame> frames;
};

/// TSDF snapshot on disk. index.bin: config fingerprint, submap poses /
/// members / pending rebuilds and every fused frame's submap, fused pose and
/// block list. Voxels live in submap_<id>_<gen>.bin; only submaps changed
/// since the last save get a new file. The index is renamed into place last,
/// so it always names complete files.
class SnapshotStore
{
public:
  SnapshotStore(std::filesystem::path dir, const VoxelGrid &grid) : dir_(std::move(dir)), grid_(grid)
  {
  }

  const std::filesystem::path &dir() const { return dir_; }

  bool writeSubmap(const Submap &submap, uint64_t gen) const;
  /// Reads voxels and tombstones into the submap (id set); mark_dirty marks
  /// every block for surface extraction.
  bool readSubmap(Submap *submap, uint64_t gen, bool mark_dirty) const;

  bool writeIndex(uint64_t gen, uint32_t active, uint32_t next,
                  const std::vector<IndexSubmap> &submaps,
                  const std::vector<const DepthFrame *> &frames) const;
  /// False with an empty *why when there is no index. data->gen is set as
  /// soon as it is read, also when a later part fails.
  bool readIndex(IndexData *data, std::string *why) const;

  /// Removes submap files other than the given (id, generation) pairs.
  void removeStaleSubmaps(const std::vector<std::pair<uint32_t, uint64_t>> &keep) const;

private:
  std::filesystem::path submapPath(uint32_t id, uint64_t gen) const;
  void putFingerprint(std::ostream &out) const;
  bool checkFingerprint(std::istream &in) const;

  std::filesystem::path dir_;
  const VoxelGrid grid_;
};

} // namespace dense
} // namespace vslam

#endif
