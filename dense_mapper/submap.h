#ifndef VSLAM_DENSE_MAPPER_SUBMAP_H
#define VSLAM_DENSE_MAPPER_SUBMAP_H

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "dense_mapper/dense_mapper.h"
#include "dense_mapper/dense_types.h"

namespace vslam
{
namespace dense
{

struct SurfacePiece
{
  std::vector<Eigen::Vector3f> xyz;
  std::vector<uint8_t> rgb;
};

/// One TSDF volume with its own frame. The submap pose follows the consensus
/// of its members, so a loop correction moves it rigidly and only members
/// that moved relative to the others are rebuilt.
struct Submap
{
  uint32_t id = 0;
  Rigid world_from_submap;
  int num_members = 0;
  BlockMap<std::vector<Voxel>> blocks;
  BlockMap<SurfacePiece> surface;
  BlockSet surface_dirty;
  BlockSet replay_dirty;
  std::unordered_set<uint32_t> replay_changed;
  bool pose_changed = false;
  std::shared_ptr<const DenseCloud> points;
  std::shared_ptr<const DenseSubmapCloud> published;
  /// Grow-only bounds of every block ever allocated, in block keys.
  bool bounded = false;
  BlockKey lo;
  BlockKey hi;
  /// Snapshot file generation holding this submap's voxels; 0 = never written.
  uint64_t file_gen = 0;
  bool snapshot_dirty = true;
  /// Freed empty blocks -> newest keyframe id that wrote or cleared them.
  BlockMap<uint32_t> tombstones;
  /// Frozen: voxels and tombstones live only in the snapshot file (file_gen),
  /// the surface stays. frozen_keys keeps the block keys for overlap and
  /// carve tests; anything that writes voxels thaws the submap first.
  bool frozen = false;
  BlockSet frozen_keys;
  Clock::time_point last_used = Clock::now();

  /// The block's voxels, allocated if missing.
  std::vector<Voxel> &block(const BlockKey &key);
  size_t numBlocks() const;
  bool hasBlock(const BlockKey &key) const;
  /// Frees a block whose voxels all lost their weight, keeping its newest
  /// writer as a tombstone.
  void freeBlock(const BlockKey &key);
  void growBounds(const BlockKey &key);
  /// Voxel by block-local index; indices one step outside the block address the neighbour.
  const Voxel *voxelAt(BlockKey key, int ix, int iy, int iz) const;
  /// Freezes: releases voxels and tombstones, keeping the block keys. The
  /// snapshot file at file_gen must hold them.
  void releaseVoxels();
};

bool AllEmpty(const std::vector<Voxel> &cells);

} // namespace dense
} // namespace vslam

#endif
