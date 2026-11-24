#ifndef VSLAM_DENSE_MAPPER_TSDF_INTEGRATOR_H
#define VSLAM_DENSE_MAPPER_TSDF_INTEGRATOR_H

#include <vector>

#include "dense_mapper/dense_mapper.h"
#include "dense_mapper/depth_frame.h"
#include "dense_mapper/submap.h"

namespace vslam
{
namespace dense
{

/// Per-tile maximum ray length of a frame's depth. A block farther than every
/// depth it projects onto cannot be in front of the observed surface.
struct RangeTiles
{
  static constexpr int kTile = 16;
  int cols = 0;
  int rows = 0;
  float global_max = 0.f;
  std::vector<float> max_range;
};

/// Projective TSDF fusion of depth frames into submaps, and free-space
/// carving of other submaps. Works on in-memory voxels only: frozen submaps
/// must be thawed by the caller.
class TsdfIntegrator
{
public:
  TsdfIntegrator(const DenseConfig &config, const VoxelGrid &grid) : config_(config), grid_(grid) {}

  bool carves() const { return config_.carve_free_space && config_.carve_weight > 0.0; }

  /// Projective TSDF update of the frame's submap at the frame's current
  /// pose: every voxel of the candidate blocks is projected into the frame
  /// once. Candidates are the blocks around the depth plus, for free-space
  /// carving, existing blocks in the frustum. only != nullptr restricts the
  /// update (rebuild). Records the pose and blocks in the frame and returns
  /// the range tiles for carving other submaps (empty without carving).
  RangeTiles integrate(DepthFrame *frame, Submap &submap, const BlockSet *only, bool log) const;

  /// Whether carveInFront() can touch the submap; tests frozen block keys too.
  bool mayCarveSubmap(const DepthFrame &frame, const RangeTiles &tiles, const Submap &other) const;

  /// Zeros existing voxels of another (thawed) submap that sit in front of
  /// this frame's depth by more than the truncation distance. Other submaps
  /// count: a revisit often lands in a different submap from the keyframe
  /// that first saw the object. Only blocks that can be in front of the
  /// depth are visited; returns false when there were none.
  bool carveInFront(const DepthFrame &frame, const RangeTiles &tiles, Submap &other) const;

private:
  double carveMinZ() const;

  const DenseConfig config_;
  const VoxelGrid grid_;
};

} // namespace dense
} // namespace vslam

#endif
