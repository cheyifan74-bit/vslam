#ifndef VSLAM_DENSE_MAPPER_SURFACE_EXTRACTOR_H
#define VSLAM_DENSE_MAPPER_SURFACE_EXTRACTOR_H

#include "dense_mapper/submap.h"

namespace vslam
{
namespace dense
{

/// Surface points of a submap: one point per voxel at the sdf zero crossings,
/// cached per block and re-extracted only for changed blocks.
class SurfaceExtractor
{
public:
  SurfaceExtractor(double min_weight, const VoxelGrid &grid) : min_weight_(min_weight), grid_(grid)
  {
  }

  /// Re-extracts the surface_dirty blocks of a thawed submap (and the blocks
  /// below them, whose crossings read them) and rebuilds submap.points.
  void update(Submap &submap) const;

private:
  void extractBlock(const Submap &submap, const BlockKey &key, const std::vector<Voxel> &voxels,
                    SurfacePiece *piece) const;

  const double min_weight_;
  const VoxelGrid grid_;
};

} // namespace dense
} // namespace vslam

#endif
