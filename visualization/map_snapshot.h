/*
 * @Description: Builds a MapVisualizationSnapshot from the online COLMAP mapper.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_MAP_SNAPSHOT_H
#define VSLAM_VISUALIZATION_MAP_SNAPSHOT_H

#include <cstdint>

#include "visualization/map_viz_types.h"

namespace colmap
{
  class OnlineIncrementalMapper;
}

namespace vslam
{

  /// Copies keyframe poses, 3D points (split into local/global around
  /// current_image_id) and confirmed loops. The caller must hold the lock that
  /// guards the mapper.
  MapVisualizationSnapshot MakeMapVisualizationSnapshot(
      const colmap::OnlineIncrementalMapper &mapper,
      std::uint32_t current_image_id);

} // namespace vslam

#endif // VSLAM_VISUALIZATION_MAP_SNAPSHOT_H
