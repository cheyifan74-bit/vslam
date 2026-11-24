/*
 * @Description: MakeMapVisualizationSnapshot implementation.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "visualization/map_snapshot.h"

#include <unordered_set>

#include "colmap/geometry/rigid3.h"
#include "online_sfm/online_incremental_mapper.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include "utility/config_reader.h"

namespace vslam
{

  MapVisualizationSnapshot MakeMapVisualizationSnapshot(
      const colmap::OnlineIncrementalMapper &mapper,
      std::uint32_t current_image_id)
  {
    MapVisualizationSnapshot snapshot;
    snapshot.current_keyframe_id = current_image_id;
    const colmap::Reconstruction &reconstruction = mapper.GetReconstruction();
    std::unordered_set<colmap::point3D_t> local_point_ids;
    std::unordered_set<colmap::image_t> local_image_ids;
    local_image_ids.insert(current_image_id);

    if (reconstruction.ExistsImage(current_image_id) &&
        reconstruction.Image(current_image_id).HasPose())
    {
      const auto &current = reconstruction.Image(current_image_id);
      for (const auto &point2D : current.Points2D())
      {
        if (!point2D.HasPoint3D() || !reconstruction.ExistsPoint3D(point2D.point3D_id))
          continue;
        const auto &point = reconstruction.Point3D(point2D.point3D_id);
        local_point_ids.insert(point2D.point3D_id);
        for (const auto &track_element : point.track.Elements())
          local_image_ids.insert(track_element.image_id);
      }
    }

    for (const auto image_id : local_image_ids)
    {
      if (!reconstruction.ExistsImage(image_id))
        continue;
      const auto &image = reconstruction.Image(image_id);
      for (const auto &point2D : image.Points2D())
        if (point2D.HasPoint3D() && reconstruction.ExistsPoint3D(point2D.point3D_id))
          local_point_ids.insert(point2D.point3D_id);
    }

    for (const auto image_id : reconstruction.RegImageIds())
    {
      const auto &image = reconstruction.Image(image_id);
      if (!image.HasPose())
        continue;
      const colmap::Rigid3d world_from_cam = colmap::Inverse(image.CamFromWorld());
      MapVizKeyframe keyframe;
      keyframe.id = image_id;
      keyframe.position = world_from_cam.translation();
      keyframe.orientation = world_from_cam.rotation();
      snapshot.keyframes.push_back(keyframe);
    }

    for (const auto point_id : reconstruction.Point3DIds())
    {
      const auto &point = reconstruction.Point3D(point_id);
      MapVizPoint viz_point;
      viz_point.position = point.xyz;
      viz_point.id = point_id;
      viz_point.observations = static_cast<std::uint32_t>(point.track.Length());
      if (local_point_ids.count(point_id))
        snapshot.local_points.push_back(viz_point);
      else
        snapshot.global_points.push_back(viz_point);
    }

    for (const auto &loop : mapper.ConfirmedLoops())
      snapshot.confirmed_loops.push_back({loop.image_id1, loop.image_id2});
    return snapshot;
  }

  void MapVizConfig::load(ConfigReader &reader)
  {
    reader.read("dense_full_cloud_period_s", dense_full_cloud_period_s);
    reader.read("map_viz_max_frustums", max_frustums);
  }

} // namespace vslam
