/*
 * @Description: Sparse-map visualization data, independent of ROS and COLMAP.
 *               Produced by MakeMapVisualizationSnapshot(), consumed by the ROS wrappers.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_MAP_VIZ_TYPES_H
#define VSLAM_VISUALIZATION_MAP_VIZ_TYPES_H

#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace vslam
{

  class ConfigReader;

  /// ROS-side visualization options from vslam.yaml (shared by the ROS1 and ROS2 wrappers).
  struct MapVizConfig {
    /// Period of the full /dense_cloud rebuild; /dense_submaps is incremental.
    double dense_full_cloud_period_s = 5.0;
    /// Newest keyframes drawn as frustums in /map_visualization (-1 = all).
    int max_frustums = 300;

    /// Reads this struct's keys from vslam.yaml.
    void load(ConfigReader &reader);
  };

  struct MapVizPoint {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    std::uint64_t id = 0;
    std::uint32_t observations = 0;
  };

  struct MapVizKeyframe {
    std::uint32_t id = 0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  };

  struct MapVizLoop {
    std::uint32_t current_id = 0;
    std::uint32_t candidate_id = 0;
  };

  struct MapVisualizationSnapshot {
    std::uint32_t current_keyframe_id = 0;
    /// Points not seen from the current keyframe's covisible neighbourhood.
    std::vector<MapVizPoint> global_points;
    /// Points observed by the current keyframe or any keyframe sharing a point with it.
    std::vector<MapVizPoint> local_points;
    std::vector<MapVizKeyframe> keyframes;
    std::vector<MapVizLoop> confirmed_loops;
  };

} // namespace vslam

#endif // VSLAM_VISUALIZATION_MAP_VIZ_TYPES_H
