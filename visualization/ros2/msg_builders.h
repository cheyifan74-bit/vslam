/*
 * @Description: Pure conversions from vslam visualization data to ROS2 messages.
 *               No publishers, no state: every function only builds a message.
 *               All outputs are in the "global" frame unless stated otherwise.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_ROS2_MSG_BUILDERS_H
#define VSLAM_VISUALIZATION_ROS2_MSG_BUILDERS_H

#include <cstdint>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "dense_mapper/dense_mapper.h"
#include "keyframe_select/keyframe.h"
#include "visualization/map_viz_types.h"

namespace vslam
{

  // ---- Sparse map (COLMAP reconstruction) ----

  /// xyz + rgb cloud, every point with the same color.
  sensor_msgs::msg::PointCloud2 MakeMapPointCloud(const std::vector<MapVizPoint> &points,
                                                  uint8_t r, uint8_t g, uint8_t b,
                                                  const builtin_interfaces::msg::Time &stamp);

  /// Keyframe positions / path / frustums, loop edges and the current keyframe
  /// (frustum + axes). Starts with a DELETEALL marker. Only the newest
  /// max_frustums keyframes get a frustum (<0: all of them).
  visualization_msgs::msg::MarkerArray MakeMapMarkers(const MapVisualizationSnapshot &snapshot,
                                                      const builtin_interfaces::msg::Time &stamp,
                                                      int max_frustums);

  // ---- Dense map (TSDF submaps) ----

  /// One POINTS marker in ns "dense_submaps", posed by world_from_submap with the
  /// points in the submap frame. An empty submap yields a DELETE marker.
  visualization_msgs::msg::Marker MakeDenseSubmapMarker(const DenseSubmapCloud &submap,
                                                        float point_size_m,
                                                        const builtin_interfaces::msg::Time &stamp);

  visualization_msgs::msg::Marker MakeDenseSubmapDeleteMarker(uint32_t id,
                                                              const builtin_interfaces::msg::Time &stamp);

  /// Every submap transformed into the world frame, as one xyz + rgb cloud.
  sensor_msgs::msg::PointCloud2::SharedPtr MakeDenseFullCloud(const DenseUpdate &update,
                                                              const builtin_interfaces::msg::Time &stamp);

  // ---- Keyframe selection debug ----

  /// Pose stamped with the keyframe's own timestamp, frame "global" if unset.
  geometry_msgs::msg::PoseStamped MakeKeyframePoseMsg(const Keyframe &kf);

} // namespace vslam

#endif // VSLAM_VISUALIZATION_ROS2_MSG_BUILDERS_H
