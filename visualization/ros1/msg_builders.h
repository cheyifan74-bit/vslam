/*
 * @Description: Pure conversions from vslam visualization data to ROS1 messages.
 *               No publishers, no state: every function only builds a message.
 *               All outputs are in the "global" frame unless stated otherwise.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_ROS1_MSG_BUILDERS_H
#define VSLAM_VISUALIZATION_ROS1_MSG_BUILDERS_H

#include <cstdint>
#include <vector>

#include <ros/time.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "dense_mapper/dense_mapper.h"
#include "keyframe_select/keyframe.h"
#include "visualization/map_viz_types.h"

namespace vslam
{

  // ---- Sparse map (COLMAP reconstruction) ----

  /// xyz + rgb cloud, every point with the same color.
  sensor_msgs::PointCloud2 MakeMapPointCloud(const std::vector<MapVizPoint> &points,
                                             uint8_t r, uint8_t g, uint8_t b,
                                             const ros::Time &stamp);

  /// Keyframe positions / path / frustums, loop edges and the current keyframe
  /// (frustum + axes). Starts with a DELETEALL marker. Only the newest
  /// max_frustums keyframes get a frustum (<0: all of them).
  visualization_msgs::MarkerArray MakeMapMarkers(const MapVisualizationSnapshot &snapshot,
                                                 const ros::Time &stamp,
                                                 int max_frustums);

  // ---- Dense map (TSDF submaps) ----

  /// One POINTS marker in ns "dense_submaps", posed by world_from_submap with the
  /// points in the submap frame. An empty submap yields a DELETE marker.
  visualization_msgs::Marker MakeDenseSubmapMarker(const DenseSubmapCloud &submap,
                                                   float point_size_m,
                                                   const ros::Time &stamp);

  visualization_msgs::Marker MakeDenseSubmapDeleteMarker(uint32_t id,
                                                         const ros::Time &stamp);

  /// Every submap transformed into the world frame, as one xyz + rgb cloud.
  sensor_msgs::PointCloud2Ptr MakeDenseFullCloud(const DenseUpdate &update,
                                                 const ros::Time &stamp);

  // ---- Keyframe selection debug ----

  /// Pose stamped with the keyframe's own timestamp, frame "global" if unset.
  geometry_msgs::PoseStamped MakeKeyframePoseMsg(const Keyframe &kf);

} // namespace vslam

#endif // VSLAM_VISUALIZATION_ROS1_MSG_BUILDERS_H
