/*
 * @Description: ROS2 message builders for the sparse map, dense map and keyframe debug output.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "visualization/ros2/msg_builders.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>

#include <rclcpp/time.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace vslam
{

  namespace
  {
    const char *const kWorldFrame = "global";

    geometry_msgs::msg::Point toPoint(const Eigen::Vector3d &v)
    {
      geometry_msgs::msg::Point p;
      p.x = v.x();
      p.y = v.y();
      p.z = v.z();
      return p;
    }

    void addSegment(visualization_msgs::msg::Marker &m, const Eigen::Vector3d &a, const Eigen::Vector3d &b)
    {
      m.points.push_back(toPoint(a));
      m.points.push_back(toPoint(b));
    }

    /// Camera pyramid (4 rays + image rectangle) plus the optical axis.
    void addFrustum(visualization_msgs::msg::Marker &m, const MapVizKeyframe &kf, double depth)
    {
      const Eigen::Quaterniond q = kf.orientation.normalized();
      const double w = depth * 0.65;
      const double h = depth * 0.45;
      const std::array<Eigen::Vector3d, 4> corners = {Eigen::Vector3d(-w, -h, depth), Eigen::Vector3d(w, -h, depth),
                                                      Eigen::Vector3d(w, h, depth), Eigen::Vector3d(-w, h, depth)};
      std::array<Eigen::Vector3d, 4> world_corners;
      for (size_t i = 0; i < corners.size(); ++i)
      {
        world_corners[i] = kf.position + q * corners[i];
      }
      for (size_t i = 0; i < corners.size(); ++i)
      {
        addSegment(m, kf.position, world_corners[i]);
        addSegment(m, world_corners[i], world_corners[(i + 1) % corners.size()]);
      }
      addSegment(m, kf.position, kf.position + q * Eigen::Vector3d(0, 0, depth * 1.2));
    }

    /// Marker size scale: 30% of the median keyframe spacing, clamped to [0.12, 1.5] m.
    double frustumDepth(const std::vector<MapVizKeyframe> &keyframes)
    {
      std::vector<double> steps;
      for (size_t i = 1; i < keyframes.size(); ++i)
      {
        const double d = (keyframes[i].position - keyframes[i - 1].position).norm();
        if (std::isfinite(d) && d > 1e-4)
        {
          steps.push_back(d);
        }
      }
      double depth = 0.25;
      if (!steps.empty())
      {
        auto middle = steps.begin() + steps.size() / 2;
        std::nth_element(steps.begin(), middle, steps.end());
        depth = std::clamp(*middle * 0.30, 0.12, 1.5);
      }
      return depth;
    }

    uint32_t packRgb(uint8_t r, uint8_t g, uint8_t b)
    {
      return (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | b;
    }
  } // namespace

  sensor_msgs::msg::PointCloud2 MakeMapPointCloud(const std::vector<MapVizPoint> &points,
                                                  uint8_t r, uint8_t g, uint8_t b,
                                                  const builtin_interfaces::msg::Time &stamp)
  {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = kWorldFrame;
    cloud.header.stamp = stamp;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(points.size());
    cloud.is_dense = true;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<uint32_t> rgb(cloud, "rgb");
    const uint32_t color = packRgb(r, g, b);
    for (const auto &point : points)
    {
      *x = static_cast<float>(point.position.x());
      *y = static_cast<float>(point.position.y());
      *z = static_cast<float>(point.position.z());
      *rgb = color;
      ++x;
      ++y;
      ++z;
      ++rgb;
    }
    return cloud;
  }

  visualization_msgs::msg::MarkerArray MakeMapMarkers(const MapVisualizationSnapshot &snapshot,
                                                      const builtin_interfaces::msg::Time &stamp,
                                                      int max_frustums)
  {
    using visualization_msgs::msg::Marker;

    visualization_msgs::msg::MarkerArray array;
    Marker clear;
    clear.header.frame_id = kWorldFrame;
    clear.header.stamp = stamp;
    clear.ns = "vslam_map";
    clear.id = 0;
    clear.action = Marker::DELETEALL;
    array.markers.push_back(clear);

    const double depth = frustumDepth(snapshot.keyframes);

    // All keyframes: positions (spheres), trajectory, frustums; plus loop edges.
    Marker positions;
    positions.header = clear.header;
    positions.ns = "keyframes";
    positions.id = 1;
    positions.type = Marker::SPHERE_LIST;
    positions.action = Marker::ADD;
    positions.scale.x = positions.scale.y = positions.scale.z = depth * 0.10;
    positions.color.r = 0.1;
    positions.color.g = 0.8;
    positions.color.b = 1.0;
    positions.color.a = 1.0;

    Marker path = positions;
    path.ns = "keyframe_path";
    path.id = 2;
    path.type = Marker::LINE_STRIP;
    path.scale.x = depth * 0.035;

    Marker frustums = positions;
    frustums.ns = "keyframe_frustums";
    frustums.id = 3;
    frustums.type = Marker::LINE_LIST;
    frustums.scale.x = depth * 0.025;

    Marker loops = frustums;
    loops.ns = "loop_edges";
    loops.id = 4;
    loops.scale.x = depth * 0.04;
    loops.color.r = 1.0;
    loops.color.g = 0.1;
    loops.color.b = 1.0;

    // Frustums are 18 points each; only the newest ones are drawn so the array stays small on big maps.
    const size_t first_frustum =
        max_frustums >= 0 && snapshot.keyframes.size() > static_cast<size_t>(max_frustums)
            ? snapshot.keyframes.size() - static_cast<size_t>(max_frustums)
            : 0;
    std::unordered_map<uint32_t, Eigen::Vector3d> positions_by_id;
    for (size_t i = 0; i < snapshot.keyframes.size(); ++i)
    {
      const auto &kf = snapshot.keyframes[i];
      positions.points.push_back(toPoint(kf.position));
      path.points.push_back(toPoint(kf.position));
      positions_by_id[kf.id] = kf.position;
      if (i >= first_frustum)
      {
        addFrustum(frustums, kf, depth);
      }
    }
    for (const auto &loop : snapshot.confirmed_loops)
    {
      if (positions_by_id.count(loop.current_id) && positions_by_id.count(loop.candidate_id))
      {
        addSegment(loops, positions_by_id.at(loop.current_id), positions_by_id.at(loop.candidate_id));
      }
    }

    // Current keyframe: larger orange sphere + orange frustum + RGB axes.
    Marker current = positions;
    current.ns = "current_keyframe";
    current.id = 5;
    current.scale.x = current.scale.y = current.scale.z = depth * 0.24;
    current.color.r = 1.0;
    current.color.g = 0.65;
    current.color.b = 0.0;
    current.points.clear();

    Marker current_frustum = frustums;
    current_frustum.ns = "current_frustum";
    current_frustum.id = 6;
    current_frustum.points.clear();
    current_frustum.color.r = 1.0;
    current_frustum.color.g = 0.65;
    current_frustum.color.b = 0.0;

    Marker axes;
    axes.header = clear.header;
    axes.ns = "current_axes";
    axes.action = Marker::ADD;
    axes.type = Marker::LINE_LIST;
    axes.scale.x = depth * 0.045;

    auto current_it = std::find_if(snapshot.keyframes.begin(), snapshot.keyframes.end(),
                                   [&](const MapVizKeyframe &kf) { return kf.id == snapshot.current_keyframe_id; });
    if (current_it != snapshot.keyframes.end())
    {
      const auto &kf = *current_it;
      current.points.push_back(toPoint(kf.position));
      addFrustum(current_frustum, kf, depth);
      const Eigen::Quaterniond q = kf.orientation.normalized();
      const double len = depth * 1.8;
      Marker xm = axes, ym = axes, zm = axes;
      xm.id = 0;
      ym.id = 1;
      zm.id = 2;
      xm.color.r = 1;
      xm.color.a = 1;
      ym.color.g = 1;
      ym.color.a = 1;
      zm.color.b = 1;
      zm.color.a = 1;
      addSegment(xm, kf.position, kf.position + q * Eigen::Vector3d::UnitX() * len);
      addSegment(ym, kf.position, kf.position + q * Eigen::Vector3d::UnitY() * len);
      addSegment(zm, kf.position, kf.position + q * Eigen::Vector3d::UnitZ() * len);
      array.markers.push_back(xm);
      array.markers.push_back(ym);
      array.markers.push_back(zm);
    }
    array.markers.push_back(positions);
    array.markers.push_back(path);
    array.markers.push_back(frustums);
    array.markers.push_back(current_frustum);
    array.markers.push_back(loops);
    array.markers.push_back(current);
    return array;
  }

  visualization_msgs::msg::Marker MakeDenseSubmapMarker(const DenseSubmapCloud &submap,
                                                        float point_size_m,
                                                        const builtin_interfaces::msg::Time &stamp)
  {
    using visualization_msgs::msg::Marker;
    Marker marker;
    marker.header.frame_id = kWorldFrame;
    marker.header.stamp = stamp;
    marker.ns = "dense_submaps";
    marker.id = static_cast<int>(submap.id);
    const DenseCloud &cloud = *submap.points;
    if (cloud.xyz.empty())
    {
      marker.action = Marker::DELETE;
      return marker;
    }
    marker.type = Marker::POINTS;
    marker.action = Marker::ADD;
    marker.pose.orientation.x = submap.q_world_submap[0];
    marker.pose.orientation.y = submap.q_world_submap[1];
    marker.pose.orientation.z = submap.q_world_submap[2];
    marker.pose.orientation.w = submap.q_world_submap[3];
    marker.pose.position.x = submap.t_world_submap[0];
    marker.pose.position.y = submap.t_world_submap[1];
    marker.pose.position.z = submap.t_world_submap[2];
    marker.scale.x = point_size_m;
    marker.scale.y = point_size_m;
    marker.color.a = 1.f;
    marker.points.resize(cloud.xyz.size());
    marker.colors.resize(cloud.xyz.size());
    for (size_t i = 0; i < cloud.xyz.size(); ++i)
    {
      marker.points[i].x = cloud.xyz[i].x();
      marker.points[i].y = cloud.xyz[i].y();
      marker.points[i].z = cloud.xyz[i].z();
      const size_t c = i * 3;
      const bool has_rgb = c + 2 < cloud.rgb.size();
      marker.colors[i].r = (has_rgb ? cloud.rgb[c] : 180) / 255.f;
      marker.colors[i].g = (has_rgb ? cloud.rgb[c + 1] : 180) / 255.f;
      marker.colors[i].b = (has_rgb ? cloud.rgb[c + 2] : 180) / 255.f;
      marker.colors[i].a = 1.f;
    }
    return marker;
  }

  visualization_msgs::msg::Marker MakeDenseSubmapDeleteMarker(uint32_t id,
                                                              const builtin_interfaces::msg::Time &stamp)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = kWorldFrame;
    marker.header.stamp = stamp;
    marker.ns = "dense_submaps";
    marker.id = static_cast<int>(id);
    marker.action = visualization_msgs::msg::Marker::DELETE;
    return marker;
  }

  sensor_msgs::msg::PointCloud2::SharedPtr MakeDenseFullCloud(const DenseUpdate &update,
                                                              const builtin_interfaces::msg::Time &stamp)
  {
    size_t count = 0;
    for (const auto &submap : update.submaps)
    {
      count += submap->points->xyz.size();
    }
    auto msg_ptr = std::make_shared<sensor_msgs::msg::PointCloud2>();
    sensor_msgs::msg::PointCloud2 &msg = *msg_ptr;
    msg.header.frame_id = kWorldFrame;
    msg.header.stamp = stamp;
    msg.height = 1;
    msg.width = static_cast<uint32_t>(count);
    msg.is_dense = true;
    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(count);
    sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    sensor_msgs::PointCloud2Iterator<uint32_t> rgb(msg, "rgb");
    for (const auto &submap : update.submaps)
    {
      const Eigen::Quaternionf q(submap->q_world_submap[3], submap->q_world_submap[0],
                                 submap->q_world_submap[1], submap->q_world_submap[2]);
      const Eigen::Matrix3f R = q.normalized().toRotationMatrix();
      const Eigen::Vector3f t(submap->t_world_submap[0], submap->t_world_submap[1],
                              submap->t_world_submap[2]);
      const DenseCloud &cloud = *submap->points;
      for (size_t i = 0; i < cloud.xyz.size(); ++i)
      {
        const Eigen::Vector3f p = R * cloud.xyz[i] + t;
        *x = p.x();
        *y = p.y();
        *z = p.z();
        const size_t c = i * 3;
        const uint8_t r = c + 2 < cloud.rgb.size() ? cloud.rgb[c] : 180;
        const uint8_t g = c + 2 < cloud.rgb.size() ? cloud.rgb[c + 1] : 180;
        const uint8_t b = c + 2 < cloud.rgb.size() ? cloud.rgb[c + 2] : 180;
        *rgb = packRgb(r, g, b);
        ++x;
        ++y;
        ++z;
        ++rgb;
      }
    }
    return msg_ptr;
  }

  geometry_msgs::msg::PoseStamped MakeKeyframePoseMsg(const Keyframe &kf)
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(kf.timestamp * 1e9), RCL_ROS_TIME);
    pose_msg.header.frame_id = kf.frame_id.empty() ? kWorldFrame : kf.frame_id;
    pose_msg.pose.position.x = kf.pose.position.x();
    pose_msg.pose.position.y = kf.pose.position.y();
    pose_msg.pose.position.z = kf.pose.position.z();
    pose_msg.pose.orientation.x = kf.pose.orientation.x();
    pose_msg.pose.orientation.y = kf.pose.orientation.y();
    pose_msg.pose.orientation.z = kf.pose.orientation.z();
    pose_msg.pose.orientation.w = kf.pose.orientation.w();
    return pose_msg;
  }

} // namespace vslam
