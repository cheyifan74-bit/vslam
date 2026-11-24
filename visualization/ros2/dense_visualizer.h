/*
 * @Description: Publishes the dense (TSDF) map: per-submap markers and a merged full cloud.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_ROS2_DENSE_VISUALIZER_H
#define VSLAM_VISUALIZATION_ROS2_DENSE_VISUALIZER_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "dense_mapper/dense_mapper.h"

namespace vslam
{

  /// Topics (both transient_local, depth 1):
  ///   dense_submaps  One POINTS marker per submap, posed by world_from_submap: a loop
  ///                  closure only re-sends poses, a surface change only its own submap.
  ///   dense_cloud    Full map in "global"; rebuilt at most every full_cloud_period_s.
  ///
  /// onUpdate() may be called from the dense fusion thread and only records the update.
  /// All conversion and publishing happen in the 200 ms timer on the viz executor.
  class DenseVisualizer
  {
  public:
    /// Publishers live on viz_node; message stamps come from clock_node.
    DenseVisualizer(const rclcpp::Node::SharedPtr &viz_node, rclcpp::Node::SharedPtr clock_node);

    void setFullCloudPeriod(double seconds) { full_cloud_period_s_ = seconds; }

    /// Updates between two timer ticks are merged: the newest snapshot holds
    /// every submap, the id sets say which markers must be re-sent.
    void onUpdate(std::shared_ptr<const DenseUpdate> update);

    /// Creates the periodic publish timer on viz_node.
    void startTimer(const rclcpp::Node::SharedPtr &viz_node);

  private:
    void tick();
    void publishSubmapMarkers(const DenseUpdate &latest, const std::unordered_set<uint32_t> &changed,
                              const std::unordered_set<uint32_t> &removed);

    rclcpp::Node::SharedPtr clock_node_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_submaps_;
    rclcpp::TimerBase::SharedPtr timer_;
    double full_cloud_period_s_ = 5.0;

    /// Written by onUpdate(), drained by tick().
    std::mutex mutex_;
    std::shared_ptr<const DenseUpdate> latest_;
    std::unordered_set<uint32_t> changed_ids_;
    std::unordered_set<uint32_t> removed_ids_;
    bool full_dirty_ = false;

    /// Only touched by tick().
    size_t submap_subscribers_ = 0;
    sensor_msgs::msg::PointCloud2::SharedPtr last_cloud_msg_;
    std::chrono::steady_clock::time_point last_cloud_publish_;
    std::chrono::steady_clock::time_point last_full_build_;
  };

} // namespace vslam

#endif // VSLAM_VISUALIZATION_ROS2_DENSE_VISUALIZER_H
