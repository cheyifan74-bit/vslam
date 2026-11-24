/*
 * @Description: DenseVisualizer implementation.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "visualization/ros2/dense_visualizer.h"

#include <utility>

#include "visualization/ros2/msg_builders.h"

namespace vslam
{

  DenseVisualizer::DenseVisualizer(const rclcpp::Node::SharedPtr &viz_node, rclcpp::Node::SharedPtr clock_node)
      : clock_node_(std::move(clock_node))
  {
    pub_cloud_ = viz_node->create_publisher<sensor_msgs::msg::PointCloud2>(
        "dense_cloud", rclcpp::QoS(1).transient_local());
    pub_submaps_ = viz_node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "dense_submaps", rclcpp::QoS(1).transient_local());
  }

  void DenseVisualizer::onUpdate(std::shared_ptr<const DenseUpdate> update)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &submap : update->changed)
    {
      changed_ids_.insert(submap->id);
      removed_ids_.erase(submap->id);
    }
    for (const uint32_t id : update->removed)
    {
      removed_ids_.insert(id);
      changed_ids_.erase(id);
    }
    latest_ = std::move(update);
    full_dirty_ = true;
  }

  void DenseVisualizer::startTimer(const rclcpp::Node::SharedPtr &viz_node)
  {
    timer_ = viz_node->create_wall_timer(std::chrono::milliseconds(200), [this]() { tick(); });
  }

  void DenseVisualizer::tick()
  {
    std::shared_ptr<const DenseUpdate> latest;
    std::unordered_set<uint32_t> changed;
    std::unordered_set<uint32_t> removed;
    bool full_dirty = false;
    const auto now = std::chrono::steady_clock::now();
    const bool full_due = now - last_full_build_ >= std::chrono::duration<double>(full_cloud_period_s_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest = latest_;
      changed.swap(changed_ids_);
      removed.swap(removed_ids_);
      full_dirty = full_dirty_ && full_due;
      if (full_dirty)
      {
        full_dirty_ = false;
      }
    }
    if (!latest)
    {
      return;
    }

    publishSubmapMarkers(*latest, changed, removed);

    if (full_dirty)
    {
      last_cloud_msg_ = MakeDenseFullCloud(*latest, clock_node_->now());
      pub_cloud_->publish(*last_cloud_msg_);
      last_cloud_publish_ = now;
      last_full_build_ = now;
    }
    else if (last_cloud_msg_ && now - last_cloud_publish_ >= std::chrono::seconds(5))
    {
      // Unchanged map: an occasional resend lets volatile late joiners (RViz default) see it.
      last_cloud_msg_->header.stamp = clock_node_->now();
      pub_cloud_->publish(*last_cloud_msg_);
      last_cloud_publish_ = now;
    }
  }

  void DenseVisualizer::publishSubmapMarkers(const DenseUpdate &latest, const std::unordered_set<uint32_t> &changed,
                                             const std::unordered_set<uint32_t> &removed)
  {
    // Transient-local keeps only the last array, so a new subscriber gets every submap once.
    const size_t subscribers = pub_submaps_->get_subscription_count();
    const bool resend_all = subscribers > submap_subscribers_;
    submap_subscribers_ = subscribers;
    if (!resend_all && changed.empty() && removed.empty())
    {
      return;
    }

    visualization_msgs::msg::MarkerArray markers;
    const rclcpp::Time stamp = clock_node_->now();
    for (const auto &submap : latest.submaps)
    {
      if (resend_all || changed.count(submap->id) != 0)
      {
        markers.markers.push_back(MakeDenseSubmapMarker(*submap, latest.point_size_m, stamp));
      }
    }
    for (const uint32_t id : removed)
    {
      markers.markers.push_back(MakeDenseSubmapDeleteMarker(id, stamp));
    }
    if (!markers.markers.empty())
    {
      pub_submaps_->publish(markers);
    }
  }

} // namespace vslam
