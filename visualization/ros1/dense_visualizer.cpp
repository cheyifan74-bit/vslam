/*
 * @Description: DenseVisualizer implementation (ROS1).
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "visualization/ros1/dense_visualizer.h"

#include <utility>

#include "visualization/ros1/msg_builders.h"

namespace vslam
{

  DenseVisualizer::DenseVisualizer(ros::NodeHandle &viz_nh)
  {
    // Latched: late subscribers (RViz) get the last cloud without a periodic resend.
    pub_cloud_ = viz_nh.advertise<sensor_msgs::PointCloud2>("dense_cloud", 1, true);
    pub_submaps_ = viz_nh.advertise<visualization_msgs::MarkerArray>("dense_submaps", 1, true);
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

  void DenseVisualizer::startTimer(ros::NodeHandle &viz_nh)
  {
    timer_ = viz_nh.createWallTimer(ros::WallDuration(0.2), [this](const ros::WallTimerEvent &) { tick(); });
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
      pub_cloud_.publish(MakeDenseFullCloud(*latest, ros::Time::now()));
      last_full_build_ = now;
    }
  }

  void DenseVisualizer::publishSubmapMarkers(const DenseUpdate &latest, const std::unordered_set<uint32_t> &changed,
                                             const std::unordered_set<uint32_t> &removed)
  {
    // Latching keeps only the last array, so a new subscriber gets every submap once.
    const uint32_t subscribers = pub_submaps_.getNumSubscribers();
    const bool resend_all = subscribers > submap_subscribers_;
    submap_subscribers_ = subscribers;
    if (!resend_all && changed.empty() && removed.empty())
    {
      return;
    }

    visualization_msgs::MarkerArray markers;
    const ros::Time stamp = ros::Time::now();
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
      pub_submaps_.publish(markers);
    }
  }

} // namespace vslam
