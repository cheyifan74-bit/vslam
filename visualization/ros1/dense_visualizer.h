/*
 * @Description: Publishes the dense (TSDF) map: per-submap markers and a merged full cloud (ROS1).
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_ROS1_DENSE_VISUALIZER_H
#define VSLAM_VISUALIZATION_ROS1_DENSE_VISUALIZER_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include "dense_mapper/dense_mapper.h"

namespace vslam
{

  /// Topics (both latched):
  ///   dense_submaps  One POINTS marker per submap, posed by world_from_submap: a loop
  ///                  closure only re-sends poses, a surface change only its own submap.
  ///   dense_cloud    Full map in "global"; rebuilt at most every full_cloud_period_s.
  ///
  /// onUpdate() may be called from the dense fusion thread and only records the update.
  /// All conversion and publishing happen in the 200 ms timer on the viz queue.
  class DenseVisualizer
  {
  public:
    /// Publishers are advertised on viz_nh.
    explicit DenseVisualizer(ros::NodeHandle &viz_nh);

    void setFullCloudPeriod(double seconds) { full_cloud_period_s_ = seconds; }

    /// Updates between two timer ticks are merged: the newest snapshot holds
    /// every submap, the id sets say which markers must be re-sent.
    void onUpdate(std::shared_ptr<const DenseUpdate> update);

    /// Creates the periodic publish timer on viz_nh (runs on its callback queue).
    void startTimer(ros::NodeHandle &viz_nh);

  private:
    void tick();
    void publishSubmapMarkers(const DenseUpdate &latest, const std::unordered_set<uint32_t> &changed,
                              const std::unordered_set<uint32_t> &removed);

    ros::Publisher pub_cloud_;
    ros::Publisher pub_submaps_;
    ros::WallTimer timer_;
    double full_cloud_period_s_ = 5.0;

    /// Written by onUpdate(), drained by tick().
    std::mutex mutex_;
    std::shared_ptr<const DenseUpdate> latest_;
    std::unordered_set<uint32_t> changed_ids_;
    std::unordered_set<uint32_t> removed_ids_;
    bool full_dirty_ = false;

    /// Only touched by tick().
    uint32_t submap_subscribers_ = 0;
    std::chrono::steady_clock::time_point last_full_build_;
  };

} // namespace vslam

#endif // VSLAM_VISUALIZATION_ROS1_DENSE_VISUALIZER_H
