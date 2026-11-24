/*
 * @Description: NodeHandle with its own callback queue + spinner thread for visualization output (ROS1).
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_ROS1_VIZ_NODE_H
#define VSLAM_VISUALIZATION_ROS1_VIZ_NODE_H

#include <memory>

#include <ros/callback_queue.h>
#include <ros/ros.h>

namespace vslam
{

  /// Map / dense / keyframe-debug timers run on their own callback queue and
  /// spinner thread, so building multi-MB messages never delays the sensor
  /// callbacks of the main AsyncSpinner. The handle is a copy of the main one,
  /// so relative topic names resolve exactly as on the main handle.
  class VizNode
  {
  public:
    explicit VizNode(const ros::NodeHandle &main_nh);
    ~VizNode();

    VizNode(const VizNode &) = delete;
    VizNode &operator=(const VizNode &) = delete;

    ros::NodeHandle &nh() { return nh_; }

    /// Spins the queue on its own thread. No-op if already running.
    void start();
    /// Stops the spinner thread. Safe to call repeatedly.
    void stop();

  private:
    /// Declared before nh_: the handle refers to it.
    ros::CallbackQueue queue_;
    ros::NodeHandle nh_;
    std::unique_ptr<ros::AsyncSpinner> spinner_;
  };

} // namespace vslam

#endif // VSLAM_VISUALIZATION_ROS1_VIZ_NODE_H
