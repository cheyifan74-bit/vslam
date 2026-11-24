/*
 * @Description: Separate rclcpp context + node + executor thread for heavy visualization output.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_ROS2_VIZ_NODE_H
#define VSLAM_VISUALIZATION_ROS2_VIZ_NODE_H

#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>

namespace vslam
{

  /// Map / dense / keyframe-debug output lives in its own context (own DDS
  /// participant) and executor thread. Its multi-MB messages then cannot hold
  /// the participant send lock that the VIO odometry/TF publishers need.
  /// The node is named "<main>_map_viz" in the main node's namespace, so
  /// relative topic names resolve exactly as on the main node.
  class VizNode
  {
  public:
    explicit VizNode(const rclcpp::Node &main_node);
    ~VizNode();

    VizNode(const VizNode &) = delete;
    VizNode &operator=(const VizNode &) = delete;

    const rclcpp::Node::SharedPtr &node() const { return node_; }

    /// Spins the executor on its own thread. No-op if already running.
    void start();
    /// Shuts the context down and joins the thread. Safe to call repeatedly.
    void stop();

  private:
    rclcpp::Context::SharedPtr context_;
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
    std::thread thread_;
  };

} // namespace vslam

#endif // VSLAM_VISUALIZATION_ROS2_VIZ_NODE_H
