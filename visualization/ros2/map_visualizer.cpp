/*
 * @Description: MapVisualizer implementation.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "visualization/ros2/map_visualizer.h"

#include <chrono>
#include <mutex>
#include <string>
#include <utility>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <std_msgs/msg/header.hpp>

#include "keyframe_select/keyframe_select.h"
#include "map_manager/map_manager.h"
#include "visualization/ros2/dense_visualizer.h"
#include "visualization/ros2/msg_builders.h"
#include "visualization/ros2/viz_node.h"

#include "utils/print.h"

namespace vslam
{

  MapVisualizer::MapVisualizer(const rclcpp::Node::SharedPtr &main_node, const MapVizConfig &config)
      : main_node_(main_node), max_frustums_(config.max_frustums)
  {
    viz_node_ = std::make_unique<VizNode>(*main_node_);
    const rclcpp::Node::SharedPtr &node = viz_node_->node();
    pub_map_global_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("map_points_global", 1);
    pub_map_local_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("map_points_local", 1);
    pub_map_markers_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("map_visualization", 1);
    dense_ = std::make_unique<DenseVisualizer>(node, main_node_);
    dense_->setFullCloudPeriod(config.dense_full_cloud_period_s);
  }

  MapVisualizer::~MapVisualizer()
  {
    detach();
    stop();
  }

  void MapVisualizer::detach()
  {
    std::unique_lock<std::shared_mutex> lock(guard_->mutex);
    guard_->alive = false;
  }

  void MapVisualizer::attachMapManager(const std::shared_ptr<MapManager> &map_manager)
  {
    if (!map_manager)
    {
      return;
    }
    // Publishes synchronously, so the publishers above must already exist.
    std::shared_ptr<Guard> guard = guard_;
    map_manager->setVisualizationCallback([this, guard](const MapVisualizationSnapshot &snapshot) {
      std::shared_lock<std::shared_mutex> lock(guard->mutex);
      if (guard->alive)
      {
        publishMap(snapshot);
      }
    });
    map_manager->setDenseCloudCallback([this, guard](std::shared_ptr<const DenseUpdate> update) {
      std::shared_lock<std::shared_mutex> lock(guard->mutex);
      if (guard->alive)
      {
        dense_->onUpdate(std::move(update));
      }
    });

    // See map_timer_: RViz joins late and its default QoS is volatile, so the
    // initial snapshot of a resumed map would otherwise be dropped.
    map_timer_ = viz_node_->node()->create_wall_timer(std::chrono::seconds(1), [map_manager]() {
      map_manager->publishVisualization();
    });
    dense_->startTimer(viz_node_->node());
  }

  void MapVisualizer::attachKeyframeSelect(const KeyframeSelectConfig &config,
                                           const std::shared_ptr<KeyframeSelect> &keyframe_select)
  {
    if (!config.publish_debug)
    {
      return;
    }
    // Relative names resolve under the node namespace (/ov_msckf/... with the default launch).
    const rclcpp::Node::SharedPtr &node = viz_node_->node();
    pub_kf_pose_ = node->create_publisher<geometry_msgs::msg::PoseStamped>(config.topic_keyframe_pose, 10);
    pub_kf_image_ = node->create_publisher<sensor_msgs::msg::Image>(config.topic_keyframe_image, 2);
    PRINT_INFO("[ROS_WRAPPER]: KeyframeSelect debug pub: %s, %s\n", pub_kf_pose_->get_topic_name(),
               pub_kf_image_->get_topic_name());

    // Debug publish runs on KeyframeSelect worker thread, not in sensor callbacks.
    if (keyframe_select)
    {
      std::shared_ptr<Guard> guard = guard_;
      keyframe_select->setKeyframeSelectedCallback([this, guard](const Keyframe &kf) {
        std::shared_lock<std::shared_mutex> lock(guard->mutex);
        if (guard->alive)
        {
          publishKeyframeDebug(kf);
        }
      });
    }
  }

  void MapVisualizer::start() { viz_node_->start(); }

  void MapVisualizer::stop() { viz_node_->stop(); }

  void MapVisualizer::publishMap(const MapVisualizationSnapshot &snapshot)
  {
    const rclcpp::Time stamp = main_node_->now();
    pub_map_global_->publish(MakeMapPointCloud(snapshot.global_points, 255, 255, 255, stamp));
    pub_map_local_->publish(MakeMapPointCloud(snapshot.local_points, 255, 0, 0, stamp));
    pub_map_markers_->publish(MakeMapMarkers(snapshot, stamp, max_frustums_));
  }

  void MapVisualizer::publishKeyframeDebug(const Keyframe &kf)
  {
    const geometry_msgs::msg::PoseStamped pose_msg = MakeKeyframePoseMsg(kf);
    pub_kf_pose_->publish(pose_msg);

    if (!kf.image.empty() && pub_kf_image_->get_subscription_count() > 0)
    {
      std_msgs::msg::Header header = pose_msg.header;
      header.frame_id = "cam0";
      const std::string encoding = (kf.image.channels() == 1) ? "mono8" : "bgr8";
      pub_kf_image_->publish(*cv_bridge::CvImage(header, encoding, kf.image).toImageMsg());
    }
  }

} // namespace vslam
