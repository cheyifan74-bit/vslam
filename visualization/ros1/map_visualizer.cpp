/*
 * @Description: MapVisualizer implementation (ROS1).
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "visualization/ros1/map_visualizer.h"

#include <mutex>
#include <string>
#include <utility>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/MarkerArray.h>

#include "keyframe_select/keyframe_select.h"
#include "map_manager/map_manager.h"
#include "visualization/ros1/dense_visualizer.h"
#include "visualization/ros1/msg_builders.h"
#include "visualization/ros1/viz_node.h"

#include "utils/print.h"

namespace vslam
{

  MapVisualizer::MapVisualizer(const ros::NodeHandle &main_nh, const MapVizConfig &config)
      : max_frustums_(config.max_frustums)
  {
    viz_node_ = std::make_unique<VizNode>(main_nh);
    ros::NodeHandle &nh = viz_node_->nh();
    pub_map_global_ = nh.advertise<sensor_msgs::PointCloud2>("map_points_global", 1);
    pub_map_local_ = nh.advertise<sensor_msgs::PointCloud2>("map_points_local", 1);
    pub_map_markers_ = nh.advertise<visualization_msgs::MarkerArray>("map_visualization", 1);
    dense_ = std::make_unique<DenseVisualizer>(nh);
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

    // See map_timer_: the initial snapshot of a resumed map is gone once RViz connects.
    map_timer_ = viz_node_->nh().createWallTimer(
        ros::WallDuration(1.0), [map_manager](const ros::WallTimerEvent &) { map_manager->publishVisualization(); });
    dense_->startTimer(viz_node_->nh());
  }

  void MapVisualizer::attachKeyframeSelect(const KeyframeSelectConfig &config,
                                           const std::shared_ptr<KeyframeSelect> &keyframe_select)
  {
    if (!config.publish_debug)
    {
      return;
    }
    ros::NodeHandle &nh = viz_node_->nh();
    pub_kf_pose_ = nh.advertise<geometry_msgs::PoseStamped>(config.topic_keyframe_pose, 10);
    pub_kf_image_ = nh.advertise<sensor_msgs::Image>(config.topic_keyframe_image, 2);
    PRINT_INFO("[ROS_WRAPPER]: KeyframeSelect debug pub: %s, %s\n", pub_kf_pose_.getTopic().c_str(),
               pub_kf_image_.getTopic().c_str());

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
    const ros::Time stamp = ros::Time::now();
    pub_map_global_.publish(MakeMapPointCloud(snapshot.global_points, 255, 255, 255, stamp));
    pub_map_local_.publish(MakeMapPointCloud(snapshot.local_points, 255, 0, 0, stamp));
    pub_map_markers_.publish(MakeMapMarkers(snapshot, stamp, max_frustums_));
  }

  void MapVisualizer::publishKeyframeDebug(const Keyframe &kf)
  {
    const geometry_msgs::PoseStamped pose_msg = MakeKeyframePoseMsg(kf);
    pub_kf_pose_.publish(pose_msg);

    if (!kf.image.empty() && pub_kf_image_.getNumSubscribers() > 0)
    {
      std_msgs::Header header = pose_msg.header;
      header.frame_id = "cam0";
      const std::string encoding = (kf.image.channels() == 1) ? "mono8" : "bgr8";
      pub_kf_image_.publish(cv_bridge::CvImage(header, encoding, kf.image).toImageMsg());
    }
  }

} // namespace vslam
