/*
 * @Description: MapVisualizer — all vslam-specific ROS2 visualization output.
 *               Owns the separate viz node and publishes:
 *                 map_points_global / map_points_local / map_visualization  (sparse COLMAP map)
 *                 dense_cloud / dense_submaps                               (dense TSDF map)
 *                 keyframe pose / image                                     (KeyframeSelect debug)
 *               OpenVINS' own output (odometry, TF, tracks) stays in ov_msckf::ROS2Visualizer.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_ROS2_MAP_VISUALIZER_H
#define VSLAM_VISUALIZATION_ROS2_MAP_VISUALIZER_H

#include <memory>
#include <shared_mutex>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "keyframe_select/keyframe.h"
#include "visualization/map_viz_types.h"

namespace vslam
{

  class DenseVisualizer;
  class KeyframeSelect;
  class MapManager;
  class VizNode;

  class MapVisualizer
  {
  public:
    /// Creates the viz node and the map / dense publishers. Reads
    /// Options come from vslam.yaml (Application::getMapVizConfig()).
    /// Message stamps use main_node's clock.
    MapVisualizer(const rclcpp::Node::SharedPtr &main_node, const MapVizConfig &config);
    ~MapVisualizer();

    MapVisualizer(const MapVisualizer &) = delete;
    MapVisualizer &operator=(const MapVisualizer &) = delete;

    /// Registers the sparse/dense callbacks on map_manager (this publishes the
    /// currently loaded map right away) and starts the periodic timers.
    void attachMapManager(const std::shared_ptr<MapManager> &map_manager);

    /// Creates the keyframe debug publishers and hooks them to keyframe_select.
    /// No-op unless config.publish_debug.
    void attachKeyframeSelect(const KeyframeSelectConfig &config,
                              const std::shared_ptr<KeyframeSelect> &keyframe_select);

    /// Starts / stops the viz executor thread.
    void start();
    void stop();

  private:
    /// Callbacks handed to MapManager / DenseMapper / KeyframeSelect can still
    /// fire after this object is destroyed (RosWrapper destroys map_viz_ before
    /// app_, and those modules call the callbacks outside their own locks).
    /// Every callback holds a shared lock while running and bails out once
    /// alive is false; the destructor flips it under the unique lock, which
    /// also waits for in-flight callbacks to finish.
    struct Guard
    {
      std::shared_mutex mutex;
      bool alive = true;
    };

    void detach();
    void publishMap(const MapVisualizationSnapshot &snapshot);
    void publishKeyframeDebug(const Keyframe &kf);

    std::shared_ptr<Guard> guard_ = std::make_shared<Guard>();
    rclcpp::Node::SharedPtr main_node_;
    std::unique_ptr<VizNode> viz_node_;
    std::unique_ptr<DenseVisualizer> dense_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_global_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_local_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_map_markers_;
    /// RViz subscribes with default (volatile) QoS, so a snapshot published before
    /// it connects is never seen. Republish the current map periodically so a
    /// resumed map appears as soon as RViz starts (and survives an RViz restart).
    rclcpp::TimerBase::SharedPtr map_timer_;
    int max_frustums_ = 300;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_kf_pose_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_kf_image_;
  };

} // namespace vslam

#endif // VSLAM_VISUALIZATION_ROS2_MAP_VISUALIZER_H
