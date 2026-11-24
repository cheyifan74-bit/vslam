/*
 * @Description: MapVisualizer — all vslam-specific ROS1 visualization output.
 *               Publishes on its own callback queue / spinner thread:
 *                 map_points_global / map_points_local / map_visualization  (sparse COLMAP map)
 *                 dense_cloud / dense_submaps                               (dense TSDF map)
 *                 keyframe pose / image                                     (KeyframeSelect debug)
 *               OpenVINS' own output (odometry, TF, tracks) stays in ov_msckf::ROS1Visualizer.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_VISUALIZATION_ROS1_MAP_VISUALIZER_H
#define VSLAM_VISUALIZATION_ROS1_MAP_VISUALIZER_H

#include <memory>
#include <shared_mutex>

#include <ros/ros.h>

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
    /// Creates the viz handle and the map / dense publishers.
    /// Options come from vslam.yaml (Application::getMapVizConfig()).
    MapVisualizer(const ros::NodeHandle &main_nh, const MapVizConfig &config);
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

    /// Starts / stops the viz spinner thread.
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
    std::unique_ptr<VizNode> viz_node_;
    std::unique_ptr<DenseVisualizer> dense_;

    ros::Publisher pub_map_global_;
    ros::Publisher pub_map_local_;
    ros::Publisher pub_map_markers_;
    /// Republishes the current map periodically, so a resumed map also reaches
    /// an RViz started (or restarted) after the initial snapshot.
    ros::WallTimer map_timer_;
    int max_frustums_ = 300;

    ros::Publisher pub_kf_pose_;
    ros::Publisher pub_kf_image_;
  };

} // namespace vslam

#endif // VSLAM_VISUALIZATION_ROS1_MAP_VISUALIZER_H
