/*
 * @Description: ROS2 Wrapper — thin ROS2 glue layer (rclcpp counterpart of ros_wrapper.h).
 *               Sensor subscription + message conversion + KeyframeSelect ROS I/O.
 *               Map / dense / keyframe-debug output: visualization/ros2/map_visualizer.h.
 * @Author: che yifan
 * @Date: 2026-10-22
 * @LastEditTime: 2026-10-22
 * @LastEditors: che yifan
 * @Reference:
 */
#ifndef VSLAM_ROS2_WRAPPER_H
#define VSLAM_ROS2_WRAPPER_H

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <opencv2/core.hpp>

#include "keyframe_select/keyframe.h"
#include "map_manager/map_manager.h"
#include "visualization/ros2/map_visualizer.h"

namespace ov_core
{
  struct ImuData;
  struct CameraData;
  class YamlParser;
} // namespace ov_core

namespace ov_msckf
{
  class VioManager;
  class ROS2Visualizer;
} // namespace ov_msckf

namespace vslam
{

  class Application;
  class KeyframeQueue;

  class RosWrapper
  {
  public:
    RosWrapper(std::shared_ptr<rclcpp::Node> node, const std::string &config_path);
    ~RosWrapper();

    void run();

    std::shared_ptr<Application> getApplication() const;
    std::shared_ptr<ov_core::YamlParser> getParser() const;
    std::shared_ptr<rclcpp::Node> getNode() const;
    std::shared_ptr<KeyframeQueue> getKeyframeQueue() const;

  private:
    using ImageMsg = sensor_msgs::msg::Image;
    using ImuMsg = sensor_msgs::msg::Imu;

    void init(const std::string &config_path);
    void setupSubscribers();
    /// Feeds VIO poses to KeyframeSelect in-process and sets up its debug publishers.
    void setupKeyframeSelect();

    void callbackInertial(const ImuMsg::ConstSharedPtr msg);
    void callbackMonocular(const ImageMsg::ConstSharedPtr msg, int cam_id);
    void callbackStereo(const ImageMsg::ConstSharedPtr msg0, const ImageMsg::ConstSharedPtr msg1, int cam_id0, int cam_id1);
    void callbackColor(const ImageMsg::ConstSharedPtr msg);

    void feedKeyframeImage(double timestamp, const cv::Mat &image, int cam_id,
                           const cv::Mat &image_right = cv::Mat());

    std::shared_ptr<Application> app_;
    std::shared_ptr<ov_msckf::ROS2Visualizer> viz_;
    std::shared_ptr<rclcpp::Node> node_;
    std::shared_ptr<ov_core::YamlParser> parser_;

    /// Separate groups so IMU / camera callbacks may run in parallel (like ros::AsyncSpinner),
    /// while callbacks of the same stream stay serialized.
    rclcpp::CallbackGroup::SharedPtr cb_group_imu_;
    rclcpp::CallbackGroup::SharedPtr cb_group_cam_;
    rclcpp::CallbackGroup::SharedPtr cb_group_color_;

    rclcpp::Subscription<ImuMsg>::SharedPtr sub_imu_;
    rclcpp::Subscription<ImageMsg>::SharedPtr sub_color_;

    std::vector<rclcpp::Subscription<ImageMsg>::SharedPtr> camera_subs_;

    typedef message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg> SyncPolicy;
    // Subscribers must outlive the Synchronizer (its dtor disconnects from them): keep this order.
    std::vector<std::shared_ptr<message_filters::Subscriber<ImageMsg>>> sync_subs_cam_;
    std::vector<std::shared_ptr<message_filters::Synchronizer<SyncPolicy>>> sync_cam_;

    /// vslam map / dense / keyframe-debug output on its own DDS participant and thread.
    std::unique_ptr<MapVisualizer> map_viz_;
  };

} // namespace vslam

#endif // VSLAM_ROS2_WRAPPER_H
