/*
 * @Description: ROS2 Wrapper — thin ROS2 glue layer (rclcpp counterpart of ros_wrapper.h).
 *               Sensor subscription + message conversion + KeyframeSelect ROS I/O.
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

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <opencv2/core.hpp>

#include "keyframe_select/keyframe.h"
#include "map_manager/map_manager.h"

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
    using PoseMsg = geometry_msgs::msg::PoseWithCovarianceStamped;
    using ImageMsg = sensor_msgs::msg::Image;
    using ImuMsg = sensor_msgs::msg::Imu;

    void init(const std::string &config_path);
    void setupSubscribers();
    /// Subscribe pose topic and optional debug publishers for KeyframeSelect.
    void setupKeyframeSelectTopics();

    void callbackInertial(const ImuMsg::ConstSharedPtr msg);
    void callbackMonocular(const ImageMsg::ConstSharedPtr msg, int cam_id);
    void callbackStereo(const ImageMsg::ConstSharedPtr msg0, const ImageMsg::ConstSharedPtr msg1, int cam_id0, int cam_id1);
    void callbackPose(const PoseMsg::ConstSharedPtr pose_msg);

    void feedKeyframeImage(double timestamp, const cv::Mat &image, int cam_id);
    void publishKeyframeDebug(const Keyframe &kf);
    void publishMapVisualization(const MapVisualizationSnapshot &snapshot);
    static Pose3d poseFromMsg(const PoseMsg &msg);
    /// Convert poseimu (IMU in global) to left-camera pose in global using OpenVINS extrinsics.
    bool convertImuPoseMsgToCameraPose(const PoseMsg &pose_msg, Pose3d &camera_pose) const;

    std::shared_ptr<Application> app_;
    std::shared_ptr<ov_msckf::ROS2Visualizer> viz_;
    std::shared_ptr<rclcpp::Node> node_;
    std::shared_ptr<ov_core::YamlParser> parser_;

    /// Separate groups so IMU / camera / pose callbacks may run in parallel (like ros::AsyncSpinner),
    /// while callbacks of the same stream stay serialized.
    rclcpp::CallbackGroup::SharedPtr cb_group_imu_;
    rclcpp::CallbackGroup::SharedPtr cb_group_cam_;
    rclcpp::CallbackGroup::SharedPtr cb_group_pose_;

    rclcpp::Subscription<ImuMsg>::SharedPtr sub_imu_;
    rclcpp::Subscription<PoseMsg>::SharedPtr sub_pose_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_kf_pose_;
    rclcpp::Publisher<ImageMsg>::SharedPtr pub_kf_image_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_global_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_local_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_map_markers_;
    bool kf_publish_debug_ = true;

    std::vector<rclcpp::Subscription<ImageMsg>::SharedPtr> camera_subs_;

    typedef message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg> SyncPolicy;
    // Subscribers must outlive the Synchronizer (its dtor disconnects from them): keep this order.
    std::vector<std::shared_ptr<message_filters::Subscriber<ImageMsg>>> sync_subs_cam_;
    std::vector<std::shared_ptr<message_filters::Synchronizer<SyncPolicy>>> sync_cam_;
  };

} // namespace vslam

#endif // VSLAM_ROS2_WRAPPER_H
