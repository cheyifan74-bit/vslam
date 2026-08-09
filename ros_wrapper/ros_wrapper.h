/*
 * @Description: ROS Wrapper — thin ROS glue layer.
 *               Sensor subscription + message conversion + KeyframeSelect ROS I/O.
 * @Author: che yifan
 * @Date: 2026-07-26 09:39:40
 * @LastEditTime: 2026-08-02
 * @LastEditors: che yifan
 * @Reference:
 */
#ifndef VSLAM_ROS_WRAPPER_H
#define VSLAM_ROS_WRAPPER_H

#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/core.hpp>

#include "keyframe_select/keyframe.h"

namespace ov_core
{
  struct ImuData;
  struct CameraData;
  class YamlParser;
} // namespace ov_core

namespace ov_msckf
{
  class VioManager;
  class ROS1Visualizer;
} // namespace ov_msckf

namespace vslam
{

  class Application;
  class KeyframeQueue;

  class RosWrapper
  {
  public:
    RosWrapper(std::shared_ptr<ros::NodeHandle> nh, const std::string &config_path);
    ~RosWrapper();

    void run();

    std::shared_ptr<Application> getApplication() const;
    std::shared_ptr<ov_core::YamlParser> getParser() const;
    std::shared_ptr<ros::NodeHandle> getNodeHandle() const;
    std::shared_ptr<KeyframeQueue> getKeyframeQueue() const;

  private:
    using PoseMsg = geometry_msgs::PoseWithCovarianceStamped;

    void init(std::shared_ptr<ros::NodeHandle> nh, const std::string &config_path);
    void setupSubscribers();
    /// Subscribe pose topic and optional debug publishers for KeyframeSelect.
    void setupKeyframeSelectTopics();

    void callbackInertial(const sensor_msgs::Imu::ConstPtr &msg);
    void callbackMonocular(const sensor_msgs::ImageConstPtr &msg, int cam_id);
    void callbackStereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1,
                        int cam_id0, int cam_id1);
    void callbackPose(const PoseMsg::ConstPtr &pose_msg);

    void feedKeyframeImage(double timestamp, const cv::Mat &image, int cam_id);
    void publishKeyframeDebug(const Keyframe &kf);
    static Pose3d poseFromMsg(const PoseMsg &msg);
    /// Convert poseimu (IMU in global) to left-camera pose in global using OpenVINS extrinsics.
    bool convertImuPoseMsgToCameraPose(const PoseMsg &pose_msg, Pose3d &camera_pose) const;

    std::shared_ptr<Application> app_;
    std::shared_ptr<ov_msckf::ROS1Visualizer> viz_;
    std::shared_ptr<ros::NodeHandle> nh_;
    std::shared_ptr<ov_core::YamlParser> parser_;

    ros::Subscriber sub_imu_;
    ros::Subscriber sub_pose_;
    ros::Publisher pub_kf_pose_;
    ros::Publisher pub_kf_image_;
    bool kf_publish_debug_ = true;

    std::vector<ros::Subscriber> camera_subs_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    std::vector<std::shared_ptr<message_filters::Synchronizer<SyncPolicy>>> sync_cam_;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> sync_subs_cam_;
  };

} // namespace vslam

#endif // VSLAM_ROS_WRAPPER_H
