/*
 * @Description: ROS Wrapper — thin ROS glue layer.
 *               Only does ROS subscription + sensor_msgs conversion.
 *               All scheduling, queueing, and processing lives in OpenVINS VioManager.
 * @Author: che yifan
 * @Date: 2026-07-26 09:39:40
 * @LastEditTime: 2026-07-26 17:27:42
 * @LastEditors: che yifan
 * @Reference:
 */
#ifndef VSLAM_ROS_WRAPPER_H
#define VSLAM_ROS_WRAPPER_H

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

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

  class RosWrapper
  {
  public:
    RosWrapper(std::shared_ptr<ros::NodeHandle> nh, const std::string &config_path);
    ~RosWrapper();

    void run();

    std::shared_ptr<Application> getApplication() const;
    std::shared_ptr<ov_core::YamlParser> getParser() const;
    std::shared_ptr<ros::NodeHandle> getNodeHandle() const;

  private:
    void init(std::shared_ptr<ros::NodeHandle> nh, const std::string &config_path);
    void setupSubscribers();

    void callbackInertial(const sensor_msgs::Imu::ConstPtr &msg);
    void callbackMonocular(const sensor_msgs::ImageConstPtr &msg, int cam_id);
    void callbackStereo(const sensor_msgs::ImageConstPtr &msg0,
                        const sensor_msgs::ImageConstPtr &msg1,
                        int cam_id0, int cam_id1);

    std::shared_ptr<Application> app_;
    std::shared_ptr<ov_msckf::ROS1Visualizer> viz_;
    std::shared_ptr<ros::NodeHandle> nh_;
    std::shared_ptr<ov_core::YamlParser> parser_;

    ros::Subscriber sub_imu_;
    std::vector<ros::Subscriber> camera_subs_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    std::vector<std::shared_ptr<message_filters::Synchronizer<SyncPolicy>>> sync_cam_;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> sync_subs_cam_;
  };

} // namespace vslam

#endif // VSLAM_ROS_WRAPPER_H
