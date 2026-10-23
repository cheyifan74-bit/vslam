/*
 * @Description: ROS2 Wrapper implementation — thin glue layer (rclcpp counterpart of ros_wrapper.cpp).
 * @Author: che yifan
 * @Date: 2026-10-22
 * @LastEditTime: 2026-10-22
 * @LastEditors: che yifan
 * @Reference:
 */
#include "ros2_wrapper.h"

#include "app/application.h"
#include "keyframe_select/keyframe_select.h"
#include "utility/math_utils.h"

#include "core/VioManager.h"
#include "ros/ROS2Visualizer.h"
#include "state/State.h"

#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>

namespace vslam
{

  namespace
  {
    double stampToSec(const builtin_interfaces::msg::Time &stamp) { return rclcpp::Time(stamp).seconds(); }

    builtin_interfaces::msg::Time secToStamp(double t)
    {
      return rclcpp::Time(static_cast<int64_t>(t * 1e9), RCL_ROS_TIME);
    }
  } // namespace

  RosWrapper::RosWrapper(std::shared_ptr<rclcpp::Node> node, const std::string &config_path) : node_(node)
  {
    init(config_path);
  }

  RosWrapper::~RosWrapper()
  {
    if (viz_)
    {
      viz_->visualize_final();
    }
  }

  std::shared_ptr<Application> RosWrapper::getApplication() const { return app_; }
  std::shared_ptr<ov_core::YamlParser> RosWrapper::getParser() const { return parser_; }
  std::shared_ptr<rclcpp::Node> RosWrapper::getNode() const { return node_; }
  std::shared_ptr<KeyframeQueue> RosWrapper::getKeyframeQueue() const
  {
    return app_ ? app_->getKeyframeQueue() : nullptr;
  }

  void RosWrapper::init(const std::string &config_path)
  {
    PRINT_INFO("[ROS_WRAPPER]: Initializing Application from config: %s\n", config_path.c_str());
    std::string verbosity = "INFO";
    node_->get_parameter("verbosity", verbosity);
    app_ = std::make_shared<Application>(config_path, verbosity);
    if (!app_->vioInitialized())
    {
      PRINT_ERROR(RED "[ROS_WRAPPER]: Application VIO initialization failed!\n" RESET);
      return;
    }

    parser_ = std::make_shared<ov_core::YamlParser>(config_path);
    parser_->set_node(node_);

    PRINT_INFO("[ROS_WRAPPER]: Setting up ROS2Visualizer...\n");
    viz_ = std::make_shared<ov_msckf::ROS2Visualizer>(node_, app_->getVioManager());

    if (!parser_->successful())
    {
      PRINT_ERROR(RED "[ROS_WRAPPER]: Failed to parse all parameters from config file!\n" RESET);
      return;
    }

    cb_group_imu_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cb_group_cam_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cb_group_pose_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    PRINT_INFO("[ROS_WRAPPER]: Setting up KeyframeSelect topics...\n");
    setupKeyframeSelectTopics();

    PRINT_INFO("[ROS_WRAPPER]: Setting up sensor subscribers...\n");
    setupSubscribers();

    PRINT_INFO("[ROS_WRAPPER]: Initialization complete.\n");
  }

  void RosWrapper::setupKeyframeSelectTopics()
  {
    const KeyframeSelectConfig &kf_cfg = app_->getKeyframeSelectConfig();
    kf_publish_debug_ = kf_cfg.publish_debug;

    rclcpp::SubscriptionOptions options;
    options.callback_group = cb_group_pose_;
    sub_pose_ = node_->create_subscription<PoseMsg>(
        kf_cfg.topic_pose, rclcpp::QoS(kf_cfg.pose_queue_size),
        std::bind(&RosWrapper::callbackPose, this, std::placeholders::_1), options);
    PRINT_INFO("[ROS_WRAPPER]: KeyframeSelect pose topic: %s\n", sub_pose_->get_topic_name());

    if (kf_publish_debug_)
    {
      // Relative names resolve under the node namespace (/ov_msckf/... with the default launch).
      pub_kf_pose_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(kf_cfg.topic_keyframe_pose, 10);
      pub_kf_image_ = node_->create_publisher<ImageMsg>(kf_cfg.topic_keyframe_image, 2);
      PRINT_INFO("[ROS_WRAPPER]: KeyframeSelect debug pub: %s, %s\n", pub_kf_pose_->get_topic_name(),
                 pub_kf_image_->get_topic_name());

      // Debug publish runs on KeyframeSelect worker thread, not in sensor callbacks.
      auto kf_select = app_->getKeyframeSelect();
      if (kf_select)
      {
        kf_select->setKeyframeSelectedCallback([this](const Keyframe &kf) { publishKeyframeDebug(kf); });
      }
    }
  }

  Pose3d RosWrapper::poseFromMsg(const PoseMsg &msg)
  {
    Pose3d pose;
    pose.position = Eigen::Vector3d(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z);
    pose.orientation = Eigen::Quaterniond(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                                          msg.pose.pose.orientation.y, msg.pose.pose.orientation.z);
    pose.orientation.normalize();
    return pose;
  }

  bool RosWrapper::convertImuPoseMsgToCameraPose(const PoseMsg &pose_msg, Pose3d &camera_pose) const
  {
    if (!app_ || !app_->getVioManager())
    {
      return false;
    }

    auto state = app_->getVioManager()->get_state();
    if (!state || state->_calib_IMUtoCAM.find(0) == state->_calib_IMUtoCAM.end() ||
        state->_calib_IMUtoCAM.at(0) == nullptr)
    {
      return false;
    }

    // OpenVINS stores cam0 extrinsics as R_ItoC / p_IinC.
    const auto &calib_imu_to_cam0 = state->_calib_IMUtoCAM.at(0);
    const Eigen::Matrix3d R_ItoC = calib_imu_to_cam0->Rot();
    const Eigen::Vector3d p_IinC = calib_imu_to_cam0->pos();

    const Pose3d imu_pose = poseFromMsg(pose_msg);
    camera_pose = transformImuPoseToCameraPose(imu_pose, R_ItoC, p_IinC);
    return true;
  }

  void RosWrapper::callbackPose(const PoseMsg::ConstSharedPtr pose_msg)
  {
    auto kf_select = app_->getKeyframeSelect();
    if (!kf_select)
    {
      return;
    }

    Pose3d camera_pose;
    if (!convertImuPoseMsgToCameraPose(*pose_msg, camera_pose))
    {
      return;
    }

    // Feed left-camera pose; selection runs on KeyframeSelect worker thread.
    kf_select->feedPose(stampToSec(pose_msg->header.stamp), camera_pose, pose_msg->header.frame_id);
  }

  void RosWrapper::feedKeyframeImage(double timestamp, const cv::Mat &image, int cam_id)
  {
    auto kf_select = app_->getKeyframeSelect();
    if (!kf_select)
    {
      return;
    }

    // Callback only enqueues; selection runs on KeyframeSelect worker thread.
    kf_select->feedImage(timestamp, image, cam_id);
  }

  void RosWrapper::publishKeyframeDebug(const Keyframe &kf)
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = secToStamp(kf.timestamp);
    pose_msg.header.frame_id = kf.frame_id.empty() ? "global" : kf.frame_id;
    pose_msg.pose.position.x = kf.pose.position.x();
    pose_msg.pose.position.y = kf.pose.position.y();
    pose_msg.pose.position.z = kf.pose.position.z();
    pose_msg.pose.orientation.x = kf.pose.orientation.x();
    pose_msg.pose.orientation.y = kf.pose.orientation.y();
    pose_msg.pose.orientation.z = kf.pose.orientation.z();
    pose_msg.pose.orientation.w = kf.pose.orientation.w();
    pub_kf_pose_->publish(pose_msg);

    if (!kf.image.empty() && pub_kf_image_->get_subscription_count() > 0)
    {
      std_msgs::msg::Header header = pose_msg.header;
      header.frame_id = "cam0";
      const std::string encoding = (kf.image.channels() == 1) ? "mono8" : "bgr8";
      pub_kf_image_->publish(*cv_bridge::CvImage(header, encoding, kf.image).toImageMsg());
    }
  }

  void RosWrapper::run()
  {
    if (!app_ || !app_->vioInitialized())
    {
      PRINT_ERROR(RED "[ROS_WRAPPER]: Cannot run, VIO not initialized!\n" RESET);
      return;
    }

    PRINT_INFO("[ROS_WRAPPER]: Starting ROS2 multi-threaded executor...\n");
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node_);
    executor.spin();
  }

  void RosWrapper::setupSubscribers()
  {
    // SensorDataQoS (best effort) accepts both reliable (ros2 bag play) and best-effort (drivers) publishers.
    {
      std::string topic_imu = "/imu0";
      node_->get_parameter("topic_imu", topic_imu);
      parser_->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);

      rclcpp::SubscriptionOptions options;
      options.callback_group = cb_group_imu_;
      sub_imu_ = node_->create_subscription<ImuMsg>(
          topic_imu, rclcpp::SensorDataQoS().keep_last(2000),
          std::bind(&RosWrapper::callbackInertial, this, std::placeholders::_1), options);
      PRINT_INFO("[ROS_WRAPPER]: Subscribing to IMU: %s\n", topic_imu.c_str());
    }

    int num_cameras = app_->getVioManager()->get_params().state_options.num_cameras;

    rclcpp::SubscriptionOptions cam_options;
    cam_options.callback_group = cb_group_cam_;

    if (num_cameras == 2)
    {
      std::string cam_topic0 = "/cam0/image_raw";
      std::string cam_topic1 = "/cam1/image_raw";
      node_->get_parameter("topic_camera0", cam_topic0);
      node_->get_parameter("topic_camera1", cam_topic1);
      parser_->parse_external("relative_config_imucam", "cam0", "rostopic", cam_topic0);
      parser_->parse_external("relative_config_imucam", "cam1", "rostopic", cam_topic1);

      rmw_qos_profile_t qos = rmw_qos_profile_sensor_data;
      qos.depth = 10;
      auto image_sub0 = std::make_shared<message_filters::Subscriber<ImageMsg>>(node_.get(), cam_topic0, qos, cam_options);
      auto image_sub1 = std::make_shared<message_filters::Subscriber<ImageMsg>>(node_.get(), cam_topic1, qos, cam_options);
      auto sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *image_sub0, *image_sub1);
      sync->registerCallback(
          std::bind(&RosWrapper::callbackStereo, this, std::placeholders::_1, std::placeholders::_2, 0, 1));

      sync_cam_.push_back(sync);
      sync_subs_cam_.push_back(image_sub0);
      sync_subs_cam_.push_back(image_sub1);

      PRINT_INFO("[ROS_WRAPPER]: Subscribing to stereo cam0: %s\n", cam_topic0.c_str());
      PRINT_INFO("[ROS_WRAPPER]: Subscribing to stereo cam1: %s\n", cam_topic1.c_str());
    }
    else
    {
      for (int i = 0; i < num_cameras; i++)
      {
        std::string cam_topic = "/cam" + std::to_string(i) + "/image_raw";
        node_->get_parameter("topic_camera" + std::to_string(i), cam_topic);
        parser_->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);

        camera_subs_.push_back(node_->create_subscription<ImageMsg>(
            cam_topic, rclcpp::SensorDataQoS().keep_last(10),
            [this, i](const ImageMsg::ConstSharedPtr msg) { callbackMonocular(msg, i); }, cam_options));
        PRINT_INFO("[ROS_WRAPPER]: Subscribing to mono cam%d: %s\n", i, cam_topic.c_str());
      }
    }
  }

  void RosWrapper::callbackInertial(const ImuMsg::ConstSharedPtr msg)
  {
    ov_core::ImuData message;
    message.timestamp = stampToSec(msg->header.stamp);
    message.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    message.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

    viz_->handleImuMeasurement(message);
  }

  void RosWrapper::callbackMonocular(const ImageMsg::ConstSharedPtr msg, int cam_id)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
      PRINT_ERROR("[ROS_WRAPPER]: cv_bridge exception for cam%d: %s\n", cam_id, e.what());
      return;
    }

    ov_core::CameraData message;
    message.timestamp = stampToSec(cv_ptr->header.stamp);
    message.sensor_ids.push_back(cam_id);
    message.images.push_back(cv_ptr->image.clone());

    auto params = app_->getVioManager()->get_params();
    if (params.use_mask)
    {
      message.masks.push_back(params.masks.at(cam_id));
    }
    else
    {
      message.masks.push_back(cv::Mat::zeros(cv_ptr->image.rows, cv_ptr->image.cols, CV_8UC1));
    }

    if (cam_id == 0)
    {
      feedKeyframeImage(message.timestamp, message.images.front(), cam_id);
    }

    viz_->handleCameraMeasurement(message);
  }

  void RosWrapper::callbackStereo(const ImageMsg::ConstSharedPtr msg0, const ImageMsg::ConstSharedPtr msg1, int cam_id0,
                                  int cam_id1)
  {
    cv_bridge::CvImageConstPtr cv_ptr0, cv_ptr1;
    try
    {
      cv_ptr0 = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
      PRINT_ERROR("[ROS_WRAPPER]: cv_bridge exception for cam%d: %s\n", cam_id0, e.what());
      return;
    }
    try
    {
      cv_ptr1 = cv_bridge::toCvShare(msg1, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
      PRINT_ERROR("[ROS_WRAPPER]: cv_bridge exception for cam%d: %s\n", cam_id1, e.what());
      return;
    }

    ov_core::CameraData message;
    message.timestamp = stampToSec(cv_ptr0->header.stamp);
    message.sensor_ids.push_back(cam_id0);
    message.sensor_ids.push_back(cam_id1);
    message.images.push_back(cv_ptr0->image.clone());
    message.images.push_back(cv_ptr1->image.clone());

    auto params = app_->getVioManager()->get_params();
    if (params.use_mask)
    {
      message.masks.push_back(params.masks.at(cam_id0));
      message.masks.push_back(params.masks.at(cam_id1));
    }
    else
    {
      message.masks.push_back(cv::Mat::zeros(cv_ptr0->image.rows, cv_ptr0->image.cols, CV_8UC1));
      message.masks.push_back(cv::Mat::zeros(cv_ptr1->image.rows, cv_ptr1->image.cols, CV_8UC1));
    }

    feedKeyframeImage(message.timestamp, message.images.front(), cam_id0);
    viz_->handleCameraMeasurement(message);
  }

} // namespace vslam

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  // YamlParser reads overrides via get_parameter(), so accept any parameter passed in.
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("ov_msckf", options);

  std::string config_path;
  if (!node->get_parameter("config_path", config_path) || config_path.empty())
  {
    RCLCPP_FATAL(node->get_logger(), "Parameter 'config_path' is required! "
                                     "Usage: ros2 run vslam vslam_node --ros-args -p config_path:=/path/to/config.yaml");
    rclcpp::shutdown();
    return 1;
  }

  {
    vslam::RosWrapper wrapper(node, config_path);
    wrapper.run();
  }

  rclcpp::shutdown();
  return 0;
}
