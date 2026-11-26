/*
 * @Description: ROS2 Wrapper implementation — thin glue layer (rclcpp counterpart of ros_wrapper.cpp).
 * @Author: che yifan
 * @Date: 2026-10-22
 * @LastEditTime: 2026-10-22
 * @LastEditors: che yifan
 * @Reference:
 */
#include "ros2_wrapper.h"

#include <cstdlib>
#include <mutex>

#include <opencv2/imgproc.hpp>

#include "app/application.h"
#include "keyframe_select/keyframe_select.h"
#include "semantic_index/color_image_buffer.h"
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

namespace vslam
{

  namespace
  {
    double stampToSec(const builtin_interfaces::msg::Time &stamp) { return rclcpp::Time(stamp).seconds(); }

    /// Left-camera pose in global for the current filter state, built from the
    /// same IMU pose and cam0 extrinsics poseimu used to carry.
    bool cameraPoseFromState(const ov_msckf::State &state, Pose3d &camera_pose)
    {
      const auto calib = state._calib_IMUtoCAM.find(0);
      if (!state._imu || calib == state._calib_IMUtoCAM.end() || calib->second == nullptr)
      {
        return false;
      }
      // OpenVINS' JPL q_GtoI has the same coefficients as the Hamilton q_ItoG.
      const Eigen::Vector4d q = state._imu->quat();
      Pose3d imu_pose;
      imu_pose.position = state._imu->pos();
      imu_pose.orientation = Eigen::Quaterniond(q(3), q(0), q(1), q(2));
      imu_pose.orientation.normalize();
      // OpenVINS stores cam0 extrinsics as R_ItoC / p_IinC.
      camera_pose = transformImuPoseToCameraPose(imu_pose, calib->second->Rot(), calib->second->pos());
      return true;
    }
  } // namespace

  RosWrapper::RosWrapper(std::shared_ptr<rclcpp::Node> node, const std::string &config_path) : node_(node)
  {
    init(config_path);
  }

  RosWrapper::~RosWrapper()
  {
    if (map_viz_)
    {
      map_viz_->stop();
    }
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

    map_viz_ = std::make_unique<MapVisualizer>(node_, app_->getMapVizConfig());
    map_viz_->attachMapManager(app_->getMapManager());

    PRINT_INFO("[ROS_WRAPPER]: Setting up KeyframeSelect...\n");
    setupKeyframeSelect();

    PRINT_INFO("[ROS_WRAPPER]: Setting up sensor subscribers...\n");
    setupSubscribers();

    PRINT_INFO("[ROS_WRAPPER]: Initialization complete.\n");
  }

  void RosWrapper::setupKeyframeSelect()
  {
    // Poses go straight from the filter to KeyframeSelect instead of through
    // /ov_msckf/poseimu: no serialization, no drops (poseimu keeps only 2), and
    // the stamp is the image time the state was updated at.
    std::shared_ptr<KeyframeSelect> kf_select = app_->getKeyframeSelect();
    if (kf_select)
    {
      viz_->set_state_callback([kf_select](const ov_msckf::State &state) {
        Pose3d camera_pose;
        if (cameraPoseFromState(state, camera_pose))
        {
          // Only enqueues; selection runs on the KeyframeSelect worker thread.
          kf_select->feedPose(state._timestamp, camera_pose, "global");
        }
      });
      PRINT_INFO("[ROS_WRAPPER]: KeyframeSelect fed in-process from the VIO state\n");
    }

    map_viz_->attachKeyframeSelect(app_->getKeyframeSelectConfig(), kf_select);
  }

  void RosWrapper::feedKeyframeImage(double timestamp, const cv::Mat &image, int cam_id,
                                     const cv::Mat &image_right)
  {
    auto kf_select = app_->getKeyframeSelect();
    if (!kf_select)
    {
      return;
    }

    // Callback only enqueues; selection runs on KeyframeSelect worker thread.
    kf_select->feedImage(timestamp, image, cam_id, image_right);
  }

  void RosWrapper::run()
  {
    if (!app_ || !app_->vioInitialized())
    {
      PRINT_ERROR(RED "[ROS_WRAPPER]: Cannot run, VIO not initialized!\n" RESET);
      return;
    }

    if (map_viz_)
    {
      map_viz_->start();
    }

    PRINT_INFO("[ROS_WRAPPER]: Starting ROS2 multi-threaded executor...\n");
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node_);
    executor.spin();

    if (map_viz_)
    {
      map_viz_->stop();
    }
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

    if (app_->getSemanticConfig().enabled && app_->getSemanticConfig().color_enabled &&
        app_->getColorImageBuffer())
    {
      cb_group_color_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      rclcpp::SubscriptionOptions color_options;
      color_options.callback_group = cb_group_color_;
      const std::string color_topic = app_->getSemanticConfig().color_topic;
      sub_color_ = node_->create_subscription<ImageMsg>(
          color_topic, rclcpp::SensorDataQoS().keep_last(30),
          [this](const ImageMsg::ConstSharedPtr msg) { callbackColor(msg); }, color_options);
      PRINT_INFO("[ROS_WRAPPER]: Subscribing to color (semantic): %s\n", color_topic.c_str());
    }
    else if (app_->getSemanticConfig().enabled && !app_->getSemanticConfig().color_enabled)
    {
      PRINT_INFO("[ROS_WRAPPER]: semantic_color_enabled=false, color pairing disabled.\n");
    }
  }

  void RosWrapper::callbackColor(const ImageMsg::ConstSharedPtr msg)
  {
    auto buffer = app_ ? app_->getColorImageBuffer() : nullptr;
    if (!buffer)
    {
      return;
    }
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      // Orbbec publishes rgb8; accept bgr8 too.
      if (msg->encoding == sensor_msgs::image_encodings::RGB8)
      {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
        cv::Mat bgr;
        cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_RGB2BGR);
        buffer->push(stampToSec(cv_ptr->header.stamp), bgr);
        return;
      }
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
      buffer->push(stampToSec(cv_ptr->header.stamp), cv_ptr->image.clone());
    }
    catch (cv_bridge::Exception &e)
    {
      PRINT_ERROR("[ROS_WRAPPER]: cv_bridge exception for color: %s\n", e.what());
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

    feedKeyframeImage(message.timestamp, message.images.front(), cam_id0,
                      message.images.size() > 1 ? message.images[1] : cv::Mat());
    viz_->handleCameraMeasurement(message);
  }

} // namespace vslam

int main(int argc, char **argv)
{
  // Humble's rmw_fastrtps defaults to synchronous sends, which serialize every publisher of the
  // process on one participant lock; asynchronous mode hands large samples to a writer thread.
  // An explicit value in the environment still wins.
  setenv("RMW_FASTRTPS_PUBLICATION_MODE", "ASYNCHRONOUS", 0);
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
