/*
 * @Description: ROS Wrapper implementation — thin glue layer.
 * @Author: che yifan
 * @Date: 2026-07-26 09:39:40
 * @LastEditTime: 2026-08-02
 * @LastEditors: che yifan
 * @Reference:
 */
#include "ros_wrapper.h"

#include "app/application.h"
#include "keyframe_select/keyframe_select.h"
#include "utility/math_utils.h"

#include "core/VioManager.h"
#include "ros/ROS1Visualizer.h"
#include "state/State.h"

#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>

namespace vslam
{

  RosWrapper::RosWrapper(std::shared_ptr<ros::NodeHandle> nh, const std::string &config_path) : nh_(nh)
  {
    init(nh, config_path);
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
  std::shared_ptr<ros::NodeHandle> RosWrapper::getNodeHandle() const { return nh_; }
  std::shared_ptr<KeyframeQueue> RosWrapper::getKeyframeQueue() const
  {
    return app_ ? app_->getKeyframeQueue() : nullptr;
  }

  void RosWrapper::init(std::shared_ptr<ros::NodeHandle> nh, const std::string &config_path)
  {
    PRINT_INFO("[ROS_WRAPPER]: Initializing Application from config: %s\n", config_path.c_str());
    app_ = std::make_shared<Application>(config_path);
    if (!app_->vioInitialized())
    {
      PRINT_ERROR(RED "[ROS_WRAPPER]: Application VIO initialization failed!\n" RESET);
      return;
    }

    parser_ = std::make_shared<ov_core::YamlParser>(config_path);
    parser_->set_node_handler(nh);

    PRINT_INFO("[ROS_WRAPPER]: Setting up ROS1Visualizer...\n");
    viz_ = std::make_shared<ov_msckf::ROS1Visualizer>(nh, app_->getVioManager());

    if (!parser_->successful())
    {
      PRINT_ERROR(RED "[ROS_WRAPPER]: Failed to parse all parameters from config file!\n" RESET);
      return;
    }

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

    sub_pose_ = nh_->subscribe(kf_cfg.topic_pose, kf_cfg.pose_queue_size, &RosWrapper::callbackPose, this);
    PRINT_INFO("[ROS_WRAPPER]: KeyframeSelect pose topic: %s\n", kf_cfg.topic_pose.c_str());

    if (kf_publish_debug_)
    {
      pub_kf_pose_ = nh_->advertise<geometry_msgs::PoseStamped>(kf_cfg.topic_keyframe_pose, 10);
      pub_kf_image_ = nh_->advertise<sensor_msgs::Image>(kf_cfg.topic_keyframe_image, 2);
      PRINT_INFO("[ROS_WRAPPER]: KeyframeSelect debug pub: %s, %s\n", kf_cfg.topic_keyframe_pose.c_str(),
                 kf_cfg.topic_keyframe_image.c_str());

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

  void RosWrapper::callbackPose(const PoseMsg::ConstPtr &pose_msg)
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
    kf_select->feedPose(pose_msg->header.stamp.toSec(), camera_pose, pose_msg->header.frame_id);
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
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = ros::Time(kf.timestamp);
    pose_msg.header.frame_id = kf.frame_id.empty() ? "global" : kf.frame_id;
    pose_msg.pose.position.x = kf.pose.position.x();
    pose_msg.pose.position.y = kf.pose.position.y();
    pose_msg.pose.position.z = kf.pose.position.z();
    pose_msg.pose.orientation.x = kf.pose.orientation.x();
    pose_msg.pose.orientation.y = kf.pose.orientation.y();
    pose_msg.pose.orientation.z = kf.pose.orientation.z();
    pose_msg.pose.orientation.w = kf.pose.orientation.w();
    pub_kf_pose_.publish(pose_msg);

    if (!kf.image.empty() && pub_kf_image_.getNumSubscribers() > 0)
    {
      std_msgs::Header header = pose_msg.header;
      header.frame_id = "cam0";
      const std::string encoding = (kf.image.channels() == 1) ? "mono8" : "bgr8";
      pub_kf_image_.publish(cv_bridge::CvImage(header, encoding, kf.image).toImageMsg());
    }
  }

  void RosWrapper::run()
  {
    if (!app_ || !app_->vioInitialized())
    {
      PRINT_ERROR(RED "[ROS_WRAPPER]: Cannot run, VIO not initialized!\n" RESET);
      return;
    }

    PRINT_INFO("[ROS_WRAPPER]: Starting ROS async spinner...\n");
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::waitForShutdown();
  }

  void RosWrapper::setupSubscribers()
  {
    {
      std::string topic_imu;
      nh_->param<std::string>("topic_imu", topic_imu, "/imu0");
      parser_->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);

      sub_imu_ = nh_->subscribe(topic_imu, 2000, &RosWrapper::callbackInertial, this,
                                ros::TransportHints().tcpNoDelay());
      PRINT_INFO("[ROS_WRAPPER]: Subscribing to IMU: %s\n", topic_imu.c_str());
    }

    int num_cameras = app_->getVioManager()->get_params().state_options.num_cameras;

    if (num_cameras == 2)
    {
      std::string cam_topic0, cam_topic1;
      nh_->param<std::string>("topic_camera0", cam_topic0, "/cam0/image_raw");
      nh_->param<std::string>("topic_camera1", cam_topic1, "/cam1/image_raw");
      parser_->parse_external("relative_config_imucam", "cam0", "rostopic", cam_topic0);
      parser_->parse_external("relative_config_imucam", "cam1", "rostopic", cam_topic1);

      auto image_sub0 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(*nh_, cam_topic0, 1);
      auto image_sub1 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(*nh_, cam_topic1, 1);
      auto sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *image_sub0, *image_sub1);
      sync->registerCallback(boost::bind(&RosWrapper::callbackStereo, this, _1, _2, 0, 1));

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
        std::string cam_topic;
        nh_->param<std::string>("topic_camera" + std::to_string(i), cam_topic,
                                "/cam" + std::to_string(i) + "/image_raw");
        parser_->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);

        camera_subs_.push_back(nh_->subscribe<sensor_msgs::Image>(
            cam_topic, 10, boost::bind(&RosWrapper::callbackMonocular, this, _1, i)));
        PRINT_INFO("[ROS_WRAPPER]: Subscribing to mono cam%d: %s\n", i, cam_topic.c_str());
      }
    }
  }

  void RosWrapper::callbackInertial(const sensor_msgs::Imu::ConstPtr &msg)
  {
    ov_core::ImuData message;
    message.timestamp = msg->header.stamp.toSec();
    message.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    message.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

    viz_->handleImuMeasurement(message);
  }

  void RosWrapper::callbackMonocular(const sensor_msgs::ImageConstPtr &msg, int cam_id)
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
    message.timestamp = cv_ptr->header.stamp.toSec();
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

  void RosWrapper::callbackStereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1,
                                  int cam_id0, int cam_id1)
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
    message.timestamp = cv_ptr0->header.stamp.toSec();
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
  ros::init(argc, argv, "vslam_node");
  auto nh = std::make_shared<ros::NodeHandle>("~");

  std::string config_path;
  if (!nh->getParam("config_path", config_path))
  {
    ROS_FATAL("Parameter 'config_path' is required! "
              "Usage: rosrun vslam vslam_node _config_path:=/path/to/config.yaml");
    return 1;
  }

  vslam::RosWrapper wrapper(nh, config_path);
  wrapper.run();

  return 0;
}
