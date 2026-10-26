/*
 * @Description: ROS Wrapper implementation — thin glue layer.
 * @Author: che yifan
 * @Date: 2026-07-26 09:39:40
 * @LastEditTime: 2026-08-02
 * @LastEditors: che yifan
 * @Reference:
 */
#include "ros_wrapper.h"

#include <unordered_map>
#include <algorithm>
#include <array>
#include <cmath>

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
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/Marker.h>
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
    std::string verbosity = "INFO";
    nh->param<std::string>("verbosity", verbosity, verbosity);
    app_ = std::make_shared<Application>(config_path, verbosity);
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

    pub_map_global_ = nh_->advertise<sensor_msgs::PointCloud2>("map_points_global", 1);
    pub_map_local_ = nh_->advertise<sensor_msgs::PointCloud2>("map_points_local", 1);
    pub_map_markers_ = nh_->advertise<visualization_msgs::MarkerArray>("map_visualization", 1);
    if (auto map_manager = app_->getMapManager())
      map_manager->setVisualizationCallback(
          [this](const MapVisualizationSnapshot &snapshot) { publishMapVisualization(snapshot); });

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

  void RosWrapper::publishMapVisualization(const MapVisualizationSnapshot &snapshot)
  {
    const auto make_cloud = [](const std::vector<MapVizPoint> &points,
                               uint8_t r, uint8_t g, uint8_t b) {
      sensor_msgs::PointCloud2 cloud;
      cloud.header.frame_id = "global";
      cloud.header.stamp = ros::Time::now();
      cloud.height = 1;
      cloud.width = static_cast<uint32_t>(points.size());
      cloud.is_dense = true;
      sensor_msgs::PointCloud2Modifier modifier(cloud);
      modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
      modifier.resize(points.size());
      sensor_msgs::PointCloud2Iterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
      sensor_msgs::PointCloud2Iterator<uint32_t> rgb(cloud, "rgb");
      const uint32_t color = (static_cast<uint32_t>(r) << 16) |
                             (static_cast<uint32_t>(g) << 8) | b;
      for (const auto &point : points) {
        *x = static_cast<float>(point.position.x());
        *y = static_cast<float>(point.position.y());
        *z = static_cast<float>(point.position.z());
        *rgb = color;
        ++x; ++y; ++z; ++rgb;
      }
      return cloud;
    };
    const ros::Time stamp = ros::Time::now();
    auto global_cloud = make_cloud(snapshot.global_points, 255, 255, 255);
    auto local_cloud = make_cloud(snapshot.local_points, 255, 0, 0);
    global_cloud.header.stamp = stamp;
    local_cloud.header.stamp = stamp;
    pub_map_global_.publish(global_cloud);
    pub_map_local_.publish(local_cloud);

    visualization_msgs::MarkerArray array;
    visualization_msgs::Marker clear;
    clear.header.frame_id = "global";
    clear.header.stamp = stamp;
    clear.ns = "vslam_map";
    clear.id = 0;
    clear.action = visualization_msgs::Marker::DELETEALL;
    array.markers.push_back(clear);

    std::vector<double> steps;
    for (size_t i = 1; i < snapshot.keyframes.size(); ++i) {
      const double distance = (snapshot.keyframes[i].position - snapshot.keyframes[i - 1].position).norm();
      if (std::isfinite(distance) && distance > 1e-4) steps.push_back(distance);
    }
    double depth = 0.25;
    if (!steps.empty()) {
      const auto middle = steps.begin() + steps.size() / 2;
      std::nth_element(steps.begin(), middle, steps.end());
      depth = std::clamp(*middle * 0.30, 0.12, 1.5);
    }

    visualization_msgs::Marker positions;
    positions.header = clear.header;
    positions.ns = "keyframes";
    positions.id = 1;
    positions.type = visualization_msgs::Marker::SPHERE_LIST;
    positions.action = visualization_msgs::Marker::ADD;
    positions.scale.x = positions.scale.y = positions.scale.z = depth * 0.10;
    positions.color.r = 0.1; positions.color.g = 0.8; positions.color.b = 1.0; positions.color.a = 1.0;

    visualization_msgs::Marker path = positions;
    path.ns = "keyframe_path"; path.id = 2;
    path.type = visualization_msgs::Marker::LINE_STRIP;
    path.scale.x = depth * 0.035;

    visualization_msgs::Marker frustums = positions;
    frustums.ns = "keyframe_frustums"; frustums.id = 3;
    frustums.type = visualization_msgs::Marker::LINE_LIST;
    frustums.scale.x = depth * 0.025;
    frustums.color.r = 0.1; frustums.color.g = 0.8; frustums.color.b = 1.0;
    visualization_msgs::Marker loops = frustums;
    loops.ns = "loop_edges"; loops.id = 4;
    loops.scale.x = depth * 0.04;
    loops.color.r = 1.0; loops.color.g = 0.1; loops.color.b = 1.0;

    const auto to_point = [](const Eigen::Vector3d &v) {
      geometry_msgs::Point p; p.x = v.x(); p.y = v.y(); p.z = v.z(); return p;
    };
    const auto add_segment = [](visualization_msgs::Marker &marker,
                                const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
      geometry_msgs::Point p, q;
      p.x=a.x(); p.y=a.y(); p.z=a.z(); q.x=b.x(); q.y=b.y(); q.z=b.z();
      marker.points.push_back(p); marker.points.push_back(q);
    };
    std::unordered_map<uint32_t, Eigen::Vector3d> positions_by_id;
    for (const auto &kf : snapshot.keyframes) {
      positions.points.push_back(to_point(kf.position));
      path.points.push_back(to_point(kf.position));
      positions_by_id[kf.id] = kf.position;
      const Eigen::Quaterniond q = kf.orientation.normalized();
      const double w = depth * 0.65, h = depth * 0.45;
      const std::array<Eigen::Vector3d, 4> corners = {
          Eigen::Vector3d(-w, -h, depth), Eigen::Vector3d(w, -h, depth),
          Eigen::Vector3d(w, h, depth), Eigen::Vector3d(-w, h, depth)};
      std::array<Eigen::Vector3d, 4> world_corners;
      for (size_t i = 0; i < corners.size(); ++i) world_corners[i] = kf.position + q * corners[i];
      for (size_t i = 0; i < corners.size(); ++i) {
        add_segment(frustums, kf.position, world_corners[i]);
        add_segment(frustums, world_corners[i], world_corners[(i + 1) % corners.size()]);
      }
      add_segment(frustums, kf.position, kf.position + q * Eigen::Vector3d(0, 0, depth * 1.2));
    }
    for (const auto &loop : snapshot.confirmed_loops) {
      if (positions_by_id.count(loop.current_id) && positions_by_id.count(loop.candidate_id))
        add_segment(loops, positions_by_id.at(loop.current_id), positions_by_id.at(loop.candidate_id));
    }

    visualization_msgs::Marker current = positions;
    current.ns = "current_keyframe"; current.id = 5;
    current.scale.x = current.scale.y = current.scale.z = depth * 0.24;
    current.color.r = 1.0; current.color.g = 0.65; current.color.b = 0.0;
    current.points.clear();
    auto current_it = std::find_if(snapshot.keyframes.begin(), snapshot.keyframes.end(),
        [&](const MapVizKeyframe &kf) { return kf.id == snapshot.current_keyframe_id; });
    visualization_msgs::Marker current_frustum = frustums;
    current_frustum.ns = "current_frustum"; current_frustum.id = 6;
    current_frustum.points.clear();
    current_frustum.color.r = 1.0; current_frustum.color.g = 0.65; current_frustum.color.b = 0.0;
    visualization_msgs::Marker axes;
    axes.header = clear.header; axes.ns = "current_axes"; axes.action = visualization_msgs::Marker::ADD;
    axes.type = visualization_msgs::Marker::LINE_LIST; axes.scale.x = depth * 0.045;
    if (current_it != snapshot.keyframes.end()) {
      const auto &kf = *current_it;
      current.points.push_back(to_point(kf.position));
      const Eigen::Quaterniond q = kf.orientation.normalized();
      const double w = depth * 0.65, h = depth * 0.45;
      const std::array<Eigen::Vector3d, 4> corners = {
          Eigen::Vector3d(-w, -h, depth), Eigen::Vector3d(w, -h, depth),
          Eigen::Vector3d(w, h, depth), Eigen::Vector3d(-w, h, depth)};
      std::array<Eigen::Vector3d, 4> world_corners;
      for (size_t i = 0; i < corners.size(); ++i) world_corners[i] = kf.position + q * corners[i];
      for (size_t i = 0; i < corners.size(); ++i) {
        add_segment(current_frustum, kf.position, world_corners[i]);
        add_segment(current_frustum, world_corners[i], world_corners[(i + 1) % corners.size()]);
      }
      add_segment(current_frustum, kf.position, kf.position + q * Eigen::Vector3d(0, 0, depth * 1.2));
      const double axis_len = depth * 1.8;
      visualization_msgs::Marker x_axis = axes, y_axis = axes, z_axis = axes;
      x_axis.id = 0; y_axis.id = 1; z_axis.id = 2;
      x_axis.color.r = 1; x_axis.color.a = 1;
      y_axis.color.g = 1; y_axis.color.a = 1;
      z_axis.color.b = 1; z_axis.color.a = 1;
      add_segment(x_axis, kf.position, kf.position + q * Eigen::Vector3d::UnitX() * axis_len);
      add_segment(y_axis, kf.position, kf.position + q * Eigen::Vector3d::UnitY() * axis_len);
      add_segment(z_axis, kf.position, kf.position + q * Eigen::Vector3d::UnitZ() * axis_len);
      array.markers.push_back(x_axis); array.markers.push_back(y_axis); array.markers.push_back(z_axis);
    }

    array.markers.push_back(positions);
    array.markers.push_back(path);
    array.markers.push_back(frustums);
    array.markers.push_back(current_frustum);
    array.markers.push_back(loops);
    array.markers.push_back(current);
    pub_map_markers_.publish(array);
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
