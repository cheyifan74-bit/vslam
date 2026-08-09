/*
 * @Description: Keyframe data definitions for KeyframeSelect module.
 * @Author: che yifan
 * @Date: 2026-08-02
 */
#ifndef VSLAM_KEYFRAME_SELECT_KEYFRAME_H
#define VSLAM_KEYFRAME_SELECT_KEYFRAME_H

#include <cstdint>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace vslam
{
  struct Pose3d
  {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct Keyframe
  {
    uint64_t id = 0;
    double timestamp = 0.0; ///< seconds, aligned with image stamp
    Pose3d pose;            ///< left-camera pose in global (converted from IMU pose)
    cv::Mat image;          ///< left image clone
    int cam_id = 0;         ///< always 0 for now (left)
    std::string frame_id;   ///< pose frame name (e.g. "global")

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct KeyframeSelectConfig
  {
    double dist_thresh = 0.3;       ///< meters
    double angle_thresh_deg = 15.0; ///< degrees
    double sync_slop = 0.02;        ///< image-pose timestamp match slop (seconds)
    int pose_queue_size = 50;
    int image_queue_size = 10;
    int output_queue_size = 20; ///< drop oldest if full

    std::string topic_pose = "/ov_msckf/poseimu";
    std::string topic_keyframe_pose = "keyframe_pose";
    std::string topic_keyframe_image = "keyframe_image";
    bool publish_debug = true;
  };

} // namespace vslam

#endif // VSLAM_KEYFRAME_SELECT_KEYFRAME_H
