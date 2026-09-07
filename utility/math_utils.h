/*
 * @Description: Common math utilities.
 * @Author: che yifan
 * @Date: 2026-08-08
 */
#ifndef VSLAM_UTILITY_MATH_UTILS_H
#define VSLAM_UTILITY_MATH_UTILS_H

#include <Eigen/Geometry>

#include "colmap/geometry/rigid3.h"
#include "keyframe_select/keyframe.h"

namespace vslam
{

  /// Shortest rotation angle (radians) from q_from to q_to.
  double rotationAngleRad(const Eigen::Quaterniond &q_from, const Eigen::Quaterniond &q_to);

  /**
   * @brief Convert global IMU pose to global camera pose.
   *
   * OpenVINS convention: calib stores R_ItoC and p_IinC (IMU expressed in camera).
   * imu_pose stores position p_IinG and orientation as R_ItoG (Eigen/Hamilton).
   *
   * Resulting camera pose: p_CinG and R_CtoG.
   */
  Pose3d transformImuPoseToCameraPose(const Pose3d &imu_pose, const Eigen::Matrix3d &R_ItoC,
                                      const Eigen::Vector3d &p_IinC);

  /// Keyframe pose is left camera in global (world_from_cam). COLMAP uses cam_from_world.
  colmap::Rigid3d vioCamFromWorld(const Pose3d &pose);

} // namespace vslam

#endif // VSLAM_UTILITY_MATH_UTILS_H
