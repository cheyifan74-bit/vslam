/*
 * @Description: Common math utilities.
 * @Author: che yifan
 * @Date: 2026-08-08
 */
#include "utility/math_utils.h"

#include <algorithm>
#include <cmath>

namespace vslam
{

  double rotationAngleRad(const Eigen::Quaterniond &q_from, const Eigen::Quaterniond &q_to)
  {
    Eigen::Quaterniond q_rel = q_from.normalized().inverse() * q_to.normalized();
    // Shortest angle: use absolute w so we ignore double-cover.
    const double w = std::min(1.0, std::abs(q_rel.w()));
    return 2.0 * std::acos(w);
  }

  Pose3d transformImuPoseToCameraPose(const Pose3d &imu_pose, const Eigen::Matrix3d &R_ItoC,
                                      const Eigen::Vector3d &p_IinC)
  {
    const Eigen::Matrix3d R_ItoG = imu_pose.orientation.normalized().toRotationMatrix();
    const Eigen::Matrix3d R_GtoI = R_ItoG.transpose();
    const Eigen::Matrix3d R_GtoC = R_ItoC * R_GtoI;
    const Eigen::Matrix3d R_CtoG = R_GtoC.transpose();

    Pose3d camera_pose;
    camera_pose.position = imu_pose.position - R_CtoG * p_IinC;
    camera_pose.orientation = Eigen::Quaterniond(R_CtoG);
    camera_pose.orientation.normalize();
    return camera_pose;
  }

} // namespace vslam
