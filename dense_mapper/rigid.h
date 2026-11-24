/*
 * @Description: dense::Rigid — the light pose type of the dense mapper's public API,
 *               so that callers need no COLMAP header.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_DENSE_MAPPER_RIGID_H
#define VSLAM_DENSE_MAPPER_RIGID_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace vslam
{
namespace dense
{

/// p_b = q * p_a + t for a transform named b_from_a.
struct Rigid
{
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Rigid operator*(const Rigid &other) const
  {
    Rigid out;
    out.q = (q * other.q).normalized();
    out.t = q * other.t + t;
    return out;
  }
  Rigid inverse() const
  {
    Rigid out;
    out.q = q.conjugate();
    out.t = -(out.q * t);
    return out;
  }
};

} // namespace dense
} // namespace vslam

#endif
