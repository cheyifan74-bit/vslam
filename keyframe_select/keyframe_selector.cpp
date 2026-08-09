/*
 * @Description: Pure keyframe selection logic (distance / angle only).
 * @Author: che yifan
 * @Date: 2026-08-02
 */
#include "keyframe_select/keyframe_selector.h"

#include "utility/math_utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vslam
{

  KeyframeSelector::KeyframeSelector(const KeyframeSelectConfig &config) : config_(config) {}

  void KeyframeSelector::setSelectConfig(const KeyframeSelectConfig &config) { config_ = config; }

  const KeyframeSelectConfig &KeyframeSelector::getSelectConfig() const { return config_; }

  void KeyframeSelector::reset()
  {
    has_last_keyframe_ = false;
    last_kf_ = Keyframe();
    next_id_ = 0;
  }

  bool KeyframeSelector::hasLastKeyframe() const { return has_last_keyframe_; }

  const Keyframe &KeyframeSelector::lastKeyframe() const { return last_kf_; }

  bool KeyframeSelector::trySelectKeyframe(double timestamp, const Pose3d &pose, const cv::Mat &image,
                                           Keyframe &out_kf, int cam_id, const std::string &frame_id)
  {
    if (image.empty())
    {
      return false;
    }

    bool selected = false;
    if (!has_last_keyframe_)
    {
      selected = true;
    }
    else
    {
      const double dist = (pose.position - last_kf_.pose.position).norm();
      const double angle_rad = rotationAngleRad(last_kf_.pose.orientation, pose.orientation);
      const double angle_thresh_rad = config_.angle_thresh_deg * M_PI / 180.0;
      selected = (dist > config_.dist_thresh) || (angle_rad > angle_thresh_rad);
    }

    if (!selected)
    {
      return false;
    }

    out_kf.id = next_id_++;
    out_kf.timestamp = timestamp;
    out_kf.pose = pose;
    out_kf.image = image.clone();
    out_kf.cam_id = cam_id;
    out_kf.frame_id = frame_id;

    last_kf_ = out_kf;
    // Keep a lightweight last pose; avoid retaining every historical image forever.
    last_kf_.image.release();
    has_last_keyframe_ = true;

    return true;
  }

} // namespace vslam
