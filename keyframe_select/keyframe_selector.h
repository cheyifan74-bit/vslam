/*
 * @Description: Pure keyframe selection logic (distance / angle only).
 * @Author: che yifan
 * @Date: 2026-08-02
 */
#ifndef VSLAM_KEYFRAME_SELECT_KEYFRAME_SELECTOR_H
#define VSLAM_KEYFRAME_SELECT_KEYFRAME_SELECTOR_H

#include "keyframe_select/keyframe.h"

namespace vslam
{

  class KeyframeSelector
  {
  public:
    KeyframeSelector() = default;
    explicit KeyframeSelector(const KeyframeSelectConfig &config);

    void setSelectConfig(const KeyframeSelectConfig &config);
    const KeyframeSelectConfig &getSelectConfig() const;

    /// Reset last-keyframe state (e.g. after VIO re-init).
    void reset();

    /// Decide whether the current synced (image, pose) should become a keyframe.
    /// Uses distance / angle relative to the last keyframe.
    /// Returns true and fills out_kf if selected; otherwise false.
    bool trySelectKeyframe(double timestamp, const Pose3d &pose, const cv::Mat &image, Keyframe &out_kf,
                           int cam_id = 0, const std::string &frame_id = "global");

    bool hasLastKeyframe() const;
    const Keyframe &lastKeyframe() const;

  private:
    KeyframeSelectConfig config_;
    bool has_last_keyframe_ = false;
    Keyframe last_kf_;
    uint64_t next_id_ = 0;
  };

} // namespace vslam

#endif // VSLAM_KEYFRAME_SELECT_KEYFRAME_SELECTOR_H
