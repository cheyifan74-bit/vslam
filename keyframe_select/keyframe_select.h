/*
 * @Description: Pure KeyframeSelect module (no ROS).
 *               Callbacks only enqueue data; a dedicated worker thread
 *               performs sync + keyframe selection to avoid blocking I/O threads.
 * @Author: che yifan
 * @Date: 2026-08-02
 */
#ifndef VSLAM_KEYFRAME_SELECT_KEYFRAME_SELECT_H
#define VSLAM_KEYFRAME_SELECT_KEYFRAME_SELECT_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "keyframe_select/keyframe.h"
#include "keyframe_select/keyframe_queue.h"
#include "keyframe_select/keyframe_selector.h"

namespace vslam
{

  class ColorImageBuffer;

  /// ROS-free keyframe selection with a dedicated worker thread:
  ///   feedImage / feedPose (enqueue only) → worker → KeyframeQueue
  class KeyframeSelect
  {
  public:
    using KeyframeCallback = std::function<void(const Keyframe &)>;

    explicit KeyframeSelect(const KeyframeSelectConfig &config);
    ~KeyframeSelect();

    KeyframeSelect(const KeyframeSelect &) = delete;
    KeyframeSelect &operator=(const KeyframeSelect &) = delete;

    /// Enqueue left-camera image (cam_id != 0 ignored). Non-blocking aside from the lock.
    /// `image_right` is kept with the same timestamp and attached when this frame
    /// is selected as a keyframe. Dense mapping uses it; sparse mapping does not.
    /// The pixels are shared, not copied (the same buffers also go to OpenVINS),
    /// so neither the caller nor any consumer may modify them in place.
    void feedImage(double timestamp, const cv::Mat &image, int cam_id = 0,
                   const cv::Mat &image_right = cv::Mat());

    /// Enqueue pose. Non-blocking aside from lock.
    void feedPose(double timestamp, const Pose3d &pose, const std::string &frame_id = "global");

    /// Optional color buffer: when set, selected keyframes attach a time-synced RGB.
    void setColorImageBuffer(const std::shared_ptr<ColorImageBuffer> &buffer,
                             double sync_slop_s = 0.05);

    /// Optional callback invoked on the worker thread after a keyframe is selected.
    void setKeyframeSelectedCallback(KeyframeCallback callback);

    std::shared_ptr<KeyframeQueue> getKeyframeQueue() const;
    const KeyframeSelectConfig &getSelectConfig() const;
    const KeyframeSelector &getKeyframeSelector() const;

  private:
    struct StampedImage
    {
      double timestamp = 0.0;
      cv::Mat image;
      cv::Mat image_right;
      int cam_id = 0;
    };

    struct StampedPose
    {
      double timestamp = 0.0;
      Pose3d pose;
      std::string frame_id;

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    /// Worker-thread entry: wait for data, sync image/pose, run keyframe selection.
    void runKeyframeSelectLoop();
    void notifyKeyframeSelectWorker();
    void syncBuffers(std::vector<std::pair<StampedImage, StampedPose>> &matched);
    void processMatched(const StampedImage &image, const StampedPose &pose);

    KeyframeSelectConfig select_config_;
    KeyframeSelector keyframe_selector_;
    std::shared_ptr<KeyframeQueue> keyframe_queue_;
    std::shared_ptr<ColorImageBuffer> color_buffer_;
    double color_sync_slop_s_ = 0.05;

    mutable std::mutex sync_buffer_mutex_;
    std::condition_variable worker_condition_;
    std::deque<StampedImage> image_buffer_;
    /// Images dropped by age since the last warning; guarded by sync_buffer_mutex_.
    size_t aged_out_images_ = 0;
    double last_age_warn_time_ = -1.0;
    /// Timestamp of the last "no color frame" warning; worker-thread only.
    double last_color_warn_time_ = -1.0;
    std::deque<StampedPose, Eigen::aligned_allocator<StampedPose>> pose_buffer_;
    bool has_pending_work_ = false;
    std::atomic<bool> stop_worker_{false};
    std::thread worker_thread_;

    std::mutex keyframe_callback_mutex_;
    KeyframeCallback keyframe_selected_callback_;
  };

} // namespace vslam

#endif // VSLAM_KEYFRAME_SELECT_KEYFRAME_SELECT_H
