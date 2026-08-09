/*
 * @Description: Pure KeyframeSelect module with dedicated worker thread.
 * @Author: che yifan
 * @Date: 2026-08-02
 */
#include "keyframe_select/keyframe_select.h"

#include <cmath>

#include "utils/print.h"

namespace vslam
{

  KeyframeSelect::KeyframeSelect(const KeyframeSelectConfig &config)
      : select_config_(config), keyframe_selector_(config),
        keyframe_queue_(std::make_shared<KeyframeQueue>(config.output_queue_size))
  {
    worker_thread_ = std::thread(&KeyframeSelect::runKeyframeSelectLoop, this);
    PRINT_INFO("[KF_SELECT]: Worker thread started (dist=%.3f m, angle=%.2f deg, sync_slop=%.3f s)\n",
               select_config_.dist_thresh, select_config_.angle_thresh_deg, select_config_.sync_slop);
  }

  KeyframeSelect::~KeyframeSelect()
  {
    {
      std::lock_guard<std::mutex> lock(sync_buffer_mutex_);
      stop_worker_ = true;
      has_pending_work_ = true;
    }
    worker_condition_.notify_all();

    if (worker_thread_.joinable())
    {
      worker_thread_.join();
    }

    if (keyframe_queue_)
    {
      keyframe_queue_->shutdown();
    }
  }

  std::shared_ptr<KeyframeQueue> KeyframeSelect::getKeyframeQueue() const { return keyframe_queue_; }

  const KeyframeSelectConfig &KeyframeSelect::getSelectConfig() const { return select_config_; }

  const KeyframeSelector &KeyframeSelect::getKeyframeSelector() const { return keyframe_selector_; }

  void KeyframeSelect::setKeyframeSelectedCallback(KeyframeCallback callback)
  {
    std::lock_guard<std::mutex> lock(keyframe_callback_mutex_);
    keyframe_selected_callback_ = std::move(callback);
  }

  void KeyframeSelect::notifyKeyframeSelectWorker()
  {
    has_pending_work_ = true;
    worker_condition_.notify_one();
  }

  void KeyframeSelect::feedImage(double timestamp, const cv::Mat &image, int cam_id)
  {
    if (cam_id != 0 || image.empty() || stop_worker_)
    {
      return;
    }

    StampedImage stamped;
    stamped.timestamp = timestamp;
    stamped.image = image.clone();
    stamped.cam_id = cam_id;

    {
      std::lock_guard<std::mutex> lock(sync_buffer_mutex_);
      image_buffer_.push_back(std::move(stamped));
      while (static_cast<int>(image_buffer_.size()) > select_config_.image_queue_size)
      {
        image_buffer_.pop_front();
      }
      notifyKeyframeSelectWorker();
    }
  }

  void KeyframeSelect::feedPose(double timestamp, const Pose3d &pose, const std::string &frame_id)
  {
    if (stop_worker_)
    {
      return;
    }

    StampedPose stamped;
    stamped.timestamp = timestamp;
    stamped.pose = pose;
    stamped.frame_id = frame_id;

    {
      std::lock_guard<std::mutex> lock(sync_buffer_mutex_);
      pose_buffer_.push_back(std::move(stamped));
      while (static_cast<int>(pose_buffer_.size()) > select_config_.pose_queue_size)
      {
        pose_buffer_.pop_front();
      }
      notifyKeyframeSelectWorker();
    }
  }

  void KeyframeSelect::runKeyframeSelectLoop()
  {
    while (true)
    {
      std::vector<std::pair<StampedImage, StampedPose>> matched;
      {
        std::unique_lock<std::mutex> lock(sync_buffer_mutex_);
        worker_condition_.wait(lock, [this] { return stop_worker_ || has_pending_work_; });

        if (stop_worker_)
        {
          break;
        }

        has_pending_work_ = false;
        syncBuffers(matched);

        // More data may still be pairable after this batch; keep pending if both buffers non-empty.
        if (!image_buffer_.empty() && !pose_buffer_.empty())
        {
          // Could be waiting for closer timestamps; do not spin. Next feed will notify.
        }
      }

      for (const auto &pair : matched)
      {
        processMatched(pair.first, pair.second);
      }
    }
  }

  void KeyframeSelect::syncBuffers(std::vector<std::pair<StampedImage, StampedPose>> &matched)
  {
    while (!image_buffer_.empty() && !pose_buffer_.empty())
    {
      const StampedImage &img = image_buffer_.front();

      size_t best_idx = 0;
      double best_dt = std::abs(pose_buffer_.front().timestamp - img.timestamp);
      for (size_t i = 1; i < pose_buffer_.size(); ++i)
      {
        const double dt = std::abs(pose_buffer_[i].timestamp - img.timestamp);
        if (dt < best_dt)
        {
          best_dt = dt;
          best_idx = i;
        }
      }

      if (best_dt <= select_config_.sync_slop)
      {
        StampedImage matched_img = std::move(image_buffer_.front());
        image_buffer_.pop_front();
        StampedPose matched_pose = std::move(pose_buffer_[best_idx]);
        pose_buffer_.erase(pose_buffer_.begin() + static_cast<std::ptrdiff_t>(best_idx));
        matched.emplace_back(std::move(matched_img), std::move(matched_pose));
        continue;
      }

      if (img.timestamp + select_config_.sync_slop < pose_buffer_.front().timestamp)
      {
        image_buffer_.pop_front();
        continue;
      }

      if (pose_buffer_.front().timestamp + select_config_.sync_slop < img.timestamp)
      {
        pose_buffer_.pop_front();
        continue;
      }

      break;
    }
  }

  void KeyframeSelect::processMatched(const StampedImage &image, const StampedPose &pose)
  {
    Keyframe kf;
    if (!keyframe_selector_.trySelectKeyframe(image.timestamp, pose.pose, image.image, kf, image.cam_id,
                                              pose.frame_id))
    {
      return;
    }

    PRINT_INFO("[KF_SELECT]: Selected keyframe id=%llu t=%.6f\n", static_cast<unsigned long long>(kf.id),
               kf.timestamp);

    KeyframeCallback callback;
    {
      std::lock_guard<std::mutex> lock(keyframe_callback_mutex_);
      callback = keyframe_selected_callback_;
    }
    if (callback)
    {
      callback(kf);
    }

    keyframe_queue_->push(std::move(kf));
  }

} // namespace vslam
