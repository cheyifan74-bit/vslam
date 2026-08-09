/*
 * @Description: Thread-safe keyframe queue for cross-thread consumers.
 * @Author: che yifan
 * @Date: 2026-08-02
 */
#ifndef VSLAM_KEYFRAME_SELECT_KEYFRAME_QUEUE_H
#define VSLAM_KEYFRAME_SELECT_KEYFRAME_QUEUE_H

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

#include <Eigen/Core>

#include "keyframe_select/keyframe.h"

namespace vslam
{

  /// Bounded queue: push from KeyframeSelect callback thread,
  /// wait/try_pop from downstream worker threads.
  class KeyframeQueue
  {
  public:
    explicit KeyframeQueue(std::size_t max_size = 20) : max_size_(max_size) {}

    void setMaxSize(std::size_t max_size)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      max_size_ = max_size;
      while (queue_.size() > max_size_)
      {
        queue_.pop_front();
      }
    }

    /// Push a keyframe. If full, drop the oldest.
    void push(Keyframe kf)
    {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (max_size_ > 0 && queue_.size() >= max_size_)
        {
          queue_.pop_front();
        }
        queue_.push_back(std::move(kf));
      }
      cv_.notify_one();
    }

    bool tryPop(Keyframe &out)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.empty())
      {
        return false;
      }
      out = std::move(queue_.front());
      queue_.pop_front();
      return true;
    }

    /// Block until a keyframe is available or timeout elapses.
    template <typename Rep, typename Period>
    bool waitPop(Keyframe &out, const std::chrono::duration<Rep, Period> &timeout)
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_; }))
      {
        return false;
      }
      if (queue_.empty())
      {
        return false;
      }
      out = std::move(queue_.front());
      queue_.pop_front();
      return true;
    }

    /// Block indefinitely until a keyframe arrives or shutdown().
    bool waitPop(Keyframe &out)
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
      if (queue_.empty())
      {
        return false;
      }
      out = std::move(queue_.front());
      queue_.pop_front();
      return true;
    }

    std::size_t size() const
    {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.size();
    }

    void clear()
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.clear();
    }

    /// Wake blocked consumers (e.g. on node shutdown).
    void shutdown()
    {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
      }
      cv_.notify_all();
    }

  private:
    using QueueType = std::deque<Keyframe, Eigen::aligned_allocator<Keyframe>>;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    QueueType queue_;
    std::size_t max_size_ = 20;
    bool shutdown_ = false;
  };

} // namespace vslam

#endif // VSLAM_KEYFRAME_SELECT_KEYFRAME_QUEUE_H
