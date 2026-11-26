/*
 * @Description: Timestamped color-image ring buffer for keyframe RGB pairing.
 * @Author: che yifan
 * @Date: 2026-11-25
 */
#ifndef VSLAM_SEMANTIC_INDEX_COLOR_IMAGE_BUFFER_H
#define VSLAM_SEMANTIC_INDEX_COLOR_IMAGE_BUFFER_H

#include <cstddef>
#include <deque>
#include <mutex>

#include <opencv2/core.hpp>

namespace vslam
{

  /// Thread-safe ring of recent color frames. Lookup returns a clone of the
  /// nearest frame within |slop| seconds, or false when none matches.
  class ColorImageBuffer
  {
  public:
    explicit ColorImageBuffer(std::size_t max_size = 60);

    void setMaxSize(std::size_t max_size);

    /// Stores a shared view of `image` (caller must not mutate the Mat later).
    void push(double timestamp, const cv::Mat &image);

    bool lookup(double timestamp, double slop_s, cv::Mat *out) const;

  private:
    struct Entry
    {
      double timestamp = 0.0;
      cv::Mat image;
    };

    mutable std::mutex mutex_;
    std::size_t max_size_ = 60;
    std::deque<Entry> buffer_;
  };

} // namespace vslam

#endif // VSLAM_SEMANTIC_INDEX_COLOR_IMAGE_BUFFER_H
