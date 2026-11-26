/*
 * @Description: ColorImageBuffer implementation.
 * @Author: che yifan
 * @Date: 2026-11-25
 */
#include "semantic_index/color_image_buffer.h"

#include <cmath>
#include <utility>

namespace vslam
{

  ColorImageBuffer::ColorImageBuffer(std::size_t max_size) : max_size_(max_size == 0 ? 1 : max_size) {}

  void ColorImageBuffer::setMaxSize(std::size_t max_size)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    max_size_ = max_size == 0 ? 1 : max_size;
    while (buffer_.size() > max_size_)
    {
      buffer_.pop_front();
    }
  }

  void ColorImageBuffer::push(double timestamp, const cv::Mat &image)
  {
    if (image.empty())
    {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.push_back(Entry{timestamp, image});
    while (buffer_.size() > max_size_)
    {
      buffer_.pop_front();
    }
  }

  bool ColorImageBuffer::lookup(double timestamp, double slop_s, cv::Mat *out) const
  {
    if (out == nullptr)
    {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty())
    {
      return false;
    }

    const Entry *best = nullptr;
    double best_dt = 0.0;
    for (const auto &entry : buffer_)
    {
      const double dt = std::abs(entry.timestamp - timestamp);
      if (best == nullptr || dt < best_dt)
      {
        best = &entry;
        best_dt = dt;
      }
    }
    if (best == nullptr || best_dt > slop_s)
    {
      return false;
    }
    *out = best->image.clone();
    return true;
  }

} // namespace vslam
