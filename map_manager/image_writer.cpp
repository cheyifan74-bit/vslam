/*
 * @Description: ImageWriter implementation.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "map_manager/image_writer.h"

#include <opencv2/imgcodecs.hpp>

#include "utils/print.h"

namespace vslam
{

  ImageWriter::ImageWriter() : thread_(&ImageWriter::run, this) {}

  ImageWriter::~ImageWriter()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    work_cv_.notify_one();
    if (thread_.joinable())
    {
      thread_.join();
    }
  }

  void ImageWriter::enqueue(const cv::Mat &image, const std::string &path)
  {
    if (image.empty())
    {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.emplace_back(image, path);
    }
    work_cv_.notify_one();
  }

  void ImageWriter::flush()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_cv_.wait(lock, [this] { return queue_.empty() && !writing_; });
  }

  void ImageWriter::run()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true)
    {
      work_cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
      // Drain the queue before honoring stop_, so nothing queued is lost.
      if (queue_.empty())
      {
        return;
      }
      std::pair<cv::Mat, std::string> item = std::move(queue_.front());
      queue_.pop_front();
      writing_ = true;
      const std::size_t backlog = queue_.size();
      lock.unlock();

      try
      {
        if (!cv::imwrite(item.second, item.first))
        {
          PRINT_ERROR("[IMAGE_WRITER]: cv::imwrite failed: %s\n", item.second.c_str());
        }
      }
      catch (const cv::Exception &e)
      {
        PRINT_ERROR("[IMAGE_WRITER]: cv::imwrite exception for %s: %s\n", item.second.c_str(), e.what());
      }
      if (backlog >= 50 && backlog % 50 == 0)
      {
        PRINT_WARNING("[IMAGE_WRITER]: %zu images waiting to be written (disk too slow?)\n", backlog);
      }

      lock.lock();
      writing_ = false;
      if (queue_.empty())
      {
        idle_cv_.notify_all();
      }
    }
  }

} // namespace vslam
