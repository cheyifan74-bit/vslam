/*
 * @Description: ImageWriter — writes keyframe images to disk on a background thread,
 *               so PNG encoding and file IO stay off the mapping thread.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_MAP_MANAGER_IMAGE_WRITER_H
#define VSLAM_MAP_MANAGER_IMAGE_WRITER_H

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <opencv2/core.hpp>

namespace vslam
{

  class ImageWriter
  {
  public:
    ImageWriter();
    /// Writes everything still queued, then stops the thread.
    ~ImageWriter();

    ImageWriter(const ImageWriter &) = delete;
    ImageWriter &operator=(const ImageWriter &) = delete;

    /// Queues image for writing to path. The pixels are shared, not copied, so
    /// the caller must not modify image afterwards.
    void enqueue(const cv::Mat &image, const std::string &path);

    /// Blocks until every image queued so far is on disk.
    void flush();

  private:
    void run();

    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable idle_cv_;
    std::deque<std::pair<cv::Mat, std::string>> queue_;
    bool writing_ = false;
    bool stop_ = false;
    std::thread thread_;
  };

} // namespace vslam

#endif // VSLAM_MAP_MANAGER_IMAGE_WRITER_H
