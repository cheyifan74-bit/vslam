/*
 * @Description: MapManager implementation.
 * @Author: che yifan
 * @Date: 2026-08-16
 */
#include "map_manager/map_manager.h"

#include <chrono>

#include <opencv2/imgcodecs.hpp>

#include "utils/print.h"

namespace vslam
{

  MapManager::MapManager(const MapManagerConfig &config) : config_(config) {}

  MapManager::~MapManager() { stopKeyframeConsumer(); }

  bool MapManager::createMap()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    map_ = std::make_shared<Map>(config_.map, config_.cameras);
    if (!map_->initialize())
    {
      map_.reset();
      return false;
    }
    frame_count_ = 0;
    PRINT_INFO("[MAP_MANAGER]: Map session created.\n");
    return true;
  }

  bool MapManager::saveKeyframeImage(const Keyframe &keyframe,
                                     const std::string &abs_image_path) const
  {
    if (keyframe.image.empty())
    {
      PRINT_ERROR("[MAP_MANAGER]: Empty image for keyframe id=%llu\n",
                  static_cast<unsigned long long>(keyframe.id));
      return false;
    }

    try
    {
      if (!cv::imwrite(abs_image_path, keyframe.image))
      {
        PRINT_ERROR("[MAP_MANAGER]: cv::imwrite failed: %s\n", abs_image_path.c_str());
        return false;
      }
    }
    catch (const cv::Exception &e)
    {
      PRINT_ERROR("[MAP_MANAGER]: cv::imwrite exception: %s\n", e.what());
      return false;
    }
    return true;
  }

  bool MapManager::addKeyframe(const Keyframe &keyframe)
  {
    std::shared_ptr<Map> map;
    std::size_t frame_index = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      map = map_;
      if (!map || !map->isReady())
      {
        PRINT_ERROR("[MAP_MANAGER]: addKeyframe failed, map is not ready.\n");
        return false;
      }
      frame_index = frame_count_;
    }

    const std::string file_name = map->makeImageFileName(frame_index, keyframe.timestamp);
    const std::string abs_path = map->makeImageAbsPath(/*left=*/true, file_name);

    if (map->config().save_images)
    {
      if (!saveKeyframeImage(keyframe, abs_path))
      {
        return false;
      }
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++frame_count_;
    }

    PRINT_INFO("[MAP_MANAGER]: Saved keyframe id=%llu -> %s (total=%zu)\n",
               static_cast<unsigned long long>(keyframe.id),
               map->makeImageRelativeName(true, file_name).c_str(), frame_index + 1);
    return true;
  }

  void MapManager::startKeyframeConsumer(const std::shared_ptr<KeyframeQueue> &keyframe_queue)
  {
    if (!keyframe_queue)
    {
      PRINT_ERROR("[MAP_MANAGER]: startKeyframeConsumer failed, queue is null.\n");
      return;
    }
    if (consumer_thread_.joinable())
    {
      PRINT_WARNING("[MAP_MANAGER]: Consumer already running.\n");
      return;
    }

    keyframe_queue_ = keyframe_queue;
    stop_consumer_ = false;
    consumer_thread_ = std::thread(&MapManager::keyframeConsumerLoop, this);
    PRINT_INFO("[MAP_MANAGER]: Keyframe consumer thread started.\n");
  }

  void MapManager::stopKeyframeConsumer()
  {
    stop_consumer_ = true;
    if (consumer_thread_.joinable())
    {
      consumer_thread_.join();
    }
  }

  void MapManager::keyframeConsumerLoop()
  {
    while (!stop_consumer_)
    {
      Keyframe kf;
      if (!keyframe_queue_->waitPop(kf, std::chrono::milliseconds(100)))
      {
        continue;
      }
      addKeyframe(kf);
    }
    PRINT_INFO("[MAP_MANAGER]: Keyframe consumer thread stopped.\n");
  }

  std::shared_ptr<Map> MapManager::getMap() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return map_;
  }

  const MapManagerConfig &MapManager::getConfig() const { return config_; }

  std::size_t MapManager::frameCount() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return frame_count_;
  }

} // namespace vslam
