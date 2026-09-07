/*
 * @Description: MapManager implementation.
 * @Author: che yifan
 * @Date: 2026-08-16
 */
#include "map_manager/map_manager.h"

#include <chrono>
#include <exception>
#include <optional>

#include <opencv2/imgcodecs.hpp>

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/database_session.h"
#include "colmap/scene/image.h"
#include "colmap/sfm/online_incremental_mapper.h"
#include "colmap/util/types.h"

#include "utility/math_utils.h"
#include "utils/print.h"

namespace vslam
{

  MapManager::MapManager(const MapManagerConfig &config) : config_(config) {}

  MapManager::~MapManager() { stopKeyframeConsumer(); }

  bool MapManager::createMap()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    incremental_mapper_.reset();
    map_ = std::make_shared<Map>(config_.map, config_.cameras);
    if (!map_->initialize())
    {
      map_.reset();
      return false;
    }
    incremental_mapper_ =
        std::make_unique<colmap::OnlineIncrementalMapper>(map_->databasePath());
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

  uint32_t MapManager::writeKeyframeImage(const Map &map,
                                          const Keyframe &keyframe,
                                          const std::string &relative_name) const
  {
    try
    {
      colmap::DatabaseSession db(map.databasePath());
      if (db->ExistsImageWithName(relative_name))
      {
        const auto existing = db->ReadImageWithName(relative_name);
        if (!existing.has_value())
        {
          PRINT_ERROR("[MAP_MANAGER]: ExistsImageWithName but Read failed: %s\n",
                      relative_name.c_str());
          return 0;
        }
        return existing->ImageId();
      }

      const colmap::camera_t camera_id =
          static_cast<colmap::camera_t>(keyframe.cam_id + 1);
      if (!db->ExistsCamera(camera_id))
      {
        PRINT_ERROR("[MAP_MANAGER]: camera_id=%u not in database\n",
                    static_cast<unsigned>(camera_id));
        return 0;
      }

      colmap::Image image;
      image.SetName(relative_name);
      image.SetCameraId(camera_id);
      const colmap::image_t image_id = db->WriteImage(image);
      PRINT_INFO("[MAP_MANAGER]: WriteImage name=%s image_id=%u camera_id=%u\n",
                 relative_name.c_str(),
                 static_cast<unsigned>(image_id),
                 static_cast<unsigned>(camera_id));
      return image_id;
    }
    catch (const std::exception &e)
    {
      PRINT_ERROR("[MAP_MANAGER]: WriteImage failed: %s\n", e.what());
      return 0;
    }
  }

  bool MapManager::addKeyframe(const Keyframe &keyframe)
  {
    std::shared_ptr<Map> map;
    colmap::OnlineIncrementalMapper *mapper = nullptr;
    std::size_t frame_index = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      map = map_;
      mapper = incremental_mapper_.get();
      if (!map || !map->isReady() || mapper == nullptr)
      {
        PRINT_ERROR("[MAP_MANAGER]: addKeyframe failed, map/mapper is not ready.\n");
        return false;
      }
      frame_index = frame_count_;
    }

    const std::string file_name = map->makeImageFileName(frame_index, keyframe.timestamp);
    const std::string abs_path = map->makeImageAbsPath(/*left=*/true, file_name);
    const std::string relative_name = map->makeImageRelativeName(/*left=*/true, file_name);

    if (map->config().save_images)
    {
      if (!saveKeyframeImage(keyframe, abs_path))
      {
        return false;
      }
    }

    const uint32_t image_id = writeKeyframeImage(*map, keyframe, relative_name);
    if (image_id == 0)
    {
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++frame_count_;
    }

    const std::optional<colmap::Rigid3d> vio_prior = vioCamFromWorld(keyframe.pose);
    if (!mapper->Process(image_id, abs_path, vio_prior))
    {
      PRINT_WARNING("[MAP_MANAGER]: Process failed for image_id=%u (image is in DB)\n",
                    static_cast<unsigned>(image_id));
    }

    PRINT_INFO("[MAP_MANAGER]: Saved keyframe id=%llu -> %s image_id=%u (total=%zu)\n",
               static_cast<unsigned long long>(keyframe.id),
               relative_name.c_str(),
               static_cast<unsigned>(image_id),
               frame_index + 1);
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
