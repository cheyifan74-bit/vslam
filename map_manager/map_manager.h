/*
 * @Description: MapManager — create Map session, ingest keyframes, save images.
 * @Author: che yifan
 * @Date: 2026-08-16
 */
#ifndef VSLAM_MAP_MANAGER_MAP_MANAGER_H
#define VSLAM_MAP_MANAGER_MAP_MANAGER_H

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "keyframe_select/keyframe.h"
#include "keyframe_select/keyframe_queue.h"
#include "map_manager/map.h"

namespace vslam
{

  struct MapManagerConfig
  {
    MapConfig map;
    std::vector<CameraParams> cameras;
  };

  /**
   * @brief Owns a Map session and consumes KeyframeQueue.
   *
   * Map owns directories + COLMAP database.
   * MapManager fills the session (currently: save left keyframe images).
   */
  class MapManager
  {
  public:
    explicit MapManager(const MapManagerConfig &config);
    ~MapManager();

    MapManager(const MapManager &) = delete;
    MapManager &operator=(const MapManager &) = delete;

    /// Create Map and initialize directories + database.
    bool createMap();

    /// Save keyframe image into Map session (left camera for now).
    bool addKeyframe(const Keyframe &keyframe);

    void startKeyframeConsumer(const std::shared_ptr<KeyframeQueue> &keyframe_queue);
    void stopKeyframeConsumer();

    std::shared_ptr<Map> getMap() const;
    const MapManagerConfig &getConfig() const;
    std::size_t frameCount() const;

  private:
    void keyframeConsumerLoop();
    bool saveKeyframeImage(const Keyframe &keyframe, const std::string &abs_image_path) const;

    MapManagerConfig config_;
    std::shared_ptr<Map> map_;
    std::size_t frame_count_ = 0;

    std::shared_ptr<KeyframeQueue> keyframe_queue_;
    std::atomic<bool> stop_consumer_{false};
    std::thread consumer_thread_;
    mutable std::mutex mutex_;
  };

} // namespace vslam

#endif // VSLAM_MAP_MANAGER_MAP_MANAGER_H
