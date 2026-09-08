/*
 * @Description: MapManager — create Map session, ingest keyframes, save images.
 * @Author: che yifan
 * @Date: 2026-08-16
 */
#ifndef VSLAM_MAP_MANAGER_MAP_MANAGER_H
#define VSLAM_MAP_MANAGER_MAP_MANAGER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "keyframe_select/keyframe.h"
#include "keyframe_select/keyframe_queue.h"
#include "map_manager/map.h"

namespace colmap
{
  class OnlineIncrementalMapper;
}

namespace vslam
{

  struct SiftExtractConfig
  {
    bool use_gpu = false;
    std::string gpu_index = "-1";
    int max_num_features = 8192;
    int first_octave = -1;
    int num_octaves = 4;
    int octave_resolution = 3;
    double peak_threshold = 0.02 / 3.0;
    double edge_threshold = 10.0;
    bool estimate_affine_shape = false;
    int max_num_orientations = 2;
    bool upright = false;
    int max_image_size = -1;
  };

  struct MapManagerConfig
  {
    MapConfig map;
    std::vector<CameraParams> cameras;
    SiftExtractConfig sift;
  };

  /**
   * @brief Owns a Map session and consumes KeyframeQueue.
   *
   * Map owns directories + COLMAP database.
   * MapManager saves images, WriteImage, then calls OnlineIncrementalMapper::Process.
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

    /// Save left image, register it in database.db, then Process() the mapper.
    bool addKeyframe(const Keyframe &keyframe);

    void startKeyframeConsumer(const std::shared_ptr<KeyframeQueue> &keyframe_queue);
    void stopKeyframeConsumer();

    std::shared_ptr<Map> getMap() const;
    const MapManagerConfig &getConfig() const;
    std::size_t frameCount() const;

  private:
    void keyframeConsumerLoop();
    bool saveKeyframeImage(const Keyframe &keyframe, const std::string &abs_image_path) const;
    /// Open DB (path lock) → WriteImage → Close. Returns COLMAP image_id, or 0 on failure.
    uint32_t writeKeyframeImage(const Map &map,
                                const Keyframe &keyframe,
                                const std::string &relative_name) const;

    MapManagerConfig config_;
    std::shared_ptr<Map> map_;
    std::unique_ptr<colmap::OnlineIncrementalMapper> incremental_mapper_;
    std::size_t frame_count_ = 0;

    std::shared_ptr<KeyframeQueue> keyframe_queue_;
    std::atomic<bool> stop_consumer_{false};
    std::thread consumer_thread_;
    mutable std::mutex mutex_;
  };

} // namespace vslam

#endif // VSLAM_MAP_MANAGER_MAP_MANAGER_H
