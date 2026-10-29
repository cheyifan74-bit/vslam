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
#include <functional>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "keyframe_select/keyframe.h"
#include "keyframe_select/keyframe_queue.h"
#include "map_manager/map.h"

namespace colmap
{
  class OnlineIncrementalMapper;
}

namespace vslam
{

  struct MapVizPoint {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    std::uint64_t id = 0;
    std::uint32_t observations = 0;
  };

  struct MapVizKeyframe {
    std::uint32_t id = 0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  };

  struct MapVizLoop {
    std::uint32_t current_id = 0;
    std::uint32_t candidate_id = 0;
  };

  struct MapVisualizationSnapshot {
    std::uint32_t current_keyframe_id = 0;
    std::vector<MapVizPoint> global_points;
    std::vector<MapVizPoint> local_points;
    std::vector<MapVizKeyframe> keyframes;
    std::vector<MapVizLoop> confirmed_loops;
  };

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

  struct MatchConfig
  {
    bool use_gpu = false;
    std::string gpu_index = "-1";
    int overlap = 10;
    double spatial_max_distance = 10.0;
    double spatial_max_angle_deg = 60.0;
    int spatial_max_num_images = 10;
    int num_threads = -1;
    int max_num_matches = 2048;
    double max_ratio = 0.8;
    double max_distance = 0.7;
    bool cross_check = true;
    bool cpu_brute_force_matcher = false;
    int min_num_inliers = 15;
    double max_error = 4.0;
    bool e_only = true;
    bool detect_watermark = false;
    bool use_degensac = false;
    int ransac_min_num_trials = 30;
    int ransac_max_num_trials = 500;
    double ransac_confidence = 0.99;
  };

  struct MapperConfig
  {
    int min_num_inliers = 15;
    int abs_pose_min_num_inliers = 50;
    double abs_pose_max_error = 1.0;
    double abs_pose_min_inlier_ratio = 0.25;
    int ba_local_num_images = 0;
    int ba_min_covisibility_points = 15;
    double filter_max_reproj_error = 1.0;
    double filter_min_tri_angle = 1.5;
    int tri_max_transitivity = 1;
    double tri_create_max_angle_error = 2.0;
    double tri_continue_max_angle_error = 2.0;
    double tri_merge_max_reproj_error = 4.0;
    double tri_complete_max_reproj_error = 4.0;
    int tri_complete_max_transitivity = 5;
    double tri_min_angle = 1.5;
    bool tri_ignore_two_view_tracks = false;
    int local_map_num_images = 15;
    double local_map_search_radius_px = 15.0;
    double local_map_refine_radius_px = 4.0;
    double local_map_max_desc_distance = 0.7;
    double local_map_max_ratio = 0.9;
    double ba_vio_prior_rot_sigma_deg = 1.0;
    double ba_vio_prior_trans_sigma_m = 0.02;
    double ba_vio_prior_trans_sigma_ratio = 0.03;
    int ba_vio_prior_max_gap = 5;
    int ba_min_window_images = 4;
    int ba_temporal_window_images = 5;
    int sparse_write_every_n = 10;
  };

  struct LoopCloserConfig
  {
    bool enabled = true;
    std::string mixvpr_engine_path;
    bool mixvpr_use_gpu = true;
    std::string mixvpr_gpu_index = "-1";
    int cooldown_num_images = 10;
    int min_frame_gap = 20;
    double max_distance = 10.0;
    double max_view_angle_deg = 60.0;
    double min_mixvpr_score = 0.4;
    int topk = 5;
    int min_covisibility_points = 15;
    int min_num_matches = 20;
    int min_num_verified_matches = 40;
    int min_num_3d_correspondences = 20;
    int min_num_3d_inliers = 20;
    bool correct_local = true;
    int max_correct_connected = 20;
    double pnp_max_error = 3.0;
    double guided_radius_px = 7.5;
    double max_rot_vs_essential_deg = 3.0;
    double max_correction_m = 3.0;
    double max_correction_deg = 20.0;
    int pose_graph_min_covis = 50;
    double fuse_radius_px = 6.0;
    double fuse_max_point_distance = 0.2;

    // Relocalization on a resumed map.
    int reloc_topk = 5;
    double reloc_min_mixvpr_score = 0.5;
    int reloc_min_num_matches = 30;
    int reloc_min_num_3d_correspondences = 40;
    int reloc_min_inliers = 80;
    double reloc_max_rot_vs_essential_deg = 3.0;
  };

  struct MapManagerConfig
  {
    MapConfig map;
    std::vector<CameraParams> cameras;
    SiftExtractConfig sift;
    MatchConfig match;
    MapperConfig mapper;
    LoopCloserConfig loop;
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
    /// Installs the visualization sink and immediately publishes the map that is
    /// currently loaded, so a resumed session shows the existing keyframes and
    /// points in RViz before the first keyframe of this session is processed.
    void setVisualizationCallback(
        std::function<void(const MapVisualizationSnapshot &)> callback);

    /// Publish the current reconstruction (restored map + every processed
    /// keyframe) to the visualization callback. No-op when no callback is set or
    /// the map is still empty.
    void publishVisualization();

  private:
    void keyframeConsumerLoop();
    /// Next free image index parsed from the database image names, so a resumed
    /// session never reuses an existing name (and therefore an image_id).
    std::size_t nextFrameIndexFromDatabase(const Map &map) const;
    MapVisualizationSnapshot makeVisualizationSnapshot(
        const colmap::OnlineIncrementalMapper &mapper,
        uint32_t current_image_id) const;
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
    /// Guards the mapper/reconstruction: Process() runs on the consumer thread
    /// while publishVisualization() can be triggered from the ROS thread.
    std::mutex mapper_mutex_;
    /// Last image handed to the mapper, used as the "current" keyframe marker.
    /// Only touched while holding mapper_mutex_.
    uint32_t last_processed_image_id_ = 0;
    std::function<void(const MapVisualizationSnapshot &)> visualization_callback_;
  };

} // namespace vslam

#endif // VSLAM_MAP_MANAGER_MAP_MANAGER_H
