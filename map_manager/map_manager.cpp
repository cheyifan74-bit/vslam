/*
 * @Description: MapManager implementation.
 * @Author: che yifan
 * @Date: 2026-08-16
 */
#include "map_manager/map_manager.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <optional>
#include <string>
#include <unordered_set>

#include <opencv2/imgcodecs.hpp>

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/database_session.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/online_sfm/online_incremental_mapper.h"
#include "colmap/util/types.h"

#include "map_manager/colmap_options.h"
#include "utility/math_utils.h"
#include "utils/print.h"

namespace vslam
{

  MapManager::MapManager(const MapManagerConfig &config) : config_(config) {}

  MapManager::~MapManager()
  {
    stopKeyframeConsumer();
    // Flush the frame(s) that sparse_write_every_n deferred, otherwise the
    // next session resumes from a map missing the tail of this one.
    if (incremental_mapper_)
    {
      incremental_mapper_->FlushSparse();
    }
  }

  std::size_t MapManager::nextFrameIndexFromDatabase(const Map &map) const
  {
    std::size_t next_index = 0;
    try
    {
      colmap::DatabaseSession db(map.databasePath());
      for (const auto &image : db->ReadAllImages())
      {
        const std::string &name = image.Name();
        const auto slash = name.find_last_of('/');
        const std::string base = slash == std::string::npos ? name : name.substr(slash + 1);
        const auto underscore = base.find('_');
        if (underscore == std::string::npos || underscore == 0)
        {
          continue;
        }
        try
        {
          const auto index = static_cast<std::size_t>(std::stoul(base.substr(0, underscore)));
          next_index = std::max(next_index, index + 1);
        }
        catch (const std::exception &)
        {
          continue;
        }
      }
    }
    catch (const std::exception &e)
    {
      PRINT_WARNING("[MAP_MANAGER]: could not read image names from database: %s\n",
                    e.what());
    }
    return next_index;
  }

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
    incremental_mapper_ = std::make_unique<colmap::OnlineIncrementalMapper>(
        map_->databasePath(),
        ToColmap(config_.sift),
        ToColmap(config_.match),
        map_->sparseDir(),
        ToColmap(config_.mapper),
        ToColmap(config_.loop));
    if (map_->isResuming())
    {
      if (!incremental_mapper_->ResumeFromDisk())
      {
        PRINT_ERROR("[MAP_MANAGER]: ResumeFromDisk failed, keeping the map on "
                    "disk untouched. Set map_resume_existing=false to start a "
                    "new session.\n");
        incremental_mapper_.reset();
        map_.reset();
        return false;
      }
      {
        // Highlight the last restored keyframe as "current" so the loaded map is
        // anchored in the view before this session registers anything.
        std::lock_guard<std::mutex> mapper_lock(mapper_mutex_);
        const std::vector<colmap::image_t> reg_ids =
            incremental_mapper_->GetReconstruction().RegImageIds();
        if (!reg_ids.empty())
        {
          last_processed_image_id_ = reg_ids.back();
        }
      }
    }
    // Image names are index_timestamp.ext. Continue the numbering of whatever
    // rows are already in the database, otherwise a saved image name (and with
    // it its image_id) would be reused and overwritten.
    frame_count_ = nextFrameIndexFromDatabase(*map_);
    if (map_->isResuming())
    {
      PRINT_INFO("[MAP_MANAGER]: Resumed map, next frame index=%zu\n",
                 frame_count_);
    }
    else if (frame_count_ > 0)
    {
      PRINT_WARNING("[MAP_MANAGER]: Database already holds images, continuing "
                    "frame numbering at %zu\n",
                    frame_count_);
    }
    PRINT_INFO("[MAP_MANAGER]: Map session created. SIFT use_gpu=%d gpu_index=%s "
               "max_num_features=%d peak=%.6f | match use_gpu=%d gpu_index=%s "
               "overlap=%d spatial=(%.1fm,%.1fdeg,n=%d) min_inliers=%d e_only=%d "
               "degensac=%d watermark=%d "
               "ransac_trials=[%d,%d] num_threads=%d cpu_brute_force=%d | "
               "mapper p3p_inliers=%d p3p_err=%.2f covis=%d "
               "filter_err=%.2f tri_angle=%.2f "
               "ignore_2view=%d | loop enabled=%d cooldown=%d max_dist=%.1f "
               "mixvpr=%s gpu=%d gpu_index=%s\n",
               static_cast<int>(config_.sift.use_gpu),
               config_.sift.gpu_index.c_str(),
               config_.sift.max_num_features,
               config_.sift.peak_threshold,
               static_cast<int>(config_.match.use_gpu),
               config_.match.gpu_index.c_str(),
               config_.match.overlap,
               config_.match.spatial_max_distance,
               config_.match.spatial_max_angle_deg,
               config_.match.spatial_max_num_images,
               config_.match.min_num_inliers,
               static_cast<int>(config_.match.e_only),
               static_cast<int>(config_.match.use_degensac),
               static_cast<int>(config_.match.detect_watermark),
               config_.match.ransac_min_num_trials,
               config_.match.ransac_max_num_trials,
               config_.match.num_threads,
               static_cast<int>(config_.match.cpu_brute_force_matcher),
               config_.mapper.abs_pose_min_num_inliers,
               config_.mapper.abs_pose_max_error,
               config_.mapper.ba_min_covisibility_points,
               config_.mapper.filter_max_reproj_error,
               config_.mapper.tri_min_angle,
               static_cast<int>(config_.mapper.tri_ignore_two_view_tracks),
               static_cast<int>(config_.loop.enabled),
               config_.loop.cooldown_num_images,
               config_.loop.max_distance,
               config_.loop.mixvpr_engine_path.c_str(),
               static_cast<int>(config_.loop.mixvpr_use_gpu),
               config_.loop.mixvpr_gpu_index.c_str());
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
    bool processed = false;
    {
      std::lock_guard<std::mutex> mapper_lock(mapper_mutex_);
      processed = mapper->Process(image_id, abs_path, vio_prior);
      last_processed_image_id_ = image_id;
    }
    if (!processed)
    {
      PRINT_WARNING("[MAP_MANAGER]: Process failed for image_id=%u (image is in DB)\n",
                    static_cast<unsigned>(image_id));
    }
    // Publish for every keyframe, not only the successfully mapped ones: while a
    // resumed session is still trying to relocalize, Process() returns true but
    // adds nothing, and the restored map must stay visible (and keep being
    // refreshed) until relocalization succeeds and incremental mapping resumes.
    publishVisualization();

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


  MapVisualizationSnapshot MapManager::makeVisualizationSnapshot(
      const colmap::OnlineIncrementalMapper &mapper,
      uint32_t current_image_id) const
  {
    MapVisualizationSnapshot snapshot;
    snapshot.current_keyframe_id = current_image_id;
    const colmap::Reconstruction &reconstruction = mapper.GetReconstruction();
    std::unordered_set<colmap::point3D_t> local_point_ids;
    std::unordered_set<colmap::image_t> local_image_ids;
    local_image_ids.insert(current_image_id);

    if (reconstruction.ExistsImage(current_image_id) &&
        reconstruction.Image(current_image_id).HasPose())
    {
      const auto &current = reconstruction.Image(current_image_id);
      for (const auto &point2D : current.Points2D())
      {
        if (!point2D.HasPoint3D() || !reconstruction.ExistsPoint3D(point2D.point3D_id))
          continue;
        const auto &point = reconstruction.Point3D(point2D.point3D_id);
        local_point_ids.insert(point2D.point3D_id);
        for (const auto &track_element : point.track.Elements())
          local_image_ids.insert(track_element.image_id);
      }
    }

    for (const auto image_id : local_image_ids)
    {
      if (!reconstruction.ExistsImage(image_id))
        continue;
      const auto &image = reconstruction.Image(image_id);
      for (const auto &point2D : image.Points2D())
        if (point2D.HasPoint3D() && reconstruction.ExistsPoint3D(point2D.point3D_id))
          local_point_ids.insert(point2D.point3D_id);
    }

    for (const auto image_id : reconstruction.RegImageIds())
    {
      const auto &image = reconstruction.Image(image_id);
      if (!image.HasPose())
        continue;
      const colmap::Rigid3d world_from_cam = colmap::Inverse(image.CamFromWorld());
      MapVizKeyframe keyframe;
      keyframe.id = image_id;
      keyframe.position = world_from_cam.translation();
      keyframe.orientation = world_from_cam.rotation();
      snapshot.keyframes.push_back(keyframe);
    }

    for (const auto point_id : reconstruction.Point3DIds())
    {
      const auto &point = reconstruction.Point3D(point_id);
      MapVizPoint viz_point;
      viz_point.position = point.xyz;
      viz_point.id = point_id;
      viz_point.observations = static_cast<std::uint32_t>(point.track.Length());
      if (local_point_ids.count(point_id))
        snapshot.local_points.push_back(viz_point);
      else
        snapshot.global_points.push_back(viz_point);
    }

    for (const auto &loop : mapper.ConfirmedLoops())
      snapshot.confirmed_loops.push_back({loop.image_id1, loop.image_id2});
    return snapshot;
  }

  void MapManager::setVisualizationCallback(
      std::function<void(const MapVisualizationSnapshot &)> callback)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      visualization_callback_ = std::move(callback);
    }
    // A resumed session already holds the whole previous map in the
    // reconstruction. Push it right away: otherwise RViz stays empty until the
    // first keyframe of this session is processed, which only happens after VIO
    // initializes and (on resume) after relocalization succeeds.
    publishVisualization();
  }

  void MapManager::publishVisualization()
  {
    std::function<void(const MapVisualizationSnapshot &)> callback;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      callback = visualization_callback_;
    }
    if (!callback)
    {
      return;
    }

    MapVisualizationSnapshot snapshot;
    {
      std::lock_guard<std::mutex> mapper_lock(mapper_mutex_);
      if (!incremental_mapper_)
      {
        return;
      }
      snapshot = makeVisualizationSnapshot(*incremental_mapper_,
                                           last_processed_image_id_);
    }

    // Do not publish an empty snapshot: its first marker is a DELETEALL, so it
    // would wipe whatever RViz is currently showing (e.g. a restored map while
    // the first frame of this session is still being relocalized).
    if (snapshot.keyframes.empty() && snapshot.global_points.empty() &&
        snapshot.local_points.empty())
    {
      return;
    }
    callback(snapshot);
  }

  std::size_t MapManager::frameCount() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return frame_count_;
  }

} // namespace vslam
