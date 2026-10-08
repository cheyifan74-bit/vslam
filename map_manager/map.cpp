/*
 * @Description: Map session implementation (directories + COLMAP database).
 * @Author: che yifan
 * @Date: 2026-08-16
 */
#include "map_manager/map.h"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <boost/filesystem.hpp>

#include "colmap/scene/camera.h"
#include "colmap/scene/database_session.h"
#include "colmap/sensor/models.h"

#include "utils/print.h"

namespace fs = boost::filesystem;

namespace vslam
{

  Map::Map(const MapConfig &config, std::vector<CameraParams> cameras)
      : config_(config), cameras_(std::move(cameras))
  {
    map_dir_ = (fs::path(config_.map_root_dir) / "map").string();
    database_dir_ = (fs::path(map_dir_) / "database").string();
    database_path_ = (fs::path(database_dir_) / "database.db").string();
    images_dir_ = (fs::path(map_dir_) / "images").string();
    images_left_dir_ = (fs::path(images_dir_) / "left_image").string();
    images_right_dir_ = (fs::path(images_dir_) / "right_image").string();
    masks_left_dir_ = (fs::path(map_dir_) / "masks" / "left_image").string();
    masks_right_dir_ = (fs::path(map_dir_) / "masks" / "right_image").string();
    sparse_dir_ = (fs::path(map_dir_) / "sparse").string();
  }

  bool Map::initialize()
  {
    ready_ = false;
    if (config_.clear_old_session && !clearOldSession())
    {
      return false;
    }
    if (!initDirectories())
    {
      return false;
    }
    if (!initDatabase())
    {
      return false;
    }
    ready_ = true;
    PRINT_INFO("[MAP]: Session ready. root=%s db=%s\n", config_.map_root_dir.c_str(),
               database_path_.c_str());
    return true;
  }

  bool Map::clearOldSession() const
  {
    try
    {
      if (fs::exists(database_path_))
      {
        fs::remove(database_path_);
        PRINT_INFO("[MAP]: Removed old database\n");
      }
      for (const char *ext : {"-wal", "-shm"})
      {
        const fs::path side = database_path_ + ext;
        if (fs::exists(side))
        {
          fs::remove(side);
        }
      }
      if (fs::exists(sparse_dir_))
      {
        fs::remove_all(sparse_dir_);
        PRINT_INFO("[MAP]: Removed old sparse dir\n");
      }
    }
    catch (const fs::filesystem_error &e)
    {
      PRINT_ERROR("[MAP]: clearOldSession failed: %s\n", e.what());
      return false;
    }
    return true;
  }

  bool Map::initDirectories() const
  {
    const std::vector<std::string> dirs = {
        config_.map_root_dir,
        map_dir_,
        database_dir_,
        images_dir_,
        images_left_dir_,
        images_right_dir_,
        masks_left_dir_,
        masks_right_dir_,
        sparse_dir_,
    };

    try
    {
      for (const auto &d : dirs)
      {
        if (!fs::exists(d))
        {
          fs::create_directories(d);
          PRINT_INFO("[MAP]: created: %s\n", d.c_str());
        }
      }
    }
    catch (const fs::filesystem_error &e)
    {
      PRINT_ERROR("[MAP]: initDirectories failed: %s\n", e.what());
      return false;
    }
    return true;
  }

  std::string Map::colmapModelName(const CameraParams &cam)
  {
    if (cam.distortion_model == "radtan")
    {
      return "OPENCV";
    }
    if (cam.distortion_model == "equidistant")
    {
      return "OPENCV_FISHEYE";
    }
    return "PINHOLE";
  }

  std::vector<double> Map::colmapParams(const CameraParams &cam)
  {
    if (cam.distortion_model == "radtan" || cam.distortion_model == "equidistant")
    {
      return {cam.fx, cam.fy, cam.cx, cam.cy, cam.k1, cam.k2, cam.k3, cam.k4};
    }
    return {cam.fx, cam.fy, cam.cx, cam.cy};
  }

  bool Map::initDatabase() const
  {
    try
    {
      colmap::DatabaseSession db(database_path_);

      for (std::size_t i = 0; i < cameras_.size(); ++i)
      {
        const auto &p = cameras_[i];
        const auto model_name = colmapModelName(p);
        const colmap::camera_t cam_id = static_cast<colmap::camera_t>(i + 1);

        if (db->ExistsCamera(cam_id))
        {
          continue;
        }

        colmap::Camera cam;
        cam.camera_id = cam_id;
        cam.model_id = colmap::CameraModelNameToId(model_name);
        cam.width = static_cast<size_t>(p.width);
        cam.height = static_cast<size_t>(p.height);
        cam.params = colmapParams(p);
        cam.has_prior_focal_length = true;

        db->WriteCamera(cam, /*use_camera_id=*/true);
        PRINT_INFO("[MAP]: camera %u: %dx%d fx=%.2f model=%s\n",
                   static_cast<unsigned>(cam_id), p.width, p.height, p.fx, model_name.c_str());
      }
    }
    catch (const std::exception &e)
    {
      PRINT_ERROR("[MAP]: initDatabase failed: %s\n", e.what());
      return false;
    }

    PRINT_INFO("[MAP]: database ready: %s\n", database_path_.c_str());
    return true;
  }

  bool Map::isReady() const { return ready_; }

  const MapConfig &Map::config() const { return config_; }

  const std::vector<CameraParams> &Map::cameras() const { return cameras_; }

  std::string Map::mapRootDir() const { return config_.map_root_dir; }
  std::string Map::mapDir() const { return map_dir_; }
  std::string Map::databaseDir() const { return database_dir_; }
  std::string Map::databasePath() const { return database_path_; }
  std::string Map::imagesDir() const { return images_dir_; }
  std::string Map::imagesLeftDir() const { return images_left_dir_; }
  std::string Map::imagesRightDir() const { return images_right_dir_; }
  std::string Map::masksLeftDir() const { return masks_left_dir_; }
  std::string Map::masksRightDir() const { return masks_right_dir_; }
  std::string Map::sparseDir() const { return sparse_dir_; }

  std::string Map::makeImageFileName(std::size_t frame_index, double timestamp_sec) const
  {
    const auto ts_ms = static_cast<std::int64_t>(timestamp_sec * 1000.0);
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << frame_index << "_" << ts_ms << "."
        << config_.image_extension;
    return oss.str();
  }

  std::string Map::makeImageAbsPath(bool left, const std::string &file_name) const
  {
    return (fs::path(left ? images_left_dir_ : images_right_dir_) / file_name).string();
  }

  std::string Map::makeImageRelativeName(bool left, const std::string &file_name) const
  {
    return std::string(left ? "left_image/" : "right_image/") + file_name;
  }

} // namespace vslam
