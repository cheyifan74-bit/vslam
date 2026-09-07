/*
 * @Description: Map — COLMAP-compatible map session (directories + database).
 * @Author: che yifan
 * @Date: 2026-08-16
 */
#ifndef VSLAM_MAP_MANAGER_MAP_H
#define VSLAM_MAP_MANAGER_MAP_H

#include <cstddef>
#include <string>
#include <vector>

namespace vslam
{

  /// Camera intrinsics used when seeding COLMAP cameras table.
  struct CameraParams
  {
    int width = 752;
    int height = 480;
    double fx = 366.0;
    double fy = 366.0;
    double cx = 376.0;
    double cy = 240.0;
    /// "radtan" | "equidistant" | "" (pinhole)
    std::string distortion_model;
    double k1 = 0.0;
    double k2 = 0.0;
    double k3 = 0.0;
    double k4 = 0.0;
  };

  struct MapConfig
  {
    /// Session root; COLMAP layout lives under <map_root_dir>/map/
    std::string map_root_dir = "/tmp/vslam_map";
    bool clear_old_session = true; ///< remove old database / sparse on init
    bool save_images = true;
    std::string image_extension = "png";
  };

  /**
   * @brief One mapping session skeleton.
   *
   * Responsibilities:
   *  - Create COLMAP-style directories (images / database / masks / sparse)
   *  - Create database.db and write camera models
   *  - Provide path / naming helpers for MapManager and future mappers
   *
   * Does NOT store unbounded in-memory keyframe history.
   */
  class Map
  {
  public:
    Map(const MapConfig &config, std::vector<CameraParams> cameras);
    ~Map() = default;

    Map(const Map &) = delete;
    Map &operator=(const Map &) = delete;

    /// clearOldSession (optional) + initDirectories + initDatabase
    bool initialize();

    bool isReady() const;
    const MapConfig &config() const;
    const std::vector<CameraParams> &cameras() const;

    std::string mapRootDir() const;
    std::string mapDir() const; ///< <root>/map
    std::string databaseDir() const;
    std::string databasePath() const; ///< .../database/database.db
    std::string imagesDir() const;
    std::string imagesLeftDir() const;
    std::string imagesRightDir() const;
    std::string masksLeftDir() const;
    std::string masksRightDir() const;
    std::string sparseDir() const;

    /// e.g. 000012_1712345678901.png
    std::string makeImageFileName(std::size_t frame_index, double timestamp_sec) const;

    /// Absolute path under left/right image folder.
    std::string makeImageAbsPath(bool left, const std::string &file_name) const;

    /// Relative COLMAP image name: left_image/xxx.png or right_image/xxx.png
    std::string makeImageRelativeName(bool left, const std::string &file_name) const;

  private:
    bool clearOldSession() const;
    bool initDirectories() const;
    bool initDatabase() const;
    static std::string colmapModelName(const CameraParams &cam);
    static std::vector<double> colmapParams(const CameraParams &cam);

    MapConfig config_;
    std::vector<CameraParams> cameras_;
    bool ready_ = false;

    std::string map_dir_;
    std::string database_dir_;
    std::string database_path_;
    std::string images_dir_;
    std::string images_left_dir_;
    std::string images_right_dir_;
    std::string masks_left_dir_;
    std::string masks_right_dir_;
    std::string sparse_dir_;
  };

} // namespace vslam

#endif // VSLAM_MAP_MANAGER_MAP_H
