/*
 * @Description: Application class implementation
 * @Author: che yifan
 * @Date: 2026-07-26 09:39:40
 * @LastEditTime: 2026-08-16
 * @LastEditors: che yifan
 * @Reference:
 */
#include "application.h"
#include "keyframe_select/keyframe_select.h"

#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

namespace vslam
{

  Application::Application(const std::string &config_path) : config_path_(config_path)
  {
    initOpenVINS(config_path);

    if (!vioInitialized())
    {
      PRINT_ERROR("[APP]: OpenVINS initialization failed!\n");
      return;
    }

    PRINT_INFO("[APP]: OpenVINS initialized successfully.\n");
    initKeyframeSelect();
    initMapManager();
  }

  bool Application::vioInitialized() const
  {
    return vio_manager_ != nullptr;
  }

  std::shared_ptr<ov_msckf::VioManager> Application::getVioManager() const
  {
    return vio_manager_;
  }

  std::shared_ptr<KeyframeSelect> Application::getKeyframeSelect() const
  {
    return keyframe_select_;
  }

  std::shared_ptr<KeyframeQueue> Application::getKeyframeQueue() const
  {
    return keyframe_select_ ? keyframe_select_->getKeyframeQueue() : nullptr;
  }

  const KeyframeSelectConfig &Application::getKeyframeSelectConfig() const
  {
    return kf_config_;
  }

  std::shared_ptr<MapManager> Application::getMapManager() const
  {
    return map_manager_;
  }

  const MapManagerConfig &Application::getMapManagerConfig() const
  {
    return map_config_;
  }

  std::string Application::resolveVslamConfigPath(const std::string &estimator_config_path)
  {
    const auto pos = estimator_config_path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
      return "vslam.yaml";
    }
    return estimator_config_path.substr(0, pos + 1) + "vslam.yaml";
  }

  bool Application::loadKeyframeSelectConfig(KeyframeSelectConfig &kf_cfg) const
  {
    const std::string vslam_yaml = resolveVslamConfigPath(config_path_);
    PRINT_INFO("[APP]: Loading VSLAM config from: %s\n", vslam_yaml.c_str());

    auto parser = std::make_shared<ov_core::YamlParser>(vslam_yaml);
    if (!parser->successful())
    {
      PRINT_ERROR(RED "[APP]: Failed to open vslam.yaml: %s\n" RESET, vslam_yaml.c_str());
      return false;
    }

    parser->parse_config("kf_dist_thresh", kf_cfg.dist_thresh, false);
    parser->parse_config("kf_angle_thresh_deg", kf_cfg.angle_thresh_deg, false);
    parser->parse_config("kf_sync_slop", kf_cfg.sync_slop, false);
    parser->parse_config("kf_pose_queue_size", kf_cfg.pose_queue_size, false);
    parser->parse_config("kf_image_queue_size", kf_cfg.image_queue_size, false);
    parser->parse_config("kf_output_queue_size", kf_cfg.output_queue_size, false);
    parser->parse_config("kf_topic_pose", kf_cfg.topic_pose, false);
    parser->parse_config("kf_topic_keyframe_pose", kf_cfg.topic_keyframe_pose, false);
    parser->parse_config("kf_topic_keyframe_image", kf_cfg.topic_keyframe_image, false);
    parser->parse_config("kf_publish_debug", kf_cfg.publish_debug, false);

    if (!parser->successful())
    {
      PRINT_ERROR(RED "[APP]: Failed to parse parameters from vslam.yaml!\n" RESET);
      return false;
    }

    return true;
  }

  bool Application::loadCameraParams(std::vector<CameraParams> &cameras) const
  {
    auto parser = std::make_shared<ov_core::YamlParser>(config_path_);
    if (!parser->successful())
    {
      PRINT_ERROR(RED "[APP]: Failed to open estimator config for cameras: %s\n" RESET,
                  config_path_.c_str());
      return false;
    }

    int max_cameras = 2;
    parser->parse_config("max_cameras", max_cameras, false);

    cameras.clear();
    cameras.reserve(static_cast<std::size_t>(max_cameras));

    for (int i = 0; i < max_cameras; ++i)
    {
      const std::string cam_name = "cam" + std::to_string(i);
      CameraParams cp;

      std::vector<double> intrinsics = {cp.fx, cp.fy, cp.cx, cp.cy};
      std::vector<int> resolution = {cp.width, cp.height};
      std::vector<double> distortion = {0, 0, 0, 0};

      parser->parse_external("relative_config_imucam", cam_name, "intrinsics", intrinsics, false);
      parser->parse_external("relative_config_imucam", cam_name, "resolution", resolution, false);
      parser->parse_external("relative_config_imucam", cam_name, "distortion_model",
                             cp.distortion_model, false);
      parser->parse_external("relative_config_imucam", cam_name, "distortion_coeffs", distortion,
                             false);

      if (intrinsics.size() >= 4)
      {
        cp.fx = intrinsics[0];
        cp.fy = intrinsics[1];
        cp.cx = intrinsics[2];
        cp.cy = intrinsics[3];
      }
      if (resolution.size() >= 2)
      {
        cp.width = resolution[0];
        cp.height = resolution[1];
      }
      if (distortion.size() >= 4)
      {
        cp.k1 = distortion[0];
        cp.k2 = distortion[1];
        cp.k3 = distortion[2];
        cp.k4 = distortion[3];
      }

      cameras.push_back(cp);
    }

    return !cameras.empty();
  }

  bool Application::loadMapManagerConfig(MapManagerConfig &map_cfg) const
  {
    const std::string vslam_yaml = resolveVslamConfigPath(config_path_);
    auto parser = std::make_shared<ov_core::YamlParser>(vslam_yaml);
    if (!parser->successful())
    {
      PRINT_ERROR(RED "[APP]: Failed to open vslam.yaml for MapManager: %s\n" RESET,
                  vslam_yaml.c_str());
      return false;
    }

    parser->parse_config("map_root_dir", map_cfg.map.map_root_dir, false);
    parser->parse_config("map_clear_old_session", map_cfg.map.clear_old_session, false);
    parser->parse_config("map_save_images", map_cfg.map.save_images, false);
    parser->parse_config("map_image_extension", map_cfg.map.image_extension, false);

    if (!parser->successful())
    {
      PRINT_ERROR(RED "[APP]: Failed to parse MapManager parameters from vslam.yaml!\n" RESET);
      return false;
    }

    if (!loadCameraParams(map_cfg.cameras))
    {
      PRINT_ERROR(RED "[APP]: Failed to load camera params for Map.\n" RESET);
      return false;
    }

    return true;
  }

  void Application::initKeyframeSelect()
  {
    if (!loadKeyframeSelectConfig(kf_config_))
    {
      PRINT_ERROR(RED "[APP]: KeyframeSelect will use in-code defaults (vslam.yaml load failed).\n" RESET);
    }

    keyframe_select_ = std::make_shared<KeyframeSelect>(kf_config_);
    PRINT_INFO("[APP]: KeyframeSelect module created.\n");
  }

  void Application::initMapManager()
  {
    if (!loadMapManagerConfig(map_config_))
    {
      PRINT_ERROR(RED "[APP]: MapManager will use in-code defaults (vslam.yaml load failed).\n" RESET);
      // Still try to load cameras if yaml partially failed.
      if (map_config_.cameras.empty())
      {
        loadCameraParams(map_config_.cameras);
      }
    }

    map_manager_ = std::make_shared<MapManager>(map_config_);
    if (!map_manager_->createMap())
    {
      PRINT_ERROR(RED "[APP]: MapManager createMap() failed!\n" RESET);
      map_manager_.reset();
      return;
    }

    auto kf_queue = getKeyframeQueue();
    if (kf_queue)
    {
      map_manager_->startKeyframeConsumer(kf_queue);
    }
    else
    {
      PRINT_ERROR(RED "[APP]: KeyframeQueue is null, MapManager consumer not started.\n" RESET);
    }

    PRINT_INFO("[APP]: MapManager module created.\n");
  }

  void Application::initOpenVINS(const std::string &config_path)
  {
    PRINT_INFO("[APP]: Loading OpenVINS configuration from: %s\n", config_path.c_str());
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);

    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity, false);
    ov_core::Printer::setPrintLevel(verbosity);

    ov_msckf::VioManagerOptions params;
    params.use_multi_threading_subs = true;
    params.print_and_load(parser);

    if (!parser->successful())
    {
      PRINT_ERROR(RED "[APP]: Failed to parse all parameters from config file!\n" RESET);
      return;
    }

    vio_manager_ = std::make_shared<ov_msckf::VioManager>(params);
    PRINT_INFO("[APP]: OpenVINS VioManager created.\n");
  }

} // namespace vslam
