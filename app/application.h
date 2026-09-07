/*
 * @Description: Application class - manages module initialization and lifecycle
 * @Author: che yifan
 * @Date: 2026-07-26 09:39:40
 * @LastEditTime: 2026-08-16
 * @LastEditors: che yifan
 * @Reference:
 */
#ifndef VSLAM_APPLICATION_H
#define VSLAM_APPLICATION_H

#include <memory>
#include <string>

#include "core/VioManager.h"
#include "keyframe_select/keyframe.h"
#include "map_manager/map_manager.h"

namespace vslam
{

  class KeyframeSelect;
  class KeyframeQueue;

  class Application
  {
  public:
    /**
     * @brief Constructor: initializes OpenVINS / KeyframeSelect / MapManager
     * @param config_path Path to estimator_config.yaml (sibling vslam.yaml is also loaded)
     * @param verbosity If non-empty, overrides estimator_config.yaml verbosity
     */
    explicit Application(const std::string &config_path, const std::string &verbosity = "");

    /// Returns true if the OpenVINS VIO system has been successfully initialized
    bool vioInitialized() const;

    /// Accessor for the VIO manager
    std::shared_ptr<ov_msckf::VioManager> getVioManager() const;

    std::shared_ptr<KeyframeSelect> getKeyframeSelect() const;
    std::shared_ptr<KeyframeQueue> getKeyframeQueue() const;
    const KeyframeSelectConfig &getKeyframeSelectConfig() const;

    std::shared_ptr<MapManager> getMapManager() const;
    const MapManagerConfig &getMapManagerConfig() const;

  private:
    /// Initializes the OpenVINS VIO system
    void initOpenVINS(const std::string &config_path);

    /// Load vslam.yaml and create pure KeyframeSelect module
    void initKeyframeSelect();

    /// Load map config + cameras, create MapManager + Map, start keyframe consumer
    void initMapManager();

    /// Resolve sibling config path: <dir(estimator_config)>/vslam.yaml
    static std::string resolveVslamConfigPath(const std::string &estimator_config_path);

    /// Load KeyframeSelectConfig from vslam.yaml
    bool loadKeyframeSelectConfig(KeyframeSelectConfig &kf_cfg) const;

    /// Load MapManagerConfig from vslam.yaml + cameras from estimator_config
    bool loadMapManagerConfig(MapManagerConfig &map_cfg) const;

    /// Load camera intrinsics from estimator_config relative_config_imucam
    bool loadCameraParams(std::vector<CameraParams> &cameras) const;

    std::string config_path_; ///< path to estimator_config.yaml
    std::string verbosity_override_; ///< from launch/ROS param; empty → use yaml
    KeyframeSelectConfig kf_config_;
    MapManagerConfig map_config_;

    /// OpenVINS VIO manager instance
    std::shared_ptr<ov_msckf::VioManager> vio_manager_;

    /// Pure keyframe selection module (no ROS)
    std::shared_ptr<KeyframeSelect> keyframe_select_;

    /// Map management module
    std::shared_ptr<MapManager> map_manager_;
  };

} // namespace vslam

#endif // VSLAM_APPLICATION_H
