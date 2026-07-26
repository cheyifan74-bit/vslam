/*
 * @Description: Application class - manages module initialization and lifecycle
 * @Author: che yifan
 * @Date: 2026-07-26 09:39:40
 * @LastEditTime: 2026-07-26 09:39:45
 * @LastEditors: che yifan
 * @Reference:
 */
#ifndef VSLAM_APPLICATION_H
#define VSLAM_APPLICATION_H

#include <memory>
#include <string>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"

namespace vslam
{

  class Application
  {
  public:
    /**
     * @brief Constructor: initializes OpenVINS from a YAML config file
     * @param config_path Path to the OpenVINS YAML configuration file
     */
    explicit Application(const std::string &config_path);

    /// Returns true if the OpenVINS VIO system has been successfully initialized
    bool vioInitialized() const;

    /// Accessor for the VIO manager
    std::shared_ptr<ov_msckf::VioManager> getVioManager() const;

  private:
    /// Initializes the OpenVINS VIO system
    void initOpenVINS(const std::string &config_path);

    /// OpenVINS VIO manager instance
    std::shared_ptr<ov_msckf::VioManager> vio_manager_;
  };

} // namespace vslam

#endif // VSLAM_APPLICATION_H
