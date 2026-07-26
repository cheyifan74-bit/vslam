/*
 * @Description: Application class implementation
 * @Author: che yifan
 * @Date: 2026-07-26 09:39:40
 * @LastEditTime: 2026-07-26 09:39:45
 * @LastEditors: che yifan
 * @Reference:
 */
#include "application.h"

#include <iostream>

#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace vslam
{

  Application::Application(const std::string &config_path)
  {
    initOpenVINS(config_path);

    if (!vioInitialized())
    {
      PRINT_ERROR("[APP]: OpenVINS initialization failed!\n");
    }
    else
    {
      PRINT_INFO("[APP]: OpenVINS initialized successfully.\n");
    }
  }

  bool Application::vioInitialized() const
  {
    return vio_manager_ != nullptr;
  }

  std::shared_ptr<ov_msckf::VioManager> Application::getVioManager() const
  {
    return vio_manager_;
  }

  void Application::initOpenVINS(const std::string &config_path)
  {
    // Step 1: Create YAML parser to load configuration
    PRINT_INFO("[APP]: Loading OpenVINS configuration from: %s\n", config_path.c_str());
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);

    // Set verbosity level (optional, defaults to DEBUG)
    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity, false);
    ov_core::Printer::setPrintLevel(verbosity);

    // Step 2: Load all VIO parameters from config file
    ov_msckf::VioManagerOptions params;
    params.print_and_load(parser);

    // Step 3: Verify all required parameters were successfully parsed
    if (!parser->successful())
    {
      PRINT_ERROR(RED "[APP]: Failed to parse all parameters from config file!\n" RESET);
      return;
    }

    // Step 4: Create the VIO manager instance
    vio_manager_ = std::make_shared<ov_msckf::VioManager>(params);

    PRINT_INFO("[APP]: OpenVINS VioManager created.\n");
  }

} // namespace vslam
