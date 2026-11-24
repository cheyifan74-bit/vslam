/*
 * @Description: VizNode implementation.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "visualization/ros2/viz_node.h"

#include <string>
#include <vector>

namespace vslam
{

  VizNode::VizNode(const rclcpp::Node &main_node)
  {
    // Remapping/parameter arguments of the main node are not passed on: topic names are
    // resolved against the same namespace, so they stay identical to the old ones.
    std::vector<std::string> args = {main_node.get_name() + std::string("_map_viz")};
    std::vector<const char *> argv;
    for (const auto &arg : args)
    {
      argv.push_back(arg.c_str());
    }
    context_ = std::make_shared<rclcpp::Context>();
    context_->init(static_cast<int>(argv.size()), argv.data());

    rclcpp::NodeOptions options;
    options.context(context_);
    options.use_global_arguments(false);
    options.start_parameter_services(false);
    options.start_parameter_event_publisher(false);
    node_ = std::make_shared<rclcpp::Node>(args.front(), main_node.get_namespace(), options);

    rclcpp::ExecutorOptions executor_options;
    executor_options.context = context_;
    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>(executor_options);
  }

  VizNode::~VizNode() { stop(); }

  void VizNode::start()
  {
    if (executor_ && node_ && !thread_.joinable())
    {
      executor_->add_node(node_);
      thread_ = std::thread([this]() { executor_->spin(); });
    }
  }

  void VizNode::stop()
  {
    if (context_ && context_->is_valid())
    {
      context_->shutdown("vslam exiting");
    }
    if (executor_)
    {
      executor_->cancel();
    }
    if (thread_.joinable())
    {
      thread_.join();
    }
  }

} // namespace vslam
