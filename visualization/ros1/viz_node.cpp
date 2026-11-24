/*
 * @Description: VizNode implementation (ROS1).
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "visualization/ros1/viz_node.h"

namespace vslam
{

  VizNode::VizNode(const ros::NodeHandle &main_nh) : nh_(main_nh) { nh_.setCallbackQueue(&queue_); }

  VizNode::~VizNode() { stop(); }

  void VizNode::start()
  {
    if (!spinner_)
    {
      spinner_ = std::make_unique<ros::AsyncSpinner>(1, &queue_);
      spinner_->start();
    }
  }

  void VizNode::stop()
  {
    if (spinner_)
    {
      spinner_->stop();
      spinner_.reset();
    }
  }

} // namespace vslam
