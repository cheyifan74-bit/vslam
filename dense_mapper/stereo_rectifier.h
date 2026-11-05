#ifndef VSLAM_DENSE_MAPPER_STEREO_RECTIFIER_H
#define VSLAM_DENSE_MAPPER_STEREO_RECTIFIER_H

#include <vector>

#include <opencv2/core.hpp>

#include "map_manager/map.h"

namespace vslam
{

/// Fisheye (or pinhole) stereo pair remapped to one fixed virtual pinhole.
/// The virtual size is the TensorRT engine's input size, so every dataset
/// shares one engine.
class StereoRectifier
{
public:
  bool init(const std::vector<CameraParams> &cameras, int rect_width, int rect_height);

  bool ready() const { return ready_; }
  int width() const { return width_; }
  int height() const { return height_; }
  float fx() const { return fx_; }
  float fy() const { return fy_; }
  float cx() const { return cx_; }
  float cy() const { return cy_; }
  /// Rectified stereo baseline in meters.
  float baseline() const { return baseline_; }
  /// Rotation taking a point in the raw left camera into the rectified camera.
  const cv::Matx33d &rectFromLeft() const { return rect_from_left_; }

  bool rectify(const cv::Mat &left, const cv::Mat &right, cv::Mat *left_rect,
               cv::Mat *right_rect, cv::Mat *valid_mask) const;

private:
  bool ready_ = false;
  int width_ = 0;
  int height_ = 0;
  float fx_ = 0;
  float fy_ = 0;
  float cx_ = 0;
  float cy_ = 0;
  float baseline_ = 0;
  cv::Matx33d rect_from_left_ = cv::Matx33d::eye();
  cv::Mat map1_left_;
  cv::Mat map2_left_;
  cv::Mat map1_right_;
  cv::Mat map2_right_;
};

} // namespace vslam

#endif
