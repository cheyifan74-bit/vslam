#include "dense_mapper/stereo_rectifier.h"

#include <cmath>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/print.h"

namespace vslam
{
namespace
{

Eigen::Matrix4d MatFromRowMajor(const double *m)
{
  Eigen::Matrix4d T;
  for (int r = 0; r < 4; ++r)
  {
    for (int c = 0; c < 4; ++c)
    {
      T(r, c) = m[r * 4 + c];
    }
  }
  return T;
}

cv::Mat CameraMatrix(const CameraParams &cam)
{
  cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
  K.at<double>(0, 0) = cam.fx;
  K.at<double>(1, 1) = cam.fy;
  K.at<double>(0, 2) = cam.cx;
  K.at<double>(1, 2) = cam.cy;
  return K;
}

cv::Mat Distortion(const CameraParams &cam)
{
  return (cv::Mat_<double>(4, 1) << cam.k1, cam.k2, cam.k3, cam.k4);
}

} // namespace

bool StereoRectifier::init(const std::vector<CameraParams> &cameras, const int rect_width,
                           const int rect_height)
{
  ready_ = false;
  if (cameras.size() < 2 || rect_width <= 0 || rect_height <= 0)
  {
    PRINT_WARNING("[DENSE]: stereo rectification needs two cameras\n");
    return false;
  }
  const CameraParams &left = cameras[0];
  const CameraParams &right = cameras[1];
  if (left.width <= 0 || left.height <= 0)
  {
    return false;
  }

  const Eigen::Matrix4d T_right_left =
      MatFromRowMajor(right.T_cam_imu) * MatFromRowMajor(left.T_cam_imu).inverse();
  cv::Mat R(3, 3, CV_64F);
  cv::Mat t(3, 1, CV_64F);
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      R.at<double>(r, c) = T_right_left(r, c);
    }
    t.at<double>(r) = T_right_left(r, 3);
  }

  const cv::Size raw_size(left.width, left.height);
  const cv::Size rect_size(rect_width, rect_height);
  cv::Mat R1, R2, P1, P2, Q;
  const bool fisheye = left.distortion_model == "equidistant" ||
                       right.distortion_model == "equidistant";
  try
  {
    if (fisheye)
    {
      cv::fisheye::stereoRectify(CameraMatrix(left), Distortion(left), CameraMatrix(right),
                                 Distortion(right), raw_size, R, t, R1, R2, P1, P2, Q,
                                 cv::CALIB_ZERO_DISPARITY, rect_size, 0.0, 1.0);
      cv::fisheye::initUndistortRectifyMap(CameraMatrix(left), Distortion(left), R1, P1,
                                           rect_size, CV_32FC1, map1_left_, map2_left_);
      cv::fisheye::initUndistortRectifyMap(CameraMatrix(right), Distortion(right), R2, P2,
                                           rect_size, CV_32FC1, map1_right_, map2_right_);
    }
    else
    {
      cv::stereoRectify(CameraMatrix(left), Distortion(left), CameraMatrix(right),
                        Distortion(right), raw_size, R, t, R1, R2, P1, P2, Q,
                        cv::CALIB_ZERO_DISPARITY, -1, rect_size);
      cv::initUndistortRectifyMap(CameraMatrix(left), Distortion(left), R1, P1, rect_size,
                                  CV_32FC1, map1_left_, map2_left_);
      cv::initUndistortRectifyMap(CameraMatrix(right), Distortion(right), R2, P2, rect_size,
                                  CV_32FC1, map1_right_, map2_right_);
    }
  }
  catch (const cv::Exception &e)
  {
    PRINT_ERROR("[DENSE]: stereoRectify failed: %s\n", e.what());
    return false;
  }

  fx_ = static_cast<float>(P1.at<double>(0, 0));
  fy_ = static_cast<float>(P1.at<double>(1, 1));
  cx_ = static_cast<float>(P1.at<double>(0, 2));
  cy_ = static_cast<float>(P1.at<double>(1, 2));
  const double px = P2.at<double>(0, 0);
  baseline_ = px == 0.0 ? 0.f : static_cast<float>(std::abs(P2.at<double>(0, 3) / px));
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      rect_from_left_(r, c) = R1.at<double>(r, c);
    }
  }
  width_ = rect_width;
  height_ = rect_height;
  ready_ = fx_ > 1.f && baseline_ > 1e-4f;
  if (ready_)
  {
    PRINT_INFO("[DENSE]: rectified camera %dx%d fx=%.2f fy=%.2f baseline=%.4f m\n", width_,
               height_, fx_, fy_, baseline_);
  }
  else
  {
    PRINT_ERROR("[DENSE]: rectified camera is degenerate (fx=%.2f baseline=%.4f)\n", fx_,
                baseline_);
  }
  return ready_;
}

bool StereoRectifier::rectify(const cv::Mat &left, const cv::Mat &right, cv::Mat *left_rect,
                              cv::Mat *right_rect, cv::Mat *valid_mask) const
{
  if (!ready_ || left.empty() || right.empty() || left_rect == nullptr || right_rect == nullptr)
  {
    return false;
  }
  cv::remap(left, *left_rect, map1_left_, map2_left_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
  cv::remap(right, *right_rect, map1_right_, map2_right_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
  if (valid_mask != nullptr)
  {
    cv::Mat ones(left.size(), CV_8U, cv::Scalar(255));
    cv::remap(ones, *valid_mask, map1_left_, map2_left_, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
  }
  return true;
}

} // namespace vslam
