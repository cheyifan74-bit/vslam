#include "dense_mapper/depth_estimator.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "utils/print.h"

namespace vslam
{
namespace dense
{
namespace
{

cv::Mat ToRgb(const cv::Mat &image)
{
  cv::Mat rgb;
  cv::cvtColor(image, rgb, image.channels() == 3 ? cv::COLOR_BGR2RGB : cv::COLOR_GRAY2RGB);
  return rgb;
}

/// Closest sparse depth in the rectified camera, spread by a small radius.
/// Zero means no sparse point covers that pixel.
cv::Mat SparseDepthCeiling(const std::vector<Eigen::Vector3f> &xyz, const Eigen::Matrix3d &R,
                           const Eigen::Vector3d &t, int width, int height, float fx, float fy,
                           float cx, float cy, int radius_px)
{
  cv::Mat ceiling(height, width, CV_32F, cv::Scalar(0));
  for (const Eigen::Vector3f &point : xyz)
  {
    const Eigen::Vector3d cam = R * point.cast<double>() + t;
    if (cam.z() < 0.3)
    {
      continue;
    }
    const int u = static_cast<int>(std::lround(fx * cam.x() / cam.z() + cx));
    const int v = static_cast<int>(std::lround(fy * cam.y() / cam.z() + cy));
    if (u < 0 || v < 0 || u >= width || v >= height)
    {
      continue;
    }
    float &cell = ceiling.at<float>(v, u);
    const float depth = static_cast<float>(cam.z());
    if (cell == 0.f || depth < cell)
    {
      cell = depth;
    }
  }
  if (radius_px <= 0 || cv::countNonZero(ceiling) == 0)
  {
    return ceiling;
  }
  // Empty pixels are 0. A minimum filter would spread those zeros, so empty
  // cells are filled with a sentinel and restored after the erode.
  cv::Mat filled = ceiling.clone();
  filled.setTo(1e6f, ceiling == 0);
  const int k = radius_px * 2 + 1;
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
  cv::Mat spread;
  cv::erode(filled, spread, kernel);
  spread.setTo(0.f, spread >= 1e5f);
  return spread;
}

} // namespace

bool DepthEstimator::init(const std::vector<CameraParams> &cameras)
{
  if (!rectifier_.init(cameras, config_.rect_width, config_.rect_height))
  {
    return false;
  }
  engine_.load(config_.engine_path, config_.rect_width, config_.rect_height);
  return true;
}

Eigen::Matrix3d DepthEstimator::rectFromLeft() const
{
  const cv::Matx33d &R_rect = rectifier_.rectFromLeft();
  return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_rect.val);
}

Rigid DepthEstimator::rectifiedPose(const Rigid &cam_from_world) const
{
  const Eigen::Matrix3d R1 = rectFromLeft();
  const Rigid &left = cam_from_world;
  Rigid out;
  out.q = Eigen::Quaterniond(R1 * left.q.toRotationMatrix()).normalized();
  out.t = R1 * left.t;
  return out;
}

bool DepthEstimator::similarToRecent(const Rigid &cam_from_world)
{
  if (config_.skip_similar_trans_m <= 0.0 || config_.skip_similar_rot_deg <= 0.0)
  {
    return false;
  }
  const Rigid &pose = cam_from_world;
  for (const Rigid &recent : recent_inferred_)
  {
    if (!Moved(pose, recent, config_.skip_similar_trans_m, config_.skip_similar_rot_deg))
    {
      return true;
    }
  }
  recent_inferred_.push_back(pose);
  while (recent_inferred_.size() > 8)
  {
    recent_inferred_.pop_front();
  }
  return false;
}

bool DepthEstimator::build(const StereoKeyframe &keyframe, DepthFrame *frame)
{
  if (frame == nullptr || !engine_.ready() || keyframe.left.empty() || keyframe.right.empty())
  {
    return false;
  }
  const auto start = Clock::now();
  cv::Mat left_rect;
  cv::Mat right_rect;
  cv::Mat valid;
  if (!rectifier_.rectify(keyframe.left, keyframe.right, &left_rect, &right_rect, &valid))
  {
    return false;
  }
  const cv::Mat left_rgb = ToRgb(left_rect);
  const cv::Mat right_rgb = ToRgb(right_rect);
  std::vector<float> disparity;
  std::vector<float> occlusion;
  std::vector<float> confidence;
  if (!engine_.infer(left_rgb, right_rgb, &disparity, &occlusion, &confidence))
  {
    return false;
  }
  const double infer_ms = MsSince(start);
  const Rigid pose = rectifiedPose(keyframe.cam_from_world);
  frame->image_id = keyframe.image_id;
  frame->q = pose.q;
  frame->t = pose.t;
  frame->width = rectifier_.width();
  frame->height = rectifier_.height();
  frame->fx = rectifier_.fx();
  frame->fy = rectifier_.fy();
  frame->cx = rectifier_.cx();
  frame->cy = rectifier_.cy();
  frame->baseline = rectifier_.baseline();
  const int width = frame->width;
  const int height = frame->height;
  const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  frame->depth_mm.assign(pixels, 0);
  frame->conf.assign(pixels, 0);
  frame->gray.assign(pixels, 0);
  cv::Mat gray;
  if (left_rect.channels() == 1)
  {
    gray = left_rect;
  }
  else
  {
    cv::cvtColor(left_rect, gray, cv::COLOR_BGR2GRAY);
  }
  const bool use_sparse = config_.sparse_depth_ratio > 1.0 && !keyframe.sparse_xyz.empty();
  const cv::Mat ceiling =
      use_sparse ? SparseDepthCeiling(keyframe.sparse_xyz, frame->q.toRotationMatrix(), frame->t,
                                       width, height, frame->fx, frame->fy, frame->cx, frame->cy,
                                       config_.sparse_dilate_px)
                 : cv::Mat();
  const float *ceiling_data =
      ceiling.empty() ? nullptr : reinterpret_cast<const float *>(ceiling.data);
  std::vector<float> depth(pixels, 0.f);
  std::vector<float> weight(pixels, 0.f);
  int saturated = 0;
  int behind_sparse = 0;
  for (size_t i = 0; i < pixels; ++i)
  {
    if (valid.empty() || valid.data[i] == 0 || !std::isfinite(disparity[i]) || disparity[i] < 0.5f ||
        occlusion[i] < static_cast<float>(config_.min_visible) ||
        confidence[i] < static_cast<float>(config_.min_confidence))
    {
      continue;
    }
    if (config_.saturate_gray > 0 && gray.data[i] >= static_cast<uint8_t>(config_.saturate_gray))
    {
      ++saturated;
      continue;
    }
    const double d = static_cast<double>(frame->fx) * frame->baseline / disparity[i];
    if (d < config_.min_range_m || d > config_.max_range_m)
    {
      continue;
    }
    if (ceiling_data != nullptr && ceiling_data[i] > 0.f)
    {
      const double sparse_z = static_cast<double>(ceiling_data[i]);
      const double limit =
          std::max(sparse_z * config_.sparse_depth_ratio, sparse_z + config_.sparse_depth_margin_m);
      if (d > limit)
      {
        ++behind_sparse;
        continue;
      }
    }
    float w = std::clamp(confidence[i], 0.f, 1.f);
    if (config_.weight_reference_range_m > 0.0)
    {
      const double z_ref = config_.weight_reference_range_m;
      const double scale = (z_ref / d) * (z_ref / d);
      w = static_cast<float>(std::min(1.0, scale) * w);
    }
    depth[i] = static_cast<float>(d);
    weight[i] = w;
  }
  int edges = 0;
  if (config_.edge_ratio > 0.0)
  {
    const float ratio = static_cast<float>(config_.edge_ratio);
    std::vector<uint8_t> drop(pixels, 0);
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        const size_t i = static_cast<size_t>(y) * width + x;
        const float d = depth[i];
        if (d <= 0.f)
        {
          continue;
        }
        const float limit = ratio * d;
        const auto jump = [&](int nx, int ny) {
          if (nx < 0 || ny < 0 || nx >= width || ny >= height)
          {
            return false;
          }
          const float other = depth[static_cast<size_t>(ny) * width + nx];
          return other > 0.f && std::abs(other - d) > limit;
        };
        if (jump(x - 1, y) || jump(x + 1, y) || jump(x, y - 1) || jump(x, y + 1))
        {
          drop[i] = 1;
        }
      }
    }
    for (size_t i = 0; i < pixels; ++i)
    {
      if (drop[i] != 0)
      {
        depth[i] = 0.f;
        ++edges;
      }
    }
  }
  int kept = 0;
  for (size_t i = 0; i < pixels; ++i)
  {
    if (depth[i] <= 0.f)
    {
      continue;
    }
    frame->depth_mm[i] = static_cast<uint16_t>(std::min(depth[i] * 1000.0, 65535.0));
    frame->conf[i] = static_cast<uint8_t>(std::clamp(weight[i], 0.f, 1.f) * 255.f);
    frame->gray[i] = gray.data[i];
    ++kept;
  }
  PRINT_INFO("[DENSE]: image %u depth pixels=%d / %zu saturated=%d behind_sparse=%d edge=%d "
             "infer=%.0fms\n",
             keyframe.image_id, kept, pixels, saturated, behind_sparse, edges, infer_ms);
  return kept > 0;
}

} // namespace dense
} // namespace vslam
