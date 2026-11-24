// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/feature/types.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

// Helpers shared by the OnlineLoopCloser translation units.

namespace colmap {
namespace online_loop_internal {

// Newest track observations used as fusion descriptor references. The minimum
// L2 among them is kept, same as local-map projection matching.
inline constexpr size_t kMaxFuseRefDescriptors = 3;

// SIFT descriptors are L2-normalized to 512.
inline constexpr float kLoopDescMaxL2Sq = 512.0f * 512.0f * 0.7f * 0.7f;

// Feature scale consistency (BAFS observation model s = f * S / range). The
// point size S is the median over its observations in the loaded images; a
// keypoint is consistent when its scale is within the ratio of the scale S
// predicts at the queried pose. Range is used instead of optical-axis depth
// because the fisheye feature scale follows range.
class LoopScaleCheck {
 public:
  explicit LoopScaleCheck(const double max_ratio)
      : max_log_(max_ratio > 1.0 ? std::log(max_ratio) : 0.0) {}

  bool Enabled() const { return max_log_ > 0.0; }

  void Load(const Database& database, const image_t image_id) {
    if (!Enabled() || scales_.count(image_id) > 0 ||
        !database.ExistsKeypoints(image_id)) {
      return;
    }
    const FeatureKeypoints keypoints = database.ReadKeypoints(image_id);
    std::vector<float>& scales = scales_[image_id];
    scales.resize(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
      scales[i] = keypoints[i].ComputeScale();
    }
  }

  // True also when there is nothing to compare against.
  bool Consistent(const Reconstruction& reconstruction,
                  const image_t image_id,
                  const point2D_t point2D_idx,
                  const point3D_t point3D_id,
                  const Rigid3d& cam_from_world,
                  const Camera& camera) const {
    if (!Enabled() || !reconstruction.ExistsPoint3D(point3D_id)) {
      return true;
    }
    const auto it = scales_.find(image_id);
    if (it == scales_.end() || point2D_idx >= it->second.size() ||
        it->second[point2D_idx] <= 0.0f) {
      return true;
    }
    const double size = PointSize(reconstruction, point3D_id, image_id);
    const double range = (reconstruction.Point3D(point3D_id).xyz -
                          cam_from_world.TgtOriginInSrc())
                             .norm();
    if (size <= 0.0 || range <= 0.0) {
      return true;
    }
    const double predicted = size * camera.MeanFocalLength() / range;
    return std::abs(std::log(it->second[point2D_idx] / predicted)) <= max_log_;
  }

 private:
  double PointSize(const Reconstruction& reconstruction,
                   const point3D_t point3D_id,
                   const image_t query_id) const {
    const auto cached = sizes_.find(point3D_id);
    if (cached != sizes_.end()) {
      return cached->second;
    }
    const Point3D& point3D = reconstruction.Point3D(point3D_id);
    std::vector<double> sizes;
    for (const TrackElement& el : point3D.track.Elements()) {
      const auto it = scales_.find(el.image_id);
      if (el.image_id == query_id || it == scales_.end() ||
          el.point2D_idx >= it->second.size() ||
          !reconstruction.ExistsImage(el.image_id)) {
        continue;
      }
      const Image& image = reconstruction.Image(el.image_id);
      if (!image.HasPose() || !image.HasCameraPtr()) {
        continue;
      }
      sizes.push_back(it->second[el.point2D_idx] *
                      (point3D.xyz - image.ProjectionCenter()).norm() /
                      image.CameraPtr()->MeanFocalLength());
    }
    double size = 0.0;
    if (!sizes.empty()) {
      std::nth_element(
          sizes.begin(), sizes.begin() + sizes.size() / 2, sizes.end());
      size = sizes[sizes.size() / 2];
    }
    sizes_.emplace(point3D_id, size);
    return size;
  }

  double max_log_ = 0.0;
  std::unordered_map<image_t, std::vector<float>> scales_;
  mutable std::unordered_map<point3D_t, double> sizes_;
};

bool GeometryUsable(const TwoViewGeometry& geometry, int min_inliers);

std::optional<Eigen::Vector2d> ProjectCam(const Camera& camera,
                                          const Eigen::Vector3d& xyz_cam);

bool InsideImage(const Camera& camera, const Eigen::Vector2d& xy);

double RotationDiffDeg(const Eigen::Quaterniond& a,
                       const Eigen::Quaterniond& b);

float DescriptorL2Sq(const FeatureDescriptors& a,
                     point2D_t ia,
                     const FeatureDescriptors& b,
                     point2D_t ib);

}  // namespace online_loop_internal
}  // namespace colmap
