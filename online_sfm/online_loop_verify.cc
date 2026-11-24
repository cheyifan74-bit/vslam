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

#include "online_sfm/online_loop_closer.h"

#include "colmap/estimators/pose.h"
#include "colmap/feature/types.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "online_sfm/online_loop_closer_internal.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database_session.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Geometry>

namespace colmap {

using online_loop_internal::DescriptorL2Sq;
using online_loop_internal::GeometryUsable;
using online_loop_internal::InsideImage;
using online_loop_internal::kLoopDescMaxL2Sq;
using online_loop_internal::kMaxFuseRefDescriptors;
using online_loop_internal::LoopScaleCheck;
using online_loop_internal::ProjectCam;
using online_loop_internal::RotationDiffDeg;

namespace {

// Keypoint noise behind the loop PnP covariance.
constexpr double kLoopPixelSigma = 1.0;
// PnP covariance treats map points as exact; inflate the loop edge sigma.
constexpr double kLoopSigmaInflation = 2.0;

// Current keypoint ↔ loop-side map point.
struct LoopPnPCorr {
  point2D_t idx_current = kInvalidPoint2DIdx;
  point3D_t point3D_id = kInvalidPoint3DId;
  Eigen::Vector2d xy_current = Eigen::Vector2d::Zero();
};

size_t CountInliers(const std::vector<char>& inliers) {
  return static_cast<size_t>(std::count(inliers.begin(), inliers.end(), 1));
}

// Matches from ReadMatches(loop_image, current): idx1 = loop image, idx2 =
// current. Every loop-side keypoint with a 3D point gives one 2D–3D pair.
void CollectLoopPnPCorrs(const Image& current,
                         const Image& loop_image,
                         const Reconstruction& reconstruction,
                         const FeatureMatches& matches,
                         std::unordered_set<point2D_t>* used_current,
                         std::unordered_set<point3D_t>* used_points,
                         std::vector<LoopPnPCorr>* corrs) {
  for (const FeatureMatch& match : matches) {
    const point2D_t idx_current = match.point2D_idx2;
    const point2D_t idx_loop = match.point2D_idx1;
    if (idx_current >= current.NumPoints2D() ||
        idx_loop >= loop_image.NumPoints2D() ||
        used_current->count(idx_current) > 0) {
      continue;
    }
    const Point2D& p_loop = loop_image.Point2D(idx_loop);
    if (!p_loop.HasPoint3D() ||
        !reconstruction.ExistsPoint3D(p_loop.point3D_id) ||
        used_points->count(p_loop.point3D_id) > 0) {
      continue;
    }
    LoopPnPCorr corr;
    corr.idx_current = idx_current;
    corr.point3D_id = p_loop.point3D_id;
    corr.xy_current = current.Point2D(idx_current).xy;
    corrs->push_back(corr);
    used_current->insert(idx_current);
    used_points->insert(p_loop.point3D_id);
  }
}

std::vector<char> ComputePnPInliers(const std::vector<LoopPnPCorr>& corrs,
                                    const Reconstruction& reconstruction,
                                    const Rigid3d& cam_from_world,
                                    const Camera& camera,
                                    const double max_error_px) {
  const double max_sq = max_error_px * max_error_px;
  std::vector<char> inliers(corrs.size(), 0);
  for (size_t i = 0; i < corrs.size(); ++i) {
    if (!reconstruction.ExistsPoint3D(corrs[i].point3D_id)) {
      continue;
    }
    const double err = CalculateSquaredReprojectionError(
        corrs[i].xy_current,
        reconstruction.Point3D(corrs[i].point3D_id).xyz,
        cam_from_world,
        camera);
    inliers[i] = err <= max_sq ? 1 : 0;
  }
  return inliers;
}

// Isotropic 1-sigma of the PnP pose from the Gauss-Newton information of the
// inlier reprojections, with a left perturbation [w, v] of cam_from_world.
// The rotation sigma covers w; the translation sigma covers the camera center
// in any fixed frame, which only moves with v.
bool EstimatePnPSigmas(const std::vector<LoopPnPCorr>& corrs,
                       const std::vector<char>& inliers,
                       const Reconstruction& reconstruction,
                       const Rigid3d& cam_from_world,
                       const Camera& camera,
                       double* rot_sigma,
                       double* trans_sigma) {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  size_t num_used = 0;
  for (size_t i = 0; i < corrs.size(); ++i) {
    if (!inliers[i] || !reconstruction.ExistsPoint3D(corrs[i].point3D_id)) {
      continue;
    }
    const Eigen::Vector3d xyz_cam =
        cam_from_world * reconstruction.Point3D(corrs[i].point3D_id).xyz;
    const auto proj = ProjectCam(camera, xyz_cam);
    if (!proj) {
      continue;
    }
    const double h = 1e-6 * std::max(1.0, xyz_cam.norm());
    Eigen::Matrix<double, 2, 3> J_proj;
    bool ok = true;
    for (int a = 0; a < 3 && ok; ++a) {
      Eigen::Vector3d xyz_h = xyz_cam;
      xyz_h[a] += h;
      const auto proj_h = ProjectCam(camera, xyz_h);
      ok = proj_h.has_value();
      if (ok) {
        J_proj.col(a) = (*proj_h - *proj) / h;
      }
    }
    if (!ok) {
      continue;
    }
    Eigen::Matrix<double, 3, 6> J_xyz;
    J_xyz.leftCols<3>() << 0.0, xyz_cam.z(), -xyz_cam.y(),
        -xyz_cam.z(), 0.0, xyz_cam.x(),
        xyz_cam.y(), -xyz_cam.x(), 0.0;
    J_xyz.rightCols<3>().setIdentity();
    const Eigen::Matrix<double, 2, 6> J = J_proj * J_xyz;
    H += J.transpose() * J;
    ++num_used;
  }
  if (num_used < 6) {
    return false;
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig_h(H);
  if (eig_h.info() != Eigen::Success || !(eig_h.eigenvalues()(0) > 1e-9)) {
    return false;
  }
  const Eigen::Matrix<double, 6, 6> cov =
      kLoopPixelSigma * kLoopPixelSigma * H.inverse();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_rot(
      cov.topLeftCorner<3, 3>());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_trans(
      cov.bottomRightCorner<3, 3>());
  *rot_sigma = std::sqrt(std::max(eig_rot.eigenvalues()(2), 0.0));
  *trans_sigma = std::sqrt(std::max(eig_trans.eigenvalues()(2), 0.0));
  return true;
}

// Project loop-map points with the PnP pose into the current image and take
// the descriptor-closest keypoint inside `radius_px` (ORB SearchByProjection).
size_t SearchLoopByProjection(
    const Image& current,
    const Reconstruction& reconstruction,
    const Rigid3d& cam_from_world,
    const std::vector<image_t>& loop_ids,
    const std::unordered_map<image_t, FeatureDescriptors>& loop_desc,
    const FeatureDescriptors& desc_current,
    const double radius_px,
    const LoopScaleCheck& scale_check,
    std::unordered_set<point2D_t>* used_current,
    std::unordered_set<point3D_t>* used_points,
    std::vector<LoopPnPCorr>* corrs) {
  if (desc_current.data.rows() == 0 || !current.HasCameraPtr()) {
    return 0;
  }
  const Camera& camera = *current.CameraPtr();
  const double radius_sq = radius_px * radius_px;
  size_t added = 0;
  for (const image_t loop_id : loop_ids) {
    const auto desc_it = loop_desc.find(loop_id);
    if (desc_it == loop_desc.end() || !reconstruction.ExistsImage(loop_id)) {
      continue;
    }
    const Image& loop_image = reconstruction.Image(loop_id);
    for (point2D_t j = 0; j < loop_image.NumPoints2D(); ++j) {
      const Point2D& p_loop = loop_image.Point2D(j);
      if (!p_loop.HasPoint3D() ||
          !reconstruction.ExistsPoint3D(p_loop.point3D_id) ||
          used_points->count(p_loop.point3D_id) > 0) {
        continue;
      }
      const auto proj = ProjectCam(
          camera,
          cam_from_world * reconstruction.Point3D(p_loop.point3D_id).xyz);
      if (!proj || !InsideImage(camera, *proj)) {
        continue;
      }
      float best_l2 = kLoopDescMaxL2Sq;
      point2D_t best_idx = kInvalidPoint2DIdx;
      for (point2D_t i = 0; i < current.NumPoints2D(); ++i) {
        if (used_current->count(i) > 0 ||
            (current.Point2D(i).xy - *proj).squaredNorm() >= radius_sq ||
            !scale_check.Consistent(reconstruction,
                                    current.ImageId(),
                                    i,
                                    p_loop.point3D_id,
                                    cam_from_world,
                                    camera)) {
          continue;
        }
        const float l2 = DescriptorL2Sq(desc_it->second, j, desc_current, i);
        if (l2 < best_l2) {
          best_l2 = l2;
          best_idx = i;
        }
      }
      if (best_idx == kInvalidPoint2DIdx) {
        continue;
      }
      LoopPnPCorr corr;
      corr.idx_current = best_idx;
      corr.point3D_id = p_loop.point3D_id;
      corr.xy_current = current.Point2D(best_idx).xy;
      corrs->push_back(corr);
      used_current->insert(best_idx);
      used_points->insert(p_loop.point3D_id);
      ++added;
    }
  }
  return added;
}

// Pose-free 2D-3D bootstrap against the loop-side island. Each current
// keypoint takes its nearest SIFT neighbour among island observations (Lowe
// ratio against the best observation of a different point) that lies within
// `max_epipolar_angle` of the epipolar plane E * x_candidate; each point keeps
// its closest keypoint. E maps candidate rays to current rays.
size_t SearchLoopIslandByDescriptor(
    const Image& current,
    const Image& candidate,
    const Reconstruction& reconstruction,
    const Eigen::Matrix3d& E,
    const std::vector<image_t>& loop_ids,
    const std::unordered_map<image_t, FeatureDescriptors>& loop_desc,
    const FeatureDescriptors& desc_current,
    const double max_epipolar_angle,
    std::unordered_set<point2D_t>* used_current,
    std::unordered_set<point3D_t>* used_points,
    std::vector<LoopPnPCorr>* corrs) {
  const int dim = static_cast<int>(desc_current.data.cols());
  if (dim == 0 || !current.HasCameraPtr() || !candidate.HasPose()) {
    return 0;
  }
  const Rigid3d& cand_from_world = candidate.CamFromWorld();

  std::vector<point3D_t> row_points;
  std::vector<Eigen::Vector3d> row_normals;
  std::vector<const uint8_t*> row_desc;
  std::unordered_map<point3D_t, size_t> rows_per_point;
  for (const image_t loop_id : loop_ids) {
    const auto desc_it = loop_desc.find(loop_id);
    if (desc_it == loop_desc.end() || !reconstruction.ExistsImage(loop_id) ||
        desc_it->second.data.cols() != dim) {
      continue;
    }
    const Image& loop_image = reconstruction.Image(loop_id);
    const FeatureDescriptorsData& data = desc_it->second.data;
    for (point2D_t j = 0; j < loop_image.NumPoints2D(); ++j) {
      const Point2D& p_loop = loop_image.Point2D(j);
      if (static_cast<int>(j) >= data.rows() || !p_loop.HasPoint3D() ||
          !reconstruction.ExistsPoint3D(p_loop.point3D_id) ||
          used_points->count(p_loop.point3D_id) > 0) {
        continue;
      }
      size_t& num_rows = rows_per_point[p_loop.point3D_id];
      if (num_rows >= kMaxFuseRefDescriptors) {
        continue;
      }
      const Eigen::Vector3d normal =
          E * (cand_from_world * reconstruction.Point3D(p_loop.point3D_id).xyz);
      if (normal.norm() < 1e-12) {
        continue;
      }
      ++num_rows;
      row_points.push_back(p_loop.point3D_id);
      row_normals.push_back(normal.normalized());
      row_desc.push_back(data.row(j).data());
    }
  }

  std::vector<point2D_t> col_idx;
  std::vector<Eigen::Vector3d> col_rays;
  for (point2D_t i = 0; i < current.NumPoints2D(); ++i) {
    if (static_cast<int>(i) >= desc_current.data.rows() ||
        used_current->count(i) > 0) {
      continue;
    }
    const auto ray = current.CameraPtr()->CamFromImg(current.Point2D(i).xy);
    if (!ray) {
      continue;
    }
    col_idx.push_back(i);
    col_rays.push_back(ray->homogeneous().normalized());
  }
  if (row_points.empty() || col_idx.empty()) {
    return 0;
  }

  Eigen::MatrixXf rows(row_points.size(), dim);
  for (size_t r = 0; r < row_desc.size(); ++r) {
    rows.row(r) = Eigen::Map<const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic>>(
                      row_desc[r], dim)
                      .cast<float>();
  }
  const Eigen::VectorXf row_sq = rows.rowwise().squaredNorm();
  const double max_sin = std::sin(max_epipolar_angle);
  constexpr float kMaxRatioSq = 0.8f * 0.8f;
  constexpr int kBlock = 256;

  // point -> (L2^2, current keypoint), mutual best.
  std::unordered_map<point3D_t, std::pair<float, point2D_t>> best_for_point;
  for (size_t c0 = 0; c0 < col_idx.size(); c0 += kBlock) {
    const int n = static_cast<int>(std::min<size_t>(kBlock, col_idx.size() - c0));
    Eigen::MatrixXf cols(n, dim);
    for (int c = 0; c < n; ++c) {
      cols.row(c) = desc_current.data.row(col_idx[c0 + c]).cast<float>();
    }
    const Eigen::VectorXf col_sq = cols.rowwise().squaredNorm();
    const Eigen::MatrixXf dots = rows * cols.transpose();
    for (int c = 0; c < n; ++c) {
      const Eigen::Vector3d& ray = col_rays[c0 + c];
      float best = std::numeric_limits<float>::infinity();
      float second = std::numeric_limits<float>::infinity();
      point3D_t best_point = kInvalidPoint3DId;
      for (size_t r = 0; r < row_points.size(); ++r) {
        if (std::abs(ray.dot(row_normals[r])) > max_sin) {
          continue;
        }
        const float d = row_sq(r) + col_sq(c) - 2.0f * dots(r, c);
        if (d < best) {
          if (row_points[r] != best_point) {
            second = best;
          }
          best = d;
          best_point = row_points[r];
        } else if (d < second && row_points[r] != best_point) {
          second = d;
        }
      }
      if (best_point == kInvalidPoint3DId || best > kLoopDescMaxL2Sq ||
          best > kMaxRatioSq * second) {
        continue;
      }
      const auto it = best_for_point.find(best_point);
      if (it == best_for_point.end() || best < it->second.first) {
        best_for_point[best_point] = {best, col_idx[c0 + c]};
      }
    }
  }

  size_t added = 0;
  for (const auto& [point3D_id, match] : best_for_point) {
    if (used_current->count(match.second) > 0) {
      continue;
    }
    LoopPnPCorr corr;
    corr.idx_current = match.second;
    corr.point3D_id = point3D_id;
    corr.xy_current = current.Point2D(match.second).xy;
    corrs->push_back(corr);
    used_current->insert(match.second);
    used_points->insert(point3D_id);
    ++added;
  }
  return added;
}

// Rotation angle between the PnP-implied relative pose and the essential
// matrix pose of the same pair (image1 = candidate, image2 = current).
// Returns -1 when the essential matrix is unavailable.
double RotationVsEssentialDeg(const TwoViewGeometry& geometry,
                              const Image& candidate,
                              const Image& current,
                              const Rigid3d& current_cam_from_world,
                              double* tdir_deg) {
  *tdir_deg = -1.0;
  if (!geometry.E || geometry.inlier_matches.size() < 8 ||
      !candidate.HasCameraPtr() || !current.HasCameraPtr()) {
    return -1.0;
  }
  std::vector<Eigen::Vector3d> rays1;
  std::vector<Eigen::Vector3d> rays2;
  rays1.reserve(geometry.inlier_matches.size());
  rays2.reserve(geometry.inlier_matches.size());
  for (const FeatureMatch& m : geometry.inlier_matches) {
    if (m.point2D_idx1 >= candidate.NumPoints2D() ||
        m.point2D_idx2 >= current.NumPoints2D()) {
      continue;
    }
    const auto n1 =
        candidate.CameraPtr()->CamFromImg(candidate.Point2D(m.point2D_idx1).xy);
    const auto n2 =
        current.CameraPtr()->CamFromImg(current.Point2D(m.point2D_idx2).xy);
    if (!n1 || !n2) {
      continue;
    }
    rays1.push_back(n1->homogeneous().normalized());
    rays2.push_back(n2->homogeneous().normalized());
  }
  if (rays1.size() < 8) {
    return -1.0;
  }
  Rigid3d e_cur_from_cand;
  std::vector<int> valid;
  PoseFromEssentialMatrix(*geometry.E, rays1, rays2, &e_cur_from_cand, &valid);
  if (valid.empty()) {
    return -1.0;
  }
  const Rigid3d pnp_cur_from_cand =
      current_cam_from_world * Inverse(candidate.CamFromWorld());
  const Eigen::Vector3d t_pnp = pnp_cur_from_cand.translation();
  const Eigen::Vector3d t_e = e_cur_from_cand.translation();
  if (t_pnp.norm() > 1e-6 && t_e.norm() > 1e-6) {
    *tdir_deg = RadToDeg(std::acos(
        std::clamp(t_pnp.normalized().dot(t_e.normalized()), -1.0, 1.0)));
  }
  return RotationDiffDeg(pnp_cur_from_cand.rotation(),
                         e_cur_from_cand.rotation());
}

}  // namespace

bool OnlineLoopCloser::VerifyCandidate(const image_t image_id,
                                       const image_t candidate_id,
                                       const float retrieval_score,
                                       const Reconstruction& reconstruction,
                                       const CandidateGate& gate,
                                       OnlineLoopPair* loop_pair) {
  if (loop_pair == nullptr) {
    return false;
  }

  loop_pair->image_id1 = image_id;
  loop_pair->image_id2 = candidate_id;
  loop_pair->retrieval_score = retrieval_score;
  loop_pair->has_loop_pose = false;

  FeatureMatches matches;
  TwoViewGeometry geometry;
  FeatureDescriptors desc_current;
  FeatureDescriptors desc_candidate;
  LoopScaleCheck scale_check(options_.scale_check_max_ratio);
  try {
    DatabaseSession session(database_path_);
    if (session->ExistsMatches(candidate_id, image_id)) {
      matches = session->ReadMatches(candidate_id, image_id);
    }
    loop_pair->num_matches = matches.size();
    if (matches.size() < gate.min_matches) {
      LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-"
                << candidate_id << ", matches=" << matches.size();
      return false;
    }
    if (session->ExistsTwoViewGeometry(candidate_id, image_id)) {
      geometry = session->ReadTwoViewGeometry(candidate_id, image_id);
    }
    loop_pair->num_inliers = geometry.inlier_matches.size();
    if (!GeometryUsable(geometry, static_cast<int>(gate.min_matches))) {
      LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-"
                << candidate_id
                << ", essential inliers=" << geometry.inlier_matches.size();
      return false;
    }
    if (session->ExistsDescriptors(image_id)) {
      desc_current = session->ReadDescriptors(image_id);
    }
    if (session->ExistsDescriptors(candidate_id)) {
      desc_candidate = session->ReadDescriptors(candidate_id);
    }
    scale_check.Load(*session, image_id);
    scale_check.Load(*session, candidate_id);
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to read pair " << candidate_id
                 << "-" << image_id << ": " << e.what();
    return false;
  }

  const FeatureMatches& verify_matches = geometry.inlier_matches.empty()
                                             ? matches
                                             : geometry.inlier_matches;
  const Image& current = reconstruction.Image(image_id);
  const Image& candidate = reconstruction.Image(candidate_id);
  if (!current.HasCameraPtr() || !candidate.HasCameraPtr() ||
      !candidate.HasPose() ||
      (gate.require_current_pose && !current.HasPose())) {
    return false;
  }

  std::vector<LoopPnPCorr> corrs;
  std::unordered_set<point2D_t> used_current;
  std::unordered_set<point3D_t> used_points;
  CollectLoopPnPCorrs(current,
                      candidate,
                      reconstruction,
                      verify_matches,
                      &used_current,
                      &used_points,
                      &corrs);
  const size_t num_candidate_corrs = corrs.size();

  std::vector<image_t> loop_ids;
  std::unordered_map<image_t, FeatureDescriptors> loop_desc;
  bool loop_map_loaded = false;
  const auto LoadLoopMap = [&]() {
    if (loop_map_loaded) {
      return;
    }
    loop_map_loaded = true;
    loop_ids = LoopMapImages(candidate_id, image_id, reconstruction);
    loop_desc.emplace(candidate_id, desc_candidate);
    try {
      DatabaseSession session(database_path_);
      for (const image_t id : loop_ids) {
        if (id != candidate_id && session->ExistsDescriptors(id)) {
          loop_desc.emplace(id, session->ReadDescriptors(id));
        }
        scale_check.Load(*session, id);
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "OnlineLoopCloser: failed to read loop-map descriptors "
                   << "for " << image_id << "-" << candidate_id << ": "
                   << e.what();
    }
  };

  // The candidate may have triangulated few of its matched keypoints; borrow
  // the 3D points of its covisibility island instead of rejecting.
  size_t num_island = 0;
  const size_t island_target =
      2 * std::max(gate.min_3d_corrs,
                   static_cast<size_t>(std::max(options_.min_num_3d_inliers, 0)));
  if (options_.island_2d3d && corrs.size() < island_target && geometry.E) {
    LoadLoopMap();
    const double focal = std::max(current.CameraPtr()->MeanFocalLength(), 1.0);
    num_island = SearchLoopIslandByDescriptor(
        current,
        candidate,
        reconstruction,
        *geometry.E,
        loop_ids,
        loop_desc,
        desc_current,
        std::atan(std::max(options_.island_epipolar_px, 0.1) / focal),
        &used_current,
        &used_points,
        &corrs);
  }
  loop_pair->num_3d_correspondences = corrs.size();
  if (corrs.size() < gate.min_3d_corrs) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", 2D-3D=" << corrs.size() << " (candidate="
              << num_candidate_corrs << " island=" << num_island << ")"
              << " inliers=" << geometry.inlier_matches.size();
    return false;
  }

  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
  const auto FillPoints = [&]() {
    points2D.clear();
    points3D.clear();
    points2D.reserve(corrs.size());
    points3D.reserve(corrs.size());
    for (const LoopPnPCorr& corr : corrs) {
      points2D.push_back(corr.xy_current);
      points3D.push_back(reconstruction.Point3D(corr.point3D_id).xyz);
    }
  };
  FillPoints();

  Camera camera = *current.CameraPtr();
  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.estimate_focal_length = false;
  abs_pose_options.ransac_options.max_error = options_.pnp_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio = 0.1;
  AbsolutePoseRefinementOptions refine_options;
  refine_options.refine_focal_length = false;
  refine_options.refine_extra_params = false;

  Rigid3d cam_from_world;
  size_t num_ransac_inliers = 0;
  std::vector<char> inliers;
  const bool pnp_ok = EstimateAbsolutePose(abs_pose_options,
                                           points2D,
                                           points3D,
                                           &cam_from_world,
                                           &camera,
                                           &num_ransac_inliers,
                                           &inliers);
  loop_pair->num_3d_inliers = pnp_ok ? num_ransac_inliers : 0;
  if (!pnp_ok || num_ransac_inliers <
                     static_cast<size_t>(options_.min_num_3d_inliers)) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", pnp ransac failed 2D-3D=" << corrs.size()
              << " island=" << num_island
              << " inliers=" << loop_pair->num_3d_inliers;
    return false;
  }
  if (!RefineAbsolutePose(
          refine_options, inliers, points2D, points3D, &cam_from_world, &camera)) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", pnp refine failed";
    return false;
  }

  // Grow matches by projecting the whole loop-side local map.
  LoadLoopMap();
  const size_t num_guided = SearchLoopByProjection(current,
                                                   reconstruction,
                                                   cam_from_world,
                                                   loop_ids,
                                                   loop_desc,
                                                   desc_current,
                                                   options_.guided_radius_px,
                                                   scale_check,
                                                   &used_current,
                                                   &used_points,
                                                   &corrs);
  FillPoints();
  // Reprojection inliers whose keypoint scale disagrees with the loop-side
  // observations are dropped; returns how many.
  const auto RemoveScaleOutliers = [&]() {
    size_t num_removed = 0;
    for (size_t i = 0; i < corrs.size(); ++i) {
      if (inliers[i] && !scale_check.Consistent(reconstruction,
                                                image_id,
                                                corrs[i].idx_current,
                                                corrs[i].point3D_id,
                                                cam_from_world,
                                                camera)) {
        inliers[i] = 0;
        ++num_removed;
      }
    }
    return num_removed;
  };
  inliers = ComputePnPInliers(
      corrs, reconstruction, cam_from_world, camera, options_.pnp_max_error);
  size_t num_scale_rejected = RemoveScaleOutliers();
  if (CountInliers(inliers) >=
          static_cast<size_t>(options_.min_num_3d_inliers) &&
      RefineAbsolutePose(refine_options,
                         inliers,
                         points2D,
                         points3D,
                         &cam_from_world,
                         &camera)) {
    inliers = ComputePnPInliers(
        corrs, reconstruction, cam_from_world, camera, options_.pnp_max_error);
    num_scale_rejected = RemoveScaleOutliers();
  }
  const size_t num_pnp_inliers = CountInliers(inliers);

  const bool has_pose_before = current.HasPose();
  if (has_pose_before) {
    const Rigid3d& pose_before = current.CamFromWorld();
    loop_pair->correction_m =
        (cam_from_world.TgtOriginInSrc() - pose_before.TgtOriginInSrc()).norm();
    loop_pair->correction_deg =
        RotationDiffDeg(cam_from_world.rotation(), pose_before.rotation());
  } else {
    loop_pair->correction_m = 0.0;
    loop_pair->correction_deg = 0.0;
  }
  double tdir_vs_essential_deg = -1.0;
  loop_pair->rot_vs_essential_deg = RotationVsEssentialDeg(
      geometry, candidate, current, cam_from_world, &tdir_vs_essential_deg);
  const double baseline_m =
      (cam_from_world.TgtOriginInSrc() -
       candidate.CamFromWorld().TgtOriginInSrc())
          .norm();
  LOG(INFO) << "OnlineLoopCloser: pnp " << image_id << "-" << candidate_id
            << " 2D-3D=" << corrs.size() << " island=" << num_island
            << " guided=" << num_guided
            << " ransac_inliers=" << num_ransac_inliers
            << " inliers=" << num_pnp_inliers
            << " scale_rejected=" << num_scale_rejected
            << " loop_map=" << loop_ids.size()
            << " baseline_m=" << baseline_m
            << " correction_m=" << loop_pair->correction_m
            << " correction_deg=" << loop_pair->correction_deg
            << " rot_vs_E_deg=" << loop_pair->rot_vs_essential_deg
            << " tdir_vs_E_deg=" << tdir_vs_essential_deg;

  if (num_pnp_inliers < gate.min_inliers) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", pnp weak inliers=" << num_pnp_inliers;
    return false;
  }
  if (gate.max_rot_vs_essential_deg > 0.0 &&
      loop_pair->rot_vs_essential_deg > gate.max_rot_vs_essential_deg) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", pnp rotation disagrees with essential by "
              << loop_pair->rot_vs_essential_deg << " deg";
    return false;
  }
  if (gate.check_correction &&
      (loop_pair->correction_m > options_.max_correction_m ||
       loop_pair->correction_deg > options_.max_correction_deg)) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", correction too large " << loop_pair->correction_m
              << " m / " << loop_pair->correction_deg << " deg";
    return false;
  }

  double pnp_rot_sigma = 0.0;
  double pnp_trans_sigma = 0.0;
  const bool has_pnp_sigma = EstimatePnPSigmas(corrs,
                                               inliers,
                                               reconstruction,
                                               cam_from_world,
                                               camera,
                                               &pnp_rot_sigma,
                                               &pnp_trans_sigma);
  const double min_rot_sigma =
      DegToRad(std::max(options_.pose_graph_loop_min_rot_sigma_deg, 1e-3));
  const double min_trans_sigma =
      std::max(options_.pose_graph_loop_min_trans_sigma_m, 1e-4);
  // Degenerate PnP information: keep the loop but trust it like odometry.
  loop_pair->rot_sigma =
      has_pnp_sigma
          ? std::max(kLoopSigmaInflation * pnp_rot_sigma, min_rot_sigma)
          : DegToRad(std::max(options_.pose_graph_odom_rot_sigma_deg, 1e-3));
  loop_pair->trans_sigma =
      has_pnp_sigma
          ? std::max(kLoopSigmaInflation * pnp_trans_sigma, min_trans_sigma)
          : std::max(options_.pose_graph_odom_trans_sigma_m, 1e-4) +
                options_.pose_graph_odom_trans_sigma_ratio * baseline_m;

  loop_pair->has_loop_pose = true;
  loop_pair->loop_cam_from_world = cam_from_world;
  loop_pair->loop_observations.clear();
  loop_pair->loop_observations.reserve(num_pnp_inliers);
  for (size_t i = 0; i < corrs.size(); ++i) {
    if (inliers[i]) {
      loop_pair->loop_observations.emplace_back(corrs[i].idx_current,
                                                corrs[i].point3D_id);
    }
  }
  LOG(INFO) << "OnlineLoopCloser: accept " << image_id << "-" << candidate_id
            << " by pnp inliers=" << num_pnp_inliers
            << " correction_m=" << loop_pair->correction_m
            << " correction_deg=" << loop_pair->correction_deg
            << " pnp_sigma_deg=" << RadToDeg(pnp_rot_sigma)
            << " pnp_sigma_m=" << pnp_trans_sigma
            << " edge_sigma_deg=" << RadToDeg(loop_pair->rot_sigma)
            << " edge_sigma_m=" << loop_pair->trans_sigma;
  return true;
}

}  // namespace colmap
