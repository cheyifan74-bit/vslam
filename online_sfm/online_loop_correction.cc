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

#include "colmap/feature/types.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "online_sfm/online_loop_closer_internal.h"
#include "online_sfm/online_pose_graph.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database_session.h"
#include "colmap/scene/track.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Geometry>

namespace colmap {

using online_loop_internal::DescriptorL2Sq;
using online_loop_internal::InsideImage;
using online_loop_internal::kLoopDescMaxL2Sq;
using online_loop_internal::kMaxFuseRefDescriptors;
using online_loop_internal::LoopScaleCheck;
using online_loop_internal::ProjectCam;
using online_loop_internal::RotationDiffDeg;

namespace {

// Historical edges whose residual is farther than this multiple of the
// correction gate are treated as broken measurements and left out.
constexpr double kHistoricalResidualFactor = 2.0;
// A corrected frame joins the fusion / loop-BA window when at least this many
// loop-side map points project inside it.
constexpr size_t kMinOverlapProjections = 50;

// RMSE between `before` and `after` camera centers once `after` is rigidly
// aligned onto `before`: yaw about +z plus translation, or full SE3.
double AlignedTrajectoryRmse(const std::vector<Eigen::Vector3d>& before,
                             const std::vector<Eigen::Vector3d>& after,
                             const bool yaw_only) {
  const size_t n = before.size();
  if (n == 0 || after.size() != n) {
    return 0.0;
  }
  Eigen::Matrix3Xd a(3, n);
  Eigen::Matrix3Xd b(3, n);
  for (size_t i = 0; i < n; ++i) {
    a.col(i) = before[i];
    b.col(i) = after[i];
  }
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  if (yaw_only) {
    const Eigen::Vector3d mean_a = a.rowwise().mean();
    const Eigen::Vector3d mean_b = b.rowwise().mean();
    const Eigen::Matrix3Xd ca = a.colwise() - mean_a;
    const Eigen::Matrix3Xd cb = b.colwise() - mean_b;
    const double s = (cb.row(0).cwiseProduct(ca.row(1)) -
                      cb.row(1).cwiseProduct(ca.row(0)))
                         .sum();
    const double c = (cb.row(0).cwiseProduct(ca.row(0)) +
                      cb.row(1).cwiseProduct(ca.row(1)))
                         .sum();
    R = Eigen::AngleAxisd(std::atan2(s, c), Eigen::Vector3d::UnitZ())
            .toRotationMatrix();
    t = mean_a - R * mean_b;
  } else if (n >= 3) {
    const Eigen::Matrix4d T = Eigen::umeyama(b, a, /*with_scaling=*/false);
    R = T.topLeftCorner<3, 3>();
    t = T.topRightCorner<3, 1>();
  } else {
    t = a.rowwise().mean() - b.rowwise().mean();
  }
  const Eigen::Matrix3Xd residual = a - ((R * b).colwise() + t);
  return std::sqrt(residual.colwise().squaredNorm().mean());
}

bool ImageIdsWithin(const image_t a, const image_t b, const int gap) {
  if (gap < 0) {
    return false;
  }
  const image_t delta = a > b ? a - b : b - a;
  return delta <= static_cast<image_t>(gap);
}

bool SetImagePose(Image& image, const Rigid3d& cam_from_world) {
  if (!image.HasFramePtr()) {
    return false;
  }
  image.FramePtr()->SetCamFromWorld(image.CameraId(), cam_from_world);
  return true;
}

}  // namespace

point3D_t ResolveMergedPoint(
    const std::unordered_map<point3D_t, point3D_t>& remap, point3D_t id) {
  while (true) {
    const auto it = remap.find(id);
    if (it == remap.end()) {
      return id;
    }
    id = it->second;
  }
}

bool MergeKeepLoopPoint(ObservationManager* obs_manager,
                        Reconstruction& reconstruction,
                        std::unordered_map<point3D_t, point3D_t>* remap,
                        const point3D_t cur_pid,
                        const point3D_t loop_pid) {
  const Eigen::Vector3d loop_xyz = reconstruction.Point3D(loop_pid).xyz;
  try {
    const point3D_t merged = obs_manager->MergePoints3D(cur_pid, loop_pid);
    reconstruction.Point3D(merged).xyz = loop_xyz;
    (*remap)[cur_pid] = merged;
    (*remap)[loop_pid] = merged;
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: merge " << cur_pid << "+" << loop_pid
                 << " failed: " << e.what();
    return false;
  }
}

bool AttachLoopObservation(ObservationManager* obs_manager,
                           const point3D_t loop_pid,
                           const image_t image_id,
                           const point2D_t idx) {
  try {
    obs_manager->AddObservation(loop_pid, TrackElement(image_id, idx));
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: attach point " << loop_pid
                 << " to image " << image_id << " failed: " << e.what();
    return false;
  }
}

struct ProjectionFuseStats {
  size_t num_fused = 0;
  size_t num_attached = 0;
  size_t num_rejected_dist = 0;
  size_t num_rejected_desc = 0;
  size_t num_rejected_scale = 0;
};

// Project loop-side points into `image_ids` and merge or attach descriptor
// matches inside the fuse radius.
ProjectionFuseStats FuseLoopMapByProjection(
    const std::string& database_path,
    const OnlineLoopCloserOptions& options,
    Reconstruction& reconstruction,
    ObservationManager* obs_manager,
    const std::unordered_set<point3D_t>& loop_point_ids,
    const std::vector<image_t>& image_ids,
    std::unordered_map<point3D_t, point3D_t>* remap) {
  ProjectionFuseStats stats;
  if (obs_manager == nullptr || image_ids.empty() || loop_point_ids.empty()) {
    return stats;
  }

  std::unordered_map<image_t, FeatureDescriptors> descriptors;
  std::unordered_map<point3D_t, std::vector<TrackElement>> loop_refs;
  LoopScaleCheck scale_check(options.scale_check_max_ratio);
  try {
    DatabaseSession session(database_path);
    const auto Load = [&](const image_t id) {
      if (descriptors.count(id) == 0 && session->ExistsDescriptors(id)) {
        descriptors.emplace(id, session->ReadDescriptors(id));
      }
      scale_check.Load(*session, id);
    };
    for (const image_t id : image_ids) {
      Load(id);
    }
    std::vector<TrackElement> track_elements;
    for (const point3D_t point3D_id_raw : loop_point_ids) {
      const point3D_t point3D_id = ResolveMergedPoint(*remap, point3D_id_raw);
      if (!reconstruction.ExistsPoint3D(point3D_id) ||
          loop_refs.count(point3D_id_raw) > 0) {
        continue;
      }
      track_elements = reconstruction.Point3D(point3D_id).track.Elements();
      std::sort(track_elements.begin(),
                track_elements.end(),
                [](const TrackElement& a, const TrackElement& b) {
                  return a.image_id > b.image_id;
                });
      std::vector<TrackElement> refs;
      for (const TrackElement& el : track_elements) {
        if (refs.size() >= kMaxFuseRefDescriptors) {
          break;
        }
        if (!reconstruction.ExistsImage(el.image_id) ||
            !reconstruction.Image(el.image_id).HasPose()) {
          continue;
        }
        Load(el.image_id);
        const auto ref_it = descriptors.find(el.image_id);
        if (ref_it == descriptors.end() ||
            static_cast<int>(el.point2D_idx) >= ref_it->second.data.rows()) {
          continue;
        }
        refs.push_back(el);
      }
      if (!refs.empty()) {
        loop_refs.emplace(point3D_id_raw, std::move(refs));
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to read fusion descriptors: "
                 << e.what();
    return stats;
  }

  const double fuse_radius_sq = options.fuse_radius_px * options.fuse_radius_px;
  for (const image_t id : image_ids) {
    if (!reconstruction.ExistsImage(id)) {
      continue;
    }
    Image& image = reconstruction.Image(id);
    const auto desc_it = descriptors.find(id);
    if (!image.HasPose() || !image.HasCameraPtr() ||
        desc_it == descriptors.end()) {
      continue;
    }
    const Camera& camera = *image.CameraPtr();
    const Rigid3d cam_from_world = image.CamFromWorld();
    const double cell = std::max(1.0, options.fuse_radius_px);
    const int grid_w = static_cast<int>(camera.width / cell) + 1;
    const int grid_h = static_cast<int>(camera.height / cell) + 1;
    std::vector<std::vector<point2D_t>> grid(grid_w * grid_h);
    for (point2D_t i = 0; i < image.NumPoints2D(); ++i) {
      const Eigen::Vector2d& xy = image.Point2D(i).xy;
      const int gx = std::clamp(static_cast<int>(xy.x() / cell), 0, grid_w - 1);
      const int gy = std::clamp(static_cast<int>(xy.y() / cell), 0, grid_h - 1);
      grid[gy * grid_w + gx].push_back(i);
    }
    std::unordered_set<point2D_t> used;
    for (const auto& [loop_pid_raw, refs] : loop_refs) {
      const point3D_t loop_pid = ResolveMergedPoint(*remap, loop_pid_raw);
      if (!reconstruction.ExistsPoint3D(loop_pid) ||
          image.HasPoint3D(loop_pid) || refs.empty()) {
        continue;
      }
      const Eigen::Vector3d loop_xyz = reconstruction.Point3D(loop_pid).xyz;
      const auto proj = ProjectCam(camera, cam_from_world * loop_xyz);
      if (!proj || !InsideImage(camera, *proj)) {
        continue;
      }
      float best_l2 = kLoopDescMaxL2Sq;
      point2D_t best_idx = kInvalidPoint2DIdx;
      bool any_in_radius = false;
      bool any_scale_ok = false;
      const int pgx = static_cast<int>(proj->x() / cell);
      const int pgy = static_cast<int>(proj->y() / cell);
      for (int gy = std::max(0, pgy - 1); gy <= std::min(grid_h - 1, pgy + 1);
           ++gy) {
        for (int gx = std::max(0, pgx - 1); gx <= std::min(grid_w - 1, pgx + 1);
             ++gx) {
          for (const point2D_t i : grid[gy * grid_w + gx]) {
            if (used.count(i) > 0 ||
                (image.Point2D(i).xy - *proj).squaredNorm() >= fuse_radius_sq) {
              continue;
            }
            any_in_radius = true;
            if (!scale_check.Consistent(
                    reconstruction, id, i, loop_pid, cam_from_world, camera)) {
              continue;
            }
            any_scale_ok = true;
            float l2 = std::numeric_limits<float>::infinity();
            for (const TrackElement& ref : refs) {
              const auto ref_desc_it = descriptors.find(ref.image_id);
              if (ref_desc_it == descriptors.end()) {
                continue;
              }
              l2 = std::min(l2,
                            DescriptorL2Sq(ref_desc_it->second,
                                           ref.point2D_idx,
                                           desc_it->second,
                                           i));
            }
            if (l2 < best_l2) {
              best_l2 = l2;
              best_idx = i;
            }
          }
        }
      }
      if (best_idx == kInvalidPoint2DIdx) {
        if (any_scale_ok) {
          ++stats.num_rejected_desc;
        } else if (any_in_radius) {
          ++stats.num_rejected_scale;
        }
        continue;
      }
      const Point2D& point2D = image.Point2D(best_idx);
      if (point2D.HasPoint3D()) {
        const point3D_t cur_pid = point2D.point3D_id;
        if (cur_pid == loop_pid || !reconstruction.ExistsPoint3D(cur_pid)) {
          continue;
        }
        if ((reconstruction.Point3D(cur_pid).xyz - loop_xyz).norm() >
            options.fuse_max_point_distance) {
          ++stats.num_rejected_dist;
          continue;
        }
        if (MergeKeepLoopPoint(
                obs_manager, reconstruction, remap, cur_pid, loop_pid)) {
          used.insert(best_idx);
          ++stats.num_fused;
        }
      } else if (AttachLoopObservation(obs_manager, loop_pid, id, best_idx)) {
        used.insert(best_idx);
        ++stats.num_attached;
      }
    }
  }
  return stats;
}

bool OnlineLoopCloser::CorrectLoop(Reconstruction& reconstruction,
                                   ObservationManager* obs_manager,
                                   const OnlineLoopPair& loop_pair) {
  if (obs_manager == nullptr || !loop_pair.has_loop_pose) {
    return false;
  }
  const image_t current_id = loop_pair.image_id1;
  const image_t loop_id = loop_pair.image_id2;
  if (!reconstruction.ExistsImage(current_id) ||
      !reconstruction.ExistsImage(loop_id) ||
      !reconstruction.Image(current_id).HasPose() ||
      !reconstruction.Image(loop_id).HasPose()) {
    return false;
  }

  // Old map = loop candidate, its local map, and everything registered before.
  const std::vector<image_t> loop_map_ids =
      LoopMapImages(loop_id, current_id, reconstruction);
  std::unordered_set<image_t> anchor_set =
      CovisibilityGroup(loop_id, reconstruction);
  anchor_set.insert(loop_map_ids.begin(), loop_map_ids.end());
  anchor_set.erase(current_id);
  image_t anchor_max = loop_id;
  for (const image_t id : anchor_set) {
    anchor_max = std::max(anchor_max, id);
  }

  std::vector<image_t> registered = reconstruction.RegImageIds();
  std::sort(registered.begin(), registered.end());
  std::vector<image_t> variable_ids;
  image_t boundary_id = kInvalidImageId;
  for (const image_t id : registered) {
    if (id <= anchor_max || anchor_set.count(id) > 0) {
      if (variable_ids.empty()) {
        boundary_id = id;
      }
      continue;
    }
    if (id <= current_id && reconstruction.Image(id).HasFramePtr()) {
      variable_ids.push_back(id);
    }
  }
  const std::unordered_set<image_t> variable_set(variable_ids.begin(),
                                                 variable_ids.end());
  if (variable_set.count(current_id) == 0) {
    LOG(INFO) << "OnlineLoopCloser: CorrectLoop skip " << current_id << "-"
              << loop_id << ", current inside old map (anchor_max="
              << anchor_max << ")";
    return false;
  }

  // Stage A: pose graph (odometry + covisibility + loop edges).
  std::unordered_map<image_t, Rigid3d> poses_old;
  for (const image_t id : variable_ids) {
    poses_old[id] = reconstruction.Image(id).CamFromWorld();
  }
  poses_old[loop_id] = reconstruction.Image(loop_id).CamFromWorld();
  if (boundary_id != kInvalidImageId) {
    poses_old[boundary_id] = reconstruction.Image(boundary_id).CamFromWorld();
  }
  const auto Rel = [&](const image_t i, const image_t j) {
    return poses_old.at(j) * Inverse(poses_old.at(i));
  };

  std::vector<PoseGraphEdge> edges;
  size_t num_vio_odom_edges = 0;
  image_t prev_id = boundary_id;
  for (const image_t id : variable_ids) {
    if (prev_id != kInvalidImageId) {
      std::optional<Rigid3d> vio_rel;
      if (options_.pose_graph_vio_odometry && vio_relative_pose_) {
        vio_rel = vio_relative_pose_(id, prev_id);
      }
      if (vio_rel) {
        ++num_vio_odom_edges;
      }
      edges.push_back(MakeOdometryEdge(
          prev_id, id, vio_rel ? *vio_rel : Rel(prev_id, id), 1.0));
    }
    prev_id = id;
  }
  // Old-map frames covisible with corrected frames (revisited areas) join as
  // fixed nodes, so the shared points they keep stay consistent.
  std::unordered_set<image_t> fixed_ids = {loop_id};
  if (boundary_id != kInvalidImageId) {
    fixed_ids.insert(boundary_id);
  }
  size_t num_covis_edges = 0;
  size_t num_old_covis_edges = 0;
  for (const image_t id : variable_ids) {
    for (const auto& [other_id, count] :
         CountCovisibilityPoints(id, reconstruction)) {
      if (count < options_.pose_graph_min_covis) {
        continue;
      }
      // More shared points, tighter edge; at most 2x tighter than odometry.
      const double covis_scale = std::clamp(
          std::sqrt(static_cast<double>(std::max(options_.pose_graph_min_covis, 1)) /
                    static_cast<double>(count)),
          0.5,
          1.0);
      if (variable_set.count(other_id) > 0) {
        if (other_id <= id) {
          continue;
        }
        edges.push_back(
            MakeOdometryEdge(id, other_id, Rel(id, other_id), covis_scale));
        ++num_covis_edges;
        continue;
      }
      if (other_id > current_id || !reconstruction.ExistsImage(other_id) ||
          !reconstruction.Image(other_id).HasPose()) {
        continue;
      }
      if (poses_old.count(other_id) == 0) {
        poses_old[other_id] = reconstruction.Image(other_id).CamFromWorld();
      }
      fixed_ids.insert(other_id);
      edges.push_back(
          MakeOdometryEdge(other_id, id, Rel(other_id, id), covis_scale));
      ++num_old_covis_edges;
    }
  }
  // Freeze the PnP relative pose against the loop frame's pre-correction pose.
  // Later graphs replay this measurement; they do not rebuild it from poses.
  const LoopConstraint current_loop{
      loop_id,
      current_id,
      loop_pair.loop_cam_from_world * Inverse(poses_old[loop_id]),
      loop_pair.loop_observations.size(),
      loop_pair.rot_sigma,
      loop_pair.trans_sigma};
  const size_t loop_edge_begin = edges.size();
  std::vector<LoopConstraint> loop_edge_constraints;
  size_t num_hist_loop_edges = AppendLoopEdges(reconstruction,
                                               variable_set,
                                               current_loop,
                                               &poses_old,
                                               &fixed_ids,
                                               &edges,
                                               &loop_edge_constraints);

  // Solve, then drop the historical loop edge most inconsistent with the rest
  // (whitened chi-square above the 99% quantile) and solve again. If the
  // current loop is the outlier, the correction is rejected.
  const PoseGraphOptions graph_options = MakePoseGraphOptions();
  const double outlier_chi2 = PoseGraphEdgeChi2Quantile(graph_options, 0.99);
  std::unordered_map<image_t, Rigid3d> poses;
  PoseGraphSummary pg;
  size_t num_dropped_loop_edges = 0;
  std::vector<LoopConstraint> dropped_loop_constraints;
  double current_loop_chi2 = -1.0;
  while (true) {
    poses = poses_old;
    pg = OptimizePoseGraph(graph_options, edges, fixed_ids, &poses);
    if (!pg.success) {
      LOG(WARNING) << "OnlineLoopCloser: CorrectLoop pose graph failed "
                   << current_id << "-" << loop_id << " nodes=" << pg.num_nodes
                   << " edges=" << pg.num_edges;
      return false;
    }
    size_t worst = edges.size();
    double worst_chi2 = outlier_chi2;
    for (size_t k = loop_edge_begin; k < edges.size(); ++k) {
      const LoopConstraint& c = loop_edge_constraints[k - loop_edge_begin];
      if (c.map_id == current_loop.map_id &&
          c.query_id == current_loop.query_id) {
        current_loop_chi2 = pg.edge_chi2[k];
      }
      if (pg.edge_chi2[k] > worst_chi2) {
        worst_chi2 = pg.edge_chi2[k];
        worst = k;
      }
    }
    if (worst == edges.size()) {
      break;
    }
    const LoopConstraint outlier = loop_edge_constraints[worst - loop_edge_begin];
    if (outlier.map_id == current_loop.map_id &&
        outlier.query_id == current_loop.query_id) {
      LOG(WARNING) << "OnlineLoopCloser: CorrectLoop reject " << current_id
                   << "-" << loop_id << ", loop edge chi2=" << worst_chi2
                   << " > " << outlier_chi2
                   << " against odometry and historical loops";
      return false;
    }
    LOG(INFO) << "OnlineLoopCloser: drop historical loop " << outlier.query_id
              << "-" << outlier.map_id << " chi2=" << worst_chi2 << " > "
              << outlier_chi2;
    dropped_loop_constraints.push_back(outlier);
    edges.erase(edges.begin() + static_cast<std::ptrdiff_t>(worst));
    loop_edge_constraints.erase(
        loop_edge_constraints.begin() +
        static_cast<std::ptrdiff_t>(worst - loop_edge_begin));
    --num_hist_loop_edges;
    ++num_dropped_loop_edges;
  }

  // Trajectory prior: frames between the loop and the current frame that the
  // graph does not move keep their pose.
  double rover_score = -1.0;
  double rover_threshold = -1.0;
  double rover_path_m = 0.0;
  if (options_.rover_enabled) {
    std::vector<Eigen::Vector3d> centers_before;
    std::vector<Eigen::Vector3d> centers_after;
    for (const image_t id : registered) {
      if (id < loop_id || id > current_id ||
          !reconstruction.Image(id).HasPose()) {
        continue;
      }
      const auto it_old = poses_old.find(id);
      const Rigid3d& before = it_old != poses_old.end()
                                  ? it_old->second
                                  : reconstruction.Image(id).CamFromWorld();
      const auto it_new = poses.find(id);
      const Rigid3d& after = it_new != poses.end() ? it_new->second : before;
      if (!centers_before.empty()) {
        rover_path_m += (before.TgtOriginInSrc() - centers_before.back()).norm();
      }
      centers_before.push_back(before.TgtOriginInSrc());
      centers_after.push_back(after.TgtOriginInSrc());
    }
    rover_score = AlignedTrajectoryRmse(
        centers_before, centers_after, graph_options.four_dof);
    rover_threshold =
        std::max(options_.rover_min_threshold_m,
                 options_.rover_threshold_ratio * rover_path_m);
    if (rover_score > rover_threshold) {
      LOG(WARNING) << "OnlineLoopCloser: CorrectLoop reject " << current_id
                   << "-" << loop_id << ", trajectory deformation "
                   << rover_score << " m > " << rover_threshold
                   << " m (frames=" << centers_before.size()
                   << " path_m=" << rover_path_m << ")";
      return false;
    }
  }
  const Rigid3d& cur_after = poses.at(current_id);
  const double cur_vs_pnp_m = (cur_after.TgtOriginInSrc() -
                               loop_pair.loop_cam_from_world.TgtOriginInSrc())
                                  .norm();
  const double cur_vs_pnp_deg = RotationDiffDeg(
      cur_after.rotation(), loop_pair.loop_cam_from_world.rotation());

  // Stage B: move each point with its reference (first observing) frame, so
  // points first seen by the old map keep their position.
  std::unordered_set<point3D_t> visited;
  size_t num_points_moved = 0;
  size_t num_points_kept = 0;
  for (const image_t id : variable_ids) {
    for (const Point2D& point2D : reconstruction.Image(id).Points2D()) {
      if (!point2D.HasPoint3D() ||
          !reconstruction.ExistsPoint3D(point2D.point3D_id) ||
          !visited.insert(point2D.point3D_id).second) {
        continue;
      }
      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      image_t ref_id = kInvalidImageId;
      for (const TrackElement& el : point3D.track.Elements()) {
        if (reconstruction.ExistsImage(el.image_id) &&
            reconstruction.Image(el.image_id).HasPose() &&
            (ref_id == kInvalidImageId || el.image_id < ref_id)) {
          ref_id = el.image_id;
        }
      }
      if (variable_set.count(ref_id) == 0) {
        ++num_points_kept;
        continue;
      }
      point3D.xyz =
          Inverse(poses.at(ref_id)) * (poses_old.at(ref_id) * point3D.xyz);
      ++num_points_moved;
    }
  }
  size_t num_poses_corrected = 0;
  for (const image_t id : variable_ids) {
    if (SetImagePose(reconstruction.Image(id), poses.at(id))) {
      ++num_poses_corrected;
    }
  }

  // Loop-side map points. Descriptor references are chosen later from the
  // newest observations on each track.
  std::unordered_set<point3D_t> loop_point_ids;
  for (const image_t id : anchor_set) {
    if (!reconstruction.ExistsImage(id)) {
      continue;
    }
    const Image& image = reconstruction.Image(id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D() &&
          reconstruction.ExistsPoint3D(point2D.point3D_id)) {
        loop_point_ids.insert(point2D.point3D_id);
      }
    }
  }

  // Overlap window: corrected frames onto which enough of the loop-side map
  // projects, strongest first, capped by max_correct_connected.
  std::vector<std::pair<size_t, image_t>> overlap;
  for (const image_t id : variable_ids) {
    const Image& image = reconstruction.Image(id);
    if (!image.HasCameraPtr()) {
      continue;
    }
    const Camera& camera = *image.CameraPtr();
    const Rigid3d cam_from_world = image.CamFromWorld();
    size_t num_visible = 0;
    for (const point3D_t loop_pid : loop_point_ids) {
      const auto proj = ProjectCam(
          camera, cam_from_world * reconstruction.Point3D(loop_pid).xyz);
      if (proj && InsideImage(camera, *proj)) {
        ++num_visible;
      }
    }
    if (id == current_id || num_visible >= kMinOverlapProjections) {
      overlap.emplace_back(id == current_id ? loop_point_ids.size() + 1
                                            : num_visible,
                           id);
    }
  }
  std::sort(overlap.begin(), overlap.end(), std::greater<>());
  std::vector<image_t> connected;
  for (const auto& [num_visible, id] : overlap) {
    if (static_cast<int>(connected.size()) >=
        std::max(1, options_.max_correct_connected)) {
      break;
    }
    connected.push_back(id);
  }

  std::unordered_map<point3D_t, point3D_t> remap;

  // Stage C1: PnP inliers of the current frame.
  size_t num_fused_direct = 0;
  size_t num_attached_direct = 0;
  {
    const Image& current = reconstruction.Image(current_id);
    for (const auto& [idx, loop_pid_raw] : loop_pair.loop_observations) {
      const point3D_t loop_pid = ResolveMergedPoint(remap, loop_pid_raw);
      if (idx >= current.NumPoints2D() ||
          !reconstruction.ExistsPoint3D(loop_pid)) {
        continue;
      }
      const Point2D& point2D = current.Point2D(idx);
      if (point2D.HasPoint3D()) {
        if (point2D.point3D_id != loop_pid &&
            MergeKeepLoopPoint(obs_manager,
                               reconstruction,
                               &remap,
                               point2D.point3D_id,
                               loop_pid)) {
          ++num_fused_direct;
        }
      } else if (!current.HasPoint3D(loop_pid) &&
                 AttachLoopObservation(obs_manager, loop_pid, current_id, idx)) {
        ++num_attached_direct;
      }
    }
  }

  // Stage C2: project the loop-side map into the overlap window.
  const ProjectionFuseStats proj = FuseLoopMapByProjection(database_path_,
                                                           options_,
                                                           reconstruction,
                                                           obs_manager,
                                                           loop_point_ids,
                                                           connected,
                                                           &remap);
  const size_t num_fused_proj = proj.num_fused;
  const size_t num_attached_proj = proj.num_attached;
  const size_t num_rejected_dist = proj.num_rejected_dist;
  const size_t num_rejected_desc = proj.num_rejected_desc;

  last_correction_.current_id = current_id;
  last_correction_.loop_id = loop_id;
  // Every pose-graph-moved frame is refined and filtered; `connected` only
  // limits fusion. The same window is projected again after loop BA.
  last_correction_.variable_ids = variable_ids;
  last_correction_.anchor_ids = loop_map_ids;
  last_correction_.fusion_ids = connected;
  last_correction_.loop_point_image_ids.assign(anchor_set.begin(),
                                               anchor_set.end());

  LOG(INFO) << "OnlineLoopCloser: CorrectLoop " << current_id << "-" << loop_id
            << " variable=[" << variable_ids.front() << ".."
            << variable_ids.back() << "](" << variable_ids.size() << ")"
            << " boundary=" << boundary_id << " anchor_max=" << anchor_max
            << " edges=" << pg.num_edges
            << " vio_odom_edges=" << num_vio_odom_edges
            << " covis_edges=" << num_covis_edges
            << " old_covis_edges=" << num_old_covis_edges
            << " hist_loop_edges=" << num_hist_loop_edges
            << " dropped_loop_edges=" << num_dropped_loop_edges
            << " loop_chi2=" << current_loop_chi2
            << " rover_score=" << rover_score << "/" << rover_threshold
            << " rover_path_m=" << rover_path_m
            << " dof=" << (graph_options.four_dof ? 4 : 6)
            << " fixed_nodes=" << fixed_ids.size()
            << " cost=" << pg.initial_cost << "->" << pg.final_cost
            << " cur_vs_pnp_m=" << cur_vs_pnp_m
            << " cur_vs_pnp_deg=" << cur_vs_pnp_deg
            << " poses=" << num_poses_corrected
            << " points_moved=" << num_points_moved
            << " points_kept=" << num_points_kept
            << " fuse_direct=" << num_fused_direct
            << " attach_direct=" << num_attached_direct
            << " fuse_proj=" << num_fused_proj
            << " attach_proj=" << num_attached_proj
            << " fuse_rejected_dist=" << num_rejected_dist
            << " fuse_rejected_desc=" << num_rejected_desc
            << " fuse_rejected_scale=" << proj.num_rejected_scale
            << " window=" << connected.size() << " ["
            << *std::min_element(connected.begin(), connected.end()) << ".."
            << *std::max_element(connected.begin(), connected.end()) << "]";
  if (num_poses_corrected == 0) {
    return false;
  }
  for (const LoopConstraint& dropped : dropped_loop_constraints) {
    ForgetLoopConstraint(dropped.map_id, dropped.query_id);
  }
  RememberLoopConstraint(current_loop);
  return true;
}

void OnlineLoopCloser::FuseAfterLoopBA(Reconstruction& reconstruction,
                                       ObservationManager* obs_manager) {
  if (obs_manager == nullptr || last_correction_.fusion_ids.empty() ||
      last_correction_.loop_point_image_ids.empty()) {
    return;
  }

  std::unordered_set<point3D_t> loop_point_ids;
  for (const image_t id : last_correction_.loop_point_image_ids) {
    if (!reconstruction.ExistsImage(id)) {
      continue;
    }
    for (const Point2D& point2D : reconstruction.Image(id).Points2D()) {
      if (point2D.HasPoint3D() &&
          reconstruction.ExistsPoint3D(point2D.point3D_id)) {
        loop_point_ids.insert(point2D.point3D_id);
      }
    }
  }

  std::unordered_map<point3D_t, point3D_t> remap;
  const ProjectionFuseStats proj =
      FuseLoopMapByProjection(database_path_,
                              options_,
                              reconstruction,
                              obs_manager,
                              loop_point_ids,
                              last_correction_.fusion_ids,
                              &remap);
  LOG(INFO) << "OnlineLoopCloser: post-BA fuse " << last_correction_.current_id
            << "-" << last_correction_.loop_id
            << " window=" << last_correction_.fusion_ids.size()
            << " points=" << loop_point_ids.size()
            << " fuse_proj=" << proj.num_fused
            << " attach_proj=" << proj.num_attached
            << " fuse_rejected_dist=" << proj.num_rejected_dist
            << " fuse_rejected_desc=" << proj.num_rejected_desc
            << " fuse_rejected_scale=" << proj.num_rejected_scale;
}

PoseGraphOptions OnlineLoopCloser::MakePoseGraphOptions() const {
  PoseGraphOptions graph_options;
  graph_options.four_dof = options_.pose_graph_4dof;
  graph_options.num_threads = options_.pose_graph_num_threads;
  return graph_options;
}

PoseGraphEdge OnlineLoopCloser::MakeOdometryEdge(const image_t i,
                                                 const image_t j,
                                                 const Rigid3d& j_from_i,
                                                 const double sigma_scale) const {
  PoseGraphEdge edge;
  edge.image_id_i = i;
  edge.image_id_j = j;
  edge.j_from_i = j_from_i;
  edge.rot_sigma =
      sigma_scale *
      DegToRad(std::max(options_.pose_graph_odom_rot_sigma_deg, 1e-3));
  edge.trans_sigma =
      sigma_scale * (std::max(options_.pose_graph_odom_trans_sigma_m, 1e-4) +
                     options_.pose_graph_odom_trans_sigma_ratio *
                         j_from_i.translation().norm());
  return edge;
}

bool OnlineLoopCloser::SameLoopCluster(
    const image_t query_a,
    const image_t map_a,
    const image_t query_b,
    const image_t map_b,
    const Reconstruction& reconstruction) const {
  if (!ImageIdsWithin(query_a, query_b, options_.cooldown_num_images)) {
    return false;
  }
  if (map_a == map_b) {
    return true;
  }
  return CovisibilityGroup(map_a, reconstruction).count(map_b) > 0;
}

bool OnlineLoopCloser::HistoricalResidualPlausible(
    const Rigid3d& measured, const Rigid3d& current_rel) const {
  const Rigid3d error = Inverse(measured) * current_rel;
  const double trans_m = error.translation().norm();
  const double rot_deg =
      RotationDiffDeg(measured.rotation(), current_rel.rotation());
  if (options_.max_correction_m > 0.0 &&
      trans_m > kHistoricalResidualFactor * options_.max_correction_m) {
    return false;
  }
  if (options_.max_correction_deg > 0.0 &&
      rot_deg > kHistoricalResidualFactor * options_.max_correction_deg) {
    return false;
  }
  return true;
}

size_t OnlineLoopCloser::AppendLoopEdges(
    const Reconstruction& reconstruction,
    const std::unordered_set<image_t>& variable_set,
    const LoopConstraint& current,
    std::unordered_map<image_t, Rigid3d>* poses_old,
    std::unordered_set<image_t>* fixed_ids,
    std::vector<PoseGraphEdge>* edges,
    std::vector<LoopConstraint>* edge_constraints) {
  std::vector<LoopConstraint> replay;
  replay.reserve(loop_constraints_.size());
  for (const LoopConstraint& stored : loop_constraints_) {
    if (stored.map_id == current.map_id &&
        stored.query_id == current.query_id) {
      continue;
    }
    if (variable_set.count(stored.query_id) == 0 ||
        stored.map_id == stored.query_id ||
        poses_old->count(stored.query_id) == 0) {
      continue;
    }
    if (!reconstruction.ExistsImage(stored.map_id) ||
        !reconstruction.ExistsImage(stored.query_id) ||
        !reconstruction.Image(stored.map_id).HasPose() ||
        !reconstruction.Image(stored.query_id).HasPose()) {
      continue;
    }
    if (poses_old->count(stored.map_id) == 0) {
      (*poses_old)[stored.map_id] =
          reconstruction.Image(stored.map_id).CamFromWorld();
    }
    const Rigid3d current_rel = poses_old->at(stored.query_id) *
                                Inverse(poses_old->at(stored.map_id));
    if (!HistoricalResidualPlausible(stored.query_from_map, current_rel)) {
      LOG(INFO) << "OnlineLoopCloser: skip historical loop " << stored.query_id
                << "-" << stored.map_id << ", residual exceeds "
                << kHistoricalResidualFactor << "x correction gate";
      continue;
    }
    replay.push_back(stored);
  }
  std::sort(replay.begin(),
            replay.end(),
            [](const LoopConstraint& a, const LoopConstraint& b) {
              if (a.num_inliers != b.num_inliers) {
                return a.num_inliers > b.num_inliers;
              }
              return a.query_id < b.query_id;
            });

  // Current edge is kept even when an older edge has more inliers, so this
  // correction is not replaced by a previous measurement of the same place.
  std::vector<LoopConstraint> selected;
  selected.push_back(current);
  size_t num_hist = 0;
  for (const LoopConstraint& candidate : replay) {
    bool clustered = false;
    for (const LoopConstraint& kept : selected) {
      if (!SameLoopCluster(candidate.query_id,
                           candidate.map_id,
                           kept.query_id,
                           kept.map_id,
                           reconstruction)) {
        continue;
      }
      clustered = true;
      LOG(INFO) << "OnlineLoopCloser: drop clustered loop "
                << candidate.query_id << "-" << candidate.map_id
                << " inliers=" << candidate.num_inliers << ", keep "
                << kept.query_id << "-" << kept.map_id
                << " inliers=" << kept.num_inliers;
      break;
    }
    if (clustered) {
      continue;
    }
    selected.push_back(candidate);
    ++num_hist;
  }

  for (const LoopConstraint& edge : selected) {
    if (poses_old->count(edge.map_id) == 0 &&
        reconstruction.ExistsImage(edge.map_id) &&
        reconstruction.Image(edge.map_id).HasPose()) {
      (*poses_old)[edge.map_id] =
          reconstruction.Image(edge.map_id).CamFromWorld();
    }
    if (poses_old->count(edge.map_id) == 0 ||
        poses_old->count(edge.query_id) == 0) {
      continue;
    }
    if (variable_set.count(edge.map_id) == 0) {
      fixed_ids->insert(edge.map_id);
    }
    PoseGraphEdge loop_edge =
        MakeOdometryEdge(edge.map_id, edge.query_id, edge.query_from_map, 1.0);
    if (edge.rot_sigma > 0.0 && edge.trans_sigma > 0.0) {
      loop_edge.rot_sigma = edge.rot_sigma;
      loop_edge.trans_sigma = edge.trans_sigma;
    }
    loop_edge.robust = true;
    edges->push_back(loop_edge);
    edge_constraints->push_back(edge);
  }
  return num_hist;
}

}  // namespace colmap
