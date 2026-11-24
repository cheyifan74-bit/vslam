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
#include "mixvpr/mixvpr_encoder.h"

#include "colmap/scene/database_session.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace colmap {
namespace {

// Camera +Z in world. COLMAP x_cam = R x_world + t, so +Z_world = R^T e_z.
Eigen::Vector3d OpticalAxisWorld(const Image& image) {
  return image.CamFromWorld().rotation().toRotationMatrix().row(2).transpose();
}

}  // namespace

bool OnlineLoopCloser::StoreMixVprDescriptor(const image_t image_id,
                                             const Eigen::VectorXf& desc) {
  if (mixvpr_ == nullptr) {
    return false;
  }
  if (desc.size() != kMixVprDescDim) {
    LOG(WARNING) << "OnlineLoopCloser: cannot store MixVPR descriptor for image "
                 << image_id << " desc_dim=" << desc.size() << " expected "
                 << kMixVprDescDim;
    return false;
  }
  try {
    DatabaseSession session(database_path_);
    if (session->ExistsMixVprDescriptor(image_id)) {
      return true;
    }
    session->WriteMixVprDescriptor(image_id, desc);
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to write MixVPR descriptor for "
                 << image_id << ": " << e.what();
    return false;
  }
  return true;
}

std::vector<std::pair<image_t, float>> OnlineLoopCloser::DetectLoop(
    const image_t image_id,
    const Eigen::VectorXf& query,
    const Reconstruction& reconstruction) {
  std::vector<std::pair<image_t, float>> result;
  if (query.size() != kMixVprDescDim) {
    return result;
  }
  if (last_loop_image_id_ != kInvalidImageId &&
      image_id < last_loop_image_id_ +
                     static_cast<image_t>(options_.cooldown_num_images)) {
    return result;
  }
  if (!reconstruction.ExistsImage(image_id) ||
      !reconstruction.Image(image_id).HasPose()) {
    return result;
  }

  const Eigen::Vector3d current_center =
      reconstruction.Image(image_id).ProjectionCenter();
  const Eigen::Vector3d current_view =
      OpticalAxisWorld(reconstruction.Image(image_id));
  const double max_distance = options_.max_distance;
  const double max_distance_sq =
      max_distance > 0.0 ? max_distance * max_distance : -1.0;
  const double max_view_angle_deg = options_.max_view_angle_deg;
  const double min_view_cos =
      max_view_angle_deg > 0.0 && max_view_angle_deg < 180.0
          ? std::cos(max_view_angle_deg * 3.14159265358979323846 / 180.0)
          : -2.0;
  const std::unordered_set<image_t> covis =
      CovisibilityGroup(image_id, reconstruction);

  const auto usable = [&](const image_t other_id) {
    if (other_id == image_id) {
      return false;
    }
    if (other_id + static_cast<image_t>(std::max(options_.min_frame_gap, 0)) >
        image_id) {
      return false;
    }
    if (covis.count(other_id) > 0) {
      return false;
    }
    if (!reconstruction.ExistsImage(other_id) ||
        !reconstruction.Image(other_id).HasPose()) {
      return false;
    }
    const Image& other = reconstruction.Image(other_id);
    if (options_.min_candidate_points > 0 &&
        other.NumPoints3D() <
            static_cast<size_t>(options_.min_candidate_points)) {
      return false;
    }
    if (max_distance_sq >= 0.0) {
      if ((other.ProjectionCenter() - current_center).squaredNorm() >
          max_distance_sq) {
        return false;
      }
    }
    if (min_view_cos > -1.5) {
      const Eigen::Vector3d other_view = OpticalAxisWorld(other);
      const double na = current_view.norm();
      const double nb = other_view.norm();
      if (na < 1e-12 || nb < 1e-12 ||
          current_view.dot(other_view) < min_view_cos * na * nb) {
        return false;
      }
    }
    return true;
  };

  std::vector<image_t> candidate_ids;
  candidate_ids.reserve(reconstruction.NumRegImages());
  for (const image_t other_id : reconstruction.RegImageIds()) {
    if (usable(other_id)) {
      candidate_ids.push_back(other_id);
    }
  }

  size_t num_mixvpr_images = 0;
  std::vector<std::pair<image_t, Eigen::VectorXf>> mixvpr_entries;
  try {
    DatabaseSession session(database_path_);
    num_mixvpr_images = session->NumMixVprDescriptors();
    if (num_mixvpr_images == 0 || candidate_ids.empty()) {
      if (candidate_ids.empty()) {
        LOG(INFO) << "OnlineLoopCloser: image " << image_id
                  << " mixvpr_images=" << num_mixvpr_images
                  << " no loop candidates (within " << max_distance
                  << "m, view<" << max_view_angle_deg << "deg, score>="
                  << options_.min_mixvpr_score
                  << ", excluding first-level covis)";
      }
      return result;
    }
    mixvpr_entries = session->ReadMixVprDescriptors(candidate_ids);
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to read MixVPR descriptors: "
                 << e.what();
    return result;
  }
  if (mixvpr_entries.empty()) {
    LOG(INFO) << "OnlineLoopCloser: image " << image_id
              << " mixvpr_images=" << num_mixvpr_images
              << " geo_candidates=" << candidate_ids.size()
              << " no MixVPR rows for candidates";
    return result;
  }

  // Candidates already geometry-filtered; rank all loaded descriptors.
  std::vector<std::pair<image_t, float>> ranked =
      RankMixVprMatches(query, mixvpr_entries, /*usable=*/{}, 0);
  if (options_.min_mixvpr_score > 0.0) {
    const float min_score = static_cast<float>(options_.min_mixvpr_score);
    ranked.erase(std::remove_if(ranked.begin(),
                                ranked.end(),
                                [min_score](const auto& item) {
                                  return item.second < min_score;
                                }),
                 ranked.end());
  }
  if (options_.topk > 0 &&
      static_cast<int>(ranked.size()) > options_.topk) {
    ranked.resize(static_cast<size_t>(options_.topk));
  }
  if (ranked.empty()) {
    LOG(INFO) << "OnlineLoopCloser: image " << image_id
              << " mixvpr_images=" << num_mixvpr_images
              << " geo_candidates=" << candidate_ids.size()
              << " loaded=" << mixvpr_entries.size()
              << " no loop candidates (score>=" << options_.min_mixvpr_score
              << ")";
    return result;
  }

  LOG(INFO) << "OnlineLoopCloser: image " << image_id
            << " mixvpr_images=" << num_mixvpr_images
            << " geo_candidates=" << candidate_ids.size()
            << " loaded=" << mixvpr_entries.size()
            << " topk=" << ranked.size() << " best=" << ranked.front().first
            << " score=" << ranked.front().second;
  return ranked;
}

}  // namespace colmap
