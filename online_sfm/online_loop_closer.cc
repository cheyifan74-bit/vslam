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

#include "colmap/math/math.h"
#include "online_sfm/online_loop_closer_internal.h"
#include "online_sfm/timing_stats.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database_session.h"
#include "colmap/scene/track.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Geometry>

namespace colmap {
namespace online_loop_internal {

bool GeometryUsable(const TwoViewGeometry& geometry, const int min_inliers) {
  return geometry.config != TwoViewGeometry::ConfigurationType::DEGENERATE &&
         geometry.inlier_matches.size() >= static_cast<size_t>(min_inliers);
}

std::optional<Eigen::Vector2d> ProjectCam(const Camera& camera,
                                          const Eigen::Vector3d& xyz_cam) {
  if (xyz_cam.z() <= std::numeric_limits<double>::epsilon()) {
    return std::nullopt;
  }
  return camera.ImgFromCam(xyz_cam);
}

bool InsideImage(const Camera& camera, const Eigen::Vector2d& xy) {
  return xy.x() >= 0.0 && xy.y() >= 0.0 &&
         xy.x() < static_cast<double>(camera.width) &&
         xy.y() < static_cast<double>(camera.height);
}

double RotationDiffDeg(const Eigen::Quaterniond& a,
                       const Eigen::Quaterniond& b) {
  return RadToDeg(Eigen::AngleAxisd(a * b.inverse()).angle());
}

float DescriptorL2Sq(const FeatureDescriptors& a,
                     const point2D_t ia,
                     const FeatureDescriptors& b,
                     const point2D_t ib) {
  if (static_cast<int>(ia) >= a.data.rows() ||
      static_cast<int>(ib) >= b.data.rows() || a.data.cols() != b.data.cols() ||
      a.data.cols() == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return static_cast<float>(
      (a.data.row(ia).cast<int>() - b.data.row(ib).cast<int>()).squaredNorm());
}

}  // namespace online_loop_internal

namespace {

// Candidate + covisibles used as the loop-side map for PnP and fusion.
constexpr size_t kMaxLoopMapImages = 10;

}  // namespace

using online_loop_internal::GeometryUsable;

OnlineLoopCloser::OnlineLoopCloser(std::string database_path,
                                   OnlineLoopCloserOptions options)
    : database_path_(std::move(database_path)), options_(std::move(options)) {
  if (!options_.enabled) {
    LOG(INFO) << "OnlineLoopCloser: disabled by options";
    return;
  }
  timing_ = std::make_unique<TimingStats>(
      TimingStats::FileBesideDatabase(database_path_, "loop_pairs.txt"),
      "loop",
      std::vector<std::string>{"candidate",
                               "score",
                               "matches",
                               "inliers",
                               "n3d",
                               "n3d_inliers",
                               "correction_m",
                               "pnp",
                               "status",
                               "candidate_ids"});
  timing_->AddMeta("mixvpr_engine_path", options_.mixvpr_engine_path);
  timing_->AddMeta("mixvpr_use_gpu", options_.mixvpr_use_gpu ? "1" : "0");
  timing_->AddMeta("cooldown_num_images",
                   std::to_string(options_.cooldown_num_images));
  timing_->AddMeta("min_frame_gap", std::to_string(options_.min_frame_gap));
  timing_->AddMeta("max_distance", std::to_string(options_.max_distance));
  timing_->AddMeta("max_view_angle_deg",
                   std::to_string(options_.max_view_angle_deg));
  timing_->AddMeta("min_mixvpr_score",
                   std::to_string(options_.min_mixvpr_score));
  timing_->AddMeta("topk", std::to_string(options_.topk));
  timing_->AddMeta("min_num_matches",
                   std::to_string(options_.min_num_matches));
  timing_->AddMeta("min_num_verified_matches",
                   std::to_string(options_.min_num_verified_matches));
  timing_->AddMeta("reloc_topk", std::to_string(options_.reloc_topk));
  timing_->AddMeta("reloc_min_mixvpr_score",
                   std::to_string(options_.reloc_min_mixvpr_score));
  timing_->AddMeta("reloc_min_num_matches",
                   std::to_string(options_.reloc_min_num_matches));
  timing_->AddMeta("reloc_min_num_3d_correspondences",
                   std::to_string(options_.reloc_min_num_3d_correspondences));
  timing_->AddMeta("reloc_min_inliers",
                   std::to_string(options_.reloc_min_inliers));
  timing_->AddMeta("reloc_max_rot_vs_essential_deg",
                   std::to_string(options_.reloc_max_rot_vs_essential_deg));
}

OnlineLoopCloser::~OnlineLoopCloser() = default;

bool OnlineLoopCloser::EncodeQuery(const image_t image_id,
                                   const std::string& image_abs_path,
                                   const Bitmap* bitmap,
                                   Eigen::VectorXf* query) {
  if (query == nullptr) {
    return false;
  }
  if (!InitMixVprModel()) {
    return false;
  }
  const bool encoded = (bitmap != nullptr && !bitmap->IsEmpty())
                           ? mixvpr_->Encode(*bitmap, query)
                           : mixvpr_->Encode(image_abs_path, query);
  if (!encoded || query->size() != kMixVprDescDim) {
    LOG(WARNING) << "OnlineLoopCloser: MixVPR encode failed for image "
                 << image_id << " path=" << image_abs_path
                 << " desc_dim=" << query->size();
    return false;
  }
  return true;
}

OnlineLoopCloser::CandidateGate OnlineLoopCloser::LoopGate() const {
  CandidateGate gate;
  gate.min_matches = static_cast<size_t>(std::max(options_.min_num_matches, 0));
  gate.min_3d_corrs =
      static_cast<size_t>(std::max(options_.min_num_3d_correspondences, 0));
  gate.min_inliers =
      static_cast<size_t>(std::max(options_.min_num_verified_matches, 0));
  gate.max_rot_vs_essential_deg = options_.max_rot_vs_essential_deg;
  gate.require_current_pose = true;
  gate.check_correction = true;
  return gate;
}

OnlineLoopCloser::CandidateGate OnlineLoopCloser::RelocGate() const {
  CandidateGate gate;
  gate.min_matches =
      static_cast<size_t>(std::max(options_.reloc_min_num_matches, 0));
  gate.min_3d_corrs =
      static_cast<size_t>(std::max(options_.reloc_min_num_3d_correspondences, 0));
  gate.min_inliers = static_cast<size_t>(std::max(options_.reloc_min_inliers, 0));
  gate.max_rot_vs_essential_deg = options_.reloc_max_rot_vs_essential_deg;
  gate.require_current_pose = false;
  gate.check_correction = false;
  return gate;
}

bool OnlineLoopCloser::Process(const image_t image_id,
                               const std::string& image_abs_path,
                               const DatabaseCache& cache,
                               Reconstruction& reconstruction,
                               ObservationManager* obs_manager,
                               OnlineFeatureMatcher* matcher,
                               const Bitmap* bitmap) {
  if (!IsLoopClosureActive()) {
    return false;
  }
  if (image_id == kInvalidImageId ||
      (image_abs_path.empty() && (bitmap == nullptr || bitmap->IsEmpty()))) {
    return false;
  }

  Timer timer;
  timer.Start();
  if (!InitMixVprModel()) {
    return false;
  }

  Eigen::VectorXf query;
  if (!EncodeQuery(image_id, image_abs_path, bitmap, &query)) {
    return false;
  }

  const std::vector<std::pair<image_t, float>> candidates =
      DetectLoop(image_id, query, reconstruction);
  if (!StoreMixVprDescriptor(image_id, query)) {
    return false;
  }

  if (candidates.empty()) {
    timing_->Record(std::to_string(image_id),
                    timer.ElapsedMicroSeconds() / 1000.0,
                    {"-",
                     "0",
                     "0",
                     "0",
                     "0",
                     "0",
                     "0",
                     "0",
                     "no_candidate",
                     "-"});
    return false;
  }

  LOG(INFO) << "OnlineLoopCloser: image " << image_id << " has "
            << candidates.size() << " loop candidate(s)";

  const auto RecordPair = [&](const OnlineLoopPair& pair, const char* status) {
    timing_->Record(
        std::to_string(image_id),
        timer.ElapsedMicroSeconds() / 1000.0,
        {pair.image_id2 == kInvalidImageId ? "-"
                                          : std::to_string(pair.image_id2),
         std::to_string(pair.retrieval_score),
         std::to_string(pair.num_matches),
         std::to_string(pair.num_inliers),
         std::to_string(pair.num_3d_correspondences),
         std::to_string(pair.num_3d_inliers),
         std::to_string(pair.correction_m),
         pair.has_loop_pose ? "1" : "0",
         status,
         "-"});
  };
  const auto BetterAttempt = [](const OnlineLoopPair& a,
                                const OnlineLoopPair& b) {
    if (a.num_3d_inliers != b.num_3d_inliers) {
      return a.num_3d_inliers > b.num_3d_inliers;
    }
    if (a.num_3d_correspondences != b.num_3d_correspondences) {
      return a.num_3d_correspondences > b.num_3d_correspondences;
    }
    if (a.num_inliers != b.num_inliers) {
      return a.num_inliers > b.num_inliers;
    }
    return a.num_matches > b.num_matches;
  };

  bool confirmed = false;
  OnlineLoopPair confirmed_pair;
  OnlineLoopPair best_attempt;
  bool has_attempt = false;
  std::vector<image_t> candidate_ids;
  candidate_ids.reserve(candidates.size());
  for (const auto& [candidate_id, score] : candidates) {
    candidate_ids.push_back(candidate_id);
  }
  // Drop weak stored pairs so the batch rematch uses the loop match threshold,
  // which is stricter than the matcher's own inlier gate.
  try {
    DatabaseSession session(database_path_);
    for (const image_t candidate_id : candidate_ids) {
      if (!session->ExistsTwoViewGeometry(candidate_id, image_id)) {
        continue;
      }
      const TwoViewGeometry existing =
          session->ReadTwoViewGeometry(candidate_id, image_id);
      if (GeometryUsable(existing, options_.min_num_matches)) {
        continue;
      }
      if (session->ExistsMatches(candidate_id, image_id)) {
        session->DeleteMatches(candidate_id, image_id);
      }
      session->DeleteTwoViewGeometry(candidate_id, image_id);
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to clear weak pairs for image "
                 << image_id << ": " << e.what();
  }
  if (matcher == nullptr ||
      !matcher->MatchSpecificPairs(image_id, candidate_ids, cache)) {
    LOG(WARNING) << "OnlineLoopCloser: batch matching image " << image_id
                 << " against " << candidate_ids.size() << " candidate(s) failed";
    return false;
  }
  for (const auto& [candidate_id, score] : candidates) {
    OnlineLoopPair loop_pair;
    const bool ok = VerifyCandidate(image_id,
                                    candidate_id,
                                    score,
                                    reconstruction,
                                    LoopGate(),
                                    &loop_pair);
    if (loop_pair.image_id2 != kInvalidImageId) {
      if (!has_attempt || BetterAttempt(loop_pair, best_attempt)) {
        best_attempt = loop_pair;
        has_attempt = true;
      }
    }
    if (!ok) {
      continue;
    }
    confirmed_loops_.push_back(loop_pair);
    last_loop_image_id_ = image_id;
    confirmed = true;
    confirmed_pair = loop_pair;
    LOG(INFO) << "OnlineLoopCloser: CONFIRMED loop " << image_id << "-"
              << candidate_id << " matches=" << loop_pair.num_matches
              << " inliers=" << loop_pair.num_inliers
              << " n3d=" << loop_pair.num_3d_correspondences
              << " n3d_inliers=" << loop_pair.num_3d_inliers
              << " correction_m=" << loop_pair.correction_m
              << " correction_deg=" << loop_pair.correction_deg;
    RecordPair(loop_pair, "confirmed");
    break;
  }

  if (!confirmed) {
    if (has_attempt) {
      RecordPair(best_attempt, "verify_failed");
    } else {
      timing_->Record(std::to_string(image_id),
                      timer.ElapsedMicroSeconds() / 1000.0,
                      {"-",
                       "0",
                       "0",
                       "0",
                       "0",
                       "0",
                       "0",
                       "0",
                       "no_candidate",
                       "-"});
    }
    return false;
  }

  if (!options_.correct_local || obs_manager == nullptr ||
      !confirmed_pair.has_loop_pose) {
    return false;
  }

  const bool ok = CorrectLoop(reconstruction, obs_manager, confirmed_pair);
  if (ok) {
    confirmed_loops_.back() = confirmed_pair;
  }
  return ok;
}

bool OnlineLoopCloser::InitMixVprModel() {
  if (disabled_) {
    return false;
  }
  if (mixvpr_ready_) {
    return mixvpr_ != nullptr;
  }
  mixvpr_ready_ = true;

#ifndef VSLAM_MIXVPR_TENSORRT_ENABLED
  LOG(ERROR) << "OnlineLoopCloser: MixVPR requires TensorRT. Disabling "
                "loop detection";
  disabled_ = true;
  return false;
#else
  const std::filesystem::path model_path = options_.mixvpr_engine_path;
  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    LOG(ERROR) << "OnlineLoopCloser: MixVPR TensorRT engine not found at "
               << (model_path.empty() ? std::string("<empty>")
                                      : model_path.string())
               << ". Set loop_mixvpr_engine_path to a valid .engine file";
    disabled_ = true;
    return false;
  }

  try {
    LOG(INFO) << "OnlineLoopCloser: loading MixVPR from " << model_path
              << " gpu=" << static_cast<int>(options_.mixvpr_use_gpu)
              << " gpu_index=" << options_.mixvpr_gpu_index;
    mixvpr_ = std::make_unique<MixVprEncoder>(
        model_path.string(),
        options_.mixvpr_use_gpu,
        options_.mixvpr_gpu_index);
    LOG(INFO) << "OnlineLoopCloser: MixVPR ready desc_dim=" << mixvpr_->DescDim()
              << " input=" << kMixVprInputSize << "x" << kMixVprInputSize
              << " used_gpu=" << static_cast<int>(mixvpr_->UsedGpu())
              << " backend=" << mixvpr_->Backend();
    timing_->AddMeta("mixvpr_desc_dim", std::to_string(mixvpr_->DescDim()));
    timing_->AddMeta("mixvpr_backend", mixvpr_->Backend());
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineLoopCloser: failed to load MixVPR: " << e.what()
               << ", disabling loop detection";
    mixvpr_.reset();
    disabled_ = true;
    return false;
  }
  return true;
#endif
}

void OnlineLoopCloser::CollectProtectedImageIds(
    std::unordered_set<image_t>* ids) const {
  for (const LoopConstraint& constraint : loop_constraints_) {
    ids->insert(constraint.map_id);
    ids->insert(constraint.query_id);
  }
  for (const OnlineLoopPair& pair : confirmed_loops_) {
    ids->insert(pair.image_id1);
    ids->insert(pair.image_id2);
  }
  if (last_correction_.current_id != kInvalidImageId) {
    ids->insert(last_correction_.current_id);
    ids->insert(last_correction_.loop_id);
  }
}

void OnlineLoopCloser::ForgetLoopConstraint(const image_t map_id,
                                            const image_t query_id) {
  loop_constraints_.erase(
      std::remove_if(loop_constraints_.begin(),
                     loop_constraints_.end(),
                     [&](const LoopConstraint& other) {
                       return other.map_id == map_id &&
                              other.query_id == query_id;
                     }),
      loop_constraints_.end());
}

void OnlineLoopCloser::RememberLoopConstraint(const LoopConstraint& constraint) {
  loop_constraints_.erase(
      std::remove_if(loop_constraints_.begin(),
                     loop_constraints_.end(),
                     [&](const LoopConstraint& other) {
                       return other.map_id == constraint.map_id &&
                              other.query_id == constraint.query_id;
                     }),
      loop_constraints_.end());
  loop_constraints_.push_back(constraint);
}

std::vector<image_t> OnlineLoopCloser::LoopMapImages(
    const image_t candidate_id,
    const image_t current_id,
    const Reconstruction& reconstruction) const {
  std::vector<image_t> ids = {candidate_id};
  const std::unordered_set<image_t> group =
      CovisibilityGroup(candidate_id, reconstruction);
  const std::unordered_set<image_t> current_group =
      CovisibilityGroup(current_id, reconstruction);
  for (const image_t id : RankedCovisible(candidate_id, reconstruction)) {
    if (ids.size() >= kMaxLoopMapImages) {
      break;
    }
    if (id == current_id || group.count(id) == 0 ||
        current_group.count(id) > 0 || !reconstruction.ExistsImage(id) ||
        !reconstruction.Image(id).HasPose()) {
      continue;
    }
    ids.push_back(id);
  }
  return ids;
}

std::unordered_map<image_t, int> OnlineLoopCloser::CountCovisibilityPoints(
    const image_t image_id, const Reconstruction& reconstruction) const {
  std::unordered_map<image_t, int> shared;
  if (!reconstruction.ExistsImage(image_id)) {
    return shared;
  }
  const Image& image = reconstruction.Image(image_id);
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D() ||
        !reconstruction.ExistsPoint3D(point2D.point3D_id)) {
      continue;
    }
    for (const TrackElement& el :
         reconstruction.Point3D(point2D.point3D_id).track.Elements()) {
      if (el.image_id == image_id) {
        continue;
      }
      shared[el.image_id] += 1;
    }
  }
  return shared;
}

std::unordered_set<image_t> OnlineLoopCloser::CovisibilityGroup(
    const image_t image_id, const Reconstruction& reconstruction) const {
  std::unordered_set<image_t> group;
  group.insert(image_id);
  const auto shared = CountCovisibilityPoints(image_id, reconstruction);
  int best_count = 0;
  image_t best_id = kInvalidImageId;
  for (const auto& [other_id, count] : shared) {
    if (count > best_count) {
      best_count = count;
      best_id = other_id;
    }
    if (count >= options_.min_covisibility_points) {
      group.insert(other_id);
    }
  }
  if (group.size() == 1 && best_id != kInvalidImageId) {
    group.insert(best_id);
  }
  return group;
}

std::vector<image_t> OnlineLoopCloser::RankedCovisible(
    const image_t image_id, const Reconstruction& reconstruction) const {
  const auto shared = CountCovisibilityPoints(image_id, reconstruction);
  std::vector<std::pair<int, image_t>> ranked;
  ranked.reserve(shared.size());
  for (const auto& [other_id, count] : shared) {
    ranked.emplace_back(count, other_id);
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first > b.first;
    }
    return a.second < b.second;
  });
  std::vector<image_t> ids;
  ids.reserve(ranked.size());
  for (const auto& item : ranked) {
    ids.push_back(item.second);
  }
  return ids;
}

}  // namespace colmap
