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

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace colmap {

bool OnlineLoopCloser::Relocalize(const image_t image_id,
                                  const DatabaseCache& cache,
                                  const Reconstruction& reconstruction,
                                  OnlineFeatureMatcher* matcher,
                                  const std::string& image_abs_path,
                                  const Bitmap* bitmap,
                                  OnlineLoopPair* loop_pair) {
  if (loop_pair == nullptr || matcher == nullptr) {
    return false;
  }
  if (!IsLoopClosureActive()) {
    LOG(WARNING) << "OnlineLoopCloser: relocalization needs an active loop "
                    "closer (MixVPR)";
    return false;
  }
  if (!reconstruction.ExistsImage(image_id) ||
      !reconstruction.Image(image_id).HasCameraPtr()) {
    LOG(WARNING) << "OnlineLoopCloser: relocalize image " << image_id
                 << " is not in the reconstruction";
    return false;
  }
  if (reconstruction.Image(image_id).HasPose()) {
    LOG(INFO) << "OnlineLoopCloser: image " << image_id
              << " already has a pose, skip relocalization";
    return false;
  }

  Eigen::VectorXf query;
  if (!EncodeQuery(image_id, image_abs_path, bitmap, &query)) {
    return false;
  }
  if (!StoreMixVprDescriptor(image_id, query)) {
    return false;
  }

  std::vector<std::pair<image_t, Eigen::VectorXf>> entries;
  try {
    DatabaseSession session(database_path_);
    entries = session->ReadAllMixVprDescriptors();
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: relocalize failed to read MixVPR "
                    "descriptors: "
                 << e.what();
    return false;
  }
  if (entries.empty()) {
    LOG(INFO) << "OnlineLoopCloser: relocalize image " << image_id
              << " has no stored map descriptors";
    return false;
  }

  // No pose yet, so retrieval cannot be gated by distance / view angle. Only
  // frames that carry a pose are valid PnP targets.
  const auto usable = [&](const image_t id) {
    return id != image_id && reconstruction.ExistsImage(id) &&
           reconstruction.Image(id).HasPose();
  };
  std::vector<std::pair<image_t, float>> ranked =
      RankMixVprMatches(query, entries, usable, options_.reloc_topk);
  if (options_.reloc_min_mixvpr_score > 0.0) {
    const float min_score =
        static_cast<float>(options_.reloc_min_mixvpr_score);
    ranked.erase(std::remove_if(ranked.begin(),
                                ranked.end(),
                                [min_score](const auto& item) {
                                  return item.second < min_score;
                                }),
                 ranked.end());
  }
  if (ranked.empty()) {
    LOG(INFO) << "OnlineLoopCloser: relocalize image " << image_id
              << " descriptors=" << entries.size()
              << " no candidate above score " << options_.reloc_min_mixvpr_score;
    return false;
  }

  std::vector<image_t> candidate_ids;
  candidate_ids.reserve(ranked.size());
  for (const auto& [id, score] : ranked) {
    candidate_ids.push_back(id);
  }
  if (!matcher->MatchSpecificPairs(image_id, candidate_ids, cache)) {
    LOG(WARNING) << "OnlineLoopCloser: relocalize image " << image_id
                 << " matching against " << candidate_ids.size()
                 << " candidate(s) failed";
    return false;
  }

  const CandidateGate gate = RelocGate();
  OnlineLoopPair best;
  bool has_best = false;
  for (const auto& [candidate_id, score] : ranked) {
    OnlineLoopPair attempt;
    const bool ok = VerifyCandidate(
        image_id, candidate_id, score, reconstruction, gate, &attempt);
    LOG(INFO) << "OnlineLoopCloser: relocalize " << image_id << "-"
              << candidate_id << " score=" << score
              << " matches=" << attempt.num_matches
              << " 2D-3D=" << attempt.num_3d_correspondences
              << " inliers=" << attempt.num_3d_inliers
              << " accepted=" << static_cast<int>(ok);
    if (!ok) {
      continue;
    }
    if (!has_best || attempt.num_3d_inliers > best.num_3d_inliers ||
        (attempt.num_3d_inliers == best.num_3d_inliers &&
         attempt.num_inliers > best.num_inliers)) {
      best = attempt;
      has_best = true;
    }
  }
  if (!has_best) {
    LOG(INFO) << "OnlineLoopCloser: relocalize image " << image_id
              << " failed, candidates=" << candidate_ids.size();
    return false;
  }

  *loop_pair = best;
  // A successful relocalization is not a loop: keep it out of the loop list so
  // the cooldown and the confirmed-loop log stay meaningful.
  last_loop_image_id_ = image_id;
  LOG(INFO) << "OnlineLoopCloser: RELOCALIZED " << image_id << "-"
            << loop_pair->image_id2 << " inliers=" << loop_pair->num_3d_inliers
            << " matches=" << loop_pair->num_matches
            << " center="
            << loop_pair->loop_cam_from_world.TgtOriginInSrc().transpose();
  return true;
}

}  // namespace colmap
