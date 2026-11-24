#include "dense_mapper/dense_map.h"

#include <algorithm>

#include "utils/print.h"

namespace vslam
{
namespace dense
{

DenseMap::DenseMap(const DenseConfig &config, FrameStore frame_store,
                   std::filesystem::path tsdf_dir)
    : config_(config), grid_(config.voxel_m, config.truncation_m),
      frame_store_(std::move(frame_store)), snapshots_(std::move(tsdf_dir), grid_),
      integrator_(config, grid_), extractor_(config.surface_min_weight, grid_)
{
}

bool DenseMap::isRebuilding(uint32_t submap_id) const
{
  return rebuild_.active && rebuild_.submap == submap_id;
}

/// Stores the new world pose and collects the submap of a fused frame.
void DenseMap::updatePose(uint32_t id, const Rigid &pose, std::unordered_set<uint32_t> *touched)
{
  auto it = frames_.find(id);
  if (it == frames_.end())
  {
    return;
  }
  DepthFrame &frame = it->second;
  if (!Moved(pose, frame.pose(), 1e-4, 1e-3))
  {
    return;
  }
  frame.q = pose.q;
  frame.t = pose.t;
  frame_store_.writePose(frame);
  if (frame.integrated)
  {
    touched->insert(frame.submap);
  }
}

/// Fuses a resident frame into its submap and carves the free space it sees
/// out of the other submaps.
void DenseMap::fuseFrame(DepthFrame *frame, Submap &submap, bool log)
{
  if (!frame->resident())
  {
    return;
  }
  thaw(submap);
  const RangeTiles tiles = integrator_.integrate(frame, submap, nullptr, log);
  if (!integrator_.carves())
  {
    return;
  }
  for (auto &item : submaps_)
  {
    Submap &other = item.second;
    if (item.first == submap.id || !integrator_.mayCarveSubmap(*frame, tiles, other))
    {
      continue;
    }
    if (other.frozen)
    {
      thaw(other);
    }
    if (integrator_.carveInFront(*frame, tiles, other))
    {
      other.last_used = Clock::now();
    }
  }
}

/// Fuses the oldest pending frames until at most keep remain.
int DenseMap::fusePending(size_t keep)
{
  int fused = 0;
  while (pending_.size() > keep)
  {
    const uint32_t id = pending_.front();
    pending_.pop_front();
    auto it = frames_.find(id);
    if (it == frames_.end() || it->second.integrated)
    {
      continue;
    }
    DepthFrame &frame = it->second;
    Submap &submap = chooseSubmap(frame);
    frame.submap = submap.id;
    fuseFrame(&frame, submap, true);
    frame.integrated = true;
    ++submap.num_members;
    if (isRebuilding(submap.id))
    {
      rebuild_.fused_live.insert(id);
    }
    frame_store_.write(frame);
    ++fused;
  }
  if (fused > 0)
  {
    evictOld();
  }
  return fused;
}

/// Keeps the pixels of the newest resident_frames fused frames and of every
/// pending frame; older frames keep only their pose and block list. Their
/// files are current, so nothing is written here.
void DenseMap::evictOld()
{
  if (config_.resident_frames <= 0)
  {
    return;
  }
  std::vector<uint32_t> resident;
  for (const auto &item : frames_)
  {
    if (item.second.integrated && item.second.resident())
    {
      resident.push_back(item.first);
    }
  }
  const size_t keep = static_cast<size_t>(config_.resident_frames);
  if (resident.size() <= keep)
  {
    return;
  }
  std::sort(resident.begin(), resident.end(), std::greater<uint32_t>());
  for (size_t i = keep; i < resident.size(); ++i)
  {
    frames_[resident[i]].releasePixels();
  }
}

Submap &DenseMap::createSubmap(const DepthFrame &first)
{
  Submap submap;
  submap.id = next_submap_++;
  submap.world_from_submap = first.pose().inverse();
  submap.pose_changed = true;
  const uint32_t id = submap.id;
  auto &stored = submaps_.emplace(id, std::move(submap)).first->second;
  PRINT_INFO("[DENSE]: new submap %u at image %u (%zu submaps)\n", id, first.image_id,
             submaps_.size());
  return stored;
}

/// Share of the frame's depth samples that land in allocated blocks of the submap.
double DenseMap::overlap(const DepthFrame &frame, const Submap &submap) const
{
  if (!frame.resident() || submap.numBlocks() == 0)
  {
    return 0.0;
  }
  const Rigid cam_from_submap = frame.pose() * submap.world_from_submap;
  const Eigen::Matrix3d Rt = cam_from_submap.q.conjugate().toRotationMatrix();
  const Eigen::Vector3d center = -(Rt * cam_from_submap.t);
  constexpr int kStride = 16;
  int samples = 0;
  int inside = 0;
  for (int y = kStride / 2; y < frame.height; y += kStride)
  {
    for (int x = kStride / 2; x < frame.width; x += kStride)
    {
      const uint16_t d = frame.depth_mm[static_cast<size_t>(y) * frame.width + x];
      if (d == 0)
      {
        continue;
      }
      const double z = d * 0.001;
      const Eigen::Vector3d p_cam((x - frame.cx) * z / frame.fx, (y - frame.cy) * z / frame.fy, z);
      ++samples;
      inside += submap.hasBlock(grid_.keyOf(Rt * p_cam + center)) ? 1 : 0;
    }
  }
  return samples > 0 ? static_cast<double>(inside) / samples : 0.0;
}

/// The submap that already holds most of what the keyframe sees (a revisit),
/// else the active submap while it has room and the keyframe stays within
/// submap_max_distance_m of its origin, else a new submap.
Submap &DenseMap::chooseSubmap(const DepthFrame &frame)
{
  if (config_.submap_max_frames <= 0)
  {
    return submaps_.empty() ? createSubmap(frame) : submaps_.begin()->second;
  }
  const Eigen::Vector3d center = frame.center();
  const double reach = config_.submap_max_distance_m + config_.max_range_m;
  const auto has_room = [&](const Submap &submap) {
    return submap.num_members < config_.submap_max_frames;
  };
  Submap *active = nullptr;
  double active_overlap = 0.0;
  Submap *best = nullptr;
  double best_overlap = 0.0;
  for (auto &item : submaps_)
  {
    Submap &submap = item.second;
    if (!has_room(submap) || (center - submap.world_from_submap.t).norm() > reach)
    {
      continue;
    }
    const double share = overlap(frame, submap);
    if (item.first == active_submap_)
    {
      active = &submap;
      active_overlap = share;
    }
    if (share > best_overlap)
    {
      best = &submap;
      best_overlap = share;
    }
  }
  const bool active_near =
      active != nullptr &&
      (center - active->world_from_submap.t).norm() <= config_.submap_max_distance_m;
  Submap *chosen = nullptr;
  if (best != nullptr && best != active && best_overlap >= 0.5 &&
      best_overlap > active_overlap + 0.3)
  {
    chosen = best;
  }
  else if (active_near)
  {
    chosen = active;
  }
  else if (best != nullptr && best_overlap >= 0.3)
  {
    chosen = best;
  }
  else
  {
    chosen = &createSubmap(frame);
  }
  active_submap_ = chosen->id;
  return *chosen;
}

/// After pose updates: every member implies a submap pose (its current world
/// pose combined with the pose it was fused with). The submap takes the mean
/// of the members that agree, so a loop correction moves it rigidly while a
/// few locally refined members stay outliers. Members that disagree with the
/// result beyond the thresholds are rebuilt.
void DenseMap::refreshSubmap(Submap &submap)
{
  std::vector<const DepthFrame *> members;
  for (const auto &item : frames_)
  {
    if (item.second.integrated && item.second.submap == submap.id)
    {
      members.push_back(&item.second);
    }
  }
  if (members.empty())
  {
    return;
  }
  const auto disagrees = [&](const DepthFrame &frame, const Rigid &world_from_submap) {
    return Moved(frame.pose() * world_from_submap, frame.fused, config_.pose_update_trans_m,
                 config_.pose_update_rot_deg);
  };
  const Eigen::Quaterniond reference = submap.world_from_submap.q;
  const auto mean = [&](const std::vector<const DepthFrame *> &set) {
    Eigen::Vector4d q_sum = Eigen::Vector4d::Zero();
    Eigen::Vector3d t_sum = Eigen::Vector3d::Zero();
    for (const DepthFrame *frame : set)
    {
      const Rigid implied = frame->pose().inverse() * frame->fused;
      Eigen::Vector4d q = implied.q.coeffs();
      if (q.dot(reference.coeffs()) < 0.0)
      {
        q = -q;
      }
      q_sum += q;
      t_sum += implied.t;
    }
    Rigid out;
    out.q = Eigen::Quaterniond(q_sum(3), q_sum(0), q_sum(1), q_sum(2)).normalized();
    out.t = t_sum / static_cast<double>(set.size());
    return out;
  };
  Rigid consensus = mean(members);
  for (int iteration = 0; iteration < 2; ++iteration)
  {
    std::vector<const DepthFrame *> inliers;
    for (const DepthFrame *frame : members)
    {
      if (!disagrees(*frame, consensus))
      {
        inliers.push_back(frame);
      }
    }
    if (inliers.empty() || inliers.size() == members.size())
    {
      break;
    }
    consensus = mean(inliers);
  }
  if (Moved(consensus.inverse(), submap.world_from_submap.inverse(), 1e-4, 1e-3))
  {
    submap.world_from_submap = consensus;
    submap.pose_changed = true;
  }
  // The replay only writes dirty blocks, so the blocks a frame lands in at its
  // new pose are marked too; otherwise a large correction drops the part of
  // its surface that moved out of its old blocks. Shifts within half the
  // truncation band only cut off the band's tail and are left out.
  const double half = 0.5 * grid_.blockSize();
  const auto add_moved_blocks = [&](const DepthFrame &frame) {
    const Rigid new_from_old =
        (frame.pose() * submap.world_from_submap).inverse() * frame.fused;
    const Eigen::Matrix3d R = new_from_old.q.toRotationMatrix();
    for (const BlockKey &key : frame.blocks)
    {
      const Eigen::Vector3d center = grid_.blockOrigin(key) + Eigen::Vector3d::Constant(half);
      const Eigen::Vector3d moved = R * center + new_from_old.t;
      if ((moved - center).norm() <= 0.5 * grid_.trunc)
      {
        continue;
      }
      for (int corner = 0; corner < 8; ++corner)
      {
        const Eigen::Vector3d offset((corner & 1) ? half : -half, (corner & 2) ? half : -half,
                                     (corner & 4) ? half : -half);
        submap.replay_dirty.insert(grid_.keyOf(moved + 0.999 * offset));
      }
    }
  };
  for (const DepthFrame *frame : members)
  {
    if (disagrees(*frame, submap.world_from_submap))
    {
      submap.replay_dirty.insert(frame->blocks.begin(), frame->blocks.end());
      add_moved_blocks(*frame);
      submap.replay_changed.insert(frame->image_id);
    }
  }
}

/// A submap whose keyframes were all retired has no member left to follow
/// later pose corrections, and newer passes of the same cells cover its area.
/// Kept, it would drift out of line after a loop and pile up on revisits.
void DenseMap::dropOrphans()
{
  std::unordered_set<uint32_t> referenced;
  for (const auto &item : frames_)
  {
    if (item.second.integrated)
    {
      referenced.insert(item.second.submap);
    }
  }
  std::vector<uint32_t> orphans;
  for (const auto &item : submaps_)
  {
    if (item.second.num_members <= 0 && referenced.count(item.first) == 0 &&
        item.first != active_submap_ && !isRebuilding(item.first))
    {
      orphans.push_back(item.first);
    }
  }
  for (const uint32_t id : orphans)
  {
    PRINT_INFO("[DENSE]: dropped submap %u: all its keyframes were replaced (%zu blocks)\n", id,
               submaps_.at(id).numBlocks());
    submaps_.erase(id);
    removed_submaps_.push_back(id);
  }
}

ApplyStats DenseMap::apply(MapJob job, bool idle, bool stopping)
{
  ApplyStats stats;
  last_job_ = Clock::now();
  index_dirty_ = true;
  // A redundant keyframe's depth is still a valid observation, so it stays in
  // the volume; it is only lost if one of its blocks is rebuilt later.
  const std::unordered_set<uint32_t> redundant(job.redundant_ids.begin(), job.redundant_ids.end());
  for (const uint32_t id : job.removed_ids)
  {
    auto it = frames_.find(id);
    if (it != frames_.end())
    {
      const DepthFrame &frame = it->second;
      const bool keep = !config_.rebuild_redundant_removals && redundant.count(id) != 0;
      auto submap = submaps_.find(frame.submap);
      if (frame.integrated && submap != submaps_.end())
      {
        if (!keep)
        {
          submap->second.replay_dirty.insert(frame.blocks.begin(), frame.blocks.end());
        }
        --submap->second.num_members;
      }
      stats.retired += keep ? 1 : 0;
      frames_.erase(it);
    }
    frame_store_.remove(id);
  }
  if (stats.retired > 0)
  {
    dropOrphans();
  }

  std::unordered_set<uint32_t> touched;
  for (const auto &item : job.pose_updates)
  {
    updatePose(item.first, item.second, &touched);
  }
  for (const uint32_t id : touched)
  {
    auto submap = submaps_.find(id);
    if (submap != submaps_.end())
    {
      refreshSubmap(submap->second);
    }
  }
  stats.moved_submaps = touched.size();

  if (job.depth)
  {
    const uint32_t id = job.depth->image_id;
    auto it = frames_.find(id);
    if (it != frames_.end() && it->second.integrated)
    {
      auto submap = submaps_.find(it->second.submap);
      if (submap != submaps_.end())
      {
        submap->second.replay_dirty.insert(it->second.blocks.begin(), it->second.blocks.end());
        --submap->second.num_members;
      }
    }
    frames_[id] = std::move(*job.depth);
    job.depth.reset();
    pending_.push_back(id);
  }

  const auto fuse_start = Clock::now();
  stats.fused = fusePending(static_cast<size_t>(std::max(0, config_.integrate_delay_frames)));
  stats.fuse_ms = MsSince(fuse_start);

  // While stopping, rebuilds are left to the next session (the snapshot keeps
  // the dirty blocks), so shutdown does not wait for them.
  if (!stopping && rebuildPending())
  {
    ++jobs_since_dirty_;
    if (idle || rebuild_.active || jobs_since_dirty_ >= std::max(1, config_.replay_max_defer_jobs))
    {
      const auto replay_start = Clock::now();
      flushReplay();
      stats.rebuild_ms = MsSince(replay_start);
    }
  }

  const auto publish_start = Clock::now();
  if (!stopping)
  {
    maybePublish();
    maybeFreeze();
    maybeSnapshot();
  }
  stats.publish_ms = MsSince(publish_start);
  stats.pending = pending_.size();
  return stats;
}

void DenseMap::onIdle()
{
  if (rebuildPending())
  {
    flushReplay();
  }
  if (!pending_.empty() && config_.idle_flush_s > 0.0 &&
      std::chrono::duration<double>(Clock::now() - last_job_).count() >= config_.idle_flush_s)
  {
    fusePending(0);
  }
  maybePublish();
  maybeFreeze();
  maybeSnapshot();
}

void DenseMap::shutdown()
{
  // Pending frames were stored with their first pose; keep the refined one.
  for (const uint32_t id : pending_)
  {
    const auto it = frames_.find(id);
    if (it != frames_.end())
    {
      frame_store_.write(it->second);
    }
  }
  saveSnapshot("stop");
}

} // namespace dense
} // namespace vslam
