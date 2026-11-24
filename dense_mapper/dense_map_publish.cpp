#include <algorithm>
#include <cmath>

#include "dense_mapper/dense_map.h"
#include "utils/print.h"

namespace vslam
{
namespace dense
{

void DenseMap::extractSurface(Submap &submap)
{
  thaw(submap);
  if (!submap.surface_dirty.empty())
  {
    submap.snapshot_dirty = true;
  }
  extractor_.update(submap);
}

bool DenseMap::hasUnpublished() const
{
  if (!removed_submaps_.empty())
  {
    return true;
  }
  for (const auto &item : submaps_)
  {
    if (!item.second.surface_dirty.empty() || item.second.pose_changed)
    {
      return true;
    }
  }
  return false;
}

/// Re-extracts only the submaps whose blocks changed; a submap that only
/// moved is re-sent with the same points.
void DenseMap::publish()
{
  if (!cloud_callback_ || !hasUnpublished())
  {
    return;
  }
  const auto start = Clock::now();
  auto update = std::make_shared<DenseUpdate>();
  update->point_size_m = static_cast<float>(grid_.voxel);
  update->removed.swap(removed_submaps_);
  size_t changed_points = 0;
  size_t total_points = 0;
  size_t num_blocks = 0;
  size_t num_frozen = 0;
  size_t frozen_blocks = 0;
  for (auto &item : submaps_)
  {
    Submap &submap = item.second;
    const bool surface_changed = !submap.surface_dirty.empty();
    if (surface_changed)
    {
      extractSurface(submap);
    }
    if (surface_changed || submap.pose_changed || !submap.published)
    {
      auto cloud = std::make_shared<DenseSubmapCloud>();
      cloud->id = submap.id;
      const Eigen::Quaterniond &q = submap.world_from_submap.q;
      const Eigen::Vector3d &t = submap.world_from_submap.t;
      cloud->q_world_submap[0] = static_cast<float>(q.x());
      cloud->q_world_submap[1] = static_cast<float>(q.y());
      cloud->q_world_submap[2] = static_cast<float>(q.z());
      cloud->q_world_submap[3] = static_cast<float>(q.w());
      cloud->t_world_submap[0] = static_cast<float>(t.x());
      cloud->t_world_submap[1] = static_cast<float>(t.y());
      cloud->t_world_submap[2] = static_cast<float>(t.z());
      cloud->points = submap.points ? submap.points : std::make_shared<DenseCloud>();
      submap.published = cloud;
      submap.pose_changed = false;
      update->changed.push_back(cloud);
      changed_points += cloud->points->xyz.size();
    }
    update->submaps.push_back(submap.published);
    total_points += submap.published->points->xyz.size();
    num_blocks += submap.blocks.size();
    if (submap.frozen)
    {
      ++num_frozen;
      frozen_blocks += submap.frozen_keys.size();
    }
  }
  last_publish_ = Clock::now();
  const size_t num_changed = update->changed.size();
  const size_t num_removed = update->removed.size();
  cloud_callback_(std::move(update));

  size_t resident = 0;
  size_t pixel_bytes = 0;
  for (const auto &item : frames_)
  {
    if (item.second.resident())
    {
      ++resident;
      pixel_bytes += item.second.depth_mm.size() * sizeof(uint16_t) + item.second.conf.size() +
                     item.second.gray.size();
    }
  }
  const double volume_mb = static_cast<double>(num_blocks) * kBlockVoxels * sizeof(Voxel) / 1e6;
  PRINT_INFO("[DENSE]: publish submaps changed=%zu removed=%zu (%zu pts) total submaps=%zu "
             "points=%zu blocks=%zu (%.0f MB) frozen=%zu (%zu blocks) frames=%zu resident=%zu "
             "(%.0f MB) pending=%zu (%.0f ms)\n",
             num_changed, num_removed, changed_points, submaps_.size(), total_points, num_blocks,
             volume_mb, num_frozen, frozen_blocks, frames_.size(), resident, pixel_bytes / 1e6,
             pending_.size(), MsSince(start));
  if (config_.stats_period_s > 0.0 &&
      std::chrono::duration<double>(last_publish_ - last_stats_).count() >= config_.stats_period_s)
  {
    last_stats_ = last_publish_;
    logStats(total_points);
  }
}

/// How much of the allocated volume is in use, to size block / voxel layouts.
void DenseMap::logStats(size_t total_points) const
{
  const auto start = Clock::now();
  // Only resident submaps: frozen ones have no voxels in memory.
  std::vector<const std::vector<Voxel> *> blocks;
  size_t tombstones = 0;
  size_t num_frozen = 0;
  size_t frozen_blocks = 0;
  for (const auto &item : submaps_)
  {
    if (item.second.frozen)
    {
      ++num_frozen;
      frozen_blocks += item.second.frozen_keys.size();
      if (item.second.points)
      {
        total_points -= std::min(total_points, item.second.points->xyz.size());
      }
      continue;
    }
    for (const auto &block : item.second.blocks)
    {
      blocks.push_back(&block.second);
    }
    tombstones += item.second.tombstones.size();
  }
  const uint16_t min_weight =
      static_cast<uint16_t>(std::lround(config_.surface_min_weight * kWeightScale));
  const float trunc = static_cast<float>(grid_.trunc);
  const float voxel = static_cast<float>(grid_.voxel);
  size_t empty = 0;
  size_t occupied = 0;
  size_t surface = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : empty, occupied, surface)
  for (int b = 0; b < static_cast<int>(blocks.size()); ++b)
  {
    size_t used = 0;
    size_t near = 0;
    for (const Voxel &cell : *blocks[b])
    {
      if (cell.weight_q != 0)
      {
        ++used;
        // Within one voxel of the zero crossing and above the surface weight.
        if (cell.weight_q >= min_weight && std::abs(cell.sdf()) * trunc <= voxel)
        {
          ++near;
        }
      }
    }
    empty += used == 0 ? 1 : 0;
    occupied += used;
    surface += near;
  }
  const double total = static_cast<double>(blocks.size()) * kBlockVoxels;
  PRINT_INFO("[DENSE]: stats blocks=%zu empty=%zu (%.1f%%) occupied_voxels=%.1f%% "
             "near_surface_voxels=%.2f%% points/block=%.1f tombstones=%zu frozen_submaps=%zu "
             "frozen_blocks=%zu (%.0f ms)\n",
             blocks.size(), empty, blocks.empty() ? 0.0 : 100.0 * empty / blocks.size(),
             total > 0 ? 100.0 * occupied / total : 0.0, total > 0 ? 100.0 * surface / total : 0.0,
             blocks.empty() ? 0.0 : static_cast<double>(total_points) / blocks.size(), tombstones,
             num_frozen, frozen_blocks, MsSince(start));
}

void DenseMap::maybePublish()
{
  const double elapsed = std::chrono::duration<double>(Clock::now() - last_publish_).count();
  if (elapsed >= config_.publish_period_s && hasUnpublished())
  {
    publish();
  }
}

} // namespace dense
} // namespace vslam
