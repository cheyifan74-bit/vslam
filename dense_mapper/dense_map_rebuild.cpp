#include <algorithm>

#include "dense_mapper/dense_map.h"
#include "utils/print.h"

namespace vslam
{
namespace dense
{

bool DenseMap::rebuildPending() const
{
  if (rebuild_.active)
  {
    return true;
  }
  for (const auto &item : submaps_)
  {
    if (!item.second.replay_dirty.empty())
    {
      return true;
    }
  }
  return false;
}

/// Takes the dirty blocks of the newest submap that has any and prepares
/// their rebuild in a shadow volume. The live blocks stay published and
/// keep taking new keyframes until finishRebuild swaps the result in.
bool DenseMap::startRebuild()
{
  Submap *found = nullptr;
  for (auto it = submaps_.rbegin(); it != submaps_.rend(); ++it)
  {
    if (!it->second.replay_dirty.empty())
    {
      found = &it->second;
      break;
    }
  }
  if (found == nullptr)
  {
    return false;
  }
  Submap &submap = *found;
  thaw(submap);
  Rebuild rb;
  rb.active = true;
  rb.submap = submap.id;
  rb.started = Clock::now();
  rb.dirty.swap(submap.replay_dirty);
  rb.changed.swap(submap.replay_changed);
  std::vector<const DepthFrame *> members;
  for (const auto &item : frames_)
  {
    rb.start_id = std::max(rb.start_id, item.first);
    if (item.second.integrated && item.second.submap == submap.id)
    {
      members.push_back(&item.second);
    }
  }
  std::sort(members.begin(), members.end(),
            [](const DepthFrame *a, const DepthFrame *b) { return a->image_id < b->image_id; });
  for (const DepthFrame *frame : members)
  {
    if (rb.changed.count(frame->image_id) != 0)
    {
      rb.changed_max = std::max(rb.changed_max, frame->image_id);
      rb.any_changed = true;
    }
  }
  // Rebuilds dirty blocks from the frames whose pose changed, oldest first.
  // Voxels a newer keyframe already wrote or cleared are left in place, so
  // replaying an older depth cannot bring a replaced surface back.
  const auto touches_dirty = [&](const DepthFrame *frame) {
    if (rb.changed.count(frame->image_id) != 0)
    {
      return true;
    }
    for (const BlockKey &key : frame->blocks)
    {
      if (rb.dirty.count(key) != 0)
      {
        return true;
      }
    }
    return false;
  };
  for (const DepthFrame *frame : members)
  {
    if ((rb.any_changed && frame->image_id > rb.changed_max) || !touches_dirty(frame))
    {
      continue;
    }
    rb.todo.push_back(frame->image_id);
  }
  rb.num_members = members.size();
  rb.shadow.id = submap.id;
  rb.shadow.world_from_submap = submap.world_from_submap;
  // Without a moved frame (only removals) the blocks restart empty.
  if (rb.any_changed)
  {
    for (const BlockKey &key : rb.dirty)
    {
      const auto tomb = submap.tombstones.find(key);
      if (tomb != submap.tombstones.end() && tomb->second > rb.changed_max)
      {
        rb.shadow.tombstones.emplace(key, tomb->second);
      }
      const auto it = submap.blocks.find(key);
      if (it == submap.blocks.end())
      {
        continue;
      }
      std::vector<Voxel> &cells = rb.shadow.blocks.emplace(key, it->second).first->second;
      for (Voxel &voxel : cells)
      {
        if (voxel.last_id() <= rb.changed_max)
        {
          voxel = Voxel{};
        }
      }
    }
  }
  rebuild_ = std::move(rb);
  return true;
}

/// Fuses a member frame into the shadow, restricted to the dirty blocks.
/// Evicted pixels are read back for the call and released again.
void DenseMap::replayInto(DepthFrame *frame)
{
  const bool resident = frame->resident();
  if (!resident && !frame_store_.loadPixels(frame))
  {
    return;
  }
  integrator_.integrate(frame, rebuild_.shadow, &rebuild_.dirty, false);
  if (!resident)
  {
    frame->releasePixels();
  }
}

/// Replays queued frames into the shadow until the deadline, at least one
/// per call, and swaps the result in once all are done.
void DenseMap::stepRebuild(Clock::time_point deadline)
{
  Rebuild &rb = rebuild_;
  const auto start = Clock::now();
  auto live = submaps_.find(rb.submap);
  if (live == submaps_.end())
  {
    rebuild_ = Rebuild();
    return;
  }
  Submap &submap = live->second;
  if (!submap.replay_changed.empty())
  {
    // Another pose change in this submap: start over with all of it.
    PRINT_INFO("[DENSE]: rebuild submap %u restarted after a pose change (%zu / %zu frames)\n",
               rb.submap, rb.next, rb.todo.size());
    submap.replay_dirty.insert(rb.dirty.begin(), rb.dirty.end());
    submap.replay_changed.insert(rb.changed.begin(), rb.changed.end());
    rebuild_ = Rebuild();
    return;
  }
  ++rb.slices;
  rb.shadow.world_from_submap = submap.world_from_submap;
  bool first = true;
  while (rb.next < rb.todo.size() && (first || Clock::now() < deadline))
  {
    const uint32_t id = rb.todo[rb.next++];
    auto it = frames_.find(id);
    if (it == frames_.end() || !it->second.integrated || it->second.submap != rb.submap ||
        rb.fused_live.count(id) != 0)
    {
      continue;
    }
    replayInto(&it->second);
    ++rb.replayed;
    rb.moved += rb.changed.count(id) != 0 ? 1 : 0;
    first = false;
  }
  rb.work_ms += MsSince(start);
  if (rb.next >= rb.todo.size())
  {
    finishRebuild(submap);
  }
}

/// Replays the keyframes fused into the live submap meanwhile, then merges
/// the shadow into the live blocks voxel by voxel: whichever keyframe wrote
/// the voxel last wins, so live writes newer than the rebuild survive.
void DenseMap::finishRebuild(Submap &submap)
{
  Rebuild &rb = rebuild_;
  const auto start = Clock::now();
  for (const uint32_t id : rb.fused_live)
  {
    auto it = frames_.find(id);
    if (it != frames_.end() && it->second.integrated && it->second.submap == rb.submap)
    {
      replayInto(&it->second);
      ++rb.replayed;
    }
  }
  thaw(submap);
  const std::vector<Voxel> empty(kBlockVoxels);
  size_t freed = 0;
  for (const BlockKey &key : rb.dirty)
  {
    const auto shadow_it = rb.shadow.blocks.find(key);
    const auto live_it = submap.blocks.find(key);
    uint32_t tomb = 0;
    const auto shadow_tomb = rb.shadow.tombstones.find(key);
    if (shadow_tomb != rb.shadow.tombstones.end())
    {
      tomb = shadow_tomb->second;
    }
    // A live block a newer keyframe freed meanwhile is only a tombstone.
    uint32_t live_cleared = 0;
    const auto live_tomb = submap.tombstones.find(key);
    if (live_tomb != submap.tombstones.end())
    {
      if (live_tomb->second > rb.start_id)
      {
        tomb = std::max(tomb, live_tomb->second);
        live_cleared = live_tomb->second;
      }
      submap.tombstones.erase(live_tomb);
    }
    if (tomb != 0)
    {
      submap.tombstones[key] = tomb;
    }
    submap.surface_dirty.insert(key);
    if (shadow_it == rb.shadow.blocks.end() && live_it == submap.blocks.end())
    {
      continue;
    }
    std::vector<Voxel> merged =
        shadow_it != rb.shadow.blocks.end() ? std::move(shadow_it->second) : empty;
    for (size_t i = 0; i < merged.size(); ++i)
    {
      const uint32_t last =
          live_it != submap.blocks.end() ? live_it->second[i].last_id() : live_cleared;
      if (last > rb.start_id && last > merged[i].last_id())
      {
        if (live_it != submap.blocks.end())
        {
          merged[i] = live_it->second[i];
        }
        else
        {
          merged[i] = Voxel{};
          merged[i].set_last_id(last);
        }
      }
    }
    // Blocks nobody wrote again (e.g. only a removed frame saw them) are freed.
    if (std::all_of(merged.begin(), merged.end(),
                    [](const Voxel &v) { return v.weight_q == 0 && v.last_id() == 0; }))
    {
      submap.blocks.erase(key);
      ++freed;
      continue;
    }
    submap.blocks[key] = std::move(merged);
    submap.growBounds(key);
    if (config_.free_empty_blocks && AllEmpty(submap.blocks[key]))
    {
      submap.freeBlock(key);
      ++freed;
    }
  }
  submap.snapshot_dirty = true;
  rb.work_ms += MsSince(start);
  PRINT_INFO("[DENSE]: rebuild submap %u blocks=%zu frames=%zu / %zu (moved=%zu live=%zu) "
             "freed=%zu slices=%d work=%.0f ms wall=%.0f ms\n",
             rb.submap, rb.dirty.size(), rb.replayed, rb.num_members, rb.moved,
             rb.fused_live.size(), freed, rb.slices, rb.work_ms, MsSince(rb.started));
  rebuild_ = Rebuild();
}

/// Advances the submap rebuilds, newest submap first, for replay_budget_ms;
/// the rest continues on later calls so new keyframes are fused in between.
/// Moved submaps are already published at their new pose in the meantime.
/// Without a budget, one whole submap per call.
void DenseMap::flushReplay()
{
  const bool unlimited = config_.replay_budget_ms <= 0.0;
  const auto deadline =
      unlimited ? Clock::time_point::max()
                : Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                     std::chrono::duration<double, std::milli>(
                                         config_.replay_budget_ms));
  while (rebuild_.active || startRebuild())
  {
    stepRebuild(deadline);
    if (unlimited ? !rebuild_.active : Clock::now() >= deadline)
    {
      break;
    }
  }
  if (!rebuildPending())
  {
    jobs_since_dirty_ = 0;
  }
  std::vector<uint32_t> emptied;
  for (auto &item : submaps_)
  {
    Submap &submap = item.second;
    if (isRebuilding(item.first))
    {
      continue;
    }
    if (submap.replay_dirty.empty())
    {
      submap.replay_changed.clear();
    }
    if (submap.numBlocks() == 0 && submap.num_members <= 0)
    {
      emptied.push_back(item.first);
    }
  }
  for (const uint32_t id : emptied)
  {
    submaps_.erase(id);
    removed_submaps_.push_back(id);
    if (active_submap_ == id)
    {
      active_submap_ = 0;
    }
  }
}

} // namespace dense
} // namespace vslam
