#include <algorithm>
#include <string>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#include "dense_mapper/dense_map.h"
#include "utils/print.h"

namespace vslam
{
namespace dense
{

/// Writes the submap file if it is not current (at the generation the next
/// snapshot will use, so the on-disk index stays valid) and drops the voxels.
bool DenseMap::freeze(Submap &submap)
{
  if (submap.file_gen == 0 || submap.snapshot_dirty)
  {
    const uint64_t gen = snapshot_gen_ + 1;
    if (!snapshots_.writeSubmap(submap, gen))
    {
      PRINT_WARNING("[DENSE]: cannot write TSDF submap %u to %s, keeping it in memory\n",
                    submap.id, snapshots_.dir().c_str());
      return false;
    }
    submap.file_gen = gen;
    submap.snapshot_dirty = false;
    index_dirty_ = true;
  }
  submap.releaseVoxels();
  return true;
}

/// Marks the submap used and reads its voxels back if it is frozen. A lost
/// file is rebuilt from the member frames.
void DenseMap::thaw(Submap &submap)
{
  submap.last_used = Clock::now();
  if (!submap.frozen)
  {
    return;
  }
  const auto start = Clock::now();
  BlockSet keys;
  keys.swap(submap.frozen_keys);
  submap.frozen = false;
  if (snapshots_.readSubmap(&submap, submap.file_gen, false))
  {
    PRINT_INFO("[DENSE]: thawed submap %u blocks=%zu in %.0f ms\n", submap.id,
               submap.blocks.size(), MsSince(start));
    return;
  }
  PRINT_WARNING("[DENSE]: cannot read frozen TSDF submap %u, rebuilding it from its frames\n",
                submap.id);
  submap.blocks.clear();
  submap.tombstones.clear();
  submap.snapshot_dirty = true;
  submap.replay_dirty.insert(keys.begin(), keys.end());
  for (const auto &item : frames_)
  {
    if (item.second.integrated && item.second.submap == submap.id)
    {
      submap.replay_dirty.insert(item.second.blocks.begin(), item.second.blocks.end());
    }
  }
}

void DenseMap::maybeFreeze()
{
  if (config_.freeze_idle_s <= 0.0)
  {
    return;
  }
  const auto start = Clock::now();
  size_t count = 0;
  size_t blocks = 0;
  for (auto &item : submaps_)
  {
    Submap &submap = item.second;
    if (submap.frozen || item.first == active_submap_ || submap.blocks.empty() ||
        isRebuilding(item.first) || !submap.replay_dirty.empty() ||
        !submap.surface_dirty.empty() ||
        std::chrono::duration<double>(start - submap.last_used).count() < config_.freeze_idle_s)
    {
      continue;
    }
    const size_t num = submap.blocks.size();
    if (!freeze(submap))
    {
      break;
    }
    ++count;
    blocks += num;
  }
  if (count == 0)
  {
    return;
  }
#ifdef __GLIBC__
  malloc_trim(0);
#endif
  PRINT_INFO("[DENSE]: froze %zu submap(s) blocks=%zu (%.0f MB) in %.0f ms\n", count, blocks,
             static_cast<double>(blocks) * kBlockVoxels * sizeof(Voxel) / 1e6, MsSince(start));
}

/// Saves changed submaps and the index. Frames still pending are left out
/// and fused again on the next load from their depth files.
void DenseMap::saveSnapshot(const char *reason)
{
  if (submaps_.empty() && snapshot_gen_ == 0)
  {
    return;
  }
  last_snapshot_ = Clock::now();
  const auto start = Clock::now();
  const uint64_t gen = snapshot_gen_ + 1;
  size_t written = 0;
  size_t written_blocks = 0;
  for (auto &item : submaps_)
  {
    Submap &submap = item.second;
    // Frozen submaps were written when they froze.
    if (submap.frozen)
    {
      continue;
    }
    if (!submap.surface_dirty.empty())
    {
      submap.snapshot_dirty = true;
    }
    if (submap.file_gen != 0 && !submap.snapshot_dirty)
    {
      continue;
    }
    if (!snapshots_.writeSubmap(submap, gen))
    {
      PRINT_WARNING("[DENSE]: cannot write TSDF submap %u to %s\n", submap.id,
                    snapshots_.dir().c_str());
      return;
    }
    submap.file_gen = gen;
    submap.snapshot_dirty = false;
    ++written;
    written_blocks += submap.blocks.size();
  }
  std::vector<IndexSubmap> entries;
  entries.reserve(submaps_.size());
  for (const auto &item : submaps_)
  {
    const Submap &submap = item.second;
    IndexSubmap entry;
    entry.id = submap.id;
    entry.file_gen = submap.file_gen;
    entry.num_members = static_cast<int32_t>(submap.num_members);
    entry.world_from_submap = submap.world_from_submap;
    entry.replay_dirty = submap.replay_dirty;
    entry.replay_changed = submap.replay_changed;
    // A rebuild in progress is saved as still to do.
    if (isRebuilding(submap.id))
    {
      entry.replay_dirty.insert(rebuild_.dirty.begin(), rebuild_.dirty.end());
      entry.replay_changed.insert(rebuild_.changed.begin(), rebuild_.changed.end());
    }
    entries.push_back(std::move(entry));
  }
  std::vector<const DepthFrame *> fused;
  for (const auto &item : frames_)
  {
    if (item.second.integrated && submaps_.count(item.second.submap) != 0)
    {
      fused.push_back(&item.second);
    }
  }
  if (!snapshots_.writeIndex(gen, active_submap_, next_submap_, entries, fused))
  {
    PRINT_WARNING("[DENSE]: cannot write TSDF index to %s\n", snapshots_.dir().c_str());
    return;
  }
  snapshot_gen_ = gen;
  index_dirty_ = false;
  // Files the new index does not name are stale.
  std::vector<std::pair<uint32_t, uint64_t>> keep;
  keep.reserve(submaps_.size());
  for (const auto &item : submaps_)
  {
    keep.emplace_back(item.first, item.second.file_gen);
  }
  snapshots_.removeStaleSubmaps(keep);
  PRINT_INFO("[DENSE]: saved TSDF snapshot (%s) submaps=%zu written=%zu (%zu blocks) "
             "frames=%zu in %.0f ms\n",
             reason, submaps_.size(), written, written_blocks, fused.size(), MsSince(start));
}

void DenseMap::maybeSnapshot()
{
  if (config_.snapshot_period_s <= 0.0 || !index_dirty_)
  {
    return;
  }
  const double elapsed = std::chrono::duration<double>(Clock::now() - last_snapshot_).count();
  if (elapsed >= config_.snapshot_period_s)
  {
    saveSnapshot("periodic");
  }
}

/// Restores submaps and the fused state of the frames already read from
/// disk. Returns false (and leaves no submap behind) when the snapshot is
/// missing, from another configuration, or incomplete.
bool DenseMap::loadSnapshot(size_t *restored)
{
  IndexData index;
  std::string why;
  const bool read = snapshots_.readIndex(&index, &why);
  const auto fail = [&](const char *reason) {
    PRINT_WARNING("[DENSE]: TSDF snapshot in %s not used (%s), re-fusing stored frames\n",
                  snapshots_.dir().c_str(), reason);
    submaps_.clear();
    // New files (also from freezing) must not reuse a generation the old
    // index still names, in case we stop before the next index is written.
    snapshot_gen_ = index.gen;
    return false;
  };
  if (!read)
  {
    return why.empty() ? false : fail(why.c_str());
  }
  for (IndexSubmap &entry : index.submaps)
  {
    Submap submap;
    submap.id = entry.id;
    submap.file_gen = entry.file_gen;
    submap.num_members = entry.num_members;
    submap.world_from_submap = entry.world_from_submap;
    submap.replay_dirty = std::move(entry.replay_dirty);
    submap.replay_changed = std::move(entry.replay_changed);
    submaps_.emplace(submap.id, std::move(submap));
  }
  size_t present = 0;
  for (const IndexFrame &entry : index.frames)
  {
    present += frames_.count(entry.image_id);
  }
  // A cleared or replaced frame directory: the volume no longer matches it.
  if (present * 2 < index.frames.size())
  {
    return fail("most indexed depth frames are gone");
  }
  size_t frozen = 0;
  for (auto &item : submaps_)
  {
    Submap &submap = item.second;
    if (!snapshots_.readSubmap(&submap, submap.file_gen, true))
    {
      return fail("missing or damaged submap file");
    }
    submap.pose_changed = true;
    // Old submaps keep only their surface; one submap's voxels at a time
    // are in memory while loading.
    if (config_.freeze_idle_s > 0.0 && item.first != index.active && submap.replay_dirty.empty())
    {
      extractSurface(submap);
      submap.releaseVoxels();
      ++frozen;
    }
    submap.snapshot_dirty = false;
  }
#ifdef __GLIBC__
  if (frozen > 0)
  {
    malloc_trim(0);
  }
#endif
  size_t removed = 0;
  for (IndexFrame &entry : index.frames)
  {
    auto submap = submaps_.find(entry.submap);
    if (submap == submaps_.end())
    {
      continue;
    }
    auto it = frames_.find(entry.image_id);
    if (it == frames_.end())
    {
      // Removed while the map was offline: rebuild what it wrote.
      submap->second.replay_dirty.insert(entry.blocks.begin(), entry.blocks.end());
      --submap->second.num_members;
      ++removed;
      continue;
    }
    DepthFrame &frame = it->second;
    frame.submap = entry.submap;
    frame.fused = entry.fused;
    frame.blocks = std::move(entry.blocks);
    frame.integrated = true;
    ++*restored;
  }
  snapshot_gen_ = index.gen;
  active_submap_ = submaps_.count(index.active) != 0 ? index.active : 0;
  next_submap_ = std::max(index.next, submaps_.empty() ? 1u : submaps_.rbegin()->first + 1);
  // Poses refined while the map was offline move submaps or rebuild members.
  for (auto &item : submaps_)
  {
    refreshSubmap(item.second);
  }
  PRINT_INFO("[DENSE]: loaded TSDF snapshot: %zu submap(s) (%zu frozen), %zu frame(s), %zu "
             "removed since\n",
             submaps_.size(), frozen, *restored, removed);
  return true;
}

void DenseMap::load(const std::function<bool()> &stopping)
{
  const auto start = Clock::now();
  std::vector<uint32_t> ids;
  for (DepthFrame &frame : frame_store_.readHeaders())
  {
    ids.push_back(frame.image_id);
    frames_[frame.image_id] = std::move(frame);
  }
  if (ids.empty())
  {
    return;
  }
  std::sort(ids.begin(), ids.end());
  size_t restored = 0;
  const bool from_snapshot = loadSnapshot(&restored);
  if (!from_snapshot)
  {
    for (auto &item : frames_)
    {
      item.second.integrated = false;
    }
  }
  // Frames the snapshot does not hold are fused with the same assignment
  // rule, in id order. Without a snapshot that is every stored frame.
  size_t replayed = 0;
  bool interrupted = false;
  for (const uint32_t id : ids)
  {
    const auto found = frames_.find(id);
    if (found == frames_.end() || found->second.integrated)
    {
      continue;
    }
    if (stopping())
    {
      interrupted = true;
      break;
    }
    DepthFrame &frame = found->second;
    // The submap choice looks at the frame's depth, so it is read before fusing.
    const bool resident = frame.resident();
    if (!resident && !frame_store_.loadPixels(&frame))
    {
      frames_.erase(id);
      continue;
    }
    Submap &submap = chooseSubmap(frame);
    frame.submap = submap.id;
    fuseFrame(&frame, submap, false);
    frame.integrated = true;
    ++submap.num_members;
    ++replayed;
    if (!resident)
    {
      frame.releasePixels();
    }
    // A long replay shows the map as it grows instead of only at the end.
    maybePublish();
    maybeFreeze();
  }
  // Frames left unfused by a stop are not part of the volume yet.
  if (interrupted)
  {
    for (auto it = frames_.begin(); it != frames_.end();)
    {
      it = it->second.integrated ? std::next(it) : frames_.erase(it);
    }
  }
  // Keep the newest stored frames resident for the next session's fusion.
  const size_t keep =
      config_.resident_frames > 0 ? static_cast<size_t>(config_.resident_frames) : ids.size();
  for (size_t i = ids.size() > keep ? ids.size() - keep : 0; i < ids.size(); ++i)
  {
    const auto it = frames_.find(ids[i]);
    if (it != frames_.end() && !it->second.resident())
    {
      frame_store_.loadPixels(&it->second);
    }
  }
  evictOld();
  PRINT_INFO("[DENSE]: loaded %zu stored depth frame(s): %zu from TSDF snapshot, %zu re-fused%s "
             "into %zu submap(s) in %.0f ms\n",
             ids.size(), restored, replayed, interrupted ? ", interrupted by stop" : "",
             submaps_.size(), MsSince(start));
  publish();
  // A long replay is saved at once, so the next start does not repeat it.
  if (replayed > 0 || !from_snapshot)
  {
    saveSnapshot(from_snapshot ? "after load" : "after replay");
  }
}

} // namespace dense
} // namespace vslam
