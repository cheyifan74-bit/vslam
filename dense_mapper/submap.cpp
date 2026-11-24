#include "dense_mapper/submap.h"

#include <algorithm>

namespace vslam
{
namespace dense
{

std::vector<Voxel> &Submap::block(const BlockKey &key)
{
  auto it = blocks.find(key);
  if (it == blocks.end())
  {
    it = blocks.emplace(key, std::vector<Voxel>(kBlockVoxels)).first;
    growBounds(key);
    // A freed block comes back with its last writer, so older frames stay out.
    const auto tomb = tombstones.find(key);
    if (tomb != tombstones.end())
    {
      for (Voxel &cell : it->second)
      {
        cell.set_last_id(tomb->second);
      }
    }
  }
  return it->second;
}

size_t Submap::numBlocks() const { return frozen ? frozen_keys.size() : blocks.size(); }

bool Submap::hasBlock(const BlockKey &key) const
{
  return frozen ? frozen_keys.count(key) != 0 : blocks.count(key) != 0;
}

void Submap::freeBlock(const BlockKey &key)
{
  const auto it = blocks.find(key);
  if (it == blocks.end())
  {
    return;
  }
  uint32_t newest = 0;
  for (const Voxel &cell : it->second)
  {
    newest = std::max(newest, cell.last_id());
  }
  if (newest != 0)
  {
    uint32_t &tomb = tombstones[key];
    tomb = std::max(tomb, newest);
  }
  blocks.erase(it);
  surface_dirty.insert(key);
}

void Submap::growBounds(const BlockKey &key)
{
  if (!bounded)
  {
    lo = key;
    hi = key;
    bounded = true;
    return;
  }
  lo = {std::min(lo.x, key.x), std::min(lo.y, key.y), std::min(lo.z, key.z)};
  hi = {std::max(hi.x, key.x), std::max(hi.y, key.y), std::max(hi.z, key.z)};
}

const Voxel *Submap::voxelAt(BlockKey key, int ix, int iy, int iz) const
{
  const auto wrap = [](int *i, int *k) {
    if (*i < 0)
    {
      *i += kBlockN;
      --*k;
    }
    else if (*i >= kBlockN)
    {
      *i -= kBlockN;
      ++*k;
    }
  };
  wrap(&ix, &key.x);
  wrap(&iy, &key.y);
  wrap(&iz, &key.z);
  const auto it = blocks.find(key);
  return it == blocks.end() ? nullptr : &it->second[static_cast<size_t>(VoxelIndex(ix, iy, iz))];
}

void Submap::releaseVoxels()
{
  frozen_keys.clear();
  frozen_keys.reserve(blocks.size());
  for (const auto &item : blocks)
  {
    frozen_keys.insert(item.first);
  }
  decltype(blocks)().swap(blocks);
  decltype(tombstones)().swap(tombstones);
  frozen = true;
}

bool AllEmpty(const std::vector<Voxel> &cells)
{
  return std::all_of(cells.begin(), cells.end(), [](const Voxel &v) { return v.weight_q == 0; });
}

} // namespace dense
} // namespace vslam
