#include "dense_mapper/surface_extractor.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace vslam
{
namespace dense
{

/// Surface points at the sdf zero crossings towards the +x/+y/+z neighbours,
/// averaged to one point per voxel. Submap frame.
void SurfaceExtractor::extractBlock(const Submap &submap, const BlockKey &key,
                                    const std::vector<Voxel> &voxels, SurfacePiece *piece) const
{
  const double voxel = grid_.voxel;
  const Eigen::Vector3d origin = grid_.blockOrigin(key);
  const float min_weight = static_cast<float>(min_weight_);
  const auto usable = [min_weight](const Voxel *cell) {
    return cell != nullptr && cell->weight() >= min_weight && std::abs(cell->sdf()) < 1.f;
  };
  for (int iz = 0; iz < kBlockN; ++iz)
  {
    for (int iy = 0; iy < kBlockN; ++iy)
    {
      for (int ix = 0; ix < kBlockN; ++ix)
      {
        const Voxel &cell = voxels[static_cast<size_t>(VoxelIndex(ix, iy, iz))];
        if (!usable(&cell))
        {
          continue;
        }
        const Eigen::Vector3d p =
            origin + Eigen::Vector3d((ix + 0.5) * voxel, (iy + 0.5) * voxel, (iz + 0.5) * voxel);
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        double gray_sum = 0.0;
        int count = 0;
        for (int axis = 0; axis < 3; ++axis)
        {
          const int nx = ix + (axis == 0 ? 1 : 0);
          const int ny = iy + (axis == 1 ? 1 : 0);
          const int nz = iz + (axis == 2 ? 1 : 0);
          const Voxel *next = (nx < kBlockN && ny < kBlockN && nz < kBlockN)
                                  ? &voxels[static_cast<size_t>(VoxelIndex(nx, ny, nz))]
                                  : submap.voxelAt(key, nx, ny, nz);
          if (!usable(next))
          {
            continue;
          }
          const float s0 = cell.sdf();
          const float s1 = next->sdf();
          // A jump this large between neighbours is a truncation edge, not a surface.
          if ((s0 >= 0.f) == (s1 >= 0.f) || std::abs(s0 - s1) > 0.5f)
          {
            continue;
          }
          const double a = s0 / (s0 - s1);
          Eigen::Vector3d crossing = p;
          crossing[axis] += a * voxel;
          sum += crossing;
          gray_sum += cell.gray() + a * (static_cast<double>(next->gray()) - cell.gray());
          ++count;
        }
        if (count == 0)
        {
          continue;
        }
        piece->xyz.push_back((sum / static_cast<double>(count)).cast<float>());
        const uint8_t gray =
            static_cast<uint8_t>(std::clamp(std::lround(gray_sum / count), 0L, 255L));
        piece->rgb.push_back(gray);
        piece->rgb.push_back(gray);
        piece->rgb.push_back(gray);
      }
    }
  }
}

void SurfaceExtractor::update(Submap &submap) const
{
  // Crossings towards +x/+y/+z read the next block, so the previous block changes too.
  BlockSet todo = submap.surface_dirty;
  for (const BlockKey &key : submap.surface_dirty)
  {
    const BlockKey lower[3] = {
        {key.x - 1, key.y, key.z}, {key.x, key.y - 1, key.z}, {key.x, key.y, key.z - 1}};
    for (const BlockKey &other : lower)
    {
      if (submap.blocks.count(other) != 0)
      {
        todo.insert(other);
      }
    }
  }
  submap.surface_dirty.clear();
  const std::vector<BlockKey> keys(todo.begin(), todo.end());
  std::vector<SurfacePiece> pieces(keys.size());
#pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < static_cast<int>(keys.size()); ++i)
  {
    const auto it = submap.blocks.find(keys[i]);
    if (it != submap.blocks.end())
    {
      extractBlock(submap, keys[i], it->second, &pieces[i]);
    }
  }
  for (size_t i = 0; i < keys.size(); ++i)
  {
    if (pieces[i].xyz.empty())
    {
      submap.surface.erase(keys[i]);
    }
    else
    {
      submap.surface[keys[i]] = std::move(pieces[i]);
    }
  }
  auto cloud = std::make_shared<DenseCloud>();
  size_t count = 0;
  for (const auto &item : submap.surface)
  {
    count += item.second.xyz.size();
  }
  cloud->xyz.reserve(count);
  cloud->rgb.reserve(count * 3);
  for (const auto &item : submap.surface)
  {
    cloud->xyz.insert(cloud->xyz.end(), item.second.xyz.begin(), item.second.xyz.end());
    cloud->rgb.insert(cloud->rgb.end(), item.second.rgb.begin(), item.second.rgb.end());
  }
  submap.points = std::move(cloud);
}

} // namespace dense
} // namespace vslam
