#include "dense_mapper/tsdf_integrator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "utils/print.h"

namespace vslam
{
namespace dense
{
namespace
{

RangeTiles BuildRangeTiles(const std::vector<float> &range, int width, int height)
{
  RangeTiles tiles;
  tiles.cols = (width + RangeTiles::kTile - 1) / RangeTiles::kTile;
  tiles.rows = (height + RangeTiles::kTile - 1) / RangeTiles::kTile;
  tiles.max_range.assign(static_cast<size_t>(tiles.cols) * tiles.rows, 0.f);
  for (int y = 0; y < height; ++y)
  {
    float *row = &tiles.max_range[static_cast<size_t>(y / RangeTiles::kTile) * tiles.cols];
    const float *src = &range[static_cast<size_t>(y) * width];
    for (int x = 0; x < width; ++x)
    {
      float &cell = row[x / RangeTiles::kTile];
      cell = std::max(cell, src[x]);
    }
  }
  for (const float r : tiles.max_range)
  {
    tiles.global_max = std::max(tiles.global_max, r);
  }
  return tiles;
}

/// Conservative test whether any point within radius of pc (camera frame)
/// can lie more than `ahead` in front of the frame's depth, i.e. be carved.
bool MayCarve(const DepthFrame &frame, const RangeTiles &tiles, const Eigen::Vector3d &pc,
              double radius, double min_z, double ahead)
{
  if (pc.z() + radius < min_z)
  {
    return false;
  }
  const double near = pc.norm() - radius;
  if (near > static_cast<double>(tiles.global_max) - ahead)
  {
    return false;
  }
  // Bounds of x/z and y/z over the box around the sphere.
  const double z_lo = std::max(pc.z() - radius, min_z);
  const double z_hi = pc.z() + radius;
  const auto ratio_bounds = [&](double c, double *lo, double *hi) {
    const double a = c - radius;
    const double b = c + radius;
    *lo = a >= 0.0 ? a / z_hi : a / z_lo;
    *hi = b >= 0.0 ? b / z_lo : b / z_hi;
  };
  double x_lo, x_hi, y_lo, y_hi;
  ratio_bounds(pc.x(), &x_lo, &x_hi);
  ratio_bounds(pc.y(), &y_lo, &y_hi);
  const double u0 = std::max(frame.fx * x_lo + frame.cx, 0.0);
  const double u1 = std::min(frame.fx * x_hi + frame.cx, frame.width - 1.0);
  const double v0 = std::max(frame.fy * y_lo + frame.cy, 0.0);
  const double v1 = std::min(frame.fy * y_hi + frame.cy, frame.height - 1.0);
  if (u0 > u1 || v0 > v1)
  {
    return false;
  }
  const int c0 = static_cast<int>(std::floor(u0)) / RangeTiles::kTile;
  const int c1 = static_cast<int>(std::ceil(u1)) / RangeTiles::kTile;
  const int r0 = static_cast<int>(std::floor(v0)) / RangeTiles::kTile;
  const int r1 = static_cast<int>(std::ceil(v1)) / RangeTiles::kTile;
  float far = 0.f;
  for (int r = r0; r <= std::min(r1, tiles.rows - 1); ++r)
  {
    for (int c = c0; c <= std::min(c1, tiles.cols - 1); ++c)
    {
      far = std::max(far, tiles.max_range[static_cast<size_t>(r) * tiles.cols + c]);
    }
  }
  return far > 0.f && near <= static_cast<double>(far) - ahead;
}

/// A newer view puts the voxel on the other side of the surface, by a margin
/// beyond depth noise (sdf values in truncation units).
bool Conflicts(float old_sdf, float new_sdf, float min_diff)
{
  return (old_sdf >= 0.f) != (new_sdf >= 0.f) && std::abs(new_sdf - old_sdf) > min_diff;
}

} // namespace

double TsdfIntegrator::carveMinZ() const { return std::max(0.05, config_.min_range_m * 0.5); }

RangeTiles TsdfIntegrator::integrate(DepthFrame *frame, Submap &submap, const BlockSet *only,
                                     bool log) const
{
  if (frame == nullptr || !frame->resident())
  {
    return RangeTiles();
  }
  // A rebuild writes every frame, old and new, at its current pose, so the
  // newest observation wins.
  const Rigid cam_from_submap = frame->pose() * submap.world_from_submap;
  frame->fused = cam_from_submap;
  const double voxel = grid_.voxel;
  const double trunc = grid_.trunc;
  const Eigen::Matrix3d R = cam_from_submap.q.toRotationMatrix();
  const Eigen::Matrix3d Rt = R.transpose();
  const Eigen::Vector3d t = cam_from_submap.t;
  const Eigen::Vector3d center = -Rt * t;
  const int width = frame->width;
  const int height = frame->height;
  const double fx = frame->fx;
  const double fy = frame->fy;
  const double cx = frame->cx;
  const double cy = frame->cy;
  // Range along the pixel ray. Every pixel with a depth is fused; other
  // keyframes see a different view, so their depth is not a gate.
  std::vector<float> range(static_cast<size_t>(width) * static_cast<size_t>(height), 0.f);
  int kept = 0;
  float max_range = 0.f;
#pragma omp parallel for schedule(dynamic, 4) reduction(+ : kept) reduction(max : max_range)
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      const size_t idx = static_cast<size_t>(y) * width + x;
      if (frame->depth_mm[idx] == 0)
      {
        continue;
      }
      const double depth = static_cast<double>(frame->depth_mm[idx]) * 0.001;
      const Eigen::Vector3d p_cam((x - cx) * depth / fx, (y - cy) * depth / fy, depth);
      const double length = p_cam.norm();
      if (length < 1e-4)
      {
        continue;
      }
      range[idx] = static_cast<float>(length);
      max_range = std::max(max_range, static_cast<float>(length));
      ++kept;
    }
  }

  // Neighbouring seed rays stay within one voxel of each other at the farthest depth.
  int stride = std::max(1, config_.integrate_pixel_stride);
  if (max_range > 0.f)
  {
    stride = std::max(1, std::min(stride, static_cast<int>(voxel * fx / max_range)));
  }
  const double block_size = grid_.blockSize();
  // Samples along the band at most half a block apart cannot skip a block.
  const double step = std::min(trunc, block_size * 0.5);
  const bool carve = carves();
  RangeTiles tiles = carve ? BuildRangeTiles(range, width, height) : RangeTiles();
  BlockSet candidates;
#pragma omp parallel
  {
    BlockSet local;
    const auto march = [&](const Eigen::Vector3d &dir, double from, double to) {
      for (double s = from; s < to; s += step)
      {
        local.insert(grid_.keyOf(center + dir * s));
      }
      local.insert(grid_.keyOf(center + dir * to));
    };
    // One ray per stride x stride cell, through the cell centre, covering
    // the nearest and farthest depth in the cell so thin structures are kept.
#pragma omp for schedule(static) nowait
    for (int y0 = 0; y0 < height; y0 += stride)
    {
      for (int x0 = 0; x0 < width; x0 += stride)
      {
        float r_min = std::numeric_limits<float>::max();
        float r_max = 0.f;
        for (int y = y0; y < std::min(y0 + stride, height); ++y)
        {
          for (int x = x0; x < std::min(x0 + stride, width); ++x)
          {
            const float r = range[static_cast<size_t>(y) * width + x];
            if (r > 0.f)
            {
              r_min = std::min(r_min, r);
              r_max = std::max(r_max, r);
            }
          }
        }
        if (r_max <= 0.f)
        {
          continue;
        }
        const double u = std::min(x0 + 0.5 * (stride - 1), width - 1.0);
        const double v = std::min(y0 + 0.5 * (stride - 1), height - 1.0);
        const Eigen::Vector3d dir =
            (Rt * Eigen::Vector3d((u - cx) / fx, (v - cy) / fy, 1.0)).normalized();
        if (r_max - r_min <= 2.0 * trunc)
        {
          march(dir, r_min - trunc, r_max + trunc);
        }
        else
        {
          march(dir, r_min - trunc, r_min + trunc);
          march(dir, r_max - trunc, r_max + trunc);
        }
      }
    }
#pragma omp critical(dense_candidates)
    candidates.insert(local.begin(), local.end());
  }
  const double min_z = carveMinZ();
  // The truncation band only covers the new surface. Existing blocks that
  // can sit in front of it are carved too.
  if (carve)
  {
    const double half_diag = block_size * 0.8660254;
    const Eigen::Vector3d half = Eigen::Vector3d::Constant(block_size * 0.5);
    for (const auto &item : submap.blocks)
    {
      if (candidates.count(item.first) != 0)
      {
        continue;
      }
      const Eigen::Vector3d pc = R * (grid_.blockOrigin(item.first) + half) + t;
      if (MayCarve(*frame, tiles, pc, half_diag, min_z, trunc))
      {
        candidates.insert(item.first);
      }
    }
  }
  if (only != nullptr)
  {
    for (auto it = candidates.begin(); it != candidates.end();)
    {
      it = only->count(*it) != 0 ? std::next(it) : candidates.erase(it);
    }
  }

  std::vector<BlockKey> keys(candidates.begin(), candidates.end());
  std::vector<Voxel *> cells(keys.size(), nullptr);
  std::vector<uint8_t> fresh(keys.size(), 0);
  for (size_t i = 0; i < keys.size(); ++i)
  {
    fresh[i] = submap.blocks.count(keys[i]) == 0 ? 1 : 0;
    cells[i] = submap.block(keys[i]).data();
  }
  std::vector<uint8_t> hit(keys.size(), 0);
  const float trunc_f = static_cast<float>(trunc);
  const float carve_scale = carve ? static_cast<float>(config_.carve_weight) : 0.f;
  const Eigen::Vector3d step_cam = R.col(0) * voxel;
  const bool replace = config_.replace_sdf_diff > 0.0;
  const float replace_diff = static_cast<float>(config_.replace_sdf_diff);
  const float replace_conf = static_cast<float>(config_.replace_min_conf);
  const float replace_weight = static_cast<float>(config_.replace_weight);
#pragma omp parallel for schedule(dynamic, 8)
  for (int b = 0; b < static_cast<int>(keys.size()); ++b)
  {
    const Eigen::Vector3d origin = grid_.blockOrigin(keys[b]);
    Voxel *block = cells[b];
    bool touched = false;
    for (int iz = 0; iz < kBlockN; ++iz)
    {
      for (int iy = 0; iy < kBlockN; ++iy)
      {
        const Eigen::Vector3d row(origin.x() + 0.5 * voxel, origin.y() + (iy + 0.5) * voxel,
                                  origin.z() + (iz + 0.5) * voxel);
        const Eigen::Vector3d row_cam = R * row + t;
        for (int ix = 0; ix < kBlockN; ++ix)
        {
          const Eigen::Vector3d pc = row_cam + step_cam * static_cast<double>(ix);
          if (pc.z() < min_z)
          {
            continue;
          }
          const int u = static_cast<int>(std::lround(fx * pc.x() / pc.z() + cx));
          const int v = static_cast<int>(std::lround(fy * pc.y() / pc.z() + cy));
          if (u < 0 || v < 0 || u >= width || v >= height)
          {
            continue;
          }
          const size_t pidx = static_cast<size_t>(v) * width + u;
          const float r = range[pidx];
          if (r <= 0.f)
          {
            continue;
          }
          const float sdf = r - static_cast<float>(pc.norm());
          if (sdf < -trunc_f)
          {
            continue;
          }
          Voxel &cell = block[VoxelIndex(ix, iy, iz)];
          // A newer keyframe already updated this voxel, including clearing it.
          if (cell.last_id() > frame->image_id)
          {
            continue;
          }
          const float w = frame->conf.empty() ? 1.f : frame->conf[pidx] / 255.f;
          const float tsdf = sdf / trunc_f;
          const float old_weight = cell.weight();
          if (sdf > trunc_f)
          {
            // Farther than the truncation band: this voxel sits in front of the
            // measured surface. A weighted nudge cannot beat a saturated
            // surface, so drop a share of the maximum weight instead.
            if (carve_scale <= 0.f || old_weight <= 0.f)
            {
              continue;
            }
            float weight = old_weight - carve_scale * kMaxWeight;
            if (weight < static_cast<float>(config_.surface_min_weight))
            {
              weight = 0.f;
            }
            cell.set(1.f, weight);
            cell.set_last_id(frame->image_id);
            touched = true;
            continue;
          }
          if (w <= 0.f)
          {
            continue;
          }
          const float g = frame->gray.empty() ? 180.f : static_cast<float>(frame->gray[pidx]);
          // The surface moved by less than the band: averaging into a
          // saturated voxel would take dozens of views, so the newer view wins.
          if (replace && old_weight > replace_weight && w >= replace_conf &&
              cell.last_id() < frame->image_id && Conflicts(cell.sdf(), tsdf, replace_diff))
          {
            cell.set(tsdf, std::max(w, replace_weight));
            cell.set_gray(static_cast<uint8_t>(std::lround(g)));
            cell.set_last_id(frame->image_id);
            touched = true;
            continue;
          }
          const float sum = old_weight + w;
          cell.set_gray(static_cast<uint8_t>(
              std::lround((static_cast<float>(cell.gray()) * old_weight + g * w) / sum)));
          cell.set((cell.sdf() * old_weight + tsdf * w) / sum, std::min(sum, kMaxWeight));
          cell.set_last_id(frame->image_id);
          touched = true;
        }
      }
    }
    hit[b] = 0;
    if (touched)
    {
      hit[b] = config_.free_empty_blocks &&
                       std::all_of(block, block + kBlockVoxels,
                                   [](const Voxel &v) { return v.weight_q == 0; })
                   ? 2
                   : 1;
    }
  }

  BlockSet touched;
  size_t freed = 0;
  for (size_t i = 0; i < keys.size(); ++i)
  {
    if (hit[i] == 2)
    {
      // Still recorded for the frame: a rebuild of this frame revisits it.
      touched.insert(keys[i]);
      submap.freeBlock(keys[i]);
      ++freed;
    }
    else if (hit[i] != 0)
    {
      touched.insert(keys[i]);
      submap.surface_dirty.insert(keys[i]);
    }
    else if (fresh[i] != 0)
    {
      submap.blocks.erase(keys[i]);
    }
  }
  if (log)
  {
    PRINT_INFO("[DENSE]: fuse image %u submap %u keep=%d blocks=%zu/%zu freed=%zu stride=%d\n",
               frame->image_id, submap.id, kept, touched.size(), keys.size(), freed, stride);
  }
  if (only == nullptr)
  {
    frame->blocks = std::move(touched);
  }
  else
  {
    for (const BlockKey &key : *only)
    {
      frame->blocks.erase(key);
    }
    frame->blocks.insert(touched.begin(), touched.end());
  }
  return tiles;
}

bool TsdfIntegrator::mayCarveSubmap(const DepthFrame &frame, const RangeTiles &tiles,
                                    const Submap &other) const
{
  if (other.numBlocks() == 0 || !other.bounded)
  {
    return false;
  }
  const Rigid cam_from = frame.pose() * other.world_from_submap;
  const Eigen::Matrix3d R = cam_from.q.toRotationMatrix();
  const Eigen::Vector3d t = cam_from.t;
  const double min_z = carveMinZ();
  // With replacement, blocks in the band around the depth are visited too.
  const double ahead = config_.replace_sdf_diff > 0.0 ? -grid_.trunc : grid_.trunc;
  const double size = grid_.blockSize();
  const Eigen::Vector3d lo = grid_.blockOrigin(other.lo);
  const Eigen::Vector3d hi = grid_.blockOrigin(other.hi) + Eigen::Vector3d::Constant(size);
  if (!MayCarve(frame, tiles, R * (0.5 * (lo + hi)) + t, 0.5 * (hi - lo).norm(), min_z, ahead))
  {
    return false;
  }
  if (!other.frozen)
  {
    return true;
  }
  const double half_diag = size * 0.8660254;
  const Eigen::Vector3d half = Eigen::Vector3d::Constant(size * 0.5);
  return std::any_of(other.frozen_keys.begin(), other.frozen_keys.end(), [&](const BlockKey &key) {
    return MayCarve(frame, tiles, R * (grid_.blockOrigin(key) + half) + t, half_diag, min_z, ahead);
  });
}

bool TsdfIntegrator::carveInFront(const DepthFrame &frame, const RangeTiles &tiles,
                                  Submap &other) const
{
  const double voxel = grid_.voxel;
  const double trunc = grid_.trunc;
  const Rigid cam_from = frame.pose() * other.world_from_submap;
  const Eigen::Matrix3d R = cam_from.q.toRotationMatrix();
  const Eigen::Vector3d t = cam_from.t;
  const double min_z = carveMinZ();
  const bool replace = config_.replace_sdf_diff > 0.0;
  const double ahead = replace ? -trunc : trunc;
  const double size = grid_.blockSize();
  const double half_diag = size * 0.8660254;
  const Eigen::Vector3d half = Eigen::Vector3d::Constant(size * 0.5);
  std::vector<std::pair<BlockKey, std::vector<Voxel> *>> blocks;
  for (auto &item : other.blocks)
  {
    const Eigen::Vector3d pc = R * (grid_.blockOrigin(item.first) + half) + t;
    if (MayCarve(frame, tiles, pc, half_diag, min_z, ahead))
    {
      blocks.emplace_back(item.first, &item.second);
    }
  }
  if (blocks.empty())
  {
    return false;
  }
  const Eigen::Vector3d step_cam = R.col(0) * voxel;
  const float trunc_f = static_cast<float>(trunc);
  const float carve_scale = static_cast<float>(config_.carve_weight);
  const float replace_diff = static_cast<float>(config_.replace_sdf_diff);
  const float replace_conf = static_cast<float>(config_.replace_min_conf);
  const float replace_weight = static_cast<float>(config_.replace_weight);
  std::vector<uint8_t> hit(blocks.size(), 0);
#pragma omp parallel for schedule(dynamic, 8)
  for (int b = 0; b < static_cast<int>(blocks.size()); ++b)
  {
    const Eigen::Vector3d origin = grid_.blockOrigin(blocks[b].first);
    std::vector<Voxel> &cells = *blocks[b].second;
    bool touched = false;
    for (int iz = 0; iz < kBlockN; ++iz)
    {
      for (int iy = 0; iy < kBlockN; ++iy)
      {
        const Eigen::Vector3d row(origin.x() + 0.5 * voxel, origin.y() + (iy + 0.5) * voxel,
                                  origin.z() + (iz + 0.5) * voxel);
        const Eigen::Vector3d row_cam = R * row + t;
        for (int ix = 0; ix < kBlockN; ++ix)
        {
          const Eigen::Vector3d pc = row_cam + step_cam * static_cast<double>(ix);
          if (pc.z() < min_z)
          {
            continue;
          }
          const int u = static_cast<int>(std::lround(frame.fx * pc.x() / pc.z() + frame.cx));
          const int v = static_cast<int>(std::lround(frame.fy * pc.y() / pc.z() + frame.cy));
          if (u < 0 || v < 0 || u >= frame.width || v >= frame.height)
          {
            continue;
          }
          const size_t pidx = static_cast<size_t>(v) * frame.width + u;
          if (frame.depth_mm[pidx] == 0)
          {
            continue;
          }
          Voxel &cell = cells[static_cast<size_t>(VoxelIndex(ix, iy, iz))];
          if (cell.last_id() > frame.image_id || cell.weight() <= 0.f)
          {
            continue;
          }
          const double depth = frame.depth_mm[pidx] * 0.001;
          const double range = depth * std::sqrt(std::pow((u - frame.cx) / frame.fx, 2) +
                                                 std::pow((v - frame.cy) / frame.fy, 2) + 1.0);
          const float sdf = static_cast<float>(range) - static_cast<float>(pc.norm());
          if (sdf < -trunc_f)
          {
            continue;
          }
          if (sdf <= trunc_f)
          {
            // The newer view puts this surface elsewhere (within the band).
            // The new surface is written to the frame's own submap; here the
            // old one is only removed.
            const float conf = frame.conf.empty() ? 1.f : frame.conf[pidx] / 255.f;
            if (replace && cell.weight() > replace_weight && conf >= replace_conf &&
                cell.last_id() < frame.image_id &&
                Conflicts(cell.sdf(), sdf / trunc_f, replace_diff))
            {
              cell.set(1.f, 0.f);
              cell.set_last_id(frame.image_id);
              touched = true;
            }
            continue;
          }
          float weight = cell.weight() - carve_scale * kMaxWeight;
          if (weight < static_cast<float>(config_.surface_min_weight))
          {
            weight = 0.f;
          }
          cell.set(1.f, weight);
          cell.set_last_id(frame.image_id);
          touched = true;
        }
      }
    }
    hit[b] = !touched ? 0 : (config_.free_empty_blocks && AllEmpty(cells)) ? 2 : 1;
  }
  for (size_t b = 0; b < blocks.size(); ++b)
  {
    if (hit[b] == 2)
    {
      other.freeBlock(blocks[b].first);
    }
    else if (hit[b] != 0)
    {
      other.surface_dirty.insert(blocks[b].first);
    }
  }
  return true;
}

} // namespace dense
} // namespace vslam
