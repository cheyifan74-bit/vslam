#ifndef VSLAM_DENSE_MAPPER_DENSE_TYPES_H
#define VSLAM_DENSE_MAPPER_DENSE_TYPES_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "dense_mapper/rigid.h"

namespace vslam
{
namespace dense
{

// Surfaces fill only a few percent of a block, so small blocks waste less.
constexpr int kBlockN = 8;
constexpr int kBlockVoxels = kBlockN * kBlockN * kBlockN;
constexpr float kMaxWeight = 64.f;
constexpr double kPi = 3.14159265358979323846;
constexpr float kSdfScale = 32767.f;
constexpr float kWeightScale = 1000.f;
constexpr uint32_t kLastIdMax = 0xFFFFFFu;

using Clock = std::chrono::steady_clock;

inline double MsSince(const Clock::time_point &start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// For two camera_from_X transforms: did the camera move by more than the thresholds?
inline bool Moved(const Rigid &a, const Rigid &b, double trans_m, double rot_deg)
{
  const double moved = (a.inverse().t - b.inverse().t).norm();
  const double turned = a.q.angularDistance(b.q) * 180.0 / kPi;
  return moved >= trans_m || turned >= rot_deg;
}

struct BlockKey
{
  int x = 0;
  int y = 0;
  int z = 0;
  bool operator==(const BlockKey &other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct BlockKeyHash
{
  size_t operator()(const BlockKey &key) const
  {
    return (static_cast<size_t>(key.x) * 73856093u) ^ (static_cast<size_t>(key.y) * 19349663u) ^
           (static_cast<size_t>(key.z) * 83492791u);
  }
};

using BlockSet = std::unordered_set<BlockKey, BlockKeyHash>;
template <typename T> using BlockMap = std::unordered_map<BlockKey, T, BlockKeyHash>;

inline int VoxelIndex(int ix, int iy, int iz) { return ix + kBlockN * (iy + kBlockN * iz); }

/// sdf in [-1, 1] and weight in [0, 65.5] as 16-bit fixed point.
/// last_id is the newest keyframe that wrote or cleared this voxel. An older
/// frame cannot put its depth back on top of that. It shares a word with the
/// gray value: 24 bits of id, so ids past kLastIdMax all compare equal.
struct Voxel
{
  int16_t sdf_q = static_cast<int16_t>(kSdfScale);
  uint16_t weight_q = 0;
  uint32_t meta = 0;

  float sdf() const { return static_cast<float>(sdf_q) / kSdfScale; }
  float weight() const { return static_cast<float>(weight_q) / kWeightScale; }
  void set(float sdf_value, float weight_value)
  {
    sdf_q = static_cast<int16_t>(std::lround(std::clamp(sdf_value, -1.f, 1.f) * kSdfScale));
    weight_q = static_cast<uint16_t>(
        std::min(std::lround(std::max(weight_value, 0.f) * kWeightScale), 65535L));
  }
  uint32_t last_id() const { return meta & kLastIdMax; }
  void set_last_id(uint32_t id) { meta = (meta & ~kLastIdMax) | std::min(id, kLastIdMax); }
  uint8_t gray() const { return static_cast<uint8_t>(meta >> 24); }
  void set_gray(uint8_t value) { meta = (meta & kLastIdMax) | (static_cast<uint32_t>(value) << 24); }
};
static_assert(kMaxWeight * kWeightScale <= 65535.f, "weight range");
static_assert(sizeof(Voxel) == 8, "voxel layout");

/// Voxel size and truncation distance, and the block layout derived from them.
struct VoxelGrid
{
  double voxel = 0.05;
  double trunc = 0.15;

  VoxelGrid(double voxel_m, double truncation_m)
      : voxel(std::max(voxel_m, 0.01)), trunc(std::max(truncation_m, voxel))
  {
  }

  double blockSize() const { return voxel * kBlockN; }

  Eigen::Vector3d blockOrigin(const BlockKey &key) const
  {
    const double size = blockSize();
    return Eigen::Vector3d(key.x * size, key.y * size, key.z * size);
  }

  BlockKey keyOf(const Eigen::Vector3d &point) const
  {
    const double size = blockSize();
    return {static_cast<int>(std::floor(point.x() / size)),
            static_cast<int>(std::floor(point.y() / size)),
            static_cast<int>(std::floor(point.z() / size))};
  }
};

} // namespace dense
} // namespace vslam

#endif
