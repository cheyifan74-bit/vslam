#include "dense_mapper/dense_mapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dense_mapper/s2m2_engine.h"
#include "dense_mapper/stereo_rectifier.h"
#include "utils/print.h"

namespace vslam
{
namespace
{

constexpr int kBlockN = 16;
constexpr int kBlockVoxels = kBlockN * kBlockN * kBlockN;
constexpr float kMaxWeight = 64.f;
constexpr double kPi = 3.14159265358979323846;
// DFR2 adds a flags word after the image id; DFR1 files are still read.
constexpr char kMagic[4] = {'D', 'F', 'R', '2'};
constexpr char kMagicV1[4] = {'D', 'F', 'R', '1'};
constexpr std::streamoff kPoseOffset = 12;
constexpr uint32_t kFlagVerified = 1u;
constexpr float kSdfScale = 32767.f;
constexpr float kWeightScale = 1000.f;

using Clock = std::chrono::steady_clock;

double MsSince(const Clock::time_point &start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// p_b = q * p_a + t for a transform named b_from_a.
struct Rigid
{
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Rigid operator*(const Rigid &other) const
  {
    Rigid out;
    out.q = (q * other.q).normalized();
    out.t = q * other.t + t;
    return out;
  }
  Rigid Inverse() const
  {
    Rigid out;
    out.q = q.conjugate();
    out.t = -(out.q * t);
    return out;
  }
};

/// For two camera_from_X transforms: did the camera move by more than the thresholds?
bool Moved(const Rigid &a, const Rigid &b, double trans_m, double rot_deg)
{
  const double moved = (a.Inverse().t - b.Inverse().t).norm();
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

int VoxelIndex(int ix, int iy, int iz) { return ix + kBlockN * (iy + kBlockN * iz); }

/// sdf in [-1, 1] and weight in [0, 65.5] as 16-bit fixed point.
/// last_id is the newest keyframe that wrote or cleared this voxel. An older
/// frame cannot put its depth back on top of that.
struct Voxel
{
  int16_t sdf_q = static_cast<int16_t>(kSdfScale);
  uint16_t weight_q = 0;
  uint8_t gray = 0;
  uint32_t last_id = 0;

  float sdf() const { return static_cast<float>(sdf_q) / kSdfScale; }
  float weight() const { return static_cast<float>(weight_q) / kWeightScale; }
  void set(float sdf_value, float weight_value)
  {
    sdf_q = static_cast<int16_t>(std::lround(std::clamp(sdf_value, -1.f, 1.f) * kSdfScale));
    weight_q = static_cast<uint16_t>(
        std::min(std::lround(std::max(weight_value, 0.f) * kWeightScale), 65535L));
  }
};
static_assert(kMaxWeight * kWeightScale <= 65535.f, "weight range");

struct DepthFrame
{
  uint32_t image_id = 0;
  /// cam_from_world (rectified camera).
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  int width = 0;
  int height = 0;
  float fx = 0;
  float fy = 0;
  float cx = 0;
  float cy = 0;
  float baseline = 0;
  std::vector<uint16_t> depth_mm;
  std::vector<uint8_t> conf;
  std::vector<uint8_t> gray;
  /// False while the frame waits for integrate_delay_frames newer keyframes.
  bool integrated = false;
  /// Kept for the depth-file flag. Fusion no longer consults other frames.
  bool verified = false;
  uint32_t submap = 0;
  /// cam_from_submap the frame was last fused with.
  Rigid fused;
  BlockSet blocks;

  bool Resident() const { return !depth_mm.empty(); }
  Rigid Pose() const
  {
    Rigid pose;
    pose.q = q;
    pose.t = t;
    return pose;
  }
  Eigen::Vector3d Center() const { return -(q.conjugate() * t); }
};

struct Job
{
  bool infer = false;
  uint32_t image_id = 0;
  cv::Mat left;
  cv::Mat right;
  colmap::MappingPose cam_from_world;
  std::vector<Eigen::Vector3f> sparse_xyz;
  std::vector<uint32_t> removed_ids;
  std::vector<uint32_t> redundant_ids;
  std::vector<std::pair<uint32_t, colmap::MappingPose>> pose_updates;
  /// Filled by the inference thread; null when no depth was built.
  std::unique_ptr<DepthFrame> depth;
  double build_ms = 0.0;
  Clock::time_point submitted;
};

struct SurfacePiece
{
  std::vector<Eigen::Vector3f> xyz;
  std::vector<uint8_t> rgb;
};

/// One TSDF volume with its own frame. The submap pose follows the consensus
/// of its members, so a loop correction moves it rigidly and only members
/// that moved relative to the others are rebuilt.
struct Submap
{
  uint32_t id = 0;
  Rigid world_from_submap;
  int num_members = 0;
  std::unordered_map<BlockKey, std::vector<Voxel>, BlockKeyHash> blocks;
  std::unordered_map<BlockKey, SurfacePiece, BlockKeyHash> surface;
  BlockSet surface_dirty;
  BlockSet replay_dirty;
  std::unordered_set<uint32_t> replay_changed;
  bool pose_changed = false;
  std::shared_ptr<const DenseCloud> points;
  std::shared_ptr<const DenseSubmapCloud> published;
};

Eigen::Quaterniond QuatFromPose(const colmap::MappingPose &pose)
{
  return Eigen::Quaterniond(pose.q_xyzw[3], pose.q_xyzw[0], pose.q_xyzw[1], pose.q_xyzw[2])
      .normalized();
}

Eigen::Vector3d TransFromPose(const colmap::MappingPose &pose)
{
  return Eigen::Vector3d(pose.t[0], pose.t[1], pose.t[2]);
}

cv::Mat ToRgb(const cv::Mat &image)
{
  if (image.channels() == 3)
  {
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    return rgb;
  }
  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  return rgb;
}

/// Closest sparse depth in the rectified camera, spread by a small radius.
/// Zero means no sparse point covers that pixel.
cv::Mat SparseDepthCeiling(const std::vector<Eigen::Vector3f> &xyz, const Eigen::Matrix3d &R,
                           const Eigen::Vector3d &t, int width, int height, float fx, float fy,
                           float cx, float cy, int radius_px)
{
  cv::Mat ceiling(height, width, CV_32F, cv::Scalar(0));
  for (const Eigen::Vector3f &point : xyz)
  {
    const Eigen::Vector3d cam = R * point.cast<double>() + t;
    if (cam.z() < 0.3)
    {
      continue;
    }
    const int u = static_cast<int>(std::lround(fx * cam.x() / cam.z() + cx));
    const int v = static_cast<int>(std::lround(fy * cam.y() / cam.z() + cy));
    if (u < 0 || v < 0 || u >= width || v >= height)
    {
      continue;
    }
    float &cell = ceiling.at<float>(v, u);
    const float depth = static_cast<float>(cam.z());
    if (cell == 0.f || depth < cell)
    {
      cell = depth;
    }
  }
  if (radius_px <= 0 || cv::countNonZero(ceiling) == 0)
  {
    return ceiling;
  }
  // Empty pixels are 0. A minimum filter would spread those zeros, so empty
  // cells are filled with a sentinel and restored after the erode.
  cv::Mat filled = ceiling.clone();
  filled.setTo(1e6f, ceiling == 0);
  const int k = radius_px * 2 + 1;
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
  cv::Mat spread;
  cv::erode(filled, spread, kernel);
  spread.setTo(0.f, spread >= 1e5f);
  return spread;
}

} // namespace

struct DenseMapper::Impl
{
  DenseConfig config;
  std::vector<CameraParams> cameras;
  std::filesystem::path frame_dir;
  StereoRectifier rectifier;
  S2M2Engine engine;
  CloudCallback cloud_callback;

  // Jobs go submit -> infer_queue -> inference thread -> fuse_queue -> fusion
  // thread, in submit order. Both queues share one mutex.
  std::mutex mutex;
  std::condition_variable infer_cv;
  std::condition_variable fuse_cv;
  std::queue<Job> infer_queue;
  std::queue<Job> fuse_queue;
  bool stop = false;
  // Inference waits for the stored frames, so no new file appears mid-load.
  bool loaded = false;
  bool infer_done = false;
  bool started = false;
  std::thread infer_thread;
  std::thread fuse_thread;

  mutable std::mutex tracked_mutex;
  std::deque<uint32_t> tracked;

  double voxel = 0.05;
  double trunc = 0.15;
  std::unordered_map<uint32_t, DepthFrame> frames;
  std::deque<uint32_t> pending;
  std::map<uint32_t, Submap> submaps;
  uint32_t active_submap = 0;
  uint32_t next_submap = 1;
  std::vector<uint32_t> removed_submaps;
  int jobs_since_dirty = 0;
  Clock::time_point last_job = Clock::now();
  Clock::time_point last_publish;

  std::filesystem::path FramePath(uint32_t image_id) const
  {
    return frame_dir / (std::to_string(image_id) + ".bin");
  }

  double BlockSize() const { return voxel * kBlockN; }

  Eigen::Vector3d BlockOrigin(const BlockKey &key) const
  {
    const double size = BlockSize();
    return Eigen::Vector3d(key.x * size, key.y * size, key.z * size);
  }

  BlockKey KeyOf(const Eigen::Vector3d &point) const
  {
    const double size = BlockSize();
    return {static_cast<int>(std::floor(point.x() / size)),
            static_cast<int>(std::floor(point.y() / size)),
            static_cast<int>(std::floor(point.z() / size))};
  }

  static std::vector<Voxel> &Block(Submap &submap, const BlockKey &key)
  {
    auto it = submap.blocks.find(key);
    if (it == submap.blocks.end())
    {
      it = submap.blocks.emplace(key, std::vector<Voxel>(kBlockVoxels)).first;
    }
    return it->second;
  }

  static void ClearBlock(Submap &submap, const BlockKey &key)
  {
    auto it = submap.blocks.find(key);
    if (it != submap.blocks.end())
    {
      it->second.assign(kBlockVoxels, Voxel{});
      submap.surface_dirty.insert(key);
    }
  }

  /// Voxel by block-local index; indices one step outside the block address the neighbour.
  static const Voxel *VoxelAt(const Submap &submap, BlockKey key, int ix, int iy, int iz)
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
    const auto it = submap.blocks.find(key);
    return it == submap.blocks.end() ? nullptr
                                     : &it->second[static_cast<size_t>(VoxelIndex(ix, iy, iz))];
  }


  /// Zeros existing voxels that sit on this frame's rays, in front of its depth
  /// by more than the truncation distance. other submaps included: a revisit
  /// often lands in a different submap from the keyframe that first saw the object.
  void CarveInFront(const DepthFrame &frame, Submap &other)
  {
    const Rigid cam_from = frame.Pose() * other.world_from_submap;
    const Eigen::Matrix3d R = cam_from.q.toRotationMatrix();
    const Eigen::Matrix3d Rt = R.transpose();
    const Eigen::Vector3d t = cam_from.t;
    const Eigen::Vector3d center = -Rt * t;
    const double near = std::max(config.min_range_m, 0.05);
    const double step = std::min(trunc, BlockSize() * 0.5);
    const float trunc_f = static_cast<float>(trunc);
    const float carve_scale = static_cast<float>(config.carve_weight);
    BlockSet keys;
    for (int y = 0; y < frame.height; ++y)
    {
      for (int x = 0; x < frame.width; ++x)
      {
        const size_t idx = static_cast<size_t>(y) * frame.width + x;
        if (frame.depth_mm[idx] == 0)
        {
          continue;
        }
        const double depth = frame.depth_mm[idx] * 0.001;
        const Eigen::Vector3d dir =
            (Rt * Eigen::Vector3d((x - frame.cx) * depth / frame.fx,
                                  (y - frame.cy) * depth / frame.fy, depth))
                .normalized();
        const double length = depth * std::sqrt(
            std::pow((x - frame.cx) / frame.fx, 2) +
            std::pow((y - frame.cy) / frame.fy, 2) + 1.0);
        for (double s = near; s < length - trunc; s += step)
        {
          const BlockKey key = KeyOf(center + dir * s);
          if (other.blocks.find(key) != other.blocks.end())
          {
            keys.insert(key);
          }
        }
      }
    }
    if (keys.empty())
    {
      return;
    }
    const Eigen::Vector3d step_cam = R.col(0) * voxel;
    const double min_z = std::max(0.05, config.min_range_m * 0.5);
    for (const BlockKey &key : keys)
    {
      const auto it = other.blocks.find(key);
      if (it == other.blocks.end())
      {
        continue;
      }
      const Eigen::Vector3d origin = BlockOrigin(key);
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
            const double depth = frame.depth_mm[pidx] * 0.001;
            const double range =
                depth * std::sqrt(std::pow((u - frame.cx) / frame.fx, 2) +
                                  std::pow((v - frame.cy) / frame.fy, 2) + 1.0);
            Voxel &cell = it->second[static_cast<size_t>(VoxelIndex(ix, iy, iz))];
            if (cell.last_id > frame.image_id || cell.weight() <= 0.f)
            {
              continue;
            }
            if (static_cast<float>(range) - static_cast<float>(pc.norm()) <= trunc_f)
            {
              continue;
            }
            float weight = cell.weight() - carve_scale * kMaxWeight;
            if (weight < static_cast<float>(config.surface_min_weight))
            {
              weight = 0.f;
            }
            cell.set(1.f, weight);
            cell.last_id = frame.image_id;
            touched = true;
          }
        }
      }
      if (touched)
      {
        other.surface_dirty.insert(key);
      }
    }
  }

  /// Projective TSDF update of the frame's submap: every voxel of the candidate
  /// blocks is projected into the frame once. Candidates are the blocks around
  /// the depth plus, for free-space carving, existing blocks in the frustum.
  /// only != nullptr restricts the update (rebuild). current_pose uses the
  /// frame's pose now, so a replay carves the surface just written by an older
  /// frame instead of the place that surface used to occupy.
  bool Integrate(DepthFrame *frame, Submap &submap, const BlockSet *only, bool log,
                 bool current_pose = false)
  {
    if (frame == nullptr || !frame->Resident())
    {
      return false;
    }
    // A partial rebuild normally keeps the pose the frame's other blocks were
    // fused with. A submap replay passes current_pose so every frame, old and
    // new, is written in the same pose and the newest observation wins.
    const Rigid cam_from_submap = (only != nullptr && !current_pose)
                                      ? frame->fused
                                      : frame->Pose() * submap.world_from_submap;
    frame->fused = cam_from_submap;
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
#pragma omp parallel for schedule(dynamic, 4) reduction(+ : kept)
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
        ++kept;
      }
    }

    const double block_size = BlockSize();
    // Samples along the band at most half a block apart cannot skip a block.
    const double step = std::min(trunc, block_size * 0.5);
    const bool carve = config.carve_free_space && config.carve_weight > 0.0;
    const double carve_near = std::max(config.min_range_m, 0.05);
    BlockSet candidates;
#pragma omp parallel
    {
      BlockSet local;
#pragma omp for schedule(static) nowait
      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
          const float r = range[static_cast<size_t>(y) * width + x];
          if (r <= 0.f)
          {
            continue;
          }
          const Eigen::Vector3d dir =
              (Rt * Eigen::Vector3d((x - cx) / fx, (y - cy) / fy, 1.0)).normalized();
          // The truncation band only covers the new surface. An older surface
          // in front of it sits on the rest of the ray and has to be visited too.
          if (carve)
          {
            for (double s = carve_near; s < static_cast<double>(r) - trunc; s += step)
            {
              const BlockKey key = KeyOf(center + dir * s);
              if (submap.blocks.find(key) != submap.blocks.end())
              {
                local.insert(key);
              }
            }
          }
          for (double s = r - trunc; s < r + trunc; s += step)
          {
            local.insert(KeyOf(center + dir * s));
          }
          local.insert(KeyOf(center + dir * (r + trunc)));
        }
      }
#pragma omp critical(dense_candidates)
      candidates.insert(local.begin(), local.end());
    }
    if (carve)
    {
      const double half_diag = block_size * 0.8660254;
      const Eigen::Vector3d half(block_size * 0.5, block_size * 0.5, block_size * 0.5);
      for (const auto &item : submap.blocks)
      {
        if (candidates.count(item.first) != 0)
        {
          continue;
        }
        const Eigen::Vector3d pc = R * (BlockOrigin(item.first) + half) + t;
        if (pc.z() + half_diag < config.min_range_m || pc.z() - half_diag > config.max_range_m)
        {
          continue;
        }
        const double z = std::max(pc.z(), half_diag);
        const double margin_u = fx * half_diag / z;
        const double margin_v = fy * half_diag / z;
        const double u = fx * pc.x() / z + cx;
        const double v = fy * pc.y() / z + cy;
        if (u < -margin_u || v < -margin_v || u > width + margin_u || v > height + margin_v)
        {
          continue;
        }
        candidates.insert(item.first);
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
      cells[i] = Block(submap, keys[i]).data();
    }
    std::vector<uint8_t> hit(keys.size(), 0);
    const float trunc_f = static_cast<float>(trunc);
    const float carve_scale = carve ? static_cast<float>(config.carve_weight) : 0.f;
    const double min_z = std::max(0.05, config.min_range_m * 0.5);
    const Eigen::Vector3d step_cam = R.col(0) * voxel;
#pragma omp parallel for schedule(dynamic, 8)
    for (int b = 0; b < static_cast<int>(keys.size()); ++b)
    {
      const Eigen::Vector3d origin = BlockOrigin(keys[b]);
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
            if (cell.last_id > frame->image_id)
            {
              continue;
            }
            float w = frame->conf.empty() ? 1.f : frame->conf[pidx] / 255.f;
            float tsdf = sdf / trunc_f;
            const bool in_band = sdf <= trunc_f;
            const float old_weight = cell.weight();
            if (!in_band)
            {
              // Farther than the truncation band: this voxel sits in front of the
              // measured surface. A weighted nudge cannot beat a saturated
              // surface, so drop a share of the maximum weight instead.
              if (carve_scale <= 0.f || old_weight <= 0.f)
              {
                continue;
              }
              float weight = old_weight - carve_scale * kMaxWeight;
              if (weight < static_cast<float>(config.surface_min_weight))
              {
                weight = 0.f;
              }
              cell.set(1.f, weight);
              cell.last_id = frame->image_id;
              touched = true;
              continue;
            }
            if (w <= 0.f)
            {
              continue;
            }
            const float sum = old_weight + w;
            if (in_band)
            {
              const float g = frame->gray.empty() ? 180.f : static_cast<float>(frame->gray[pidx]);
              cell.gray = static_cast<uint8_t>(
                  std::lround((static_cast<float>(cell.gray) * old_weight + g * w) / sum));
            }
            cell.set((cell.sdf() * old_weight + tsdf * w) / sum, std::min(sum, kMaxWeight));
            cell.last_id = frame->image_id;
            touched = true;
          }
        }
      }
      hit[b] = touched ? 1 : 0;
    }

    BlockSet touched;
    for (size_t i = 0; i < keys.size(); ++i)
    {
      if (hit[i] != 0)
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
      PRINT_INFO("[DENSE]: fuse image %u submap %u keep=%d blocks=%zu/%zu\n",
                 frame->image_id, submap.id, kept, touched.size(), keys.size());
    }
    if (only == nullptr)
    {
      frame->blocks = std::move(touched);
      if (carve)
      {
        for (auto &item : submaps)
        {
          if (item.first != submap.id)
          {
            CarveInFront(*frame, item.second);
          }
        }
      }
    }
    else
    {
      for (const BlockKey &key : *only)
      {
        frame->blocks.erase(key);
      }
      frame->blocks.insert(touched.begin(), touched.end());
    }
    return false;
  }

  bool WriteFrame(const DepthFrame &frame) const
  {
    std::ofstream out(FramePath(frame.image_id), std::ios::binary);
    if (!out)
    {
      return false;
    }
    out.write(kMagic, 4);
    out.write(reinterpret_cast<const char *>(&frame.image_id), sizeof(frame.image_id));
    const uint32_t flags = frame.verified ? kFlagVerified : 0u;
    out.write(reinterpret_cast<const char *>(&flags), sizeof(flags));
    const double q[4] = {frame.q.x(), frame.q.y(), frame.q.z(), frame.q.w()};
    const double t[3] = {frame.t.x(), frame.t.y(), frame.t.z()};
    out.write(reinterpret_cast<const char *>(q), sizeof(q));
    out.write(reinterpret_cast<const char *>(t), sizeof(t));
    const int wh[2] = {frame.width, frame.height};
    out.write(reinterpret_cast<const char *>(wh), sizeof(wh));
    const float cam[5] = {frame.fx, frame.fy, frame.cx, frame.cy, frame.baseline};
    out.write(reinterpret_cast<const char *>(cam), sizeof(cam));
    out.write(reinterpret_cast<const char *>(frame.depth_mm.data()),
              static_cast<std::streamsize>(frame.depth_mm.size() * sizeof(uint16_t)));
    out.write(reinterpret_cast<const char *>(frame.conf.data()),
              static_cast<std::streamsize>(frame.conf.size()));
    out.write(reinterpret_cast<const char *>(frame.gray.data()),
              static_cast<std::streamsize>(frame.gray.size()));
    return static_cast<bool>(out);
  }

  /// Rewrites only the pose; the file must already be DFR2.
  bool WritePose(const DepthFrame &frame) const
  {
    std::fstream io(FramePath(frame.image_id), std::ios::in | std::ios::out | std::ios::binary);
    if (!io)
    {
      return false;
    }
    const double q[4] = {frame.q.x(), frame.q.y(), frame.q.z(), frame.q.w()};
    const double t[3] = {frame.t.x(), frame.t.y(), frame.t.z()};
    io.seekp(kPoseOffset);
    io.write(reinterpret_cast<const char *>(q), sizeof(q));
    io.write(reinterpret_cast<const char *>(t), sizeof(t));
    return static_cast<bool>(io);
  }

  /// pixels = false reads only the header (pose, intrinsics, flags).
  /// *legacy is set for DFR1 files, which must be rewritten before WritePose works.
  bool ReadFrame(const std::filesystem::path &path, DepthFrame *frame, bool pixels,
                 bool *legacy = nullptr) const
  {
    std::ifstream in(path, std::ios::binary);
    if (!in || frame == nullptr)
    {
      return false;
    }
    char magic[4];
    in.read(magic, 4);
    if (!in)
    {
      return false;
    }
    const bool v1 = std::equal(magic, magic + 4, kMagicV1);
    if (!v1 && !std::equal(magic, magic + 4, kMagic))
    {
      return false;
    }
    if (legacy != nullptr)
    {
      *legacy = v1;
    }
    in.read(reinterpret_cast<char *>(&frame->image_id), sizeof(frame->image_id));
    uint32_t flags = 0;
    if (!v1)
    {
      in.read(reinterpret_cast<char *>(&flags), sizeof(flags));
    }
    frame->verified = (flags & kFlagVerified) != 0;
    double q[4];
    double t[3];
    in.read(reinterpret_cast<char *>(q), sizeof(q));
    in.read(reinterpret_cast<char *>(t), sizeof(t));
    frame->q = Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized();
    frame->t = Eigen::Vector3d(t[0], t[1], t[2]);
    int wh[2];
    in.read(reinterpret_cast<char *>(wh), sizeof(wh));
    frame->width = wh[0];
    frame->height = wh[1];
    float cam[5];
    in.read(reinterpret_cast<char *>(cam), sizeof(cam));
    frame->fx = cam[0];
    frame->fy = cam[1];
    frame->cx = cam[2];
    frame->cy = cam[3];
    frame->baseline = cam[4];
    if (!pixels)
    {
      return static_cast<bool>(in);
    }
    const size_t count = static_cast<size_t>(frame->width) * static_cast<size_t>(frame->height);
    frame->depth_mm.resize(count);
    frame->conf.resize(count);
    frame->gray.resize(count);
    in.read(reinterpret_cast<char *>(frame->depth_mm.data()),
            static_cast<std::streamsize>(count * sizeof(uint16_t)));
    in.read(reinterpret_cast<char *>(frame->conf.data()), static_cast<std::streamsize>(count));
    in.read(reinterpret_cast<char *>(frame->gray.data()), static_cast<std::streamsize>(count));
    return static_cast<bool>(in);
  }

  /// Reads the pixels of an evicted frame back; the in-memory pose wins.
  bool LoadPixels(DepthFrame *frame) const
  {
    DepthFrame stored;
    if (!ReadFrame(FramePath(frame->image_id), &stored, true) ||
        stored.width != frame->width || stored.height != frame->height)
    {
      PRINT_WARNING("[DENSE]: cannot reload depth frame %u\n", frame->image_id);
      return false;
    }
    frame->depth_mm.swap(stored.depth_mm);
    frame->conf.swap(stored.conf);
    frame->gray.swap(stored.gray);
    frame->verified = stored.verified;
    return true;
  }

  static void ReleasePixels(DepthFrame *frame)
  {
    std::vector<uint16_t>().swap(frame->depth_mm);
    std::vector<uint8_t>().swap(frame->conf);
    std::vector<uint8_t>().swap(frame->gray);
  }

  /// Integrate() for any fused frame: evicted pixels are read back for the
  /// call and released again.
  void Fuse(DepthFrame *frame, Submap &submap, const BlockSet *only, bool log,
            bool current_pose = false)
  {
    const bool resident = frame->Resident();
    if (!resident && !LoadPixels(frame))
    {
      return;
    }
    if (Integrate(frame, submap, only, log, current_pose))
    {
      WriteFrame(*frame);
    }
    if (!resident)
    {
      ReleasePixels(frame);
    }
  }

  /// Keeps the pixels of the newest resident_frames fused frames (verification
  /// references) and of every pending frame; older frames keep only their pose
  /// and block list. Their files are current, so nothing is written here.
  void EvictOld()
  {
    if (config.resident_frames <= 0)
    {
      return;
    }
    std::vector<uint32_t> resident;
    for (const auto &item : frames)
    {
      if (item.second.integrated && item.second.Resident())
      {
        resident.push_back(item.first);
      }
    }
    const size_t keep = static_cast<size_t>(config.resident_frames);
    if (resident.size() <= keep)
    {
      return;
    }
    std::sort(resident.begin(), resident.end(), std::greater<uint32_t>());
    for (size_t i = keep; i < resident.size(); ++i)
    {
      ReleasePixels(&frames[resident[i]]);
    }
  }

  // ------------------------------------------------------------------ submaps

  Submap &CreateSubmap(const DepthFrame &first)
  {
    Submap submap;
    submap.id = next_submap++;
    submap.world_from_submap = first.Pose().Inverse();
    submap.pose_changed = true;
    const uint32_t id = submap.id;
    auto &stored = submaps.emplace(id, std::move(submap)).first->second;
    PRINT_INFO("[DENSE]: new submap %u at image %u (%zu submaps)\n", id, first.image_id,
               submaps.size());
    return stored;
  }

  /// Share of the frame's depth samples that land in allocated blocks of the submap.
  double Overlap(const DepthFrame &frame, const Submap &submap) const
  {
    if (!frame.Resident() || submap.blocks.empty())
    {
      return 0.0;
    }
    const Rigid cam_from_submap = frame.Pose() * submap.world_from_submap;
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
        inside += submap.blocks.count(KeyOf(Rt * p_cam + center)) != 0 ? 1 : 0;
      }
    }
    return samples > 0 ? static_cast<double>(inside) / samples : 0.0;
  }

  /// The submap that already holds most of what the keyframe sees (a revisit),
  /// else the active submap while it has room and the keyframe stays within
  /// submap_max_distance_m of its origin, else a new submap.
  Submap &ChooseSubmap(const DepthFrame &frame)
  {
    if (config.submap_max_frames <= 0)
    {
      return submaps.empty() ? CreateSubmap(frame) : submaps.begin()->second;
    }
    const Eigen::Vector3d center = frame.Center();
    const double reach = config.submap_max_distance_m + config.max_range_m;
    const auto has_room = [&](const Submap &submap) {
      return submap.num_members < config.submap_max_frames;
    };
    Submap *active = nullptr;
    double active_overlap = 0.0;
    Submap *best = nullptr;
    double best_overlap = 0.0;
    for (auto &item : submaps)
    {
      Submap &submap = item.second;
      if (!has_room(submap) || (center - submap.world_from_submap.t).norm() > reach)
      {
        continue;
      }
      const double overlap = Overlap(frame, submap);
      if (item.first == active_submap)
      {
        active = &submap;
        active_overlap = overlap;
      }
      if (overlap > best_overlap)
      {
        best = &submap;
        best_overlap = overlap;
      }
    }
    const bool active_near =
        active != nullptr &&
        (center - active->world_from_submap.t).norm() <= config.submap_max_distance_m;
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
      chosen = &CreateSubmap(frame);
    }
    active_submap = chosen->id;
    return *chosen;
  }

  /// After pose updates: every member implies a submap pose (its current world
  /// pose combined with the pose it was fused with). The submap takes the mean
  /// of the members that agree, so a loop correction moves it rigidly while a
  /// few locally refined members stay outliers. Members that disagree with the
  /// result beyond the thresholds are rebuilt.
  void RefreshSubmap(Submap &submap)
  {
    std::vector<const DepthFrame *> members;
    for (const auto &item : frames)
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
      return Moved(frame.Pose() * world_from_submap, frame.fused, config.pose_update_trans_m,
                   config.pose_update_rot_deg);
    };
    const Eigen::Quaterniond reference = submap.world_from_submap.q;
    const auto mean = [&](const std::vector<const DepthFrame *> &set) {
      Eigen::Vector4d q_sum = Eigen::Vector4d::Zero();
      Eigen::Vector3d t_sum = Eigen::Vector3d::Zero();
      for (const DepthFrame *frame : set)
      {
        const Rigid implied = frame->Pose().Inverse() * frame->fused;
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
    if (Moved(consensus.Inverse(), submap.world_from_submap.Inverse(), 1e-4, 1e-3))
    {
      submap.world_from_submap = consensus;
      submap.pose_changed = true;
    }
    for (const DepthFrame *frame : members)
    {
      if (disagrees(*frame, submap.world_from_submap))
      {
        submap.replay_dirty.insert(frame->blocks.begin(), frame->blocks.end());
        submap.replay_changed.insert(frame->image_id);
      }
    }
  }

  // ------------------------------------------------------------------ surface

  /// Surface points at the sdf zero crossings towards the +x/+y/+z neighbours,
  /// averaged to one point per voxel. Submap frame.
  void ExtractBlock(const Submap &submap, const BlockKey &key, const std::vector<Voxel> &voxels,
                    SurfacePiece *piece) const
  {
    const Eigen::Vector3d origin = BlockOrigin(key);
    const float min_weight = static_cast<float>(config.surface_min_weight);
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
          const Eigen::Vector3d p = origin + Eigen::Vector3d((ix + 0.5) * voxel, (iy + 0.5) * voxel,
                                                             (iz + 0.5) * voxel);
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
                                    : VoxelAt(submap, key, nx, ny, nz);
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
            gray_sum += cell.gray + a * (static_cast<double>(next->gray) - cell.gray);
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

  void ExtractDirtySurface(Submap &submap)
  {
    // Crossings towards +x/+y/+z read the next block, so the previous block changes too.
    BlockSet todo = submap.surface_dirty;
    for (const BlockKey &key : submap.surface_dirty)
    {
      const BlockKey lower[3] = {{key.x - 1, key.y, key.z}, {key.x, key.y - 1, key.z},
                                 {key.x, key.y, key.z - 1}};
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
        ExtractBlock(submap, keys[i], it->second, &pieces[i]);
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

  bool HasUnpublished() const
  {
    if (!removed_submaps.empty())
    {
      return true;
    }
    for (const auto &item : submaps)
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
  void Publish()
  {
    if (!cloud_callback || !HasUnpublished())
    {
      return;
    }
    const auto start = Clock::now();
    auto update = std::make_shared<DenseUpdate>();
    update->point_size_m = static_cast<float>(voxel);
    update->removed.swap(removed_submaps);
    size_t changed_points = 0;
    size_t total_points = 0;
    size_t num_blocks = 0;
    for (auto &item : submaps)
    {
      Submap &submap = item.second;
      const bool surface_changed = !submap.surface_dirty.empty();
      if (surface_changed)
      {
        ExtractDirtySurface(submap);
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
    }
    last_publish = Clock::now();
    const size_t num_changed = update->changed.size();
    const size_t num_removed = update->removed.size();
    cloud_callback(std::move(update));

    size_t resident = 0;
    size_t pixel_bytes = 0;
    for (const auto &item : frames)
    {
      if (item.second.Resident())
      {
        ++resident;
        pixel_bytes += item.second.depth_mm.size() * sizeof(uint16_t) + item.second.conf.size() +
                       item.second.gray.size();
      }
    }
    const double volume_mb = static_cast<double>(num_blocks) * kBlockVoxels * sizeof(Voxel) / 1e6;
    PRINT_INFO("[DENSE]: publish submaps changed=%zu removed=%zu (%zu pts) total submaps=%zu "
               "points=%zu blocks=%zu (%.0f MB) frames=%zu resident=%zu (%.0f MB) pending=%zu "
               "(%.0f ms)\n",
               num_changed, num_removed, changed_points, submaps.size(), total_points, num_blocks,
               volume_mb, frames.size(), resident, pixel_bytes / 1e6, pending.size(),
               MsSince(start));
  }

  void MaybePublish()
  {
    const double elapsed = std::chrono::duration<double>(Clock::now() - last_publish).count();
    if (elapsed >= config.publish_period_s && HasUnpublished())
    {
      Publish();
    }
  }

  // ------------------------------------------------------------------ depth

  bool BuildDepth(const Job &job, DepthFrame *frame)
  {
    if (frame == nullptr || !engine.ready() || job.left.empty() || job.right.empty())
    {
      return false;
    }
    const auto start = Clock::now();
    cv::Mat left_rect;
    cv::Mat right_rect;
    cv::Mat valid;
    if (!rectifier.rectify(job.left, job.right, &left_rect, &right_rect, &valid))
    {
      return false;
    }
    const cv::Mat left_rgb = ToRgb(left_rect);
    const cv::Mat right_rgb = ToRgb(right_rect);
    std::vector<float> disparity;
    std::vector<float> occlusion;
    std::vector<float> confidence;
    if (!engine.infer(left_rgb, right_rgb, &disparity, &occlusion, &confidence))
    {
      return false;
    }
    const double infer_ms = MsSince(start);
    const Eigen::Matrix3d R1 = RectFromLeft();
    const Eigen::Matrix3d R_left = QuatFromPose(job.cam_from_world).toRotationMatrix();
    const Eigen::Vector3d t_left = TransFromPose(job.cam_from_world);
    frame->image_id = job.image_id;
    frame->q = Eigen::Quaterniond(R1 * R_left).normalized();
    frame->t = R1 * t_left;
    frame->width = rectifier.width();
    frame->height = rectifier.height();
    frame->fx = rectifier.fx();
    frame->fy = rectifier.fy();
    frame->cx = rectifier.cx();
    frame->cy = rectifier.cy();
    frame->baseline = rectifier.baseline();
    const int width = frame->width;
    const int height = frame->height;
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    frame->depth_mm.assign(pixels, 0);
    frame->conf.assign(pixels, 0);
    frame->gray.assign(pixels, 0);
    cv::Mat gray;
    if (left_rect.channels() == 1)
    {
      gray = left_rect;
    }
    else
    {
      cv::cvtColor(left_rect, gray, cv::COLOR_BGR2GRAY);
    }
    const bool use_sparse = config.sparse_depth_ratio > 1.0 && !job.sparse_xyz.empty();
    const cv::Mat ceiling =
        use_sparse ? SparseDepthCeiling(job.sparse_xyz, frame->q.toRotationMatrix(), frame->t,
                                         width, height, frame->fx, frame->fy, frame->cx,
                                         frame->cy, config.sparse_dilate_px)
                   : cv::Mat();
    const float *ceiling_data =
        ceiling.empty() ? nullptr : reinterpret_cast<const float *>(ceiling.data);
    std::vector<float> depth(pixels, 0.f);
    std::vector<float> weight(pixels, 0.f);
    int saturated = 0;
    int behind_sparse = 0;
    for (size_t i = 0; i < pixels; ++i)
    {
      if (valid.empty() || valid.data[i] == 0 || !std::isfinite(disparity[i]) || disparity[i] < 0.5f ||
          occlusion[i] < static_cast<float>(config.min_visible) ||
          confidence[i] < static_cast<float>(config.min_confidence))
      {
        continue;
      }
      if (config.saturate_gray > 0 && gray.data[i] >= static_cast<uint8_t>(config.saturate_gray))
      {
        ++saturated;
        continue;
      }
      const double d = static_cast<double>(frame->fx) * frame->baseline / disparity[i];
      if (d < config.min_range_m || d > config.max_range_m)
      {
        continue;
      }
      if (ceiling_data != nullptr && ceiling_data[i] > 0.f)
      {
        const double sparse_z = static_cast<double>(ceiling_data[i]);
        const double limit = std::max(sparse_z * config.sparse_depth_ratio,
                                       sparse_z + config.sparse_depth_margin_m);
        if (d > limit)
        {
          ++behind_sparse;
          continue;
        }
      }
      float w = std::clamp(confidence[i], 0.f, 1.f);
      if (config.weight_reference_range_m > 0.0)
      {
        const double z_ref = config.weight_reference_range_m;
        const double scale = (z_ref / d) * (z_ref / d);
        w = static_cast<float>(std::min(1.0, scale) * w);
      }
      depth[i] = static_cast<float>(d);
      weight[i] = w;
    }
    int edges = 0;
    if (config.edge_ratio > 0.0)
    {
      const float ratio = static_cast<float>(config.edge_ratio);
      std::vector<uint8_t> drop(pixels, 0);
      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
          const size_t i = static_cast<size_t>(y) * width + x;
          const float d = depth[i];
          if (d <= 0.f)
          {
            continue;
          }
          const float limit = ratio * d;
          const auto jump = [&](int nx, int ny) {
            if (nx < 0 || ny < 0 || nx >= width || ny >= height)
            {
              return false;
            }
            const float other = depth[static_cast<size_t>(ny) * width + nx];
            return other > 0.f && std::abs(other - d) > limit;
          };
          if (jump(x - 1, y) || jump(x + 1, y) || jump(x, y - 1) || jump(x, y + 1))
          {
            drop[i] = 1;
          }
        }
      }
      for (size_t i = 0; i < pixels; ++i)
      {
        if (drop[i] != 0)
        {
          depth[i] = 0.f;
          ++edges;
        }
      }
    }
    int kept = 0;
    for (size_t i = 0; i < pixels; ++i)
    {
      if (depth[i] <= 0.f)
      {
        continue;
      }
      frame->depth_mm[i] = static_cast<uint16_t>(std::min(depth[i] * 1000.0, 65535.0));
      frame->conf[i] = static_cast<uint8_t>(std::clamp(weight[i], 0.f, 1.f) * 255.f);
      frame->gray[i] = gray.data[i];
      ++kept;
    }
    PRINT_INFO("[DENSE]: image %u depth pixels=%d / %zu saturated=%d behind_sparse=%d edge=%d "
               "infer=%.0fms\n",
               job.image_id, kept, pixels, saturated, behind_sparse, edges, infer_ms);
    return kept > 0;
  }

  Eigen::Matrix3d RectFromLeft() const
  {
    const cv::Matx33d &R_rect = rectifier.rectFromLeft();
    return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_rect.val);
  }

  // ------------------------------------------------------------------ fusion

  /// Stores the new world pose and collects the submap of a fused frame.
  void UpdatePose(uint32_t id, const colmap::MappingPose &pose, const Eigen::Matrix3d &R1,
                  std::unordered_set<uint32_t> *touched)
  {
    auto it = frames.find(id);
    if (it == frames.end())
    {
      return;
    }
    DepthFrame &frame = it->second;
    Rigid updated;
    updated.q = Eigen::Quaterniond(R1 * QuatFromPose(pose).toRotationMatrix()).normalized();
    updated.t = R1 * TransFromPose(pose);
    if (!Moved(updated, frame.Pose(), 1e-4, 1e-3))
    {
      return;
    }
    frame.q = updated.q;
    frame.t = updated.t;
    WritePose(frame);
    if (frame.integrated)
    {
      touched->insert(frame.submap);
    }
  }

  /// Fuses the oldest pending frames until at most keep remain.
  int FusePending(size_t keep)
  {
    int fused = 0;
    while (pending.size() > keep)
    {
      const uint32_t id = pending.front();
      pending.pop_front();
      auto it = frames.find(id);
      if (it == frames.end() || it->second.integrated)
      {
        continue;
      }
      DepthFrame &frame = it->second;
      Submap &submap = ChooseSubmap(frame);
      frame.submap = submap.id;
      Integrate(&frame, submap, nullptr, true);
      frame.integrated = true;
      ++submap.num_members;
      WriteFrame(frame);
      ++fused;
    }
    if (fused > 0)
    {
      EvictOld();
    }
    return fused;
  }

  bool ReplayPending() const
  {
    for (const auto &item : submaps)
    {
      if (!item.second.replay_dirty.empty())
      {
        return true;
      }
    }
    return false;
  }

  /// Rebuilds dirty blocks from the frames whose pose changed, oldest first.
  /// Voxels a newer keyframe already wrote or cleared are left in place, so
  /// replaying an older depth cannot bring a replaced surface back.
  void FlushSubmap(Submap &submap)
  {
    const auto start = Clock::now();
    std::vector<DepthFrame *> members;
    for (auto &item : frames)
    {
      if (item.second.integrated && item.second.submap == submap.id)
      {
        members.push_back(&item.second);
      }
    }
    std::sort(members.begin(), members.end(),
              [](const DepthFrame *a, const DepthFrame *b) { return a->image_id < b->image_id; });
    uint32_t changed_max = 0;
    bool any_changed = false;
    for (const DepthFrame *frame : members)
    {
      if (submap.replay_changed.count(frame->image_id) == 0)
      {
        continue;
      }
      changed_max = std::max(changed_max, frame->image_id);
      any_changed = true;
    }
    if (!any_changed)
    {
      for (const BlockKey &key : submap.replay_dirty)
      {
        ClearBlock(submap, key);
      }
    }
    else
    {
      for (const BlockKey &key : submap.replay_dirty)
      {
        const auto it = submap.blocks.find(key);
        if (it == submap.blocks.end())
        {
          continue;
        }
        for (Voxel &voxel : it->second)
        {
          if (voxel.last_id <= changed_max)
          {
            voxel = Voxel{};
          }
        }
        submap.surface_dirty.insert(key);
      }
    }
    const auto touches_dirty = [&](const DepthFrame *frame) {
      if (submap.replay_changed.count(frame->image_id) != 0)
      {
        return true;
      }
      for (const BlockKey &key : frame->blocks)
      {
        if (submap.replay_dirty.count(key) != 0)
        {
          return true;
        }
      }
      return false;
    };
    size_t replayed = 0;
    size_t moved = 0;
    for (DepthFrame *frame : members)
    {
      if (any_changed && frame->image_id > changed_max)
      {
        continue;
      }
      if (!touches_dirty(frame))
      {
        continue;
      }
      Fuse(frame, submap, &submap.replay_dirty, false, true);
      ++replayed;
      if (submap.replay_changed.count(frame->image_id) != 0)
      {
        ++moved;
      }
    }
    // Blocks nobody wrote again (e.g. only a removed frame saw them) are freed.
    size_t freed = 0;
    for (const BlockKey &key : submap.replay_dirty)
    {
      const auto it = submap.blocks.find(key);
      if (it != submap.blocks.end() &&
          std::all_of(it->second.begin(), it->second.end(),
                      [](const Voxel &v) { return v.weight_q == 0 && v.last_id == 0; }))
      {
        submap.blocks.erase(it);
        ++freed;
      }
    }
    PRINT_INFO("[DENSE]: rebuild submap %u blocks=%zu frames=%zu / %zu (moved=%zu) freed=%zu in "
               "%.0f ms\n",
               submap.id, submap.replay_dirty.size(), replayed, members.size(), moved, freed,
               MsSince(start));
    submap.replay_dirty.clear();
    submap.replay_changed.clear();
  }

  /// Rebuilds one submap per call, newest first, so a loop that dirties many
  /// submaps never blocks the fusion thread for more than one submap. Moved
  /// submaps are already published at their new pose in the meantime.
  void FlushReplay()
  {
    for (auto it = submaps.rbegin(); it != submaps.rend(); ++it)
    {
      if (!it->second.replay_dirty.empty())
      {
        FlushSubmap(it->second);
        break;
      }
    }
    if (!ReplayPending())
    {
      jobs_since_dirty = 0;
    }
    std::vector<uint32_t> emptied;
    for (auto &item : submaps)
    {
      Submap &submap = item.second;
      if (submap.replay_dirty.empty())
      {
        submap.replay_changed.clear();
      }
      if (submap.blocks.empty() && submap.num_members <= 0)
      {
        emptied.push_back(item.first);
      }
    }
    for (const uint32_t id : emptied)
    {
      submaps.erase(id);
      removed_submaps.push_back(id);
      if (active_submap == id)
      {
        active_submap = 0;
      }
    }
  }

  void ApplyJob(Job job, bool idle, size_t queued)
  {
    const auto start = Clock::now();
    last_job = start;
    // A redundant keyframe's depth is still a valid observation, so it stays in
    // the volume; it is only lost if one of its blocks is rebuilt later.
    const std::unordered_set<uint32_t> redundant(job.redundant_ids.begin(), job.redundant_ids.end());
    int retired = 0;
    for (const uint32_t id : job.removed_ids)
    {
      auto it = frames.find(id);
      if (it != frames.end())
      {
        const DepthFrame &frame = it->second;
        const bool keep = !config.rebuild_redundant_removals && redundant.count(id) != 0;
        auto submap = submaps.find(frame.submap);
        if (frame.integrated && submap != submaps.end())
        {
          if (!keep)
          {
            submap->second.replay_dirty.insert(frame.blocks.begin(), frame.blocks.end());
          }
          --submap->second.num_members;
        }
        retired += keep ? 1 : 0;
        frames.erase(it);
      }
      std::filesystem::remove(FramePath(id));
    }

    const Eigen::Matrix3d R1 = RectFromLeft();
    std::unordered_set<uint32_t> touched;
    for (const auto &item : job.pose_updates)
    {
      UpdatePose(item.first, item.second, R1, &touched);
    }
    for (const uint32_t id : touched)
    {
      auto submap = submaps.find(id);
      if (submap != submaps.end())
      {
        RefreshSubmap(submap->second);
      }
    }

    if (job.depth)
    {
      const uint32_t id = job.depth->image_id;
      auto it = frames.find(id);
      if (it != frames.end() && it->second.integrated)
      {
        auto submap = submaps.find(it->second.submap);
        if (submap != submaps.end())
        {
          submap->second.replay_dirty.insert(it->second.blocks.begin(), it->second.blocks.end());
          --submap->second.num_members;
        }
      }
      frames[id] = std::move(*job.depth);
      job.depth.reset();
      pending.push_back(id);
    }

    const auto fuse_start = Clock::now();
    const int fused = FusePending(static_cast<size_t>(std::max(0, config.integrate_delay_frames)));
    const double fuse_ms = MsSince(fuse_start);

    double replay_ms = 0.0;
    if (ReplayPending())
    {
      ++jobs_since_dirty;
      if (idle || jobs_since_dirty >= std::max(1, config.replay_max_defer_jobs))
      {
        const auto replay_start = Clock::now();
        FlushReplay();
        replay_ms = MsSince(replay_start);
      }
    }

    const auto publish_start = Clock::now();
    MaybePublish();
    const double publish_ms = MsSince(publish_start);
    // build runs on the inference thread; apply is the fusion-thread time;
    // latency is submit -> applied, including both queues.
    PRINT_INFO("[DENSE]: job image %u build=%.0fms fuse=%.0fms (%d) rebuild=%.0fms publish=%.0fms "
               "apply=%.0fms latency=%.0fms queue=%zu pending=%zu retired=%d moved_submaps=%zu\n",
               job.image_id, job.build_ms, fuse_ms, fused, replay_ms, publish_ms, MsSince(start),
               MsSince(job.submitted), queued, pending.size(), retired, touched.size());
  }

  void OnIdle()
  {
    if (ReplayPending())
    {
      FlushReplay();
    }
    if (!pending.empty() && config.idle_flush_s > 0.0 &&
        std::chrono::duration<double>(Clock::now() - last_job).count() >= config.idle_flush_s)
    {
      FusePending(0);
    }
    MaybePublish();
  }

  void LoadStoredFrames()
  {
    if (!std::filesystem::exists(frame_dir))
    {
      return;
    }
    const auto start = Clock::now();
    std::vector<uint32_t> ids;
    std::unordered_set<uint32_t> legacy_ids;
    for (const auto &entry : std::filesystem::directory_iterator(frame_dir))
    {
      if (entry.path().extension() != ".bin")
      {
        continue;
      }
      DepthFrame frame;
      bool legacy = false;
      if (!ReadFrame(entry.path(), &frame, false, &legacy))
      {
        continue;
      }
      ids.push_back(frame.image_id);
      if (legacy)
      {
        legacy_ids.insert(frame.image_id);
      }
      frames[frame.image_id] = std::move(frame);
    }
    if (ids.empty())
    {
      return;
    }
    std::sort(ids.begin(), ids.end());
    // Submaps are not stored: the same assignment rule rebuilds them in id order.
    for (const uint32_t id : ids)
    {
      DepthFrame &frame = frames[id];
      if (legacy_ids.count(id) != 0 && !frame.Resident())
      {
        frames.erase(id);
        continue;
      }
      // The submap choice looks at the frame's depth, so it is read before fusing.
      const bool resident = frame.Resident();
      if (!resident && !LoadPixels(&frame))
      {
        frames.erase(id);
        continue;
      }
      Submap &submap = ChooseSubmap(frame);
      frame.submap = submap.id;
      Fuse(&frame, submap, nullptr, false);
      frame.integrated = true;
      ++submap.num_members;
      if (legacy_ids.count(id) != 0)
      {
        WriteFrame(frame);
      }
      if (!resident)
      {
        ReleasePixels(&frame);
      }
    }
    // Keep the newest stored frames resident for the next session's fusion.
    const size_t keep = config.resident_frames > 0 ? static_cast<size_t>(config.resident_frames) : ids.size();
    for (size_t i = ids.size() > keep ? ids.size() - keep : 0; i < ids.size(); ++i)
    {
      const auto it = frames.find(ids[i]);
      if (it != frames.end() && !it->second.Resident())
      {
        LoadPixels(&it->second);
      }
    }
    EvictOld();
    PRINT_INFO("[DENSE]: replayed %zu stored depth frame(s) (%zu legacy) into %zu submap(s) in "
               "%.0f ms\n",
               ids.size(), legacy_ids.size(), submaps.size(), MsSince(start));
    Publish();
  }

  // ------------------------------------------------------------------ threads

  /// Rectification, S²M² and depth filtering. Touches only the job and its own
  /// new frame file, never the volume or the frame table.
  void InferLoop()
  {
    while (true)
    {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        infer_cv.wait(lock, [this] { return loaded && (stop || !infer_queue.empty()); });
        if (infer_queue.empty())
        {
          infer_done = true;
          break;
        }
        job = std::move(infer_queue.front());
        infer_queue.pop();
      }
      if (job.infer)
      {
        const auto start = Clock::now();
        auto frame = std::make_unique<DepthFrame>();
        if (BuildDepth(job, frame.get()))
        {
          WriteFrame(*frame);
          job.depth = std::move(frame);
        }
        job.left.release();
        job.right.release();
        job.sparse_xyz.clear();
        job.build_ms = MsSince(start);
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        fuse_queue.push(std::move(job));
      }
      fuse_cv.notify_one();
    }
    fuse_cv.notify_all();
  }

  void FuseLoop()
  {
#ifdef _OPENMP
    // Bound fusion/rebuild parallelism so Dense does not starve the VIO thread.
    if (config.num_threads > 0)
    {
      omp_set_num_threads(config.num_threads);
    }
#endif
    LoadStoredFrames();
    {
      std::lock_guard<std::mutex> lock(mutex);
      loaded = true;
    }
    infer_cv.notify_all();
    const std::chrono::duration<double> wait(std::clamp(config.publish_period_s * 0.5, 0.05, 0.5));
    while (true)
    {
      Job job;
      bool has_job = false;
      bool idle = false;
      size_t queued = 0;
      {
        std::unique_lock<std::mutex> lock(mutex);
        fuse_cv.wait_for(lock, wait, [this] { return infer_done || !fuse_queue.empty(); });
        if (fuse_queue.empty() && infer_done)
        {
          break;
        }
        if (!fuse_queue.empty())
        {
          job = std::move(fuse_queue.front());
          fuse_queue.pop();
          has_job = true;
        }
        // A frame still on the GPU does not block a rebuild: running it now
        // overlaps it with the inference instead of delaying the next fusion.
        idle = fuse_queue.empty();
        queued = fuse_queue.size() + infer_queue.size();
      }
      if (has_job)
      {
        ApplyJob(std::move(job), idle, queued);
      }
      else
      {
        OnIdle();
      }
    }
    // Pending frames were stored with their first pose; keep the refined one.
    for (const uint32_t id : pending)
    {
      const auto it = frames.find(id);
      if (it != frames.end())
      {
        WriteFrame(it->second);
      }
    }
  }
};

DenseMapper::DenseMapper(DenseConfig config, std::vector<CameraParams> cameras, std::string map_dir)
    : impl_(std::make_unique<Impl>())
{
  impl_->config = std::move(config);
  impl_->cameras = std::move(cameras);
  impl_->frame_dir = std::filesystem::path(map_dir) / "dense" / "frames";
  impl_->voxel = std::max(impl_->config.voxel_m, 0.01);
  impl_->trunc = std::max(impl_->config.truncation_m, impl_->voxel);
}

DenseMapper::~DenseMapper() { stop(); }

void DenseMapper::setCloudCallback(CloudCallback callback)
{
  impl_->cloud_callback = std::move(callback);
}

void DenseMapper::start()
{
  if (!impl_->config.enabled || impl_->started)
  {
    return;
  }
  std::filesystem::create_directories(impl_->frame_dir);
  if (!impl_->rectifier.init(impl_->cameras, impl_->config.rect_width, impl_->config.rect_height))
  {
    PRINT_WARNING("[DENSE]: rectifier failed, dense mapping disabled\n");
    impl_->config.enabled = false;
    return;
  }
  impl_->engine.load(impl_->config.engine_path, impl_->config.rect_width, impl_->config.rect_height);
  impl_->started = true;
  impl_->fuse_thread = std::thread([this] { impl_->FuseLoop(); });
  impl_->infer_thread = std::thread([this] { impl_->InferLoop(); });
  PRINT_INFO("[DENSE]: mapper started, engine=%s\n",
             impl_->engine.ready() ? "ready" : "missing");
}

void DenseMapper::stop()
{
  if (!impl_->started)
  {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stop = true;
  }
  impl_->infer_cv.notify_all();
  impl_->fuse_cv.notify_all();
  if (impl_->infer_thread.joinable())
  {
    impl_->infer_thread.join();
  }
  if (impl_->fuse_thread.joinable())
  {
    impl_->fuse_thread.join();
  }
  impl_->started = false;
}

std::vector<uint32_t> DenseMapper::trackedIds() const
{
  std::lock_guard<std::mutex> lock(impl_->tracked_mutex);
  return std::vector<uint32_t>(impl_->tracked.begin(), impl_->tracked.end());
}

void DenseMapper::submit(const uint32_t image_id, const bool registered, const cv::Mat &left,
                         const cv::Mat &right, const colmap::MappingPose &cam_from_world,
                         const std::vector<uint32_t> &removed_ids,
                         const std::vector<uint32_t> &redundant_ids,
                         const std::vector<std::pair<uint32_t, colmap::MappingPose>> &pose_updates,
                         const std::vector<Eigen::Vector3f> &sparse_xyz)
{
  if (!impl_->config.enabled || !impl_->started)
  {
    return;
  }
  Job job;
  job.submitted = Clock::now();
  job.removed_ids = removed_ids;
  job.redundant_ids = redundant_ids;
  job.pose_updates = pose_updates;
  if (registered && impl_->engine.ready() && !left.empty() && !right.empty())
  {
    job.infer = true;
    job.image_id = image_id;
    job.left = left.clone();
    job.right = right.clone();
    job.cam_from_world = cam_from_world;
    job.sparse_xyz = sparse_xyz;
    std::lock_guard<std::mutex> lock(impl_->tracked_mutex);
    impl_->tracked.push_back(image_id);
    const size_t cap = static_cast<size_t>(
        std::max({impl_->config.pose_track_frames, impl_->config.integrate_delay_frames + 1, 1}));
    while (impl_->tracked.size() > cap)
    {
      impl_->tracked.pop_front();
    }
  }
  if (!job.infer && job.removed_ids.empty() && job.pose_updates.empty())
  {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->infer_queue.push(std::move(job));
  }
  impl_->infer_cv.notify_one();
}

} // namespace vslam
