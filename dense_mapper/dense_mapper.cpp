#include "dense_mapper/dense_mapper.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#include "dense_mapper/s2m2_engine.h"
#include "dense_mapper/stereo_rectifier.h"
#include "utils/print.h"

namespace vslam
{
namespace
{

constexpr int kBlockN = 80;
constexpr char kMagic[4] = {'D', 'F', 'R', '1'};

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

struct Voxel
{
  float sdf = 1.f;
  float weight = 0.f;
  uint8_t gray = 0;
};

struct DepthFrame
{
  uint32_t image_id = 0;
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
  std::unordered_set<BlockKey, BlockKeyHash> blocks;
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
  std::vector<std::pair<uint32_t, colmap::MappingPose>> dirty_poses;
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

  std::mutex mutex;
  std::condition_variable cv;
  std::queue<Job> queue;
  bool stop = false;
  bool started = false;
  std::thread thread;

  double voxel = 0.05;
  double trunc = 0.15;
  std::unordered_map<BlockKey, std::vector<Voxel>, BlockKeyHash> blocks;
  std::unordered_map<uint32_t, DepthFrame> frames;
  int integrates_since_publish = 0;
  // Surface points of blocks that have not changed since the last publish.
  struct SurfacePiece
  {
    std::vector<Eigen::Vector3f> xyz;
    std::vector<uint8_t> rgb;
  };
  std::unordered_map<BlockKey, SurfacePiece, BlockKeyHash> surface;
  std::unordered_set<BlockKey, BlockKeyHash> surface_dirty;

  std::filesystem::path FramePath(uint32_t image_id) const
  {
    return frame_dir / (std::to_string(image_id) + ".bin");
  }

  BlockKey KeyOf(const Eigen::Vector3d &point) const
  {
    const double size = voxel * kBlockN;
    return {static_cast<int>(std::floor(point.x() / size)),
            static_cast<int>(std::floor(point.y() / size)),
            static_cast<int>(std::floor(point.z() / size))};
  }

  std::vector<Voxel> &Block(const BlockKey &key)
  {
    auto it = blocks.find(key);
    if (it == blocks.end())
    {
      it = blocks.emplace(key, std::vector<Voxel>(kBlockN * kBlockN * kBlockN)).first;
    }
    return it->second;
  }

  void ClearBlock(const BlockKey &key)
  {
    auto it = blocks.find(key);
    if (it != blocks.end())
    {
      it->second.assign(kBlockN * kBlockN * kBlockN, Voxel{});
      surface_dirty.insert(key);
    }
  }

  void UpdateVoxel(const Eigen::Vector3d &point, float sdf_m, float weight, uint8_t gray,
                   const std::unordered_set<BlockKey, BlockKeyHash> *only,
                   std::unordered_set<BlockKey, BlockKeyHash> *touched)
  {
    const BlockKey key = KeyOf(point);
    if (only != nullptr && only->count(key) == 0)
    {
      return;
    }
    const double size = voxel * kBlockN;
    const Eigen::Vector3d origin(key.x * size, key.y * size, key.z * size);
    const int ix = static_cast<int>(std::floor((point.x() - origin.x()) / voxel));
    const int iy = static_cast<int>(std::floor((point.y() - origin.y()) / voxel));
    const int iz = static_cast<int>(std::floor((point.z() - origin.z()) / voxel));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= kBlockN || iy >= kBlockN || iz >= kBlockN)
    {
      return;
    }
    Voxel &voxel_cell = Block(key)[ix + kBlockN * (iy + kBlockN * iz)];
    const float tsdf = std::clamp(sdf_m / static_cast<float>(trunc), -1.f, 1.f);
    const float w = std::max(weight, 0.f);
    const float sum = voxel_cell.weight + w;
    if (sum <= 0.f)
    {
      return;
    }
    voxel_cell.sdf = (voxel_cell.sdf * voxel_cell.weight + tsdf * w) / sum;
    voxel_cell.weight = std::min(sum, 64.f);
    if (w >= voxel_cell.weight * 0.5f || voxel_cell.gray == 0)
    {
      voxel_cell.gray = gray;
    }
    if (touched != nullptr && touched->insert(key).second)
    {
      surface_dirty.insert(key);
    }
  }

  const Voxel *FindVoxel(const Eigen::Vector3d &point) const
  {
    const BlockKey key = KeyOf(point);
    const auto it = blocks.find(key);
    if (it == blocks.end())
    {
      return nullptr;
    }
    const double size = voxel * kBlockN;
    const Eigen::Vector3d origin(key.x * size, key.y * size, key.z * size);
    const int ix = static_cast<int>(std::floor((point.x() - origin.x()) / voxel));
    const int iy = static_cast<int>(std::floor((point.y() - origin.y()) / voxel));
    const int iz = static_cast<int>(std::floor((point.z() - origin.z()) / voxel));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= kBlockN || iy >= kBlockN || iz >= kBlockN)
    {
      return nullptr;
    }
    return &it->second[static_cast<size_t>(ix + kBlockN * (iy + kBlockN * iz))];
  }

  // True when this ray already hits an extractable surface outside the band
  // this sample will write, but still within verify_agree_m. The stored sdf
  // points at that surface, so the shell of a surface inside the truncation
  // band does not count and later views can still accumulate weight.
  bool AnotherLayerOnRay(const Eigen::Vector3d &center, const Eigen::Vector3d &dir,
                         double length) const
  {
    const double agree = config.verify_agree_m;
    if (agree <= trunc)
    {
      return false;
    }
    const double step = voxel;
    auto band = [&](double begin, double end) {
      for (double s = begin; s <= end; s += step)
      {
        if (s <= 0.0)
        {
          continue;
        }
        const Voxel *cell = FindVoxel(center + dir * s);
        if (cell == nullptr || cell->weight < 1.f || std::abs(cell->sdf) > 0.25f)
        {
          continue;
        }
        const double crossed = s + static_cast<double>(cell->sdf) * trunc;
        const double gap = std::abs(crossed - length);
        if (gap > trunc && gap <= agree)
        {
          return true;
        }
      }
      return false;
    };
    return band(length - agree, length - trunc) || band(length + trunc, length + agree);
  }

  struct VerifyRef
  {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double fx = 0;
    double fy = 0;
    double cx = 0;
    double cy = 0;
    int width = 0;
    int height = 0;
    const uint16_t *depth = nullptr;
  };

  /// Earlier keyframes with enough baseline, most recent first. The raw depth
  /// images stay on those frames; this only decides which pixels may be fused.
  std::vector<VerifyRef> SelectVerifyRefs(const DepthFrame &frame, const Eigen::Vector3d &center) const
  {
    std::vector<VerifyRef> refs;
    if (config.verify_frames <= 0)
    {
      return refs;
    }
    struct Candidate
    {
      uint32_t id = 0;
      const DepthFrame *frame = nullptr;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(frames.size());
    for (const auto &item : frames)
    {
      if (item.first >= frame.image_id || item.second.depth_mm.empty())
      {
        continue;
      }
      const Eigen::Matrix3d rotation = item.second.q.toRotationMatrix();
      const Eigen::Vector3d other_center = -rotation.transpose() * item.second.t;
      if ((other_center - center).norm() < config.verify_min_baseline_m)
      {
        continue;
      }
      candidates.push_back({item.first, &item.second});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) { return a.id > b.id; });
    if (static_cast<int>(candidates.size()) > config.verify_frames)
    {
      candidates.resize(static_cast<size_t>(config.verify_frames));
    }
    refs.reserve(candidates.size());
    for (const Candidate &candidate : candidates)
    {
      const DepthFrame &other = *candidate.frame;
      VerifyRef ref;
      ref.R = other.q.toRotationMatrix();
      ref.t = other.t;
      ref.fx = other.fx;
      ref.fy = other.fy;
      ref.cx = other.cx;
      ref.cy = other.cy;
      ref.width = other.width;
      ref.height = other.height;
      ref.depth = other.depth_mm.data();
      refs.push_back(ref);
    }
    return refs;
  }

  void Integrate(DepthFrame *frame, const std::unordered_set<BlockKey, BlockKeyHash> *only)
  {
    if (frame == nullptr || frame->depth_mm.empty())
    {
      return;
    }
    const Eigen::Matrix3d R = frame->q.toRotationMatrix();
    const Eigen::Vector3d t = frame->t;
    const Eigen::Vector3d center = -R.transpose() * t;
    const bool verify = config.verify_frames > 0;
    const std::vector<VerifyRef> refs = verify ? SelectVerifyRefs(*frame, center) : std::vector<VerifyRef>();
    const double agree = config.verify_agree_m;
    const double gap = config.verify_gap_m;
    std::unordered_set<BlockKey, BlockKeyHash> touched;
    const double step = voxel;
    int kept = 0;
    int contradicted = 0;
    int unconfirmed = 0;
    int layered = 0;
    for (int y = 0; y < frame->height; ++y)
    {
      for (int x = 0; x < frame->width; ++x)
      {
        const size_t idx = static_cast<size_t>(y * frame->width + x);
        if (frame->depth_mm[idx] == 0)
        {
          continue;
        }
        const double depth = static_cast<double>(frame->depth_mm[idx]) * 0.001;
        const Eigen::Vector3d p_cam((x - frame->cx) * depth / frame->fx,
                                    (y - frame->cy) * depth / frame->fy, depth);
        const Eigen::Vector3d p_world = R.transpose() * (p_cam - t);
        const Eigen::Vector3d ray = p_world - center;
        const double length = ray.norm();
        if (length < 1e-4)
        {
          continue;
        }
        if (verify)
        {
          bool support = false;
          bool contra = false;
          for (const VerifyRef &ref : refs)
          {
            const Eigen::Vector3d pc = ref.R * p_world + ref.t;
            if (pc.z() < 0.3)
            {
              continue;
            }
            const int u = static_cast<int>(std::lround(ref.fx * pc.x() / pc.z() + ref.cx));
            const int v = static_cast<int>(std::lround(ref.fy * pc.y() / pc.z() + ref.cy));
            if (u < 0 || v < 0 || u >= ref.width || v >= ref.height)
            {
              continue;
            }
            const uint16_t observed = ref.depth[static_cast<size_t>(v * ref.width + u)];
            if (observed == 0)
            {
              continue;
            }
            const double z = static_cast<double>(observed) * 0.001;
            const double du = (static_cast<double>(u) - ref.cx) / ref.fx;
            const double dv = (static_cast<double>(v) - ref.cy) / ref.fy;
            const double range_d = z * std::sqrt(du * du + dv * dv + 1.0);
            const double range_p = pc.norm();
            if (range_d > range_p + gap)
            {
              contra = true;
              break;
            }
            if (std::abs(range_d - range_p) <= agree)
            {
              support = true;
            }
          }
          if (contra)
          {
            ++contradicted;
            continue;
          }
          if (!support)
          {
            ++unconfirmed;
            continue;
          }
        }
        const Eigen::Vector3d dir = ray / length;
        if (AnotherLayerOnRay(center, dir, length))
        {
          ++layered;
          continue;
        }
        ++kept;
        const uint8_t gray = frame->gray.empty() ? 180 : frame->gray[idx];
        const float weight = frame->conf.empty() ? 1.f : frame->conf[idx] / 255.f;
        for (double s = std::max(0.0, length - trunc); s <= length + trunc; s += step)
        {
          UpdateVoxel(center + dir * s, static_cast<float>(length - s), weight, gray, only,
                      &touched);
        }
      }
    }
    if (verify)
    {
      PRINT_INFO("[DENSE]: image %u verify refs=%zu keep=%d contradict=%d unconfirmed=%d layered=%d\n",
                 frame->image_id, refs.size(), kept, contradicted, unconfirmed, layered);
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

  bool ReadFrame(const std::filesystem::path &path, DepthFrame *frame) const
  {
    std::ifstream in(path, std::ios::binary);
    if (!in || frame == nullptr)
    {
      return false;
    }
    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != std::string(kMagic, 4))
    {
      return false;
    }
    in.read(reinterpret_cast<char *>(&frame->image_id), sizeof(frame->image_id));
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
    const size_t pixels = static_cast<size_t>(frame->width) * static_cast<size_t>(frame->height);
    frame->depth_mm.resize(pixels);
    frame->conf.resize(pixels);
    frame->gray.resize(pixels);
    in.read(reinterpret_cast<char *>(frame->depth_mm.data()),
            static_cast<std::streamsize>(pixels * sizeof(uint16_t)));
    in.read(reinterpret_cast<char *>(frame->conf.data()), static_cast<std::streamsize>(pixels));
    in.read(reinterpret_cast<char *>(frame->gray.data()), static_cast<std::streamsize>(pixels));
    return static_cast<bool>(in);
  }

  void LoadStoredFrames()
  {
    if (!std::filesystem::exists(frame_dir))
    {
      return;
    }
    std::vector<DepthFrame> stored;
    for (const auto &entry : std::filesystem::directory_iterator(frame_dir))
    {
      if (entry.path().extension() != ".bin")
      {
        continue;
      }
      DepthFrame frame;
      if (!ReadFrame(entry.path(), &frame))
      {
        continue;
      }
      stored.push_back(std::move(frame));
    }
    std::sort(stored.begin(), stored.end(),
              [](const DepthFrame &a, const DepthFrame &b) { return a.image_id < b.image_id; });
    size_t loaded = 0;
    for (DepthFrame &frame : stored)
    {
      Integrate(&frame, nullptr);
      frames.emplace(frame.image_id, std::move(frame));
      ++loaded;
    }
    if (loaded > 0)
    {
      PRINT_INFO("[DENSE]: replayed %zu stored depth frame(s)\n", loaded);
      Publish();
    }
  }

  void ExtractBlock(const BlockKey &key, const std::vector<Voxel> &voxels, SurfacePiece *piece) const
  {
    const double size = voxel * kBlockN;
    const Eigen::Vector3d origin(key.x * size, key.y * size, key.z * size);
    for (int i = 0; i < static_cast<int>(voxels.size()); ++i)
    {
      const Voxel &voxel_cell = voxels[i];
      if (voxel_cell.weight < 1.f || std::abs(voxel_cell.sdf) > 0.25f)
      {
        continue;
      }
      const int ix = i % kBlockN;
      const int iy = (i / kBlockN) % kBlockN;
      const int iz = i / (kBlockN * kBlockN);
      piece->xyz.emplace_back(static_cast<float>(origin.x() + (ix + 0.5) * voxel),
                              static_cast<float>(origin.y() + (iy + 0.5) * voxel),
                              static_cast<float>(origin.z() + (iz + 0.5) * voxel));
      piece->rgb.push_back(voxel_cell.gray);
      piece->rgb.push_back(voxel_cell.gray);
      piece->rgb.push_back(voxel_cell.gray);
    }
  }

  DenseCloud ExtractCloud()
  {
    for (const BlockKey &key : surface_dirty)
    {
      const auto it = blocks.find(key);
      if (it == blocks.end())
      {
        surface.erase(key);
        continue;
      }
      SurfacePiece piece;
      ExtractBlock(key, it->second, &piece);
      if (piece.xyz.empty())
      {
        surface.erase(key);
      }
      else
      {
        surface[key] = std::move(piece);
      }
    }
    surface_dirty.clear();

    DenseCloud cloud;
    size_t count = 0;
    for (const auto &item : surface)
    {
      count += item.second.xyz.size();
    }
    cloud.xyz.reserve(count);
    cloud.rgb.reserve(count * 3);
    for (const auto &item : surface)
    {
      cloud.xyz.insert(cloud.xyz.end(), item.second.xyz.begin(), item.second.xyz.end());
      cloud.rgb.insert(cloud.rgb.end(), item.second.rgb.begin(), item.second.rgb.end());
    }
    return cloud;
  }

  void Publish()
  {
    if (!cloud_callback)
    {
      return;
    }
    DenseCloud cloud = ExtractCloud();
    if (cloud.xyz.empty())
    {
      return;
    }
    cloud_callback(cloud);
    PRINT_INFO("[DENSE]: surface points=%zu blocks=%zu frames=%zu\n", cloud.xyz.size(),
               blocks.size(), frames.size());
  }

  bool BuildDepth(const Job &job, DepthFrame *frame)
  {
    if (frame == nullptr || !engine.ready() || job.left.empty() || job.right.empty())
    {
      return false;
    }
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
    const cv::Matx33d &R_rect = rectifier.rectFromLeft();
    const Eigen::Matrix3d R1 =
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_rect.val);
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
    const size_t pixels = static_cast<size_t>(frame->width) * static_cast<size_t>(frame->height);
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
                                         frame->width, frame->height, frame->fx, frame->fy,
                                         frame->cx, frame->cy, config.sparse_dilate_px)
                   : cv::Mat();
    const float *ceiling_data =
        ceiling.empty() ? nullptr : reinterpret_cast<const float *>(ceiling.data);
    int kept = 0;
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
      const double depth = static_cast<double>(frame->fx) * frame->baseline / disparity[i];
      if (depth < config.min_range_m || depth > config.max_range_m)
      {
        continue;
      }
      if (ceiling_data != nullptr && ceiling_data[i] > 0.f)
      {
        const double sparse_z = static_cast<double>(ceiling_data[i]);
        const double limit = std::max(sparse_z * config.sparse_depth_ratio,
                                       sparse_z + config.sparse_depth_margin_m);
        if (depth > limit)
        {
          ++behind_sparse;
          continue;
        }
      }
      frame->depth_mm[i] = static_cast<uint16_t>(std::min(depth * 1000.0, 65535.0));
      float weight = std::clamp(confidence[i], 0.f, 1.f);
      if (config.weight_reference_range_m > 0.0)
      {
        const double z_ref = config.weight_reference_range_m;
        const double scale = (z_ref / depth) * (z_ref / depth);
        weight = static_cast<float>(std::min(1.0, scale) * weight);
      }
      frame->conf[i] = static_cast<uint8_t>(std::clamp(weight, 0.f, 1.f) * 255.f);
      frame->gray[i] = gray.data[i];
      ++kept;
    }
    PRINT_INFO("[DENSE]: image %u depth pixels=%d / %zu saturated=%d behind_sparse=%d\n",
               job.image_id, kept, pixels, saturated, behind_sparse);
    return kept > 0;
  }

  void ApplyJob(Job job)
  {
    std::unordered_set<BlockKey, BlockKeyHash> dirty;
    for (const uint32_t id : job.removed_ids)
    {
      auto it = frames.find(id);
      if (it == frames.end())
      {
        std::filesystem::remove(FramePath(id));
        continue;
      }
      dirty.insert(it->second.blocks.begin(), it->second.blocks.end());
      frames.erase(it);
      std::filesystem::remove(FramePath(id));
    }
    const cv::Matx33d &R_rect = rectifier.rectFromLeft();
    const Eigen::Matrix3d R1 =
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_rect.val);
    std::unordered_set<uint32_t> changed_pose;
    changed_pose.reserve(job.dirty_poses.size());
    for (const auto &item : job.dirty_poses)
    {
      changed_pose.insert(item.first);
      auto it = frames.find(item.first);
      if (it == frames.end())
      {
        continue;
      }
      dirty.insert(it->second.blocks.begin(), it->second.blocks.end());
      const Eigen::Matrix3d R_left = QuatFromPose(item.second).toRotationMatrix();
      const Eigen::Vector3d t_left = TransFromPose(item.second);
      it->second.q = Eigen::Quaterniond(R1 * R_left).normalized();
      it->second.t = R1 * t_left;
      WriteFrame(it->second);
    }
    if (job.infer)
    {
      DepthFrame frame;
      if (BuildDepth(job, &frame))
      {
        Integrate(&frame, nullptr);
        WriteFrame(frame);
        frames[frame.image_id] = std::move(frame);
      }
      job.left.release();
      job.right.release();
      job.sparse_xyz.clear();
    }
    if (!dirty.empty())
    {
      for (const BlockKey &key : dirty)
      {
        ClearBlock(key);
      }
      std::vector<DepthFrame *> ordered;
      ordered.reserve(frames.size());
      for (auto &item : frames)
      {
        ordered.push_back(&item.second);
      }
      std::sort(ordered.begin(), ordered.end(),
                [](const DepthFrame *a, const DepthFrame *b) { return a->image_id < b->image_id; });
      size_t replayed = 0;
      for (DepthFrame *frame : ordered)
      {
        // Unchanged pose and no cleared block: every sample would be rejected.
        if (changed_pose.count(frame->image_id) == 0)
        {
          bool hits = false;
          for (const BlockKey &key : frame->blocks)
          {
            if (dirty.count(key) != 0)
            {
              hits = true;
              break;
            }
          }
          if (!hits)
          {
            continue;
          }
        }
        Integrate(frame, &dirty);
        ++replayed;
      }
      PRINT_INFO("[DENSE]: replay %zu / %zu frames\n", replayed, ordered.size());
    }
    ++integrates_since_publish;
    if (integrates_since_publish >= 3 || !dirty.empty())
    {
      integrates_since_publish = 0;
      Publish();
    }
  }

  void Loop()
  {
    LoadStoredFrames();
    while (true)
    {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return stop || !queue.empty(); });
        if (stop && queue.empty())
        {
          break;
        }
        job = std::move(queue.front());
        queue.pop();
      }
      ApplyJob(std::move(job));
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
  impl_->thread = std::thread([this] { impl_->Loop(); });
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
  impl_->cv.notify_all();
  if (impl_->thread.joinable())
  {
    impl_->thread.join();
  }
  impl_->started = false;
}

void DenseMapper::submit(const uint32_t image_id, const bool registered, const cv::Mat &left,
                         const cv::Mat &right, const colmap::MappingPose &cam_from_world,
                         const std::vector<uint32_t> &removed_ids,
                         const std::vector<std::pair<uint32_t, colmap::MappingPose>> &dirty_poses,
                         const std::vector<Eigen::Vector3f> &sparse_xyz)
{
  if (!impl_->config.enabled || !impl_->started)
  {
    return;
  }
  Job job;
  job.removed_ids = removed_ids;
  job.dirty_poses = dirty_poses;
  if (registered && impl_->engine.ready() && !left.empty() && !right.empty())
  {
    job.infer = true;
    job.image_id = image_id;
    job.left = left.clone();
    job.right = right.clone();
    job.cam_from_world = cam_from_world;
    job.sparse_xyz = sparse_xyz;
  }
  if (!job.infer && job.removed_ids.empty() && job.dirty_poses.empty())
  {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->queue.push(std::move(job));
  }
  impl_->cv.notify_one();
}

} // namespace vslam
