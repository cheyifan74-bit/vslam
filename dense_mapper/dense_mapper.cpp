#include "dense_mapper/dense_mapper.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dense_mapper/dense_map.h"
#include "dense_mapper/depth_estimator.h"
#include "dense_mapper/frame_store.h"
#include "utility/config_reader.h"
#include "utils/print.h"

namespace vslam
{

using dense::Clock;
using dense::MsSince;
using dense::Rigid;

namespace
{

struct Job
{
  bool infer = false;
  /// Inference skipped: the view was too close to a recently inferred keyframe.
  bool skipped = false;
  dense::StereoKeyframe keyframe;
  std::vector<uint32_t> removed_ids;
  std::vector<uint32_t> redundant_ids;
  std::vector<std::pair<uint32_t, Rigid>> pose_updates;
  /// Filled by the inference thread; null when no depth was built.
  std::unique_ptr<dense::DepthFrame> depth;
  double build_ms = 0.0;
  Clock::time_point submitted;

  void releaseImages()
  {
    keyframe.left.release();
    keyframe.right.release();
    keyframe.sparse_xyz.clear();
  }
};

} // namespace

/// Two threads: inference (rectification, S²M², depth filtering) and fusion
/// (everything in DenseMap). They meet only at the job queues.
struct DenseMapper::Impl
{
  Impl(DenseConfig config_in, std::vector<CameraParams> cameras_in, const std::string &map_dir)
      : config(std::move(config_in)), cameras(std::move(cameras_in)),
        frame_store(std::filesystem::path(map_dir) / "dense" / "frames"), estimator(config),
        map(config, frame_store, std::filesystem::path(map_dir) / "dense" / "tsdf")
  {
  }

  DenseConfig config;
  std::vector<CameraParams> cameras;
  const dense::FrameStore frame_store;
  dense::DepthEstimator estimator;
  dense::DenseMap map;

  std::mutex mutex;
  std::condition_variable infer_cv;
  std::condition_variable fuse_cv;
  std::deque<Job> infer_queue;
  std::deque<Job> fuse_queue;
  /// Built depths waiting in fuse_queue.
  int fuse_depths = 0;
  size_t dropped = 0;
  bool stop = false;
  bool loaded = false;
  bool infer_done = false;
  bool started = false;
  std::thread infer_thread;
  std::thread fuse_thread;

  mutable std::mutex tracked_mutex;
  std::deque<uint32_t> tracked;

  bool stopping()
  {
    std::lock_guard<std::mutex> lock(mutex);
    return stop;
  }

  /// Touches only the job and its own new frame file, never the map.
  void inferLoop()
  {
    while (true)
    {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        // Backpressure: while fusion is behind, keyframes wait here, where
        // submit() can drop the oldest images instead of queuing depth.
        const int cap = config.max_queue_frames;
        infer_cv.wait(lock, [this, cap] {
          return loaded && (stop || (!infer_queue.empty() && (cap <= 0 || fuse_depths < cap)));
        });
        if (infer_queue.empty())
        {
          infer_done = true;
          break;
        }
        job = std::move(infer_queue.front());
        infer_queue.pop_front();
        // Stopping: drop the depth of queued keyframes but keep their pose
        // updates and removals, which the fusion thread still applies.
        if (stop)
        {
          job.infer = false;
          job.keyframe.left.release();
          job.keyframe.right.release();
        }
      }
      if (job.infer && estimator.similarToRecent(job.keyframe.cam_from_world))
      {
        job.infer = false;
        job.skipped = true;
        job.releaseImages();
      }
      if (job.infer)
      {
        const auto start = Clock::now();
        auto frame = std::make_unique<dense::DepthFrame>();
        if (estimator.build(job.keyframe, frame.get()))
        {
          frame_store.write(*frame);
          job.depth = std::move(frame);
        }
        job.releaseImages();
        job.build_ms = MsSince(start);
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        fuse_depths += job.depth ? 1 : 0;
        fuse_queue.push_back(std::move(job));
      }
      fuse_cv.notify_one();
    }
    fuse_cv.notify_all();
  }

  void applyJob(Job job, bool idle, size_t queued, size_t dropped_total)
  {
    const auto start = Clock::now();
    const bool is_stopping = stopping();
    dense::MapJob map_job;
    map_job.removed_ids = std::move(job.removed_ids);
    map_job.redundant_ids = std::move(job.redundant_ids);
    map_job.pose_updates.reserve(job.pose_updates.size());
    for (const auto &item : job.pose_updates)
    {
      map_job.pose_updates.emplace_back(item.first, estimator.rectifiedPose(item.second));
    }
    map_job.depth = std::move(job.depth);
    const dense::ApplyStats stats = map.apply(std::move(map_job), idle, is_stopping);
    // build runs on the inference thread; apply is the fusion-thread time;
    // latency is submit -> applied, including both queues.
    PRINT_INFO("[DENSE]: job image %u build=%.0fms fuse=%.0fms (%d) rebuild=%.0fms publish=%.0fms "
               "apply=%.0fms latency=%.0fms queue=%zu pending=%zu retired=%d moved_submaps=%zu "
               "skipped=%d dropped=%zu\n",
               job.keyframe.image_id, job.build_ms, stats.fuse_ms, stats.fused, stats.rebuild_ms,
               stats.publish_ms, MsSince(start), MsSince(job.submitted), queued, stats.pending,
               stats.retired, stats.moved_submaps, job.skipped ? 1 : 0, dropped_total);
  }

  void fuseLoop()
  {
#ifdef _OPENMP
    // Bound fusion/rebuild parallelism so Dense does not starve the VIO thread.
    if (config.num_threads > 0)
    {
      omp_set_num_threads(config.num_threads);
    }
#endif
    map.load([this] { return stopping(); });
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
      size_t dropped_total = 0;
      {
        std::unique_lock<std::mutex> lock(mutex);
        // A rebuild in progress continues as soon as no job is waiting.
        const auto timeout = map.rebuildPending() ? std::chrono::duration<double>(0) : wait;
        fuse_cv.wait_for(lock, timeout, [this] { return infer_done || !fuse_queue.empty(); });
        if (fuse_queue.empty() && infer_done)
        {
          break;
        }
        if (!fuse_queue.empty())
        {
          job = std::move(fuse_queue.front());
          fuse_queue.pop_front();
          fuse_depths -= job.depth ? 1 : 0;
          has_job = true;
        }
        // A frame still on the GPU does not block a rebuild: running it now
        // overlaps it with the inference instead of delaying the next fusion.
        idle = fuse_queue.empty();
        queued = fuse_queue.size() + infer_queue.size();
        dropped_total = dropped;
      }
      if (has_job)
      {
        infer_cv.notify_one();
        applyJob(std::move(job), idle, queued, dropped_total);
      }
      else
      {
        map.onIdle();
      }
    }
    map.shutdown();
  }
};

DenseMapper::DenseMapper(DenseConfig config, std::vector<CameraParams> cameras, std::string map_dir)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(cameras), map_dir))
{
}

DenseMapper::~DenseMapper() { stop(); }

void DenseMapper::setCloudCallback(CloudCallback callback)
{
  impl_->map.setCloudCallback(std::move(callback));
}

void DenseMapper::start()
{
  if (!impl_->config.enabled || impl_->started)
  {
    return;
  }
  std::filesystem::create_directories(impl_->frame_store.dir());
  if (!impl_->estimator.init(impl_->cameras))
  {
    PRINT_WARNING("[DENSE]: rectifier failed, dense mapping disabled\n");
    impl_->config.enabled = false;
    return;
  }
  impl_->started = true;
  impl_->fuse_thread = std::thread([this] { impl_->fuseLoop(); });
  impl_->infer_thread = std::thread([this] { impl_->inferLoop(); });
  PRINT_INFO("[DENSE]: mapper started, engine=%s\n",
             impl_->estimator.ready() ? "ready" : "missing");
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
                         const cv::Mat &right, const Rigid &cam_from_world,
                         const std::vector<uint32_t> &removed_ids,
                         const std::vector<uint32_t> &redundant_ids,
                         const std::vector<std::pair<uint32_t, Rigid>> &pose_updates,
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
  if (registered && impl_->estimator.ready() && !left.empty() && !right.empty())
  {
    job.infer = true;
    job.keyframe.image_id = image_id;
    job.keyframe.left = left.clone();
    job.keyframe.right = right.clone();
    job.keyframe.cam_from_world = cam_from_world;
    job.keyframe.sparse_xyz = sparse_xyz;
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
    impl_->infer_queue.push_back(std::move(job));
    // Too many images waiting: the oldest lose theirs. Their pose updates and
    // removals still go through.
    const int cap = impl_->config.max_queue_frames;
    if (cap > 0)
    {
      int images = 0;
      for (auto it = impl_->infer_queue.rbegin(); it != impl_->infer_queue.rend(); ++it)
      {
        if (!it->infer)
        {
          continue;
        }
        if (++images > cap)
        {
          it->infer = false;
          it->releaseImages();
          ++impl_->dropped;
        }
      }
    }
  }
  impl_->infer_cv.notify_one();
}


void DenseConfig::load(ConfigReader &reader)
{
  reader.read("dense_enabled", enabled);
  reader.read("dense_engine_path", engine_path);
  reader.read("dense_rect_width", rect_width);
  reader.read("dense_rect_height", rect_height);
  reader.read("dense_voxel_m", voxel_m);
  reader.read("dense_truncation_m", truncation_m);
  reader.read("dense_min_range_m", min_range_m);
  reader.read("dense_max_range_m", max_range_m);
  reader.read("dense_min_confidence", min_confidence);
  reader.read("dense_min_visible", min_visible);
  reader.read("dense_weight_reference_range_m", weight_reference_range_m);
  reader.read("dense_saturate_gray", saturate_gray);
  reader.read("dense_sparse_depth_ratio", sparse_depth_ratio);
  reader.read("dense_sparse_depth_margin_m", sparse_depth_margin_m);
  reader.read("dense_sparse_dilate_px", sparse_dilate_px);
  reader.read("dense_integrate_delay_frames", integrate_delay_frames);
  reader.read("dense_pose_track_frames", pose_track_frames);
  reader.read("dense_pose_update_trans_m", pose_update_trans_m);
  reader.read("dense_pose_update_rot_deg", pose_update_rot_deg);
  reader.read("dense_idle_flush_s", idle_flush_s);
  reader.read("dense_carve_free_space", carve_free_space);
  reader.read("dense_carve_weight", carve_weight);
  reader.read("dense_surface_min_weight", surface_min_weight);
  reader.read("dense_edge_ratio", edge_ratio);
  reader.read("dense_replay_max_defer_jobs", replay_max_defer_jobs);
  reader.read("dense_rebuild_redundant_removals", rebuild_redundant_removals);
  reader.read("dense_resident_frames", resident_frames);
  reader.read("dense_publish_period_s", publish_period_s);
  reader.read("dense_submap_max_frames", submap_max_frames);
  reader.read("dense_submap_max_distance_m", submap_max_distance_m);
  reader.read("dense_num_threads", num_threads);
  reader.read("dense_snapshot_period_s", snapshot_period_s);
  reader.read("dense_stats_period_s", stats_period_s);
  reader.read("dense_integrate_pixel_stride", integrate_pixel_stride);
  reader.read("dense_free_empty_blocks", free_empty_blocks);
  reader.read("dense_skip_similar_trans_m", skip_similar_trans_m);
  reader.read("dense_skip_similar_rot_deg", skip_similar_rot_deg);
  reader.read("dense_max_queue_frames", max_queue_frames);
  reader.read("dense_freeze_idle_s", freeze_idle_s);
  reader.read("dense_replay_budget_ms", replay_budget_ms);
  reader.read("dense_replace_sdf_diff", replace_sdf_diff);
  reader.read("dense_replace_min_conf", replace_min_conf);
  reader.read("dense_replace_weight", replace_weight);
}

} // namespace vslam
