#ifndef VSLAM_DENSE_MAPPER_FRAME_STORE_H
#define VSLAM_DENSE_MAPPER_FRAME_STORE_H

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

#include "dense_mapper/depth_frame.h"

namespace vslam
{
namespace dense
{

/// Depth frames on disk, one <image_id>.bin per keyframe: "DFR2", image id,
/// a reserved flags word, cam_from_world (q xyzw, t), width, height,
/// fx fy cx cy baseline, then depth (mm), confidence and gray per pixel.
/// Holds no state besides the directory, so both dense threads can use it.
class FrameStore
{
public:
  explicit FrameStore(std::filesystem::path dir) : dir_(std::move(dir)) {}

  const std::filesystem::path &dir() const { return dir_; }
  std::filesystem::path path(uint32_t image_id) const;

  bool write(const DepthFrame &frame) const;
  /// Rewrites only the pose of an existing file.
  bool writePose(const DepthFrame &frame) const;
  /// pixels = false reads only the header (pose, intrinsics).
  bool read(const std::filesystem::path &path, DepthFrame *frame, bool pixels) const;
  /// Headers of every readable frame file.
  std::vector<DepthFrame> readHeaders() const;
  /// Reads the pixels of an evicted frame back; the in-memory pose wins.
  bool loadPixels(DepthFrame *frame) const;
  void remove(uint32_t image_id) const;

private:
  std::filesystem::path dir_;
};

} // namespace dense
} // namespace vslam

#endif
