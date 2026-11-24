#include "dense_mapper/frame_store.h"

#include <algorithm>
#include <fstream>
#include <string>

#include "utils/print.h"

namespace vslam
{
namespace dense
{
namespace
{

constexpr char kMagic[4] = {'D', 'F', 'R', '2'};
// magic + image id + flags.
constexpr std::streamoff kPoseOffset = 12;

} // namespace

std::filesystem::path FrameStore::path(uint32_t image_id) const
{
  return dir_ / (std::to_string(image_id) + ".bin");
}

bool FrameStore::write(const DepthFrame &frame) const
{
  std::ofstream out(path(frame.image_id), std::ios::binary);
  if (!out)
  {
    return false;
  }
  out.write(kMagic, 4);
  out.write(reinterpret_cast<const char *>(&frame.image_id), sizeof(frame.image_id));
  const uint32_t flags = 0;
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

bool FrameStore::writePose(const DepthFrame &frame) const
{
  std::fstream io(path(frame.image_id), std::ios::in | std::ios::out | std::ios::binary);
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

bool FrameStore::read(const std::filesystem::path &file, DepthFrame *frame, bool pixels) const
{
  std::ifstream in(file, std::ios::binary);
  if (!in || frame == nullptr)
  {
    return false;
  }
  char magic[4];
  in.read(magic, 4);
  if (!in || !std::equal(magic, magic + 4, kMagic))
  {
    return false;
  }
  in.read(reinterpret_cast<char *>(&frame->image_id), sizeof(frame->image_id));
  uint32_t flags = 0;
  in.read(reinterpret_cast<char *>(&flags), sizeof(flags));
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

std::vector<DepthFrame> FrameStore::readHeaders() const
{
  std::vector<DepthFrame> frames;
  if (!std::filesystem::exists(dir_))
  {
    return frames;
  }
  for (const auto &entry : std::filesystem::directory_iterator(dir_))
  {
    if (entry.path().extension() != ".bin")
    {
      continue;
    }
    DepthFrame frame;
    if (read(entry.path(), &frame, false))
    {
      frames.push_back(std::move(frame));
    }
  }
  return frames;
}

bool FrameStore::loadPixels(DepthFrame *frame) const
{
  DepthFrame stored;
  if (!read(path(frame->image_id), &stored, true) || stored.width != frame->width ||
      stored.height != frame->height)
  {
    PRINT_WARNING("[DENSE]: cannot reload depth frame %u\n", frame->image_id);
    return false;
  }
  frame->depth_mm.swap(stored.depth_mm);
  frame->conf.swap(stored.conf);
  frame->gray.swap(stored.gray);
  return true;
}

void FrameStore::remove(uint32_t image_id) const { std::filesystem::remove(path(image_id)); }

} // namespace dense
} // namespace vslam
