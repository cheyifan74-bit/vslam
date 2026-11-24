#include "dense_mapper/snapshot_store.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace vslam
{
namespace dense
{
namespace
{

// v2 submap files carry the tombstones of freed blocks; v1 snapshots are re-fused.
constexpr char kIndexMagic[4] = {'D', 'T', 'I', '2'};
constexpr char kSubmapMagic[4] = {'D', 'T', 'S', '2'};

template <typename T> void Put(std::ostream &out, const T &value)
{
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T> bool Get(std::istream &in, T *value)
{
  in.read(reinterpret_cast<char *>(value), sizeof(T));
  return static_cast<bool>(in);
}

void PutRigid(std::ostream &out, const Rigid &pose)
{
  const double v[7] = {pose.q.x(), pose.q.y(), pose.q.z(), pose.q.w(),
                       pose.t.x(), pose.t.y(), pose.t.z()};
  out.write(reinterpret_cast<const char *>(v), sizeof(v));
}

bool GetRigid(std::istream &in, Rigid *pose)
{
  double v[7];
  in.read(reinterpret_cast<char *>(v), sizeof(v));
  pose->q = Eigen::Quaterniond(v[3], v[0], v[1], v[2]).normalized();
  pose->t = Eigen::Vector3d(v[4], v[5], v[6]);
  return static_cast<bool>(in);
}

void PutKeys(std::ostream &out, const BlockSet &keys)
{
  Put(out, static_cast<uint64_t>(keys.size()));
  for (const BlockKey &key : keys)
  {
    const int32_t k[3] = {key.x, key.y, key.z};
    out.write(reinterpret_cast<const char *>(k), sizeof(k));
  }
}

bool GetKeys(std::istream &in, BlockSet *keys)
{
  uint64_t count = 0;
  if (!Get(in, &count) || count > (1ull << 32))
  {
    return false;
  }
  std::vector<int32_t> raw(static_cast<size_t>(count) * 3);
  in.read(reinterpret_cast<char *>(raw.data()),
          static_cast<std::streamsize>(raw.size() * sizeof(int32_t)));
  if (!in)
  {
    return false;
  }
  keys->reserve(static_cast<size_t>(count));
  for (size_t i = 0; i < raw.size(); i += 3)
  {
    keys->insert({raw[i], raw[i + 1], raw[i + 2]});
  }
  return true;
}

/// Writes to path + ".tmp" and renames, so a crash never leaves a torn file.
template <typename Writer> bool WriteAtomic(const std::filesystem::path &path, Writer write)
{
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out)
    {
      return false;
    }
    write(out);
    out.flush();
    if (!out)
    {
      out.close();
      std::filesystem::remove(tmp, ec);
      return false;
    }
  }
  std::filesystem::rename(tmp, path, ec);
  return !ec;
}

} // namespace

std::filesystem::path SnapshotStore::submapPath(uint32_t id, uint64_t gen) const
{
  return dir_ / ("submap_" + std::to_string(id) + "_" + std::to_string(gen) + ".bin");
}

void SnapshotStore::putFingerprint(std::ostream &out) const
{
  Put(out, grid_.voxel);
  Put(out, grid_.trunc);
  Put(out, static_cast<int32_t>(kBlockN));
  Put(out, static_cast<int32_t>(sizeof(Voxel)));
}

bool SnapshotStore::checkFingerprint(std::istream &in) const
{
  double v = 0.0;
  double tr = 0.0;
  int32_t n = 0;
  int32_t size = 0;
  return Get(in, &v) && Get(in, &tr) && Get(in, &n) && Get(in, &size) &&
         std::abs(v - grid_.voxel) < 1e-9 && std::abs(tr - grid_.trunc) < 1e-9 && n == kBlockN &&
         size == static_cast<int32_t>(sizeof(Voxel));
}

bool SnapshotStore::writeSubmap(const Submap &submap, uint64_t gen) const
{
  return WriteAtomic(submapPath(submap.id, gen), [&](std::ostream &out) {
    out.write(kSubmapMagic, 4);
    putFingerprint(out);
    Put(out, submap.id);
    Put(out, gen);
    Put(out, static_cast<uint64_t>(submap.blocks.size()));
    for (const auto &item : submap.blocks)
    {
      const int32_t k[3] = {item.first.x, item.first.y, item.first.z};
      out.write(reinterpret_cast<const char *>(k), sizeof(k));
      out.write(reinterpret_cast<const char *>(item.second.data()),
                static_cast<std::streamsize>(item.second.size() * sizeof(Voxel)));
    }
    Put(out, static_cast<uint64_t>(submap.tombstones.size()));
    for (const auto &item : submap.tombstones)
    {
      const int32_t k[3] = {item.first.x, item.first.y, item.first.z};
      out.write(reinterpret_cast<const char *>(k), sizeof(k));
      Put(out, item.second);
    }
  });
}

bool SnapshotStore::readSubmap(Submap *submap, uint64_t gen, bool mark_dirty) const
{
  std::ifstream in(submapPath(submap->id, gen), std::ios::binary);
  char magic[4];
  if (!in || !in.read(magic, 4) || !std::equal(magic, magic + 4, kSubmapMagic) ||
      !checkFingerprint(in))
  {
    return false;
  }
  uint32_t id = 0;
  uint64_t file_gen = 0;
  uint64_t count = 0;
  if (!Get(in, &id) || !Get(in, &file_gen) || !Get(in, &count) || id != submap->id ||
      file_gen != gen || count > (1ull << 28))
  {
    return false;
  }
  submap->blocks.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i)
  {
    int32_t k[3];
    in.read(reinterpret_cast<char *>(k), sizeof(k));
    std::vector<Voxel> cells(kBlockVoxels);
    in.read(reinterpret_cast<char *>(cells.data()),
            static_cast<std::streamsize>(cells.size() * sizeof(Voxel)));
    if (!in)
    {
      return false;
    }
    const BlockKey key{k[0], k[1], k[2]};
    submap->blocks.emplace(key, std::move(cells));
    if (mark_dirty)
    {
      submap->surface_dirty.insert(key);
    }
    submap->growBounds(key);
  }
  uint64_t num_tombs = 0;
  if (!Get(in, &num_tombs) || num_tombs > (1ull << 28))
  {
    return false;
  }
  submap->tombstones.reserve(static_cast<size_t>(num_tombs));
  for (uint64_t i = 0; i < num_tombs; ++i)
  {
    int32_t k[3];
    uint32_t last = 0;
    in.read(reinterpret_cast<char *>(k), sizeof(k));
    if (!Get(in, &last))
    {
      return false;
    }
    submap->tombstones[BlockKey{k[0], k[1], k[2]}] = last;
  }
  return true;
}

bool SnapshotStore::writeIndex(uint64_t gen, uint32_t active, uint32_t next,
                               const std::vector<IndexSubmap> &submaps,
                               const std::vector<const DepthFrame *> &frames) const
{
  return WriteAtomic(dir_ / "index.bin", [&](std::ostream &out) {
    out.write(kIndexMagic, 4);
    putFingerprint(out);
    Put(out, gen);
    Put(out, active);
    Put(out, next);
    Put(out, static_cast<uint32_t>(submaps.size()));
    for (const IndexSubmap &submap : submaps)
    {
      Put(out, submap.id);
      Put(out, submap.file_gen);
      Put(out, submap.num_members);
      PutRigid(out, submap.world_from_submap);
      PutKeys(out, submap.replay_dirty);
      Put(out, static_cast<uint32_t>(submap.replay_changed.size()));
      for (const uint32_t id : submap.replay_changed)
      {
        Put(out, id);
      }
    }
    Put(out, static_cast<uint32_t>(frames.size()));
    for (const DepthFrame *frame : frames)
    {
      Put(out, frame->image_id);
      Put(out, frame->submap);
      PutRigid(out, frame->fused);
      PutKeys(out, frame->blocks);
    }
  });
}

bool SnapshotStore::readIndex(IndexData *data, std::string *why) const
{
  why->clear();
  std::ifstream in(dir_ / "index.bin", std::ios::binary);
  if (!in)
  {
    return false;
  }
  const auto fail = [why](const char *reason) {
    *why = reason;
    return false;
  };
  char magic[4];
  if (!in.read(magic, 4) || !std::equal(magic, magic + 4, kIndexMagic))
  {
    return fail("bad index");
  }
  if (!checkFingerprint(in))
  {
    return fail("voxel, truncation or block layout changed");
  }
  uint32_t num_submaps = 0;
  if (!Get(in, &data->gen) || !Get(in, &data->active) || !Get(in, &data->next) ||
      !Get(in, &num_submaps))
  {
    return fail("truncated index");
  }
  for (uint32_t i = 0; i < num_submaps; ++i)
  {
    IndexSubmap submap;
    uint32_t num_changed = 0;
    if (!Get(in, &submap.id) || !Get(in, &submap.file_gen) || !Get(in, &submap.num_members) ||
        !GetRigid(in, &submap.world_from_submap) || !GetKeys(in, &submap.replay_dirty) ||
        !Get(in, &num_changed))
    {
      return fail("truncated index");
    }
    for (uint32_t j = 0; j < num_changed; ++j)
    {
      uint32_t id = 0;
      if (!Get(in, &id))
      {
        return fail("truncated index");
      }
      submap.replay_changed.insert(id);
    }
    data->submaps.push_back(std::move(submap));
  }
  uint32_t num_frames = 0;
  if (!Get(in, &num_frames))
  {
    return fail("truncated index");
  }
  data->frames.resize(num_frames);
  for (IndexFrame &frame : data->frames)
  {
    if (!Get(in, &frame.image_id) || !Get(in, &frame.submap) || !GetRigid(in, &frame.fused) ||
        !GetKeys(in, &frame.blocks))
    {
      return fail("truncated index");
    }
  }
  return true;
}

void SnapshotStore::removeStaleSubmaps(
    const std::vector<std::pair<uint32_t, uint64_t>> &keep_files) const
{
  std::unordered_set<std::string> keep;
  for (const auto &item : keep_files)
  {
    keep.insert(submapPath(item.first, item.second).filename().string());
  }
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator(dir_, ec))
  {
    const std::string name = entry.path().filename().string();
    if (name.rfind("submap_", 0) == 0 && keep.count(name) == 0)
    {
      std::filesystem::remove(entry.path(), ec);
    }
  }
}

} // namespace dense
} // namespace vslam
