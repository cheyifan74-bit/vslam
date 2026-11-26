/*
 * @Description: MapStore — persistent ledger of keyframe lifecycle state.
 * @Author: che yifan
 * @Date: 2026-11-26
 *
 * The COLMAP database keeps geometry (image rows, keypoints, descriptors,
 * two-view geometries). This class answers the orthogonal question "is this
 * keyframe committed / pending / culled?", which lets the write path know
 * whether a Process() result should be committed, kept as a bounded pending
 * candidate, or rolled back, and lets startup recovery reconcile the on-disk
 * state without replaying an append-only event log.
 *
 * Stored in its own SQLite file <map>/state/state.db (WAL, synchronous=NORMAL)
 * so it never conflicts with COLMAP's schema or its serialized sessions.
 */
#ifndef VSLAM_MAP_MANAGER_MAP_STORE_H
#define VSLAM_MAP_MANAGER_MAP_STORE_H

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct sqlite3;

namespace vslam
{

  struct KeyframeRecord
  {
    uint32_t image_id = 0;
    std::string name;      ///< COLMAP image name, e.g. left_image/000014_...png
    std::string rgb_name;  ///< left_rgb/... sibling; empty when no RGB frame
    std::string state;     ///< "pending" | "committed" | "culled"
  };

  /// Lifecycle ledger for keyframes. Single long-lived connection; every method
  /// is a short autocommit statement so the mapping thread is never held up.
  class MapStore
  {
  public:
    MapStore() = default;
    ~MapStore();

    MapStore(const MapStore &) = delete;
    MapStore &operator=(const MapStore &) = delete;

    /// Create <state_dir> and open (or create) state.db.
    bool open(const std::string &state_dir);
    void close();

    bool insertPending(uint32_t image_id, const std::string &name,
                       const std::string &rgb_name);
    bool setState(uint32_t image_id, const char *state);
    bool erase(uint32_t image_id);
    bool getRecord(uint32_t image_id, KeyframeRecord *record) const;

    std::vector<KeyframeRecord> records() const;
    std::vector<uint32_t> idsByState(const char *state) const;

    const std::string &path() const { return db_path_; }

  private:
    bool exec(const char *sql) const;

    mutable std::mutex mutex_;
    sqlite3 *db_ = nullptr;
    std::string db_path_;
  };

} // namespace vslam

#endif // VSLAM_MAP_MANAGER_MAP_STORE_H
