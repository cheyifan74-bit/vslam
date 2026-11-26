/*
 * @Description: MapStore implementation (SQLite keyframe lifecycle ledger).
 * @Author: che yifan
 * @Date: 2026-11-26
 */
#include "map_manager/map_store.h"

#include <sqlite3.h>

#include <ctime>

#include <boost/filesystem.hpp>

#include "utils/print.h"

namespace fs = boost::filesystem;

namespace vslam
{

  namespace
  {
    constexpr const char *kSchema =
        "CREATE TABLE IF NOT EXISTS keyframes ("
        " image_id INTEGER PRIMARY KEY,"
        " name TEXT NOT NULL,"
        " rgb_name TEXT NOT NULL DEFAULT '',"
        " state TEXT NOT NULL,"
        " updated_at REAL NOT NULL);"
        "CREATE INDEX IF NOT EXISTS idx_keyframes_state ON keyframes(state);";
  } // namespace

  MapStore::~MapStore()
  {
    close();
  }

  bool MapStore::open(const std::string &state_dir)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    try
    {
      if (!fs::exists(state_dir))
      {
        fs::create_directories(state_dir);
      }
    }
    catch (const fs::filesystem_error &e)
    {
      PRINT_ERROR("[MAP_STORE]: failed to create state dir %s: %s\n",
                  state_dir.c_str(), e.what());
      return false;
    }
    db_path_ = (fs::path(state_dir) / "state.db").string();

    if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK)
    {
      PRINT_ERROR("[MAP_STORE]: failed to open %s: %s\n", db_path_.c_str(),
                  db_ ? sqlite3_errmsg(db_) : "out of memory");
      if (db_)
      {
        sqlite3_close(db_);
        db_ = nullptr;
      }
      return false;
    }

    // WAL + NORMAL keeps the ledger durable enough for crash recovery (only the
    // last transaction may be lost) while avoiding a per-keyframe fsync.
    exec("PRAGMA journal_mode=WAL;");
    exec("PRAGMA synchronous=NORMAL;");
    exec("PRAGMA busy_timeout=5000;");
    if (!exec(kSchema))
    {
      PRINT_ERROR("[MAP_STORE]: failed to create schema in %s\n", db_path_.c_str());
      close();
      return false;
    }
    PRINT_INFO("[MAP_STORE]: state ledger ready: %s\n", db_path_.c_str());
    return true;
  }

  void MapStore::close()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (db_)
    {
      sqlite3_close(db_);
      db_ = nullptr;
    }
  }

  bool MapStore::exec(const char *sql) const
  {
    if (!db_)
    {
      return false;
    }
    char *err = nullptr;
    const int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK)
    {
      PRINT_WARNING("[MAP_STORE]: sql error: %s\n", err ? err : sqlite3_errmsg(db_));
      sqlite3_free(err);
      return false;
    }
    return true;
  }

  bool MapStore::insertPending(uint32_t image_id, const std::string &name,
                               const std::string &rgb_name)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_)
    {
      return false;
    }
    const char *sql =
        "INSERT OR REPLACE INTO keyframes(image_id,name,rgb_name,state,updated_at)"
        " VALUES(?1,?2,?3,'pending',?4);";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
      PRINT_WARNING("[MAP_STORE]: prepare insert failed: %s\n", sqlite3_errmsg(db_));
      return false;
    }
    sqlite3_bind_int64(stmt, 1, image_id);
    sqlite3_bind_text(stmt, 2, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, rgb_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 4, static_cast<double>(std::time(nullptr)));
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE)
    {
      PRINT_WARNING("[MAP_STORE]: insert pending image_id=%u failed: %s\n",
                    image_id, sqlite3_errmsg(db_));
      return false;
    }
    return true;
  }

  bool MapStore::setState(uint32_t image_id, const char *state)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_)
    {
      return false;
    }
    const char *sql = "UPDATE keyframes SET state=?2 WHERE image_id=?1;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
      PRINT_WARNING("[MAP_STORE]: prepare setState failed: %s\n", sqlite3_errmsg(db_));
      return false;
    }
    sqlite3_bind_int64(stmt, 1, image_id);
    sqlite3_bind_text(stmt, 2, state, -1, SQLITE_TRANSIENT);
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE)
    {
      PRINT_WARNING("[MAP_STORE]: setState image_id=%u failed: %s\n",
                    image_id, sqlite3_errmsg(db_));
      return false;
    }
    return true;
  }

  bool MapStore::erase(uint32_t image_id)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_)
    {
      return false;
    }
    const char *sql = "DELETE FROM keyframes WHERE image_id=?1;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
      PRINT_WARNING("[MAP_STORE]: prepare erase failed: %s\n", sqlite3_errmsg(db_));
      return false;
    }
    sqlite3_bind_int64(stmt, 1, image_id);
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE)
    {
      PRINT_WARNING("[MAP_STORE]: erase image_id=%u failed: %s\n",
                    image_id, sqlite3_errmsg(db_));
      return false;
    }
    return true;
  }

  bool MapStore::getRecord(uint32_t image_id, KeyframeRecord *record) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_ || record == nullptr)
    {
      return false;
    }
    const char *sql =
        "SELECT image_id,name,rgb_name,state FROM keyframes WHERE image_id=?1;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
      return false;
    }
    sqlite3_bind_int64(stmt, 1, image_id);
    bool found = false;
    if (sqlite3_step(stmt) == SQLITE_ROW)
    {
      record->image_id = static_cast<uint32_t>(sqlite3_column_int64(stmt, 0));
      const char *name = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
      const char *rgb = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
      const char *state = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 3));
      record->name = name ? name : "";
      record->rgb_name = rgb ? rgb : "";
      record->state = state ? state : "";
      found = true;
    }
    sqlite3_finalize(stmt);
    return found;
  }

  std::vector<KeyframeRecord> MapStore::records() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<KeyframeRecord> out;
    if (!db_)
    {
      return out;
    }
    const char *sql = "SELECT image_id,name,rgb_name,state FROM keyframes;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
      return out;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
      KeyframeRecord r;
      r.image_id = static_cast<uint32_t>(sqlite3_column_int64(stmt, 0));
      const char *name = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
      const char *rgb = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
      const char *state = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 3));
      r.name = name ? name : "";
      r.rgb_name = rgb ? rgb : "";
      r.state = state ? state : "";
      out.push_back(std::move(r));
    }
    sqlite3_finalize(stmt);
    return out;
  }

  std::vector<uint32_t> MapStore::idsByState(const char *state) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint32_t> out;
    if (!db_)
    {
      return out;
    }
    const char *sql = "SELECT image_id FROM keyframes WHERE state=?1;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
      return out;
    }
    sqlite3_bind_text(stmt, 1, state, -1, SQLITE_TRANSIENT);
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
      out.push_back(static_cast<uint32_t>(sqlite3_column_int64(stmt, 0)));
    }
    sqlite3_finalize(stmt);
    return out;
  }

} // namespace vslam
