/*
 * @Description: SemanticEventWriter — JSONL event log for SigLIP2 indexer.
 * @Author: che yifan
 * @Date: 2026-11-25
 */
#include "semantic_index/semantic_event_writer.h"

#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>

#include "utility/config_reader.h"
#include "utils/print.h"

namespace fs = boost::filesystem;

namespace vslam
{

  void SemanticConfig::load(ConfigReader &reader)
  {
    reader.read("semantic_enabled", enabled);
    // Color follows semantic_enabled unless explicitly overridden, so a
    // config that only sets semantic_enabled behaves as before.
    color_enabled = enabled;
    reader.read("semantic_color_enabled", color_enabled);
    reader.read("semantic_color_topic", color_topic);
    reader.read("semantic_color_sync_slop", color_sync_slop);
    reader.read("semantic_color_buffer_size", color_buffer_size);
    reader.read("semantic_siglip_model", siglip_model);
    reader.read("semantic_topk", topk);
  }

  bool SemanticEventWriter::open(const std::string &semantic_dir)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    try
    {
      if (!fs::exists(semantic_dir))
      {
        fs::create_directories(semantic_dir);
      }
      events_path_ = (fs::path(semantic_dir) / "events.jsonl").string();
      // Touch the file so the Python tailer can open it immediately.
      std::ofstream touch(events_path_, std::ios::app);
      if (!touch)
      {
        PRINT_ERROR("[SEMANTIC]: Failed to open events file: %s\n", events_path_.c_str());
        open_ = false;
        return false;
      }
      open_ = true;
      PRINT_INFO("[SEMANTIC]: Event log ready: %s\n", events_path_.c_str());
      return true;
    }
    catch (const fs::filesystem_error &e)
    {
      PRINT_ERROR("[SEMANTIC]: open failed: %s\n", e.what());
      open_ = false;
      return false;
    }
  }

  void SemanticEventWriter::close()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    open_ = false;
    events_path_.clear();
  }

  void SemanticEventWriter::appendLine(const std::string &line)
  {
    if (!open_ || events_path_.empty())
    {
      return;
    }
    std::ofstream out(events_path_, std::ios::app);
    if (!out)
    {
      PRINT_ERROR("[SEMANTIC]: Failed to append event to %s\n", events_path_.c_str());
      return;
    }
    out << line << '\n';
  }

  namespace
  {
    std::string jsonEscape(const std::string &s)
    {
      std::string out;
      out.reserve(s.size() + 8);
      for (char c : s)
      {
        if (c == '\\' || c == '"')
        {
          out.push_back('\\');
        }
        out.push_back(c);
      }
      return out;
    }
  } // namespace

  void SemanticEventWriter::writeUpsert(uint32_t image_id, const std::string &rgb_abs_path,
                                        const std::string &file_name)
  {
    std::ostringstream oss;
    oss << "{\"type\":\"upsert\",\"image_id\":" << image_id
        << ",\"rgb_path\":\"" << jsonEscape(rgb_abs_path) << "\""
        << ",\"file_name\":\"" << jsonEscape(file_name) << "\"}";
    std::lock_guard<std::mutex> lock(mutex_);
    appendLine(oss.str());
  }

  void SemanticEventWriter::writeDelete(const std::vector<uint32_t> &image_ids)
  {
    if (image_ids.empty())
    {
      return;
    }
    std::ostringstream oss;
    oss << "{\"type\":\"delete\",\"image_ids\":[";
    for (std::size_t i = 0; i < image_ids.size(); ++i)
    {
      if (i > 0)
      {
        oss << ',';
      }
      oss << image_ids[i];
    }
    oss << "]}";
    std::lock_guard<std::mutex> lock(mutex_);
    appendLine(oss.str());
  }

} // namespace vslam
