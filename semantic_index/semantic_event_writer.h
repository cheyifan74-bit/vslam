/*
 * @Description: Append-only JSONL events for the semantic indexer (SigLIP2).
 * @Author: che yifan
 * @Date: 2026-11-25
 */
#ifndef VSLAM_SEMANTIC_INDEX_SEMANTIC_EVENT_WRITER_H
#define VSLAM_SEMANTIC_INDEX_SEMANTIC_EVENT_WRITER_H

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vslam
{

  struct SemanticConfig
  {
    bool enabled = false;
    /// Whether a color camera is expected and RGB should be paired/recorded.
    /// Defaults to `enabled` in load(); set semantic_color_enabled=false to run
    /// semantic indexing off a sensor that publishes no color stream.
    bool color_enabled = true;
    std::string color_topic = "/camera/color/image_raw";
    /// Max |t_color - t_keyframe| to accept a pair. Color and IR are separate
    /// streams, so 50 ms is safer than the old 10 ms (which dropped frames).
    double color_sync_slop = 0.05;
    int color_buffer_size = 120;
    /// Local directory under data/models (pre-downloaded). No HF id at runtime.
    std::string siglip_model =
        "/home/cyf/vslam_ws/src/vslam/data/models/siglip2-so400m-patch14-384";
    int topk = 5;

    void load(class ConfigReader &reader);
  };

  /// Writes upsert/delete events under <map>/semantic/events.jsonl for the
  /// Python SigLIP2 node. Mapping stays ROS-free; the indexer tails this file.
  class SemanticEventWriter
  {
  public:
    SemanticEventWriter() = default;

    bool open(const std::string &semantic_dir);
    void close();

    void writeUpsert(uint32_t image_id, const std::string &rgb_abs_path,
                     const std::string &file_name);
    void writeDelete(const std::vector<uint32_t> &image_ids);

    const std::string &eventsPath() const { return events_path_; }

  private:
    void appendLine(const std::string &line);

    std::mutex mutex_;
    std::string events_path_;
    bool open_ = false;
  };

} // namespace vslam

#endif // VSLAM_SEMANTIC_INDEX_SEMANTIC_EVENT_WRITER_H
