/*
 * @Description: ConfigReader — reads vslam.yaml once and checks it.
 *               Each config struct lists its keys in its own load(ConfigReader &).
 *               Keys nobody asked for (typos, removed options) and values of the
 *               wrong type (e.g. 0.0 for an int) are reported instead of ignored.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#ifndef VSLAM_UTILITY_CONFIG_READER_H
#define VSLAM_UTILITY_CONFIG_READER_H

#include <cstddef>
#include <set>
#include <string>

#include <opencv2/core/persistence.hpp>

namespace vslam
{

  class ConfigReader
  {
  public:
    explicit ConfigReader(const std::string &path);

    bool isOpened() const { return fs_.isOpened(); }
    const std::string &path() const { return path_; }

    /// A missing key keeps the default in value. A value of the wrong type is
    /// reported, counted in numErrors() and also leaves the default.
    void read(const std::string &key, bool &value);
    void read(const std::string &key, int &value);
    void read(const std::string &key, double &value);
    void read(const std::string &key, std::string &value);

    /// Warns about every top-level key that no read() asked for. Returns their count.
    std::size_t warnUnusedKeys() const;

    std::size_t numErrors() const { return num_errors_; }
    /// Keys asked for that were present in the file.
    std::size_t numFound() const { return num_found_; }

  private:
    /// Marks key as used; returns an empty node when the file lacks it.
    cv::FileNode find(const std::string &key);
    void typeError(const std::string &key, const char *expected, const cv::FileNode &node);

    std::string path_;
    cv::FileStorage fs_;
    std::set<std::string> used_;
    std::size_t num_found_ = 0;
    std::size_t num_errors_ = 0;
  };

} // namespace vslam

#endif // VSLAM_UTILITY_CONFIG_READER_H
