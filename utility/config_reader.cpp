/*
 * @Description: ConfigReader implementation.
 * @Author: che yifan
 * @Date: 2026-11-23
 */
#include "utility/config_reader.h"

#include <opencv2/core.hpp>

#include "utils/colors.h"
#include "utils/print.h"

namespace vslam
{

  namespace
  {
    std::string describe(const cv::FileNode &node)
    {
      if (node.isInt())
      {
        return std::to_string(static_cast<int>(node)) + " (int)";
      }
      if (node.isReal())
      {
        return std::to_string(static_cast<double>(node)) + " (float)";
      }
      if (node.isString())
      {
        return "\"" + static_cast<std::string>(node) + "\" (string)";
      }
      if (node.isMap())
      {
        return "a map";
      }
      if (node.isSeq())
      {
        return "a list";
      }
      return "an empty value";
    }

    /// OpenCV keeps an inline comment as part of an unquoted value
    /// ("key: true  # note" reads as "true  # note"): cut at " #" and trim.
    std::string stripComment(const std::string &raw)
    {
      std::string v = raw;
      for (std::size_t i = 1; i < v.size(); ++i)
      {
        if (v[i] == '#' && (v[i - 1] == ' ' || v[i - 1] == '\t'))
        {
          v.erase(i);
          break;
        }
      }
      const std::size_t end = v.find_last_not_of(" \t");
      return end == std::string::npos ? std::string() : v.substr(0, end + 1);
    }
  } // namespace

  ConfigReader::ConfigReader(const std::string &path) : path_(path)
  {
    try
    {
      fs_.open(path, cv::FileStorage::READ);
    }
    catch (const cv::Exception &e)
    {
      PRINT_ERROR(RED "[CONFIG]: Failed to parse %s: %s\n" RESET, path.c_str(), e.what());
    }
  }

  cv::FileNode ConfigReader::find(const std::string &key)
  {
    used_.insert(key);
    if (!fs_.isOpened())
    {
      return cv::FileNode();
    }
    cv::FileNode node = fs_[key];
    if (node.empty())
    {
      PRINT_DEBUG("[CONFIG]: %s not set, using the default\n", key.c_str());
    }
    else
    {
      ++num_found_;
    }
    return node;
  }

  void ConfigReader::typeError(const std::string &key, const char *expected, const cv::FileNode &node)
  {
    ++num_errors_;
    PRINT_WARNING(YELLOW "[CONFIG]: %s expects %s but %s has %s; keeping the default\n" RESET,
                  key.c_str(), expected, path_.c_str(), describe(node).c_str());
  }

  void ConfigReader::read(const std::string &key, bool &value)
  {
    const cv::FileNode node = find(key);
    if (node.empty())
    {
      return;
    }
    if (node.isInt())
    {
      const int v = static_cast<int>(node);
      if (v == 0 || v == 1)
      {
        value = v == 1;
        return;
      }
    }
    else if (node.isString())
    {
      // OpenCV has no bool type: true/false arrive as strings.
      const std::string v = stripComment(static_cast<std::string>(node));
      if (v == "true" || v == "True" || v == "TRUE")
      {
        value = true;
        return;
      }
      if (v == "false" || v == "False" || v == "FALSE")
      {
        value = false;
        return;
      }
    }
    typeError(key, "a bool (true/false/1/0)", node);
  }

  void ConfigReader::read(const std::string &key, int &value)
  {
    const cv::FileNode node = find(key);
    if (node.empty())
    {
      return;
    }
    if (!node.isInt())
    {
      typeError(key, "an int", node);
      return;
    }
    value = static_cast<int>(node);
  }

  void ConfigReader::read(const std::string &key, double &value)
  {
    const cv::FileNode node = find(key);
    if (node.empty())
    {
      return;
    }
    if (!node.isInt() && !node.isReal())
    {
      typeError(key, "a number", node);
      return;
    }
    value = static_cast<double>(node);
  }

  void ConfigReader::read(const std::string &key, std::string &value)
  {
    const cv::FileNode node = find(key);
    if (node.empty())
    {
      return;
    }
    if (!node.isString())
    {
      typeError(key, "a string (quote numbers, e.g. \"-1\")", node);
      return;
    }
    value = stripComment(static_cast<std::string>(node));
  }

  std::size_t ConfigReader::warnUnusedKeys() const
  {
    if (!fs_.isOpened())
    {
      return 0;
    }
    std::size_t unused = 0;
    const cv::FileNode root = fs_.root();
    for (auto it = root.begin(); it != root.end(); ++it)
    {
      const std::string key = (*it).name();
      if (used_.count(key) == 0)
      {
        ++unused;
        PRINT_WARNING(YELLOW "[CONFIG]: %s: key %s is not used by any module (typo or removed option?)\n" RESET,
                      path_.c_str(), key.c_str());
      }
    }
    return unused;
  }

} // namespace vslam
