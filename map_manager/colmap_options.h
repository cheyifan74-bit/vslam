/*
 * @Description: vslam yaml DTOs → COLMAP online SfM options.
 * @Author: che yifan
 * @Date: 2026-09-10
 */
#ifndef VSLAM_MAP_MANAGER_COLMAP_OPTIONS_H
#define VSLAM_MAP_MANAGER_COLMAP_OPTIONS_H

#include "colmap/feature/extractor.h"
#include "colmap/online_sfm/online_feature_matcher.h"
#include "colmap/online_sfm/online_loop_closer.h"
#include "colmap/online_sfm/online_mapper.h"
#include "map_manager/map_manager.h"

namespace vslam
{

  colmap::FeatureExtractionOptions ToColmap(const SiftExtractConfig &cfg);
  colmap::OnlineMatchingOptions ToColmap(const MatchConfig &cfg);
  colmap::OnlineMapperOptions ToColmap(const MapperConfig &cfg);
  colmap::OnlineLoopCloserOptions ToColmap(const LoopCloserConfig &cfg);

} // namespace vslam

#endif // VSLAM_MAP_MANAGER_COLMAP_OPTIONS_H
