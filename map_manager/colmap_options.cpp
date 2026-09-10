/*
 * @Description: vslam yaml DTOs → COLMAP online SfM options.
 * @Author: che yifan
 * @Date: 2026-09-10
 */
#include "map_manager/colmap_options.h"

#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"

namespace vslam
{

  colmap::FeatureExtractionOptions ToColmap(const SiftExtractConfig &cfg)
  {
    colmap::FeatureExtractionOptions options(colmap::FeatureExtractorType::SIFT);
    options.use_gpu = cfg.use_gpu;
    options.gpu_index = cfg.gpu_index;
    options.max_image_size = cfg.max_image_size;
    options.sift->max_num_features = cfg.max_num_features;
    options.sift->first_octave = cfg.first_octave;
    options.sift->num_octaves = cfg.num_octaves;
    options.sift->octave_resolution = cfg.octave_resolution;
    options.sift->peak_threshold = cfg.peak_threshold;
    options.sift->edge_threshold = cfg.edge_threshold;
    options.sift->estimate_affine_shape = cfg.estimate_affine_shape;
    options.sift->max_num_orientations = cfg.max_num_orientations;
    options.sift->upright = cfg.upright;
    return options;
  }

  colmap::OnlineMatchingOptions ToColmap(const MatchConfig &cfg)
  {
    colmap::OnlineMatchingOptions options;
    options.overlap = cfg.overlap;
    options.matching = colmap::FeatureMatchingOptions(
        colmap::FeatureMatcherType::SIFT_BRUTEFORCE);
    options.matching.use_gpu = cfg.use_gpu;
    options.matching.gpu_index = cfg.gpu_index;
    options.matching.max_num_matches = cfg.max_num_matches;
    options.matching.num_threads = cfg.use_gpu ? 1 : cfg.num_threads;
    options.matching.guided_matching = false;
    options.matching.sift->max_ratio = cfg.max_ratio;
    options.matching.sift->max_distance = cfg.max_distance;
    options.matching.sift->cross_check = cfg.cross_check;
    options.matching.sift->cpu_brute_force_matcher = cfg.cpu_brute_force_matcher;
    options.e_only = cfg.e_only;
    options.geometry.min_num_inliers = cfg.min_num_inliers;
    options.geometry.detect_watermark = cfg.detect_watermark;
    options.geometry.use_degensac = cfg.use_degensac;
    options.geometry.ransac_options.max_error = cfg.max_error;
    options.geometry.ransac_options.min_num_trials = cfg.ransac_min_num_trials;
    options.geometry.ransac_options.max_num_trials = cfg.ransac_max_num_trials;
    options.geometry.ransac_options.confidence = cfg.ransac_confidence;
    return options;
  }

} // namespace vslam
